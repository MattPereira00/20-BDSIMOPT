import logging
import math
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood


from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning

from opt_problem import OptProblem
from opt_obj import *

class BDSIMOpt:
    def __init__(self, problem: OptProblem, filename: str):
            self.problem = problem
            self.config = problem.config
            self.filename = filename

            self.device = self.config.device
            self.dtype = torch.double

            self.run_dir = Path(self.problem.model.builder.data_dir) / filename
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.logfile = self.run_dir / f"{filename}_optlog.txt"

            self.logger = logging.getLogger(filename)
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                fh = logging.FileHandler(self.logfile, mode="a", encoding="utf-8")
                sh = logging.StreamHandler()
                fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
                fh.setFormatter(fmt)
                sh.setFormatter(fmt)
                self.logger.addHandler(fh)
                self.logger.addHandler(sh)

            self.train_X: Optional[torch.Tensor] = None
            self.train_Y: Optional[torch.Tensor] = None
            self.train_raw: List[dict] = []

            # Trust-region state (only used when config.use_trust_region=True)
            self.tr_center: Optional[torch.Tensor] = None  # normalised [0,1]^d
            self.tr_length: Optional[float] = None
            self.tr_succ_count = 0
            self.tr_fail_count = 0

    def _write_log(self, text: str) -> None:
        self.logger.info(text)

    def _scalar_transform(self, y: torch.Tensor) -> torch.Tensor:
        """
        Map loss (to be minimised) to -log(loss) (to be maximised).
        Compresses the loss's huge dynamic range and puts it in the
        "bigger is better" convention BoTorch's EI-family acquisitions assume.
        """
        return -torch.log(y.clamp_min(1e-12))

    def build_model(self, x, y):
        if self.config.mode == "scalar":
            y_fit = self._scalar_transform(y)
            gp = SingleTaskGP(x, y_fit)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            return gp

        models = []
        for i in range(y.shape[-1]):
            gp = SingleTaskGP(x, y[..., i:i + 1])
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            models.append(gp)
        return ModelListGP(*models)

    def build_acquisition(self, model):
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.config.mc_samples]))

        if self.config.mode == "scalar":
            best_f = self._scalar_transform(self.train_Y).view(-1).max().item()
            return qLogExpectedImprovement(model=model, best_f=best_f, sampler=sampler, eta=1e-3)

        _, pareto_y, _ = self.get_pareto_front()
        y_for_part = pareto_y if pareto_y.shape[0] > 0 else self.train_Y.new_empty((0, self.train_Y.shape[1]))

        partitioning = NondominatedPartitioning(ref_point=self.problem.get_ref_point(), Y=y_for_part)

        return qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.problem.get_ref_point().tolist(),
            partitioning=partitioning,
            sampler=sampler,
            # An empty list (no constraints) still reads as "not None" internally,
            # which triggers a codepath needing an eta buffer that only gets
            # registered when constraints is truthy - pass None instead of [].
            constraints=self.problem.get_constraints() or None,
        )

    def get_pareto_front(self) -> Tuple[torch.Tensor, torch.Tensor, List[dict]]:
        input_dim = self.train_X.shape[1] if self.train_X is not None else len(self.config.bounds)
        obj_dim = self.train_Y.shape[1] if self.train_Y is not None else len(self.config.objectives)

        if self.train_X is None or self.train_Y is None:
            empty_x = torch.empty((0, input_dim), dtype=self.dtype, device=self.device)
            empty_y = torch.empty((0, obj_dim), dtype=self.dtype, device=self.device)
            return empty_x, empty_y, []

        feasible = self.problem.get_feasible_mask(self.train_Y)  # shape (N,)
        if int(feasible.sum().item()) == 0:
            empty_x = self.train_X.new_empty((0, self.train_X.shape[1]))
            empty_y = self.train_Y.new_empty((0, self.train_Y.shape[1]))
            return empty_x, empty_y, []

        x_f = self.train_X[feasible]
        y_f = self.train_Y[feasible]
        raw_f = [r for r, ok in zip(self.train_raw, feasible.tolist()) if ok]

        pareto_mask = is_non_dominated(y_f)

        pareto_x = x_f[pareto_mask]
        pareto_y = y_f[pareto_mask]
        pareto_raw = [r for r, p in zip(raw_f, pareto_mask.tolist()) if p]

        return pareto_x, pareto_y, pareto_raw

    def generate_initial_data(self) -> None:
        sobol = torch.quasirandom.SobolEngine(dimension=len(self.config.bounds))
        x = sobol.draw(self.config.n_initial).to(dtype=self.dtype, device=self.device)

        bounds = torch.tensor(list(self.config.bounds.values()), dtype=self.dtype, device=self.device).T

        x = unnormalize(x, bounds)

        y, raw = self.problem.evaluate(x)

        self.train_X = x
        self.train_Y = y
        self.train_raw = list(raw)

    def _within_tolerance(self, raw) -> bool:
        for key, target in self.problem.targets.items():
            scale = self.problem.scales.get(key, 1.0)
            if abs(raw[key] - target) > scale:
                return False
        return True

    # ------------------------------------------------------------------
    # Trust-region search (config.use_trust_region=True): shrinks/expands a
    # window around the current incumbent based on whether recent batches
    # are actually improving on it, instead of searching the full `bounds`
    # box every iteration. This automates what the wide -> zoom -> zoom2 ->
    # zoom3 runs did by hand across this whole project: watch whether the
    # search is paying off nearby, and narrow in (or back off) accordingly.
    # ------------------------------------------------------------------

    def _tr_bounds_tensor(self) -> torch.Tensor:
        return torch.tensor(
            list(self.config.bounds.values()), dtype=self.dtype, device=self.device
        ).T

    def _tr_fail_tol(self) -> int:
        if self.config.tr_fail_tol is not None:
            return self.config.tr_fail_tol
        dim = len(self.config.bounds)
        return int(math.ceil(max(4.0 / self.config.batch_size, dim / self.config.batch_size)))

    def _tr_reset(self, center: torch.Tensor) -> None:
        """(Re)initialise the trust region around `center` (normalised [0,1]^d)."""
        self.tr_center = center.clone()
        self.tr_length = self.config.tr_length_init
        self.tr_succ_count = 0
        self.tr_fail_count = 0

    def _tr_window(self) -> torch.Tensor:
        """Current trust-region window, in physical units, for optimize_acqf."""
        half = self.tr_length / 2.0
        lower = torch.clamp(self.tr_center - half, 0.0, 1.0)
        upper = torch.clamp(self.tr_center + half, 0.0, 1.0)
        return unnormalize(torch.stack([lower, upper]), self._tr_bounds_tensor())

    def _hypervolume(self, y: torch.Tensor) -> float:
        if y.shape[0] == 0:
            return 0.0
        feasible = self.problem.get_feasible_mask(y)
        y_f = y[feasible]
        if y_f.shape[0] == 0:
            return 0.0
        pareto_mask = is_non_dominated(y_f)
        hv = Hypervolume(ref_point=self.problem.get_ref_point())
        return hv.compute(y_f[pareto_mask])

    def _tr_check_success(self, model, x_new: torch.Tensor, n_new: int) -> bool:
        """
        Did this batch actually improve things? Uses the GP's posterior mean
        at the new points rather than the raw observations, since BDSIM's
        finite-statistics noise can otherwise read a lucky/unlucky batch as a
        false success/failure.
        """
        with torch.no_grad():
            posterior_mean = model.posterior(x_new).mean

        y_before = self.train_Y[:-n_new]

        if self.config.mode == "scalar":
            # model was fit on _scalar_transform(y) already, so posterior_mean
            # comes back pre-transformed - only y_before needs transforming here.
            prior_best = self._scalar_transform(y_before).view(-1).max()
            new_best = posterior_mean.view(-1).max()
            return bool(new_best > prior_best + 1e-9)

        hv_before = self._hypervolume(y_before)
        hv_after = self._hypervolume(torch.cat([y_before, posterior_mean]))
        return hv_after > hv_before + 1e-9

    def _tr_recentre(self) -> None:
        """Move the trust-region centre to whichever point best represents
        the current incumbent: the best (scalar) loss, or (mobo) the Pareto
        point closest to the ideal point - i.e. the best simultaneous
        trade-off found so far, not just whichever objective improved most."""
        bounds = self._tr_bounds_tensor()

        if self.config.mode == "scalar":
            best_idx = torch.argmin(self.train_Y.view(-1))
            best_x = self.train_X[best_idx : best_idx + 1]
        else:
            xp, yp, _ = self.get_pareto_front()
            if xp.shape[0] == 0:
                return
            ideal = yp.max(dim=0).values
            dists = (yp - ideal).pow(2).sum(dim=-1)
            best_idx = torch.argmin(dists)
            best_x = xp[best_idx : best_idx + 1]

        self.tr_center = normalize(best_x, bounds).squeeze(0).clamp(0.0, 1.0)

    def _tr_update(self, model, x_new: torch.Tensor, n_new: int) -> None:
        improved = self._tr_check_success(model, x_new, n_new)

        if improved:
            self.tr_succ_count += 1
            self.tr_fail_count = 0
        else:
            self.tr_fail_count += 1
            self.tr_succ_count = 0

        if self.tr_succ_count >= self.config.tr_succ_tol:
            self.tr_length = min(self.tr_length * 2.0, self.config.tr_length_max)
            self.tr_succ_count = 0
        elif self.tr_fail_count >= self._tr_fail_tol():
            self.tr_length = self.tr_length / 2.0
            self.tr_fail_count = 0

        self._tr_recentre()

        if self.tr_length < self.config.tr_length_min:
            # Region has collapsed (converged locally) - restart fresh
            # elsewhere in the box to avoid getting stuck at a local optimum.
            bounds = self._tr_bounds_tensor()
            sobol = torch.quasirandom.SobolEngine(dimension=bounds.shape[1])
            fresh = sobol.draw(1).to(dtype=self.dtype, device=self.device).squeeze(0)
            self._write_log(
                f"[trust-region] window collapsed (length < {self.config.tr_length_min}); "
                f"restarting around a fresh point"
            )
            self._tr_reset(fresh)

    def optimise(self) -> None:
        self.generate_initial_data()

        if self.config.use_trust_region:
            dim = len(self.config.bounds)
            self._tr_reset(center=torch.full((dim,), 0.5, dtype=self.dtype, device=self.device))
            self._tr_recentre()  # immediately overwritten to the best initial point

        for it in range(self.config.n_iter):
            model = self.build_model(self.train_X, self.train_Y).to(self.device)
            acq = self.build_acquisition(model)

            if self.config.use_trust_region:
                bounds = self._tr_window()
                self._write_log(
                    f"[trust-region] length={self.tr_length:.4f}  bounds={bounds.tolist()}"
                )
            else:
                bounds = self._tr_bounds_tensor()

            x_new, _ = optimize_acqf(
                acq_function=acq,
                bounds=bounds,
                q=self.config.batch_size,
                num_restarts=10,
                raw_samples=64,
            )

            y_new, raw_new = self.problem.evaluate(x_new)

            self.train_X = torch.cat([self.train_X, x_new])
            self.train_Y = torch.cat([self.train_Y, y_new])
            self.train_raw.extend(raw_new)

            if self.config.use_trust_region:
                self._tr_update(model, x_new, n_new=x_new.shape[0])

            feasible = (
                self.problem.get_feasible_mask(y_new)
                if self.config.mode == "mobo"
                else torch.ones(y_new.shape[0], dtype=torch.bool)
            )

            for i in range(y_new.shape[0]):
                raw = raw_new[i]
                if raw is None:
                    prefix = "[CRASH]"
                    raw = "No output"
                elif not feasible[i]:
                    prefix = "[INFEASIBLE]"
                else:
                    prefix = "[OK]"

                self._write_log(prefix + " " + self.problem.format_result(x_new[i], y_new[i], raw))
                #if self.config.mode == "scalar" and feasible[i] and self._within_tolerance(raw):
                #    self._write_log("Target achieved within tolerance, stopping optimisation.")
                #    return

        if self.config.mode == "mobo":
            xp, yp, rawp = self.get_pareto_front()
            self._write_log("\n--- Optimisation Complete ---")
            self._write_log(f"Pareto front size: {xp.shape[0]}")
            for x, y, raw in zip(xp, yp, rawp):
                self._write_log(self.problem.format_result(x, y, raw))
        else:
            best = torch.argmin(self.train_Y)
            self._write_log("\n--- Optimisation Complete ---")
            self._write_log("Best solution:")
            self._write_log(self.problem.format_result(self.train_X[best], self.train_Y[best], self.train_raw[best]))

    def _run_candidates_at_high_stats(self, candidate_idx: List[int], x_np, ngenerate: int) -> List[Dict]:
        """
        Re-run the given training-set row indices at a higher ngenerate,
        restoring the original value afterward. Shared by the scalar top-k
        and MOBO Pareto-front refinement helpers below.
        """
        builder = self.problem.model.builder
        original_ngenerate = builder.Options["ngenerate"]
        builder.Options.SetNGenerate(ngenerate)

        results: List[Dict] = []
        try:
            with ProcessPoolExecutor(max_workers=min(len(candidate_idx), self.config.batch_size)) as pool:
                futures = {
                    pool.submit(self.problem.model.run, x_np[i], f"refine_{i}"): i
                    for i in candidate_idx
                }
                for fut in as_completed(futures):
                    i = futures[fut]
                    raw = fut.result()
                    results.append({"idx": i, "x": x_np[i], "raw": raw})
        finally:
            builder.Options.SetNGenerate(original_ngenerate)

        return results

    def refine_top_candidates(self, top_k: int = 5, ngenerate: int = 100_000) -> List[Dict]:
        """
        Re-run the top_k distinct best-loss candidates at much higher statistics.

        The search itself runs at a cheap, noisy particle count; this reruns
        only the shortlisted candidates at ngenerate to check them against the
        real physical tolerance without paying that cost on every evaluation.

        Scalar-mode only (ranks by a single loss column) - for mode="mobo"
        use refine_pareto_front instead.
        """
        x_np = self.train_X.detach().cpu().numpy()
        y_np = self.train_Y.view(-1).detach().cpu().numpy()

        # De-duplicate near-identical candidates so we don't re-verify the same point
        rounded = np.round(x_np, 4)
        _, unique_idx = np.unique(rounded, axis=0, return_index=True)
        ranked_idx = sorted(unique_idx.tolist(), key=lambda i: y_np[i])[:top_k]

        results = self._run_candidates_at_high_stats(ranked_idx, x_np, ngenerate)
        for r in results:
            r["search_loss"] = float(y_np[r["idx"]])
        results.sort(key=lambda r: r["search_loss"])

        self._write_log(f"\n--- High-statistics refinement (ngenerate={ngenerate}) ---")
        for r in results:
            metric_str = "  ".join(f"{k}={v:.4e}" for k, v in r["raw"].items())
            self._write_log(f"X={r['x']}  search_loss={r['search_loss']:.3e}  {metric_str}")

        return results

    def refine_pareto_front(self, ngenerate: int = 100_000, max_candidates: Optional[int] = None) -> List[Dict]:
        """
        Re-run every (or up to max_candidates) distinct point currently on the
        Pareto front at much higher statistics.

        Mode="mobo" analogue of refine_top_candidates: rather than ranking by
        a single loss, this takes whatever the current non-dominated set is
        and checks each of those trade-off points against real statistics,
        since the search itself runs cheap and noisy.
        """
        xp, yp, _ = self.get_pareto_front()
        if xp.shape[0] == 0:
            self._write_log("\n--- High-statistics Pareto refinement: front is empty, nothing to refine ---")
            return []

        x_np = self.train_X.detach().cpu().numpy()
        xp_np = xp.detach().cpu().numpy()

        # Map each Pareto point back to its row in train_X (de-duplicating
        # near-identical x so we don't re-verify the same point twice)
        rounded_all = np.round(x_np, 6)
        pareto_idx: List[int] = []
        seen = set()
        for row in np.round(xp_np, 6):
            matches = np.where((rounded_all == row).all(axis=1))[0]
            if matches.size == 0:
                continue
            key = tuple(row.tolist())
            if key in seen:
                continue
            seen.add(key)
            pareto_idx.append(int(matches[0]))

        if max_candidates is not None:
            pareto_idx = pareto_idx[:max_candidates]

        results = self._run_candidates_at_high_stats(pareto_idx, x_np, ngenerate)
        for r in results:
            r["search_y"] = self.train_Y[r["idx"]].detach().cpu().tolist()
            r["refined_y"] = self.problem.pack_objectives(r["raw"]).detach().cpu().tolist()

        self._write_log(f"\n--- High-statistics Pareto-front refinement (ngenerate={ngenerate}) ---")
        self._write_log(f"Refining {len(results)} of {xp.shape[0]} Pareto points")
        for r in results:
            metric_str = "  ".join(f"{k}={v:.4e}" for k, v in r["raw"].items())
            self._write_log(
                f"X={r['x']}  search_y={r['search_y']}  refined_y={r['refined_y']}  {metric_str}"
            )

        return results

    def plot_results(self) -> None:
        if self.config.mode == "scalar":
            losses = self.train_Y.view(-1).cpu().numpy()
            best_so_far = np.minimum.accumulate(losses)

            plt.figure()
            plt.semilogy(best_so_far)
            plt.xlabel("Evaluation")
            plt.ylabel("Best Loss")
            plt.title("Scalar BO Convergence")
            plt.tight_layout()
            plt.savefig(str(self.run_dir / "best_loss.png"))
            return

        self._plot_pareto()

    def _plot_pareto(self) -> None:
        xp, yp, rawp = self.get_pareto_front()
        labels = self.problem.objective_labels()

        feasible = self.problem.get_feasible_mask(self.train_Y)
        y_f = self.train_Y[feasible]

        obj_dim = y_f.shape[-1]
        fig = plt.figure(figsize=(8, 6))

        if obj_dim == 2:
            ax = fig.add_subplot(111)
            ax.scatter(y_f[:, 0], y_f[:, 1], alpha=0.3, label="Feasible")
            ax.scatter(yp[:, 0], yp[:, 1], s=80, label="Pareto")
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
        elif obj_dim == 3:
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(y_f[:, 0], y_f[:, 1], y_f[:, 2], alpha=0.3, label="Feasible")
            ax.scatter(yp[:, 0], yp[:, 1], yp[:, 2], s=80, label="Pareto")
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_zlabel(labels[2])
        else:
            raise ValueError(f"_plot_pareto only supports 2 or 3 objectives, got {obj_dim}")

        ax.legend()
        plt.tight_layout()
        plt.savefig(self.run_dir / "pareto.png")
