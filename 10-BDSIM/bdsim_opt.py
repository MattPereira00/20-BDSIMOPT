import logging
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood


from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.transforms import unnormalize
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

    def _write_log(self, text: str) -> None:
        self.logger.info(text)

    def build_model(self, x, y):
        if self.config.mode == "scalar":
            gp = SingleTaskGP(x, y)
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
            best_f = self.train_Y.view(-1).min().item()
            return qLogExpectedImprovement(model=model, best_f=best_f, sampler=sampler, eta=1e-3)

        _, pareto_y, _ = self.get_pareto_front()
        y_for_part = pareto_y if pareto_y.shape[0] > 0 else self.train_Y.new_empty((0, self.train_Y.shape[1]))

        partitioning = NondominatedPartitioning(ref_point=self.problem.get_ref_point(), Y=y_for_part)

        return qExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.problem.get_ref_point().tolist(),
            partitioning=partitioning,
            sampler=sampler,
            constraints=self.problem.get_constraints(),
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

    def optimise(self) -> None:
        self.generate_initial_data()

        for it in range(self.config.n_iter):
            model = self.build_model(self.train_X, self.train_Y).to(self.device)
            acq = self.build_acquisition(model)

            bounds = torch.tensor(
                list(self.config.bounds.values()),
                dtype=self.dtype,
                device=self.device,
            ).T

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
                if self.config.mode == "scalar" and feasible[i] and self._within_tolerance(raw):
                    self._write_log("Target achieved within tolerance, stopping optimisation.")
                    return

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
            plt.savefig(str(self.runDir / "best_loss.png"))
            return

        self._plot_pareto()

    def _plot_pareto(self) -> None:
        xp, yp, rawp = self.get_pareto_front()
        labels = self.problem.objective_labels()

        feasible = self.problem.get_feasible_mask(self.train_Y)
        y_f = self.train_Y[feasible]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(y_f[:, 0], y_f[:, 1], y_f[:, 2], alpha=0.3, label="Feasible")
        ax.scatter(yp[:, 0], yp[:, 1], yp[:, 2], s=80, label="Pareto")

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.runDir / "pareto.png")
