import torch
import os
import matplotlib.pyplot as plt

from opt_config import *
from opt_model import *
from opt_problem import *

from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood


from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.transforms import unnormalize
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning

from loss import *
from builder import Builder


class BDSIMMoBO:
    def __init__(self, problem: OptProblem, filename: str):
            self.problem = problem
            self.config = problem.config
            self.filename = filename

            self.device = self.config.device
            self.dtype = torch.double

            self.runDir = os.path.join(self.problem.model.builder.data_dir, filename)
            os.makedirs(self.runDir, exist_ok=True)
            self.logfile = os.path.join(self.runDir, f"{filename}_optlog.txt")

            self.train_X = None
            self.train_Y = None
            self.train_raw = None

    def _write_log(self, text):
        print(text)
        with open(self.logfile, "a") as f:
            f.write(text + "\n")


    def _sanitize(self, T, A, D):
        if not np.isfinite(T) or not np.isfinite(A) or not np.isfinite(D):
            return 0.0, 1.0, 1.0  # worst possible, dominated

        return T, A, D

    def build_model(self, X, Y):
        models = []
        for i in range(Y.shape[-1]):
            gp = SingleTaskGP(X, Y[..., i:i+1])
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            models.append(gp)
        return ModelListGP(*models)

    def get_pareto_front(self):
        """
        Returns:
            pareto_X : Tensor [M, d]
            pareto_Y : Tensor [M, m]
            pareto_raw : list[dict]
        """

        # 1. Feasibility mask over ALL evaluated points
        feasible = self.problem.get_feasible_mask(self.train_Y)  # shape (N,)

        if feasible.sum() == 0:
            return (
                self.train_X.new_empty((0, self.train_X.shape[1])),
                self.train_Y.new_empty((0, self.train_Y.shape[1])),
                [],
            )

        # 2. Filter feasible points
        X_f = self.train_X[feasible]
        Y_f = self.train_Y[feasible]
        raw_f = [r for r, ok in zip(self.train_raw, feasible.tolist()) if ok]

        # 3. Pareto dominance on OBJECTIVES
        pareto_mask = is_non_dominated(Y_f)

        # 4. Final Pareto sets
        pareto_X = X_f[pareto_mask]
        pareto_Y = Y_f[pareto_mask]
        pareto_raw = [r for r, p in zip(raw_f, pareto_mask.tolist()) if p]

        return pareto_X, pareto_Y, pareto_raw

    def generate_initial_data(self):
        sobol = torch.quasirandom.SobolEngine(dimension=len(self.config.bounds))
        X = sobol.draw(self.config.n_initial).to(
            dtype=self.dtype,
            device=self.device,
        )

        bounds = torch.tensor(
            list(self.config.bounds.values()),
            dtype=self.dtype,
            device=self.device,
        ).T

        X = unnormalize(X, bounds)

        Y, raw = self.problem.evaluate(X)

        self.train_X = X
        self.train_Y = Y
        self.train_raw = list(raw)

    def optimise(self):
        self.generate_initial_data()

        for it in range(self.config.n_iter):
            model = self.build_model(self.train_X, self.train_Y).to(self.device)

            # Get current feasible Pareto front
            _, pareto_Y, _ = self.get_pareto_front()

            # If no feasible Pareto points, pass an empty tensor with correct shape/dtype,
            # otherwise use the pareto front
            if pareto_Y.shape[0] == 0:
                Y_for_part = self.train_Y.new_empty((0, self.train_Y.shape[1]))
            else:
                Y_for_part = pareto_Y

            partitioning = NondominatedPartitioning(
                ref_point=self.problem.get_ref_point(),
                Y=Y_for_part,
            )

            # partitioning = NondominatedPartitioning(
            #     ref_point=self.problem.get_ref_point(),
            #     Y=self.train_Y,
            # )

            sampler = SobolQMCNormalSampler(
                sample_shape=torch.Size([self.config.mc_samples])
            )

            acq = qExpectedHypervolumeImprovement(
                model=model,
                ref_point=self.problem.get_ref_point().tolist(),
                partitioning=partitioning,
                sampler=sampler,
                constraints=self.problem.get_constraints(),
            )

            #acq.eta = 1e-3

            bounds = torch.tensor(
                list(self.config.bounds.values()),
                dtype=self.dtype,
                device=self.device,
            ).T

            X_new, _ = optimize_acqf(
                acq_function=acq,
                bounds=bounds,
                q=self.config.batch_size,
                num_restarts=10,
                raw_samples=64,
            )

            Y_new, raw_new = self.problem.evaluate(X_new)

            self.train_X = torch.cat([self.train_X, X_new])
            self.train_Y = torch.cat([self.train_Y, Y_new])
            self.train_raw.extend(raw_new)

            feasible = self.problem.get_feasible_mask(Y_new)

            for i in range(Y_new.shape[0]):
                raw = raw_new[i]

                if raw is None:
                    prefix = "[CRASH]"
                    raw = {"T": 0.0, "A": 1e3, "D": 1e3}
                elif not feasible[i]:
                    prefix = "[INFEASIBLE]"
                else:
                    prefix = "[OK]"

                self._write_log(
                    prefix + " " +
                    self.problem.format_result(X_new[i], Y_new[i], raw)
                )

        # Log the Pareto Front
        Xp, Yp, rawp = self.get_pareto_front()
        self._write_log("\n--- Optimisation Complete ---")
        self._write_log(f"Pareto front size: {Xp.shape[0]}")

        for x, y, raw in zip(Xp, Yp, rawp):
            self._write_log(self.problem.format_result(x, y, raw))


    def plot_pareto(self):
        Xp, Yp, rawp = self.get_pareto_front()
        labels = self.problem.objective_labels()

        feasible = self.problem.get_feasible_mask(self.train_Y)

        Y_f = self.train_Y[feasible]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            Y_f[:, 0],
            Y_f[:, 1],
            Y_f[:, 2],
            alpha=0.3,
            label="Feasible"
        )

        ax.scatter(
            Yp[:, 0],
            Yp[:, 1],
            Yp[:, 2],
            s=80,
            label="Pareto"
        )

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        ax.legend()

        plt.tight_layout()
        plt.savefig(f"{self.runDir}/pareto.png")
