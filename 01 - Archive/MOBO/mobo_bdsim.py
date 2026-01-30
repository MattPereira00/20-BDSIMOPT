import torch
from torch import Tensor
import os, subprocess
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed


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
from builder import *


class BDSIMMoBO:
    def __init__(
        self,
        builder:Builder,
        model:str,
        filename:str,
        n_initial=10,
        n_iter=100,
        batch_size=4,
        mc_samples=256,
        device="cpu"
    ):
        self.builder = builder
        self.model = model
        self.filename = filename
        self.runDir = self.builder.data_dir + "/" + self.filename
        os.makedirs(self.runDir, exist_ok=True)
        self.logfile = os.path.join(self.runDir, f"{self.filename}_optlog.txt")

        self.device = torch.device(device)
        self.dtype = torch.double

        # BO hyperparameters
        self.n_initial = n_initial
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.mc_samples = mc_samples
        self.run_id = 0
        self.T_min = 0.01

        # Parameter bounds: 3 drift lengths, 3 quad lengths, 3 quad apertures - min then max
        self.bounds = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.03, 0.03, 0.03, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.03, 0.03, 0.03, 0.01, 0.01, 0.01],
                [0.5, 0.5, 0.5, 0.05, 0.05, 0.05, 0.03, 0.03, 0.03, 0.5, 0.5, 0.5, 0.05, 0.05, 0.05, 0.03, 0.03, 0.03],
            ],
            dtype=self.dtype,
            device=self.device,
        )

        # EHVI reference point: (T, -A, -D) - the worst possible result outside of feasibility
        self.ref_point = torch.tensor([0.0, 1.0, 1.0], dtype=self.dtype)

        # Containers for training data
        self.train_X = None
        self.train_Y = None

    @property
    def n_objectives(self):
        return len(self.objectives)

    def _write_log(self, text):
        """Helper to write text to log file and stdout"""
        print(text)
        with open(self.logfile, "a") as f:
            f.write(text + "\n")

    def _sanitize(self, T, A, D):
        if not np.isfinite(T) or not np.isfinite(A) or not np.isfinite(D):
            return 0.0, 1.0, 1.0  # worst possible, dominated

        return T, A, D

    def _run(self, lengths, index, cleanup=True):

        # Run BDSIM (temporary file)
        modelpath = f"{self.builder.model_dir}/{self.model.casefold()}_{index}.gmad"
        outfile = f"{self.runDir}/output-{index}"

        self.builder.build_halbach_double_triplet(index, lengths)
        n_generate = self.builder.Options["ngenerate"]

        pybdsim.Run.Bdsim(
            gmadpath=modelpath,
            outfile=outfile,
            batch=True,
            seed=1999,
            silent=True,
            ngenerate=n_generate,
        )

        subprocess.call(["rebdsimOptics", f"{outfile}.root", f"{outfile}_optics.root", "--emittanceOnTheFly"],
                               stdout=open(os.devnull, 'wb'))

        # Extract variables for metrics from RebdsimOptics File
        T, A, D = extract_metrics_and_uncertainties(outfile)[::2]
        T, A, D = self._sanitize(T, A, D)

        # Cleanup
        if cleanup:
            os.remove(modelpath)
            os.remove(f"{self.builder.model_dir}/{self.model.casefold()}_{index}_beam.gmad")
            os.remove(f"{self.builder.model_dir}/{self.model.casefold()}_{index}_components.gmad")
            os.remove(f"{self.builder.model_dir}/{self.model.casefold()}_{index}_options.gmad")
            os.remove(f"{self.builder.model_dir}/{self.model.casefold()}_{index}_sequence.gmad")
            os.remove(outfile + ".root")
            os.remove(outfile + "_optics.root")

        return T, A, D

    def evaluate_objectives(self, X: Tensor):
        """
        Parallel evaluation of objective vectors.
        Returns Y = [T, -A, -D] for each row in X.
        """
        X_np = X.cpu().numpy()
        n = X_np.shape[0]

        results = [None] * n

        with ProcessPoolExecutor(max_workers=self.batch_size) as pool:
            futures = {}
            for i in range(n):
                run_id = self.run_id
                self.run_id += 1
                futures[pool.submit(self._run, X_np[i], run_id)] = i

            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    T, A, D = fut.result()
                except Exception as e:
                    print(f"[Worker {i}] ERROR:", e)
                    T, A, D = 0.0, 1.0, 1.0  # bad fallback
                results[i] = [T, -A, -D]

        return torch.tensor(results, dtype=self.dtype, device=self.device)

    def transmission_constraint(self, Y: Tensor):
        return Y[..., 0] - self.T_min  # >=0 if T >= T_min

    def build_model(self, X, Y):
        models = []
        for i in range(Y.shape[-1]):
            gp = SingleTaskGP(X, Y[..., i: i + 1])
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            models.append(gp)
        return ModelListGP(*models)

    def get_pareto_front(self):
        # Feasibility mask
        feasible_mask = self.train_Y[:, 0] >= self.T_min

        X_feasible = self.train_X[feasible_mask]
        Y_feasible = self.train_Y[feasible_mask]

        if Y_feasible.numel() == 0:
            return X_feasible, Y_feasible

        pareto_mask = is_non_dominated(Y_feasible)

        pareto_X = X_feasible[pareto_mask]
        pareto_Y = Y_feasible[pareto_mask]

        return pareto_X, pareto_Y

    def generate_initial_data(self):
        sobol = torch.quasirandom.SobolEngine(dimension=18)
        X = sobol.draw(self.n_initial).to(dtype=self.dtype, device=self.device)
        X = unnormalize(X, self.bounds)

        Y = self.evaluate_objectives(X)

        self.train_X = X
        self.train_Y = Y

    def optimise(self):
        # Initial SOBOL points
        self.generate_initial_data()

        # Batches
        for it in range(1, self.n_iter + 1):
            model = self.build_model(self.train_X, self.train_Y).to(self.device)
            partitioning = NondominatedPartitioning(
                ref_point=self.ref_point,
                Y=self.train_Y,
            )
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_samples]))
            acq = qExpectedHypervolumeImprovement(
                model=model,
                ref_point=self.ref_point.tolist(),
                sampler=sampler,
                partitioning=partitioning,
                constraints=[self.transmission_constraint],
            )

            X_new, _ = optimize_acqf(
                acq_function=acq, bounds=self.bounds, q=self.batch_size,
                num_restarts=10, raw_samples=64,
            )

            Y_new = self.evaluate_objectives(X_new)

            self.train_X = torch.cat([self.train_X, X_new])
            self.train_Y = torch.cat([self.train_Y, Y_new])

            # Log results
            for i in range(self.batch_size):
                T = float(Y_new[i, 0])
                if T < self.T_min:
                    self._write_log(f"Infeasible candidate  T={T:.4f}")
                    continue
                A = float(-Y_new[i, 1])
                D = float(-Y_new[i, 2])
                self._write_log(
                    f"T={T:.4f}  A={A:.4f}  D={D:.4f}   "
                    f"X={X_new[i].cpu().numpy()}"
                )
        # Log the Pareto Front
        pareto_X, pareto_Y = self.get_pareto_front()
        self._write_log("\n--- Optimisation Complete ---")
        self._write_log(f"Pareto front size: {pareto_X.shape[0]}")

        for i, (x, y) in enumerate(zip(pareto_X, pareto_Y)):
            T = float(y[0])
            A = float(-y[1])
            D = float(-y[2])
            self._write_log(
                f"T={T:.4f}  A={A:.4f}  D={D:.4f}   "
                f"X={x.cpu().numpy()}"
            )

        # Plot the Pareto Front

        feasible_mask = self.train_Y[:, 0] >= self.T_min
        Y_feasible = self.train_Y[feasible_mask]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(Y_feasible[:, 0], -Y_feasible[:, 1], -Y_feasible[:, 2], c='gray', alpha=0.5, label='Non-Pareto')
        ax.scatter(pareto_Y[:, 0], -pareto_Y[:, 1], -pareto_Y[:, 2], c='red', s=80, label='Pareto')

        ax.set_xlabel("Transmission T")
        ax.set_ylabel("Asymmetry A")
        ax.set_zlabel("Divergence D")
        ax.set_title(f"Pareto Front - {self.filename}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{self.runDir}/3D_pareto_front.png")