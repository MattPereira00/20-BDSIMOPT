from builder import *
from loss import *
import pybdsim
import os, subprocess
import numpy as np
if not hasattr(np, "int"):
    np.int = int
from skopt import gp_minimize
from skopt.space import Real
import matplotlib.pyplot as plt
import pandas as pd


class BDSIMBayes:
    def __init__(self, builder:Builder, model:str, filename:str):
        self.builder = builder
        self.model = model
        self.filename = filename
        self.runDir = self.builder.data_dir + "/" + self.filename
        os.makedirs(self.runDir, exist_ok=True)
        self.logfile = os.path.join(self.runDir, f"{self.filename}_optlog.txt")

        self.call_count = None
        self.results = None

        assert model.casefold() in ['doublet', 'triplet'], "Invalid model name - must be 'doublet' or 'triplet'"

    def _write_log(self, text):
        """Helper to write text to log file and stdout"""
        print(text)
        with open(self.logfile, "a") as f:
            f.write(text + "\n")

    def _run(self, filename, n_generate, cleanup=True):

        # Run BDSIM (temporary file)
        modelpath = f"{self.builder.model_dir}/{self.model.casefold()}_{filename}.gmad"
        outfile = f"{self.runDir}/output-{filename}"

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
        transmission, asymmetry, mean_alpha, loss = extract_loss(outfile)


        # Cleanup
        if cleanup:
            os.remove(modelpath)
            os.remove(f"{self.builder.model_dir}/{self.model.casefold()}_{filename}_beam.gmad")
            os.remove(f"{self.builder.model_dir}/{self.model.casefold()}_{filename}_components.gmad")
            os.remove(f"{self.builder.model_dir}/{self.model.casefold()}_{filename}_options.gmad")
            os.remove(f"{self.builder.model_dir}/{self.model.casefold()}_{filename}_sequence.gmad")
            os.remove(outfile + ".root")
            os.remove(outfile + "_optics.root")

        return transmission, asymmetry, mean_alpha, loss

    def run100k(self):
        transmission, asymmetry, mean_alpha, loss = self._run("Optimised_100k", 100000, cleanup=False)
        
        self._write_log(f"--- 100k Validation of {self.model.casefold()} ---")
        self._write_log(f"transmission = {transmission:.4f}, asymmetry = {asymmetry:.4f},"
                        f" mean_alpha= {mean_alpha:.4f}, loss = {loss:.4f}")


    def objective(self, x_candidate):
        model_name = "BO_candidate"
        if self.model.casefold() == 'triplet':
            self.builder.build_pmq_triplet(model_name, np.array(x_candidate))

        if self.model.casefold() == 'doublet':
            self.builder.build_pmq_doublet(model_name, np.array(x_candidate))

        n_generate = self.builder.Options["ngenerate"]
        transmission, asymmetry, mean_alpha, loss = self._run(model_name, n_generate, cleanup=True)

        # Handle non-finite results
        if not np.isfinite(loss):
            return 1e6  # large penalty to discourage invalid configurations

        # Store results
        self.call_count += 1
        self.results.append((self.call_count, x_candidate, transmission, asymmetry, mean_alpha, loss))

        # Log every 10 evaluations
        if self.call_count % 10 == 0:
            self._write_log(
                f"[{self.call_count:03d}] loss={loss:.4f}, transmission={transmission:.4f}, asymmetry={asymmetry:.4f},"
                f" mean_alpha={mean_alpha:.4f}, X={np.round(x_candidate,4)}"
            )

        return loss

    def optimise(self, drift_length_max:float, n_calls:int):
        # Define bounds (never negative, quad lengths categorical)
        bounds = []
        if self.model.casefold() == 'doublet':
            bounds = [
                Real(0.0, drift_length_max),
                Real(0.0, drift_length_max),
                Real(0.02, 0.04),
                Real(0.02, 0.04)
            ]

        if self.model.casefold() == 'triplet':
            bounds = [
                Real(0.0, drift_length_max),
                Real(0.0, drift_length_max),
                Real(0.0, drift_length_max),
                Real(0.02, 0.04),
                Real(0.02, 0.04),
                Real(0.02, 0.04)
            ]


        self.call_count = 0
        self.results = []

        if os.path.exists(self.logfile):
            os.remove(self.logfile)
        self._write_log(f"--- Bayesian Optimization Run ({self.model}) ---")
        self._write_log(f"Max drift length: {drift_length_max}, n_calls: {n_calls}\n")

        result = gp_minimize(
            func=self.objective,
            dimensions=bounds,
            n_calls=n_calls,
            n_initial_points=10,
            initial_point_generator="lhs",
            acq_func="EI",  # Expected Negative improvement
            random_state=1999
        )

        print("--- Bayesian Optimization Complete ---")
        print("Best input configuration (X_pred):", np.round(result.x, 3))
        print("Predicted reward (|reward|):", result.fun)

        # run final BDSIM to validate
        x_pred = np.array(result.x)
        if self.model.casefold() == 'triplet':
            self.builder.build_pmq_triplet("Optimised", x_pred)

        if self.model.casefold() == 'doublet':
            self.builder.build_pmq_doublet("Optimised", x_pred)

        pred_model = f"{self.builder.model_dir}/{self.model.casefold()}_Optimised.gmad"
        pred_outfile = f"{self.runDir}/output-opt"
        pybdsim.Run.Bdsim(
            gmadpath=pred_model,
            outfile=pred_outfile,
            batch=True,
            seed=1999,
            silent=True,
            ngenerate=self.builder.Options["ngenerate"]
        )

        subprocess.call(["rebdsimOptics", f"{pred_outfile}.root", f"{pred_outfile}_optics.root", "--emittanceOnTheFly"],
                               stdout=open(os.devnull, 'wb'))

        T, Sigma_T, A, Sigma_A, D, Sigma_D = extract_metrics_and_uncertainties(pred_outfile)

        # Create and write a .csv of ALL the results, each call and its results.
        df = pd.DataFrame(self.results, columns=["call", "X", "transmission", "asymmetry", "mean_alpha", "loss"])
        csv_path = os.path.join(self.runDir, f"{self.model}_results.csv")
        df.to_csv(csv_path, index=False)
        self._write_log(f"\nSaved detailed results to: {csv_path}")

        # Give and write final opt result to summary log
        self._write_log("\n--- Optimization Complete ---")
        self._write_log(f"Optimised Configuration: {np.round(x_pred, 4)}")
        self._write_log(f"transmission = {T:.3f} ± {Sigma_T:.3f}, asymmetry = {A:.3f} ± {Sigma_A:.3f},"
                        f"divergence= {D:.3f} ± {Sigma_D:.3f},")

        # Plot the call vs loss
        calls = [r[0] for r in self.results]
        losses = [r[5] for r in self.results]

        # Compute running minimum (best loss so far)
        best_so_far = []
        current_best = float('inf')
        for l in losses:
            current_best = min(current_best, l)
            best_so_far.append(current_best)

        # Plot convergence ignoring penalised samples
        plt.figure(figsize=(6, 4))
        plt.plot(calls, best_so_far)
        plt.xlabel("Evaluation")
        plt.ylabel("Best Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.runDir, "convergence.png"), dpi=200)
        plt.close()

