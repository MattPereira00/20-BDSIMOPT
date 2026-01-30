from builder import *
from loss import *
import os, subprocess

from abc import ABC, abstractmethod
import numpy as np


def sanitize(T, A, D):
    if not np.isfinite(T) or not np.isfinite(A) or not np.isfinite(D):
        return 0.0, 1.0, 1.0  # worst possible, dominated

    return T, A, D

class OptModel(ABC):
    @abstractmethod
    def run(self, x: np.ndarray, run_id: int) -> dict:
        pass

class TripletModel(OptModel):
    def __init__(self, builder:Builder):
        self.builder = builder
        self.model_dir = self.builder.model_dir
        self.data_dir = self.builder.data_dir

    def run(self, x, run_id, cleanup=True):
        """
        :param x:
        :param run_id:
        :param cleanup:
        :return: dict of objective values after run
        """
        self.builder.build_halbach_triplet(run_id, x)

        modelpath = f"{self.builder.model_dir}/triplet_{run_id}.gmad"
        outfile = f"{self.data_dir}/output-{run_id}"

        # Run BDSIM (temporary files)
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
        T, A, D = sanitize(T, A, D)

        # Cleanup
        if cleanup:
            os.remove(modelpath)
            os.remove(f"{self.builder.model_dir}/triplet_{run_id}_beam.gmad")
            os.remove(f"{self.builder.model_dir}/triplet_{run_id}_components.gmad")
            os.remove(f"{self.builder.model_dir}/triplet_{run_id}_options.gmad")
            os.remove(f"{self.builder.model_dir}/triplet_{run_id}_sequence.gmad")
            os.remove(outfile + ".root")
            os.remove(outfile + "_optics.root")

        return {"T": T, "A": A, "D": D}

class DoubleTripletModel(OptModel):
    def __init__(self, builder:Builder):
        self.builder = builder
        self.model_dir = self.builder.model_dir
        self.data_dir = self.builder.data_dir

    def run(self, X, run_id, cleanup=True):
        """
        :param X:
        :param run_id:
        :param cleanup:
        :return: dict of objective values after run
        """
        self.builder.build_halbach_double_triplet(run_id, X)

        modelpath = f"{self.builder.model_dir}/double_triplet_{run_id}.gmad"
        outfile = f"{self.data_dir}/output-{run_id}"

        # Run BDSIM (temporary files)
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
        T, A, D = sanitize(T, A, D)

        # Cleanup
        if cleanup:
            os.remove(modelpath)
            os.remove(f"{self.builder.model_dir}/double_triplet_{run_id}_beam.gmad")
            os.remove(f"{self.builder.model_dir}/double_triplet_{run_id}_components.gmad")
            os.remove(f"{self.builder.model_dir}/double_triplet_{run_id}_options.gmad")
            os.remove(f"{self.builder.model_dir}/double_triplet_{run_id}_sequence.gmad")
            os.remove(outfile + ".root")
            os.remove(outfile + "_optics.root")

        return {"T": T, "A": A, "D": D}