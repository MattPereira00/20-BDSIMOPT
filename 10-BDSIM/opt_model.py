from builder import *
from opt_obj import *
import os, subprocess

from abc import ABC, abstractmethod
import numpy as np


def sanitize(t, a, d):
    if not np.isfinite(t) or not np.isfinite(a) or not np.isfinite(d):
        return 0.0, 1.0, 1.0  # worst possible, dominated

    return t, a, d

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
        self.builder.build_halbach_triplet(run_id, x)

        model_path = f"{self.builder.model_dir}/triplet_{run_id}.gmad"
        outfile = f"{self.data_dir}/output-{run_id}"

        n_generate = self.builder.Options["ngenerate"]
        pybdsim.Run.Bdsim(
            gmadpath=model_path,
            outfile=outfile,
            batch=True,
            seed=1999,
            silent=True,
            ngenerate=n_generate,
        )

        subprocess.call(["rebdsimOptics", f"{outfile}.root", f"{outfile}_optics.root", "--emittanceOnTheFly"],
                               stdout=open(os.devnull, 'wb'))

        # Extract variables for metrics from RebdsimOptics File
        t, a, d = extract_metrics_and_uncertainties(outfile)[::2]
        t, a, d = sanitize(t, a, d)

        # Cleanup
        if cleanup:
            os.remove(model_path)
            os.remove(f"{self.builder.model_dir}/triplet_{run_id}_beam.gmad")
            os.remove(f"{self.builder.model_dir}/triplet_{run_id}_components.gmad")
            os.remove(f"{self.builder.model_dir}/triplet_{run_id}_options.gmad")
            os.remove(f"{self.builder.model_dir}/triplet_{run_id}_sequence.gmad")
            os.remove(outfile + ".root")
            os.remove(outfile + "_optics.root")

        return {"T": t, "A": a, "D": d}

class DoubleTripletModel(OptModel):
    def __init__(self, builder:Builder, frozen_triplet):
        self.builder = builder
        self.model_dir = self.builder.model_dir
        self.data_dir = self.builder.data_dir
        self.fixed = frozen_triplet

    def run(self, x, run_id, cleanup=True):
        """
        :param x:
        :param run_id:
        :param cleanup:
        :return: dict of objective values after run
        """

        triplet2 = self.unpack_triplet2(x)

        params = {}
        params.update(self.fixed)
        params.update(triplet2)

        self.builder.build_halbach_double_triplet(run_id, self.pack_params(params))

        model_path = f"{self.builder.model_dir}/double_triplet_{run_id}.gmad"
        outfile = f"{self.data_dir}/output-{run_id}"

        # Run BDSIM (temporary files)
        n_generate = self.builder.Options["ngenerate"]
        pybdsim.Run.Bdsim(
            gmadpath=model_path,
            outfile=outfile,
            batch=True,
            seed=1999,
            silent=True,
            ngenerate=n_generate,
        )

        subprocess.call(["rebdsimOptics", f"{outfile}.root", f"{outfile}_optics.root", "--emittanceOnTheFly"],
                               stdout=open(os.devnull, 'wb'))

        # Extract variables for metrics from RebdsimOptics File
        t, a, d = extract_metrics_and_uncertainties(outfile)[::2]
        t, a, d = sanitize(t, a, d)

        # Cleanup
        if cleanup:
            os.remove(model_path)
            os.remove(f"{self.builder.model_dir}/double_triplet_{run_id}_beam.gmad")
            os.remove(f"{self.builder.model_dir}/double_triplet_{run_id}_components.gmad")
            os.remove(f"{self.builder.model_dir}/double_triplet_{run_id}_options.gmad")
            os.remove(f"{self.builder.model_dir}/double_triplet_{run_id}_sequence.gmad")
            os.remove(outfile + ".root")
            os.remove(outfile + "_optics.root")

        return {"T": t, "A": a, "D": d}

    @staticmethod
    def unpack_triplet2(x):
        return {
            "d4": float(x[0]),
            "d5": float(x[1]),
            "d6": float(x[2]),
            "q4_L": float(x[3]),
            "q5_L": float(x[4]),
            "q6_L": float(x[5]),
            "aper4": float(x[6]),
            "aper5": float(x[7]),
            "aper6": float(x[8]),
        }

    @staticmethod
    def pack_params(params):
        return [
            params["d1"], params["d2"], params["d3"],
            params["q1_L"], params["q2_L"], params["q3_L"],
            params["aper1"], params["aper2"], params["aper3"],
            params["d4"], params["d5"], params["d6"],
            params["q4_L"], params["q5_L"], params["q6_L"],
            params["aper4"], params["aper5"], params["aper6"],
        ]

class S1GLModel(OptModel):
    def __init__(self, builder):
        self.builder = builder
        self.model_dir = builder.model_dir
        self.data_dir = builder.data_dir

    def run(self, x, run_id, cleanup=True):
        # X = [b4, b5, b6, b7]
        self.builder.build_s1_gl(run_id, x)

        model_path = f"{self.model_dir}/S1GL_{run_id}.gmad"
        outfile = f"{self.data_dir}/output-{run_id}"

        pybdsim.Run.Bdsim(
            gmadpath=model_path,
            outfile=outfile,
            batch=True,
            seed=1999,
            silent=True,
            ngenerate=self.builder.Options["ngenerate"],
        )

        subprocess.call([
            "rebdsimOptics",
            f"{outfile}.root",
            f"{outfile}_optics.root",
            "--emittanceOnTheFly"
        ], stdout=open(os.devnull, "wb"))

        sigma_x, sigma_y, sigma_xp, sigma_yp = extract_optics(outfile)

        # Cleanup
        if cleanup:
            os.remove(model_path)
            os.remove(f"{self.builder.model_dir}/S1GL_{run_id}_beam.gmad")
            os.remove(f"{self.builder.model_dir}/S1GL_{run_id}_components.gmad")
            os.remove(f"{self.builder.model_dir}/S1GL_{run_id}_options.gmad")
            os.remove(f"{self.builder.model_dir}/S1GL_{run_id}_sequence.gmad")
            os.remove(outfile + ".root")
            os.remove(outfile + "_optics.root")

        return {"sigma_x": sigma_x, "sigma_y": sigma_y, "sigma_xp": sigma_xp, "sigma_yp": sigma_yp}

class S1ColModel(OptModel):
    def __init__(self, builder:Builder):
        self.builder = builder
        self.model_dir = self.builder.model_dir
        self.data_dir = self.builder.data_dir

    def run(self, x, run_id, cleanup=True):
        self.builder.build_s1_col1(run_id, x)

        model_path = f"{self.builder.model_dir}/S1_COL1_{run_id}.gmad"
        outfile = f"{self.data_dir}/output-{run_id}"

        n_generate = self.builder.Options["ngenerate"]
        pybdsim.Run.Bdsim(
            gmadpath=model_path,
            outfile=outfile,
            batch=True,
            seed=1999,
            silent=True,
            ngenerate=n_generate,
        )

        # Extract variables for metrics from root file
        nominal_survival_fraction = extract_nominal_survival_fraction(f"{outfile}.root")

        # Cleanup
        if cleanup:
            os.remove(model_path)
            os.remove(f"{self.builder.model_dir}/S1_COL1_{run_id}_beam.gmad")
            os.remove(f"{self.builder.model_dir}/S1_COL1_{run_id}_components.gmad")
            os.remove(f"{self.builder.model_dir}/S1_COL1_{run_id}_options.gmad")
            os.remove(f"{self.builder.model_dir}/S1_COL1_{run_id}_sequence.gmad")
            os.remove(outfile + ".root")

        return {"nominal_survival_fraction": nominal_survival_fraction}
