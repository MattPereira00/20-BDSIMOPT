import os, subprocess
from opt_obj import extract_metrics_and_uncertainties
import pybdsim

modelDir = "10-Model"
dataDir = "90-BDSIMData"
runDir = f"{dataDir}/Baseline_Nozzle"
if not os.path.exists(runDir):
    os.mkdir(runDir)


modelPath = f"{modelDir}/baseline.gmad"
outfile = f"{runDir}/baseline_100k"
n_gen = 100000

pybdsim.Run.Bdsim(
    gmadpath=modelPath,
    outfile=outfile,
    batch=True,
    seed=1999,
    silent=False,
    ngenerate=n_gen
)

subprocess.call(["rebdsimOptics", f"{outfile}.root", f"{outfile}_optics.root", "--emittanceOnTheFly"],
                stdout=open(os.devnull, 'wb'))

T, Sigma_T, A, Sigma_A, D, Sigma_D = extract_metrics_and_uncertainties(outfile)

print(f"transmission = {T:.3f} ± {Sigma_T:.3f}, "
      f"asymmetry = {A:.3f} ± {Sigma_A:.3f}, "
      f"divergence= {D:.3f} ± {Sigma_D:.3f},")