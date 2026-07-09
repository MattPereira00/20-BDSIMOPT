"""
Checks whether the fixed GL1-3 lens strengths satisfy their own design
conditions for the current input beam:
  - the beam should be parallel (alpha_x ~ 0) all through the GL2->GL3 drift
  - there should be a real sigma_x waist (strong focus) shortly after GL3

Run standalone: python diagnose_stage1.py
"""
import os
import subprocess

import pybdsim

from builder import Builder
from opt_obj import extract_stage1_diagnostics

Beam = pybdsim.Beam.Beam()
Beam.SetParticleType("proton")
Beam.SetEnergy(953.27231, "MeV")
Beam.SetDistributionType("userfile")
Beam._SetDistrFile("../11-Beam/RFT_n124k.dat")
Beam._SetDistrFileFormat("x[m]:y[m]:S[m]:xp[rad]:yp[rad]:E[MeV]")

Options = pybdsim.Options.Options()
Options.SetBeamPipeRadius(3.65, "cm")
Options.SetBeamPipeThickness(5, "mm")
Options.SetIntegratorSet(integratorSet='"geant4"')
Options.SetSamplerDiameter(0.1, "m")
Options.SetIncludeFringeFields(on=True)
Options.SetPhysicsList("g4QGSP_BIC_EMZ")
Options.SetStopSecondaries(stop=True)
Options.SetGeneralOption("collimatorsAreInfiniteAbsorbers", 1)
Options.SetNGenerate(100_000)

# Parallel-tolerance / waist-quality thresholds for the pass/fail report
ALPHA_PARALLEL_TOL = 0.15


if __name__ == "__main__":
    builder = Builder(Beam, Options, "10-Model", "90-BDSIMData")

    regions = builder.build_s1_gl_diagnostic(
        "Stage1Check",
        b1=1.400, b2=0.579, b3=0.817,
    )

    model_path = f"{builder.model_dir}/S1GL_diag_Stage1Check.gmad"
    outfile = f"{builder.data_dir}/stage1_diag_Stage1Check"

    pybdsim.Run.Bdsim(
        gmadpath=model_path,
        outfile=outfile,
        batch=True,
        silent=True,
        ngenerate=builder.Options["ngenerate"],
    )

    subprocess.call([
        "rebdsimOptics", f"{outfile}.root", f"{outfile}_optics.root", "--emittanceOnTheFly"
    ], stdout=open(os.devnull, "wb"))

    diag = extract_stage1_diagnostics(outfile, regions)

    print("=== GL2 -> GL3 parallel-beam check ===")
    for s, a in zip(diag["gl2_to_gl3_S"], diag["gl2_to_gl3_alpha_x"]):
        print(f"  S={s:7.4f} m   alpha_x={a:+.4f}")
    max_alpha = diag["gl2_to_gl3_max_abs_alpha"]
    verdict = "PASS" if max_alpha is not None and max_alpha <= ALPHA_PARALLEL_TOL else "FAIL"
    print(f"  max|alpha_x| = {max_alpha:.4f}  (tolerance {ALPHA_PARALLEL_TOL})  -> {verdict}")

    print()
    print("=== Post-GL3 focus check ===")
    for s, sig in zip(diag["gl3_to_gl4_S"], diag["gl3_to_gl4_sigma_x"]):
        print(f"  S={s:7.4f} m   sigma_x={sig * 1e3:.4f} mm")
    print(
        f"  waist: S={diag['waist_S']:.4f} m "
        f"({diag['waist_distance_from_gl3']:.4f} m after GL3), "
        f"sigma_x={diag['waist_sigma_x'] * 1e3:.4f} mm"
    )
