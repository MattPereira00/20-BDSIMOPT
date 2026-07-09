"""
A/B check: does enabling Geant4 physics (vs BDSIM's default pure-transport
tracking) change the measured sigma_x/alpha_x for our chosen S1GL solution?

Same isolated S1GL-only line, same beam file, same b4-b7 values - only
physicsList/stopSecondaries toggled - to isolate whether physics-driven
absorption of halo particles explains the ~4x sigma_x growth seen when the
same strengths were run in the full CCAP-v4.6 model.

Run standalone: python verify_physics_ab.py
"""
import os
import subprocess

import pybdsim

from builder import Builder
from opt_obj import extract_optics

BEST_B = [0.7, 1.4, 0.1, 0.93116529]  # b4, b5, b6, b7
NGENERATE = 100_000


def make_beam():
    beam = pybdsim.Beam.Beam()
    beam.SetParticleType("proton")
    beam.SetEnergy(953.27231, "MeV")
    beam.SetDistributionType("userfile")
    beam._SetDistrFile("../11-Beam/RFT_n124k.dat")
    beam._SetDistrFileFormat("x[m]:y[m]:S[m]:xp[rad]:yp[rad]:E[MeV]")
    return beam


def make_options(with_physics: bool):
    options = pybdsim.Options.Options()
    options.SetBeamPipeRadius(3.65, "cm")
    options.SetBeamPipeThickness(5, "mm")
    options.SetIntegratorSet(integratorSet='"geant4"')
    options.SetSamplerDiameter(0.1, "m")
    options.SetIncludeFringeFields(on=True)
    if with_physics:
        options.SetPhysicsList("g4QGSP_BIC_EMZ")
        options.SetStopSecondaries(stop=True)
        options.SetGeneralOption("collimatorsAreInfiniteAbsorbers", 1)
    options.SetNGenerate(NGENERATE)
    return options


def run_case(label, with_physics):
    beam = make_beam()
    options = make_options(with_physics)
    builder = Builder(beam, options, "10-Model", "90-BDSIMData")

    fname = f"PhysicsAB_{label}"
    builder.build_s1_gl(fname, BEST_B)

    model_path = f"{builder.model_dir}/S1GL_{fname}.gmad"
    outfile = f"{builder.data_dir}/{fname}"

    pybdsim.Run.Bdsim(
        gmadpath=model_path,
        outfile=outfile,
        batch=True,
        silent=True,
        ngenerate=NGENERATE,
    )

    subprocess.call([
        "rebdsimOptics", f"{outfile}.root", f"{outfile}_optics.root", "--emittanceOnTheFly"
    ], stdout=open(os.devnull, "wb"))

    sigma_x, sigma_y, alpha_x, alpha_y = extract_optics(outfile)

    os.remove(model_path)
    os.remove(f"{builder.model_dir}/S1GL_{fname}_beam.gmad")
    os.remove(f"{builder.model_dir}/S1GL_{fname}_components.gmad")
    os.remove(f"{builder.model_dir}/S1GL_{fname}_options.gmad")
    os.remove(f"{builder.model_dir}/S1GL_{fname}_sequence.gmad")
    os.remove(f"{outfile}.root")
    os.remove(f"{outfile}_optics.root")

    return sigma_x, sigma_y, alpha_x, alpha_y


if __name__ == "__main__":
    print(f"Testing X = {BEST_B} at ngenerate={NGENERATE}\n")

    sx_phys, sy_phys, ax_phys, ay_phys = run_case("WithPhysics", with_physics=True)
    print(
        f"[Physics ON ]  sigma_x={sx_phys * 1e3:.4f} mm  sigma_y={sy_phys * 1e3:.4f} mm  "
        f"alpha_x={ax_phys:.4f}  alpha_y={ay_phys:.4f}"
    )

    sx_none, sy_none, ax_none, ay_none = run_case("NoPhysics", with_physics=False)
    print(
        f"[Physics OFF]  sigma_x={sx_none * 1e3:.4f} mm  sigma_y={sy_none * 1e3:.4f} mm  "
        f"alpha_x={ax_none:.4f}  alpha_y={ay_none:.4f}"
    )

    ratio = sx_none / sx_phys if sx_phys else float("nan")
    print(f"\nsigma_x ratio (no-physics / with-physics) = {ratio:.2f}x")
