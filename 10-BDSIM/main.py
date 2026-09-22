from opt_problem import *
from opt_model import *
from bdsim_opt import BDSIMOpt

Beam = pybdsim.Beam.Beam()
Beam.SetParticleType("proton")
Beam.SetEnergy(953.27231, "MeV")
Beam.SetDistributionType("userfile")
Beam._SetDistrFile("../11-Beam/RFT_pm100_BetterCut.dat")
Beam._SetDistrFileFormat("x[m]:y[m]:S[m]:xp[rad]:yp[rad]:E[MeV]")

Options = pybdsim.Options.Options()
Options.SetBeamPipeRadius(3.65, "cm")
Options.SetBeamPipeThickness(5, "mm")
Options.SetIntegratorSet(integratorSet='"geant4"')
Options.SetSamplerDiameter(0.073, "m")
Options.SetIncludeFringeFields(on=True)
Options.SetPhysicsList("g4QGSP_BIC_EMZ")
Options.SetStopSecondaries(stop=True)
Options.SetNGenerate(251261)

if __name__ == "__main__":
    Builder = Builder(Beam, Options, "10-Model", "90-BDSIMData")

    # Stage 1 Energy Selection: col_aper/b3/rf_voltage, maximizing purity
    # and yield. Bounds and ngenerate are unvalidated first guesses.
    bounds_eneselect = {
        "rf_voltage": (5e5, 15e5),
        "GL3B": (0.1, 1.4),
        "col_aper": (0.5e-3, 3e-3),
    }
    config = OptConfig(
        objectives=["purity", "yield"],
        constraints={},
        bounds=bounds_eneselect,
        n_initial=4,
        n_iter=4,
        batch_size=4,
        mode="mobo",
    )
    model = Stage1EnergySelection(Builder)
    problem = Stage1EnergySelectionMOBO(model, config)
    mobo = BDSIMOpt(problem, "MOBO_Stage1EnergySelection_4_4_4_Test")
    mobo.optimise()
    mobo.plot_results()
