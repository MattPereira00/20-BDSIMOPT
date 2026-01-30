from mobo_bdsim import *
from bdsim_mobo import *

import time

Beam = pybdsim.Beam.Beam()
Beam.SetParticleType("proton")
Beam.SetEnergy(953.27231, "MeV")
Beam.SetDistributionType("userfile")
Beam._SetDistrFile("../11-Beam/LhARA_0cm_pm2-100k.dat")
Beam._SetDistrFileFormat("x[m]:y[m]:t[ns]:xp[rad]:yp[rad]:E[MeV]")

Options = pybdsim.Options.Options()
Options.SetBeamPipeRadius(10, "mm")
Options.SetBeamPipeThickness(5, "mm")
Options.SetIntegratorSet(integratorSet='"geant4"')
Options.SetSamplerDiameter(0.1, "m")
Options.SetIncludeFringeFields(on=True)
Options.SetGeneralOption("collimatorsAreInfiniteAbsorbers", 1)
Options.SetNGenerate(100000)


if __name__ == "__main__":
    start = time.time()
    Builder = Builder(Beam, Options, "10-Model", "90-BDSIMData")
    config = OptConfig(
        objectives=["T", "D"],
        constraints = {"T_min": 0.04, "D_max": 0.04},
        bounds={
            "d1": (0.0, 0.25),
            "d2": (0.0, 0.25),
            "d3": (0.0, 0.25),
            "q1_L": (0.02, 0.05),
            "q2_L": (0.02, 0.05),
            "q3_L": (0.02, 0.05),
            "aper1": (0.01, 0.04),
            "aper2": (0.01, 0.04),
            "aper3": (0.01, 0.04),
        },
        n_initial=10,
        n_iter=10,
        batch_size=4,
    )
    model = TripletModel(Builder)
    problem = TripletCapture(model, config)
    mobo = BDSIMMoBO(problem, "MOBO_Refactor_Test_Constraints")
    mobo.optimise()
    mobo.plot_pareto()

    end = time.time()
    print("Time Taken: ", (end - start)/60, "mins")