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
        objectives=["T", "A", "D"],
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
        n_initial=15,
        n_iter=15,
        batch_size=4,
    )
    model = TripletModel(Builder)
    problem = TripletCapture(model, config)
    mobo = BDSIMMoBO(problem, "MOBO_Triplet1_15_15_4")
    mobo.optimise()
    mobo.plot_pareto()

    # frozen_triplet1 = {
    #     "d1": 0.0241,
    #     "d2": 0.0223,
    #     "d3": 0.161,
    #     "q1_L": 0.0305,
    #     "q2_L": 0.0366,
    #     "q3_L": 0.022,
    #     "aper1": 0.0254,
    #     "aper2": 0.0283,
    #     "aper3": 0.03,
    # }
    #
    # config = OptConfig(
    #     objectives=["T", "D", "A"],
    #     constraints={"T_min": 0.04, "D_max": 0.015, "A_max": 0.1},
    #     bounds={
    #         "d4": (0.0, 0.5),
    #         "d5": (0.0, 0.5),
    #         "d6": (0.0, 0.5),
    #         "q4_L": (0.02, 0.05),
    #         "q5_L": (0.02, 0.05),
    #         "q6_L": (0.02, 0.05),
    #         "aper4": (0.01, 0.04),
    #         "aper5": (0.01, 0.04),
    #         "aper6": (0.01, 0.04),
    #     },
    #     n_initial=15,
    #     n_iter=15,
    #     batch_size=4,
    # )
    # model = DoubleTripletModel(Builder, frozen_triplet1)
    # problem = DoubleTripletProblem(model, config)
    # mobo = BDSIMMoBO(problem, "MOBO_Triplet2_15_15_4")
    # mobo.optimise()
    # mobo.plot_pareto()

    end = time.time()
    print("Time Taken: ", (end - start)/60, "mins")