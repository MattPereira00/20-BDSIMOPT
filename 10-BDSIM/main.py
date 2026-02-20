from opt_problem import *
from opt_model import *
from bdsim_opt import BDSIMOpt

import time

Beam = pybdsim.Beam.Beam()
Beam.SetParticleType("proton")
Beam.SetEnergy(953.27231, "MeV")
Beam.SetDistributionType("userfile")
#Beam._SetDistrFile("../11-Beam/LhARA_0cm_pm2-100k.dat")
#Beam._SetDistrFileFormat("x[m]:y[m]:t[ns]:xp[rad]:yp[rad]:E[MeV]")
Beam._SetDistrFile("../11-Beam/LLO_n123k.txt")
Beam._SetDistrFileFormat("x[m]:y[m]:z[m]:xp[rad]:yp[rad]:E[MeV]:t[s]")

Options = pybdsim.Options.Options()
Options.SetBeamPipeRadius(10, "cm")
Options.SetBeamPipeThickness(5, "mm")
Options.SetIntegratorSet(integratorSet='"geant4"')
Options.SetSamplerDiameter(0.1, "m")
Options.SetIncludeFringeFields(on=True)
Options.SetGeneralOption("collimatorsAreInfiniteAbsorbers", 1)
Options.SetNGenerate(10000)


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
        n_iter=20,
        batch_size=4,
    )
    model = TripletModel(Builder)
    problem = TripletCapture(model, config)
    mobo = BDSIMOpt(problem, "MOBO_Triplet1_15_20_4")
    mobo.optimise()
    mobo.plot_pareto()

    # config = OptConfig(
    #     objectives=["T", "A", "D"],
    #     constraints={"T_min": 0.04, "A_max": 1.0, "D_max": 0.015},
    #     bounds={
    #         "d4": (0.0, 0.25),
    #         "d5": (0.0, 0.25),
    #         "d6": (0.0, 0.25),
    #         "q4_L": (0.02, 0.05),
    #         "q5_L": (0.02, 0.05),
    #         "q6_L": (0.02, 0.05),
    #         "aper4": (0.01, 0.04),
    #         "aper5": (0.01, 0.04),
    #         "aper6": (0.01, 0.04),
    #     },
    #     n_initial=20,
    #     n_iter=20,
    #     batch_size=4,
    # )
    #
    # triplet1_balanced = {
    #     "d1": 0.0589,
    #     "d2": 0.00913,
    #     "d3": 0.0338,
    #     "q1_L": 0.0205,
    #     "q2_L": 0.0315,
    #     "q3_L": 0.0416,
    #     "aper1": 0.0135,
    #     "aper2": 0.0292,
    #     "aper3": 0.0370,
    # }
    # model = DoubleTripletModel(Builder, triplet1_balanced)
    # problem = DoubleTripletProblem(model, config)
    # mobo = BDSIMMoBO(problem, "Triplet2_20_20_4_T1Balanced")
    # mobo.optimise()
    # mobo.plot_pareto()
    #
    # triplet1_lowD = {
    #     "d1": 0.0399,
    #     "d2": 0.0651,
    #     "d3": 0.1470,
    #     "q1_L": 0.0255,
    #     "q2_L": 0.0304,
    #     "q3_L": 0.0474,
    #     "aper1": 0.0324,
    #     "aper2": 0.0108,
    #     "aper3": 0.0348,
    # }
    #
    # model = DoubleTripletModel(Builder, triplet1_lowD)
    # problem = DoubleTripletProblem(model, config)
    # mobo = BDSIMMoBO(problem, "Triplet2_20_20_4_T1LowD")
    # mobo.optimise()
    # mobo.plot_pareto()
    #
    # triplet1_lowA = {
    #     "d1": 0.000177,
    #     "d2": 0.1270,
    #     "d3": 0.00411,
    #     "q1_L": 0.0275,
    #     "q2_L": 0.0348,
    #     "q3_L": 0.0389,
    #     "aper1": 0.0190,
    #     "aper2": 0.0127,
    #     "aper3": 0.0118,
    # }
    #
    # model = DoubleTripletModel(Builder, triplet1_lowA)
    # problem = DoubleTripletProblem(model, config)
    # mobo = BDSIMMoBO(problem, "Triplet2_20_20_4_T1LowA")
    # mobo.optimise()
    # mobo.plot_pareto()

    # Builder.build_s1GL("S1GL1Test", [1.309, 0.535, 0.785, 0.065])
    #
    # pybdsim.Run.Bdsim(f"{Builder.model_dir}/S1GL_S1GL1Test.gmad",
    #                   f"{Builder.data_dir}/S1GL_S1GL1Test", ngenerate=10000)
    #
    # pybdsim.Run.RebdsimOptics(f"{Builder.data_dir}/S1GL_S1GL1Test.root",
    #                           f"{Builder.data_dir}/S1GL_S1GL1Test_optics.root")
    #
    # pybdsim.Plot.BDSIMOptics(f"{Builder.data_dir}/S1GL_S1GL1Test_optics.root",
    #                          outputfilename=f"{Builder.data_dir}/S1GL_S1GL1Test_optics.pdf",
    #                          survey=f"{Builder.data_dir}/S1GL_S1GL1Test.root")


    # bounds_3cm = {
    #     "b4": (1.2, 1.35),
    #     "b5": (0.55, 0.7),
    #     "b6": (0.6, 0.8),
    #     "b7": (0.25, 0.45),
    # }

    bounds_2_5cm = {
        "b4": (1.2, 1.3),
        "b5": (0.6, 0.65),
        "b6": (0.8, 0.85),
        "b7": (0.225, 0.275),
    }

    bounds_2cm = {
        "b4": (1.0, 1.3),
        "b5": (0.5, 0.9),
        "b6": (0.7, 1.1),
        "b7": (0.1, 0.5),
    }

    targets = {
        "sigma_x": 5e-3,
        # "sigma_y": 7.5e-3,
        "alpha_x": 0.0,
        # "alpha_y": 0.0,
    }

    weights = {
        "sigma_x": 2.0,
        #"sigma_y": 1.0,
        "alpha_x": 1.0,
        #"alpha_y": 1.0,
    }

    scales = {
        "sigma_x": 0.5e-4,
        #"sigma_y": 1.5e-4,
        "alpha_x": 1e-1,
        #"alpha_y": 3.0,
    }

    config = OptConfig(
        objectives=["sigma_x", "alpha_x"],
        constraints={},
        bounds=bounds_2cm,
        n_initial=24,
        n_iter=40,
        batch_size=4,
        mode = "scalar",
    )
    model = S1GLModel(Builder)
    problem = OpticsMatchProblem(model, config, targets, weights=weights, scales=scales)
    mobo = BDSIMOpt(problem, "SOBO_S1GL_24_40_4_2cm_scales")
    mobo.optimise()
    mobo.plot_results()

    end = time.time()
    print("Time Taken: ", (end - start)/60, "mins")