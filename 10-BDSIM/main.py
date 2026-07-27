from opt_problem import *
from opt_model import *
from bdsim_opt import BDSIMOpt

Beam = pybdsim.Beam.Beam()
Beam.SetParticleType("proton")
Beam.SetEnergy(953.27231, "MeV")
Beam.SetDistributionType("userfile")
#Beam._SetDistrFile("../11-Beam/LLO_n250k_pm100.txt")
#Beam._SetDistrFileFormat("x[m]:y[m]:z[m]:xp[rad]:yp[rad]:E[MeV]:t[s]")
#Beam._SetDistrFile("../11-Beam/LhARA_0cm_pm2-100k.dat")
#Beam._SetDistrFileFormat("x[m]:y[m]:t[ns]:xp[rad]:yp[rad]:E[MeV]")
#Beam._SetDistrFile("../11-Beam/LLO_n123k.txt")
#Beam._SetDistrFileFormat("x[m]:y[m]:z[m]:xp[rad]:yp[rad]:E[MeV]:t[s]")
Beam._SetDistrFile("../11-Beam/RFT_n124k.dat")
Beam._SetDistrFileFormat("x[m]:y[m]:S[m]:xp[rad]:yp[rad]:E[MeV]")

Options = pybdsim.Options.Options()
Options.SetBeamPipeRadius(3.65, "cm")
Options.SetBeamPipeThickness(5, "mm")
Options.SetIntegratorSet(integratorSet='"geant4"')
Options.SetSamplerDiameter(0.1, "m")
Options.SetIncludeFringeFields(on=True)
#Options.SetPhysicsList("g4QGSP_BIC_EMZ")
#Options.SetStopSecondaries(stop=True)
Options.SetGeneralOption("collimatorsAreInfiniteAbsorbers", 1)
Options.SetNGenerate(40000)


if __name__ == "__main__":
    Builder = Builder(Beam, Options, "10-Model", "90-BDSIMData")
    # config = OptConfig(
    #     objectives=["fraction_nominal_total"],
    #     constraints={},
    #     bounds={"col_aper": (0.0025, 0.0075)},
    #     n_initial=1,
    #     n_iter=1,
    #     batch_size=1,
    #     mode = "scalar",
    # )
    # model = S1ColModel(Builder)
    # problem = S1ColProblem(model, config, targets={"fraction_nominal_total": 0.03})
    # bo = BDSIMOpt(problem, "S1Col_Scaler_test")
    # bo.optimise()
    # bo.plot_results()

    # config = OptConfig(
    #     objectives=["T", "A", "D"],
    #     constraints = {"T_min": 0.04, "D_max": 0.04},
    #     bounds={
    #         "d1": (0.0, 0.25),
    #         "d2": (0.0, 0.25),
    #         "d3": (0.0, 0.25),
    #         "q1_L": (0.02, 0.05),
    #         "q2_L": (0.02, 0.05),
    #         "q3_L": (0.02, 0.05),
    #         "aper1": (0.01, 0.03),
    #         "aper2": (0.01, 0.03),
    #         "aper3": (0.01, 0.03),
    #     },
    #     n_initial=4,
    #     n_iter=5,
    #     batch_size=4,
    #     mode= "mobo",
    # )
    # model = TripletModel(Builder)
    # problem = TripletCapture(model, config)
    # mobo = BDSIMOpt(problem, "MOBO_Triplet_Demo")
    # mobo.optimise()
    # mobo.plot_results()

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


    bounds_3cm = {
        "b4": (0.1, 1.4),
        "b5": (0.1, 1.3),
        "b6": (0.1, 1.3),
        "b7": (0.1, 1.3),
    }

    # bounds_2_5cm = {
    #     "b4": (1.2, 1.3),
    #     "b5": (0.6, 0.65),
    #     "b6": (0.8, 0.85),
    #     "b7": (0.225, 0.275),
    # }
    #
    # bounds_2cm = {
    #     "b4": (1.0, 1.3),
    #     "b5": (0.5, 0.9),
    #     "b6": (0.7, 1.1),
    #     "b7": (0.1, 0.5),
    # }
    #

    # +/-25% zoom around the best-balanced point found on the wide-box Pareto
    # front (X=[0.967, 1.193, 0.400, 0.828]), clipped to the physical b range,
    # to test whether the wide box's qEHVI search under-sampled the interior
    # "balanced" region in favour of frontier-extending corner solutions.
    bounds_zoom = {
        "b4": (0.725, 1.209),
        "b5": (0.895, 1.3),
        "b6": (0.300, 0.500),
        "b7": (0.621, 1.035),
    }

    # Round 2: the zoom-1 Pareto front pinned b5 at its upper edge (1.3) and
    # b6 at its lower edge (0.3) in most points, including the best combined
    # trade-off - both were artificial window edges, not physical limits, so
    # extend in those directions (b5 toward the true 1.4 ceiling, b6 back
    # toward the original 0.1 floor) while keeping the search concentrated.
    bounds_zoom2 = {
        "b4": (0.7, 1.3),
        "b5": (1.0, 1.4),
        "b6": (0.1, 0.4),
        "b7": (0.6, 1.0),
    }

    # New wide-box search against the corrected (nozzle-cut) beam found its
    # best-combined point at X=[0.602, 0.916, 0.360, 0.902], fully interior
    # (nothing pinned at a bound) - zoom +/-25% around it to concentrate the
    # same search budget on this neighbourhood instead of the whole box.
    bounds_zoom3 = {
        "b4": (0.4515, 0.7525),
        "b5": (0.687, 1.145),
        "b6": (0.270, 0.450),
        "b7": (0.6765, 1.1275),
    }

    # Zoom around the wide-box front's *other* extreme instead: X=[1.167,
    # 0.790, 0.877, 0.1] (near-perfect alpha, badly-off sigma) - a genuinely
    # different part of the box than the zoom3 neighbourhood, which plateaued.
    # b7 sits exactly at the physical floor (0.1) here, so rather than a
    # near-degenerate +/-25% window around it, anchor it at the floor and
    # give it the same absolute room to explore upward as the other params.
    bounds_zoom_alt = {
        "b4": (0.876, 1.4),
        "b5": (0.592, 0.987),
        "b6": (0.657, 1.096),
        "b7": (0.1, 0.4),
    }

    # zoom_alt found a new best-combined point, X=[1.009, 0.932, 0.921,
    # 0.235], fully interior (nothing pinned) - zoom +/-25% around it.
    bounds_zoom_alt2 = {
        "b4": (0.757, 1.262),
        "b5": (0.699, 1.164),
        "b6": (0.690, 1.151),
        "b7": (0.177, 0.294),
    }
    targets = {
        "sigma_x": 7.5e-3,
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
        "sigma_x": 1e-5,
        #"sigma_y": 1.5e-4,
        "alpha_x": 0.15,
        #"alpha_y": 3.0,
    }

    # config = OptConfig(
    #     objectives=["sigma_x", "alpha_x"],
    #     constraints={},
    #     bounds=bounds_3cm,
    #     n_initial=40,
    #     n_iter=40,
    #     batch_size=5,
    #     mode = "scalar",
    # )
    # model = S1GLModel(Builder)
    # problem = OpticsMatchProblem(model, config, targets, weights=weights, scales=scales)
    # mobo = BDSIMOpt(problem, "SOBO_S1GL_40_40_5_3cm_CoupledLoss_Refine")
    # mobo.optimise()
    # mobo.plot_results()
    # mobo.refine_top_candidates(top_k=5, ngenerate=100_000)

    # config = OptConfig(
    #     objectives=["sigma_x_err", "alpha_x_err"],
    #     constraints={},
    #     bounds=bounds_3cm,
    #     n_initial=40,
    #     n_iter=40,
    #     batch_size=5,
    #     mode = "mobo",
    # )
    # model = S1GLModel(Builder)
    # problem = S1GLMatchMOBO(model, config, targets, scales=scales)
    # mobo = BDSIMOpt(problem, "MOBO_S1GL_40_40_5_3cm_SigmaAlphaPareto")
    # mobo.optimise()
    # mobo.plot_results()
    # mobo.refine_pareto_front(ngenerate=100000)

    # config = OptConfig(
    #     objectives=["sigma_x_err", "alpha_x_err"],
    #     constraints={},
    #     bounds=bounds_zoom,
    #     n_initial=40,
    #     n_iter=40,
    #     batch_size=5,
    #     mode = "mobo",
    # )
    # model = S1GLModel(Builder)
    # problem = S1GLMatchMOBO(model, config, targets, scales=scales)
    # mobo = BDSIMOpt(problem, "MOBO_S1GL_40_40_5_Zoom_SigmaAlphaPareto")
    # mobo.optimise()
    # mobo.plot_results()
    # mobo.refine_pareto_front(ngenerate=100000)

    # config = OptConfig(
    #     objectives=["sigma_x_err", "alpha_x_err"],
    #     constraints={},
    #     bounds=bounds_zoom2,
    #     n_initial=40,
    #     n_iter=40,
    #     batch_size=5,
    #     mode = "mobo",
    # )
    # model = S1GLModel(Builder)
    # problem = S1GLMatchMOBO(model, config, targets, scales=scales)
    # mobo = BDSIMOpt(problem, "MOBO_S1GL_40_40_5_Zoom2_SigmaAlphaPareto")
    # mobo.optimise()
    # mobo.plot_results()
    # mobo.refine_pareto_front(ngenerate=100000)

    # Restart from the wide box: RFT_n124k.dat now contains the fully nozzle-cut
    # beam (both the 2mm entrance and 2.87mm exit radial cuts already applied by
    # RF-Track), replacing the previously-uncut file every prior search here was
    # optimised against. All earlier zoom windows were centred on solutions
    # tuned for the wrong beam, so this starts the search over from scratch.
    # config = OptConfig(
    #     objectives=["sigma_x_err", "alpha_x_err"],
    #     constraints={},
    #     bounds=bounds_3cm,
    #     n_initial=40,
    #     n_iter=40,
    #     batch_size=5,
    #     mode = "mobo",
    # )
    # model = S1GLModel(Builder)
    # problem = S1GLMatchMOBO(model, config, targets, scales=scales)
    # mobo = BDSIMOpt(problem, "MOBO_S1GL_40_40_5_3cm_CutBeam_SigmaAlphaPareto")
    # mobo.optimise()
    # mobo.plot_results()
    # mobo.refine_pareto_front(ngenerate=124264)

    # Trust-region search: replaces the manual wide -> zoom -> zoom2 -> zoom3
    # process with a single run over the full wide box (bounds_3cm) that
    # shrinks/expands its own search window automatically based on whether
    # recent batches are actually improving the Pareto front. Spent 29 of 40
    # iterations pinned at max window size (tr_fail_tol=1 auto-computed was
    # too trigger-happy); best result (combined error ~16.8) was worse than
    # the plain wide-box search it was meant to improve on. Parked here
    # pending retuning - back to manual zooming for now.
    # config = OptConfig(
    #     objectives=["sigma_x_err", "alpha_x_err"],
    #     constraints={},
    #     bounds=bounds_3cm,
    #     n_initial=40,
    #     n_iter=40,
    #     batch_size=5,
    #     mode = "mobo",
    #     use_trust_region=True,
    # )
    # model = S1GLModel(Builder)
    # problem = S1GLMatchMOBO(model, config, targets, scales=scales)
    # mobo = BDSIMOpt(problem, "MOBO_S1GL_40_40_5_3cm_CutBeam_TrustRegion_SigmaAlphaPareto")
    # mobo.optimise()
    # mobo.plot_results()
    # mobo.refine_pareto_front(ngenerate=124264)

    # config = OptConfig(
    #     objectives=["sigma_x_err", "alpha_x_err"],
    #     constraints={},
    #     bounds=bounds_zoom3,
    #     n_initial=40,
    #     n_iter=40,
    #     batch_size=5,
    #     mode = "mobo",
    # )
    # model = S1GLModel(Builder)
    # problem = S1GLMatchMOBO(model, config, targets, scales=scales)
    # mobo = BDSIMOpt(problem, "MOBO_S1GL_40_40_5_Zoom3_CutBeam_SigmaAlphaPareto")
    # mobo.optimise()
    # mobo.plot_results()
    # mobo.refine_pareto_front(ngenerate=124264)

    # config = OptConfig(
    #     objectives=["sigma_x_err", "alpha_x_err"],
    #     constraints={},
    #     bounds=bounds_zoom_alt,
    #     n_initial=40,
    #     n_iter=40,
    #     batch_size=5,
    #     mode = "mobo",
    # )
    # model = S1GLModel(Builder)
    # problem = S1GLMatchMOBO(model, config, targets, scales=scales)
    # mobo = BDSIMOpt(problem, "MOBO_S1GL_40_40_5_ZoomAlt_CutBeam_SigmaAlphaPareto")
    # mobo.optimise()
    # mobo.plot_results()
    # mobo.refine_pareto_front(ngenerate=124264)

    # config = OptConfig(
    #     objectives=["sigma_x_err", "alpha_x_err"],
    #     constraints={},
    #     bounds=bounds_zoom_alt2,
    #     n_initial=40,
    #     n_iter=40,
    #     batch_size=5,
    #     mode = "mobo",
    # )
    # model = S1GLModel(Builder)
    # problem = S1GLMatchMOBO(model, config, targets, scales=scales)
    # mobo = BDSIMOpt(problem, "MOBO_S1GL_40_40_5_ZoomAlt2_CutBeam_SigmaAlphaPareto")
    # mobo.optimise()
    # mobo.plot_results()
    # mobo.refine_pareto_front(ngenerate=124264)

    # Mac Studio run (12 cores): batch_size raised to 10 to use the extra
    # cores, and ngenerate raised from 10k to 40k (top of file) to bring the
    # search's own shot noise down from ~53um to ~26um, closer to the 10um
    # target tolerance, without paying full-beam (124k) cost on every single
    # evaluation. Back to the wide box (bounds_3cm) rather than continuing to
    # zoom zoom_alt2, to get a cleaner look at the whole space now that the
    # search itself is less noisy - if it lands back in the same basin as
    # zoom_alt2 that's a good confirmation; if not, worth knowing.
    config = OptConfig(
        objectives=["sigma_x_err", "alpha_x_err"],
        constraints={},
        bounds=bounds_3cm,
        n_initial=40,
        n_iter=40,
        batch_size=10,
        mode = "mobo",
    )
    model = S1GLModel(Builder)
    problem = S1GLMatchMOBO(model, config, targets, scales=scales)
    mobo = BDSIMOpt(problem, "MOBO_S1GL_40_40_10_3cm_CutBeam_N40k_SigmaAlphaPareto")
    mobo.optimise()
    mobo.plot_results()
    mobo.refine_pareto_front(ngenerate=124264)

