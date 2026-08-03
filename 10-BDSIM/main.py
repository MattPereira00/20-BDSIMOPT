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
Options.SetNGenerate(124262)


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
        "sigma_xp": 0.0,
        # "sigma_yp": 0.0,
    }

    weights = {
        "sigma_x": 2.0,
        #"sigma_y": 1.0,
        "sigma_xp": 1.0,
        #"sigma_yp": 1.0,
    }

    scales = {
        "sigma_x": 1e-5,
        #"sigma_y": 1.5e-4,
        "sigma_xp": 1e-4,
        #"sigma_yp": 3.0,
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
    # config = OptConfig(
    #     objectives=["sigma_x_err", "alpha_x_err"],
    #     constraints={},
    #     bounds=bounds_3cm,
    #     n_initial=40,
    #     n_iter=40,
    #     batch_size=10,
    #     mode = "mobo",
    # )
    # model = S1GLModel(Builder)
    # problem = S1GLMatchMOBO(model, config, targets, scales=scales)
    # mobo = BDSIMOpt(problem, "MOBO_S1GL_40_40_10_3cm_CutBeam_N40k_SigmaAlphaPareto")
    # mobo.optimise()
    # mobo.plot_results()
    # mobo.refine_pareto_front(ngenerate=124264)  # BUG: should be 124227 (see recover_refine.py)

    # Zoom (with the same batch_size=10/ngenerate=40000 settings) around the
    # best point found in the N40k wide box, X=[0.709, 0.891, 0.440, 0.868]
    # (combined error 6.03 - sigma much better than before at this stat level,
    # alpha worse; same trade-off curve, different point on it).
    # bounds_zoom_n40k = {
    #     "b4": (0.532, 0.887),
    #     "b5": (0.668, 1.114),
    #     "b6": (0.330, 0.550),
    #     "b7": (0.651, 1.085),
    # }
    # config = OptConfig(
    #     objectives=["sigma_x_err", "alpha_x_err"],
    #     constraints={},
    #     bounds=bounds_zoom_n40k,
    #     n_initial=40,
    #     n_iter=40,
    #     batch_size=10,
    #     mode = "mobo",
    # )
    # model = S1GLModel(Builder)
    # problem = S1GLMatchMOBO(model, config, targets, scales=scales)
    # mobo = BDSIMOpt(problem, "MOBO_S1GL_40_40_10_ZoomN40k_CutBeam_SigmaAlphaPareto")
    # mobo.optimise()
    # mobo.plot_results()
    # mobo.refine_pareto_front(ngenerate=124262)
    # Result: converged back to the same point the zoom was centred on
    # (X=[0.7095, 0.891, 0.44, 0.868], combined error 6.05 - statistically the
    # same as the 6.03 that point already scored). No improvement from 40 more
    # iterations concentrated on this neighbourhood.

    # Sensitivity scan: every search region tried so far (zoom, zoom2, zoom3,
    # alt, alt2, wide box, this last zoom) has produced the same *shape* of
    # sigma/alpha trade-off - a 1D curve, not a 2D region of good solutions -
    # even though there are 4 free lenses (b4-b7) for only 2 targets. That
    # points at two of the lenses not moving sigma_x/alpha_x independently at
    # this location. To check directly (rather than spending another BO
    # round): hold 3 of the 4 lenses fixed at the current best point and
    # sweep the 4th across its zoom window, one lens at a time. If sigma_x and
    # alpha_x move independently along some axis, that's real search room left
    # to exploit; if they move in lockstep along every axis, it's a genuine
    # optics constraint at this location, not a search/noise artifact.
    # X0 = {"b4": 0.7095, "b5": 0.891, "b6": 0.44, "b7": 0.868}
    # sweep_windows = {
    #     "b4": (0.532, 0.887),
    #     "b5": (0.668, 1.114),
    #     "b6": (0.330, 0.550),
    #     "b7": (0.651, 1.085),
    # }
    #
    # points = {}  # dedup key -> (param, frac, X list)
    # for param, (lo, hi) in sweep_windows.items():
    #     half = (hi - lo) / 2
    #     centre = X0[param]
    #     for frac in (-1.0, -0.5, 0.0, 0.5, 1.0):
    #         x = dict(X0)
    #         x[param] = centre + frac * half
    #         key = tuple(round(x[p], 6) for p in ("b4", "b5", "b6", "b7"))
    #         if key not in points:
    #             points[key] = (param, frac, [x["b4"], x["b5"], x["b6"], x["b7"]])
    #
    # model = S1GLModel(Builder)
    # run_ids = list(points.keys())
    # results = {}
    # with ProcessPoolExecutor(max_workers=min(len(run_ids), 12)) as pool:
    #     futures = {
    #         pool.submit(model.run, np.array(points[k][2]), f"sens_{i}"): k
    #         for i, k in enumerate(run_ids)
    #     }
    #     for fut in as_completed(futures):
    #         k = futures[fut]
    #         results[k] = fut.result()
    #
    # for param in sweep_windows:
    #     print(f"\n--- Sweeping {param} (others fixed at best point) ---")
    #     rows = sorted(
    #         (points[k][1], points[k][2], results[k])
    #         for k in points if points[k][0] == param
    #     )
    #     for frac, x, r in rows:
    #         xr = [round(v, 4) for v in x]
    #         print(f"  frac={frac:+.2f}  X={xr}  "
    #               f"sigma_x={r['sigma_x']*1e3:.4f}mm  alpha_x={r['alpha_x']:.4f}")
    # Result: the four lenses have very different alpha/sigma sensitivity
    # ratios (b4=-2.35, b5=-39.0, b6=-0.14, b7=-0.77 alpha-units per mm) -
    # not locked together. b5 is close to a pure alpha knob (weak sigma
    # effect, huge alpha effect); b7 is close to a pure sigma knob. Since the
    # two gradients aren't parallel, a small *combined* move (not a single
    # axis) should be able to hit both targets - solving the local 2x4 linear
    # system for the step from here to sigma=7.5mm/alpha=0 gives a tiny
    # predicted correction: b4 0.7095->0.7123, b5 0.891->0.9105,
    # b6 0.44->0.4396 (negligible), b7 0.868->0.8634. Small enough that no
    # single-axis sweep or wide BO search would land on it by chance.

    # Local polish grid around the predicted joint-correction point, to
    # verify the linear prediction and absorb any residual nonlinearity.
    # 3 levels each of b4, b5, b7 (b6 fixed - its measured effect on both
    # sigma and alpha was negligible in the sweep above).
    # grid_b4 = [0.7023, 0.7123, 0.7223]
    # grid_b5 = [0.8905, 0.9105, 0.9305]
    # grid_b7 = [0.8534, 0.8634, 0.8734]
    # b6_fixed = 0.4396
    #
    # points = {}
    # i = 0
    # for b4 in grid_b4:
    #     for b5 in grid_b5:
    #         for b7 in grid_b7:
    #             points[i] = [b4, b5, b6_fixed, b7]
    #             i += 1
    #
    # model = S1GLModel(Builder)
    # results = {}
    # with ProcessPoolExecutor(max_workers=12) as pool:
    #     futures = {
    #         pool.submit(model.run, np.array(x), f"polish_{k}"): k
    #         for k, x in points.items()
    #     }
    #     for fut in as_completed(futures):
    #         k = futures[fut]
    #         results[k] = fut.result()
    #
    # target_sigma, target_alpha = 7.5e-3, 0.0
    # tol_sigma, tol_alpha = 1e-5, 0.15
    # rows = []
    # for k, x in points.items():
    #     r = results[k]
    #     sigma_units = abs(r["sigma_x"] - target_sigma) / tol_sigma
    #     alpha_units = abs(r["alpha_x"] - target_alpha) / tol_alpha
    #     combined = (sigma_units**2 + alpha_units**2) ** 0.5
    #     rows.append((combined, x, r["sigma_x"], r["alpha_x"]))
    #
    # rows.sort(key=lambda row: row[0])
    # for combined, x, sigma_x, alpha_x in rows:
    #     print(f"X={[round(v,4) for v in x]}  sigma_x={sigma_x*1e3:.4f}mm  "
    #           f"alpha_x={alpha_x:.4f}  combined={combined:.2f}")
    # Result: best at 40k stats was X=[0.7123,...] (combined=1.78), but a
    # 4-point high-stat (ngenerate=124262) verification of the top grid
    # candidates flipped the ranking - X=[0.7223, 0.9105, 0.4396, 0.8634] came
    # out best at full stats (sigma=7.4739mm, alpha=0.1817, combined=2.88).
    # All four candidates shifted sigma_x by ~+0.022mm between 40k and full
    # stats, essentially identically regardless of X - i.e. plain shot noise
    # at 40k stats is large enough to matter for both the ranking and the
    # absolute readings, not a beam-file artifact (beam file confirmed
    # randomly ordered). Decision: run every evaluation at full statistics
    # from here on rather than search-cheap-then-refine.

    # Full-statistics MOBO round: every evaluation (not just a final refine)
    # now runs at the full beam count (ngenerate=124262, set at top of file).
    # Bounds are +/-20% around the best full-stat point found so far
    # (X=[0.7223, 0.9105, 0.4396, 0.8634], combined=2.88). batch_size=12
    # matches the Mac Studio's core count exactly (no oversubscription).
    # Expect ~30-45 min per evaluation at this ngenerate, so each 12-wide
    # wave takes about that long regardless of parallelism - 24 initial (2
    # waves) + 15 iterations x 12 (15 waves) = 17 waves total, roughly
    # 8.5-13 hours. Adjust n_initial/n_iter below to fit your time budget.
    # bounds_full_stat = {
    #     "b4": (0.5778, 0.8668),
    #     "b5": (0.7284, 1.0926),
    #     "b6": (0.3517, 0.5275),
    #     "b7": (0.6907, 1.0361),
    # }
    # config = OptConfig(
    #     objectives=["sigma_x_err", "alpha_x_err"],
    #     constraints={},
    #     bounds=bounds_full_stat,
    #     n_initial=24,
    #     n_iter=15,
    #     batch_size=12,
    #     mode="mobo",
    # )
    # model = S1GLModel(Builder)
    # problem = S1GLMatchMOBO(model, config, targets, scales=scales)
    # mobo = BDSIMOpt(problem, "MOBO_S1GL_24_15_12_FullStat_SigmaAlphaPareto")
    # mobo.optimise()
    # mobo.plot_results()
    # Result: converged to the same point as the hand-derived local search
    # (X=[0.7223, 0.9105, 0.4396, 0.8634], combined=2.88 vs alpha target) -
    # confirmed clean local optimum for the alpha-based metric, ~6h42m wall
    # time (204 evals, faster than the 8.5-13h estimate). Superseded below:
    # realised alpha_x is normalised by emittance the same way beta is by
    # size, so a fixed alpha tolerance doesn't track a fixed physical
    # divergence once the Gabor lenses grow emittance unevenly across
    # candidates. Switched the second objective to sigma_xp (actual RMS
    # divergence, extract_optics/S1GLModel/S1GLMatchMOBO updated accordingly)
    # with the same target/tolerance convention (0.0 +/- 0.15) as alpha had.

    # Quick sanity check before committing to a full search on the new
    # objective: evaluate the Pareto front from the last full-stat round
    # under the new sigma_xp metric, to see its actual magnitude here and
    # confirm 0.15 is a sensible tolerance (not, e.g., orders of magnitude
    # looser than any physically plausible divergence) before spending
    # hours on a search that might turn out to have a near-flat 2nd
    # objective.
    # check_points = {
    #     0: [0.7223, 0.9105, 0.4396, 0.8634],
    #     1: [0.74671348, 0.89498484, 0.36677166, 0.87259713],
    #     2: [0.60468059, 0.93467773, 0.45552769, 0.86996422],
    #     3: [0.82631137, 0.8771421, 0.50997987, 0.82043973],
    #     4: [0.70587819, 0.86507612, 0.46345636, 0.87363737],
    #     5: [0.68954866, 0.87351239, 0.44176398, 0.88052624],
    # }
    #
    # model = S1GLModel(Builder)
    # results = {}
    # with ProcessPoolExecutor(max_workers=len(check_points)) as pool:
    #     futures = {
    #         pool.submit(model.run, np.array(x), f"sigmaxp_check_{k}"): k
    #         for k, x in check_points.items()
    #     }
    #     for fut in as_completed(futures):
    #         k = futures[fut]
    #         results[k] = fut.result()
    #
    # sigma_tol = scales["sigma_x"]
    # sigma_xp_tol = scales["sigma_xp"]
    # rows = []
    # for k, x in check_points.items():
    #     r = results[k]
    #     sigma_units = abs(r["sigma_x"] - targets["sigma_x"]) / sigma_tol
    #     sigma_xp_units = abs(r["sigma_xp"] - targets["sigma_xp"]) / sigma_xp_tol
    #     combined = (sigma_units**2 + sigma_xp_units**2) ** 0.5
    #     rows.append((combined, x, r["sigma_x"], r["sigma_xp"]))
    #
    # rows.sort(key=lambda row: row[0])
    # for combined, x, sigma_x, sigma_xp in rows:
    #     print(f"X={[round(v,4) for v in x]}  sigma_x={sigma_x*1e3:.4f}mm  "
    #           f"sigma_xp={sigma_xp:.4e}  combined={combined:.2f}")
    # Result: sigma_xp read 5.3e-4 to 1.25e-3 rad across these 6 points - with
    # the original 0.15 tolerance every one scored <0.01 tolerance-units,
    # making sigma_xp a total no-op (the "best" point by that combined score
    # was actually the *worst* for divergence). Tightened tolerance to 1e-4
    # (chosen to be meaningfully restrictive relative to the observed
    # spread, per instruction to prefer a more restrictive tolerance).
    # Re-scoring this same data at tol=1e-4: X=[0.7223, 0.9105, 0.4396,
    # 0.8634] is still the best (combined=6.19) - reassuring cross-check
    # that the earlier alpha-based optimum remains good under the corrected,
    # properly-scaled divergence metric too.

    # Full-statistics MOBO round on the corrected objective pair
    # (sigma_x_err, sigma_xp_err), same +/-20% window as before since the
    # underlying physical region hasn't changed - only which trade-off point
    # within it is preferred.
    bounds_full_stat = {
        "b4": (0.5778, 0.8668),
        "b5": (0.7284, 1.0926),
        "b6": (0.3517, 0.5275),
        "b7": (0.6907, 1.0361),
    }
    config = OptConfig(
        objectives=["sigma_x_err", "sigma_xp_err"],
        constraints={},
        bounds=bounds_full_stat,
        n_initial=24,
        n_iter=15,
        batch_size=12,
        mode="mobo",
    )
    model = S1GLModel(Builder)
    problem = S1GLMatchMOBO(model, config, targets, scales=scales)
    mobo = BDSIMOpt(problem, "MOBO_S1GL_24_15_12_FullStat_SigmaXpPareto")
    mobo.optimise()
    mobo.plot_results()

