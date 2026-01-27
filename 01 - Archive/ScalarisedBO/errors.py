import numpy as np
import pybdsim


def compute_metrics_and_uncertainties(
        sigma_x, sigma_y, sigma_xp, sigma_yp,
        emitt_x, emitt_y,
        Nin, Nout,
        variances=None,  # dict of variances OR
        cov=None  # full covariance matrix
):
    """
    Computes:
      - Transmission T
      - Asymmetry A = A_sigma + A_sigma' + A_emitt
      - Divergence asymmetry D from sigma_xp/sigma_yp

    Supports uncertainty propagation for all 3 metrics.

    Inputs:
      sigma_x, sigma_y, sigma_xp, sigma_yp, emitt_x, emitt_y, Nin, Nout : floats
      variances: dict with entries var_sigma_x, var_sigma_y, ..., var_Nin, var_Nout
      cov: optional 8×8 covariance matrix in order:
           [σx, σy, σxp, σyp, εx, εy, Nin, Nout]

    Returns:
      Dictionary with both metric values and uncertainties:
      {
        "T": (value, sigma_T),
        "A": (value, sigma_A),
        "D": (value, sigma_D)
      }
    """

    # --- Step 1: store variables in vector form ---
    x = np.array([
        sigma_x, sigma_y, sigma_xp, sigma_yp,
        emitt_x, emitt_y, Nin, Nout
    ], dtype=float)

    # ---------- Helper: get covariance matrix ------------
    if cov is not None:
        C = np.asarray(cov, float)
        if C.shape != (8, 8):
            raise ValueError("cov must be 8x8 in order [σx,σy,σxp,σyp,εx,εy,Nin,Nout]")
    elif variances is not None:
        C = np.diag([
            variances.get("var_sigma_x", 0),
            variances.get("var_sigma_y", 0),
            variances.get("var_sigma_xp", 0),
            variances.get("var_sigma_yp", 0),
            variances.get("var_emitt_x", 0),
            variances.get("var_emitt_y", 0),
            variances.get("var_Nin", 0),
            variances.get("var_Nout", 0),
        ])
    else:
        raise ValueError("Provide either 'variances' or 'cov'.")

    # ============================================================
    #  METRIC 1: TRANSMISSION  T = Nout / Nin
    # ============================================================
    Nin_val, Nout_val = Nin, Nout
    T = Nout_val / Nin_val

    # Gradient of T wrt variables
    dT_dNin = -Nout_val / (Nin_val ** 2)
    dT_dNout = 1.0 / Nin_val

    J_T = np.array([0, 0, 0, 0, 0, 0, dT_dNin, dT_dNout])
    var_T = J_T @ C @ J_T.T
    sigma_T = np.sqrt(max(var_T, 0))

    # ============================================================
    #  ASYMMETRY  A = Aσ + Aσ' + Aε
    # ============================================================

    def asym_and_derivs(x, y):
        """Returns asymmetry component and analytic jacobian entries."""
        D = x + y if abs(x + y) > 1e-12 else 1e-12
        a_signed = 2 * (x - y) / D
        a = abs(a_signed)
        s = np.sign(a_signed)
        da_dx = s * (4 * y) / (D ** 2)
        da_dy = s * (-4 * x) / (D ** 2)
        return a, da_dx, da_dy

    # Components
    A1, dA1_dx, dA1_dy = asym_and_derivs(sigma_x, sigma_y)
    A2, dA2_dxp, dA2_dyp = asym_and_derivs(sigma_xp, sigma_yp)
    A3, dA3_dex, dA3_dey = asym_and_derivs(emitt_x, emitt_y)

    A = A1 + A2 + A3

    # Full gradient of A wrt all 8 variables
    J_A = np.array([
        dA1_dx,  # dA/dσx
        dA1_dy,  # dA/dσy
        dA2_dxp,  # dA/dσxp
        dA2_dyp,  # dA/dσyp
        dA3_dex,  # dA/dεx
        dA3_dey,  # dA/dεy
        0, 0  # Nin, Nout do not affect A
    ])

    var_A = J_A @ C @ J_A.T
    sigma_A = np.sqrt(max(var_A, 0))

    # ---------------------------------------------
    # Divergence D = 0.5 * (sigma_xp + sigma_yp)
    # ---------------------------------------------
    D = 0.5 * (sigma_xp + sigma_yp)
    # derivatives are simple constants
    dD_dxp = 0.5
    dD_dyp = 0.5
    J_D = np.array([0.0, 0.0, dD_dxp, dD_dyp, 0.0, 0.0, 0.0, 0.0])
    var_D = float(J_D @ C @ J_D.T)
    sigma_D = np.sqrt(max(var_D, 0.0))

    # ============================================================
    #  RETURN RESULTS
    # ============================================================
    return {
        "T": (T, sigma_T),
        "A": (A, sigma_A),
        "D": (D, sigma_D)
    }

fname = "Baseline_Nozzle"

datadir = "90-BDSIMData"
rundir = datadir + "/" + fname

output_optics = pybdsim.Data.Load(rundir + "/" + "baseline_optics.root").optics

init_alpha_x, init_alpha_y = output_optics.Alpha_x()[0], output_optics.Alpha_y()[0]

sigma_x, sigma_y = output_optics.Sigma_x()[-1], output_optics.Sigma_y()[-1]
sigma_xp, sigma_yp = output_optics.Sigma_xp()[-1], output_optics.Sigma_yp()[-1]
emitt_x, emitt_y = output_optics.Emitt_x()[-1], output_optics.Emitt_y()[-1]
n_parts_in = output_optics.Npart()[0]
n_parts_out = output_optics.Npart()[-1]

# Assume these are your reported standard errors (floats)
dsx   = output_optics.Sigma_Sigma_x()[-1]   # δσ_x
dsy   = output_optics.Sigma_Sigma_y()[-1]
dsxp  = output_optics.Sigma_Sigma_xp()[-1]
dsyp  = output_optics.Sigma_Sigma_yp()[-1]
demittx = output_optics.Sigma_Emitt_x()[-1]
demitty = output_optics.Sigma_Emitt_y()[-1]
dNin  = 0.0    # often Nin is known exactly; set to 0 if so
dNout = (n_parts_out**0.5)  # if Poisson approx or an experimental SE you were given

# Build variances dict expected by compute_metrics_and_uncertainties
variances = {
    "var_sigma_x":   dsx**2,
    "var_sigma_y":   dsy**2,
    "var_sigma_xp":  dsxp**2,
    "var_sigma_yp":  dsyp**2,
    "var_emitt_x":   demittx**2,
    "var_emitt_y":   demitty**2,
    "var_Nin":       dNin**2,
    "var_Nout":      dNout**2,
}

# call the function (assuming compute_metrics_and_uncertainties is in scope)
results = compute_metrics_and_uncertainties(
    sigma_x, sigma_y, sigma_xp, sigma_yp,
    emitt_x, emitt_y,
    n_parts_in, n_parts_out,
    variances=variances,
    cov=None
)

print("Transmission T = {:.6f} ± {:.6f}".format(*results["T"]))
print("Asymmetry A   = {:.6f} ± {:.6f}".format(*results["A"]))
print("Divergence D  = {:.6f} ± {:.6f}".format(*results["D"]))

