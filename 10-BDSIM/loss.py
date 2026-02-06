import numpy as np
import pybdsim

def extract_loss(outfile):
    optics = pybdsim.Data.Load(outfile + "_optics.root").optics

    # Extract run parameters
    sigma_x, sigma_y   = optics.Sigma_x()[-1],  optics.Sigma_y()[-1]
    sigma_xp, sigma_yp = optics.Sigma_xp()[-1], optics.Sigma_yp()[-1]
    emitt_x, emitt_y   = optics.Emitt_x()[-1],  optics.Emitt_y()[-1]

    n_in  = optics.Npart()[0]
    n_out = optics.Npart()[-1]
    transmission = n_out / n_in

    # Calculate Metrics
    asym_sigma   = ((sigma_x  - sigma_y ) / (0.5*(sigma_x  + sigma_y)))**2
    asym_sigmap  = ((sigma_xp - sigma_yp) / (0.5*(sigma_xp + sigma_yp)))**2
    asym_emitt   = ((emitt_x  - emitt_y)  / (0.5*(emitt_x  + emitt_y)))**2

    asymmetry = asym_sigma + asym_sigmap + asym_emitt

    divergence = 0.5 * (sigma_xp + sigma_yp)

    # Targets
    target_T = 0.05
    target_A = 0.05
    target_D = 0.013

    # Limits
    T_MIN = 0.041
    T_MIN_PEN = 1e3
    A_MAX = 0.1
    A_MAX_PEN = 1e3
    D_MAX = 0.015
    D_MAX_PEN = 1e3
    MAX_LOSS = 1e4

    # Normalised squared deviations (smooth, GP-friendly)
    L_T = ((transmission / target_T) - 1)**2
    L_A = ((asymmetry    / target_A) - 1)**2
    L_D = ((divergence   / target_D) - 1)**2

    # Weighting
    w_T = 1.0
    w_A = 1.0
    w_D = 1.0

    # Final loss
    loss = w_T * L_T + w_A * L_A + w_D * L_D

    if transmission < T_MIN:
        loss += T_MIN_PEN * ((T_MIN - transmission) / T_MIN) ** 2

    if asymmetry > A_MAX:
        loss += A_MAX_PEN * ((asymmetry - A_MAX) / A_MAX) ** 2

    if divergence > D_MAX:
        loss += D_MAX_PEN * ((divergence - D_MAX) / D_MAX) ** 2

    loss = float(min(loss, MAX_LOSS))

    return transmission, asymmetry, divergence, loss

def extract_metrics_and_uncertainties(outfile):
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
    output_optics = pybdsim.Data.Load(outfile + "_optics.root").optics

    sigma_x, sigma_y = output_optics.Sigma_x()[-1], output_optics.Sigma_y()[-1]
    sigma_xp, sigma_yp = output_optics.Sigma_xp()[-1], output_optics.Sigma_yp()[-1]
    emitt_x, emitt_y = output_optics.Emitt_x()[-1], output_optics.Emitt_y()[-1]
    n_parts_in = output_optics.Npart()[0]
    n_parts_out = output_optics.Npart()[-1]

    # Errors from Rebdsim or Poisson Statistics
    dsx = output_optics.Sigma_Sigma_x()[-1]
    dsy = output_optics.Sigma_Sigma_y()[-1]
    dsxp = output_optics.Sigma_Sigma_xp()[-1]
    dsyp = output_optics.Sigma_Sigma_yp()[-1]
    demittx = output_optics.Sigma_Emitt_x()[-1]
    demitty = output_optics.Sigma_Emitt_y()[-1]
    dNin = 0.0  # Known exactly
    dNout = (n_parts_out ** 0.5)  # Poisson approx

    variances = {
        "var_sigma_x": dsx ** 2,
        "var_sigma_y": dsy ** 2,
        "var_sigma_xp": dsxp ** 2,
        "var_sigma_yp": dsyp ** 2,
        "var_emitt_x": demittx ** 2,
        "var_emitt_y": demitty ** 2,
        "var_Nin": dNin ** 2,
        "var_Nout": dNout ** 2,
    }
    # --- Step 1: store variables in vector form ---
    x = np.array([
        sigma_x, sigma_y, sigma_xp, sigma_yp,
        emitt_x, emitt_y, n_parts_in, n_parts_out
    ], dtype=float)

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
    # ============================================================
    #  METRIC 1: TRANSMISSION  T = Nout / Nin
    # ============================================================
    Nin_val, Nout_val = n_parts_in, n_parts_out
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
    return T, sigma_T, A, sigma_A, D, sigma_D


def extract_loss_old(outfile):
    output_optics = pybdsim.Data.Load(outfile + "_optics.root").optics

    sigma_x, sigma_y = output_optics.Sigma_x()[-1], output_optics.Sigma_y()[-1]
    sigma_xp, sigma_yp = output_optics.Sigma_xp()[-1], output_optics.Sigma_yp()[-1]
    emitt_x, emitt_y = output_optics.Emitt_x()[-1], output_optics.Emitt_y()[-1]
    n_parts_in, n_parts_out = output_optics.Npart()[0], output_optics.Npart()[-1]

    # Calculate metrics
    asymmetry_sigma = (sigma_x - sigma_y)**2 / ((sigma_x + sigma_y) / 2)**2
    asymmetry_sigma_p = (sigma_xp - sigma_yp)**2 / ((sigma_xp + sigma_yp) / 2)**2
    asymettry_emitt = (emitt_x - emitt_y)**2 / ((emitt_x + emitt_y) / 2)**2

    transmission = n_parts_out / n_parts_in
    asymmetry = asymmetry_sigma + asymmetry_sigma_p + asymettry_emitt
    divergence = 0.5 * (sigma_xp + sigma_yp)

    reward_scaling = 10.0

    weight_transmission = 1.0
    weight_asymmetry = 1.0
    weight_divergence= 1.0

    target_transmission = 0.05
    target_asymmetry = 0.02
    target_divergence = 0.012

    loss_transmission = (target_transmission-transmission)**2 / target_transmission**2
    loss_asymmetry = (asymmetry-target_asymmetry)**2 / target_asymmetry**2
    loss_divergence = (divergence-target_divergence)**2 / target_divergence**2



    loss = reward_scaling * (weight_transmission * loss_transmission
                             + weight_asymmetry * loss_asymmetry
                             + weight_divergence * loss_divergence)

    return transmission, asymmetry, divergence, loss


def extract_optics(outfile):
    output_optics = pybdsim.Data.Load(outfile + "_optics.root").optics

    sigma_x, sigma_y = output_optics.Sigma_x()[-1], output_optics.Sigma_y()[-1]
    sigma_xp, sigma_yp = output_optics.Sigma_xp()[-1], output_optics.Sigma_yp()[-1]


    return sigma_x, sigma_y, sigma_xp, sigma_yp