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
    target_t = 0.05
    target_a = 0.05
    target_d = 0.013

    # Limits
    t_min = 0.041
    t_min_pen = 1e3
    a_max = 0.1
    a_max_pen = 1e3
    d_max = 0.015
    d_max_pen = 1e3
    max_loss = 1e4

    # Normalised squared deviations (smooth, GP-friendly)
    l_t = ((transmission / target_t) - 1)**2
    l_a = ((asymmetry    / target_a) - 1)**2
    l_d = ((divergence   / target_d) - 1)**2

    # Weighting
    w_t = 1.0
    w_a = 1.0
    w_d = 1.0

    # Final loss
    loss = w_t * l_t + w_a * l_a + w_d * l_d

    if transmission < t_min:
        loss += t_min_pen * ((t_min - transmission) / t_min) ** 2

    if asymmetry > a_max:
        loss += a_max_pen * ((asymmetry - a_max) / a_max) ** 2

    if divergence > d_max:
        loss += d_max_pen * ((divergence - d_max) / d_max) ** 2

    loss = float(min(loss, max_loss))

    return transmission, asymmetry, divergence, loss

def extract_metrics_and_uncertainties(outfile):
    """
    Computes:
      - Transmission T
      - Asymmetry A = A_sigma + A_sigma + A_emitt
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
    dn_in = 0.0  # Known exactly
    dn_out = (n_parts_out ** 0.5)  # Poisson approx

    variances = {
        "var_sigma_x": dsx ** 2,
        "var_sigma_y": dsy ** 2,
        "var_sigma_xp": dsxp ** 2,
        "var_sigma_yp": dsyp ** 2,
        "var_emitt_x": demittx ** 2,
        "var_emitt_y": demitty ** 2,
        "var_Nin": dn_in ** 2,
        "var_Nout": dn_out ** 2,
    }

    x = np.array([
        sigma_x, sigma_y, sigma_xp, sigma_yp,
        emitt_x, emitt_y, n_parts_in, n_parts_out
    ], dtype=float)

    c = np.diag([
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
    n_in_val, n_out_val = n_parts_in, n_parts_out
    t = n_out_val / n_in_val

    # Gradient of T wrt variables
    dt_dn_in = -n_out_val / (n_in_val ** 2)
    dt_dn_out = 1.0 / n_in_val

    j_t = np.array([0, 0, 0, 0, 0, 0, dt_dn_in, dt_dn_out])
    var_t = j_t @ c @ j_t.T
    sigma_t = np.sqrt(max(var_t, 0))

    # ============================================================
    #  ASYMMETRY  A = Aσ + Aσ' + Aε
    # ============================================================

    def asym_and_derivs(x, y):
        """Returns asymmetry component and analytic jacobian entries."""
        d = x + y if abs(x + y) > 1e-12 else 1e-12
        a_signed = 2 * (x - y) / d
        a = abs(a_signed)
        s = np.sign(a_signed)
        da_dx = s * (4 * y) / (d ** 2)
        da_dy = s * (-4 * x) / (d ** 2)
        return a, da_dx, da_dy

    # Components
    a1, d_a1_dx, d_a1_dy = asym_and_derivs(sigma_x, sigma_y)
    a2, d_a2_dxp, d_a2_dyp = asym_and_derivs(sigma_xp, sigma_yp)
    a3, d_a3_dex, d_a3_dey = asym_and_derivs(emitt_x, emitt_y)

    a = a1 + a2 + a3

    # Full gradient of A wrt all 8 variables
    j_a = np.array([
        d_a1_dx,  # dA/dσx
        d_a1_dy,  # dA/dσy
        d_a2_dxp,  # dA/dσxp
        d_a2_dyp,  # dA/dσyp
        d_a3_dex,  # dA/dεx
        d_a3_dey,  # dA/dεy
        0, 0  # Nin, Nout do not affect A
    ])

    var_a = j_a @ c @ j_a.T
    sigma_a = np.sqrt(max(var_a, 0))

    # ---------------------------------------------
    # Divergence D = 0.5 * (sigma_xp + sigma_yp)
    # ---------------------------------------------
    d = 0.5 * (sigma_xp + sigma_yp)
    # derivatives are simple constants
    dd_dxp = 0.5
    dd_dyp = 0.5
    j_d = np.array([0.0, 0.0, dd_dxp, dd_dyp, 0.0, 0.0, 0.0, 0.0])
    var_d = float(j_d @ c @ j_d.T)
    sigma_d = np.sqrt(max(var_d, 0.0))

    # ============================================================
    #  RETURN RESULTS
    # ============================================================
    return t, sigma_t, a, sigma_a, d, sigma_d


def extract_loss_old(outfile):
    output_optics = pybdsim.Data.Load(outfile + "_optics.root").optics

    sigma_x, sigma_y = output_optics.Sigma_x()[-1], output_optics.Sigma_y()[-1]
    sigma_xp, sigma_yp = output_optics.Sigma_xp()[-1], output_optics.Sigma_yp()[-1]
    emitt_x, emitt_y = output_optics.Emitt_x()[-1], output_optics.Emitt_y()[-1]
    n_parts_in, n_parts_out = output_optics.Npart()[0], output_optics.Npart()[-1]

    # Calculate metrics
    asymmetry_sigma = (sigma_x - sigma_y)**2 / ((sigma_x + sigma_y) / 2)**2
    asymmetry_sigma_p = (sigma_xp - sigma_yp)**2 / ((sigma_xp + sigma_yp) / 2)**2
    asymmetry_emit = (emitt_x - emitt_y)**2 / ((emitt_x + emitt_y) / 2)**2

    transmission = n_parts_out / n_parts_in
    asymmetry = asymmetry_sigma + asymmetry_sigma_p + asymmetry_emit
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
    alpha_x, alpha_y = output_optics.Alpha_x()[-1], output_optics.Alpha_y()[-1]

    return sigma_x, sigma_y, alpha_x, alpha_y