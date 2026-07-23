from typing import Callable, Dict, Optional

def make_weighted_target_loss(
        targets: Dict[str, float],
        weights: Optional[Dict[str, float]],
        scales: Optional[Dict[str, float]],
) -> Callable[[Dict[str, float]], float]:
    """
    Create a weighted squared-error loss function from user targets, weights, and scales.

    Returns a function that accepts a metrics dict and returns a scalar loss.
    """
    if weights is None:
        weights = {k: 1.0 for k in targets.keys()}

    if scales is None:
        scales = {k: abs(v) if abs(v) > 0 else 1.0 for k, v in targets.items()}

    def loss_fn(metrics: dict):
        loss = 0.0

        for key, target in targets.items():
            weight = weights.get(key, 1.0)
            scale = scales.get(key, 1.0)
            metric_value = metrics.get(key, float(1e6))

            loss += weight * ((metric_value - target) / scale) ** 2
        return float(loss)

    return loss_fn

def make_custom_metric_loss(
    targets: Dict[str, float],
    weights: Optional[Dict[str, float]],
    scales: Optional[Dict[str, float]],
) -> Callable[[Dict[str, float]], float]:
    """
    Create a weighted loss where each metric can use a different penalty.

    Inputs match make_weighted_target_loss:
      - targets: desired values for each metric
      - weights: per-metric weighting (default 1.0)
      - scales: per-metric normalization scales (default abs(target) or 1.0)

    Returns:
      loss_fn(metrics_dict) -> float
    """
    if weights is None:
        weights = {k: 1.0 for k in targets.keys()}

    if scales is None:
        scales = {k: abs(v) if abs(v) > 0 else 1.0 for k, v in targets.items()}

    def loss_fn(metrics: Dict[str, float]) -> float:
        loss = 0.0

        for key, target in targets.items():
            weight = float(weights.get(key, 1.0))
            scale = float(scales.get(key, 1.0))
            metric_value = float(metrics.get(key, 1e6))

            # Normalized signed error
            e = (metric_value - target) / scale

            # Per-metric penalty choice:
            # - sigma_* : Huber-like (quadratic near target, linear far away)
            # - alpha_* : asymmetric quadratic (penalize positive alpha more)
            # - default : squared normalized error
            if key.startswith("sigma_"):
                # Huber with delta=1 in normalized units
                ae = abs(e)
                if ae <= 1.0:
                    contribution = 0.5 * ae * ae
                else:
                    contribution = ae - 0.5

            elif key.startswith("alpha_"):
                # Asymmetric: overshoot (e > 0) penalized more
                over_penalty = 2.0
                under_penalty = 1.0
                if e > 0.0:
                    contribution = over_penalty * (e * e)
                else:
                    contribution = under_penalty * (e * e)

            else:
                contribution = e * e

            loss += weight * contribution

        return float(loss)

    return loss_fn


def make_coupled_metric_loss(
        targets: Dict[str, float],
        weights: Optional[Dict[str, float]],
        scales: Optional[Dict[str, float]],
        sigma_coupling_power: float = 1.5,  # Was implicitly 2.0
        sigma_allowance: float = 1.5,
) -> Callable[[Dict[str, float]], float]:
    """
    Strongly couple sigma and alpha: alpha penalty explodes if sigma drifts.
    """
    if weights is None:
        weights = {k: 1.0 for k in targets.keys()}
    if scales is None:
        scales = {k: abs(v) if abs(v) > 0 else 1.0 for k, v in targets.items()}

    sigma_target = targets.get("sigma_x", 7.5e-3)
    alpha_target = targets.get("alpha_x", 0.0)
    sigma_scale = scales.get("sigma_x", 1.0)
    alpha_scale = scales.get("alpha_x", 1.0)
    w_sigma = weights.get("sigma_x", 1.5)
    w_alpha = weights.get("alpha_x", 1.0)

    def loss_fn(metrics: Dict[str, float]) -> float:
        sigma_x = metrics.get("sigma_x", 1e6)
        alpha_x = metrics.get("alpha_x", 1e6)

        e_sigma = (sigma_x - sigma_target) / sigma_scale
        e_alpha = (alpha_x - alpha_target) / alpha_scale

        # Sigma term (Huber)
        ae_sigma = abs(e_sigma)
        if ae_sigma <= 1.0:
            L_sigma = 0.5 * ae_sigma * ae_sigma
        else:
            L_sigma = ae_sigma - 0.5

        # Alpha term with STRONG coupling
        ae_alpha = abs(e_alpha)

        # If sigma is off, alpha penalty explodes
        if ae_sigma > sigma_allowance:
            # Sigmoid-like gate: huge penalty if sigma drifts
            coupling_factor = 1.0 + 100.0 * ((ae_sigma - sigma_allowance) ** sigma_coupling_power)
        else:
            coupling_factor = 1.0

        if ae_alpha <= 0.5:
            L_alpha = 0.5 * ae_alpha * ae_alpha
        else:
            L_alpha = ae_alpha - 0.25

        loss = w_sigma * L_sigma + w_alpha * coupling_factor * L_alpha

        return float(loss)

    return loss_fn