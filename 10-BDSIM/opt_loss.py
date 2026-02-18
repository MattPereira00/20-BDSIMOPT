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
