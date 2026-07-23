from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class OptConfig:
    objectives: List[str]
    constraints: Dict[str, float]
    bounds: Dict[str, tuple]
    n_initial: int
    n_iter: int
    batch_size: int
    mc_samples: int = 64
    mode: str = 'mobo' # 'mobo' or 'scalar'
    device: str = 'cpu'

    # Trust-region search (see BDSIMOpt._tr_*): instead of a fixed `bounds`
    # box for every iteration, shrink/expand a window around the current
    # incumbent based on whether recent batches are improving on it. `bounds`
    # above still sets the outer physical limits the window is clipped to.
    use_trust_region: bool = False
    tr_length_init: float = 0.8   # initial window size, as a fraction of `bounds`
    tr_length_min: float = 0.5 ** 7  # below this, the region has collapsed
    tr_length_max: float = 1.6    # capped at 1.0 in practice (can't exceed `bounds`)
    tr_succ_tol: int = 3          # consecutive improving batches before expanding
    tr_fail_tol: Optional[int] = None  # consecutive non-improving batches before
                                        # shrinking; None = auto from dim/batch_size
