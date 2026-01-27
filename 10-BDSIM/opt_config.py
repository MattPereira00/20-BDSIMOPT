from dataclasses import dataclass
from typing import Callable, List, Dict

@dataclass
class OptimisationConfig:
    model_name: str                  # "triplet", "halbach_triplet", "halbach_double_triplet"
    objectives: List[str]             # ["T", "A", "D"]
    constraints: Dict[str, float]     # {"T_min": 0.01}
    bounds: Dict[str, tuple]          # {"k1": (0,1), "ap1": (0.02,0.06), ...}
    n_initial: int
    n_iter: int
    batch_size: int
    mc_samples: int = 64