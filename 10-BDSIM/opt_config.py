from dataclasses import dataclass
from typing import List, Dict

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
