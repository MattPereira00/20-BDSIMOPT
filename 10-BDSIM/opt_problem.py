import torch
from typing import Callable, List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from opt_config import OptimisationConfig

class OptProblem:
    def __init__(self, config: OptimisationConfig):
        self.config = config
        self.objective = config.objective
        self.constraints = config.constraints

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """
        Run the model

        :param X: torch.Tensor
        :return Y: torch.Tensor
        """
        raise NotImplementedError

    def get_constraints(self) -> List[Callable]:
        """
        Return constraints as BoTorch requires them
        """
        return []


class NozzleQuadProblem(OptProblem):
    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """
        Parallel evaluation of objective vectors.
        Returns Y = [T, -A, -D] for each row in X.
        """
        self.run_id = 0

        X_np = X.detach().cpu().numpy()
        n = X_np.shape[0]
        results = [None] * n

        with ProcessPoolExecutor(max_workers=self.config.batch_size) as pool:

            futures = {}
            for i in range(n):
                run_id = self.run_id
                self.run_id += 1
                futures[pool.submit(self.model.run_single, X_np[i], run_id)] = i

            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    T, A, D = fut.result()
                except Exception as e:
                    print(f"[Worker {i}] ERROR:", e)
                    T, A, D = 0.0, 1.0, 1.0  # safe fallback

                results[i] = [T, -A, -D]

        return torch.tensor(
            results,
            dtype=torch.double,
            device=self.config.device,
        )
