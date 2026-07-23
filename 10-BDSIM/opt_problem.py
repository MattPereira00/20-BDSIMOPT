import torch
from opt_loss import *
from typing import Callable, List, Dict
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from opt_config import OptConfig

class OptProblem(ABC):
    def __init__(self, model, config: OptConfig):
        self.model = model
        self.config = config
        self.run_id = 0
        self.last_raw_results = None

    @abstractmethod
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the model

        :param x: torch.Tensor
        :return y: torch.Tensor
        """
        pass

    @abstractmethod
    def pack_objectives(self, raw: Dict[str, float]) -> torch.Tensor:
        """
        Convert raw physical outputs into a BoTorch objective vector.

        Returns:
            torch.Tensor shape (n_objectives)
        """
        pass

    def objective_index(self, name: str) -> int:
        try:
            return self.objective_names.index(name)
        except ValueError:
            raise ValueError(
                f"Constraint refers to objective '{name}', "
                f"but this problem has objectives {self.objective_names}"
            )

    @abstractmethod
    def get_constraints(self) -> List[Callable]:
        """
        Return constraints as BoTorch requires them
        """
        pass

    def get_feasible_mask(self, y: torch.Tensor) -> torch.Tensor:
        """
        Returns a boolean mask of shape (n)
        """
        n = y.shape[0]

        if not self.get_constraints():
            return torch.ones(n, dtype=torch.bool, device=y.device)

        feasible = torch.ones(n, dtype=torch.bool, device=y.device)

        for c in self.get_constraints():
            val = c(y)

            if val.ndim != 1:
                val = val.view(-1)

            feasible &= (val >= 0)

        return feasible

    @abstractmethod
    def objective_labels(self) -> List[str]:
        """
        Human-readable labels for objectives
        """
        pass

    @abstractmethod
    def format_result(self, x, y, raw) -> str:
        """
        Returns a formatted string for logging
        """
        pass

    def get_ref_point(self) -> torch.Tensor:
        """
        Generic pessimistic reference point:
        works for capture, optics matching, and anything else.
        """
        ref = [-1e3] * len(self.config.objectives)
        return torch.tensor(ref, dtype=torch.double, device=self.config.device)


class TripletCapture(OptProblem):
    def __init__(self, model, config: OptConfig):
        super().__init__(model, config)
        self.objective_names = ["T", "A", "D"]
        self.objective_signs = {
            "T": +1,
            "A": -1,
            "D": -1,
        }
    def evaluate(self, x: torch.Tensor):
        x_np = x.detach().cpu().numpy()
        n = x_np.shape[0]
        raw_results = [None] * n
        packed_results = [None] * n

        with ProcessPoolExecutor(max_workers=self.config.batch_size) as pool:
            futures = {}
            for i in range(n):
                run_id = self.run_id
                self.run_id += 1
                futures[pool.submit(self.model.run, x_np[i], run_id)] = i

            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    raw = fut.result()  # {"T":..., "A":..., "D":...}
                    raw_results[i] = raw
                    packed_results[i] = self.pack_objectives(raw)
                except Exception as e:
                    print(f"[Worker {i}] ERROR:", e)
                    fallback = {"T": 0.0, "A": 1e3, "D": 1e3}
                    raw_results[i] = fallback
                    packed_results[i] = self.pack_objectives(fallback)

            y = torch.stack(packed_results, dim=0)
            return y, raw_results

    def get_constraints(self):
        constraints = []

        for key, value in self.config.constraints.items():
            obj, kind = key.rsplit("_", 1)

            idx = self.objective_index(obj)
            sign = self.objective_signs[obj]

            if kind == "min":
                constraints.append(
                    lambda y, i=idx, s=sign, v=value: s * y[..., i] - v
                )
            elif kind == "max":
                constraints.append(
                    lambda y, i=idx, s=sign, v=value: v - s * y[..., i]
                )
            else:
                raise ValueError(f"Unknown constraint type '{kind}'")

        return constraints

    def get_feasible_mask(self, y: torch.Tensor) -> torch.Tensor:
        """
        Returns a boolean mask of shape (n)
        True if all constraints are satisfied.
        """
        n = y.shape[0]

        if not self.get_constraints():
            return torch.ones(n, dtype=torch.bool, device=y.device)

        feasible = torch.ones(n, dtype=torch.bool, device=y.device)

        for c in self.get_constraints():
            val = c(y)
            if val.ndim != 1:
                val = val.view(-1)
            feasible &= (val >= 0)

        return feasible

    def objective_labels(self):
        return ["Transmission", "Asymmetry", "Divergence"]

    def format_result(self, x, y, raw):
        return (
            f"T={raw['T']:.4f}  "
            f"A={raw['A']:.4f}  "
            f"D={raw['D']:.4f}   "
            f"X={x.cpu().numpy()}"
        )

    def pack_objectives(self, raw: dict) -> torch.Tensor:
        out = []
        for obj in self.config.objectives:
            val = float(raw[obj])
            if obj in ("A", "D"):  # minimise
                val = -val
            out.append(val)

        return torch.tensor(out, dtype=torch.double, device=self.config.device)

class DoubleTripletProblem(OptProblem):
    def __init__(self, model, config: OptConfig):
        super().__init__(model, config)

        self.objective_names = ["T", "A", "D"]
        self.objective_signs = {
            "T": +1,
            "A": -1,
            "D": -1,
        }

    def evaluate(self, x: torch.Tensor):
        x_np = x.detach().cpu().numpy()
        n = x_np.shape[0]

        raw_results = [None] * n
        packed = [None] * n

        with ProcessPoolExecutor(max_workers=self.config.batch_size) as pool:
            futures = {
                pool.submit(self.model.run, x_np[i], self.run_id + i): i
                for i in range(n)
            }
            self.run_id += n

            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    raw = fut.result()
                    raw_results[i] = raw
                    packed[i] = self.pack_objectives(raw)
                except Exception as e:
                    print(f"[Worker {i}] ERROR:", e)
                    fallback = {"T": 0.0, "A": 1e3, "D": 1e3}
                    raw_results[i] = fallback
                    packed[i] = self.pack_objectives(fallback)

        y = torch.stack(packed, dim=0)
        return y, raw_results

    def pack_objectives(self, raw: dict) -> torch.Tensor:
        out = []
        for obj in self.config.objectives:
            val = float(raw[obj])
            if obj in ("A", "D"):
                val = -val
            out.append(val)

        return torch.tensor(out, dtype=torch.double, device=self.config.device)

    def get_constraints(self):
        constraints = []

        for key, value in self.config.constraints.items():
            obj, kind = key.rsplit("_", 1)

            idx = self.objective_index(obj)
            sign = self.objective_signs[obj]

            if kind == "min":
                constraints.append(
                    lambda y, i=idx, s=sign, v=value: s * y[..., i] - v
                )
            elif kind == "max":
                constraints.append(
                    lambda y, i=idx, s=sign, v=value: v - s * y[..., i]
                )
            else:
                raise ValueError(f"Unknown constraint type '{kind}'")

        return constraints

    def objective_labels(self):
        return ["Transmission", "Asymmetry", "Divergence"]

    def format_result(self, x, y, raw):
        return (
            f"T={raw['T']:.4f}  "
            f"A={raw['A']:.4f}  "
            f"D={raw['D']:.4f}   "
            f"X={x.cpu().numpy()}"
        )

class S1GLProblem(OptProblem):
    def __init__(
        self,
        model,
        config: OptConfig,
        targets: dict,
    ):
        super().__init__(model, config)

        # Targets at end of line
        self.targets = targets
        # e.g. {"sigma_x": 3e-3, "sigma_y": 3e-3}

        self.objective_names = ["sig_x_err", "sig_y_err", "alpha_x_err", "alpha_y_err"]

    def evaluate(self, x: torch.Tensor):
        x_np = x.detach().cpu().numpy()
        n = x_np.shape[0]

        raw_results = [None] * n
        packed = [None] * n

        with ProcessPoolExecutor(max_workers=self.config.batch_size) as pool:
            futures = {
                pool.submit(self.model.run, x_np[i], self.run_id + i): i
                for i in range(n)
            }
            self.run_id += n

            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    raw = fut.result()
                except Exception as e:
                    print(f"[Worker {i}] ERROR:", e)
                    raw = self.fallback_raw()

                raw_results[i] = raw
                packed[i] = self.pack_objectives(raw)

        y = torch.stack(packed, dim=0)
        return y, raw_results

    def pack_objectives(self, raw: dict) -> torch.Tensor:
        sigma_x = raw["sigma_x"]
        sigma_y = raw["sigma_y"]
        alpha_x = raw["alpha_x"]
        alpha_y = raw["alpha_y"]

        obj_x = -abs(sigma_x - self.targets["sigma_x"])
        obj_y = -abs(sigma_y - self.targets["sigma_y"])
        obj_ax = -abs(alpha_x - self.targets["alpha_x"])
        obj_ay = -abs(alpha_y - self.targets["alpha_y"])


        return torch.tensor(
            [obj_x, obj_y, obj_ax, obj_ay],
            dtype=torch.double,
            device=self.config.device,
        )

    def get_constraints(self):
        constraints = []

        for key, value in self.config.constraints.items():
            obj, kind = key.rsplit("_", 1)

            idx = self.objective_index(obj)

            if kind == "min":
                constraints.append(
                    lambda y, i=idx, v=value: torch.abs(y[..., i]) - v
                )
            elif kind == "max":
                constraints.append(
                    lambda y, i=idx, v=value: v - torch.abs(y[..., i])
                )
            else:
                raise ValueError(f"Unknown constraint type '{kind}'")

        return constraints

    def format_result(self, x, y, raw):
        return (
            f"sig_x={raw['sigma_x']:.4e}  "
            f"sig_y={raw['sigma_y']:.4e}  "
            f"alpha_x={raw['alpha_x']:.4e}  "
            f"alpha_y={raw['alpha_y']:.4e}  "
            f"X={x.cpu().numpy()}"
        )

    def objective_labels(self):
        return [
            "sigma_x error",
            "sigma_y error",
            "alpha_x error",
            "alpha_y error",
        ]
    @staticmethod
    def fallback_raw():
        # Very bad optics
        return {
            "sig_x": 1.0,
            "sig_y": 1.0,
            "alpha_x": 1.0,
            "alpha_y": 1.0,
        }

class OpticsMatchProblem(OptProblem):
    def __init__(self, model, config: OptConfig, targets: dict, weights: dict | None = None, scales: dict | None = None):
        super().__init__(model, config)
        self.targets = targets
        self.scales = scales if scales is not None else {k: 1.0 for k in targets}
        self.loss_fn = make_coupled_metric_loss(self.targets, weights, scales)
        self.objective_names = ["loss"]

    def get_constraints(self):
        return [] # No constraints for optics matching (probably bad this is in the template)

    def get_ref_point(self) -> torch.Tensor:
        # Worst possible optics match: very large loss
        return torch.tensor([1e6], dtype=torch.double, device=self.config.device)

    def objective_labels(self):
        return ["Weighted Optics Loss"]

    def pack_objectives(self, raw: dict) -> torch.Tensor:
        loss = self.loss_fn(raw)
        return torch.tensor([loss], dtype=torch.double, device=self.config.device)

    def format_result(self, x, y, raw):
        if raw is None:
            return f"loss={y.item():.3e}  X={x.cpu().numpy()}"

        metric_str = "  ".join(
            f"{k}={raw[k]:.3e} (tgt={self.targets[k]:.3e})"
            for k in self.targets
            if k in raw
        )

        return f"loss={y.item():.3e} {metric_str} X={x.cpu().numpy()}"


    def evaluate(self, x: torch.Tensor):
        x_np = x.detach().cpu().numpy()
        n = x_np.shape[0]

        raw_results = [None] * n
        y = torch.zeros((n, 1), dtype=torch.double, device=self.config.device)

        with ProcessPoolExecutor(max_workers=self.config.batch_size) as pool:
            futures = {
                pool.submit(self.model.run, x_np[i], self.run_id + i): i
                for i in range(n)
            }
            self.run_id += n

            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    metrics = fut.result()
                    raw_results[i] = metrics

                    loss = self.loss_fn(metrics)
                    loss = min(loss, 1e6)

                    y[i, 0] = float(loss)

                except Exception as e:
                    print(f"[Worker {i}] ERROR:", e)
                    raw_results[i] = None
                    y[i, 0] = 1e6  # hard penalty

        return y, raw_results

class S1GLMatchMOBO(OptProblem):
    """
    Two-objective version of the S1GL spot-size match: rather than
    collapsing sigma_x/alpha_x error into one hand-weighted scalar loss,
    this tracks them as separate objectives and lets qEHVI map the actual
    Pareto front between them. Each error is normalised by its physical
    tolerance (from `scales`), so both objectives read as "how many
    tolerances away from target" and (0, 0) means exactly on-target in
    both simultaneously.
    """
    def __init__(self, model, config: OptConfig, targets: dict, scales: dict | None = None):
        super().__init__(model, config)
        self.targets = targets
        self.scales = scales if scales is not None else {k: 1.0 for k in targets}
        self.objective_names = ["sigma_x_err", "alpha_x_err"]

    def get_constraints(self):
        return []  # No constraints; we want the raw trade-off, unconstrained

    def objective_labels(self):
        return ["-|sigma_x error| / tol", "-|alpha_x error| / tol"]

    def pack_objectives(self, raw: dict) -> torch.Tensor:
        sigma_tol = self.scales.get("sigma_x", 1.0)
        alpha_tol = self.scales.get("alpha_x", 1.0)

        obj_sigma = -abs(raw["sigma_x"] - self.targets["sigma_x"]) / sigma_tol
        obj_alpha = -abs(raw["alpha_x"] - self.targets["alpha_x"]) / alpha_tol

        return torch.tensor([obj_sigma, obj_alpha], dtype=torch.double, device=self.config.device)

    def format_result(self, x, y, raw):
        if raw is None:
            return f"y={y.cpu().numpy()}  X={x.cpu().numpy()}"

        return (
            f"sigma_x={raw['sigma_x']:.4e} (tgt={self.targets['sigma_x']:.4e})  "
            f"alpha_x={raw['alpha_x']:.4e} (tgt={self.targets['alpha_x']:.4e})  "
            f"X={x.cpu().numpy()}"
        )

    def evaluate(self, x: torch.Tensor):
        x_np = x.detach().cpu().numpy()
        n = x_np.shape[0]

        raw_results = [None] * n
        packed = [None] * n

        with ProcessPoolExecutor(max_workers=self.config.batch_size) as pool:
            futures = {
                pool.submit(self.model.run, x_np[i], self.run_id + i): i
                for i in range(n)
            }
            self.run_id += n

            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    raw = fut.result()
                    raw_results[i] = raw
                    packed[i] = self.pack_objectives(raw)
                except Exception as e:
                    print(f"[Worker {i}] ERROR:", e)
                    raw_results[i] = None
                    fallback = {"sigma_x": 1e6, "alpha_x": 1e6}
                    packed[i] = self.pack_objectives(fallback)

        y = torch.stack(packed, dim=0)
        return y, raw_results

class S1ColProblem(OptProblem):
    def __init__(self, model, config: OptConfig, targets: dict):
        super().__init__(model, config)
        self.targets = targets
        self.objective_names = ["fraction_nominal_total"]

    def get_constraints(self):
        return [] # No constraints for optics matching (probably bad this is in the template)

    def get_ref_point(self) -> torch.Tensor:
        # Worst possible optics match: very large loss
        return torch.tensor([1e-6], dtype=torch.double, device=self.config.device)

    def objective_labels(self):
        return ["fraction_nominal_total"]

    def pack_objectives(self, raw: dict) -> torch.Tensor:
        return torch.tensor([raw], dtype=torch.double, device=self.config.device)

    def format_result(self, x, y, raw):
        metric_str = "  ".join(
            f"{k}={raw[k]:.3e} (tgt={self.targets[k]:.3e})"
            for k in self.targets
            if k in raw
        )

        return f"{metric_str} X={x.cpu().numpy()}"

    def evaluate(self, x: torch.Tensor):
        x_np = x.detach().cpu().numpy()
        n = x_np.shape[0]

        raw_results = [None] * n
        y = torch.zeros((n, 1), dtype=torch.double, device=self.config.device)

        with ProcessPoolExecutor(max_workers=self.config.batch_size) as pool:
            futures = {
                pool.submit(self.model.run, x_np[i], self.run_id + i): i
                for i in range(n)
            }
            self.run_id += n

            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    raw = fut.result()  # Should be a dict
                    raw_results[i] = raw
                    # Extract the relevant value for the objective
                    y[i, 0] = float(raw["fraction_nominal_total"])
                except Exception as e:
                    print(f"[Worker {i}] ERROR:", e)
                    raw_results[i] = None
                    y[i, 0] = 1e-6  # hard penalty

        return y, raw_results
