import re
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from botorch.utils.multi_objective.pareto import is_non_dominated

# --- Config ---
logfile = "90-BDSIMData/MOBO_test_32/MOBO_test_32_optlog.txt"  # path to your log file

# --- Parse log file ---
pattern = re.compile(
    r"T=(?P<T>[0-9.eE+-]+)\s+A=(?P<A>[0-9.eE+-]+)\s+D=(?P<D>[0-9.eE+-]+)\s+X=\[(?P<X>[0-9.\s]+)\]"
)

records = []

with open(logfile, "r") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            T = float(m.group("T"))
            A = float(m.group("A"))
            D = float(m.group("D"))
            X = [float(v) for v in m.group("X").split()]
            records.append([T, A, D] + X)

# Build DataFrame
columns = ["T", "A", "D", "dl1", "dl2", "dl3", "ql1", "ql2", "ql3"]
df = pd.DataFrame(records, columns=columns)

# --- Convert to tensor for Pareto check ---
# Objectives: maximize T, minimize A and D
T_tensor = torch.tensor(df["T"].values, dtype=torch.float64).unsqueeze(1)
A_tensor = torch.tensor(-df["A"].values, dtype=torch.float64).unsqueeze(1)  # invert to maximize
D_tensor = torch.tensor(-df["D"].values, dtype=torch.float64).unsqueeze(1)  # invert to maximize
Y = torch.cat([T_tensor, A_tensor, D_tensor], dim=1)

mask = is_non_dominated(Y)
df["pareto"] = mask.numpy()

# --- 3D Plot ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Non-Pareto points
ax.scatter(df.loc[~df["pareto"], "T"], df.loc[~df["pareto"], "A"], df.loc[~df["pareto"], "D"],
           c='gray', alpha=0.5, label='Non-Pareto')

# Pareto points
ax.scatter(df.loc[df["pareto"], "T"], df.loc[df["pareto"], "A"], df.loc[df["pareto"], "D"],
           c='red', s=80, label='Pareto')

ax.set_xlabel("Transmission T")
ax.set_ylabel("Asymmetry A")
ax.set_zlabel("Divergence D")
ax.set_title("MOBO Pareto Front in 3D")
ax.legend()
plt.tight_layout()
plt.show()
