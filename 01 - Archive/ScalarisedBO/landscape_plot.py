import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("Triplet_NoNozzle_DL0.5_n200_W111/triplet_results.csv")

T = df["T"]
A = df["A"]
D = df["div"]

# Identify best scalar solution (minimum loss)
best_idx = df["loss"].idxmin()

plt.figure(figsize=(6, 4))
plt.scatter(T, D, s=20, alpha=0.6, label="BO evaluations")


plt.xlabel("Transmission")
plt.ylabel("Divergence")
plt.legend()
plt.tight_layout()
plt.show()
