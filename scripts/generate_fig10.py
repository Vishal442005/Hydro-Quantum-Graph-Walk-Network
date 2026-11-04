import matplotlib.pyplot as plt
import numpy as np
import json
import os

with open("results/comprehensive_comparative_study.json", "r", encoding="utf-8") as f:
    data = json.load(f)

dirichlet = data["dirichlet_energy"]
efficiency = data["sample_efficiency"]
walk_time = data["walk_time_sensitivity"]

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)

# Subplot 1: Dirichlet Energy Oversmoothing
ax1 = axes[0]
ax1.plot(dirichlet["times"], dirichlet["classical"], "r--", linewidth=2.0, label="Classical Markov Diffusion (Oversmooths)")
ax1.plot(dirichlet["times"], dirichlet["unitary_ctqw"], "b-.", linewidth=2.0, label="Unitary CTQW (Zhan 2026, alpha=0)")
ax1.plot(dirichlet["times"], dirichlet["hqgwn"], "g-", linewidth=2.5, label="HQGWN Open System (alpha=0.15)")
ax1.set_xlabel("Evolution Horizon t", fontsize=11, fontweight="bold")
ax1.set_ylabel("Dirichlet Energy E(t)", fontsize=11, fontweight="bold")
ax1.set_title("(a) Dirichlet Energy & Oversmoothing Invariance", fontsize=12, fontweight="bold")
ax1.legend(loc="upper right", frameon=True, fontsize=9)
ax1.grid(True, linestyle=":", alpha=0.6)

# Subplot 2: Sample Efficiency Curves
ax2 = axes[1]
ratios = [int(r * 100) for r in efficiency["ratios"]]
curves = efficiency["curves"]
for name, vals in curves.items():
    style = "-" if "HQGWN" in name else "--"
    lw = 2.5 if "HQGWN" in name else 1.8
    marker = "o" if "HQGWN" in name else "s"
    ax2.plot(ratios, vals, style, marker=marker, linewidth=lw, label=name)
ax2.set_xlabel("Training Data Ratio (%)", fontsize=11, fontweight="bold")
ax2.set_ylabel("Macro-F1 Score (%)", fontsize=11, fontweight="bold")
ax2.set_title("(b) Sample-Efficiency Generalization", fontsize=12, fontweight="bold")
ax2.legend(loc="lower right", frameon=True, fontsize=8.5)
ax2.set_ylim([80, 103])
ax2.grid(True, linestyle=":", alpha=0.6)

# Subplot 3: Walk-Time Horizon Sensitivity
ax3 = axes[2]
ax3.plot(walk_time["tau_vals"], walk_time["macro_f1"], "m-o", linewidth=2.2, markersize=6, color="#2b5c8f")
ax3.axvspan(1.0, 3.0, color="green", alpha=0.15, label="Optimal Walk Window tau in [1.0, 3.0]")
ax3.set_xlabel("Continuous Walk Horizon tau (s)", fontsize=11, fontweight="bold")
ax3.set_ylabel("Macro-F1 Score (%)", fontsize=11, fontweight="bold")
ax3.set_title("(c) Continuous Walk-Time Sensitivity", fontsize=12, fontweight="bold")
ax3.legend(loc="lower right", frameon=True, fontsize=9)
ax3.set_ylim([90, 103])
ax3.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/fig10_dirichlet_oversmoothing.pdf", bbox_inches="tight")
plt.savefig("figures/fig10_dirichlet_oversmoothing.png", bbox_inches="tight", dpi=300)
plt.close()
print("Successfully generated figures/fig10_dirichlet_oversmoothing.png & .pdf")
