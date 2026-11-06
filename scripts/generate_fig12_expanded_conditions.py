"""
scripts/generate_fig12_expanded_conditions.py

Generates Figure 12: Multi-Condition Physical Stress Benchmarks & Operational Limits:
(a) Colored 1/f^alpha Noise Sweep (White -> Pink -> Brownian)
(b) 50 Hz Power-Line Harmonic Hum Contamination
(c) Heavy-Tailed Impulsive Cauchy Spikes (Failure Mode & Robust Recovery)
(d) Sensor Array Density Scaling (M = 4 to 16 Channels)
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Publication typography styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 8.5,
    'figure.titlesize': 13,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'grid.alpha': 0.35,
    'grid.linestyle': '--'
})

def generate_figure12():
    with open("results/expanded_conditions_benchmark.json", "r") as f:
        data = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Color palette
    c_hqgn = "#1b9e77"        # Emerald green
    c_hqgn_rob = "#2ca25f"    # Robust green
    c_cnn = "#386cb0"         # Steel blue
    c_gcn = "#fdc086"         # Orange/gold
    c_ctqw = "#7570b3"        # Purple
    c_oqw = "#e7298a"         # Pink/Magenta
    c_riemann = "#666666"     # Charcoal grey

    # Panel (a): Colored 1/f Noise Sweep
    ax = axes[0, 0]
    p_data = data["condition_a_colored_pink_noise"]
    alphas = p_data["alpha_levels"]
    ax.plot(alphas, p_data["hqgn"], 'o-', color=c_hqgn, label="Proposed HQGN", linewidth=2.5, markersize=7)
    ax.plot(alphas, p_data["dissipative_oqw"], 's--', color=c_oqw, label="Dissipative OQW [Linck 2026]")
    ax.plot(alphas, p_data["dense_1d_cnn"], '^-', color=c_cnn, label="Dense 1D-CNN [Schirrmeister 2017]")
    ax.plot(alphas, p_data["dynamic_st_gcn"], 'd:', color=c_gcn, label="Dynamic ST-GCN [Hajisafi 2024]")
    ax.plot(alphas, p_data["unitary_ctqw"], 'x-.', color=c_ctqw, label="Unitary CTQW-GNN [Zhan 2026]")
    ax.set_title("(a) Colored $1/f^\\alpha$ Noise Sweep (White $\\to$ Pink $\\to$ Brownian)", fontweight="bold")
    ax.set_xlabel(r"Spectral Slope $\alpha$ (0=White, 1=Pink, 2=Brownian)")
    ax.set_ylabel("Macro-F1 Score (%)")
    ax.set_ylim(35, 95)
    ax.grid(True)
    ax.legend(loc="lower left", framealpha=0.9)

    # Panel (b): Harmonic Hum Sweep
    ax = axes[0, 1]
    h_data = data["condition_b_harmonic_hum"]
    hums = h_data["hum_amplitudes"]
    ax.plot(hums, h_data["hqgn"], 'o-', color=c_hqgn, label="Proposed HQGN", linewidth=2.5, markersize=7)
    ax.plot(hums, h_data["dissipative_oqw"], 's--', color=c_oqw, label="Dissipative OQW [Linck 2026]")
    ax.plot(hums, h_data["dense_1d_cnn"], '^-', color=c_cnn, label="Dense 1D-CNN [Schirrmeister 2017]")
    ax.plot(hums, h_data["dynamic_st_gcn"], 'd:', color=c_gcn, label="Dynamic ST-GCN [Hajisafi 2024]")
    ax.plot(hums, h_data["unitary_ctqw"], 'x-.', color=c_ctqw, label="Unitary CTQW-GNN [Zhan 2026]")
    ax.set_title("(b) 50 Hz Power-Line Harmonic Hum Contamination", fontweight="bold")
    ax.set_xlabel("Hum Amplitude Relative to Signal Std Dev")
    ax.set_ylabel("Macro-F1 Score (%)")
    ax.set_ylim(40, 95)
    ax.grid(True)
    ax.legend(loc="lower left", framealpha=0.9)

    # Panel (c): Heavy-Tailed Cauchy Impulsive Spikes (Failure Mode Analysis)
    ax = axes[1, 0]
    c_data = data["condition_c_cauchy_impulsive_noise"]
    probs = [p * 100 for p in c_data["contamination_probability"]]
    ax.plot(probs, c_data["hqgn_robust_clipped"], 'o-', color=c_hqgn, label="HQGN + Median Clipping", linewidth=2.5, markersize=7)
    ax.plot(probs, c_data["hqgn_raw"], 'o--', color="#d95f02", label="HQGN (Unclipped, Failure Mode)", linewidth=2.0)
    ax.plot(probs, c_data["dissipative_oqw"], 's--', color=c_oqw, label="Dissipative OQW")
    ax.plot(probs, c_data["dense_1d_cnn"], '^-', color=c_cnn, label="Dense 1D-CNN")
    ax.plot(probs, c_data["dynamic_st_gcn"], 'd:', color=c_gcn, label="Dynamic ST-GCN")
    ax.set_title("(c) Impulsive Cauchy Spikes & Robust Clipping Recourse", fontweight="bold")
    ax.set_xlabel("Contamination Probability per Timepoint (%)")
    ax.set_ylabel("Macro-F1 Score (%)")
    ax.set_ylim(30, 95)
    ax.grid(True)
    ax.legend(loc="lower left", framealpha=0.9)

    # Panel (d): Sensor Array Density Scaling
    ax = axes[1, 1]
    s_data = data["condition_d_channel_density_scaling"]
    chans = s_data["channel_counts"]
    ax.plot(chans, s_data["hqgn"], 'o-', color=c_hqgn, label="Proposed HQGN", linewidth=2.5, markersize=7)
    ax.plot(chans, s_data["dense_1d_cnn"], '^-', color=c_cnn, label="Dense 1D-CNN")
    ax.plot(chans, s_data["dynamic_st_gcn"], 'd:', color=c_gcn, label="Dynamic ST-GCN")
    ax.plot(chans, s_data["dissipative_oqw"], 's--', color=c_oqw, label="Dissipative OQW")
    ax.plot(chans, s_data["unitary_ctqw"], 'x-.', color=c_ctqw, label="Unitary CTQW")
    ax.set_title("(d) Sensor Array Density Scaling ($M = 4$ to $16$ Channels)", fontweight="bold")
    ax.set_xlabel("Number of Active Sensor Channels ($M$)")
    ax.set_ylabel("Macro-F1 Score (%)")
    ax.set_ylim(65, 95)
    ax.grid(True)
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()

    out_paths = [
        "figures/fig12_expanded_stress_conditions.png",
        "figures/fig12_expanded_stress_conditions.pdf",
        "Updated_significant/figures/fig12_expanded_stress_conditions.png",
        "Updated_significant/figures/fig12_expanded_stress_conditions.pdf"
    ]
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        plt.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Generated Figure 12 at: {p}")

    plt.close()

if __name__ == "__main__":
    generate_figure12()
