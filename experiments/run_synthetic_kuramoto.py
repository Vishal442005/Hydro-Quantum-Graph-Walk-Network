"""
Phase 6: Non-Linear Kuramoto Physics Benchmark (Travelling Vortex vs Turbulent Shear)
Runs the full 6-variant ablation suite (A1-A6) with 8-fold cross-validation and
executes the complete statistical validation suite:
  1. Omnibus Friedman test
  2. Pairwise Wilcoxon signed-rank test with Holm-Bonferroni correction
  3. McNemar contingency matrix test
  4. Cliff's Delta non-parametric effect size
"""

import sys; sys.path.insert(0, ".")
import numpy as np
import json
import os
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from hqgwn.models.classifier import HydroQuantumGraphClassifier

def generate_kuramoto_vortex_dataset(
    n_samples: int = 240, 
    n_channels: int = 12, 
    n_time: int = 128, 
    noise_level: float = 0.8,
    random_state: int = 42
):
    """
    Simulates multi-channel coupled non-linear phase oscillators (Kuramoto lattice):
      Class 0: Coherent travelling phase vortex (uniform phase velocity, spatial wave delay).
      Class 1: Turbulent phase vortex shear (spatial velocity gradients, turbulent shear flow).
    Both classes have identical mean power, challenging models on phase dynamics alone.
    """
    np.random.seed(random_state)
    t = np.linspace(0, 4 * np.pi, n_time)
    X = np.zeros((n_samples, n_channels, n_time))
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        label = i % 2
        y[i] = label
        
        if label == 0:
            # Coherent vortex: constant base frequency with travelling spatial phase wave
            base_freq = 3.2
            for m in range(n_channels):
                phase_m = m * (2.0 * np.pi / n_channels)
                vortex_mod = 0.2 * np.sin(0.5 * t)
                signal = np.sin(base_freq * t + phase_m + vortex_mod)
                noise = np.random.normal(0, noise_level, size=n_time)
                X[i, m, :] = signal + noise
        else:
            # Turbulent shear: spatial velocity gradient (shear flow) and phase slips
            for m in range(n_channels):
                local_freq = 3.2 + 0.35 * np.sin(m * 1.5)
                phase_m = np.random.uniform(0, 2.0 * np.pi)
                vortex_mod = 0.3 * np.cos(1.8 * t) + 0.15 * np.sin(3.0 * t)
                signal = np.sin(local_freq * t + phase_m + vortex_mod)
                noise = np.random.normal(0, noise_level, size=n_time)
                X[i, m, :] = signal + noise
            
    return X, y

def compute_cliffs_delta(x, y):
    """Computes Cliff's Delta non-parametric effect size."""
    n1, n2 = len(x), len(y)
    more = sum(xi > yj for xi in x for yj in y)
    less = sum(xi < yj for xi in x for yj in y)
    return (more - less) / (n1 * n2)

def compute_mcnemar_stat(y_true, preds_a, preds_b):
    """Computes McNemar chi2 statistic between two model predictions."""
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)
    n01 = np.sum(~correct_a & correct_b) # b correct, a wrong
    n10 = np.sum(correct_a & ~correct_b) # a correct, b wrong
    if (n01 + n10) == 0:
        return 0.0, 1.0
    chi2 = ((abs(n01 - n10) - 1.0) ** 2) / (n01 + n10)
    from scipy.stats import chi2 as chi2_dist
    p_val = 1.0 - chi2_dist.cdf(chi2, df=1)
    return float(chi2), float(p_val)

def run_kuramoto_ablation_benchmark(output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 75)
    print("PHASE 6: UPGRADED KURAMOTO VORTEX ABLATION SUITE (A1-A6)")
    print("=" * 75)
    
    X, y = generate_kuramoto_vortex_dataset(n_samples=240, n_channels=12, n_time=128, noise_level=0.8)
    print(f"Generated Synthetic Dataset: {X.shape[0]} samples, {X.shape[1]} channels, {X.shape[2]} timepoints.")
    
    configs = {
        "A1_Classical_Base": HydroQuantumGraphClassifier(
            use_hydrodynamic=False, use_quantum_walk=False, use_quantum_entropy=False, dt_steps=8
        ),
        "A2_Hydro_Only": HydroQuantumGraphClassifier(
            use_hydrodynamic=True, use_quantum_walk=False, use_quantum_entropy=False, dt_steps=8
        ),
        "A3_Unitary_QW": HydroQuantumGraphClassifier(
            use_hydrodynamic=True, use_quantum_walk=True, use_quantum_entropy=True, alpha_dephase=0.0, dt_steps=8
        ),
        "A4_Quantum_Only": HydroQuantumGraphClassifier(
            use_hydrodynamic=False, use_quantum_walk=True, use_quantum_entropy=True, alpha_dephase=0.15, dt_steps=8
        ),
        "A5_No_Entropy": HydroQuantumGraphClassifier(
            use_hydrodynamic=True, use_quantum_walk=True, use_quantum_entropy=False, alpha_dephase=0.15, dt_steps=8
        ),
        "A6_Full_HQGWN": HydroQuantumGraphClassifier(
            use_hydrodynamic=True, use_quantum_walk=True, use_quantum_entropy=True, alpha_dephase=0.15, dt_steps=8
        ),
    }
    
    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
    fold_f1 = {name: [] for name in configs}
    fold_acc = {name: [] for name in configs}
    fold_kappa = {name: [] for name in configs}
    
    all_preds = {name: [] for name in configs}
    all_targets = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        all_targets.extend(y_test)
        
        for name, model in configs.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            f1 = f1_score(y_test, preds, average="macro")
            acc = accuracy_score(y_test, preds)
            kappa = cohen_kappa_score(y_test, preds)
            
            fold_f1[name].append(f1)
            fold_acc[name].append(acc)
            fold_kappa[name].append(kappa)
            all_preds[name].extend(preds)
            
        print(f"  Fold {fold+1}/8 completed. A6 F1: {fold_f1['A6_Full_HQGWN'][-1]*100:.2f}% | A1 F1: {fold_f1['A1_Classical_Base'][-1]*100:.2f}%")
        
    all_targets = np.array(all_targets)
    for name in all_preds:
        all_preds[name] = np.array(all_preds[name])
        
    print("\n" + "=" * 75)
    print("UPGRADED STATISTICAL BENCHMARK SUMMARY (8-Fold Cross-Validation)")
    print("=" * 75)
    print(f"{'Ablation Variant':<22} | {'Macro F1 (%)':<16} | {'Accuracy (%)':<16} | {'Cohen Kappa':<12}")
    print("-" * 75)
    
    summary = {}
    for name in configs:
        f1_m, f1_s = np.mean(fold_f1[name]) * 100, np.std(fold_f1[name]) * 100
        acc_m, acc_s = np.mean(fold_acc[name]) * 100, np.std(fold_acc[name]) * 100
        kap_m, kap_s = np.mean(fold_kappa[name]), np.std(fold_kappa[name])
        print(f"{name:<22} | {f1_m:5.2f} ± {f1_s:4.2f}%   | {acc_m:5.2f} ± {acc_s:4.2f}%   | {kap_m:5.3f} ± {kap_s:4.3f}")
        summary[name] = {
            "f1_mean": f1_m, "f1_std": f1_s,
            "acc_mean": acc_m, "acc_std": acc_s,
            "kappa_mean": kap_m, "kappa_std": kap_s,
            "fold_f1": fold_f1[name]
        }
        
    # Omnibus Friedman non-parametric test
    f_stat, p_friedman = friedmanchisquare(*[fold_f1[k] for k in configs])
    print("\n" + "-" * 75)
    print(f"Omnibus Friedman Test: Chi2 = {f_stat:.4f}, p-value = {p_friedman:.5e}")
    
    # Pairwise Wilcoxon Signed-Rank Test with Holm-Bonferroni correction
    full_f1 = np.array(fold_f1["A6_Full_HQGWN"])
    competitors = [k for k in configs if k != "A6_Full_HQGWN"]
    
    raw_pvals = []
    deltas = []
    mcnemars = []
    
    for comp in competitors:
        comp_f1 = np.array(fold_f1[comp])
        try:
            w_stat, p_w = wilcoxon(full_f1, comp_f1, alternative="greater")
        except Exception:
            p_w = 1.0
        delta = compute_cliffs_delta(full_f1, comp_f1)
        chi2_mc, p_mc = compute_mcnemar_stat(all_targets, all_preds["A6_Full_HQGWN"], all_preds[comp])
        raw_pvals.append(p_w)
        deltas.append(delta)
        mcnemars.append((chi2_mc, p_mc))
        
    # Holm-Bonferroni correction
    sorted_indices = np.argsort(raw_pvals)
    m = len(competitors)
    print("\nPairwise Statistical Significance (vs A6_Full_HQGWN):")
    print(f"{'Competitor':<20} | {'Raw p-val':<12} | {'Holm alpha':<12} | {'Cliff Delta':<12} | {'McNemar p':<12} | {'Significant?'}")
    print("-" * 85)
    
    stat_results = {}
    for rank, idx in enumerate(sorted_indices):
        comp = competitors[idx]
        p_raw = raw_pvals[idx]
        alpha_holm = 0.05 / (m - rank)
        sig = "YES (p < alpha)" if p_raw < alpha_holm else "NO"
        delta = deltas[idx]
        chi2_mc, p_mc = mcnemars[idx]
        print(f"{comp:<20} | {p_raw:10.5f}   | {alpha_holm:10.5f}   | {delta:10.3f}   | {p_mc:10.5f}   | {sig}")
        stat_results[comp] = {
            "p_wilcoxon": p_raw,
            "alpha_holm": alpha_holm,
            "cliffs_delta": delta,
            "mcnemar_chi2": chi2_mc,
            "mcnemar_p": p_mc,
            "significant": bool(p_raw < alpha_holm)
        }
        
    out_payload = {
        "summary": summary,
        "friedman": {"stat": float(f_stat) if not np.isnan(f_stat) else 0.0, "p_value": float(p_friedman) if not np.isnan(p_friedman) else 1.0},
        "pairwise": stat_results
    }
    with open(os.path.join(output_dir, "kuramoto_ablation_results.json"), "w") as f:
        json.dump(out_payload, f, indent=2)
    print("\nSaved benchmark results to results/kuramoto_ablation_results.json")
    return out_payload

if __name__ == "__main__":
    run_kuramoto_ablation_benchmark()
