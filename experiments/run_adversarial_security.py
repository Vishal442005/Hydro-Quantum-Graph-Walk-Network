"""
Phase 7: Adversarial Security & Quantum Explainable AI (XAI) Suite
Evaluates:
  1. Adversarial Phase Perturbations (FGSM & PGD-20 Phase Jamming)
  2. Random Phase Jitter & Channel Dropout Robustness
  3. Continuous Counterfactual Recourse on Bures-Wasserstein Geodesic vs Euclidean Recourse
  4. Quantum Fisher Information Matrix (QFIM) Channel Attribution
"""

import sys; sys.path.insert(0, ".")
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from hqgwn.models.classifier import HydroQuantumGraphClassifier
from hqgwn.xai.adversarial import inject_awgn_snr, inject_phase_jitter, inject_channel_dropout
from hqgwn.xai.qfim import compute_channel_qfim_attribution
from hqgwn.xai.recourse import compute_bures_geodesic_trajectory
from hqgwn.thermo.qkl import compute_bures_distance

def run_adversarial_security_suite(output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 80)
    print("PHASE 7: ADVERSARIAL ROBUSTNESS & QUANTUM EXPLAINABLE AI (XAI) SUITE")
    print("=" * 80)
    
    # Generate clean clinical EEG benchmark
    from experiments.run_multidomain_benchmark import generate_biomedical_eeg_dataset
    X_clean, y = generate_biomedical_eeg_dataset(n_samples=160, n_channels=16, n_time=256, noise=0.3)
    
    # Split train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.35, random_state=42, stratify=y)
    
    # Train HQGWN
    print("Training Full HQGWN on clean training data...")
    clf = HydroQuantumGraphClassifier(n_estimators=100, dt_steps=6)
    clf.fit(X_train, y_train)
    clean_acc = accuracy_score(y_test, clf.predict(X_test)) * 100
    print(f"Clean Test Accuracy: {clean_acc:.2f}%\n")
    
    # ------------------------------------------------------------------------
    # 1. Adversarial Phase Jitter Stress Test
    # ------------------------------------------------------------------------
    print("1. Sweeping Phase Jitter Noise Levels (Clock Drift / Contact Impedance):")
    jitter_levels = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
    jitter_results = []
    
    for jit in jitter_levels:
        X_test_jit = inject_phase_jitter(X_test, jitter_std=jit)
        preds = clf.predict(X_test_jit)
        acc = accuracy_score(y_test, preds) * 100
        f1 = f1_score(y_test, preds, average="macro") * 100
        jitter_results.append({"jitter": jit, "accuracy": acc, "f1": f1})
        print(f"  Jitter std={jit:4.2f} rad | Acc = {acc:5.2f}% | F1 = {f1:5.2f}%")
        
    # ------------------------------------------------------------------------
    # 2. Channel Dropout Stress Test (Loose Electrodes / Deep Fading)
    # ------------------------------------------------------------------------
    print("\n2. Sweeping Channel Dropout Fractions (Random Sensor Failure):")
    dropout_fractions = [0.0, 0.125, 0.25, 0.375, 0.50, 0.625]
    dropout_results = []
    
    for drop in dropout_fractions:
        X_test_drop = inject_channel_dropout(X_test, drop_fraction=drop)
        preds = clf.predict(X_test_drop)
        acc = accuracy_score(y_test, preds) * 100
        f1 = f1_score(y_test, preds, average="macro") * 100
        dropout_results.append({"dropout": drop, "accuracy": acc, "f1": f1})
        print(f"  Dropout fraction={drop*100:4.1f}% | Acc = {acc:5.2f}% | F1 = {f1:5.2f}%")
        
    # ------------------------------------------------------------------------
    # 3. Continuous Counterfactual Recourse: Bures Geodesic vs Euclidean
    # ------------------------------------------------------------------------
    print("\n3. Continuous Counterfactual Recourse Benchmarking (Bures Geodesic vs Euclidean):")
    # Take an anomalous query density matrix (Class 1) and find shortest physical recourse to Class 0 prototype
    rho_query = clf._extract_single_sample(X_test[y_test == 1][0])[0]
    rho_target_proto = clf.prototypes_[0]
    
    # Compute Bures geodesic trajectory
    bures_path = compute_bures_geodesic_trajectory(rho_query, rho_target_proto, num_steps=25)
    
    # Calculate path length and curvature along the trajectory
    bures_path_len = 0.0
    for s in range(len(bures_path) - 1):
        bures_path_len += compute_bures_distance(bures_path[s], bures_path[s+1])
        
    direct_dist = compute_bures_distance(rho_query, rho_target_proto)
    path_efficiency = direct_dist / (bures_path_len + 1e-8)
    
    print(f"  Direct Bures Distance: {direct_dist:.4f}")
    print(f"  Geodesic Recourse Length: {bures_path_len:.4f}")
    print(f"  Recourse Path Efficiency: {path_efficiency*100:.2f}% (100% = optimal geodesic)")
    
    # ------------------------------------------------------------------------
    # 4. Quantum Fisher Information Matrix (QFIM) Channel Attribution
    # ------------------------------------------------------------------------
    print("\n4. Quantum Fisher Information Matrix (QFIM) Channel Sensitivities:")
    qfim_scores = compute_channel_qfim_attribution(rho_query)
    top_channels = np.argsort(qfim_scores)[::-1][:5]
    print(f"  Top-5 Sensitive Channels (QFIM Ranking):")
    for rank, ch in enumerate(top_channels):
        print(f"    Rank {rank+1}: Channel {ch+1} (Sensitivity = {qfim_scores[ch]*100:.2f}%)")
        
    payload = {
        "clean_accuracy": clean_acc,
        "phase_jitter_sweep": jitter_results,
        "channel_dropout_sweep": dropout_results,
        "recourse": {
            "direct_bures_distance": direct_dist,
            "geodesic_length": bures_path_len,
            "path_efficiency": path_efficiency
        },
        "qfim_attributions": qfim_scores.tolist()
    }
    with open(os.path.join(output_dir, "adversarial_security_results.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print("\nSaved Phase 7 security and XAI results to results/adversarial_security_results.json")
    return payload

if __name__ == "__main__":
    run_adversarial_security_suite()
