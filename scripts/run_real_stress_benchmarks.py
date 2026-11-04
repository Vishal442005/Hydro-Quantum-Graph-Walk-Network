import sys
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from scipy.signal import welch
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, ".")
from experiments.run_real_multidomain_benchmarks import load_domain1_epilepsy_ecog
from hqgwn.models.classifier import HydroQuantumGraphClassifier
from hqgwn.xai.adversarial import inject_awgn_snr, inject_phase_jitter, inject_channel_dropout
from hqgwn.xai.qfim import compute_channel_qfim_attribution
from hqgwn.xai.recourse import compute_bures_geodesic_trajectory
from hqgwn.thermo.qkl import compute_bures_distance

class WelchPsdSvmBaseline:
    def __init__(self):
        self.clf = SVC(kernel="rbf", C=1.0)
    def fit(self, X, y):
        feats = self._extract(X)
        self.clf.fit(feats, y)
        return self
    def _extract(self, X):
        N, M, L = X.shape
        feats = []
        for i in range(N):
            _, psd = welch(X[i], nperseg=min(64, L), axis=-1)
            feats.append(psd.reshape(-1))
        return np.array(feats)
    def predict(self, X):
        feats = self._extract(X)
        return self.clf.predict(feats)

class RiemannianCovarianceBaseline:
    def __init__(self):
        self.clf = LogisticRegression(max_iter=500)
    def _extract(self, X):
        N, M, L = X.shape
        feats = []
        for i in range(N):
            cov = np.cov(X[i]) + 1e-6 * np.eye(M)
            eigvals, eigvecs = np.linalg.eigh(cov)
            log_cov = eigvecs @ np.diag(np.log(np.maximum(eigvals, 1e-8))) @ eigvecs.T
            triu_idx = np.triu_indices(M)
            feats.append(log_cov[triu_idx])
        return np.array(feats)
    def fit(self, X, y):
        feats = self._extract(X)
        self.clf.fit(feats, y)
        return self
    def predict(self, X):
        feats = self._extract(X)
        return self.clf.predict(feats)

def run_real_benchmarks():
    print("Loading Real Clinical Intracranial ECoG Telemetry...")
    X_raw, y = load_domain1_epilepsy_ecog()
    # Downsample time dimension to 125 points for efficiency
    X = X_raw[:, :, ::3]
    print(f"Loaded: X.shape = {X.shape}, y.shape = {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42, stratify=y)

    # 1. Real SNR Stress Test
    print("\n--- Running Real SNR Stress Sweep on Clinical ECoG ---")
    models = {
        "HQGWN_Full": HydroQuantumGraphClassifier(use_hydrodynamic=True, use_quantum_walk=True, use_quantum_entropy=True, alpha_dephase=0.15, dt_steps=6),
        "Unitary_QW_alpha0": HydroQuantumGraphClassifier(use_hydrodynamic=True, use_quantum_walk=True, use_quantum_entropy=True, alpha_dephase=0.0, dt_steps=6),
        "Classical_Markov_RW": HydroQuantumGraphClassifier(use_hydrodynamic=False, use_quantum_walk=False, use_quantum_entropy=False, dt_steps=6),
        "Welch_PSD_SVM": WelchPsdSvmBaseline(),
        "Riemannian_Cov": RiemannianCovarianceBaseline(),
    }

    for name, m in models.items():
        m.fit(X_train, y_train)

    snr_levels = [20.0, 15.0, 10.0, 5.0, 0.0, -5.0, -10.0]
    snr_results = {name: [] for name in models}

    for snr in snr_levels:
        X_test_noisy = inject_awgn_snr(X_test, snr_db=snr)
        for name, m in models.items():
            preds = m.predict(X_test_noisy)
            f1 = f1_score(y_test, preds, average="macro") * 100
            snr_results[name].append(round(f1, 2))
        print(f"SNR {snr:+5.1f} dB -> HQGWN: {snr_results['HQGWN_Full'][-1]}%, Unitary: {snr_results['Unitary_QW_alpha0'][-1]}%, Markov: {snr_results['Classical_Markov_RW'][-1]}%")

    with open("results/snr_stress_results.json", "w") as f:
        json.dump({"snr_levels": snr_levels, "results": snr_results}, f, indent=2)
    print("Saved authentic results to results/snr_stress_results.json")

    # 2. Real Adversarial Security Suite
    print("\n--- Running Real Adversarial Security Sweep on Clinical ECoG ---")
    clf = models["HQGWN_Full"]
    clean_acc = accuracy_score(y_test, clf.predict(X_test)) * 100

    jitter_levels = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
    jitter_results = []
    for jit in jitter_levels:
        X_test_jit = inject_phase_jitter(X_test, jitter_std=jit)
        preds = clf.predict(X_test_jit)
        acc = accuracy_score(y_test, preds) * 100
        f1 = f1_score(y_test, preds, average="macro") * 100
        jitter_results.append({"jitter": jit, "accuracy": round(acc, 2), "f1": round(f1, 2)})

    dropout_fractions = [0.0, 0.125, 0.25, 0.375, 0.50, 0.625]
    dropout_results = []
    for drop in dropout_fractions:
        X_test_drop = inject_channel_dropout(X_test, drop_fraction=drop)
        preds = clf.predict(X_test_drop)
        acc = accuracy_score(y_test, preds) * 100
        f1 = f1_score(y_test, preds, average="macro") * 100
        dropout_results.append({"dropout": drop, "accuracy": round(acc, 2), "f1": round(f1, 2)})

    # Recourse and QFIM
    rho_query = clf._extract_single_sample(X_test[y_test == 1][0])[0]
    rho_target_proto = clf.prototypes_[0]
    bures_path = compute_bures_geodesic_trajectory(rho_query, rho_target_proto, num_steps=25)
    bures_path_len = sum(compute_bures_distance(bures_path[s], bures_path[s+1]) for s in range(len(bures_path)-1))
    direct_dist = compute_bures_distance(rho_query, rho_target_proto)
    path_efficiency = direct_dist / (bures_path_len + 1e-8)

    qfim_scores = compute_channel_qfim_attribution(rho_query)

    adv_payload = {
        "clean_accuracy": round(clean_acc, 2),
        "phase_jitter_sweep": jitter_results,
        "channel_dropout_sweep": dropout_results,
        "recourse": {
            "direct_bures_distance": round(direct_dist, 4),
            "geodesic_length": round(bures_path_len, 4),
            "path_efficiency": round(path_efficiency, 4)
        },
        "qfim_attributions": [round(float(q), 4) for q in qfim_scores]
    }
    with open("results/adversarial_security_results.json", "w") as f:
        json.dump(adv_payload, f, indent=2)
    print("Saved authentic results to results/adversarial_security_results.json")

if __name__ == '__main__':
    run_real_benchmarks()
