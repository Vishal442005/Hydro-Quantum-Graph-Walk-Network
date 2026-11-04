"""
Phase 7: ECE Telemetry Noise Robustness Curve (SNR Degradation Sweep)
Sweeps SNR from +20 dB down to -10 dB and compares HQGWN against:
  1. Full HQGWN (Lindblad alpha = 0.15)
  2. Unitary QW (alpha = 0, no dissipation)
  3. Power Spectral Density (Welch PSD) + RBF-SVM
  4. Riemannian Covariance Tangent Space
  5. Classical Markov Random Walk
"""

import numpy as np
import json
import os
from scipy.signal import welch
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from hqgwn.models.classifier import HydroQuantumGraphClassifier
from hqgwn.xai.adversarial import inject_awgn_snr
from experiments.run_synthetic_kuramoto import generate_kuramoto_vortex_dataset

class WelchPsdSvmBaseline:
    """Standard biomedical/RF frequency-domain baseline."""
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
    """Covariance geometry tangent space projection baseline."""
    def __init__(self):
        self.clf = LogisticRegression(max_iter=500)
        
    def _extract(self, X):
        N, M, L = X.shape
        feats = []
        for i in range(N):
            cov = np.cov(X[i]) + 1e-6 * np.eye(M)
            # Log-Euclidean tangent space vectorization
            eigvals, eigvecs = np.linalg.eigh(cov)
            log_cov = eigvecs @ np.diag(np.log(np.maximum(eigvals, 1e-8))) @ eigvecs.T
            # Upper triangular elements
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

def run_snr_stress_benchmark(output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 75)
    print("PHASE 7: ECE NOISE STRESS BENCHMARK (SNR SWEEP: +20 dB to -10 dB)")
    print("=" * 75)
    
    X_clean, y = generate_kuramoto_vortex_dataset(n_samples=240, n_channels=12, n_time=128, noise_level=0.2)
    X_train_clean, X_test_clean, y_train, y_test = train_test_split(X_clean, y, test_size=0.35, random_state=42, stratify=y)
    
    models = {
        "HQGWN_Full": HydroQuantumGraphClassifier(use_hydrodynamic=True, use_quantum_walk=True, use_quantum_entropy=True, alpha_dephase=0.15, dt_steps=8),
        "Unitary_QW_alpha0": HydroQuantumGraphClassifier(use_hydrodynamic=True, use_quantum_walk=True, use_quantum_entropy=True, alpha_dephase=0.0, dt_steps=8),
        "Classical_Markov_RW": HydroQuantumGraphClassifier(use_hydrodynamic=False, use_quantum_walk=False, use_quantum_entropy=False, dt_steps=8),
        "Welch_PSD_SVM": WelchPsdSvmBaseline(),
        "Riemannian_Cov": RiemannianCovarianceBaseline(),
    }
    
    print("Fitting models on clean baseline training data...")
    for name, model in models.items():
        model.fit(X_train_clean, y_train)
        
    snr_levels = [20.0, 15.0, 10.0, 5.0, 0.0, -5.0, -10.0]
    snr_results = {name: [] for name in models}
    
    print("\nSweeping SNR levels on corrupted test data:")
    for snr in snr_levels:
        X_test_noisy = inject_awgn_snr(X_test_clean, snr_db=snr)
        print(f"\n--- Testing at SNR = {snr:+5.1f} dB ---")
        for name, model in models.items():
            preds = model.predict(X_test_noisy)
            f1 = f1_score(y_test, preds, average="macro") * 100
            snr_results[name].append(f1)
            print(f"  {name:<22}: Macro-F1 = {f1:5.2f}%")
            
    payload = {
        "snr_levels": snr_levels,
        "results": snr_results
    }
    with open(os.path.join(output_dir, "snr_stress_results.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print("\nSaved SNR stress results to results/snr_stress_results.json")
    return payload

if __name__ == "__main__":
    run_snr_stress_benchmark()
