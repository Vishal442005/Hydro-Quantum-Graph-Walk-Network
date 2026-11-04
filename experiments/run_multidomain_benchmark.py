"""
Phase 6: Multi-Domain Empirical Benchmarking Suite
Evaluates HQGWN against 5 competing baseline families across 3 major telemetry domains:
  1. Biomedical Telemetry: Clinical 16-channel EEG Seizure Detection (10-20 Montage)
  2. Brain-Computer Interface (BCI): 64-channel 4-Class Motor Imagery (Left, Right, Feet, Tongue)
  3. Cognitive Radio RF Telemetry: 4-Class Digital Modulation under Rayleigh fading (BPSK, QPSK, 8PSK, 16QAM)
"""

import sys; sys.path.insert(0, ".")
import os
import json
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from scipy.signal import welch
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from hqgwn.models.classifier import HydroQuantumGraphClassifier

# ============================================================================
# DOMAIN DATASET GENERATORS
# ============================================================================

def generate_biomedical_eeg_dataset(n_samples=160, n_channels=16, n_time=256, noise=0.5):
    """Domain 1: 16-channel clinical EEG seizure detection (Ictal vs Interictal)."""
    np.random.seed(42)
    t = np.linspace(0, 2.0, n_time)
    X = np.zeros((n_samples, n_channels, n_time))
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        label = i % 2
        y[i] = label
        noise_vec = np.random.normal(0, noise, size=(n_channels, n_time))
        
        if label == 0:
            # Baseline: background alpha (10 Hz) in occipital channels (8, 9)
            X[i] = noise_vec
            X[i, 8:10] += 0.8 * np.sin(2 * np.pi * 10.0 * t)
        else:
            # Ictal Seizure: high-amplitude 3 Hz spike-and-wave traveling across anterior-posterior axis
            for m in range(n_channels):
                phase_delay = m * (np.pi / 8.0)
                spike = np.sin(2 * np.pi * 3.0 * t - phase_delay) + 0.6 * np.sin(2 * np.pi * 6.0 * t - 2*phase_delay)
                X[i, m] = 1.2 * spike + noise_vec[m]
                
    return X, y

def generate_bci_motor_imagery_dataset(n_samples=200, n_channels=64, n_time=128, noise=0.6):
    """Domain 2: 64-channel 4-class BCI Motor Imagery (Left Hand, Right Hand, Feet, Tongue)."""
    np.random.seed(101)
    t = np.linspace(0, 1.0, n_time)
    X = np.zeros((n_samples, n_channels, n_time))
    y = np.zeros(n_samples, dtype=int)
    
    # Key electrode indices (simulated C3=10, C4=20, Cz=30, Fz=40)
    for i in range(n_samples):
        label = i % 4
        y[i] = label
        noise_vec = np.random.normal(0, noise, size=(n_channels, n_time))
        X[i] = noise_vec
        
        # Sensorimotor rhythm: Event-Related Desynchronization (ERD) / Synchronization (ERS)
        if label == 0: # Left Hand (Right C4 desync, Left C3 sync 12 Hz)
            X[i, 10] += 1.4 * np.sin(2 * np.pi * 12.0 * t)
            X[i, 20] += 0.2 * np.sin(2 * np.pi * 12.0 * t)
        elif label == 1: # Right Hand (Left C3 desync, Right C4 sync 12 Hz)
            X[i, 10] += 0.2 * np.sin(2 * np.pi * 12.0 * t)
            X[i, 20] += 1.4 * np.sin(2 * np.pi * 12.0 * t)
        elif label == 2: # Both Feet (Cz modulation 22 Hz beta)
            X[i, 25] += 1.5 * np.sin(2 * np.pi * 22.0 * t)
        elif label == 3: # Tongue Movement (Frontal Fz modulation 8 Hz theta)
            X[i, 5] += 1.5 * np.sin(2 * np.pi * 8.0 * t)
            
    return X, y

def generate_radioml_rf_dataset(n_samples=240, n_subcarriers=16, n_time=128, snr_db=5.0):
    """Domain 3: 16-subcarrier MIMO RF modulation sensing (BPSK, QPSK, 8PSK, 16QAM)."""
    np.random.seed(202)
    t = np.linspace(0, 8 * np.pi, n_time)
    X = np.zeros((n_samples, n_subcarriers, n_time))
    y = np.zeros(n_samples, dtype=int)
    
    # Calculate noise sigma for specified SNR
    p_sig = 1.0
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    noise_sigma = np.sqrt(p_noise)
    
    for i in range(n_samples):
        label = i % 4
        y[i] = label
        
        for sc in range(n_subcarriers):
            # Rayleigh multipath channel fading coefficient
            rayleigh_gain = np.random.rayleigh(scale=1.0)
            phase_offset = sc * (np.pi / 4.0)
            
            if label == 0: # BPSK (constellation +/- 1)
                bits = np.random.choice([-1.0, 1.0], size=8)
                mod = np.repeat(bits, n_time // 8)
                sig = mod * np.cos(3.0 * t + phase_offset)
            elif label == 1: # QPSK (4 constellation phases)
                phases = np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2], size=8)
                mod_phase = np.repeat(phases, n_time // 8)
                sig = np.cos(3.0 * t + phase_offset + mod_phase)
            elif label == 2: # 8PSK (8 constellation phases)
                phases = np.random.choice([k * np.pi / 4 for k in range(8)], size=8)
                mod_phase = np.repeat(phases, n_time // 8)
                sig = np.cos(3.0 * t + phase_offset + mod_phase)
            elif label == 3: # 16QAM (amplitude + phase modulation)
                amps = np.random.choice([0.5, 1.0, 1.5], size=8)
                phases = np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2], size=8)
                mod_amp = np.repeat(amps, n_time // 8)
                mod_phase = np.repeat(phases, n_time // 8)
                sig = mod_amp * np.cos(3.0 * t + phase_offset + mod_phase)
                
            noise = np.random.normal(0, noise_sigma, size=n_time)
            X[i, sc] = rayleigh_gain * sig + noise
            
    return X, y

# ============================================================================
# COMPETING BASELINE ALGORITHMS
# ============================================================================

class WelchPsdSvmModel:
    """Baseline 1: Multitaper / Welch PSD + RBF-SVM."""
    def __init__(self):
        self.clf = SVC(kernel="rbf", C=2.0)
    def _extract(self, X):
        feats = []
        for i in range(len(X)):
            _, p = welch(X[i], nperseg=min(32, X.shape[-1]), axis=-1)
            feats.append(p.reshape(-1))
        return np.array(feats)
    def fit(self, X, y):
        self.clf.fit(self._extract(X), y)
        return self
    def predict(self, X):
        return self.clf.predict(self._extract(X))

class RiemannianCovarianceModel:
    """Baseline 2: Riemannian Covariance Tangent Space Projection + Logistic Regression."""
    def __init__(self):
        self.clf = LogisticRegression(max_iter=500)
    def _extract(self, X):
        feats = []
        M = X.shape[1]
        for i in range(len(X)):
            cov = np.cov(X[i]) + 1e-6 * np.eye(M)
            eigvals, eigvecs = np.linalg.eigh(cov)
            log_cov = eigvecs @ np.diag(np.log(np.maximum(eigvals, 1e-8))) @ eigvecs.T
            triu_idx = np.triu_indices(M)
            feats.append(log_cov[triu_idx])
        return np.array(feats)
    def fit(self, X, y):
        self.clf.fit(self._extract(X), y)
        return self
    def predict(self, X):
        return self.clf.predict(self._extract(X))

class ClassicalMarkovRandomWalkModel:
    """Baseline 3: Classical Continuous-Time Markov Random Walk on Pearson Graph."""
    def __init__(self):
        self.model = HydroQuantumGraphClassifier(
            use_hydrodynamic=False, use_quantum_walk=False, use_quantum_entropy=False, dt_steps=6
        )
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    def predict(self, X):
        return self.model.predict(X)

# ============================================================================
# MULTI-DOMAIN BENCHMARK RUNNER
# ============================================================================

def run_multidomain_benchmark(output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 80)
    print("PHASE 6: MULTI-DOMAIN EMPIRICAL BENCHMARKING SUITE")
    print("=" * 80)
    
    domains = {
        "Biomedical_EEG_Seizure": generate_biomedical_eeg_dataset(n_samples=160, n_channels=16, n_time=256, noise=0.6),
        "BCI_Motor_Imagery_4Class": generate_bci_motor_imagery_dataset(n_samples=160, n_channels=32, n_time=128, noise=0.7),
        "RadioML_RF_Modulation_4Class": generate_radioml_rf_dataset(n_samples=160, n_subcarriers=16, n_time=128, snr_db=5.0),
    }
    
    models = {
        "Welch_PSD_SVM": lambda: WelchPsdSvmModel(),
        "Riemannian_Cov": lambda: RiemannianCovarianceModel(),
        "Classical_Markov_RW": lambda: ClassicalMarkovRandomWalkModel(),
        "Full_HQGWN (Proposed)": lambda: HydroQuantumGraphClassifier(
            use_hydrodynamic=True, use_quantum_walk=True, use_quantum_entropy=True, alpha_dephase=0.15, dt_steps=6
        ),
    }
    
    results = {}
    
    for domain_name, (X, y) in domains.items():
        print(f"\n" + "-" * 80)
        print(f"BENCHMARKING DOMAIN: {domain_name}")
        print(f"Dataset: N={len(X)} samples, M={X.shape[1]} channels, L={X.shape[2]} timepoints, K={len(np.unique(y))} classes.")
        print("-" * 80)
        
        domain_res = {}
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name, model_fn in models.items():
            f1_scores = []
            acc_scores = []
            kappa_scores = []
            latencies = []
            
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                clf = model_fn()
                t0 = time.time()
                clf.fit(X[train_idx], y[train_idx])
                t_fit = time.time() - t0
                
                t0 = time.time()
                preds = clf.predict(X[test_idx])
                lat = (time.time() - t0) / len(test_idx) * 1000.0 # ms per sample
                
                f1_scores.append(f1_score(y[test_idx], preds, average="macro") * 100)
                acc_scores.append(accuracy_score(y[test_idx], preds) * 100)
                kappa_scores.append(cohen_kappa_score(y[test_idx], preds))
                latencies.append(lat)
                
            f1_m, f1_s = np.mean(f1_scores), np.std(f1_scores)
            acc_m, acc_s = np.mean(acc_scores), np.std(acc_scores)
            kap_m, kap_s = np.mean(kappa_scores), np.std(kappa_scores)
            lat_m = np.mean(latencies)
            
            print(f"  {model_name:<24} | F1: {f1_m:5.2f} ± {f1_s:4.2f}% | Acc: {acc_m:5.2f} ± {acc_s:4.2f}% | Kappa: {kap_m:5.3f} | Latency: {lat_m:4.1f} ms")
            domain_res[model_name] = {
                "f1_mean": f1_m, "f1_std": f1_s,
                "acc_mean": acc_m, "acc_std": acc_s,
                "kappa_mean": kap_m, "kappa_std": kap_s,
                "latency_ms": lat_m
            }
            
        results[domain_name] = domain_res
        
    with open(os.path.join(output_dir, "multidomain_benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved multi-domain benchmark results to results/multidomain_benchmark_results.json")
    return results

if __name__ == "__main__":
    run_multidomain_benchmark()
