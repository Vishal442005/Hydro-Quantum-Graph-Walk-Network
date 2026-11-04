"""
Hydro-Quantum Graph Walk Network (HQGWN) Classifier
Scikit-learn compliant estimator unifying Hydrodynamics, Lindblad Quantum Walks, and Quantum Thermodynamics.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from scipy.linalg import eigh
from typing import Dict, List, Tuple

from ..hydro.laplacian import build_hydrodynamic_laplacian, build_hamiltonian
from ..quantum.lindblad import evolve_lindblad_quantum_walk
from ..thermo.entropy import extract_quantum_invariants
from ..thermo.qkl import compute_umegaki_qkl, compute_bures_distance, compute_class_prototypes

class HydroQuantumGraphClassifier(BaseEstimator, ClassifierMixin):
    """
    Hydro-Quantum Graph Walk Network (HQGWN) Classifier
    
    A physics-grounded classification model unifying:
      1. Navier-Stokes Poincaré-Cartan Phase-Vorticity Graphs
      2. Open-System Lindblad Stochastic Quantum Walk (SLQW) Evolution
      3. Quantum Thermodynamic Invariant Extraction & Umegaki QKL Divergence Collapse
    """
    def __init__(
        self,
        use_hydrodynamic: bool = True,
        use_quantum_walk: bool = True,
        use_quantum_entropy: bool = True,
        gamma_hop: float = 1.2,
        alpha_dephase: float = 0.15,
        walk_time: float = 1.5,
        dt_steps: int = 12,
        beta_shear: float = 2.0,
        adaptive_annealing: bool = False,
        use_bures_metric: bool = True,
        n_estimators: int = 150,
        random_state: int = 42
    ):
        self.use_hydrodynamic = use_hydrodynamic
        self.use_quantum_walk = use_quantum_walk
        self.use_quantum_entropy = use_quantum_entropy
        self.gamma_hop = gamma_hop
        self.alpha_dephase = alpha_dephase
        self.walk_time = walk_time
        self.dt_steps = dt_steps
        self.beta_shear = beta_shear
        self.adaptive_annealing = adaptive_annealing
        self.use_bures_metric = use_bures_metric
        self.n_estimators = n_estimators
        self.random_state = random_state

    def _extract_single_sample(self, X_sample: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processes a single multi-channel time-series window X_sample of shape (M, L).
        Returns evolved density matrix rho(tau) and base thermodynamic feature vector.
        """
        # Stage I: Hydrodynamic Navier-Stokes Phase-Vorticity Graph
        Laplacian, energies, W, z = build_hydrodynamic_laplacian(
            X_sample, 
            use_hydrodynamic=self.use_hydrodynamic,
            beta_shear=self.beta_shear
        )
        
        # System Hamiltonian
        H = build_hamiltonian(Laplacian, energies, gamma_hop=self.gamma_hop)
        
        # Stage II: Stochastic Lindblad Quantum Walk (with spatial phase modulation z)
        rho_tau = evolve_lindblad_quantum_walk(
            H=H,
            Laplacian=Laplacian,
            energies=energies,
            z=z if self.use_hydrodynamic else None,
            walk_time=self.walk_time,
            dt_steps=self.dt_steps,
            alpha_dephase=self.alpha_dephase,
            use_quantum_walk=self.use_quantum_walk,
            adaptive_annealing=self.adaptive_annealing,
            method="strang"
        )
        
        # Stage III: Macro Graph Descriptors + Quantum Thermodynamic Invariants
        mask = ~np.eye(W.shape[0], dtype=bool)
        macro_stats = [
            float(np.mean(W[mask])), 
            float(np.std(W[mask])), 
            float(np.max(W[mask]))
        ]
        
        eigvals_L = np.sort(np.real(eigh(Laplacian, eigvals_only=True)))
        algebraic_conn = float(eigvals_L[1]) if len(eigvals_L) > 1 else 0.0
        spectral_radius = float(eigvals_L[-1])
        graph_invariants = macro_stats + [algebraic_conn, spectral_radius]
        
        if not self.use_quantum_entropy:
            # Ablation A1/A2/A5: Only macro graph stats + diagonal node probabilities
            base_feat = np.concatenate([graph_invariants, np.real(np.diag(rho_tau))])
        else:
            invariants, _ = extract_quantum_invariants(rho_tau)
            base_feat = np.concatenate([graph_invariants, invariants])
            
        return rho_tau, base_feat

    def _extract_dataset_features(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts density matrices and thermodynamic features across all samples.
        X: shape (N_samples, M_channels, L_timepoints)
        """
        N = X.shape[0]
        rho_list = []
        base_features = []
        
        for i in range(N):
            rho_i, feat_i = self._extract_single_sample(X[i])
            rho_list.append(rho_i)
            base_features.append(feat_i)
            
        return np.array(rho_list), np.array(base_features)

    def _compute_quantum_distance_features(self, rho_list: np.ndarray) -> np.ndarray:
        """
        Computes Umegaki QKL Divergence and Bures Geodesic distances to class prototypes.
        """
        N = len(rho_list)
        K = len(self.classes_)
        num_dist_metrics = 2 if self.use_bures_metric else 1
        dist_features = np.zeros((N, K * num_dist_metrics))
        
        for i, rho in enumerate(rho_list):
            for c_idx, c in enumerate(self.classes_):
                proto = self.prototypes_[c]
                qkl_val = compute_umegaki_qkl(rho, proto)
                dist_features[i, c_idx] = qkl_val
                
                if self.use_bures_metric:
                    bures_val = compute_bures_distance(rho, proto)
                    dist_features[i, K + c_idx] = bures_val
                    
        return dist_features

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the HQGWN model:
          1. Evolve all samples into quantum density matrices.
          2. Compute class prototype Gibbs states bar_rho_c.
          3. Extract thermodynamic invariants and quantum relative distances.
          4. Train downstream ensemble classifier.
        """
        self.classes_ = np.unique(y)
        rho_list, base_feats = self._extract_dataset_features(X)
        
        # Compute Class Prototype Density Matrices
        self.prototypes_ = compute_class_prototypes(rho_list, y)
        
        if self.use_quantum_entropy:
            dist_feats = self._compute_quantum_distance_features(rho_list)
            full_feats = np.hstack([base_feats, dist_feats])
        else:
            full_feats = base_feats
            
        self.clf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.clf_.fit(full_feats, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms input signals into quantum thermodynamic representation vectors.
        """
        rho_list, base_feats = self._extract_dataset_features(X)
        if self.use_quantum_entropy:
            dist_feats = self._compute_quantum_distance_features(rho_list)
            return np.hstack([base_feats, dist_feats])
        return base_feats

    def predict(self, X: np.ndarray) -> np.ndarray:
        feats = self.transform(X)
        return self.clf_.predict(feats)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        feats = self.transform(X)
        return self.clf_.predict_proba(feats)
