"""
Stage III: Quantum Thermodynamic Invariant Extraction
Computes Von Neumann Entanglement Entropy, Quantum Purity, l1-Coherence, and Spectral Invariants.
"""

import numpy as np
from scipy.linalg import eigh
from typing import Dict, Tuple

def extract_quantum_invariants(rho: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Extracts topological and thermodynamic invariants from density matrix rho.
    
    Parameters:
        rho: shape (M, M), Hermitian density matrix
    Returns:
        feature_vector: 1D array of extracted features
        metrics_dict: dictionary of individual scalar invariants
    """
    M = rho.shape[0]
    
    # 1. Site occupation probabilities (diagonal elements)
    site_probs = np.real(np.diag(rho))
    
    # 2. Spectral decomposition for Von Neumann Entropy & Purity
    eigvals = np.real(eigh(rho, eigvals_only=True))
    eigvals = np.clip(eigvals, 1e-14, 1.0)
    eigvals /= np.sum(eigvals)
    
    # Von Neumann Entropy: S_VN = - sum lambda_i * ln(lambda_i)
    vn_entropy = -float(np.sum(eigvals * np.log(eigvals)))
    
    # Quantum Purity: gamma = Tr(rho^2) = sum lambda_i^2
    purity = float(np.sum(eigvals ** 2))
    
    # Linear entropy: S_L = 1 - Tr(rho^2)
    linear_entropy = 1.0 - purity
    
    # 3. Off-diagonal l1-norm Quantum Coherence
    off_diag = rho - np.diag(np.diag(rho))
    coherence_l1 = float(np.sum(np.abs(off_diag)))
    
    # 4. Off-diagonal relative phase coherence
    phase_angles = np.angle(off_diag)
    phase_entropy = float(-np.sum(np.var(phase_angles, axis=0)))
    
    metrics = {
        "vn_entropy": vn_entropy,
        "purity": purity,
        "linear_entropy": linear_entropy,
        "coherence_l1": coherence_l1,
        "phase_entropy": phase_entropy
    }
    
    # Concatenate full feature vector:
    # [site_probs (M), sorted eigenvalues (M), vn_entropy, purity, linear_entropy, coherence_l1]
    feature_vector = np.concatenate([
        site_probs,
        np.sort(eigvals),
        [vn_entropy, purity, linear_entropy, coherence_l1]
    ])
    
    return feature_vector, metrics
