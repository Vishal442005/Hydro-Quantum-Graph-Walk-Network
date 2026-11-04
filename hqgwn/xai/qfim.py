"""
Stage VII: Quantum Fisher Information Matrix (QFIM) Channel Attribution
Computes the exact quantum geometric sensitivity of each sensor channel.
"""

import numpy as np
from scipy.linalg import eigh

def compute_channel_qfim_attribution(rho: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Computes the diagonal Quantum Fisher Information Matrix (QFIM) elements:
        F_mm = 2 * sum_{j, k: lambda_j + lambda_k > eps} [ (lambda_j - lambda_k)^2 / (lambda_j + lambda_k) ] * |<psi_j | m><m | psi_k>|^2
    
    Since the perturbation generator is the site projection operator |m><m|,
        |<psi_j | m><m | psi_k>|^2 = |psi_j(m)|^2 * |psi_k(m)|^2
        
    Parameters:
        rho: shape (M, M), density matrix of an input sample
    Returns:
        attribution: shape (M,), normalized sensitivity score for each channel
    """
    M = rho.shape[0]
    eigvals, eigvecs = eigh(rho)
    
    # Clip small eigenvalues
    eigvals = np.clip(np.real(eigvals), 0.0, 1.0)
    
    # Probabilities of measuring site m in eigenstate j: P[j, m] = |psi_j(m)|^2
    P = np.abs(eigvecs) ** 2  # shape (M_sites, M_eigenstates)
    
    qfim_diag = np.zeros(M)
    
    for j in range(M):
        for k in range(M):
            denom = eigvals[j] + eigvals[k]
            if denom > eps:
                weight = ((eigvals[j] - eigvals[k]) ** 2) / denom
                # Vectorized across all sites m: P[:, j] * P[:, k]
                qfim_diag += weight * (P[:, j] * P[:, k])
                
    qfim_diag *= 2.0
    total = np.sum(qfim_diag)
    if total > 1e-12:
        qfim_diag /= total
    else:
        qfim_diag = np.ones(M) / M
        
    return qfim_diag
