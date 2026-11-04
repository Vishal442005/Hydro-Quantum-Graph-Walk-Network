"""
Stage III: Umegaki Quantum Relative Entropy & Bures-Wasserstein Metric Geometry
Computes quantum relative free energy and geodesic manifold distances against class prototype states.
"""

import numpy as np
from scipy.linalg import eigh, sqrtm

def matrix_log(rho: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Computes Hermitian matrix logarithm ln(rho) via spectral decomposition.
    """
    eigvals, eigvecs = eigh(rho)
    eigvals_clipped = np.clip(np.real(eigvals), eps, 1.0)
    log_diag = np.diag(np.log(eigvals_clipped))
    return eigvecs @ log_diag @ eigvecs.conj().T

def matrix_sqrt(rho: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Computes Hermitian matrix square root rho^(1/2) via spectral decomposition.
    """
    eigvals, eigvecs = eigh(rho)
    eigvals_clipped = np.clip(np.real(eigvals), eps, 1.0)
    sqrt_diag = np.diag(np.sqrt(eigvals_clipped))
    return eigvecs @ sqrt_diag @ eigvecs.conj().T

def compute_umegaki_qkl(rho: np.ndarray, sigma: np.ndarray, eps: float = 1e-12) -> float:
    """
    Computes Umegaki Quantum Relative Entropy (Quantum KL Divergence):
        D_QKL(rho || sigma) = Tr( rho * (ln(rho) - ln(sigma)) )
    
    Always non-negative (Klein's Inequality), zero iff rho == sigma.
    """
    log_rho = matrix_log(rho, eps=eps)
    log_sigma = matrix_log(sigma, eps=eps)
    
    diff = log_rho - log_sigma
    qkl = np.real(np.trace(rho @ diff))
    return float(max(qkl, 0.0))

def compute_bures_distance(rho: np.ndarray, sigma: np.ndarray, eps: float = 1e-12) -> float:
    """
    Computes the Riemannian Bures-Wasserstein geodesic metric between density matrices:
        d_Bures^2(rho, sigma) = 2 * (1 - Tr( (rho^(1/2) sigma rho^(1/2))^(1/2) ))
    """
    rho_half = matrix_sqrt(rho, eps=eps)
    inner = rho_half @ sigma @ rho_half
    inner_half = matrix_sqrt(inner, eps=eps)
    
    fidelity_root = np.real(np.trace(inner_half))
    fidelity_root = np.clip(fidelity_root, 0.0, 1.0)
    
    dist_sq = 2.0 * (1.0 - fidelity_root)
    return float(np.sqrt(max(dist_sq, 0.0)))

def compute_class_prototypes(rho_list: np.ndarray, y: np.ndarray, eps_mix: float = 0.05) -> dict:
    """
    Computes the Gibbs class prototype density matrices with Laplace thermal regularization:
        bar_rho_c = (1 - eps_mix) * (1 / N_c) * sum_{y_i = c} rho_i + eps_mix * (I / M)
    Guarantees strictly full-rank support, eliminating kernel singularities in matrix logarithm.
    """
    classes = np.unique(y)
    M = rho_list.shape[-1]
    prototypes = {}
    for c in classes:
        subset = rho_list[y == c]
        proto = np.mean(subset, axis=0)
        # Ensure Hermiticity & unit trace
        proto = 0.5 * (proto + proto.conj().T)
        proto /= (np.real(np.trace(proto)) + 1e-12)
        
        # Thermal Laplace smoothing
        if eps_mix > 0:
            proto = (1.0 - eps_mix) * proto + eps_mix * (np.eye(M, dtype=complex) / M)
            
        prototypes[c] = proto
    return prototypes
