"""
Stage V: Krylov-Lanczos Subspace Quantum Walk Approximation
Accelerates matrix exponential evaluation from O(M^3) to O(k M^2 + k^3) for high-density sensor arrays.
"""

import numpy as np

def lanczos_iteration(H: np.ndarray, v0: np.ndarray, k_dim: int = 15):
    """
    Constructs an orthonormal basis V_k and tridiagonal matrix T_k for Krylov subspace K_k(H, v0):
        H * V_k = V_k * T_k + beta_{k} * v_{k+1} * e_k^T
    """
    M = H.shape[0]
    k_dim = min(k_dim, M)
    
    V = np.zeros((M, k_dim), dtype=complex)
    alpha = np.zeros(k_dim)
    beta = np.zeros(k_dim - 1)
    
    norm_v0 = np.linalg.norm(v0)
    if norm_v0 < 1e-12:
        return np.eye(M, k_dim), np.zeros((k_dim, k_dim))
    V[:, 0] = v0 / norm_v0
    
    for j in range(k_dim):
        w = H @ V[:, j]
        if j > 0:
            w -= beta[j - 1] * V[:, j - 1]
        alpha[j] = np.real(np.vdot(V[:, j], w))
        w -= alpha[j] * V[:, j]
        
        # Gram-Schmidt reorthogonalization
        for p in range(j + 1):
            w -= np.vdot(V[:, p], w) * V[:, p]
            
        if j < k_dim - 1:
            beta_val = np.linalg.norm(w)
            if beta_val < 1e-12:
                # Invariant subspace reached
                k_actual = j + 1
                T_sub = np.diag(alpha[:k_actual]) + np.diag(beta[:k_actual - 1], 1) + np.diag(beta[:k_actual - 1], -1)
                return V[:, :k_actual], T_sub
            beta[j] = beta_val
            V[:, j + 1] = w / beta_val
            
    T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
    return V, T

def krylov_matrix_exp_action(H: np.ndarray, v: np.ndarray, dt: float, k_dim: int = 15) -> np.ndarray:
    """
    Approximates exp(-i * H * dt) * v using Krylov-Lanczos subspace projection:
        exp(-i H dt) v approx ||v|| * V_k * exp(-i T_k dt) * e_1
    """
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-12:
        return np.zeros_like(v)
        
    V_k, T_k = lanczos_iteration(H, v, k_dim=k_dim)
    # Exponentiate small k x k tridiagonal matrix
    eigvals_T, eigvecs_T = np.linalg.eigh(T_k)
    exp_T = eigvecs_T @ np.diag(np.exp(-1j * eigvals_T * dt)) @ eigvecs_T.conj().T
    
    e1 = np.zeros(T_k.shape[0])
    e1[0] = 1.0
    
    res = norm_v * (V_k @ (exp_T @ e1))
    return res
