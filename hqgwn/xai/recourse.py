"""
Stage VII: Quantum Counterfactual Recourse via Bures Geodesic Interpolation
Generates explainable recourse trajectories between an anomalous query state and healthy target prototypes
along the Riemannian Bures-Wasserstein geodesic.
"""

import numpy as np
from scipy.linalg import sqrtm, inv
from ..thermo.qkl import compute_bures_distance

def compute_bures_geodesic_trajectory(
    rho_query: np.ndarray,
    rho_target: np.ndarray,
    num_steps: int = 20
) -> list:
    """
    Computes the continuous counterfactual path on the Riemannian Bures-Wasserstein manifold:
        gamma(s) for s in [0, 1]
    connecting rho_query (s=0) to rho_target (s=1) via the exact McCann / quantum optimal transport geodesic:
        T = rho_0^{-1/2} (rho_0^{1/2} rho_1 rho_0^{1/2})^{1/2} rho_0^{-1/2}
        rho(s) = (1 / Z(s)) * ((1-s) I + s T) rho_0 ((1-s) I + s T^dagger)
    where Z(s) guarantees exact unit trace for all s in [0, 1].
    """
    M = rho_query.shape[0]
    eps = 1e-7 * np.eye(M)
    rho_0 = rho_query + eps
    rho_1 = rho_target + eps
    
    # Eigendecomposition for stable fractional powers
    vals0, vecs0 = np.linalg.eigh(0.5 * (rho_0 + rho_0.conj().T))
    vals0 = np.maximum(vals0, 1e-8)
    sqrt_rho0 = vecs0 @ np.diag(np.sqrt(vals0)) @ vecs0.conj().T
    inv_sqrt_rho0 = vecs0 @ np.diag(1.0 / np.sqrt(vals0)) @ vecs0.conj().T
    
    middle = sqrt_rho0 @ rho_1 @ sqrt_rho0
    middle = 0.5 * (middle + middle.conj().T)
    vals_m, vecs_m = np.linalg.eigh(middle)
    vals_m = np.maximum(vals_m, 1e-8)
    sqrt_middle = vecs_m @ np.diag(np.sqrt(vals_m)) @ vecs_m.conj().T
    
    T = inv_sqrt_rho0 @ sqrt_middle @ inv_sqrt_rho0
    
    s_vals = np.linspace(0.0, 1.0, num_steps)
    path = []
    
    for s in s_vals:
        A_s = (1.0 - s) * np.eye(M) + s * T
        rho_s = A_s @ rho_0 @ A_s.conj().T
        # Strict Hermiticity and unit trace normalization
        rho_s = 0.5 * (rho_s + rho_s.conj().T)
        trace_val = np.real(np.trace(rho_s))
        if trace_val > 1e-12:
            rho_s /= trace_val
        path.append(rho_s)
        
    return path
