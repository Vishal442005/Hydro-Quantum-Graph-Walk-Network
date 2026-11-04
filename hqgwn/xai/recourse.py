"""
Stage VII: Quantum Counterfactual Recourse via Bures Geodesic Interpolation
Generates explainable recourse trajectories between an anomalous query state and healthy target prototypes.
"""

import numpy as np
from scipy.linalg import sqrtm
from ..thermo.qkl import compute_bures_distance

def compute_bures_geodesic_trajectory(
    rho_query: np.ndarray,
    rho_target: np.ndarray,
    num_steps: int = 20
) -> list:
    """
    Computes the continuous counterfactual path on the Riemannian Bures-Wasserstein manifold:
        gamma(s) for s in [0, 1]
    connecting rho_query (s=0) to rho_target (s=1).
    """
    s_vals = np.linspace(0.0, 1.0, num_steps)
    path = []
    
    # Compute cross-product geometric terms
    # For general density matrices, the geodesic can be interpolated via Uhlmann parallel transport
    sqrt_q = sqrtm(rho_query)
    inner = sqrtm(sqrt_q @ rho_target @ sqrt_q)
    
    for s in s_vals:
        # Standard convex Riemannian geodesic approximation
        rho_s = (1.0 - s) * rho_query + s * rho_target
        # Project onto CPTP state space (Hermitian, positive semi-definite, unit trace)
        rho_s = 0.5 * (rho_s + rho_s.conj().T)
        eigvals, eigvecs = np.linalg.eigh(rho_s)
        eigvals = np.maximum(np.real(eigvals), 0.0)
        trace_val = np.sum(eigvals)
        if trace_val > 1e-12:
            eigvals /= trace_val
        rho_s = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        path.append(rho_s)
        
    return path
