"""
Stage I: Hydrodynamic Vorticity Shear Tensor & Phase-Locking Viscosity
Computes dimensionless normalized vorticity shear and Kuramoto phase-locking viscosity.
"""

import numpy as np
from typing import Tuple

def compute_vorticity_and_viscosity(
    phase_velocity: np.ndarray, 
    phase_unwrapped: np.ndarray,
    z: np.ndarray = None,
    eps: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes pairwise dimensionless vorticity shear Omega_tilde_mn in [0, 1] 
    and phase-locking viscosity nu_mn in [0, 1].
    
    Parameters:
        phase_velocity: shape (M, L)
        phase_unwrapped: shape (M, L)
        z: optional complex analytic signal shape (M, L)
        eps: numerical stability epsilon
    Returns:
        omega_tilde: shape (M, M), normalized pairwise velocity divergence in [0, 1]
        nu_viscosity: shape (M, M), phase synchronization index in [0, 1]
    """
    M = phase_velocity.shape[0]
    
    # 1. Dimensionless Normalized Vorticity Shear:
    # num = sum_t (v_m(t) - v_n(t))^2
    # denom = sum_t (v_m^2(t) + v_n^2(t))
    v_diff = phase_velocity[:, np.newaxis, :] - phase_velocity[np.newaxis, :, :]
    num = np.sum(v_diff ** 2, axis=-1)
    
    v_sq = np.sum(phase_velocity ** 2, axis=-1)
    denom = v_sq[:, np.newaxis] + v_sq[np.newaxis, :] + eps
    omega_tilde = num / denom
    
    # Clip to [0, 1] and zero diagonal
    omega_tilde = np.clip(omega_tilde, 0.0, 1.0)
    np.fill_diagonal(omega_tilde, 0.0)
    
    # 2. Phase-Locking Viscosity (Kuramoto Order Parameter)
    if z is not None:
        phasors = z / (np.abs(z) + eps)
        cross_phasor = phasors[:, np.newaxis, :] * phasors[np.newaxis, :, :].conj()
        nu_viscosity = np.abs(np.mean(cross_phasor, axis=-1))
    else:
        phase_diff = phase_unwrapped[:, np.newaxis, :] - phase_unwrapped[np.newaxis, :, :]
        nu_viscosity = np.abs(np.mean(np.exp(1j * phase_diff), axis=-1))
        
    nu_viscosity = np.clip(nu_viscosity, 0.0, 1.0)
    np.fill_diagonal(nu_viscosity, 1.0)
    
    return omega_tilde, nu_viscosity

def compute_hydrodynamic_adjacency(
    omega_shear: np.ndarray,
    nu_viscosity: np.ndarray,
    beta_shear: float = 2.0
) -> np.ndarray:
    """
    Computes Reynolds-weighted hydrodynamic adjacency matrix:
        W_mn = nu_mn * exp( - beta_shear * Omega_tilde_mn )
        
    Parameters:
        omega_shear: normalized shear in [0, 1]
        nu_viscosity: phase locking in [0, 1]
        beta_shear: shear penalty sensitivity
    Returns:
        W: shape (M, M), strictly in [0, 1], zero on diagonal
    """
    W = nu_viscosity * np.exp(-beta_shear * omega_shear)
    np.fill_diagonal(W, 0.0)
    return W
