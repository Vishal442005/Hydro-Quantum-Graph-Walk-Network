"""
Stage I & II Interface: Hydrodynamic Laplacian & System Hamiltonian
Constructs the normalized graph Laplacian and maps it to Hilbert space Hamiltonian H.
"""

import numpy as np
from typing import Tuple
from .analytic import compute_phase_velocity_field, compute_analytic_signal
from .vorticity import compute_vorticity_and_viscosity, compute_hydrodynamic_adjacency

def build_hydrodynamic_laplacian(
    X_sample: np.ndarray,
    use_hydrodynamic: bool = True,
    beta_shear: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds the normalized graph Laplacian, extracts channel energies, and computes analytic signal.
    
    Parameters:
        X_sample: shape (M_channels, L_timepoints)
        use_hydrodynamic: If True, uses Navier-Stokes phase-vorticity.
                          If False, falls back to Pearson correlation baseline.
        beta_shear: sensitivity to normalized velocity shear
    Returns:
        Laplacian: shape (M, M), normalized graph Laplacian L
        energies: shape (M,), signal energy per channel
        W: shape (M, M), adjacency matrix
        z: shape (M, L), complex analytic representation
    """
    M, L = X_sample.shape
    energies = np.mean(X_sample ** 2, axis=1) + 1e-8
    z = compute_analytic_signal(X_sample)
    
    if not use_hydrodynamic:
        # Ablation baseline: Pearson Correlation Adjacency
        corr = np.abs(np.corrcoef(X_sample))
        np.fill_diagonal(corr, 0.0)
        W = np.nan_to_num(corr, nan=0.0)
    else:
        # Upgraded Stage I: Poincaré-Cartan phase-velocity & normalized vorticity shear
        _, phase_unwrapped, phase_velocity = compute_phase_velocity_field(X_sample)
        omega_shear, nu_viscosity = compute_vorticity_and_viscosity(phase_velocity, phase_unwrapped, z=z)
        W = compute_hydrodynamic_adjacency(omega_shear, nu_viscosity, beta_shear=beta_shear)
        
    deg = np.sum(W, axis=1) + 1e-8
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    Laplacian = np.eye(M) - D_inv_sqrt @ W @ D_inv_sqrt
    
    # Enforce exact symmetry
    Laplacian = 0.5 * (Laplacian + Laplacian.T)
    return Laplacian, energies, W, z

def build_hamiltonian(
    Laplacian: np.ndarray, 
    energies: np.ndarray, 
    gamma_hop: float = 1.2
) -> np.ndarray:
    """
    Constructs the system Hamiltonian H in Hilbert space H_M:
        H = - gamma_hop * Laplacian + diag(E / max(E))
        
    Parameters:
        Laplacian: shape (M, M)
        energies: shape (M,)
        gamma_hop: continuous-time quantum walk hopping rate
    Returns:
        H: shape (M, M), Hermitian Hamiltonian matrix
    """
    E_norm = energies / (np.max(energies) + 1e-12)
    H = -gamma_hop * Laplacian + np.diag(E_norm)
    return 0.5 * (H + H.conj().T)
