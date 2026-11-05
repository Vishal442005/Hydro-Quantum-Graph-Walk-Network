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
    Omega: np.ndarray = None,
    gamma_hop: float = 1.2,
    alpha_vort: float = 0.4
) -> np.ndarray:
    """
    Constructs the strictly Hermitian system Hamiltonian H in Hilbert space H_M:
        H = - gamma_hop * L_S + i * alpha_vort * Omega + diag(E / max(E))
    where L_S is the symmetric graph Laplacian and Omega is the real skew-symmetric
    vorticity curl operator. Because (i * Omega)^dagger = -i * Omega^T = i * Omega,
    the Hamiltonian H is guaranteed to be strictly Hermitian (H^dagger = H), ensuring
    exact unitary evolution U(tau) = exp(-i * H * tau) and complete positivity.
        
    Parameters:
        Laplacian: shape (M, M), symmetric Laplacian L_S
        energies: shape (M,), signal energy per channel
        Omega: shape (M, M), optional real skew-symmetric vorticity matrix
        gamma_hop: continuous-time quantum walk hopping rate
        alpha_vort: coupling strength of chiral vorticity gauge rotation
    Returns:
        H: shape (M, M), strictly Hermitian Hamiltonian matrix
    """
    M = Laplacian.shape[0]
    E_norm = energies / (np.max(energies) + 1e-12)
    H = -gamma_hop * Laplacian + np.diag(E_norm)
    if Omega is not None:
        # Enforce exact skew-symmetry for Omega
        Omega_skew = 0.5 * (Omega - Omega.T)
        H = H + 1j * alpha_vort * Omega_skew
    return 0.5 * (H + H.conj().T)
