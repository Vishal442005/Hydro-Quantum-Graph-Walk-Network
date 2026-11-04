"""
Stage II: Stochastic Lindblad Quantum Walk (SLQW) Evolution
Evolves density matrix rho(t) under GKSL master equation using Strang-splitting.
Includes spatial phase-amplitude modulated state initialization:
    psi_{0, m} = sqrt(E_m) * exp(i * theta_m(0)) / sqrt(sum E_k)
Ensures probability amplitude correctly reflects channel power while spatial phases
activate the unitary commutator [H, rho] to drive true quantum wave propagation.
"""

import numpy as np
from scipy.linalg import expm, eigh
from .integrators import strang_splitting_step

def initialize_density_matrix(energies: np.ndarray, z: np.ndarray = None) -> np.ndarray:
    """
    Initializes pure state density matrix rho(0) = |psi_0><psi_0|.
    Encodes both channel power and instantaneous spatial phase:
        psi_{0, m} = sqrt(E_m) * exp(i * theta_m(0)) / sqrt(sum E_k)
    This guarantees:
      1. Diagonal probability |psi_{0, m}|^2 is proportional to channel signal energy.
      2. Relative phase differences exp(i (theta_m - theta_n)) activate [H, rho] != 0.
      3. Wave propagation dynamically probes the hydrodynamic graph topology.
    """
    M = len(energies)
    energy_sum = np.sum(energies)
    if energy_sum < 1e-12:
        return np.eye(M, dtype=complex) / M
        
    amp = np.sqrt(energies) / np.sqrt(energy_sum)
    
    if z is not None:
        # Extract initial spatial phase angle of each channel at t=0
        phase_0 = np.angle(z[:, 0])
        psi_0 = amp * np.exp(1j * phase_0)
    else:
        psi_0 = amp.astype(complex)
        
    norm_psi = np.linalg.norm(psi_0)
    if norm_psi > 1e-12:
        psi_0 = psi_0 / norm_psi
    return np.outer(psi_0, psi_0.conj())

def evolve_lindblad_quantum_walk(
    H: np.ndarray,
    Laplacian: np.ndarray,
    energies: np.ndarray,
    z: np.ndarray = None,
    walk_time: float = 1.5,
    dt_steps: int = 15,
    alpha_dephase: float = 0.15,
    use_quantum_walk: bool = True,
    adaptive_annealing: bool = False,
    method: str = "strang"
) -> np.ndarray:
    """
    Evolves quantum density matrix rho(t) over interval [0, walk_time].
    
    Parameters:
        H: System Hamiltonian (M, M)
        Laplacian: Hydrodynamic graph Laplacian (M, M)
        energies: Channel energies (M,)
        z: Complex analytic signal (M, L)
        walk_time: Total propagation horizon tau
        dt_steps: Number of integration time steps
        alpha_dephase: Lindblad dephasing coupling alpha in [0, 1]
        use_quantum_walk: If False, executes classical Markov diffusion baseline
        adaptive_annealing: If True, uses dynamic alpha(t) schedule
        method: "strang" (default, exact CPTP) or "rk4"
    Returns:
        rho(tau): Evolved density matrix of shape (M, M)
    """
    M = Laplacian.shape[0]
    
    # Ablation Baseline: Classical Markov Diffusive Random Walk
    if not use_quantum_walk:
        P_diff = expm(-walk_time * Laplacian)
        p0 = energies / (np.sum(energies) + 1e-12)
        p_classical = P_diff @ p0
        p_classical = np.maximum(p_classical, 1e-12)
        p_classical /= np.sum(p_classical)
        return np.diag(p_classical).astype(complex)
        
    rho = initialize_density_matrix(energies, z=z)
    dt = walk_time / max(dt_steps, 1)
    
    # Pre-diagonalize H once for Strang splitting: O(M^3) once, then O(M^2) per step
    eigvals_H, eigvecs_H = eigh(H)
    factor_u = 1.0 - alpha_dephase
    
    for step in range(dt_steps):
        t_current = step * dt
        if adaptive_annealing:
            # Quantum annealing: start coherent, finish dissipative
            alpha_curr = 0.01 + (alpha_dephase - 0.01) * (0.5 * (1.0 - np.cos(np.pi * t_current / walk_time)))
            factor_u = 1.0 - alpha_curr
        else:
            alpha_curr = alpha_dephase
            
        rho = strang_splitting_step(
            rho=rho,
            eigvals_H=eigvals_H,
            eigvecs_H=eigvecs_H,
            alpha=alpha_curr,
            dt=dt,
            factor_unitary=factor_u
        )
        
    return rho
