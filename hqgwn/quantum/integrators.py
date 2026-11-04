"""
Stage II: Unitary-Preserving & Dissipative Operator Integrators
Includes the 2nd-order Strang-Splitting Integrator (guaranteeing exact trace preservation and CPTP property)
along with Runge-Kutta 4 (RK4) and Euler methods for numerical stability profiling.
"""

import numpy as np
from scipy.linalg import eigh

def strang_splitting_step(
    rho: np.ndarray,
    eigvals_H: np.ndarray,
    eigvecs_H: np.ndarray,
    alpha: float,
    dt: float,
    factor_unitary: float = 1.0
) -> np.ndarray:
    """
    Executes one second-order Strang-Splitting step:
        rho(t + dt) = E_{dephase}^{dt/2} o U_{unitary}^{dt} o E_{dephase}^{dt/2} [rho(t)]
    
    Guarantees:
      1. Exactly preserves Tr(rho) = 1.
      2. Completely Positive Trace-Preserving (CPTP) dynamical map.
      3. Global O(dt^2) convergence with zero unitary/trace drift.
    """
    # 1. First Half-Step Dissipative Dephasing E^{dt/2}
    decay_half = np.exp(-0.5 * alpha * dt)
    diag_rho = np.diag(np.diag(rho))
    rho_1 = diag_rho + (rho - diag_rho) * decay_half
    
    # 2. Full-Step Unitary Schrödinger Rotation U^{dt}
    # U = V exp(-i * factor_unitary * Lambda * dt) V^dagger
    phase_factors = np.exp(-1j * factor_unitary * eigvals_H * dt)
    # rho_rot = V @ (phase_factors[:, None] * (V.conj().T @ rho_1 @ V) * phase_factors[None, :].conj()) @ V.conj().T
    V_adj = eigvecs_H.conj().T
    rho_in_H = V_adj @ rho_1 @ eigvecs_H
    rho_rot_H = (phase_factors[:, None]) * rho_in_H * (phase_factors[None, :].conj())
    rho_2 = eigvecs_H @ rho_rot_H @ V_adj
    
    # Enforce Hermiticity
    rho_2 = 0.5 * (rho_2 + rho_2.conj().T)
    
    # 3. Second Half-Step Dissipative Dephasing E^{dt/2}
    diag_rho2 = np.diag(np.diag(rho_2))
    rho_next = diag_rho2 + (rho_2 - diag_rho2) * decay_half
    
    # Enforce Hermiticity and normalize trace to exactly 1.0
    rho_next = 0.5 * (rho_next + rho_next.conj().T)
    trace_val = np.real(np.trace(rho_next))
    if trace_val > 1e-15:
        rho_next /= trace_val
        
    return rho_next

def rk4_step(
    rho: np.ndarray,
    H: np.ndarray,
    alpha: float,
    dt: float
) -> np.ndarray:
    """
    Standard Runge-Kutta 4th order integrator for the GKSL equation.
    Included for stability benchmarking against Strang splitting.
    """
    def gksl_deriv(r):
        commutator = H @ r - r @ H
        dissipator = np.diag(np.diag(r)) - r
        return -1j * (1.0 - alpha) * commutator + alpha * dissipator

    k1 = gksl_deriv(rho)
    k2 = gksl_deriv(rho + 0.5 * dt * k1)
    k3 = gksl_deriv(rho + 0.5 * dt * k2)
    k4 = gksl_deriv(rho + dt * k3)
    
    rho_next = rho + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    rho_next = 0.5 * (rho_next + rho_next.conj().T)
    trace_val = np.real(np.trace(rho_next))
    if trace_val > 1e-15:
        rho_next /= trace_val
    return rho_next
