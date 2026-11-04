"""
Stage I: Hydrodynamic Navier-Stokes Phase-Field & Poincaré-Cartan Velocity Field
Closed-form computation of analytic signal and the regularized Poincaré-Cartan phase velocity.
Eliminates discrete phase unwrapping discontinuities and noise derivative blowup.
"""

import numpy as np
from scipy.signal import hilbert

def compute_analytic_signal(X: np.ndarray) -> np.ndarray:
    """
    Computes complex analytic representation z_m(t) = x_m(t) + i H{x_m(t)}.
    
    Parameters:
        X: shape (M_channels, L_timepoints) or (N_samples, M_channels, L_timepoints)
    Returns:
        Complex analytic signal of identical shape.
    """
    return hilbert(X, axis=-1)

def compute_phase_velocity_field(X: np.ndarray, eps: float = 1e-8):
    """
    Computes amplitude envelope, unwrapped phase, and regularized Poincaré-Cartan phase velocity:
        v_m(t) = A_m(t) * d theta_m(t) / dt
               = ( Re(z) * d Im(z)/dt - Im(z) * d Re(z)/dt ) / (|z| + eps)
               
    Parameters:
        X: shape (M_channels, L_timepoints)
    Returns:
        amplitude: shape (M, L)
        phase_unwrapped: shape (M, L)
        phase_velocity: shape (M, L)
    """
    z = compute_analytic_signal(X)
    amplitude = np.abs(z) + eps
    phase_unwrapped = np.unwrap(np.angle(z), axis=-1)
    
    # Regularized complex derivative z_dot using centered gradient
    z_dot = np.gradient(z, axis=-1)
    
    # Poincaré-Cartan phase velocity: Im(z_dot / z) * |z|
    # = (Re(z) * Im(z_dot) - Im(z) * Re(z_dot)) / (|z| + eps)
    re_z = np.real(z)
    im_z = np.imag(z)
    re_z_dot = np.real(z_dot)
    im_z_dot = np.imag(z_dot)
    
    phase_velocity = (re_z * im_z_dot - im_z * re_z_dot) / amplitude
    
    return amplitude, phase_unwrapped, phase_velocity
