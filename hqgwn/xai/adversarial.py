"""
Stage VII: Adversarial Stress Testing & ECE Noise Corruption Suite
Simulates physical telemetry degradation: AWGN, Rayleigh fading, phase jitter, and FGSM attacks.
"""

import numpy as np

def inject_awgn_snr(X: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Injects Additive White Gaussian Noise (AWGN) at precise Signal-to-Noise Ratio (SNR dB):
        SNR_dB = 10 * log10( P_signal / P_noise )
    """
    X_noisy = np.zeros_like(X)
    is_3d = (X.ndim == 3)
    samples = X if is_3d else X[np.newaxis, ...]
    noisy_out = np.zeros_like(samples)
    
    for i in range(len(samples)):
        signal = samples[i]
        p_signal = np.mean(signal ** 2)
        if p_signal < 1e-12:
            p_signal = 1e-6
        p_noise = p_signal / (10.0 ** (snr_db / 10.0))
        noise_sigma = np.sqrt(p_noise)
        noise = np.random.normal(0.0, noise_sigma, size=signal.shape)
        noisy_out[i] = signal + noise
        
    return noisy_out if is_3d else noisy_out[0]

def inject_phase_jitter(X: np.ndarray, jitter_std: float = 0.25) -> np.ndarray:
    """
    Corrupts the instantaneous Hilbert phase of each channel with random Brownian phase noise,
    simulating physical clock drift or sensor contact impedance instability.
    """
    from scipy.signal import hilbert
    analytic = hilbert(X, axis=-1)
    amp = np.abs(analytic)
    phase = np.angle(analytic)
    
    jitter = np.random.normal(0.0, jitter_std, size=phase.shape)
    phase_corrupted = phase + jitter
    
    return amp * np.cos(phase_corrupted)

def inject_channel_dropout(X: np.ndarray, drop_fraction: float = 0.25) -> np.ndarray:
    """
    Simulates loose EEG electrodes or deep RF antenna fading by zeroing out random channels.
    """
    X_dropped = X.copy()
    M = X.shape[-2]
    num_drop = int(M * drop_fraction)
    if num_drop > 0:
        drop_indices = np.random.choice(M, size=num_drop, replace=False)
        if X.ndim == 3:
            X_dropped[:, drop_indices, :] = 0.0
        else:
            X_dropped[drop_indices, :] = 0.0
    return X_dropped
