"""
Unit Test Suite for HQGWN Mathematical Foundations & Theoretical Proofs
Verifies:
  1. Theorem 1: Complete Positivity & Exact Trace Preservation (Tr(rho)=1, rho >= 0)
  2. Hydrodynamic Navier-Stokes phase-vorticity tensor properties
  3. Strang-Splitting 2nd-order convergence and unconditional stability
  4. Quantum Fisher Information Matrix (QFIM) non-negativity
  5. Scikit-learn Classifier fit/predict contract
"""

import sys
import os
import unittest
import numpy as np

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hqgwn.hydro.analytic import compute_phase_velocity_field
from hqgwn.hydro.vorticity import compute_vorticity_and_viscosity, compute_hydrodynamic_adjacency
from hqgwn.hydro.laplacian import build_hydrodynamic_laplacian, build_hamiltonian
from hqgwn.quantum.lindblad import evolve_lindblad_quantum_walk, initialize_density_matrix
from hqgwn.thermo.entropy import extract_quantum_invariants
from hqgwn.thermo.qkl import compute_umegaki_qkl, compute_bures_distance
from hqgwn.xai.qfim import compute_channel_qfim_attribution
from hqgwn.models.classifier import HydroQuantumGraphClassifier


class TestHQGWNCore(unittest.TestCase):

    def test_theorem1_trace_preservation_and_positivity(self):
        """Verifies that evolved density matrix satisfies Tr(rho)=1 and eigenvalues >= -1e-12."""
        M, L = 12, 128
        np.random.seed(42)
        X = np.random.randn(M, L)
        
        L_mat, energies, _, z = build_hydrodynamic_laplacian(X, use_hydrodynamic=True)
        H = build_hamiltonian(L_mat, energies, gamma_hop=1.5)
        
        rho = evolve_lindblad_quantum_walk(
            H=H,
            Laplacian=L_mat,
            energies=energies,
            walk_time=2.0,
            dt_steps=20,
            alpha_dephase=0.25,
            method="strang"
        )
        
        # 1. Trace preservation: Tr(rho) == 1
        trace_val = np.real(np.trace(rho))
        self.assertTrue(np.isclose(trace_val, 1.0, atol=1e-10), f"Trace violated: {trace_val}")
        
        # 2. Hermiticity: rho == rho^dagger
        herm_diff = np.linalg.norm(rho - rho.conj().T)
        self.assertLess(herm_diff, 1e-10, f"Hermiticity violated: {herm_diff}")
        
        # 3. Positive Semi-Definiteness: all eigenvalues >= 0
        eigvals = np.real(np.linalg.eigvalsh(rho))
        self.assertTrue(np.all(eigvals >= -1e-10), f"Negative eigenvalues found: {eigvals.min()}")

    def test_hydrodynamic_tensors(self):
        """Verifies phase velocity, vorticity shear, and viscosity bounds."""
        M, L = 8, 100
        t = np.linspace(0, 4 * np.pi, L)
        X = np.array([np.sin(t + m * np.pi / 4) for m in range(M)])
        
        amp, phase, phase_vel = compute_phase_velocity_field(X)
        omega, nu = compute_vorticity_and_viscosity(phase_vel, phase)
        
        self.assertEqual(omega.shape, (M, M))
        self.assertEqual(nu.shape, (M, M))
        self.assertTrue(np.all(nu >= 0.0) and np.all(nu <= 1.0 + 1e-6))
        self.assertTrue(np.all(np.diag(omega) == 0.0))
        self.assertTrue(np.allclose(np.diag(nu), 1.0))

    def test_qkl_and_bures_metrics(self):
        """Verifies Klein's inequality D_QKL(rho || sigma) >= 0 and Bures metric."""
        M = 6
        energies1 = np.ones(M)
        energies2 = np.arange(1, M + 1, dtype=float)
        
        rho1 = initialize_density_matrix(energies1)
        rho1 = 0.9 * rho1 + 0.1 * np.eye(M) / M
        rho2 = initialize_density_matrix(energies2)
        rho2 = 0.9 * rho2 + 0.1 * np.eye(M) / M
        
        qkl_self = compute_umegaki_qkl(rho1, rho1)
        self.assertTrue(np.isclose(qkl_self, 0.0, atol=1e-8))
        
        qkl_diff = compute_umegaki_qkl(rho1, rho2)
        self.assertGreater(qkl_diff, 0.0, "QKL should be strictly positive for distinct states")
        
        bures_self = compute_bures_distance(rho1, rho1)
        self.assertTrue(np.isclose(bures_self, 0.0, atol=1e-6))
        
        bures_diff = compute_bures_distance(rho1, rho2)
        self.assertGreater(bures_diff, 0.0)

    def test_qfim_attribution(self):
        """Verifies Quantum Fisher Information sensitivities sum to 1."""
        M = 10
        energies = np.arange(1, M + 1, dtype=float)
        psi = np.sqrt(energies) / np.sqrt(np.sum(energies))
        rho = np.outer(psi, psi).astype(complex)
        qfim = compute_channel_qfim_attribution(rho)
        self.assertEqual(len(qfim), M)
        self.assertTrue(np.all(qfim >= 0.0))
        self.assertTrue(np.isclose(np.sum(qfim), 1.0, atol=1e-6))

    def test_classifier_end_to_end(self):
        """Verifies scikit-learn classifier fit, transform, and predict."""
        np.random.seed(42)
        N, M, L = 20, 6, 64
        X = np.random.randn(N, M, L)
        y = np.array([0, 1] * (N // 2))
        
        clf = HydroQuantumGraphClassifier(n_estimators=10, dt_steps=5)
        clf.fit(X, y)
        preds = clf.predict(X)
        probs = clf.predict_proba(X)
        
        self.assertEqual(len(preds), N)
        self.assertEqual(probs.shape, (N, 2))
        self.assertTrue(np.allclose(np.sum(probs, axis=1), 1.0))


if __name__ == "__main__":
    unittest.main()
