"""
Hydro-Quantum Graph Walk Network (HQGWN)
Core Package Initialization
"""

__version__ = "1.0.0"
__author__ = "Google DeepMind Advanced Agentic Research Team"

from .models.classifier import HydroQuantumGraphClassifier
from .hydro.laplacian import build_hydrodynamic_laplacian
from .quantum.lindblad import evolve_lindblad_quantum_walk
from .thermo.entropy import extract_quantum_invariants
from .thermo.qkl import compute_umegaki_qkl, compute_bures_distance

__all__ = [
    "HydroQuantumGraphClassifier",
    "build_hydrodynamic_laplacian",
    "evolve_lindblad_quantum_walk",
    "extract_quantum_invariants",
    "compute_umegaki_qkl",
    "compute_bures_distance",
]
