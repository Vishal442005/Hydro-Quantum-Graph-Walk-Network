# Hydro-Quantum Graph Network (HQGN)
[![DOI](https://zenodo.org/badge/1445760922.svg)](https://zenodo.org/badge/latestdoi/1445760922)
[![GitHub Release](https://img.shields.io/github/v/release/Vishal442005/Hydro-Quantum-Graph-Walk-Network?color=blue&label=Release)](https://github.com/Vishal442005/Hydro-Quantum-Graph-Walk-Network/releases/tag/v1.1.0)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0006--0552--8551-green.svg)](https://orcid.org/0009-0006-0552-8551)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-3_Domains_Empirical-success)](results/real_multidomain_comparative_results.json)

> **Official Implementation & Research Artifact for:**  
> *Hydro-Quantum Graph Network (HQGN): Physics-Informed Open Quantum Dynamics and Thermodynamic Invariant Collapse for Robust Spatio-Temporal Telemetry Sensing*  
> **Author:** Vishal Chandel (Department of Electronics and Communication Engineering)  
> **ORCID:** [`0009-0006-0552-8551`](https://orcid.org/0009-0006-0552-8551)  
> **Contact:** `vishal442005@outlook.com`  
> **Target Journal:** *Engineering Applications of Artificial Intelligence* (Elsevier)

---

## 1. Overview & Theoretical Architecture

`hqgwn` is a physics-grounded machine learning framework designed for multi-channel spatio-temporal telemetry classification across three real physical regimes: human clinical intracranial ECoG epilepsy telemetry, sensorimotor brain-computer interfaces (BCIs), and non-stationary cognitive radio wireless RF sensing.

```
Multi-Channel Time-Series X(t) in R^{M x L}
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage I: 2D Navier-Stokes Poincaré-Cartan Phase-Vorticity   │
│   • Analytic phase signal: z_m(t) = A_m(t) exp(i theta_m(t))│
│   • Fluid velocity field: u(x, y, t) = grad theta(x, y, t)  │
│   • Antisymmetric curl vorticity tensor: Omega_ij           │
│   • Chiral Hamiltonian: H = gamma L_S + i alpha Omega + diag│
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage II: Open Lindblad Stochastic Quantum Walk (SLQW)      │
│   • Master Equation: d rho/d tau = -i [H, rho] + D[rho]     │
│   • Unconditional trace preservation: Tr(rho(tau)) = 1.0     │
│   • 2nd-Order Symmetric Strang Operator Splitting           │
│   • Complete Positivity & Lindbladian noise dissipation     │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage III: Quantum Thermodynamic Invariant Extraction       │
│   • Von Neumann Entanglement Entropy: S_vN = -Tr(rho ln rho)│
│   • Quantum Purity: gamma = Tr(rho^2), Linear Entropy       │
│   • Off-diagonal l1-norm Quantum Coherence                  │
│   • Umegaki Quantum Relative Entropy against class states   │
│   • Riemannian Bures-Wasserstein Metric Coordinates         │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
  Ensemble / Downstream Classifier -> Target Telemetry Class
```

---

## 2. Multi-Domain Cross-Paradigm Benchmarks

HQGN was evaluated against five leading comparative paradigms from 2020–2026 literature across **3 distinct physical telemetry domains** under **5-Fold Stratified Cross-Validation**:

| Model Paradigm | Domain 1: Clinical ECoG (16 Ch) | Domain 2: Scalp BCI (16 Ch) | Domain 3: Multipath RF (16 Ch) | Mean Latency |
| :--- | :---: | :---: | :---: | :---: |
| **Dense 1D-CNN** *(Schirrmeister et al. 2017)* | $91.26 \pm 2.72\%$ | $59.59 \pm 5.74\%$ | $33.24 \pm 5.48\%$ | $0.71$ ms |
| **Dynamic ST-GCN** *(Hajisafi et al. 2024)* | $89.92 \pm 2.58\%$ | $46.15 \pm 9.74\%$ | $40.40 \pm 4.20\%$ | $1.48$ ms |
| **Unitary CTQW-GNN** *(Zhan et al. 2026)* | $88.50 \pm 3.35\%$ | $54.65 \pm 4.17\%$ | $38.96 \pm 4.53\%$ | $3.23$ ms |
| **Dissipative OQW** *(Linck et al. 2026)* | $87.83 \pm 1.78\%$ | $50.71 \pm 4.30\%$ | $43.01 \pm 2.31\%$ | $2.33$ ms |
| **Riemannian Covariance** *(Barachant et al. 2012)* | $34.06 \pm 0.65\%$ | $71.53 \pm 4.12\%$ | $25.96 \pm 5.63\%$ | **$0.12$ ms** |
| **Proposed HQGN** | **$92.61 \pm 3.26\%$** | **$83.98 \pm 4.25\%$** | **$58.65 \pm 3.61\%$** | $2.98$ ms |

---

## 3. Head-to-Head Clinical Seizure Telemetry Benchmark

Evaluated on real human intracranial ECoG recordings ([MNE Epilepsy ECoG](C:/Users/Visha/mne_data/MNE-epilepsy-ecog-data), patient `sub-pt1`, 149 epochs of 1.5s):

| Model Paradigm | Accuracy (%) | Macro-F1 (%) | Sensitivity (%) | Specificity (%) | AUROC (%) | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Dense 1D-CNN *(Schirrmeister et al. 2017)* | $91.31 \pm 2.63$ | $91.26 \pm 2.72$ | $87.25 \pm 4.81$ | $94.67 \pm 3.67$ | $95.02 \pm 2.71$ | $0.68$ ms |
| Dynamic ST-GCN *(Hajisafi et al. 2024)* | $89.95 \pm 2.60$ | $89.92 \pm 2.58$ | $86.58 \pm 4.47$ | $93.48 \pm 3.30$ | $94.87 \pm 2.64$ | $1.52$ ms |
| Unitary CTQW-GNN *(Zhan et al. 2026)* | $88.54 \pm 3.35$ | $88.50 \pm 3.35$ | $84.42 \pm 5.41$ | $93.38 \pm 4.32$ | $94.21 \pm 2.90$ | $2.98$ ms |
| Dissipative OQW *(Linck et al. 2026)* | $87.91 \pm 1.72$ | $87.83 \pm 1.78$ | $86.92 \pm 7.39$ | $88.95 \pm 7.15$ | $93.47 \pm 2.44$ | $2.29$ ms |
| Riemannian Covariance *(Barachant et al. 2012)* | $51.68 \pm 1.49$ | $34.06 \pm 0.65$ | $100.00 \pm 0.00$ | $0.00 \pm 0.00$ | $24.33 \pm 21.14$ | **$0.12$ ms** |
| **Proposed HQGN** | **$92.62 \pm 3.26$** | **$92.61 \pm 3.26$** | **$89.58 \pm 3.25$** | **$95.90 \pm 5.38$** | **$96.85 \pm 1.85$** | $2.84$ ms |

---

## 4. Architectural Ablation Benchmark

| Ablation Variant | Macro-F1 (%) | Accuracy (%) | Cohen's $\kappa$ | Role |
| :--- | :---: | :---: | :---: | :--- |
| **A1 (Classical Pearson Base)** | $89.92 \pm 2.58$ | $89.95 \pm 2.60$ | $0.798 \pm 0.045$ | Static symmetric correlation graph |
| **A2 (Hydro Graph Only)** | $91.21 \pm 3.54$ | $91.24 \pm 3.53$ | $0.825 \pm 0.071$ | Vorticity-aware, classical heat diffusion |
| **A3 (Unitary QW, $\alpha=0$)** | $88.55 \pm 4.57$ | $88.57 \pm 4.57$ | $0.771 \pm 0.091$ | Coherent quantum walk without dissipation |
| **A4 (Quantum Only, Pearson)** | $88.53 \pm 3.41$ | $88.60 \pm 3.38$ | $0.772 \pm 0.068$ | Lindblad quantum walk on static graph |
| **A5 (No Entropy Collapse)** | $88.54 \pm 3.43$ | $88.60 \pm 3.38$ | $0.772 \pm 0.068$ | Diagonal probabilities only, no QKL/Bures |
| **A6 (Proposed HQGN)** | **$92.61 \pm 3.26$** | **$92.62 \pm 3.26$** | **$0.852 \pm 0.065$** | Full tri-domain physical synthesis |

---

## 5. Key Theoretical Theorems

1. **Theorem 1 (Complete Positivity & Strict Trace Preservation)**: Unconditionally guarantees $\operatorname{Tr}(\rho(\tau)) = 1$ and $\rho(\tau) \succeq 0$ for all $\tau \ge 0$ under Strang splitting integration.
2. **Theorem 2 (Asymptotic Noise Dissipation)**: Proves that off-diagonal stochastic noise perturbations contract exponentially: $\|\delta\rho_{mn}(\tau)\| \le \|\delta\rho_{mn}(0)\| e^{-\gamma_d \tau}$.
3. **Theorem 3 (Bures Non-Expansiveness)**: Guarantees contractive geodesic geometry: $d_{\text{Bures}}(\Phi_\tau(\rho_1), \Phi_\tau(\rho_2)) \le d_{\text{Bures}}(\rho_1, \rho_2)$.
4. **Theorem 4 (Non-Collapsing Dirichlet Energy)**: Proves $\inf_{\tau > 0} E(\rho(\tau)) \ge E_{\min} > 0$ ($E(5.0) = 0.023288$), rigorously preventing GNN oversmoothing.

---

## 6. Directory Structure

```
├── Elseiver_EAAI/                   # Elsevier EAAI Double-Blind Submission Package
│   ├── Manuscript_without_author_details.docx  # Blinded manuscript (286 native equations, 11 figures, 5 tables)
│   ├── Title_page_with_author_details.docx     # Dedicated Title page with author, affiliations & statements
│   ├── Cover_Letter_EAAI.docx                  # Formal Cover Letter to EAAI Editor-in-Chief
│   ├── Highlights.docx                         # 5 core highlights (<= 85 chars each)
│   ├── CRediT_Authorship_Statement.docx        # Formal CRediT author statement
│   ├── Declaration_of_Competing_Interest.docx  # Signed declaration of interest
│   └── README_Submission_Checklist.md          # Step-by-step Editorial Manager upload mapping
├── figures/                         # 11 Publication-grade 300 DPI figures
│   ├── fig1_hydrodynamic_adjacency.{png,pdf}
│   ├── fig2_density_matrix_3d.{png,pdf}
│   ├── fig3_entropy_purity_dynamics.{png,pdf}
│   ├── fig4_strang_trace_stability.{png,pdf}
│   ├── fig5_qfim_attribution.{png,pdf}
│   ├── fig6_snr_robustness_curve.{png,pdf}
│   ├── fig7_ablation_comparison.{png,pdf}
│   ├── fig8_multidomain_benchmarks.{png,pdf}
│   ├── fig9_adversarial_security.{png,pdf}
│   ├── fig10_dirichlet_oversmoothing.{png,pdf}
│   └── fig11_bures_geodesic_recourse.{png,pdf}
├── hqgwn/                           # Core Python Package
│   ├── hydro/                       # Stage I: Poincaré-Cartan phase vorticity
│   ├── quantum/                     # Stage II: Lindblad Master Equation & Strang splitting
│   ├── thermo/                      # Stage III: Quantum Relative Entropy & Bures Metric
│   ├── models/                      # Scikit-learn compliant HQGN Classifier
│   └── xai/                         # Explainable AI: QFIM & Bures Counterfactuals
├── experiments/                     # Multi-domain benchmarking scripts
├── results/                         # Empirical JSON results (ECoG, BCI, RF, SNR, Adversarial)
└── pyproject.toml                   # uv / pip package configuration
```

---

## 7. Quickstart & Installation

```bash
# Clone the repository
git clone https://github.com/Vishal442005/Hydro-Quantum-Graph-Walk-Network.git
cd Hydro-Quantum-Graph-Walk-Network

# Install dependencies via uv or pip
pip install -e .

# Run unit test suite
pytest tests/

# Execute 3-domain real benchmark suite
python experiments/run_real_multidomain_benchmarks.py

# Regenerate all 11 publication figures
python scripts/generate_all_figures.py

# Regenerate Elsevier EAAI submission docx package
python scripts/generate_elsevier_pr_docx.py
python scripts/generate_eaai_submission_package.py
```

---

## 8. Citation

```bibtex
@article{chandel2026hydroquantum,
  author    = {Chandel, Vishal},
  title     = {Hydro-Quantum Graph Network ({HQGN}): Physics-Informed Open Quantum Dynamics and Thermodynamic Invariant Collapse for Robust Spatio-Temporal Telemetry Sensing},
  journal   = {Engineering Applications of Artificial Intelligence},
  year      = {2026},
  note      = {Under review}
}
```

---
*Maintained by Vishal Chandel (`vishal442005@outlook.com`). ORCID: [0009-0006-0552-8551](https://orcid.org/0009-0006-0552-8551). Apache 2.0 License.*
