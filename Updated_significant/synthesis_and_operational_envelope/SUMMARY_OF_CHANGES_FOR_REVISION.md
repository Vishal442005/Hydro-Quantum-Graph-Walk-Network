# Point-by-Point Author Response & Revision Map

**Target Journal:** *Engineering Applications of Artificial Intelligence* (Elsevier)  
**Manuscript Title:** *Hydro-Quantum Graph Network (HQGN): Physics-Informed Open Quantum Dynamics and Thermodynamic Invariant Collapse for Robust Spatio-Temporal Telemetry Sensing*  
**Author:** Vishal Chandel (Department of Electronics and Communication Engineering)

---

### Editorial / Reviewer Point 1: Archival Value
> *"Although your manuscript falls within the aim and scope of this journal, it is being declined as the research results reported, in their present form, lack the archival value required by this Journal."*

**Author Action & Resolution:**
* Transformed the manuscript from a descriptive experimental report into an **archival-grade reference treatise**. We introduced:
  1. A formal mathematical and empirical analysis of the **operational envelope** of open quantum graph walks (`OPERATIONAL_ENVELOPE_AND_FAILURE_MODES.md`, Table 7).
  2. Four rigorous mathematical theorems establishing unconditional trace preservation, exponential noise contractivity, Bures non-expansiveness, and non-vanishing Dirichlet energy ($E(5.0) = 0.023288 > 0$).
  3. A formal system architect decision flowchart guiding engineers on when to deploy fluid-quantum walks versus classical 1D-CNN, ST-GCN, or Riemannian geometry.

---

### Editorial / Reviewer Point 2: Explanatory Details on Curves and Trends
> *"The performance evaluation campaign is rich in results which, however, are not commented on in a clear enough manner as required by the standards of this Journal. Many of the curves are only commented without sufficient explanatory details on the reason for their trend..."*

**Author Action & Resolution:**
* Conducted an exhaustive mechanistic overhaul across all figures (`EXPLANATORY_CURVE_DYNAMICS_GUIDE.md` and Section 6 & 7 in the manuscript). We provide detailed physical, mathematical, and algorithmic explanations for every curve:
  * **AWGN Noise Sweep (Fig. 7):** Explained the $0\text{ dB}$ transition where classical Markov diffusion collapses into node oversmoothing, contrasted with the Lindblad dissipator's off-diagonal exponential noise damping $\mathcal{O}(e^{-\gamma_d \tau})$ preserving $66.80\%$ Macro-F1 at $-10\text{ dB}$.
  * **Dirichlet Energy Floor (Fig. 8a):** Formally proved why the chiral Hamiltonian $i\alpha\Omega$ prevents projection onto the uniform consensus eigenspace $\operatorname{ker}(L_S) = \operatorname{span}\{\mathbf{1}\}$, stabilizing at $E(5.0) = 0.023288$.
  * **Sample-Efficiency Scaling (Fig. 8b):** Explained why HQGN retains $85.40\%$ at 20% data via physics-informed GKSL inductive constraints restricting hypothesis complexity on the compact manifold $\mathcal{S}(\mathcal{H})$.
  * **Brownian Phase Jitter (Fig. 11a):** Explained the non-linear stochastic resonance peak at $\sigma = 0.25\text{ rad}$ ($94.34\%$) through sub-threshold phase dithering.
  * **Channel Dropout Transition (Fig. 11b):** Explained the critical percolation threshold beyond $37.5\%$ dropped channels where spatial triangular simplices fragment, disrupting Green-Gauss curl computation.

---

### Editorial / Reviewer Point 3: Synthesis of Generality (When HQGN Performs Better vs. Worse)
> *"...and there is a lack of a synthesis that can testify to the generality of the results and that can make it clearer in which conditions the algorithm performs better and under which conditions worse."*

**Author Action & Resolution:**
* Created **Section 6.8 & Table 7** in the manuscript and the dedicated synthesis document `OPERATIONAL_ENVELOPE_AND_FAILURE_MODES.md`, systematically classifying operating conditions into three distinct regimes:
  * **Conditions Where HQGN Performs Better (Excels):**
    1. Rotating phase vortices and non-stationary circulation ($\Omega_{ij} \ne 0$).
    2. Harsh Gaussian and $1/f$ pink noise down to $-10\text{ dB}$ SNR.
    3. Severe electrode cross-talk and volume conduction.
    4. Low-sample training regimes ($N < 200$ epochs).
  * **Conditions Where HQGN Performs Worse (Boundary & Failure Modes):**
    1. *Spatially uncoupled telemetry:* Mitigated by spatial cross-correlation screening ($\bar{C} < 0.15$).
    2. *Ultra-sparse arrays ($M < 4$ channels):* Mitigated by state-space delay embedding.
    3. *Heavy-tailed Cauchy impulsive spikes:* Mitigated by robust median phase derivative clipping, recovering performance from $42.10\%$ back to $83.60\%$.
    4. *Sub-100 microsecond hard real-time budgets:* Addressed by Riemannian log-Euclidean tangent projections ($0.12\text{ ms}$).

---

### Editorial / Reviewer Point 4: Wider Range of Conditions & Literature Baselines
> *"Besides, the comparison with other solutions should definitely be conducted under a wider range of different conditions and for several other similar approaches available in the literature. More work is needed to substantiate the conclusions in your manuscript."*

**Author Action & Resolution:**
* Expanded the experimental campaign across **five new physical conditions** (Table 6, Figure 12):
  1. *Colored $1/f^\alpha$ pink/Brownian noise sweep* ($\alpha \in [0.0, 2.0]$).
  2. *50 Hz powerline harmonic hum injection* (amplitudes $0.0$ to $2.0$).
  3. *Impulsive Cauchy spike contamination* ($p \in [1\%, 10\%]$).
  4. *Sensor array density scaling* ($M \in [4, 8, 12, 16]$ channels).
  5. *Window duration scaling* ($L \in [0.5\text{s}, 3.0\text{s}]$).
* Expanded the comparative baseline suite to **nine modern literature models (2017–2026)** (Table 8):
  * Dense 1D-CNN (Schirrmeister et al. 2017)
  * 1D-ResNet-18 (Wang et al. 2017)
  * Dynamic ST-GCN (Hajisafi et al. 2024)
  * PatchTST / Spatio-Temporal Transformer (Nie et al. 2023)
  * Magnetic GCN / MagNet (Zhang et al. 2021)
  * Unitary CTQW-GNN (Zhan et al. 2026)
  * Dissipative Open Quantum Walk (Linck & Petruccione 2026)
  * Riemannian Covariance Tangent Space (Barachant et al. 2012)
  * Proposed HQGN (Physics-Informed Fluid-Quantum Synthesis)
