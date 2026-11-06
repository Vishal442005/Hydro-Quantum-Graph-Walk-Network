# Archival Explanatory Guide: Physical Mechanisms & Dynamical Explanations for All Manuscript Curves

**Manuscript Title:** *Hydro-Quantum Graph Network (HQGN): Physics-Informed Open Quantum Dynamics and Thermodynamic Invariant Collapse for Robust Spatio-Temporal Telemetry Sensing*  
**Document Purpose:** Providing complete physical, mathematical, and algorithmic explanations for the behavior, inflection points, asymptotes, and sensitivity thresholds across all curves in Figures 1 through 12.

---

## 1. Figure 1: Navier-Stokes Phase Vorticity Vector Fields & Hydrodynamic Adjacency

### What is plotted:
* **Panel (a):** 2D Phase Velocity Vector Field $\mathbf{u}(x, y) = \nabla \theta(x, y)$ computed from the analytic Hilbert signal $z_m(t) = A_m(t) e^{i\theta_m(t)}$.
* **Panel (b):** Antisymmetric Vorticity Curl Tensor $\Omega_{ij} = \frac{\partial u_y}{\partial x} - \frac{\partial u_x}{\partial y}$.
* **Panel (c):** Vorticity-Modulated Hydrodynamic Graph Adjacency Matrix $W_{ij} = \exp\left(-\frac{\|\mathbf{u}_i - \mathbf{u}_j\|^2}{2\sigma_u^2}\right) + \beta \tanh(\Omega_{ij})$.

### Physical Mechanism Explaining Trends:
1. **Pre-Ictal vs. Ictal Vorticity Contrast:** In pre-ictal baseline states, the vector field $\mathbf{u}(x, y)$ exhibits low-amplitude, irrotational laminar wave propagation ($\nabla \times \mathbf{u} \approx 0$), resulting in symmetric adjacency matrices dominated by the Gaussian kinetic shear term.
2. **Phase Singularity During Seizure Discharge:** At the onset of clinical seizure activity, high-frequency synchronization produces rotating spiral wave rotors. The non-zero curl creates strong antisymmetric off-diagonal circulation peaks in $\Omega_{ij}$ (reaching magnitudes of $\pm 0.85$). The hyperbolic tangent $\tanh(\Omega_{ij})$ bounds the contribution to $[-1, 1]$, preventing runaway edge weights while breaking graph symmetry ($W_{ij} \ne W_{ji}$).

---

## 2. Figure 3: Thermodynamic Invariant Evolution Trajectories ($\tau \in [0, 5.0]$)

### What is plotted:
Temporal trajectories of three quantum thermodynamic observables during the Lindblad evolution:
1. **Von Neumann Entanglement Entropy:** $S_{vN}(\tau) = -\operatorname{Tr}(\rho(\tau) \ln \rho(\tau))$
2. **Quantum Purity:** $\gamma(\tau) = \operatorname{Tr}(\rho(\tau)^2)$
3. **$\ell_1$-Norm Quantum Coherence:** $\mathcal{C}(\tau) = \sum_{m \ne n} |\rho_{mn}(\tau)|$

### Physical Mechanism Explaining Trends:
* **Initial Transient Phase ($\tau \in [0, 0.5]$):** Purity starts at $\gamma(0) = 1.0$ (pure state $|\psi_0\rangle\langle\psi_0|$) and decreases rapidly, while Von Neumann entropy rises sharply from $S_{vN}(0) = 0$. This reflects the initial coupling with the Markovian thermal reservoir, where environmental degrees of freedom induce rapid decoherence.
* **Coherence Damping Plateau ($\tau \in [0.5, 2.5]$):** Quantum coherence $\mathcal{C}(\tau)$ exhibits damped oscillatory behavior. The oscillations arise from the unitary commutator $-i[H, \rho]$, where chiral vorticity generates cyclic state transitions between coupled channels. The exponential decay envelope is governed by the Lindblad dissipator:
  $$\mathcal{C}(\tau) \sim \mathcal{C}(0) e^{-\gamma_d \tau} \cos(\omega_{\text{vort}} \tau)$$
* **Asymptotic Thermalization ($\tau > 3.0$):** Purity and entropy plateau toward stable non-equilibrium steady-state (NESS) values. The system does not collapse to the trivial maximally mixed state $\frac{1}{M}I$ because the continuous non-Hermitian vorticity drive balances the dissipator sink, creating a persistent, class-discriminative thermodynamic signature.

---

## 3. Figure 4: Strang Splitting Integration Stability & Trace Preservation

### What is plotted:
Numerical deviation of the density matrix trace $|\operatorname{Tr}(\rho(\tau)) - 1.0|$ and positive semi-definiteness error $\min \operatorname{eig}(\rho(\tau))$ over 500 integration steps comparing:
1. 1st-Order Euler Forward Integration
2. 4th-Order Runge-Kutta (RK4)
3. 2nd-Order Symmetric Strang Splitting (Proposed)

### Mathematical Mechanism Explaining Trends:
* **Euler Divergence:** Forward Euler $\rho_{k+1} = \rho_k + \Delta \tau \mathcal{L}[\rho_k]$ fails to preserve the Lie algebra of completely positive trace-preserving (CPTP) maps. Trace error grows exponentially ($> 10^{-2}$ within 50 steps), and the smallest eigenvalue becomes negative, violating physical density matrix postulates.
* **RK4 Instability:** Although RK4 achieves 4th-order local truncation error, it is not a geometric or symplectic integrator. Under stiff Lindblad damping, numerical dissipation causes gradual trace drift ($|\operatorname{Tr}(\rho) - 1.0| \sim 10^{-5}$).
* **Strang Splitting Exactness:** By factorizing the generator into unitary rotation and dissipative contractive channels:
  $$\rho(\tau + \Delta \tau) = e^{-i H \frac{\Delta \tau}{2}} \left( \Phi_{\text{diss}} \left[ e^{-i H \frac{\Delta \tau}{2}} \rho(\tau) e^{i H \frac{\Delta \tau}{2}} \right] \right) e^{i H \frac{\Delta \tau}{2}}$$
  every step is an exact composition of Kraus operators. Consequently, trace deviation is identically zero to machine precision ($|\operatorname{Tr}(\rho) - 1.0| < 10^{-16}$), and complete positivity ($\rho \succeq 0$) is unconditionally guaranteed for all step sizes $\Delta \tau$.

---

## 4. Figure 7: Harsh AWGN Noise Robustness Curve (+20 dB down to -10 dB)

### What is plotted:
Macro-F1 (%) classification degradation of Proposed HQGN against Unitary CTQW, Classical Markov Diffusion, Welch PSD + SVM, and Riemannian Covariance as SNR sweeps from $+20\text{ dB}$ (clean) to $-10\text{ dB}$ (extreme noise).

### Mathematical Mechanism Explaining Trends:
1. **The High-SNR Zone ($+20\text{ dB}$ to $+10\text{ dB}$):**
   * *Trend:* All models maintain near-clean performance ($> 85\%$).
   * *Reason:* Signal power dominates ($P_{\text{signal}} \ge 10 \cdot P_{\text{noise}}$); phase differentials $\nabla \theta$ and spectral features are uncorrupted.
2. **The Inflection Zone ($+5\text{ dB}$ to $0\text{ dB}$):**
   * *Trend:* Classical Markov diffusion drops precipitously from $83.50\%$ down to $52.40\%$ (approaching the $50\%$ random guessing threshold), while Welch PSD drops to $50.80\%$. In contrast, Proposed HQGN gracefully degrades to $81.40\%$.
   * *Reason:* At $0\text{ dB}$, noise power equals signal power. Classical Markov convolution averages node signals linearly; when noise variance matches signal variance, spatial graph smoothing averages noise into node embeddings, obliterating feature discriminability. 
3. **The Severe Noise Zone ($-5\text{ dB}$ to $-10\text{ dB}$):**
   * *Trend:* Classical baselines completely collapse to single-class guessing ($34.60\%--35.80\%$), while HQGN sustains $66.80\%$ Macro-F1.
   * *Reason:* Theorem 2 proves that the Lindblad dissipator actively damps random phase fluctuations in the off-diagonal density matrix elements at rate $e^{-\gamma_d \tau}$. The unitary Hamiltonian preserves the topological phase rotor frequencies while the dissipative bath extinguishes incoherent Gaussian noise, preserving a $+31.0\%$ engineering margin over classical diffusion.

---

## 5. Figure 8: Theoretical & Empirical Diagnostics

### Panel (a): Dirichlet Energy Decay vs. Walk Horizon $\tau$
* **Trend:** Classical Markov diffusion decays exponentially toward zero ($E(5.0) = 4.18 \times 10^{-8} \approx 0$). HQGN decays initially but reaches a strictly positive lower bound ($E(5.0) = 0.023288$).
* **Mechanism:** Classical diffusion is governed by the symmetric positive semi-definite Laplacian $L_S$, whose smallest eigenvalue is $\lambda_1 = 0$ corresponding to the uniform eigenvector $\mathbf{1} = [1, 1, \dots, 1]^T$. Over continuous time, state trajectories project asymptotically onto $\operatorname{ker}(L_S)$, causing all node representations to become identical (GNN oversmoothing). In HQGN, the non-zero vorticity matrix $\Omega$ introduces an imaginary skew-symmetric term $i \alpha \Omega$ into the Hamiltonian. Skew-symmetric matrices possess purely imaginary eigenvalues $\pm i \omega_k$ ($\omega_k \ne 0$ for non-trivial circulation). This prevents projection onto the uniform state, lower-bounding the Dirichlet energy by $\frac{1}{2} \min_k \omega_k^2 > 0$ (Theorem 4).

### Panel (b): Limited-Data Scaling and Sample Efficiency
* **Trend:** When training data is reduced to $20\%$, HQGN maintains $85.40\%$ Macro-F1, whereas Unitary CTQW drops to $74.20\%$ and Classical Markov GNN collapses to $66.80\%$.
* **Mechanism:** Classical GNNs must learn both spatial propagation rules and feature filtering weights from scratch. In low-data regimes, high parameter variance leads to severe over-fitting. HQGN constrains spatial propagation via the Navier-Stokes hydrodynamic relation and restricts temporal state evolution to the Lindblad CPTP manifold. This inductive constraint reduces the effective Rademacher complexity of the model, enabling rapid generalization from few clinical examples.

### Panel (c): Walk Horizon Sensitivity $\tau \in [0.2, 10.0]$
* **Trend:** Macro-F1 exhibits a concave, unimodal curve: low at $\tau = 0.2$ ($78.2\%$), peaking across $\tau \in [1.0, 3.0]$ ($92.6\%$), and gradually degrading past $\tau = 6.0$ ($86.1\%$).
* **Mechanism:** 
  - *Short Horizons ($\tau < 0.5$):* The quantum walk has not had sufficient time to propagate across the graph diameter; multi-hop spatial dependencies remain uncaptured.
  - *Optimal Plateau ($\tau \in [1.0, 3.0]$):* Wave interference and dissipative noise filtering reach equilibrium, maximizing thermodynamic feature separation.
  - *Long Horizons ($\tau > 6.0$):* Prolonged exposure to the Lindblad dissipator causes progressive thermalization, gradually eroding transient quantum coherences $\rho_{mn}$ that encode fine-grained wave differences.

---

## 6. Figure 11: Adversarial Security, Phase Jitter, and Geometric Recourse

### Panel (a): Brownian Phase Jitter $\sigma \in [0, 1.5]$ rad
* **Trend:** Performance stays above $90.5\%$ up to $\sigma = 0.5\text{ rad}$, exhibits a subtle peak at $\sigma = 0.25\text{ rad}$ ($94.34\%$), and declines smoothly to $80.96\%$ at $\sigma = 1.5\text{ rad}$.
* **Mechanism (Stochastic Resonance):** The mild performance improvement at $\sigma = 0.25\text{ rad}$ is attributable to **stochastic resonance in non-linear phase networks**. Sub-threshold phase jitter acts as a dither signal, breaking marginal phase-locking ties and enhancing the non-linear contrast of the tanh vorticity activation. When jitter exceeds $\sigma = 0.5\text{ rad}$, phase coherence is disrupted, leading to monotonic degradation.

### Panel (b): Sensor Channel Dropout Sweep (0% to 62.5%)
* **Trend:** Graceful linear degradation from $92.61\%$ down to $71.89\%$ at $12.5\%$ dropped channels (2/16 leads disconnected), followed by a steeper decline past $37.5\%$ dropped channels.
* **Mechanism:** Calculating spatial curl $\nabla \times \mathbf{u}$ requires localized triangular simplices. Dropping 1 or 2 channels leaves the remaining 14 channels interconnected through alternative simplices, allowing the Krylov projection to reroute walk paths. When $>37.5\%$ of channels are dropped, the spatial graph fragments into disconnected components, preventing global hydrodynamic circulation.

### Panel (c): Bures-Wasserstein Geodesic Counterfactual Recourse
* **Trend:** The continuous trajectory connecting anomalous query state $\rho_0$ to healthy prototype $\rho_1$ yields direct Bures distance $d_{\text{Bures}} = 0.7500$ and geodesic length $L = 0.8033$ (efficiency $93.37\%$).
* **Mechanism:** Unlike naive linear interpolation $\rho(s) = (1-s)\rho_0 + s\rho_1$ (which produces intermediate states with distorted purity and non-physical entropy trajectories), the McCann optimal transport geodesic moves states along the Riemannian manifold of the quantum state space $\mathcal{S}(\mathcal{H})$, guaranteeing that every intermediate point $\rho(s)$ is a mathematically valid, positive semi-definite state with minimal action.

---

## 7. Figure 12: Expanded Multi-Condition Stress Benchmarks

### Panel (a): Colored $1/f^\alpha$ Noise Sweep ($\alpha = 0$ White to $\alpha = 2$ Brownian)
* **Trend:** Classical CNN and ST-GCN degrade rapidly as $\alpha \to 2$ ($74.5\% \to 62.9\%$ and $71.4\% \to 62.4\%$), whereas HQGN exhibits exceptional stability ($86.2\% \to 83.8\%$).
* **Mechanism:** Brownian noise ($\alpha = 2$) is concentrated at ultra-low frequencies, manifesting as non-stationary baseline wander. Baseline drift causes severe DC offsets in time-domain convolutions (1D-CNN) and Pearson correlations. In HQGN, the spatial curl operator $\nabla \times \mathbf{u}$ computes spatial spatial derivatives that act as an intrinsic high-pass filter, completely eliminating uniform low-frequency baseline drifts.

### Panel (b): 50 Hz Power-Line Harmonic Hum Contamination
* **Trend:** At hum amplitude $1.0\times$ signal standard deviation, 1D-CNN drops to $71.30\%$ and ST-GCN drops to $68.50\%$, whereas HQGN sustains $84.70\%$.
* **Mechanism:** Power-line hum injects an identical 50 Hz sinusoid across all recording channels. Classical models interpret this common-mode hum as strong inter-channel synchrony, distorting the correlation graph. In HQGN, identical hum across channels produces zero phase velocity differential ($\Delta u_{ij} = 0$) and zero curl ($\Omega_{ij} = 0$), rendering the hydrodynamic graph invariant to common-mode electromagnetic interference.

### Panel (c): Cauchy Impulsive Spikes & Robust Recovery
* **Trend:** Raw HQGN drops sharply to $42.10\%$ under $10\%$ Cauchy spike contamination, but robust median clipping restores performance to $71.80\%$.
* **Mechanism:** Heavy-tailed Cauchy spikes introduce unbounded outliers into the Hilbert transform phase derivative, creating artificial phase wraps. Applying the non-linear median threshold $\operatorname{clip}(\Delta u, -3\text{MAD}, +3\text{MAD})$ bounds the gradient operator, restoring numerical stability to the hydrodynamic kernel.

### Panel (d): Sensor Array Density Scaling ($M = 4$ to $16$)
* **Trend:** At $M = 4$, Dense 1D-CNN ($78.60\%$) outperforms HQGN ($74.20\%$). As $M$ increases to 16, HQGN overtakes all models ($92.61\%$).
* **Mechanism:** Hydrodynamic vorticity requires spatial dimensionality to compute curl. When $M=4$, finite difference approximations of spatial gradients suffer high truncation error. As $M$ grows, the spatial resolution of the 2D fluid manifold increases, allowing the continuous vorticity tensor to capture fine-grained wave dynamics that 1D-CNNs cannot resolve.
