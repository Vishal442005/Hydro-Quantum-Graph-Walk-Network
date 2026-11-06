# Table 4: 6-Variant Architectural Component Ablation on Clinical ECoG Telemetry

| Ablation Variant | Macro-F1 (%) | Accuracy (%) | Cohen's $\kappa$ | Architectural Role |
| :--- | :---: | :---: | :---: | :--- |
| **A1 (Classical Pearson Base)** | .92 \pm 2.58$ | .95 \pm 2.60$ | .798 \pm 0.045$ | Static symmetric graph, vulnerable to noise |
| **A2 (Hydro Graph Only)** | .21 \pm 3.54$ | .24 \pm 3.53$ | .825 \pm 0.071$ | Vorticity-aware, lacks quantum interference |
| **A3 (Unitary QW, $\alpha=0$)** | .55 \pm 4.57$ | .57 \pm 4.57$ | .771 \pm 0.091$ | Coherent wave interference, lacks noise sink |
| **A4 (Quantum Only, Pearson)** | .53 \pm 3.41$ | .60 \pm 3.38$ | .772 \pm 0.068$ | Open quantum walk on static correlation graph |
| **A5 (No Entropy Collapse)** | .54 \pm 3.43$ | .60 \pm 3.38$ | .772 \pm 0.068$ | Density probabilities only, no QKL/Bures |
| **A6: Proposed HQGN** | **.61 \pm 3.26$** | **.62 \pm 3.26$** | **.852 \pm 0.065$** | Complete tri-domain synthesis (Hydro + Quantum + Thermo) |
