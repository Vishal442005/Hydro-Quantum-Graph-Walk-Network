# Table 8: Expanded Comparative Baseline Suite across 2017-2026 Literature (Clinical ECoG 5-Fold CV)

| Model Paradigm | Architectural Class | Accuracy (%) | Macro-F1 (%) | AUROC (%) | Time Complexity | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Dense 1D-CNN *(Schirrmeister et al. 2017)* | Deep Temporal Convolutional | .31 \pm 2.63$ | .26 \pm 2.72$ | .02 \pm 2.71$ | $\mathcal{O}(M L K)$ | .68$ ms |
| 1D-ResNet-18 *(Wang et al. 2017)* | Deep Residual Temporal ConvNet | .84 \pm 2.45$ | .79 \pm 2.50$ | .48 \pm 2.30$ | $\mathcal{O}(M L K)$ | .94$ ms |
| Dynamic ST-GCN *(Hajisafi et al. 2024)* | Spatio-Temporal Graph ConvNet | .95 \pm 2.60$ | .92 \pm 2.58$ | .87 \pm 2.64$ | $\mathcal{O}(M^2 L + M d^2)$ | .52$ ms |
| PatchTST / ST-Transformer *(Nie et al. 2023)* | Spatio-Temporal Patch Attention | .62 \pm 2.85$ | .58 \pm 2.90$ | .12 \pm 2.45$ | $\mathcal{O}((L/P)^2 + M^2)$ | .15$ ms |
| Magnetic GCN / MagNet *(Zhang et al. 2021)* | Complex Hermitian Directional GCN | .28 \pm 2.92$ | .24 \pm 2.95$ | .95 \pm 2.55$ | $\mathcal{O}(M^2 d)$ | .85$ ms |
| Unitary CTQW-GNN *(Zhan et al. 2026)* | Closed Continuous Quantum Walk | .54 \pm 3.35$ | .50 \pm 3.35$ | .21 \pm 2.90$ | $\mathcal{O}(M^3 + M^2 d)$ | .98$ ms |
| Dissipative OQW *(Linck et al. 2026)* | Markovian Open Quantum Walk | .91 \pm 1.72$ | .83 \pm 1.78$ | .47 \pm 2.44$ | $\mathcal{O}(M^3 K_{\text{lind}})$ | .29$ ms |
| Riemannian Covariance *(Barachant 2012)* | Information Geometry Tangent Space | .68 \pm 1.49$ | .06 \pm 0.65$ | .33 \pm 21.14$ | $\mathcal{O}(M^3)$ | **.12$ ms** |
| **Proposed HQGN** | **Physics-Informed Fluid-Quantum** | **.62 \pm 3.26$** | **.61 \pm 3.26$** | **.85 \pm 1.85$** | $\mathcal{O}(M^2 + M K_{\text{kry}})$ | .84$ ms |
