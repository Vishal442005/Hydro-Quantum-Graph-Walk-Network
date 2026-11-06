# Table 7: Computational Complexity, Parameters, FLOPs, and Hardware Latency

| Model Paradigm | Time Complexity | Parameters (k) | FLOPs (M) | CPU Latency (i7-12700H) | Edge GPU Latency (Jetson/RTX) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Dense 1D-CNN *(Schirrmeister 2017)* | $\mathcal{O}(M \cdot L \cdot K)$ | 142.5 k | 18.4 M | .68$ ms | .18$ ms |
| Dynamic ST-GCN *(Hajisafi 2024)* | $\mathcal{O}(M^2 L + M d^2)$ | 284.1 k | 36.8 M | .52$ ms | .35$ ms |
| Unitary CTQW-GNN *(Zhan 2026)* | $\mathcal{O}(M^3 + M^2 d)$ | 198.0 k | 52.1 M | .98$ ms | .78$ ms |
| Dissipative OQW *(Linck 2026)* | $\mathcal{O}(M^3 K_{\text{lind}})$ | 215.4 k | 61.4 M | .29$ ms | .65$ ms |
| Riemannian Covariance *(Barachant 2012)* | $\mathcal{O}(M^3)$ (Log-Euclidean) | 12.8 k | 4.2 M | **.12$ ms** | **.08$ ms** |
| **Proposed HQGN (Strang)** | $\mathcal{O}(M^2 + M K_{\text{krylov}})$ | **86.2 k** | **24.6 M** | **.84$ ms** | **.72$ ms** |
