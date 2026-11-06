# Table 5: Multi-Condition Physical Stress Benchmark: Noise Pathologies & Sensor Array Scaling (Macro-F1 %)

| Model Paradigm | Clean ECoG | Colored Pink Noise (/f$, +5 dB) | Power-Line Hum (50 Hz, Ampl=1.0) | Impulsive Cauchy Spikes (=5\%$) | Array Scaling (=4$ Ch) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Dense 1D-CNN *(Schirrmeister 2017)* | .26\%$ | .70\%$ | .30\%$ | .40\%$ | **.60\%$** |
| 1D-ResNet-18 *(Wang 2017)* | .79\%$ | .20\%$ | .10\%$ | .50\%$ | .10\%$ |
| Dynamic ST-GCN *(Hajisafi 2024)* | .92\%$ | .90\%$ | .50\%$ | .10\%$ | .40\%$ |
| PatchTST / ST-Transformer *(Nie 2023)* | .58\%$ | .10\%$ | .80\%$ | .20\%$ | .80\%$ |
| Unitary CTQW-GNN *(Zhan 2026)* | .50\%$ | .00\%$ | .10\%$ | .40\%$ | .10\%$ |
| Dissipative OQW *(Linck 2026)* | .83\%$ | .00\%$ | .80\%$ | .10\%$ | .80\%$ |
| Riemannian Covariance *(Barachant 2012)* | .06\%$ | .30\%$ | .60\%$ | .50\%$ | .06\%$ |
| **Proposed HQGN (Raw)** | **.61\%$** | **.00\%$** | **.70\%$** | .40\%$ *(Failure Mode)* | .20\%$ *(Boundary)* |
| **Proposed HQGN + Robust Clipping** | **.61\%$** | **.00\%$** | **.70\%$** | **.60\%$** | .20\%$ |
