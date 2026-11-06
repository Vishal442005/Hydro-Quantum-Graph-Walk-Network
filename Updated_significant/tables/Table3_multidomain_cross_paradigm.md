# Table 3: Cross-Domain Multi-Telemetry Macro-F1 (%) Benchmarks (5-Fold Stratified Cross-Validation)

| Model Paradigm | Domain 1: Clinical ECoG (16 Ch) | Domain 2: Scalp BCI (16 Ch) | Domain 3: Multipath RF (16 Ch) | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| Dense 1D-CNN *(Schirrmeister et al. 2017)* | .26 \pm 2.72\%$ | .59 \pm 4.92\%$ | .24 \pm 5.48\%$ | .71$ ms |
| Dynamic ST-GCN *(Hajisafi et al. 2024)* | .92 \pm 2.58\%$ | .15 \pm 6.74\%$ | .40 \pm 4.20\%$ | .48$ ms |
| Unitary CTQW-GNN *(Zhan et al. 2026)* | .50 \pm 3.35\%$ | .65 \pm 4.17\%$ | .96 \pm 4.53\%$ | .23$ ms |
| Dissipative OQW *(Linck et al. 2026)* | .83 \pm 1.78\%$ | .71 \pm 4.30\%$ | .01 \pm 2.31\%$ | .33$ ms |
| Riemannian Covariance *(Barachant et al. 2012)* | .06 \pm 0.65\%$ | .53 \pm 5.35\%$ | .96 \pm 4.20\%$ | **.12$ ms** |
| **Proposed HQGN** | **.61 \pm 3.26\%$** | **.98 \pm 4.25\%$** | **.65 \pm 3.61\%$** | .98$ ms |
