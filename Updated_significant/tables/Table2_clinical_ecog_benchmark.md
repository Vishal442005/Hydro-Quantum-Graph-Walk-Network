# Table 2: Real Human Clinical Intracranial ECoG Seizure Benchmark (5-Fold Stratified Cross-Validation)

| Model Paradigm | Accuracy (%) | Macro-F1 (%) | Sensitivity (%) | Specificity (%) | AUROC (%) | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Dense 1D-CNN *(Schirrmeister et al. 2017)* | .31 \pm 2.63$ | .26 \pm 2.72$ | .25 \pm 4.81$ | .67 \pm 3.67$ | .02 \pm 2.71$ | .68$ ms |
| Dynamic ST-GCN *(Hajisafi et al. 2024)* | .95 \pm 2.60$ | .92 \pm 2.58$ | .58 \pm 4.47$ | .48 \pm 3.30$ | .87 \pm 2.64$ | .52$ ms |
| Unitary CTQW-GNN *(Zhan et al. 2026)* | .54 \pm 3.35$ | .50 \pm 3.35$ | .42 \pm 5.41$ | .38 \pm 4.32$ | .21 \pm 2.90$ | .98$ ms |
| Dissipative OQW *(Linck et al. 2026)* | .91 \pm 1.72$ | .83 \pm 1.78$ | .92 \pm 7.39$ | .95 \pm 7.15$ | .47 \pm 2.44$ | .29$ ms |
| Riemannian Covariance *(Barachant et al. 2012)* | .68 \pm 1.49$ | .06 \pm 0.65$ | .00 \pm 0.00$ | .00 \pm 0.00$ | .33 \pm 21.14$ | **.12$ ms** |
| **Proposed HQGN** | **.62 \pm 3.26$** | **.61 \pm 3.26$** | **.58 \pm 3.25$** | **.90 \pm 5.38$** | **.85 \pm 1.85$** | .84$ ms |
