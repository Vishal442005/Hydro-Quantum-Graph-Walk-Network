import json

with open('results/real_multidomain_comparative_results.json') as f:
    d = json.load(f)

print('Top level keys:', list(d.keys()))
for dom in ['domain1_clinical_ecog', 'domain2_bci_eeg', 'domain3_wireless_rf']:
    if dom in d:
        print(f'\n=== {dom} ===')
        for model, res in d[dom].items():
            acc = res.get('accuracy_mean', 'N/A')
            f1 = res.get('f1_mean', 'N/A')
            sens = res.get('sensitivity_mean', 'N/A')
            spec = res.get('specificity_mean', 'N/A')
            auc = res.get('auroc_mean', 'N/A')
            print(f'  {model:32s}: Acc={acc}% F1={f1}% Sens={sens}% Spec={spec}% AUC={auc}%')

if 'ablation_study_d1' in d:
    print('\n=== ablation_study_d1 ===')
    for var, res in d['ablation_study_d1'].items():
        print(f'  {var:32s}: Acc={res.get("accuracy_mean")}% F1={res.get("f1_mean")}%')
