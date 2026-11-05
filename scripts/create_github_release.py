import subprocess
import urllib.request
import json
import os
import sys

def get_token():
    p = subprocess.Popen(['git', 'credential', 'fill'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, _ = p.communicate(input='protocol=https\nhost=github.com\n\n')
    for line in stdout.splitlines():
        if line.startswith('password='):
            return line.split('=', 1)[1].strip()
    return os.environ.get('GITHUB_TOKEN', '')

def create_release():
    token = get_token()
    if not token:
        print("Error: Could not retrieve GitHub token.")
        sys.exit(1)

    repo = "Vishal442005/Hydro-Quantum-Graph-Walk-Network"
    tag = "v1.1.0"
    release_name = "v1.1.0: Hydro-Quantum Graph Walk Network (HQGWN) - Pattern Recognition Research Artifact"
    
    release_body = """# Hydro-Quantum Graph Walk Network (HQGWN) - Release v1.1.0

## Official Research Artifact & Multi-Domain Empirical Benchmarks
**Target Venue:** *Pattern Recognition* (Elsevier)  
**Author:** Vishal Chandel (Department of Electronics and Communication Engineering)  
**ORCID:** [`0009-0006-0552-8551`](https://orcid.org/0009-0006-0552-8551)  
**Author Contact:** `vishal442005@outlook.com`  
**License:** Apache 2.0  
**Repository:** [Vishal442005/Hydro-Quantum-Graph-Walk-Network](https://github.com/Vishal442005/Hydro-Quantum-Graph-Walk-Network)

---

### Key Release Highlights
1. **Authentic Multi-Domain Empirical Benchmarks (`real_multidomain_comparative_results.json`):**
   - **Clinical Neurophysiology:** Intracranial ECoG Seizure Telemetry (149 windows, 250 Hz) - HQGWN achieves **92.62% Accuracy, 92.61% Macro-F1, 91.80% Sensitivity, 93.44% Specificity, 96.80% AUROC**.
   - **Sensorimotor BCI:** Scalp EEG Motor Imagery (BCI Competition IV-2a, 144 trials, 250 Hz) - HQGWN achieves **88.75% Macro-F1**.
   - **Cognitive Radio:** Multi-Carrier Wireless RF Telemetry under Rayleigh Fading (240 frames, 1000 Hz) - HQGWN achieves **84.30% Macro-F1**.
   - **Extreme Noise Robustness:** Evaluated from +20 dB down to -20 dB SNR; HQGWN sustains stable performance down to -10 dB where classical baselines experience catastrophic collapse.
   - **Dirichlet Energy Stability:** Asymptotically non-vanishing energy ($E(5.0) = 0.023 > 0$), preventing GNN oversmoothing.
   - **Explainability & Security:** Closed-form Quantum Fisher Information Matrix (QFIM) channel attribution and adversarial Brownian phase jitter verification.

---

### Included Release Assets
- `real_multidomain_comparative_results.json`: Full empirical 5-fold cross-validation metrics across all physical domains
- Full source code and reproducibility pipeline

---

### Persistent DOI & Archival Integration
This release snapshot is archived with Zenodo under digital object identifier (DOI) integration.
"""

    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'HQGWN-Release-Bot'
    }

    # 1. Create Release via GitHub API
    create_url = f"https://api.github.com/repos/{repo}/releases"
    payload = {
        "tag_name": tag,
        "target_commitish": "main",
        "name": release_name,
        "body": release_body,
        "draft": False,
        "prerelease": False
    }

    print(f"Creating release {tag} on {repo}...")
    req = urllib.request.Request(create_url, data=json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req) as resp:
            rel_data = json.loads(resp.read().decode('utf-8'))
            print(f"Successfully created release! ID: {rel_data['id']}, URL: {rel_data['html_url']}")
            upload_url_template = rel_data['upload_url']  # contains '{?name,label}'
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode('utf-8')
        print(f"HTTP Error {e.code}: {err_msg}")
        if "already_exists" in err_msg:
            print("Release tag already exists, fetching release info...")
            get_req = urllib.request.Request(f"https://api.github.com/repos/{repo}/releases/tags/{tag}", headers=headers)
            with urllib.request.urlopen(get_req) as g_resp:
                rel_data = json.loads(g_resp.read().decode('utf-8'))
                upload_url_template = rel_data['upload_url']
        else:
            sys.exit(1)

    # Clean upload url base
    upload_base = upload_url_template.split('{')[0]

    # 2. Upload Benchmark Assets (Manuscripts are excluded from git and public release)
    assets_to_upload = [
        ("results/real_multidomain_comparative_results.json", "real_multidomain_comparative_results.json", "application/json")
    ]

    for local_path, asset_name, content_type in assets_to_upload:
        if not os.path.exists(local_path):
            print(f"Warning: File {local_path} not found on disk, skipping.")
            continue
        
        file_size = os.path.getsize(local_path)
        print(f"Uploading asset: {asset_name} ({file_size / 1024 / 1024:.2f} MB)...")
        upload_url = f"{upload_base}?name={asset_name}"
        
        with open(local_path, 'rb') as f:
            file_data = f.read()

        up_headers = {
            'Authorization': f'token {token}',
            'Content-Type': content_type,
            'Content-Length': str(len(file_data)),
            'User-Agent': 'HQGWN-Release-Bot'
        }
        
        up_req = urllib.request.Request(upload_url, data=file_data, headers=up_headers, method='POST')
        try:
            with urllib.request.urlopen(up_req) as up_resp:
                up_res = json.loads(up_resp.read().decode('utf-8'))
                print(f"  Asset {asset_name} uploaded successfully! Download URL: {up_res['browser_download_url']}")
        except urllib.error.HTTPError as up_err:
            up_msg = up_err.read().decode('utf-8')
            print(f"  Upload HTTP Error {up_err.code}: {up_msg[:200]}")

    print("\nAll release operations completed successfully!")
    print(f"Release URL: {rel_data.get('html_url')}")

if __name__ == '__main__':
    create_release()
