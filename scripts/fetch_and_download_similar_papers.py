#!/usr/bin/env python3
"""
fetch_and_download_similar_papers.py

Finds and downloads the 15 most similar papers (published 2020-2026) across:
1. Continuous-time quantum walks on graphs & quantum graph neural networks
2. Open quantum systems & Lindblad master equations for learning / dynamics
3. Kuramoto phase synchronization, fluid vorticity & phase-locking graphs for EEG
4. Quantum information geometry, Von Neumann entropy & Bures-Wasserstein metrics
5. Dynamic spatio-temporal graph networks for biomedical EEG & telemetry

Uses official literature_search_arxiv utility scripts and saves PDFs into related_papers/.
"""

import os
import sys
import json
import subprocess
import re
import tempfile
import glob
import time

SEARCH_SCRIPT = r"C:\Users\Visha\.gemini\config\plugins\science\skills\literature_search_arxiv\scripts\search_arxiv.py"
DOWNLOAD_SCRIPT = r"C:\Users\Visha\.gemini\config\plugins\science\skills\literature_search_arxiv\scripts\download_paper.py"
OUTPUT_DIR = "related_papers"

# Clean stale locks
def clear_stale_locks():
    for pattern in ['*polite*', 'uv-*.lock']:
        for f in glob.glob(os.path.join(tempfile.gettempdir(), pattern)):
            try:
                os.remove(f)
            except Exception:
                pass

clear_stale_locks()

# Targeted queries across the 5 core pillars of HQGWN
QUERIES = [
    # Pillar 1: Quantum Walks on Graphs & Quantum Graph Networks
    'all:quantum AND all:walk AND (all:graph OR all:network)',
    'all:"continuous-time quantum walk" AND all:graph',
    'ti:"quantum walk" AND (all:machine OR all:learning OR all:graph)',
    
    # Pillar 2: Open Quantum Systems & Lindblad Master Equations
    'all:Lindblad AND (all:graph OR all:network OR all:learning)',
    'all:"dissipative quantum" AND all:walk',
    
    # Pillar 3: Kuramoto Phase Synchronization & Oscillators for EEG
    'all:Kuramoto AND all:EEG AND (all:graph OR all:network)',
    'all:"phase locking" AND all:EEG AND all:graph',
    
    # Pillar 4: Quantum Information Geometry, Density Matrices & Von Neumann Entropy
    'all:"von Neumann entropy" AND (all:graph OR all:network)',
    'all:"Bures" AND (all:"density matrix" OR all:manifold OR all:classification)',
    'all:"quantum Fisher information" AND (all:graph OR all:network OR all:learning)',
    
    # Pillar 5: Dynamic Spatial-Temporal Graphs for Multi-Channel Telemetry (EEG / RF)
    'all:EEG AND all:"dynamic graph" AND all:classification'
]

KEYWORD_WEIGHTS = {
    "quantum walk": 8,
    "continuous-time quantum walk": 9,
    "lindblad": 8,
    "open quantum": 7,
    "dissipative": 6,
    "von neumann": 7,
    "entropy": 4,
    "density matrix": 6,
    "bures": 7,
    "kuramoto": 7,
    "phase locking": 6,
    "vorticity": 8,
    "graph neural network": 5,
    "graph laplacian": 6,
    "eeg": 6,
    "telemetry": 5,
    "fisher information": 6,
    "seizure": 5,
    "synchronization": 5
}


def score_paper(paper: dict) -> float:
    text = (paper.get("title", "") + " " + paper.get("summary", "")).lower()
    score = 0.0
    for kw, w in KEYWORD_WEIGHTS.items():
        if kw in text:
            score += w * (1 + min(text.count(kw), 5) * 0.2)
    return score


def parse_papers_from_stdout(stdout_str: str) -> list:
    papers = []
    # Find all json blocks in stdout
    blocks = re.split(r'\{\s*"status":\s*"success"', stdout_str)
    for b in reversed(blocks):
        if '"papers":' in b:
            full_b = '{\n  "status": "success"' + b
            try:
                data = json.loads(full_b)
                return data.get("papers", [])
            except Exception:
                continue
    return papers


def run_searches():
    candidates = {}
    print(f"Executing {len(QUERIES)} targeted arXiv searches...")
    
    for idx, q in enumerate(QUERIES):
        print(f"[{idx+1}/{len(QUERIES)}] Query: {q}")
        clear_stale_locks()
        cmd = [
            sys.executable, SEARCH_SCRIPT,
            "--query", q,
            "--max_results", "6",
            "--sort_by", "submittedDate",
            "--sort_order", "descending"
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=25)
            if res.returncode == 0 and res.stdout.strip():
                items = parse_papers_from_stdout(res.stdout)
                for item in items:
                    pid = item["id"].split("/abs/")[-1].strip()
                    # Filter for 2020 to 2026
                    pub_str = item.get("published", "")
                    year = int(pub_str[:4]) if len(pub_str) >= 4 and pub_str[:4].isdigit() else 2000
                    if 2020 <= year <= 2026:
                        if pid not in candidates:
                            item["clean_id"] = pid
                            item["year"] = year
                            item["relevance_score"] = score_paper(item)
                            candidates[pid] = item
                print(f"  Total unique candidates so far: {len(candidates)}")
            else:
                print(f"  Warning: code {res.returncode}, stderr: {res.stderr[:80]}")
        except Exception as e:
            print(f"  Exception: {e}")
            
    return list(candidates.values())


def select_optimal_15(candidates):
    # Sort strictly by relevance score descending
    candidates_sorted = sorted(candidates, key=lambda x: x["relevance_score"], reverse=True)
    
    selected = []
    seen_titles = set()
    for c in candidates_sorted:
        title = c["title"].strip()
        if title not in seen_titles:
            seen_titles.add(title)
            selected.append(c)
        if len(selected) == 15:
            break
            
    return selected


def download_papers(papers):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    downloaded = []
    
    print(f"\nDownloading {len(papers)} papers into '{OUTPUT_DIR}/'...")
    for idx, p in enumerate(papers):
        pid = p["clean_id"]
        safe_title = re.sub(r"[^a-zA-Z0-9_\-]+", "_", p["title"][:45]).strip("_")
        fname = f"{idx+1:02d}_{pid}_{safe_title}.pdf"
        fpath = os.path.join(OUTPUT_DIR, fname)
        
        print(f"[{idx+1}/{len(papers)}] ({p['year']}) {p['title']}")
        print(f"     arXiv ID: {pid} | Score: {p['relevance_score']:.1f}")
        
        clear_stale_locks()
        cmd = [
            sys.executable, DOWNLOAD_SCRIPT,
            "--id", pid,
            "--format", "pdf",
            "--output", fpath
        ]
        
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=30)
            if os.path.exists(fpath) and os.path.getsize(fpath) > 1000:
                size_kb = os.path.getsize(fpath) / 1024.0
                print(f"     Saved: {fname} ({size_kb:.1f} KB)")
                p["pdf_path"] = fpath
                p["pdf_filename"] = fname
                p["file_size_kb"] = round(size_kb, 1)
                downloaded.append(p)
            else:
                print(f"     Download failed. Stderr: {res.stderr[:100]}")
        except Exception as e:
            print(f"     Exception: {e}")
            
    return downloaded


def generate_catalog(papers):
    # JSON catalog
    idx_path = os.path.join(OUTPUT_DIR, "papers_index.json")
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2)
        
    # Markdown catalog
    md_path = os.path.join(OUTPUT_DIR, "README.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Related Scientific Literature (2020–2026)\n\n")
        f.write("A curated collection of the 15 most similar research papers published between 2020 and 2026 directly relating to the theoretical, mathematical, and empirical foundations of **Hydro-Quantum Graph Walk Networks (HQGWN)**.\n\n")
        f.write("Downloaded via arXiv API in accordance with terms of use.\n\n")
        f.write("| # | Year | arXiv ID | Title | Downloaded PDF | Key Relevance to HQGWN |\n")
        f.write("| :-: | :--: | :------: | :---- | :------------- | :---------------------- |\n")
        
        for idx, p in enumerate(papers):
            authors = ", ".join(p.get("authors", [])[:3])
            if len(p.get("authors", [])) > 3:
                authors += " et al."
            f.write(f"| {idx+1} | {p['year']} | [{p['clean_id']}](https://arxiv.org/abs/{p['clean_id']}) | **{p['title']}**<br>*{authors}* | [`{p['pdf_filename']}`](./{p['pdf_filename']}) | {p.get('summary', '')[:160]}... |\n")
            
    print(f"\nCatalog successfully written to:\n - {idx_path}\n - {md_path}")


if __name__ == "__main__":
    t0 = time.time()
    candidates = run_searches()
    print(f"Found {len(candidates)} total candidates across 2020-2026.")
    selected = select_optimal_15(candidates)
    downloaded = download_papers(selected)
    generate_catalog(downloaded)
    print(f"\nCompleted in {time.time()-t0:.1f}s. Successfully downloaded {len(downloaded)} papers.")
