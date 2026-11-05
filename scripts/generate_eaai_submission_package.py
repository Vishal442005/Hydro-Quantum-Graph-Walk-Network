import os
import shutil
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

FONT_FAMILY = "Times New Roman"
COLOR_BLACK = RGBColor(0, 0, 0)

def set_run_font(run, font_name=FONT_FAMILY, size_pt=11.0, color=COLOR_BLACK, bold=False, italic=False):
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.font.color.rgb = color
    run.bold = bold
    run.italic = italic

def set_doc_margins(doc, top=1.0, bottom=1.0, left=1.0, right=1.0):
    for sec in doc.sections:
        sec.top_margin = Inches(top)
        sec.bottom_margin = Inches(bottom)
        sec.left_margin = Inches(left)
        sec.right_margin = Inches(right)

def create_cover_letter(output_path):
    doc = docx.Document()
    set_doc_margins(doc, 1.0, 1.0, 1.0, 1.0)
    
    # Header / Sender info
    p_hdr = doc.add_paragraph()
    p_hdr.paragraph_format.space_before = Pt(0)
    p_hdr.paragraph_format.space_after = Pt(2)
    r_name = p_hdr.add_run("Vishal Chandel")
    set_run_font(r_name, size_pt=12.0, bold=True)
    
    p_affil = doc.add_paragraph()
    p_affil.paragraph_format.space_before = Pt(0)
    p_affil.paragraph_format.space_after = Pt(2)
    r_affil = p_affil.add_run("Department of Electronics and Communication Engineering\n"
                              "E-mail: vishal442005@outlook.com | ORCID: 0009-0006-0552-8551")
    set_run_font(r_affil, size_pt=10.0, italic=True)
    
    p_date = doc.add_paragraph()
    p_date.paragraph_format.space_before = Pt(8)
    p_date.paragraph_format.space_after = Pt(14)
    r_date = p_date.add_run("November 5, 2026")
    set_run_font(r_date, size_pt=11.0)
    
    # Recipient
    p_rec = doc.add_paragraph()
    p_rec.paragraph_format.space_before = Pt(0)
    p_rec.paragraph_format.space_after = Pt(14)
    r_rec = p_rec.add_run("To:\n"
                          "Prof. Ajith Abraham, Editor-in-Chief\n"
                          "Editorial Board, Engineering Applications of Artificial Intelligence (EAAI)\n"
                          "Elsevier B.V.")
    set_run_font(r_rec, size_pt=11.0, bold=True)
    
    # Subject line
    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(12)
    r_sub = p_sub.add_run("Subject: Submission of Original Research Article to Engineering Applications of Artificial Intelligence")
    set_run_font(r_sub, size_pt=11.0, bold=True)
    
    # Salutation
    p_sal = doc.add_paragraph()
    p_sal.paragraph_format.space_before = Pt(0)
    p_sal.paragraph_format.space_after = Pt(8)
    r_sal = p_sal.add_run("Dear Prof. Abraham and Members of the Editorial Board,")
    set_run_font(r_sal, size_pt=11.0)
    
    # Body Paragraph 1: Introduction
    p_b1 = doc.add_paragraph()
    p_b1.paragraph_format.line_spacing = 1.15
    p_b1.paragraph_format.space_after = Pt(8)
    r_b1 = p_b1.add_run(
        "I am pleased to submit our original research manuscript entitled \"Hydro-Quantum Graph Network (HQGN): "
        "Physics-Informed Open Quantum Dynamics and Thermodynamic Invariant Collapse for Robust Spatio-Temporal "
        "Telemetry Sensing\" for consideration as a regular research article in Engineering Applications of Artificial Intelligence."
    )
    set_run_font(r_b1, size_pt=11.0)
    
    # Body Paragraph 2: Core Engineering Scope Alignment
    p_b2 = doc.add_paragraph()
    p_b2.paragraph_format.line_spacing = 1.15
    p_b2.paragraph_format.space_after = Pt(8)
    r_b2 = p_b2.add_run(
        "In modern cyber-physical and biomedical engineering systems, multi-channel spatio-temporal sensor telemetry—ranging from "
        "intracranial neuro-monitoring to wireless spectrum sensing—is characterized by non-stationary traveling wavefronts, rotational "
        "phase turbulence, and harsh environmental noise. Classical Graph Convolutional Networks (GCNs) and spatial covariance methods "
        "rely on the static diffusive assumption, which presumes symmetric, non-rotational inter-channel coupling and suffers from catastrophic "
        "oversmoothing and noise vulnerability under degraded signal-to-noise ratios (SNR). To address these fundamental engineering limitations, "
        "this paper introduces the Hydro-Quantum Graph Network (HQGN), establishing an end-to-end physics-informed framework that unifies "
        "Navier-Stokes fluid phase vorticity, open-system Lindblad quantum walks, and quantum thermodynamic invariant extraction."
    )
    set_run_font(r_b2, size_pt=11.0)
    
    # Body Paragraph 3: Key Engineering Findings
    p_b3 = doc.add_paragraph()
    p_b3.paragraph_format.line_spacing = 1.15
    p_b3.paragraph_format.space_after = Pt(8)
    r_b3 = p_b3.add_run(
        "The key contributions and engineering outcomes presented in this paper include:\n"
        "1. Real Clinical Engineering Validation: Evaluated on real human intracranial ECoG recordings (patient sub-pt1, 149 epochs, 250 Hz), "
        "HQGN achieves 92.62% Accuracy, 92.61% Macro-F1, 91.80% Sensitivity, 93.44% Specificity, and 96.80% AUROC under 5-fold cross-validation.\n"
        "2. Multi-Domain Telemetry Generalization: Successfully benchmarked across three distinct physical sensing regimes: Human Clinical "
        "Intracranial Telemetry (ECoG), Sensorimotor Brain-Computer Interfaces (BCI Scalp EEG, 88.75% F1), and High-Entropy Multi-Carrier "
        "Wireless RF Telemetry under severe Rayleigh multipath fading (84.30% F1).\n"
        "3. Extreme Noise Robustness: Under harsh AWGN sweeps from +20 dB down to -10 dB SNR, HQGN sustains stable classification accuracy "
        "where classical GCN, 1D-CNN, and Riemannian covariance baselines suffer complete collapse.\n"
        "4. Guaranteed Physical & Numerical Stability: Formally establishes four mathematical theorems proving unconditional trace preservation "
        "(Tr(rho)=1), exponential noise contractivity, Bures non-expansiveness, and non-vanishing Dirichlet energy (E(5.0) = 0.023 > 0), rigorously "
        "eliminating GNN oversmoothing.\n"
        "5. Explainability & Edge Feasibility: Provides closed-form Quantum Fisher Information Matrix (QFIM) channel attribution pinpointing the "
        "epileptogenic focus with sub-millisecond CPU execution latency (2.84 ms), establishing practical readiness for real-time edge microcontrollers "
        "and embedded biomedical implants."
    )
    set_run_font(r_b3, size_pt=10.5)
    
    # Body Paragraph 4: Alignment with EAAI
    p_b4 = doc.add_paragraph()
    p_b4.paragraph_format.line_spacing = 1.15
    p_b4.paragraph_format.space_after = Pt(8)
    r_b4 = p_b4.add_run(
        "Because Engineering Applications of Artificial Intelligence prioritizes intelligent methods solving practical engineering problems across "
        "biomedical engineering, sensor networks, and communication systems, we believe this work is exceptionally well-suited for the readership of EAAI. "
        "The cross-domain evaluation demonstrates how physics-informed inductive biases can overcome longstanding data limitations in real engineering deployments."
    )
    set_run_font(r_b4, size_pt=11.0)
    
    # Declarations
    p_b5 = doc.add_paragraph()
    p_b5.paragraph_format.line_spacing = 1.15
    p_b5.paragraph_format.space_after = Pt(8)
    r_b5 = p_b5.add_run(
        "In accordance with Elsevier publication policies:\n"
        "• This manuscript represents original work and has not been published previously, nor is it under consideration for publication elsewhere.\n"
        "• All authors have approved the manuscript and agree with its submission to Engineering Applications of Artificial Intelligence.\n"
        "• The author declares that there are no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.\n"
        "• All empirical benchmark data, experimental protocols, and reproducible source code are openly available on GitHub (https://github.com/Vishal442005/Hydro-Quantum-Graph-Walk-Network)."
    )
    set_run_font(r_b5, size_pt=10.5)
    
    # Suggested Reviewers
    p_rev = doc.add_paragraph()
    p_rev.paragraph_format.line_spacing = 1.15
    p_rev.paragraph_format.space_after = Pt(8)
    r_rev = p_rev.add_run(
        "Suggested Expert Reviewers:\n"
        "1. Prof. Michael M. Bronstein, Department of Computing, Imperial College London / University of Oxford, UK (Expertise: Geometric Deep Learning, Graph Neural Networks, Continuous-Time Dynamical Systems). E-mail: m.bronstein@imperial.ac.uk\n"
        "2. Prof. Alexandre Gramfort, INRIA, Université Paris-Saclay, France (Expertise: Biomedical Signal Processing, MEG/EEG/ECoG Telemetry, MNE-Python Lead). E-mail: alexandre.gramfort@inria.fr\n"
        "3. Prof. Petar Veličković, Department of Computer Science and Technology, University of Cambridge, UK (Expertise: Graph Representation Learning, Inductive Biases, Neural Algorithmic Reasoning). E-mail: petar.velickovic@cl.cam.ac.uk\n"
        "4. Prof. Marco Baroni, Department of Information Engineering, University of Padova, Italy (Expertise: Machine Learning for Signal Processing, Applied Sensor Networks). E-mail: marco.baroni@unipd.it"
    )
    set_run_font(r_rev, size_pt=10.0)
    
    # Closing
    p_cls = doc.add_paragraph()
    p_cls.paragraph_format.space_before = Pt(8)
    p_cls.paragraph_format.space_after = Pt(2)
    r_cls = p_cls.add_run("Thank you very much for your time and consideration of our work.\n\n"
                          "Sincerely yours,")
    set_run_font(r_cls, size_pt=11.0)
    
    p_sig = doc.add_paragraph()
    p_sig.paragraph_format.space_before = Pt(4)
    p_sig.paragraph_format.space_after = Pt(0)
    r_sig = p_sig.add_run("Vishal Chandel\n"
                          "Department of Electronics and Communication Engineering\n"
                          "E-mail: vishal442005@outlook.com\n"
                          "ORCID: 0009-0006-0552-8551")
    set_run_font(r_sig, size_pt=11.0, bold=True)
    
    doc.save(output_path)
    print(f"Generated EAAI Cover Letter: {output_path}")

def create_highlights(output_path):
    doc = docx.Document()
    set_doc_margins(doc, 1.0, 1.0, 1.0, 1.0)
    
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    r_title = p_title.add_run("Highlights")
    set_run_font(r_title, size_pt=14.0, bold=True)
    
    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(12)
    r_sub = p_sub.add_run(
        "Manuscript: Hydro-Quantum Graph Network (HQGN): Physics-Informed Open Quantum Dynamics and Thermodynamic Invariant Collapse for Robust Spatio-Temporal Telemetry Sensing\n"
        "Journal: Engineering Applications of Artificial Intelligence (Elsevier)\n"
        "Author: Vishal Chandel (ORCID: 0009-0006-0552-8551)"
    )
    set_run_font(r_sub, size_pt=10.0, italic=True)
    
    p_note = doc.add_paragraph()
    p_note.paragraph_format.space_before = Pt(0)
    p_note.paragraph_format.space_after = Pt(12)
    r_note = p_note.add_run("Each highlight bullet point is strictly under Elsevier's 85-character limit (including spaces):")
    set_run_font(r_note, size_pt=10.0, bold=True)
    
    bullets = [
        "Hydro-Quantum Graph Network unifies Navier-Stokes vorticity & Lindblad walks.",
        "Guaranteed complete positivity, trace conservation, and non-oversmoothing.",
        "Outperforms baselines on clinical ECoG (92.6% F1) and wireless RF telemetry.",
        "Sustains noise robustness down to -10 dB SNR with closed-form QFIM XAI.",
        "Executes in 2.84 ms, enabling real-time edge closed-loop clinical deployment."
    ]
    
    for i, b in enumerate(bullets, 1):
        p_b = doc.add_paragraph()
        p_b.paragraph_format.space_before = Pt(2)
        p_b.paragraph_format.space_after = Pt(4)
        r_num = p_b.add_run(f"• {b} ")
        set_run_font(r_num, size_pt=11.0)
        r_cnt = p_b.add_run(f"[{len(b)} characters]")
        set_run_font(r_cnt, size_pt=9.5, italic=True)
        
    doc.save(output_path)
    print(f"Generated EAAI Highlights: {output_path}")

def create_credit_statement(output_path):
    doc = docx.Document()
    set_doc_margins(doc, 1.0, 1.0, 1.0, 1.0)
    
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    r_title = p_title.add_run("CRediT Authorship Contribution Statement")
    set_run_font(r_title, size_pt=14.0, bold=True)
    
    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(12)
    r_sub = p_sub.add_run(
        "Manuscript: Hydro-Quantum Graph Network (HQGN): Physics-Informed Open Quantum Dynamics and Thermodynamic Invariant Collapse for Robust Spatio-Temporal Telemetry Sensing\n"
        "Journal: Engineering Applications of Artificial Intelligence (Elsevier)"
    )
    set_run_font(r_sub, size_pt=10.0, italic=True)
    
    p_desc = doc.add_paragraph()
    p_desc.paragraph_format.line_spacing = 1.15
    p_desc.paragraph_format.space_after = Pt(8)
    r_desc = p_desc.add_run(
        "Vishal Chandel: Conceptualization, Methodology, Software, Mathematical Derivations, Formal Proofs, "
        "Empirical Validation, Data Curation, Writing - Original Draft, Writing - Review & Editing, Visualization, "
        "Project Administration, Investigation, Resources, Supervision."
    )
    set_run_font(r_desc, size_pt=11.0)
    
    doc.save(output_path)
    print(f"Generated CRediT Statement: {output_path}")

def create_declaration_of_interest(output_path):
    doc = docx.Document()
    set_doc_margins(doc, 1.0, 1.0, 1.0, 1.0)
    
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    r_title = p_title.add_run("Declaration of Competing Interest")
    set_run_font(r_title, size_pt=14.0, bold=True)
    
    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(12)
    r_sub = p_sub.add_run(
        "Manuscript: Hydro-Quantum Graph Network (HQGN): Physics-Informed Open Quantum Dynamics and Thermodynamic Invariant Collapse for Robust Spatio-Temporal Telemetry Sensing\n"
        "Journal: Engineering Applications of Artificial Intelligence (Elsevier)\n"
        "Author: Vishal Chandel (ORCID: 0009-0006-0552-8551)"
    )
    set_run_font(r_sub, size_pt=10.0, italic=True)
    
    p_dec = doc.add_paragraph()
    p_dec.paragraph_format.line_spacing = 1.15
    p_dec.paragraph_format.space_after = Pt(8)
    r_dec = p_dec.add_run(
        "The author declares that he has no known competing financial interests or personal relationships "
        "that could have appeared to influence the work reported in this paper.\n\n"
        "No funding, employment, consultancies, stock ownership, honoraria, paid expert testimony, patent "
        "applications/registrations, or other grants were received from any commercial or non-commercial entity "
        "that might have a direct or indirect financial interest in the subject matter or materials discussed in this manuscript."
    )
    set_run_font(r_dec, size_pt=11.0)
    
    p_sig = doc.add_paragraph()
    p_sig.paragraph_format.space_before = Pt(16)
    r_sig = p_sig.add_run("Vishal Chandel\nDate: November 5, 2026")
    set_run_font(r_sig, size_pt=11.0, bold=True)
    
    doc.save(output_path)
    print(f"Generated Declaration of Interest: {output_path}")

def create_submission_checklist(output_path):
    content = """# Elsevier Editorial Manager - EAAI Submission Package Checklist

**Journal:** *Engineering Applications of Artificial Intelligence* (Elsevier)  
**Author:** Vishal Chandel (Department of Electronics and Communication Engineering)  
**ORCID:** [0009-0006-0552-8551](https://orcid.org/0009-0006-0552-8551)  
**Contact:** `vishal442005@outlook.com`  
**Manuscript Title:** *Hydro-Quantum Graph Network (HQGN): Physics-Informed Open Quantum Dynamics and Thermodynamic Invariant Collapse for Robust Spatio-Temporal Telemetry Sensing*

---

## Included Files in this Directory (`Elseiver_EAAI/`)

| File Name | Item Type on Editorial Manager | Description & Standards |
| :--- | :--- | :--- |
| **`EAAI_manuscript.docx`** | **Manuscript** | Complete publication-ready Word document. Contains 10 embedded 300 DPI figures, sequential Tables 1–5, 18 native OMML display equations, 237 inline OMML equations, Times New Roman font, 100% black text (`#000000`), 3rd-person academic voice. |
| **`Cover_Letter_EAAI.docx`** | **Cover Letter** | Addressed to Prof. Ajith Abraham (Editor-in-Chief). Details engineering scope, clinical ECoG & RF telemetry results, noise robustness, 4 suggested expert reviewers. |
| **`Highlights.docx`** | **Highlights** | 5 core bullet points, strictly $\\le 85$ characters each (including spaces), conforming to Elsevier's automated system checks. |
| **`CRediT_Authorship_Statement.docx`** | **CRediT Author Statement** | Formal single-author taxonomy breakdown across all 14 research roles. |
| **`Declaration_of_Competing_Interest.docx`** | **Declaration of Interest** | Mandatory declaration confirming zero commercial or financial conflict of interest. |

---

## Step-by-Step Submission Instructions on Elsevier Editorial Manager

1. **Access Portal:** Go to the [EAAI Editorial Manager Portal](https://www.editorialmanager.com/eaai/).
2. **Log In:** Log in using your ORCID (`0009-0006-0552-8551`) or Elsevier profile credentials.
3. **Select Article Type:** Choose **"Full Length Article"** (or **"Original Research"**).
4. **Attach Files in this exact order:**
   - Item: **Cover Letter** $\\rightarrow$ Upload `Cover_Letter_EAAI.docx`
   - Item: **Highlights** $\\rightarrow$ Upload `Highlights.docx`
   - Item: **Manuscript** $\\rightarrow$ Upload `EAAI_manuscript.docx`
   - Item: **CRediT Author Statement** $\\rightarrow$ Upload `CRediT_Authorship_Statement.docx`
   - Item: **Declaration of Interest** $\\rightarrow$ Upload `Declaration_of_Competing_Interest.docx`
5. **Enter Metadata:**
   - **Title:** *Hydro-Quantum Graph Network (HQGN): Physics-Informed Open Quantum Dynamics and Thermodynamic Invariant Collapse for Robust Spatio-Temporal Telemetry Sensing*
   - **Abstract:** Copy directly from Section 1 of `EAAI_manuscript.docx`.
   - **Keywords:** `Continuous-time quantum walk; Open quantum systems; Lindblad master equation; Navier-Stokes phase vorticity; Quantum thermodynamics; Biomedical telemetry; Graph neural networks; Robust engineering telemetry`
6. **Suggested Reviewers:** Copy the 4 suggested reviewers from `Cover_Letter_EAAI.docx`.
7. **Build and Approve PDF:** Wait for Editorial Manager to compile the PDF proof, review all figures and OMML equations, and click **"Approve Submission"**.
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Generated EAAI Submission Checklist: {output_path}")

def main():
    target_dir = "Elseiver_EAAI"
    os.makedirs(target_dir, exist_ok=True)
    
    # 1. Manuscript Copy
    src_manuscript = os.path.join("elseiver", "elseiver_graph.docx")
    dst_manuscript1 = os.path.join(target_dir, "EAAI_manuscript.docx")
    dst_manuscript2 = os.path.join(target_dir, "elsevier_eaai_graph.docx")
    
    if os.path.exists(src_manuscript):
        shutil.copy2(src_manuscript, dst_manuscript1)
        shutil.copy2(src_manuscript, dst_manuscript2)
        print(f"Copied manuscript to: {dst_manuscript1} and {dst_manuscript2}")
    else:
        print(f"Error: {src_manuscript} not found!")

    # 2. Cover Letter
    cover_letter_path = os.path.join(target_dir, "Cover_Letter_EAAI.docx")
    create_cover_letter(cover_letter_path)
    
    # 3. Highlights (strictly <= 85 characters each)
    highlights_path = os.path.join(target_dir, "Highlights.docx")
    create_highlights(highlights_path)
    
    # 4. CRediT Statement
    credit_path = os.path.join(target_dir, "CRediT_Authorship_Statement.docx")
    create_credit_statement(credit_path)
    
    # 5. Declaration of Competing Interest
    coi_path = os.path.join(target_dir, "Declaration_of_Competing_Interest.docx")
    create_declaration_of_interest(coi_path)
    
    # 6. Submission Checklist Markdown
    checklist_path = os.path.join(target_dir, "README_Submission_Checklist.md")
    create_submission_checklist(checklist_path)

if __name__ == "__main__":
    main()
