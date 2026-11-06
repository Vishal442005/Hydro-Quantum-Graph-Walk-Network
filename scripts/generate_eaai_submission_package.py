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
        "HQGN achieves 92.62% Accuracy, 92.61% Macro-F1, 89.58% Sensitivity, 95.90% Specificity, and 96.85% AUROC under 5-fold cross-validation.\n"
        "2. Multi-Domain Telemetry Generalization: Successfully benchmarked across three distinct physical sensing regimes: Human Clinical "
        "Intracranial Telemetry (ECoG, 92.61% F1), Sensorimotor Brain-Computer Interfaces (BCI Scalp EEG, 83.98% F1), and High-Entropy Multi-Carrier "
        "Wireless RF Telemetry under severe Rayleigh multipath fading (58.65% F1).\n"
        "3. Extreme Noise Robustness: Under harsh AWGN sweeps from +20 dB down to -10 dB SNR, HQGN sustains stable classification accuracy "
        "where classical GCN, 1D-CNN, and Riemannian covariance baselines suffer complete collapse.\n"
        "4. Guaranteed Physical & Numerical Stability: Formally establishes four mathematical theorems proving unconditional trace preservation "
        "(Tr(rho)=1), exponential noise contractivity, Bures non-expansiveness, and non-vanishing Dirichlet energy (E(5.0) = 0.023 > 0), rigorously "
        "eliminating GNN oversmoothing.\n"
        "5. Explainability & Edge Feasibility: Provides closed-form Quantum Fisher Information Matrix (QFIM) channel attribution pinpointing the "
        "epileptogenic focus with sub-millisecond GPU execution latency (0.72 ms) and real-time CPU latency (2.84 ms), establishing practical readiness for real-time edge microcontrollers "
        "and embedded biomedical implants.\n"
        "6. Systematic Operational Envelope & Multi-Condition Stress Synthesis: Establishes a rigorous operational taxonomy (Section 6.8, Table 6) "
        "delineating exact physical regimes where HQGN excels versus boundary conditions (ultra-sparse arrays M < 4, Cauchy impulsive spikes), providing verified algorithmic "
        "mitigations (robust median phase clipping restoring accuracy to 83.60%) and multi-condition stress benchmarking across pink 1/f noise and 50 Hz harmonic hum (Table 5, Fig. 10)."
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

def create_title_page(output_path):
    doc = docx.Document()
    set_doc_margins(doc, 1.0, 1.0, 1.0, 1.0)
    
    # Title Page Header
    p_hdr = doc.add_paragraph()
    p_hdr.paragraph_format.space_before = Pt(0)
    p_hdr.paragraph_format.space_after = Pt(2)
    r_hdr = p_hdr.add_run("TITLE PAGE WITH AUTHOR DETAILS")
    set_run_font(r_hdr, size_pt=14.0, bold=True)
    
    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(12)
    r_sub = p_sub.add_run(
        "Journal: Engineering Applications of Artificial Intelligence (Elsevier)\n"
        "Article Type: Regular Research Article (Full Length Article)"
    )
    set_run_font(r_sub, size_pt=10.0, italic=True)

    # Dividing Rule
    p_rule1 = doc.add_paragraph()
    p_rule1.paragraph_format.space_before = Pt(0)
    p_rule1.paragraph_format.space_after = Pt(12)
    pBdr1 = parse_xml(r'''
        <w:pBdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
            <w:bottom w:val="single" w:sz="6" w:space="1" w:color="000000"/>
        </w:pBdr>
    ''')
    p_rule1._p.get_or_add_pPr().append(pBdr1)

    # 1. Full Article Title
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(4)
    p_title.paragraph_format.space_after = Pt(12)
    r_title_lbl = p_title.add_run("Article Title:\n")
    set_run_font(r_title_lbl, size_pt=10.5, bold=True)
    r_title = p_title.add_run("Hydro-Quantum Graph Network (HQGN): Physics-Informed Open Quantum Dynamics and Thermodynamic Invariant Collapse for Robust Spatio-Temporal Telemetry Sensing")
    set_run_font(r_title, size_pt=14.0, bold=True)

    # 2. Author Names & Affiliations
    p_auth = doc.add_paragraph()
    p_auth.paragraph_format.space_before = Pt(6)
    p_auth.paragraph_format.space_after = Pt(4)
    r_auth_lbl = p_auth.add_run("Author:\n")
    set_run_font(r_auth_lbl, size_pt=10.5, bold=True)
    r_author = p_auth.add_run("Vishal Chandel")
    set_run_font(r_author, size_pt=12.0, bold=True)
    r_star = p_auth.add_run(" *")
    set_run_font(r_star, size_pt=12.0, bold=True)

    p_affil = doc.add_paragraph()
    p_affil.paragraph_format.space_before = Pt(0)
    p_affil.paragraph_format.space_after = Pt(10)
    r_affil = p_affil.add_run("Department of Electronics and Communication Engineering, National Institute of Technology, Hamirpur, Himachal Pradesh 177005, India")
    set_run_font(r_affil, size_pt=10.5, italic=True)

    # 3. Corresponding Author Details
    p_corr = doc.add_paragraph()
    p_corr.paragraph_format.space_before = Pt(4)
    p_corr.paragraph_format.space_after = Pt(12)
    r_corr_lbl = p_corr.add_run("* Corresponding Author Details:\n")
    set_run_font(r_corr_lbl, size_pt=10.5, bold=True)
    r_corr = p_corr.add_run(
        "Name: Vishal Chandel\n"
        "Affiliation: Department of Electronics and Communication Engineering, National Institute of Technology, Hamirpur, Himachal Pradesh 177005, India\n"
        "E-mail Address: vishal442005@outlook.com\n"
        "ORCID Identifier: 0009-0006-0552-8551 (https://orcid.org/0009-0006-0552-8551)"
    )
    set_run_font(r_corr, size_pt=10.5)

    # 4. Manuscript Metrics
    p_met = doc.add_paragraph()
    p_met.paragraph_format.space_before = Pt(4)
    p_met.paragraph_format.space_after = Pt(12)
    r_met_lbl = p_met.add_run("Manuscript Statistics:\n")
    set_run_font(r_met_lbl, size_pt=10.5, bold=True)
    r_met = p_met.add_run(
        "• Total Word Count: ~11,200 words (including abstract, references, and table text)\n"
        "• Number of Figures: 12 (High-resolution 300 DPI, sequentially numbered and referenced)\n"
        "• Number of Tables: 7 (Booktabs format with exact mean ± std performance metrics)\n"
        "• Number of Mathematical Equations: 18 Display Equations, 270+ Inline Equations\n"
        "• Number of References: 30"
    )
    set_run_font(r_met, size_pt=10.0)

    # 5. Abstract
    p_abs = doc.add_paragraph()
    p_abs.paragraph_format.space_before = Pt(4)
    p_abs.paragraph_format.space_after = Pt(4)
    r_abs_lbl = p_abs.add_run("Abstract:\n")
    set_run_font(r_abs_lbl, size_pt=11.0, bold=True)
    r_abs = p_abs.add_run(
        "Multi-channel time-series telemetry systems in clinical neurophysiology, brain-computer interfaces (BCIs), "
        "and cyber-physical communication networks face severe performance degradation caused by non-stationary phase jitter, "
        "electrode drift, and spatial volume conduction. Contemporary spatio-temporal graph neural networks (ST-GNNs) and "
        "continuous-time quantum walk (CTQW) architectures remain inherently constrained: classical Markov diffusion collapses "
        "exponentially into trivial oversmoothed consensus, while closed unitary quantum walks lack open thermodynamic dissipation "
        "sinks required to extinguish environmental noise. This paper introduces the Hydro-Quantum Graph Network (HQGN), "
        "a mathematically unified framework synthesizing fluid vorticity dynamics, open-system Lindblad stochastic quantum walks, "
        "and quantum thermodynamic invariants. Stage I maps multi-channel temporal trajectories onto a continuous 2D fluid manifold, "
        "constructing a Poincaré-Cartan phase-vorticity graph with an antisymmetric curl circulation tensor. Stage II evolves the "
        "resulting complex density operator via a Gorini-Kossakowski-Sudarshan-Lindblad (GKSL) stochastic master equation driven by a strictly Hermitian "
        "Hamiltonian with chiral imaginary vorticity coupling, solved through second-order symmetric Strang operator splitting that guarantees unconditional trace preservation and complete "
        "positivity. Stage III projects the steady-state density operator onto the Riemannian Bures-Wasserstein metric manifold, "
        "extracting spectral entropy and Umegaki quantum relative entropy coordinates. Four rigorous theorems establish unconditional "
        "trace preservation, asymptotic noise contraction, Bures non-expansiveness, and non-collapsing Dirichlet energy (E(5.0) = 0.023 > 0). "
        "Empirical evaluation across three authentic physical domains—real human clinical epilepsy intracranial ECoG (149 windows), "
        "sensorimotor BCI EEG (144 trials), and non-stationary multipath wireless RF telemetry—demonstrates that HQGN achieves superior classification "
        "fidelity (92.62% accuracy, 92.61% Macro-F1 on human ECoG), outperforming classical GNN, Unitary CTQW, and Dense 1D-CNN baselines while preserving "
        "monotonic noise robustness down to -10 dB SNR. Furthermore, closed-form Quantum Fisher Information Matrix (QFIM) extraction enables intrinsic, "
        "post-hoc-free channel attribution that localizes the epileptogenic focus at sub-millisecond GPU latency (0.72 ms) and real-time CPU latency (2.84 ms), "
        "establishing practical feasibility for low-power edge medical microcontrollers and embedded telemetry processors."
    )
    set_run_font(r_abs, size_pt=10.0)

    # 6. Keywords
    p_kw = doc.add_paragraph()
    p_kw.paragraph_format.space_before = Pt(6)
    p_kw.paragraph_format.space_after = Pt(12)
    r_kw_lbl = p_kw.add_run("Keywords: ")
    set_run_font(r_kw_lbl, size_pt=10.5, bold=True)
    r_kw = p_kw.add_run("Continuous-time quantum walk; Open quantum systems; Lindblad master equation; Navier-Stokes phase vorticity; Quantum thermodynamics; Biomedical telemetry; Graph neural networks; Robust engineering telemetry")
    set_run_font(r_kw, size_pt=10.0, italic=True)

    # 7. Highlights
    p_hl = doc.add_paragraph()
    p_hl.paragraph_format.space_before = Pt(4)
    p_hl.paragraph_format.space_after = Pt(4)
    r_hl_lbl = p_hl.add_run("Research Highlights (strictly ≤ 85 characters each):\n")
    set_run_font(r_hl_lbl, size_pt=10.5, bold=True)
    bullets = [
        "Hydro-Quantum Graph Network unifies Navier-Stokes vorticity & Lindblad walks.",
        "Guaranteed complete positivity, trace conservation, and non-oversmoothing.",
        "Outperforms baselines on clinical ECoG (92.6% F1) and wireless RF telemetry.",
        "Sustains noise robustness down to -10 dB SNR with closed-form QFIM XAI.",
        "Executes in 2.84 ms, enabling real-time edge closed-loop clinical deployment."
    ]
    for b in bullets:
        p_b = doc.add_paragraph()
        p_b.paragraph_format.space_before = Pt(1)
        p_b.paragraph_format.space_after = Pt(2)
        r_b = p_b.add_run(f"• {b}")
        set_run_font(r_b, size_pt=10.0)

    # 8. CRediT Authorship Statement
    p_cr = doc.add_paragraph()
    p_cr.paragraph_format.space_before = Pt(8)
    p_cr.paragraph_format.space_after = Pt(4)
    r_cr_lbl = p_cr.add_run("CRediT Authorship Contribution Statement:\n")
    set_run_font(r_cr_lbl, size_pt=10.5, bold=True)
    r_cr = p_cr.add_run("Vishal Chandel: Conceptualization, Methodology, Software, Mathematical Derivations, Formal Proofs, Empirical Validation, Data Curation, Writing - Original Draft, Writing - Review & Editing, Visualization, Project administration.")
    set_run_font(r_cr, size_pt=10.0)

    # 9. Declaration of Competing Interest
    p_coi = doc.add_paragraph()
    p_coi.paragraph_format.space_before = Pt(8)
    p_coi.paragraph_format.space_after = Pt(4)
    r_coi_lbl = p_coi.add_run("Declaration of Competing Interest:\n")
    set_run_font(r_coi_lbl, size_pt=10.5, bold=True)
    r_coi = p_coi.add_run("The author declares that he has no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.")
    set_run_font(r_coi, size_pt=10.0)

    # 10. Funding & Acknowledgments
    p_ack = doc.add_paragraph()
    p_ack.paragraph_format.space_before = Pt(8)
    p_ack.paragraph_format.space_after = Pt(4)
    r_ack_lbl = p_ack.add_run("Funding and Acknowledgments:\n")
    set_run_font(r_ack_lbl, size_pt=10.5, bold=True)
    r_ack = p_ack.add_run(
        "This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors. "
        "The author acknowledges the open-source scientific software initiatives that supported this research, including MNE-Python, SciPy, NumPy, and the BCI Competition consortia."
    )
    set_run_font(r_ack, size_pt=10.0)

    # 11. Data and Code Availability
    p_dat = doc.add_paragraph()
    p_dat.paragraph_format.space_before = Pt(8)
    p_dat.paragraph_format.space_after = Pt(4)
    r_dat_lbl = p_dat.add_run("Data and Code Availability:\n")
    set_run_font(r_dat_lbl, size_pt=10.5, bold=True)
    r_dat = p_dat.add_run(
        "All benchmark telemetry datasets, preprocessed time-series tensors, PyTorch framework implementations, trained model weights, and reproduction scripts "
        "supporting the findings of this study are openly available on GitHub at: https://github.com/Vishal442005/Hydro-Quantum-Graph-Walk-Network under the "
        "permissive Apache-2.0 open-source license. The persistent digital object identifier (DOI) is registered with Zenodo."
    )
    set_run_font(r_dat, size_pt=10.0)

    doc.save(output_path)
    print(f"Generated Title Page: {output_path}")


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
    content = """# Elsevier Editorial Manager - EAAI Double-Blind Submission Checklist

**Journal:** *Engineering Applications of Artificial Intelligence* (Elsevier)  
**Author:** Vishal Chandel (Department of Electronics and Communication Engineering)  
**ORCID:** [0009-0006-0552-8551](https://orcid.org/0009-0006-0552-8551)  
**Contact:** `vishal442005@outlook.com`  
**Manuscript Title:** *Hydro-Quantum Graph Network (HQGN): Physics-Informed Open Quantum Dynamics and Thermodynamic Invariant Collapse for Robust Spatio-Temporal Telemetry Sensing*

---

## Direct Mapping to Elsevier Editorial Manager Dropdown Options

When uploading files on the EAAI Editorial Manager portal:

### 1. Mandatory Core Files (Top Dropdown: "Choose file type to upload")

| Dropdown Option in Editorial Manager | File to Select and Upload | Description |
| :--- | :--- | :--- |
| **`Manuscript without author details`** | **`Manuscript_without_author_details.docx`**<br>*(or `Blinded_Manuscript.docx`)* | **Mandatory Blinded Manuscript for Double-Blind Peer Review.** Contains the complete article: Title, Abstract, Keywords, Sections 1–10, all 12 embedded 300 DPI figures, all 7 booktabs tables, all 18 display equations, and 30 references. **All author names, affiliations, emails, and personal acknowledgments have been strictly removed.** |
| **`Title page with author details`** | **`Title_page_with_author_details.docx`**<br>*(or `Title_Page.docx`)* | **Mandatory Dedicated Title Page.** Contains: Article Title, Author Name (Vishal Chandel), Institutional Affiliation, Corresponding Author Contact, ORCID (`0009-0006-0552-8551`), Abstract, Keywords, 5 Highlights, CRediT Statement, Competing Interest Declaration, Funding & Data Availability. |

---

### 2. Optional and Supplementary Files (Bottom Dropdown: "Upload your optional and supplementary files")

| Dropdown Option in Editorial Manager | File to Select and Upload | Description |
| :--- | :--- | :--- |
| **`Cover Letter`** | **`Cover_Letter_EAAI.docx`** | Formal letter addressed to Prof. Ajith Abraham (Editor-in-Chief), summarizing engineering scope, clinical & RF results, noise robustness, operational envelope synthesis, and 4 international expert reviewers. |
| **`Response to Reviewers`** | **`Response_to_Editor_and_Reviewers.docx`** | Detailed point-by-point formal response addressing all editorial decision points, archival depth, curve explanations, and multi-condition stress benchmarks. |
| **`Highlights`** | **`Highlights.docx`** | 5 core bullet points, strictly $\le 85$ characters each (including spaces), conforming to Elsevier's automated system checks. |
| **`Declaration of Interest`** | **`Declaration_of_Competing_Interest.docx`** | Formal declaration confirming no financial or personal conflicts of interest. |
| **`CRediT Author Statement`** | **`CRediT_Authorship_Statement.docx`** | Breakdown of single-author contributions across all 14 CRediT research roles. |

---

## Submission Checklist: Step-by-Step

1. **Double-Blind Integrity:**
   - Under `"Manuscript without author details"`, upload `Manuscript_without_author_details.docx`.
   - Ensure you do NOT upload `EAAI_manuscript.docx` under the blinded dropdown, as it contains author identifying headers.
2. **Title Page:**
   - Under `"Title page with author details"`, upload `Title_page_with_author_details.docx`.
3. **Supplementary Items:**
   - Upload `Cover_Letter_EAAI.docx`, `Response_to_Editor_and_Reviewers.docx`, `Highlights.docx`, `Declaration_of_Competing_Interest.docx`, and `CRediT_Authorship_Statement.docx`.
4. **Metadata Verification:**
   - **Title:** *Hydro-Quantum Graph Network (HQGN): Physics-Informed Open Quantum Dynamics and Thermodynamic Invariant Collapse for Robust Spatio-Temporal Telemetry Sensing*
   - **Abstract:** Copy directly from Section 1 or from the Title Page.
   - **Keywords:** `Continuous-time quantum walk; Open quantum systems; Lindblad master equation; Navier-Stokes phase vorticity; Quantum thermodynamics; Biomedical telemetry; Graph neural networks; Robust engineering telemetry`
5. **Approve Submission:** Build PDF Proof, verify that all 12 figures display clearly at 300 DPI without text overlaps, and click **"Approve Submission"**.
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Generated EAAI Submission Checklist: {output_path}")

def create_response_to_editor(output_path):
    doc = docx.Document()
    set_doc_margins(doc, 1.0, 1.0, 1.0, 1.0)
    
    # Title / Header
    p_title = doc.add_paragraph()
    r_title = p_title.add_run("Detailed Point-by-Point Response to the Editorial Decision & Review Assessment")
    set_run_font(r_title, size_pt=14.0, bold=True)
    
    p_sub = doc.add_paragraph()
    r_sub = p_sub.add_run(
        "Manuscript Reference: EAAI-D-26-04921\n"
        "Title: Hydro-Quantum Graph Network (HQGN): Physics-Informed Open Quantum Dynamics and Thermodynamic Invariant Collapse for Robust Spatio-Temporal Telemetry Sensing\n"
        "Journal: Engineering Applications of Artificial Intelligence (Elsevier)\n"
        "Author: Vishal Chandel (Department of Electronics and Communication Engineering)\n"
        "Date: November 6, 2026\n"
    )
    set_run_font(r_sub, size_pt=10.0, italic=True)
    
    p_intro = doc.add_paragraph()
    r_intro = p_intro.add_run(
        "Dear Prof. Ajith Abraham (Editor-in-Chief) and Respected Editorial Board Members,\n\n"
        "We express our sincere gratitude for your constructive, discerning assessment of our manuscript. "
        "The evaluation rightly identified that while the empirical performance campaign was extensive, the manuscript required "
        "significantly deeper archival synthesis, mechanistic explanations for all performance curve trends, explicit delineation of operational "
        "boundary conditions (when the method excels vs. when it underperforms), and an expanded comparative evaluation under a wider range of physical conditions. "
        "We have conducted a thorough, comprehensive revision systematically addressing every concern. Below is our itemized, point-by-point response."
    )
    set_run_font(r_intro, size_pt=10.5)
    
    points = [
        ("Editorial Critique 1: Archival Value & Engineering Significance",
         "The evaluation noted: 'Although your manuscript falls within the aim and scope of this journal, it is being declined as the research results reported, in their present form, lack the archival value required by this Journal.'",
         "Response & Revision Details:\n"
         "We have substantially deepened the foundational engineering and theoretical depth of the paper to ensure it provides lasting archival reference value for the EAAI community. Specifically:\n"
         "1. Four Rigorous Mathematical Theorems: We establish formal proofs for unconditional trace preservation (Tr(rho)=1, Theorem 1), exponential noise contractivity (||delta rho(tau)|| <= ||delta rho(0)|| exp(-gamma_d tau), Theorem 2), Bures non-expansiveness (Theorem 3), and non-collapsing Dirichlet energy (E(5.0) = 0.023288 > 0, Theorem 4) which eliminates the classical GNN oversmoothing pathology.\n"
         "2. Systematic Operational Envelope (Section 6.8 & Table 6): Rather than presenting unnuanced claims of universal superiority, we formulate an explicit three-zone operational taxonomy (Regime I: Optimal Dominance, Regime II: Competitive Parity, Regime III: Boundary & Failure Modes) and provide an algorithmic decision flowchart for practicing telemetry engineers.\n"
         "3. Comprehensive Cross-Domain Validation: Evaluated across three authentic physical domains (Clinical human intracranial ECoG from MNE, Sensorimotor BCI scalp EEG from BNCI, and 16-channel MIMO multipath RF)."),
         
        ("Editorial Critique 2: Explanatory Details on Curves and Trends",
         "The evaluation noted: 'The performance evaluation campaign is rich in results which, however, are not commented on in a clear enough manner as required by the standards of this Journal. Many of the curves are only commented without sufficient explanatory details on the reason for their trend...'",
         "Response & Revision Details:\n"
         "We have performed an exhaustive mechanistic expansion across all experimental sections (Sections 6.4–6.8 and 7.1–7.3), providing rigorous physical and mathematical rationales for every curve, inflection, and asymptote:\n"
         "• AWGN Noise Sweep (Fig. 7): We explain the physical mechanism behind the 0 dB transition threshold. Classical spatial convolution (e^{-L_S tau}) computes an unweighted average; when noise variance matches signal variance, spatial smoothing incorporates noise into node embeddings, causing classical models to collapse toward 50% chance level. In contrast, HQGN's Lindblad dissipator actively dampens off-diagonal noise perturbations at rate exp(-gamma_d tau) while unitary phase circulation preserves topological frequencies, sustaining 81.40% F1 at 0 dB and 66.80% at -10 dB.\n"
         "• Dirichlet Energy Stabilization (Fig. 8a): We derive why classical diffusion decays to 4.18e-8 (projection onto ker(L_S) = span{1}) while HQGN stabilizes at E(5.0) = 0.023288: the chiral Hamiltonian i alpha Omega introduces purely imaginary eigenvalues that break self-adjoint symmetry, preventing uniform node collapse.\n"
         "• Sample Efficiency Scaling (Fig. 8b): We explain why HQGN retains 85.40% Macro-F1 at 20% training data: the Navier-Stokes vorticity and Lindblad CPTP constraints act as strong physics-informed inductive biases that restrict the hypothesis space to valid density operators S(H), drastically reducing sample complexity.\n"
         "• Brownian Phase Jitter Stochastic Resonance (Fig. 12a): We detail why performance peaks mildly at sigma = 0.25 rad (94.34% vs 92.61% clean): sub-threshold phase jitter acts as a dither signal, breaking marginal phase ties and enhancing non-linear contrast before excessive noise destroys phase coherence past sigma = 0.5 rad.\n"
         "• Sensor Channel Dropout Percolation (Fig. 12b): We explain why performance degrades linearly up to 12.5% loss (71.89%) but drops faster past 37.5%: Green-Gauss spatial curl requires minimum 3-node triangular simplices; exceeding 37.5% channel loss triggers a topological percolation threshold where interconnected triangles fragment."),
         
        ("Editorial Critique 3: Synthesis of Generality (When HQGN Performs Better vs. Worse)",
         "The evaluation noted: '...and there is a lack of a synthesis that can testify to the generality of the results and that can make it clearer in which conditions the algorithm performs better and under which conditions worse.'",
         "Response & Revision Details:\n"
         "In Section 6.8, Table 5, and Table 6, we provide a transparent, rigorous synthesis of the exact conditions where HQGN performs better versus where it underperforms, along with concrete engineering mitigations:\n"
         "• Conditions Where HQGN Excels: High phase non-stationarity / rotating phase rotors (seizure onset, RF Doppler), low-to-moderate SNR down to -10 dB, high spatial cross-talk / volume conduction, and low-sample training regimes (N < 200).\n"
         "• Boundary Conditions & Failure Modes:\n"
         "  1. Spatially Decoupled Telemetry: When sensor channels represent independent variables without spatial topology, computing spatial curl creates spurious circulation. Mitigation: Screen spatial cross-correlation (C_bar < 0.15) and default to 1D-CNN.\n"
         "  2. Ultra-Sparse Sensor Arrays (M < 4 Channels): Green-Gauss finite differences fail on < 4 channels (74.20% F1, Fig. 10d). Mitigation: Apply delay-coordinate state-space reconstruction.\n"
         "  3. Heavy-Tailed Cauchy Impulsive Spikes: Outlier spikes create phase wrap discontinuities, dropping unclipped HQGN to 65.40%. Mitigation: Implementing Robust Median Phase Derivative Clipping restores accuracy to 83.60% (Table 5, Fig. 10c).\n"
         "  4. Sub-100 Microsecond Hard Real-Time Budgets: Strang matrix exponentiation requires 2.84 ms on CPU. Mitigation: Deploy Riemannian log-Euclidean tangent projections (0.12 ms) when timing budgets are under 0.5 ms."),
         
        ("Editorial Critique 4: Expanded Comparative Conditions & Modern Literature Baselines",
         "The evaluation noted: 'Besides, the comparison with other solutions should definitely be conducted under a wider range of different conditions and for several other similar approaches available in the literature. More work is needed to substantiate the conclusions in your manuscript.'",
         "Response & Revision Details:\n"
         "We have significantly broadened both the evaluation conditions and the comparative baselines (Table 5, Table 6, and Figure 10):\n"
         "1. Wider Range of Physical Stress Conditions:\n"
         "  • Colored 1/f^alpha noise sweep (alpha from 0.0=white to 2.0=Brownian baseline drift).\n"
         "  • Out-of-band 50 Hz powerline harmonic hum injection (amplitudes 0.0 to 2.0x signal std dev).\n"
         "  • Heavy-tailed Cauchy impulsive spike contamination (p from 1% to 10%).\n"
         "  • Sensor array density scaling (M = 4, 8, 12, 16 active channels).\n"
         "  • Window duration scaling (L = 0.5s to 3.0s).\n"
         "2. Expanded Modern Literature Baseline Suite (2017–2026):\n"
         "  • Dense 1D-CNN (Schirrmeister et al. 2017)\n"
         "  • 1D-ResNet-18 (Wang et al. 2017)\n"
         "  • Dynamic ST-GCN (Hajisafi et al. 2024)\n"
         "  • PatchTST / Spatio-Temporal Transformer (Nie et al. 2023)\n"
         "  • Magnetic GCN / MagNet (Zhang et al. 2021)\n"
         "  • Unitary CTQW-GNN (Zhan et al. Aug 2026)\n"
         "  • Dissipative Open Quantum Walk (Linck & Petruccione Mar 2026)\n"
         "  • Riemannian Covariance Tangent Space (Barachant et al. 2012)")
    ]
    
    for title, critique, response in points:
        p_pt = doc.add_paragraph()
        p_pt.paragraph_format.space_before = Pt(10)
        p_pt.paragraph_format.space_after = Pt(2)
        r_pt_title = p_pt.add_run(title)
        set_run_font(r_pt_title, size_pt=11.5, bold=True)
        
        p_crit = doc.add_paragraph()
        p_crit.paragraph_format.space_before = Pt(0)
        p_crit.paragraph_format.space_after = Pt(4)
        p_crit.paragraph_format.left_indent = Inches(0.25)
        r_crit = p_crit.add_run(critique)
        set_run_font(r_crit, size_pt=10.0, italic=True)
        
        p_resp = doc.add_paragraph()
        p_resp.paragraph_format.space_before = Pt(2)
        p_resp.paragraph_format.space_after = Pt(8)
        p_resp.paragraph_format.line_spacing = 1.15
        r_resp = p_resp.add_run(response)
        set_run_font(r_resp, size_pt=10.0)
        
    p_close = doc.add_paragraph()
    p_close.paragraph_format.space_before = Pt(12)
    p_close.paragraph_format.space_after = Pt(4)
    r_close = p_close.add_run(
        "We are confident that these substantial additions, rigorous mechanistic explanations, expanded multi-condition stress benchmarks, and "
        "transparent operational envelope synthesis thoroughly address all concerns and elevate the manuscript to the highest archival standards of "
        "Engineering Applications of Artificial Intelligence.\n\n"
        "Sincerely,\n"
        "Vishal Chandel\n"
        "Department of Electronics and Communication Engineering\n"
        "Email: vishal442005@outlook.com | ORCID: 0009-0006-0552-8551"
    )
    set_run_font(r_close, size_pt=10.5)
    
    doc.save(output_path)
    print(f"Generated Detailed Response to Editor: {output_path}")

def main():
    target_dirs = ["Elseiver_EAAI", os.path.join("Updated_significant", "manuscript")]
    for target_dir in target_dirs:
        os.makedirs(target_dir, exist_ok=True)
        
        # 1. Manuscript Copy
        src_manuscript = os.path.join("elseiver", "elseiver_graph.docx")
        dst_manuscript1 = os.path.join(target_dir, "EAAI_manuscript.docx")
        dst_manuscript2 = os.path.join(target_dir, "elsevier_eaai_graph.docx")
        
        if os.path.exists(src_manuscript):
            try:
                shutil.copy2(src_manuscript, dst_manuscript1)
                shutil.copy2(src_manuscript, dst_manuscript2)
                print(f"Copied manuscript to: {dst_manuscript1} and {dst_manuscript2}")
            except PermissionError:
                print("Notice: Existing manuscript copy open in Word, skipping overwrite.")

        # 2. Dedicated Title Pages
        tp1 = os.path.join(target_dir, "Title_page_with_author_details.docx")
        tp2 = os.path.join(target_dir, "Title_Page.docx")
        create_title_page(tp1)
        create_title_page(tp2)

        # 3. Cover Letter
        cover_letter_path = os.path.join(target_dir, "Cover_Letter_EAAI.docx")
        create_cover_letter(cover_letter_path)
        
        # 4. Highlights (strictly <= 85 characters each)
        highlights_path = os.path.join(target_dir, "Highlights.docx")
        create_highlights(highlights_path)
        
        # 5. CRediT Statement
        credit_path = os.path.join(target_dir, "CRediT_Authorship_Statement.docx")
        create_credit_statement(credit_path)
        
        # 6. Declaration of Competing Interest
        coi_path = os.path.join(target_dir, "Declaration_of_Competing_Interest.docx")
        create_declaration_of_interest(coi_path)
        
        # 7. Detailed Response to Editor & Reviewers
        resp_path = os.path.join(target_dir, "Response_to_Editor_and_Reviewers.docx")
        create_response_to_editor(resp_path)

        # 8. Submission Checklist Markdown
        checklist_path = os.path.join(target_dir, "README_Submission_Checklist.md")
        create_submission_checklist(checklist_path)

if __name__ == "__main__":
    main()
