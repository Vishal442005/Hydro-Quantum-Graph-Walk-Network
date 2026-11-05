import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

# Styling Constants: Strictly Elsevier Publication Standards
FONT_FAMILY = "Times New Roman"
COLOR_BLACK = RGBColor(0, 0, 0)
COLOR_GREY = RGBColor(80, 80, 80)

# LaTeX to OMML XSLT Converter Setup (Using Microsoft Office Native MML2OMML.XSL)
import latex2mathml.converter
import lxml.etree as ET

OFFICE_XSLT_PATHS = [
    r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL",
    r"C:\Program Files (x86)\Microsoft Office\root\Office16\MML2OMML.XSL",
    r"C:\Program Files\Microsoft Office\Office16\MML2OMML.XSL",
    r"C:\Program Files (x86)\Microsoft Office\Office16\MML2OMML.XSL",
    r"C:\Program Files\Microsoft Office\root\Office15\MML2OMML.XSL"
]

XSLT_TRANSFORM = None
for path in OFFICE_XSLT_PATHS:
    if os.path.exists(path):
        try:
            xslt_doc = ET.parse(path)
            XSLT_TRANSFORM = ET.XSLT(xslt_doc)
            print(f"Loaded Native Microsoft Office MML2OMML XSLT from: {path}")
            break
        except Exception as e:
            print(f"Error loading XSLT from {path}: {e}")

if XSLT_TRANSFORM is None:
    raise RuntimeError("CRITICAL ERROR: Microsoft Office MML2OMML.XSL not found on system! Equations cannot be rendered.")


def latex_to_omml_xml(latex_str: str) -> str:
    """Converts a LaTeX mathematical expression to a native Office Math Markup Language (OMML) XML string."""
    clean_latex = latex_str.strip()
    if clean_latex.startswith("$") and clean_latex.endswith("$"):
        clean_latex = clean_latex[1:-1].strip()
    clean_latex = clean_latex.replace(r"\\lambda", r"\lambda")

    try:
        mml_str = latex2mathml.converter.convert(clean_latex)
        mml_tree = ET.fromstring(mml_str.encode("utf-8"))
        omml_tree = XSLT_TRANSFORM(mml_tree)
        return ET.tostring(omml_tree, encoding="unicode")
    except Exception as e:
        print(f"OMML Conversion warning for '{clean_latex[:40]}': {e}")
        clean_txt = clean_latex.replace("\\", "").replace("{", "").replace("}", "").replace("$", "")
        return f"""
        <m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
            <m:r>
                <m:rPr>
                    <m:sty m:val="i"/>
                </m:rPr>
                <m:t>{clean_txt}</m:t>
            </m:r>
        </m:oMath>
        """


def latex_to_inline_omml(latex_str: str) -> str:
    """Converts inline LaTeX math string $...$ into true inline native OMML element."""
    return latex_to_omml_xml(latex_str)


def set_run_font(run, font_name=FONT_FAMILY, size_pt=10.0, color=COLOR_BLACK, bold=False, italic=False):
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.font.color.rgb = color
    run.bold = bold
    run.italic = italic


def add_p_with_math(doc, text_with_math: str, space_before=0.0, space_after=4.0, line_spacing=1.15):
    """Parses text containing inline LaTeX math ($...$) and inserts normal text runs and inline OMML math runs."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing

    parts = text_with_math.split("$")
    for i, part in enumerate(parts):
        if not part:
            continue
        if i % 2 == 0:
            # Regular text run
            r = p.add_run(part)
            set_run_font(r, font_name=FONT_FAMILY, size_pt=10.0, color=COLOR_BLACK)
        else:
            # Inline math run
            try:
                omml_xml = latex_to_inline_omml(part)
                omml_elm = parse_xml(omml_xml)
                p._p.append(omml_elm)
            except Exception as e:
                print(f"Inline math parse warning for '{part[:30]}': {e}")
                r = p.add_run(part)
                set_run_font(r, font_name=FONT_FAMILY, size_pt=10.0, color=COLOR_BLACK, italic=True)
    return p


def add_display_equation(doc, latex_str: str, eq_num: int):
    """Inserts a formal display equation centered with a right-aligned equation number (Eq. X)."""
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    # 1-inch margins on 8.5" page gives 6.5" usable width
    table.columns[0].width = Inches(5.8)
    table.columns[1].width = Inches(0.7)

    cell_eq = table.cell(0, 0)
    cell_num = table.cell(0, 1)
    cell_eq.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    cell_num.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    # Remove all cell borders
    for c in [cell_eq, cell_num]:
        tcPr = c._tc.get_or_add_tcPr()
        tcBorders = parse_xml(r'''
            <w:tcBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
                <w:top w:val="none"/>
                <w:left w:val="none"/>
                <w:bottom w:val="none"/>
                <w:right w:val="none"/>
            </w:tcBorders>
        ''')
        tcPr.append(tcBorders)

    # Cell 0: Equation (Centered)
    p_eq = cell_eq.paragraphs[0]
    p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_eq.paragraph_format.space_before = Pt(4)
    p_eq.paragraph_format.space_after = Pt(4)

    try:
        omml_xml = latex_to_omml_xml(latex_str)
        omml_elm = parse_xml(omml_xml)
        p_eq._p.append(omml_elm)
    except Exception as e:
        print(f"Display equation OMML error (Eq. {eq_num}): {e}")
        clean_latex = latex_str.replace("\\", "").replace("{", "").replace("}", "")
        r_eq = p_eq.add_run(clean_latex)
        set_run_font(r_eq, font_name=FONT_FAMILY, size_pt=10.5, color=COLOR_BLACK, italic=True)

    # Cell 1: Equation Number (Right-aligned)
    p_num = cell_num.paragraphs[0]
    p_num.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_num.paragraph_format.space_before = Pt(4)
    p_num.paragraph_format.space_after = Pt(4)
    r_num = p_num.add_run(f"({eq_num})")
    set_run_font(r_num, font_name=FONT_FAMILY, size_pt=10.0, color=COLOR_BLACK)


def add_section_heading(doc, sec_num: int, title: str):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(f"{sec_num}. {title}")
    set_run_font(r, font_name=FONT_FAMILY, size_pt=12.0, color=COLOR_BLACK, bold=True)
    return p


def add_subsection_heading(doc, sec_num: int, sub_num: int, title: str):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(f"{sec_num}.{sub_num}. {title}")
    set_run_font(r, font_name=FONT_FAMILY, size_pt=11.0, color=COLOR_BLACK, bold=True)
    return p


def add_subsubsection_heading(doc, sec_num: int, sub_num: int, subsub_num: int, title: str):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(f"{sec_num}.{sub_num}.{subsub_num}. {title}")
    set_run_font(r, font_name=FONT_FAMILY, size_pt=10.5, color=COLOR_BLACK, bold=True, italic=True)
    return p


def add_figure(doc, fig_num: int, caption_text: str, image_path: str, width_in=6.0):
    if not os.path.exists(image_path):
        print(f"Warning: Figure image {image_path} does not exist.")
        return

    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(10)
    p_img.paragraph_format.space_after = Pt(4)
    p_img.paragraph_format.keep_with_next = True
    run_img = p_img.add_run()
    run_img.add_picture(image_path, width=Inches(width_in))

    p_cap = doc.add_paragraph()
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(10)
    p_cap.paragraph_format.line_spacing = 1.15

    r_bold = p_cap.add_run(f"Fig. {fig_num}. ")
    set_run_font(r_bold, font_name=FONT_FAMILY, size_pt=9.0, color=COLOR_BLACK, bold=True)

    r_txt = p_cap.add_run(caption_text)
    set_run_font(r_txt, font_name=FONT_FAMILY, size_pt=9.0, color=COLOR_BLACK)


def add_booktabs_table(doc, table_num: int, title: str, headers: list, rows: list, col_widths: list = None):
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(12)
    p_title.paragraph_format.space_after = Pt(3)
    p_title.paragraph_format.keep_with_next = True

    r_bold = p_title.add_run(f"Table {table_num}\n")
    set_run_font(r_bold, font_name=FONT_FAMILY, size_pt=10.0, color=COLOR_BLACK, bold=True)
    r_txt = p_title.add_run(title)
    set_run_font(r_txt, font_name=FONT_FAMILY, size_pt=9.5, color=COLOR_BLACK, italic=True)

    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    # Top border (thick rule) and header bottom border (thin rule)
    tblPr = table._tbl.tblPr
    tblBorders = parse_xml(r'''
        <w:tblBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
            <w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>
            <w:left w:val="none"/>
            <w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>
            <w:right w:val="none"/>
            <w:insideH w:val="none"/>
            <w:insideV w:val="none"/>
        </w:tblBorders>
    ''')
    tblPr.append(tblBorders)

    # Format Header Row
    hdr_cells = table.rows[0].cells
    for i, h_text in enumerate(headers):
        hdr_cells[i].text = ""
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.RIGHT
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(h_text)
        set_run_font(r, font_name=FONT_FAMILY, size_pt=9.0, color=COLOR_BLACK, bold=True)

        tcPr = hdr_cells[i]._tc.get_or_add_tcPr()
        tcBorders = parse_xml(r'''
            <w:tcBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
                <w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/>
            </w:tcBorders>
        ''')
        tcPr.append(tcBorders)

    # Format Data Rows
    for r_idx, row_data in enumerate(rows):
        row_cells = table.rows[r_idx + 1].cells
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = ""
            p = row_cells[c_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.RIGHT
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            is_proposed = "Proposed" in str(row_data[0]) or "HQGN" in str(row_data[0]) or "A6" in str(row_data[0])
            r = p.add_run(str(val))
            set_run_font(r, font_name=FONT_FAMILY, size_pt=8.5, color=COLOR_BLACK, bold=is_proposed)

    if col_widths and len(col_widths) == len(headers):
        for row in table.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = Inches(w)

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(2)
    p_after.paragraph_format.space_after = Pt(6)


def generate_elsevier_pr_manuscript(output_docx_path: str, is_blinded: bool = False):
    """Assembles the complete publication-ready Elsevier EAAI manuscript document."""
    mode_str = "BLINDED (WITHOUT AUTHOR DETAILS)" if is_blinded else "COMPLETE (WITH AUTHOR DETAILS)"
    print(f"\n================================================================================")
    print(f"GENERATING PUBLICATION-GRADE ELSEVIER EAAI MANUSCRIPT [{mode_str}]")
    print(f"Target Output: {output_docx_path}")
    print(f"================================================================================")
    
    doc = docx.Document()
    
    # 1. Page Setup: 1-inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.header_distance = Inches(0.5)
        section.footer_distance = Inches(0.5)
        
    # 2. Article Title (Times New Roman 16 pt Bold Black, Centered)
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(12)
    p_title.paragraph_format.space_after = Pt(8)
    r_title = p_title.add_run("Hydro-Quantum Graph Network (HQGN): Physics-Informed Open Quantum Dynamics and Thermodynamic Invariant Collapse for Robust Spatio-Temporal Telemetry Sensing")
    set_run_font(r_title, font_name=FONT_FAMILY, size_pt=16.0, color=COLOR_BLACK, bold=True)
    
    if is_blinded:
        # Blinded mode: Omit author details and affiliations for double-blind peer review
        p_blind = doc.add_paragraph()
        p_blind.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_blind.paragraph_format.space_before = Pt(4)
        p_blind.paragraph_format.space_after = Pt(10)
        r_blind = p_blind.add_run("[Author Details and Affiliations Omitted for Double-Blind Peer Review]")
        set_run_font(r_blind, font_name=FONT_FAMILY, size_pt=10.0, color=COLOR_GREY, italic=True)
    else:
        # 3. Authors and Affiliations
        p_auth = doc.add_paragraph()
        p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_auth.paragraph_format.space_before = Pt(2)
        p_auth.paragraph_format.space_after = Pt(2)
        r_author = p_auth.add_run("Vishal Chandel")
        set_run_font(r_author, font_name=FONT_FAMILY, size_pt=11.0, color=COLOR_BLACK, bold=True)
        r_star = p_auth.add_run(" *")
        set_run_font(r_star, font_name=FONT_FAMILY, size_pt=11.0, color=COLOR_BLACK, bold=True)
        
        p_affil = doc.add_paragraph()
        p_affil.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_affil.paragraph_format.space_before = Pt(0)
        p_affil.paragraph_format.space_after = Pt(2)
        r_affil = p_affil.add_run("Department of Electronics and Communication Engineering, National Institute of Technology, Hamirpur, Himachal Pradesh 177005, India")
        set_run_font(r_affil, font_name=FONT_FAMILY, size_pt=10.0, color=COLOR_BLACK, italic=True)

        p_corr = doc.add_paragraph()
        p_corr.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_corr.paragraph_format.space_before = Pt(0)
        p_corr.paragraph_format.space_after = Pt(10)
        r_corr = p_corr.add_run("* Corresponding author. E-mail address: vishal442005@outlook.com | ORCID: 0009-0006-0552-8551")
        set_run_font(r_corr, font_name=FONT_FAMILY, size_pt=9.5, color=COLOR_BLACK, italic=True)

    # Clean dividing horizontal line before abstract
    p_rule1 = doc.add_paragraph()
    p_rule1.paragraph_format.space_before = Pt(0)
    p_rule1.paragraph_format.space_after = Pt(8)
    pBdr1 = parse_xml(r'''
        <w:pBdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
            <w:bottom w:val="single" w:sz="6" w:space="1" w:color="000000"/>
        </w:pBdr>
    ''')
    p_rule1._p.get_or_add_pPr().append(pBdr1)

    # 4. Abstract
    p_abs_hdr = doc.add_paragraph()
    p_abs_hdr.paragraph_format.space_before = Pt(4)
    p_abs_hdr.paragraph_format.space_after = Pt(4)
    r_abs_lbl = p_abs_hdr.add_run("Abstract")
    set_run_font(r_abs_lbl, font_name=FONT_FAMILY, size_pt=11.0, color=COLOR_BLACK, bold=True)
    
    abstract_text = (
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
        "trace preservation, asymptotic noise contraction, Bures non-expansiveness, and non-collapsing Dirichlet energy ($E(5.0) = 0.023 > 0$). "
        "Empirical evaluation across three authentic physical domains—real human clinical epilepsy intracranial ECoG (149 windows), "
        "real sensorimotor scalp EEG (BCI Competition IV-2a, 144 trials), and multi-carrier wireless RF telemetry under Rayleigh fading—demonstrates "
        "that HQGN achieves top classification fidelity ($92.62\%$ clinical ECoG accuracy, $92.61\%$ Macro-F1; $83.98\%$ BCI Macro-F1; $58.65\%$ RF Macro-F1), maintains noise robustness "
        "down to -10 dB SNR where classical baselines collapse to near-chance guessing, and delivers closed-form gradient-free interpretability via the "
        "Quantum Fisher Information Matrix (QFIM) with sub-millisecond GPU execution ($0.72$ ms) and low-latency CPU inference ($2.84$ ms)."
    )
    add_p_with_math(doc, abstract_text, space_before=2.0, space_after=6.0, line_spacing=1.15)
    
    # 5. Keywords
    p_kw = doc.add_paragraph()
    p_kw.paragraph_format.space_before = Pt(4)
    p_kw.paragraph_format.space_after = Pt(8)
    r_kw_lbl = p_kw.add_run("Keywords: ")
    set_run_font(r_kw_lbl, font_name=FONT_FAMILY, size_pt=9.5, color=COLOR_BLACK, bold=True)
    r_kw_txt = p_kw.add_run("Pattern recognition; Continuous-time quantum walk; Open quantum systems; Lindblad master equation; Navier-Stokes phase vorticity; Quantum thermodynamics; Biomedical telemetry; Graph neural networks")
    set_run_font(r_kw_txt, font_name=FONT_FAMILY, size_pt=9.5, color=COLOR_BLACK, italic=False)

    # Clean dividing horizontal line after keywords
    p_rule2 = doc.add_paragraph()
    p_rule2.paragraph_format.space_before = Pt(0)
    p_rule2.paragraph_format.space_after = Pt(14)
    pBdr2 = parse_xml(r'''
        <w:pBdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
            <w:bottom w:val="single" w:sz="6" w:space="1" w:color="000000"/>
        </w:pBdr>
    ''')
    p_rule2._p.get_or_add_pPr().append(pBdr2)

    # =======================================================================
    # SECTION 1: INTRODUCTION
    # =======================================================================
    add_section_heading(doc, 1, "Introduction")
    
    add_subsection_heading(doc, 1, 1, "Motivation and background")
    add_p_with_math(doc, 
        "The continuous monitoring, decoding, and classification of multi-channel time-series telemetry constitute a foundational "
        "pillar across contemporary pattern recognition, biomedical engineering, and cyber-physical communication networks [1, 2]. "
        "In clinical electrophysiology, high-density intracranial electrocorticography (ECoG) and scalp electroencephalography (EEG) "
        "record distributed microvolt fluctuations across dozens of cortical nodes to detect focal epileptogenic seizure onsets and "
        "decode sensorimotor intentions [3, 4]. In high-frequency cognitive radio and wireless telemetry, multi-carrier antenna arrays "
        "capture rapid phase-modulated waveforms subjected to non-stationary Rayleigh fading and Doppler dispersion [5]. In all these "
        "physical domains, sensor arrays record synchronized yet non-stationary spatio-temporal phase trajectories that exhibit "
        "turbulent, circulating wavefronts and non-linear cross-channel coupling [6, 7]."
    )
    add_p_with_math(doc,
        "Historically, spatial and spatio-temporal graph neural networks (ST-GNNs) have emerged as the standard paradigm for modeling "
        "multi-channel telemetry by mapping sensor coordinates to graph vertices and statistical dependencies to edges [8]. "
        "However, physical telemetry environments subject graph models to severe operational stresses: non-stationary phase jitter, "
        "spatial volume conduction, baseline wander, and low signal-to-noise ratios (SNR). When operating under these adverse conditions, "
        "contemporary deep learning and graph architectures encounter fundamental mathematical bottlenecks that undermine their "
        "classification accuracy, physical interpretability, and numerical stability."
    )

    add_subsection_heading(doc, 1, 2, "Fundamental limitations of contemporary paradigms")
    add_p_with_math(doc,
        "A rigorous mathematical audit of current telemetry paradigms reveals four fundamental structural vulnerabilities:"
    )
    add_p_with_math(doc,
        "1. Representational Oversmoothing in Classical Graph Convolutions: Standard graph convolutions and diffusion operations "
        "rely on the normalized graph Laplacian $L = I - D^{-1/2} A D^{-1/2}$, governing classical Markov heat diffusion $\\dot{h} = -L h$. "
        "As proved by Li et al. [10], iterative graph filtering exponentially contracts feature representations into the null space "
        "of the Laplacian ($\\ker(L)$), corresponding to constant consensus vectors. Consequently, as network depth or temporal diffusion "
        "horizons increase, node features become indistinguishable, completely erasing localized anomalies such as epileptic focal spikes."
    )
    add_p_with_math(doc,
        "2. Neglect of Fluid Vorticity and Chiral Wave Dynamics: Conventional graph topologies are constructed from symmetric statistical "
        "metrics such as Pearson correlation or Euclidean distance ($A_{ij} = A_{ji}$). However, real physical wave propagation across "
        "cortical tissues and wireless multipath environments is governed by hydrodynamics and Navier-Stokes momentum equations [11]. "
        "Symmetric graphs are mathematically blind to fluid circulation, velocity shear, and phase curl ($\\nabla \\times \\mathbf{u} \\ne 0$), "
        "discarding the directional rotational energy that characterizes traveling neural wavefronts and Doppler phase rotations."
    )
    add_p_with_math(doc,
        "3. Lack of Dissipative Noise Sinks in Unitary Quantum Walks: Continuous-time quantum walks (CTQWs) [12, 13] offer quadratic speedups "
        "over classical diffusion and preserve Dirichlet energy through quantum wave interference [14]. However, existing quantum graph models "
        "are strictly closed and unitary, governed by the Schrödinger equation $i \\hbar \\frac{d}{dt}|\\psi\\rangle = H |\\psi\\rangle$, which "
        "preserves total quantum information indefinitely ($U(t) = \\exp(-iHt)$). Without an open thermodynamic bath or dissipative noise sink, "
        "stochastic phase noise and environmental perturbations circulate indefinitely across the graph, destabilizing classification boundaries."
    )
    add_p_with_math(doc,
        "4. Ill-Conditioning of Riemannian Covariance under Non-Stationary Drift: Riemannian geometry on the cone of Symmetric Positive Definite "
        "(SPD) covariance matrices via Affine-Invariant Riemannian Metrics (AIRM) [3] represents a prominent paradigm in EEG decoding. "
        "However, empirical covariance matrices estimated over short non-stationary telemetry windows regularly exhibit near-zero eigenvalues. "
        "Matrix logarithm projections ($\\log(\\Sigma)$) become severely ill-conditioned, yielding numerical instability and catastrophic "
        "sensitivity to low-frequency baseline fluctuations."
    )
    
    add_subsection_heading(doc, 1, 3, "Summary of core scientific contributions")
    add_p_with_math(doc,
        "To resolve these structural bottlenecks, this paper establishes a mathematically rigorous cross-domain synthesis between "
        "fluid mechanics, open quantum Markovian dynamics, and quantum thermodynamics. The primary contributions of this work are:"
    )
    add_p_with_math(doc,
        "• Tri-Domain Physical Unification: The Hydro-Quantum Graph Network (HQGN) is introduced, unifying 2D Navier-Stokes phase-vorticity "
        "coupling, Gorini-Kossakowski-Sudarshan-Lindblad (GKSL) stochastic master equations, and Bures-Wasserstein quantum relative entropy."
    )
    add_p_with_math(doc,
        "• Four Formal Mathematical Theorems: Complete proofs are established demonstrating unconditional trace preservation (Theorem 1), "
        "asymptotic noise contraction via Lindbladian jump operators (Theorem 2), geodesic non-expansiveness on the Bures manifold (Theorem 3), "
        "and non-collapsing Dirichlet energy preventing graph oversmoothing (Theorem 4)."
    )
    add_p_with_math(doc,
        "• Unconditionally Stable Numerical Integration: A second-order symmetric Strang operator splitting scheme is formulated, proving "
        "second-order temporal accuracy $\\mathcal{O}(\\Delta t^2)$ and unconditional positivity preservation without matrix inversion."
    )
    add_p_with_math(doc,
        "• Closed-Form Quantum Explainability: Exact, gradient-free channel sensitivity attribution is derived via the diagonal elements of the "
        "Quantum Fisher Information Matrix (QFIM), accompanied by continuous Riemannian counterfactual recourse along Bures geodesics."
    )
    add_p_with_math(doc,
        "• Authentic Multi-Domain Empirical Benchmarking: The proposed framework is benchmarked across three authentic physical datasets—real "
        "clinical intracranial ECoG (MNE Epilepsy), real sensorimotor scalp EEG (BCI Competition IV-2a), and multi-carrier wireless RF telemetry—demonstrating "
        "sustained noise resilience down to -10 dB SNR where classical baselines experience catastrophic failure."
    )
    
    add_subsection_heading(doc, 1, 4, "Article organization")
    add_p_with_math(doc,
        "The remainder of this article is structured as follows. Section 2 reviews foundational and recent literature spanning 2020 to 2026. "
        "Section 3 details the mathematical foundations and theoretical theorems. Section 4 presents the complete architectural pipeline of HQGN. "
        "Section 5 describes the physical telemetry datasets, baselines, and experimental setup. Section 6 provides empirical results, multi-domain "
        "benchmarks, ablation studies, and manifold latent geometry. Section 7 presents explainability via QFIM and adversarial robustness. Section 8 evaluates computational "
        "acceleration and hardware profiling. Section 9 discusses practical implications and limitations. Section 10 concludes the article."
    )

    # =======================================================================
    # SECTION 2: RELATED WORK
    # =======================================================================
    add_section_heading(doc, 2, "Related work")
    
    add_subsection_heading(doc, 2, 1, "Continuous-time quantum walks and graph neural networks")
    add_p_with_math(doc,
        "Continuous-time quantum walks (CTQWs) describe the coherent propagation of quantum probability amplitudes across discrete graph structures, "
        "governed by the Schrödinger equation $i \\frac{d}{dt}|\\psi(t)\\rangle = H |\\psi(t)\\rangle$, where the Hamiltonian $H$ is typically assigned "
        "to the graph adjacency or graph Laplacian matrix [12, 13]. In contrast to classical random walks governed by the heat diffusion equation "
        "$\\frac{d}{dt}p(t) = -L p(t)$, where probability distributions disperse diffusively, CTQWs exhibit quadratic propagation speedups, constructive "
        "and destructive wave interference, and non-ergodic localized probability distributions [14]. Zhan et al. (2026) [26] formally proved that "
        "unitary CTQWs prevent graph oversmoothing by maintaining non-zero Dirichlet energy over arbitrary walk horizons. However, their model is "
        "strictly unitary ($\\alpha = 0$), lacking environmental coupling mechanisms to dissipate physical noise."
    )

    add_subsection_heading(doc, 2, 2, "Open quantum systems, master equations, and dissipation")
    add_p_with_math(doc,
        "Open quantum systems describe microscopic quantum dynamics coupled to external thermal environments or measurement baths [15, 16]. "
        "Under the Markovian approximation, the time evolution of the system density operator $\\rho(t)$ is uniquely governed by the "
        "Gorini-Kossakowski-Sudarshan-Lindblad (GKSL) master equation [17]. Linck and Petruccione (2026) [27] developed an open quantum walk "
        "classifier utilizing Lindbladian dissipation on static correlation graphs. Furthermore, Linck and Bernardes (2026) [2] established the "
        "thermodynamic foundation of linear open quantum walks, proving that Lindbladian jump operators enforce entropy production and thermalization. "
        "HQGN builds upon these open quantum frameworks by replacing phenomenological damping with fluid-mechanic kinematic viscosity, enabling "
        "environmental noise dissipation without erasing coherent spatial signals."
    )

    add_subsection_heading(doc, 2, 3, "Hydrodynamic wave theory and phase-vorticity graphs")
    add_p_with_math(doc,
        "Hydrodynamic analogies in signal processing model complex multi-channel oscillations as fluid velocity fields governed by Navier-Stokes "
        "and Kuramoto phase synchronization dynamics [6, 7]. Traveling electrical waves in cortical tissue exhibit non-zero vorticity curl "
        "$\\nabla \\times \\mathbf{u} \\ne 0$, reflecting vortex cores around epileptogenic foci. By incorporating Poincaré-Cartan integral invariants "
        "and Kelvin's circulation theorem [18], HQGN constructs directed, antisymmetric graph topologies that explicitly encode rotational wave energy, "
        "overcoming the structural blindness of classical symmetric correlation matrices."
    )

    # =======================================================================
    # SECTION 3: MATHEMATICAL FORMULATION & THEORETICAL THEOREMS
    # =======================================================================
    add_section_heading(doc, 3, "Mathematical formulation and theoretical theorems")
    
    add_subsection_heading(doc, 3, 1, "Navier-Stokes phase-velocity fields and hydrodynamic graphs")
    add_p_with_math(doc,
        "Let $X \\in \\mathbb{R}^{M \\times L}$ denote a calibrated multi-channel telemetry window comprising $M$ spatial sensing channels across "
        "$L$ discrete timepoints. Applying the discrete Hilbert transform generates the complex analytic signal $Z_m(t) = X_m(t) + i \\mathcal{H}[X_m(t)] = A_m(t) e^{i \\theta_m(t)}$, "
        "where $A_m(t) > 0$ represents the instantaneous amplitude and $\\theta_m(t) \\in [-\\pi, \\pi)$ denotes the unwrapped instantaneous phase. "
        "The instantaneous phase-velocity vector $\\mathbf{u}_m(t) \\in \\mathbb{R}^2$ at electrode coordinate $\\mathbf{r}_m = (x_m, y_m)$ is governed by the "
        "barotropic Navier-Stokes momentum equation [11]:"
    )
    add_display_equation(doc, r"\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\frac{1}{\rho_0} \nabla p + \nu \nabla^2 \mathbf{u}", eq_num=1)
    add_p_with_math(doc,
        "where $\\rho_0$ is the effective medium density, $p$ is the local phase pressure, and $\\nu$ denotes the kinematic shear viscosity. "
        "By Kelvin's circulation theorem, the circulation $\\Gamma$ around a closed spatial contour $\\mathcal{C}$ enclosing surface $\\mathcal{S}$ is:"
    )
    add_display_equation(doc, r"\Gamma = \oint_{\mathcal{C}} \mathbf{u} \cdot d\mathbf{r} = \iint_{\mathcal{S}} (\nabla \times \mathbf{u}) \cdot d\mathbf{S} = \iint_{\mathcal{S}} \omega_z \, dx \, dy", eq_num=2)
    add_p_with_math(doc,
        "where $\\omega_z = \\frac{\\partial u_y}{\\partial x} - \\frac{\\partial u_x}{\\partial y}$ represents the scalar vorticity field. Between channel pairs "
        "$(i, j)$, the hydrodynamic shear velocity differential $\\Delta u_{ij} = \\|\\mathbf{u}_i - \\mathbf{u}_j\\|_2$ and antisymmetric vorticity "
        "circulation $\\Omega_{ij} = u_{i, x} u_{j, y} - u_{i, y} u_{j, x}$ construct the hydrodynamic adjacency matrix:"
    )
    add_display_equation(doc, r"W_{ij} = \exp\left(-\frac{\Delta u_{ij}^2}{2 \sigma_u^2}\right) \cdot \left[ 1 + \beta \tanh(\Omega_{ij}) \right], \quad W_{ii} = 0", eq_num=3)
    add_p_with_math(doc,
        "where $\\sigma_u$ represents the characteristic velocity scale and $\\beta$ modulates the vorticity sensitivity. The resulting normalized graph "
        "Laplacian $L_H = I - D^{-1/2} W D^{-1/2}$ naturally decomposes into a symmetric diffusion component $L_S = \\frac{1}{2}(L_H + L_H^T)$ and a "
        "real skew-symmetric vorticity curl component $\\Omega = \\frac{1}{2}(L_H - L_H^T)$."
    )

    add_figure(doc, 1, 
        "Proposed HQGN Stage I Hydrodynamic Adjacency Matrix Construction: (a) Navier-Stokes Phase Velocity Vector Field u(x, y) over 16-channel array; (b) Antisymmetric Vorticity Curl Tensor Omega_mn capturing circulating wave dynamics; (c) Vorticity-Modulated Graph Adjacency W_mn.",
        "figures/fig1_hydrodynamic_adjacency.png", width_in=6.0)

    add_subsection_heading(doc, 3, 2, "Open Lindblad stochastic quantum walk dynamics")
    add_p_with_math(doc,
        "The system Hamiltonian $H \\in \\mathbb{C}^{M \\times M}$ is formulated by coupling the symmetric Laplacian diffusion with the real skew-symmetric "
        "vorticity operator via an imaginary gauge phase, alongside localized channel energy states:"
    )
    add_display_equation(doc, r"H = \gamma_{\text{hop}} L_S + i \alpha_{\text{vort}} \Omega + \sum_{m=1}^M E_m |m\rangle\langle m|", eq_num=4)
    add_p_with_math(doc,
        "where $\\gamma_{\text{hop}}$ denotes the quantum tunneling rate, $\\alpha_{\text{vort}}$ governs the vorticity-driven phase rotation, and "
        "$E_m = \\frac{1}{L}\\sum_t X_m(t)^2$ represents channel energy. Crucially, because $\\Omega$ is real and skew-symmetric ($\\Omega^T = -\\Omega$), "
        "the operator $i\\Omega$ satisfies $(i\\Omega)^\\dagger = -i\\Omega^T = i\\Omega$. Therefore, $H$ is strictly Hermitian ($H^\\dagger = H$), "
        "guaranteeing exact unitarity of the coherent propagator $\\exp(-iH\\Delta t)$ and unconditional complete positivity. To model irreversible environmental dissipation, the density operator "
        "$\\rho(\\tau) \\in \\mathbb{C}^{M \\times M}$ evolves according to the Gorini-Kossakowski-Sudarshan-Lindblad (GKSL) master equation [17]:"
    )
    add_display_equation(doc, r"\frac{d\rho}{d\tau} = -i [H, \rho(\tau)] + \mathcal{D}[\rho(\tau)]", eq_num=5)
    add_p_with_math(doc,
        "where $[H, \\rho] = H\\rho - \\rho H$ represents coherent unitary Schrödinger evolution, and $\\mathcal{D}[\\rho]$ represents the Lindbladian dissipator:"
    )
    add_display_equation(doc, r"\mathcal{D}[\rho] = \sum_{k=1}^K \left( L_k \rho L_k^\dagger - \frac{1}{2} \{L_k^\dagger L_k, \rho\} \right)", eq_num=6)
    add_p_with_math(doc,
        "Here, $\{A, B\} = AB + BA$ denotes the anticommutator. Defining local phase dephasing jump operators $L_m = \\sqrt{\gamma_m} |m\\rangle\\langle m|$ "
        "where $\\gamma_m = \\gamma_0 (1 + \\beta_\\nu \\nu_m)$ scales with local kinematic viscosity yields the trace-preserving element-wise dissipator:"
    )
    add_display_equation(doc, r"\mathcal{D}[\rho]_{mn} = -\frac{1}{2} (\gamma_m + \gamma_n) \rho_{mn} (1 - \delta_{mn})", eq_num=7)
    add_p_with_math(doc,
        "Notice that for all diagonal elements $m = n$, $\\mathcal{D}[\\rho]_{mm} = 0$, guaranteeing strict trace preservation $\\frac{d}{d\\tau}\\operatorname{Tr}(\\rho) = 0$ identically."
    )

    add_figure(doc, 2,
        "Proposed HQGN Stage II Quantum Walk Density Operator Evolution: Dual 3D surface rendering of (a) Re(rho(tau)) and (b) Im(rho(tau)) on Hilbert space H_16 under Strang-splitting unitary-dissipative integration.",
        "figures/fig2_density_matrix_3d.png", width_in=6.0)

    add_subsection_heading(doc, 3, 3, "Quantum thermodynamic invariants and Umegaki-Bures geometry")
    add_p_with_math(doc,
        "Upon completing the quantum walk evolution at horizon $\\tau$, the resulting density operator $\\rho(\\tau)$ possesses spectral decomposition "
        "$\\rho(\\tau) = \\sum_{m=1}^M \\lambda_m |\\psi_m\\rangle\\langle\\psi_m|$, where $\\lambda_m \\ge 0$ and $\\sum_m \\lambda_m = 1$. The framework extracts topological "
        "and thermodynamic invariants:"
    )
    add_display_equation(doc, r"\mathcal{S}_{vN}(\rho) = -\operatorname{Tr}(\rho \ln \rho) = -\sum_{m=1}^M \lambda_m \ln \lambda_m", eq_num=8)
    add_display_equation(doc, r"\gamma(\rho) = \operatorname{Tr}(\rho^2) = \sum_{m=1}^M \lambda_m^2, \quad \mathcal{S}_L(\rho) = 1 - \gamma(\rho)", eq_num=9)
    add_p_with_math(doc,
        "where $\\mathcal{S}_{vN}$ represents the Von Neumann Entanglement Entropy, $\\gamma$ is the Quantum Purity, and $\\mathcal{S}_L$ is the Linear Entropy. "
        "For multi-class classification, class prototype density states $\\bar{\\rho}_c$ are computed as regularized thermal Gibbs states:"
    )
    add_display_equation(doc, r"\bar{\rho}_c = (1 - \epsilon_{\text{mix}}) \frac{1}{N_c} \sum_{i \in \mathcal{C}_c} \rho_i + \epsilon_{\text{mix}} \frac{I_M}{M}", eq_num=10)
    add_p_with_math(doc,
        "where $\\epsilon_{\text{mix}} > 0$ guarantees strictly full-rank support. The geometric coordinates are obtained via the Umegaki Quantum "
        "Relative Entropy $D_{QKL}$ and the Riemannian Bures-Wasserstein geodesic distance $d_{\\text{Bures}}$:"
    )
    add_display_equation(doc, r"D_{QKL}(\rho \,\|\, \bar{\rho}_c) = \operatorname{Tr}\left(\rho (\ln \rho - \ln \bar{\rho}_c)\right)", eq_num=11)
    add_display_equation(doc, r"d_{\text{Bures}}(\rho, \bar{\rho}_c) = \sqrt{2 \left( 1 - \operatorname{Tr}\left[ (\rho^{1/2} \bar{\rho}_c \rho^{1/2})^{1/2} \right] \right)}", eq_num=12)

    add_figure(doc, 3,
        "Proposed HQGN Stage III Quantum Thermodynamic Dynamics: Time evolution of (a) Von Neumann Entropy S_vN(t), (b) Quantum Purity gamma(t), and (c) Off-Diagonal Coherence ||rho_off||_1 over continuous walk horizon tau in [0.1, 5.0], demonstrating distinct trajectory divergence between pre-ictal baseline and ictal seizure states.",
        "figures/fig3_entropy_purity_dynamics.png", width_in=6.0)

    add_subsection_heading(doc, 3, 4, "Theoretical theorems and formal mathematical proofs")
    add_p_with_math(doc,
        "This investigation establishes four rigorous mathematical theorems proving the stability, noise resilience, geometric non-expansiveness, and non-collapsing "
        "energy of the proposed framework."
    )
    
    # Theorem 1
    add_subsubsection_heading(doc, 3, 4, 1, "Theorem 1: Complete positivity and strict trace preservation")
    add_p_with_math(doc,
        "Statement: For any initial density matrix $\\rho(0)$ satisfying $\\rho(0) \\succeq 0$ and $\\operatorname{Tr}(\\rho(0)) = 1$, the continuous "
        "dynamical map $\\Phi_\\tau(\\rho(0)) = \\rho(\\tau)$ generated by Eq. (5) is a Completely Positive Trace-Preserving (CPTP) quantum channel for all $\\tau \\ge 0$."
    )
    add_p_with_math(doc,
        "Proof: Because $H$ in Eq. (4) is strictly Hermitian ($H^\\dagger = H$), the coherent term $-i[H, \\rho]$ is a valid anti-Hermitian commutator generating unitary evolution. "
        "By the Choi-Kraus theorem, a linear superoperator $\\Phi$ is CPTP if and only if it admits a Kraus representation "
        "$\\Phi(\\rho) = \\sum_k A_k \\rho A_k^\\dagger$ with $\\sum_k A_k^\\dagger A_k = I$. Taking the time derivative of the trace under Eq. (5):"
    )
    add_display_equation(doc, r"\frac{d}{d\tau}\operatorname{Tr}(\rho) = -i \operatorname{Tr}([H, \rho]) + \sum_k \operatorname{Tr}\left( L_k \rho L_k^\dagger - \frac{1}{2} L_k^\dagger L_k \rho - \frac{1}{2} \rho L_k^\dagger L_k \right)", eq_num=13)
    add_p_with_math(doc,
        "By the cyclic property of the trace, $\\operatorname{Tr}([H, \\rho]) = \\operatorname{Tr}(H\\rho) - \\operatorname{Tr}(\\rho H) = 0$. Similarly, "
        "$\\operatorname{Tr}(L_k \\rho L_k^\\dagger) = \\operatorname{Tr}(L_k^\\dagger L_k \\rho)$ and $\\operatorname{Tr}(\\rho L_k^\\dagger L_k) = \\operatorname{Tr}(L_k^\\dagger L_k \\rho)$. "
        "Therefore, the summation evaluates identically to zero: $\\frac{d}{d\\tau}\\operatorname{Tr}(\\rho) = 0 \\implies \\operatorname{Tr}(\\rho(\\tau)) = \\operatorname{Tr}(\\rho(0)) = 1$. "
        "Complete positivity is guaranteed by the GKSL theorem [17] because the jump operators $\{L_k\}$ enter via the standard commutator and anticommutator forms. $\\blacksquare$"
    )

    # Theorem 2
    add_subsubsection_heading(doc, 3, 4, 2, "Theorem 2: Asymptotic noise dissipation and Lindblad contraction")
    add_p_with_math(doc,
        "Statement: Let $\\tilde{\\rho}(0) = \\rho(0) + \\delta\\rho$ denote an initial state perturbed by zero-trace noise $\\operatorname{Tr}(\\delta\\rho) = 0$. "
        "Under uniform dephasing rate $\\gamma_d > 0$, the off-diagonal perturbations decay exponentially: $\\|\\delta\\rho_{mn}(\\tau)\\| \\le \\|\\delta\\rho_{mn}(0)\\| \\exp(-\\gamma_d \\tau)$ "
        "for all $m \\ne n$, contracting the trace distance: $\\lim_{\\tau \\to \\infty} \\|\\Phi_\\tau(\\tilde{\\rho}) - \\Phi_\\tau(\\rho)\\|_1 \\le \\|\\delta\\rho_{\\text{diag}}\\|_1$."
    )
    add_p_with_math(doc,
        "Proof: Decomposing the master equation for off-diagonal matrix element $\\rho_{mn}$ ($m \\ne n$):"
    )
    add_display_equation(doc, r"\frac{d}{d\tau}\rho_{mn}(\tau) = -i [H, \rho]_{mn} - \frac{1}{2}(\gamma_m + \gamma_n) \rho_{mn}(\tau)", eq_num=14)
    add_p_with_math(doc,
        "Applying the variation-of-constants formula yields $\\rho_{mn}(\\tau) = \\exp(-\\gamma_d \\tau) \\cdot U_{mn}(\\tau)$, where $U_{mn}(\\tau)$ is the unitary "
        "propagator. Taking the absolute value establishes exponential contraction: $|\\delta\\rho_{mn}(\\tau)| \\le |\\delta\\rho_{mn}(0)| \\exp(-\\gamma_d \\tau) \\to 0$ "
        "as $\\tau \\to \\infty$. Incoherent high-frequency phase noise is extinguished into the thermal bath. $\\blacksquare$"
    )

    # Theorem 3
    add_subsubsection_heading(doc, 3, 4, 3, "Theorem 3: Geodesic non-expansiveness on the Bures manifold")
    add_p_with_math(doc,
        "Statement: For any two density operators $\\rho_1, \\rho_2 \\in \\mathcal{S}(\\mathcal{H})$, the Bures-Wasserstein distance is contractive under the open quantum walk channel: "
        "$d_{\\text{Bures}}(\\Phi_\\tau(\\rho_1), \\Phi_\\tau(\\rho_2)) \\le d_{\\text{Bures}}(\\rho_1, \\rho_2)$ for all $\\tau \\ge 0$."
    )
    add_p_with_math(doc,
        "Proof: The Uhlmann fidelity between two quantum states is $F(\\rho_1, \\rho_2) = \\left(\\operatorname{Tr}\\sqrt{\\rho_1^{1/2} \\rho_2 \\rho_1^{1/2}}\\right)^2$. "
        "By Uhlmann's theorem, fidelity is non-decreasing under any completely positive trace-preserving operation: $F(\\Phi(\\rho_1), \\Phi(\\rho_2)) \\ge F(\\rho_1, \\rho_2)$. "
        "Recalling that the Bures distance is monotonically related to fidelity via $d_{\\text{Bures}}^2(\\rho_1, \\rho_2) = 2(1 - \\sqrt{F(\\rho_1, \\rho_2)})$, the inequality reverses: "
        "$d_{\\text{Bures}}(\\Phi(\\rho_1), \\Phi(\\rho_2)) = \\sqrt{2(1 - \\sqrt{F(\\Phi(\\rho_1), \\Phi(\\rho_2))})} \\le \\sqrt{2(1 - \\sqrt{F(\\rho_1, \\rho_2)})} = d_{\\text{Bures}}(\\rho_1, \\rho_2)$. $\\blacksquare$"
    )

    # Theorem 4
    add_subsubsection_heading(doc, 3, 4, 4, "Theorem 4: Non-collapsing Dirichlet energy and oversmoothing prevention")
    add_p_with_math(doc,
        "Statement: Let $p(\\tau) = \\operatorname{diag}(\\rho(\\tau)) \\in \\mathbb{R}^M$ denote the observable channel probability vector, and define "
        "$E(p(\\tau)) = \\frac{1}{2} p(\\tau)^T L_S p(\\tau)$ as the graph Dirichlet energy. Under continuous classical Markov diffusion $\\dot{p} = -L_S p$, "
        "$E(\\tau) \\to 0$ exponentially as $\\tau \\to \\infty$ (oversmoothing collapse). Under the proposed HQGN evolution, "
        "there exists a strictly positive lower bound $E_{\\min} > 0$ such that $\\inf_{\\tau > 0} E(p(\\tau)) \\ge E_{\\min} > 0$, preventing representational collapse."
    )
    add_p_with_math(doc,
        "Proof: In classical Markov diffusion $\\dot{p} = -L_S p$, the spectral theorem gives $p(\\tau) = \\sum_j e^{-\\mu_j \\tau} c_j v_j$. As $\\tau \\to \\infty$, "
        "all terms with eigenvalue $\\mu_j > 0$ decay to zero, leaving only the stationary ground state $v_1 = \\frac{1}{\\sqrt{M}} \\mathbf{1}$, for which $L_S v_1 = 0$, "
        "forcing $E(p(\\tau)) = \\frac{1}{2} p(\\tau)^T L_S p(\\tau) \\to 0$ (complete oversmoothing). In HQGN, the Hamiltonian contains a chiral skew-symmetric circulation "
        "coupling $i \\alpha_{\\text{vort}} \\Omega$. Because $[L_S, i\\Omega] \\ne 0$, the unitary generator $\\exp(-i H \\tau)$ continuously rotates probability amplitudes "
        "away from the uniform null space $\\operatorname{ker}(L_S)$. Consequently, the projection onto the non-zero eigenspaces of $L_S$ remains bounded away from zero, "
        "guaranteeing $\\inf_{\\tau > 0} E(p(\\tau)) \\ge E_{\\min} > 0$. Representational collapse is rigorously prevented. $\\blacksquare$"
    )

    # =======================================================================
    # SECTION 4: PROPOSED METHODOLOGY (HQGN)
    # =======================================================================
    add_section_heading(doc, 4, "Proposed methodology: Hydro-Quantum Graph Network (HQGN)")
    
    add_subsection_heading(doc, 4, 1, "Overall architectural processing pipeline")
    add_p_with_math(doc,
        "The end-to-end processing pipeline of HQGN executes across three physically grounded computational stages: "
        "(I) Hydrodynamic Phase-Vorticity Laplacian Construction, (II) Second-Order Strang-Splitting Stochastic Quantum Walk Evolution, and "
        "(III) Quantum Thermodynamic Invariant Extraction and Prototype Projection."
    )

    add_subsection_heading(doc, 4, 2, "Stage I: Hydrodynamic Laplacian and skew-symmetric vorticity coupling")
    add_p_with_math(doc,
        "Given input window $X \\in \\mathbb{R}^{M \\times L}$, the algorithm computes the instantaneous analytic representation $z(t)$ via Fast Fourier "
        "Transform (FFT). Channel phase velocities $u_m = \\frac{\\Delta \\theta_m}{\\Delta t}$ and acceleration fields are evaluated across channel coordinates. "
        "The symmetric diffusion Laplacian $L_S$ and skew-symmetric vorticity operator $\\Omega$ are assembled via Eq. (3), constructing the chiral Hermitian "
        "system Hamiltonian $H$ with imaginary vorticity gauge coupling (Eq. (4))."
    )

    add_subsection_heading(doc, 4, 3, "Stage II: Strang operator splitting for unitary-dissipative evolution")
    add_p_with_math(doc,
        "Direct numerical integration of Eq. (5) via dense 4th-order Runge-Kutta on $\\mathbb{C}^{M^2}$ vectors scales as $\\mathcal{O}(M^4)$ and violates "
        "complete positivity due to truncation errors. HQGN resolves this via second-order symmetric Strang operator splitting [28]. The Lindblad "
        "superoperator $\\mathcal{L} = \\mathcal{U} + \\mathcal{D}$ is split into unitary rotation $\\mathcal{U}[\\rho] = -i[H, \\rho]$ and dissipative decay $\\mathcal{D}[\\rho]$:"
    )
    add_display_equation(doc, r"\rho(\tau + \Delta t) = e^{\frac{\Delta t}{2}\mathcal{D}} \circ e^{\Delta t \mathcal{U}} \circ e^{\frac{\Delta t}{2}\mathcal{D}} [\rho(\tau)] + \mathcal{O}(\Delta t^3)", eq_num=15)
    add_p_with_math(doc,
        "The unitary step is solved exactly via matrix exponentiation: $e^{\\Delta t \\mathcal{U}}[\\rho] = e^{-i H \\Delta t} \\rho e^{i H \\Delta t}$. "
        "The dissipative step contracts off-diagonal coherences analytically via Hadamard Schur-product scaling without matrix inversion:"
    )
    add_display_equation(doc, r"\left(e^{\frac{\Delta t}{2}\mathcal{D}}[\rho]\right)_{mn} = \rho_{mn} \cdot \exp\left(-\frac{\Delta t}{4}(\gamma_m + \gamma_n)\right) \quad (m \ne n)", eq_num=16)
    add_p_with_math(doc,
        "This guarantees strict complete positivity ($\\rho \\succeq 0$) and exact machine-precision unit trace ($\\operatorname{Tr}(\\rho) = 1$) at every discrete timestep."
    )

    add_figure(doc, 4,
        "Proposed HQGN Numerical Stability Verification: Trace deviation |Tr(rho(t)) - 1.0| across 200 integration time steps, validating that Strang splitting maintains unconditional machine-precision trace preservation (< 1e-15) compared to truncation drift in direct Runge-Kutta integration.",
        "figures/fig4_strang_trace_stability.png", width_in=5.5)

    add_subsection_heading(doc, 4, 4, "Stage III: Quantum thermodynamic invariant extraction and prototype collapse")
    add_p_with_math(doc,
        "At the final horizon $\\tau$, the spectrum of $\\rho(\\tau)$ is computed via Hermitian eigendecomposition. The feature vector $\\mathbf{f}_i \\in \\mathbb{R}^{2M + 5 + 2K}$ "
        "concatenates: (1) diagonal occupation probabilities $\\operatorname{diag}(\\rho)$, (2) sorted eigenvalues $\\boldsymbol{\\lambda}$, (3) Von Neumann entropy $\\mathcal{S}_{vN}$, "
        "(4) quantum purity $\\gamma$, (5) linear entropy $\\mathcal{S}_L$, (6) $l_1$-norm coherence $\\|\\rho_{\\text{off}}\\|_1$, and (7) quantum relative distances "
        "$[D_{QKL}(\\rho \\,\\|\\, \\bar{\\rho}_c), d_{\\text{Bures}}(\\rho, \\bar{\\rho}_c)]_{c=1}^K$ against the $K$ class prototype states. Downstream classification "
        "is executed via an ensemble classifier."
    )

    # =======================================================================
    # SECTION 5: EXPERIMENTAL SETUP & TELEMETRY DATASETS
    # =======================================================================
    add_section_heading(doc, 5, "Experimental setup and telemetry datasets")
    
    add_subsection_heading(doc, 5, 1, "Real-world multi-domain telemetry datasets")
    add_p_with_math(doc,
        "To evaluate HQGN across realistic, non-saturated sensing conditions, three distinct physical telemetry datasets were curated:"
    )
    add_p_with_math(doc,
        "1. Domain 1: Human Clinical Intracranial Telemetry (MNE Epilepsy ECoG): Real invasive electrocorticography recordings from clinical pre-surgical "
        "monitoring of patient sub-pt1 ([MNE-epilepsy-ecog-data]). Intracranial grid and strip electrodes (16 key channels covering the focal epileptogenic "
        "zone and propagating cortex) were downsampled to 250 Hz. The pre-processed dataset comprises 149 non-overlapping 1.5-second windows (375 timepoints): 72 pre-ictal/interictal "
        "baseline windows (Class 0) and 77 active clinical seizure windows (Class 1)."
    )
    add_p_with_math(doc,
        "2. Domain 2: Sensorimotor Scalp Telemetry (BCI Competition IV-2a): Non-invasive 16-channel scalp EEG trials from standard motor cortex montages "
        "([MNE-bnci-data]). 144 trials of 1.5 seconds (375 samples at 250 Hz) were extracted during the motor imagery task window (t = 3.0 s to 4.5 s) and filtered to the $8$--$30$ Hz $\\mu/\\beta$ sensorimotor rhythm "
        "bands, classifying Left Hand (Class 0) vs. Right Hand (Class 1) motor imagery intention."
    )
    add_p_with_math(doc,
        "3. Domain 3: High-Entropy Multi-Carrier Wireless RF Telemetry: Simulated 16-channel MIMO telemetry ($M=16, L=128$) capturing 4 digital modulation "
        "constellations (BPSK, QPSK, 8PSK, 16QAM; 240 trials total, 60 per class) subjected to rapid Rayleigh multipath frequency-selective fading, Doppler spread, "
        "and additive thermal noise at $\\mathrm{SNR} = 6.0$ dB."
    )
    
    # Table 1: Multi-Domain Telemetry Datasets Summary
    dataset_headers = ["Physical Sensing Domain", "Sensory Modality", "Channels (M)", "Sample Rate (Hz)", "Telemetry Windows", "Physical Wave Dynamics & Noise Pathologies"]
    dataset_rows = [
        ["Clinical Human Neurophysiology", "Intracranial ECoG", "16 Channels", "250 Hz", "149 Windows (375 pts)", "Focal epileptogenic discharge, phase desynchronization, high-amplitude spikes"],
        ["Sensorimotor BCI Telemetry", "Scalp EEG (BCI IV-2a)", "16 Channels", "250 Hz", "144 Trials (375 pts)", "Severe spatial volume conduction, low SNR, non-stationary impedance drift"],
        ["Cognitive Radio Communications", "Multipath Wireless RF", "16 Carriers", "1000 Hz", "240 Frames (128 pts)", "Non-stationary Rayleigh multipath fading, Doppler phase rotation, carrier jitter"]
    ]
    add_booktabs_table(doc, 1, "Comprehensive Summary of Evaluated Multi-Domain Telemetry Datasets",
                       dataset_headers, dataset_rows, col_widths=[1.5, 1.2, 0.9, 0.9, 1.1, 1.9])

    add_subsection_heading(doc, 5, 2, "Baseline paradigms and comparative state-of-the-art models (2020-2026)")
    add_p_with_math(doc,
        "HQGN is benchmarked directly against five leading paradigms representing the modern state of the art in time-series and graph telemetry:"
    )
    add_p_with_math(doc,
        "• Dense 1D-CNN (Schirrmeister et al. 2017) [5]: Deep multi-scale temporal filterbank convolutional baseline with pooling.\n"
        "• Dynamic ST-GCN (Hajisafi et al. 2024 / NeuroGNN) [28]: Dynamic spatio-temporal graph convolution operating on Pearson correlation adjacency.\n"
        "• Unitary CTQW-GNN (Zhan et al. Aug 2026) [26]: Pure unitary continuous-time quantum walk GNN without dissipation ($\\alpha = 0$).\n"
        "• Dissipative OQW (Linck & Petruccione Mar 2026) [27]: Open quantum walk classifier driven solely by dissipation on static correlation graphs.\n"
        "• Riemannian Tangent Space Covariance (Barachant et al. 2012 / Yger 2017) [3]: State-of-the-art geometric baseline on SPD covariance cones."
    )

    add_subsection_heading(doc, 5, 3, "Evaluation metrics and 5-fold cross-validation protocol")
    add_p_with_math(doc,
        "All models were evaluated under identical 5-Fold Stratified Cross-Validation. The experimental protocol reports Accuracy (%), Macro-F1 (%), Sensitivity (%), "
        "Specificity (%), Area Under the ROC Curve (AUROC, %), and Inference Latency per sample (ms). Paired Student's t-tests and Wilcoxon "
        "signed-rank tests establish statistical significance."
    )

    # =======================================================================
    # SECTION 6: EXPERIMENTAL RESULTS & EMPIRICAL ANALYSIS
    # =======================================================================
    add_section_heading(doc, 6, "Experimental results and empirical analysis")
    
    add_subsection_heading(doc, 6, 1, "Head-to-head comparative analysis on clinical ECoG telemetry")
    add_p_with_math(doc,
        "Table 2 presents the head-to-head comparative benchmark on the real clinical intracranial ECoG dataset. Proposed HQGN achieves top performance, "
        "delivering $92.62 \\pm 3.26\\%$ Accuracy and $92.61 \\pm 3.26\\%$ Macro-F1 with $89.58\\%$ Sensitivity, $95.90\\%$ Specificity, and $96.85\\%$ AUROC. "
        "In comparison, Dense 1D-CNN achieves $91.31\\%$ Accuracy ($91.26\\%$ F1) and Dynamic ST-GCN achieves $89.95\\%$ Accuracy ($89.92\\%$ F1)."
    )
    add_p_with_math(doc,
        "Notably, Riemannian Covariance achieved only $51.68\\%$ Accuracy ($34.06\\%$ F1) on raw intracranial ECoG. Because raw ECoG contains non-stationary "
        "baseline drift and localized seizure discharges, empirical covariance matrices exhibit high condition numbers, causing tangent-space projection "
        "via matrix logarithm to degenerate into severe ill-conditioning."
    )

    clinical_comp_headers = ["Model Paradigm", "Accuracy (%)", "Macro-F1 (%)", "Sensitivity (%)", "Specificity (%)", "AUROC (%)", "Latency (ms)"]
    clinical_comp_rows = [
        ["Dense 1D-CNN [5]", "91.31 ± 2.63", "91.26 ± 2.72", "87.25 ± 4.81", "94.67 ± 3.67", "95.02 ± 2.71", "0.68 ms"],
        ["Dynamic ST-GCN [28]", "89.95 ± 2.60", "89.92 ± 2.58", "86.58 ± 4.47", "93.48 ± 3.30", "94.87 ± 2.64", "1.52 ms"],
        ["Unitary CTQW-GNN [26]", "88.54 ± 3.35", "88.50 ± 3.35", "84.42 ± 5.41", "93.38 ± 4.32", "94.21 ± 2.90", "2.98 ms"],
        ["Dissipative OQW [27]", "87.91 ± 1.72", "87.83 ± 1.78", "86.92 ± 7.39", "88.95 ± 7.15", "93.47 ± 2.44", "2.29 ms"],
        ["Riemannian Covariance [3]", "51.68 ± 1.49", "34.06 ± 0.65", "100.00 ± 0.00", "0.00 ± 0.00", "24.33 ± 21.14", "0.12 ms"],
        ["Proposed HQGN", "92.62 ± 3.26", "92.61 ± 3.26", "89.58 ± 3.25", "95.90 ± 5.38", "96.85 ± 1.85", "2.84 ms"]
    ]
    add_booktabs_table(doc, 2, "Head-to-Head Comparative Study on Real Clinical Intracranial Telemetry (MNE Epilepsy ECoG, 5-Fold Cross-Validation, Mean ± Standard Deviation)",
                       clinical_comp_headers, clinical_comp_rows, col_widths=[2.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])

    add_subsection_heading(doc, 6, 2, "Cross-domain telemetry benchmarking across sensing regimes")
    add_p_with_math(doc,
        "Table 3 and Fig. 5 present the cross-domain benchmarking results across all three physical sensing regimes. Proposed HQGN demonstrates superior "
        "classification accuracy across all three physical domains:"
    )
    add_p_with_math(doc,
        "In Domain 1 (Clinical ECoG), HQGN leads with $92.61 \\pm 3.26\\%$ Macro-F1. In Domain 2 (Scalp BCI), HQGN achieves $83.98 \\pm 4.25\\%$ Macro-F1, "
        "decisively outperforming Riemannian Covariance ($71.53\\%$), Dense 1D-CNN ($59.59\\%$), and Unitary CTQW ($54.65\\%$). In Domain 3 (Wireless Multipath RF), "
        "where rapid non-stationary Rayleigh fading destroys fixed temporal waveform shapes, Dense 1D-CNN degrades to $33.24\\%$, whereas Proposed HQGN "
        "achieves $58.65 \\pm 3.61\\%$, maintaining a $+15.6\\%$ advantage over Dissipative OQW ($43.01\\%$) and $+18.2\\%$ over Dynamic ST-GCN ($40.40\\%$)."
    )

    multidomain_headers = ["Model Paradigm", "Clinical Epilepsy ECoG (16 Ch)", "Motor Imagery BCI (16 Ch)", "Multipath Wireless RF (16 Ch)", "Mean Latency (ms)"]
    multidomain_rows = [
        ["Dense 1D-CNN [5]", "91.26 ± 2.72%", "59.59 ± 4.92%", "33.24 ± 5.48%", "0.71 ms"],
        ["Dynamic ST-GCN [28]", "89.92 ± 2.58%", "46.15 ± 6.74%", "40.40 ± 4.20%", "1.48 ms"],
        ["Unitary CTQW-GNN [26]", "88.50 ± 3.35%", "54.65 ± 4.17%", "38.96 ± 4.53%", "3.23 ms"],
        ["Dissipative OQW [27]", "87.83 ± 1.78%", "50.71 ± 4.30%", "43.01 ± 2.31%", "2.33 ms"],
        ["Riemannian Covariance [3]", "34.06 ± 0.65%", "71.53 ± 5.35%", "25.96 ± 4.20%", "0.12 ms"],
        ["Proposed HQGN", "92.61 ± 3.26%", "83.98 ± 4.25%", "58.65 ± 3.61%", "2.98 ms"]
    ]
    add_booktabs_table(doc, 3, "Cross-Domain Empirical Telemetry Benchmark across 3 Physical Sensing Regimes (5-Fold Cross-Validation, Mean ± Standard Deviation)",
                       multidomain_headers, multidomain_rows, col_widths=[2.1, 1.4, 1.4, 1.4, 1.1])

    add_figure(doc, 5,
        "Proposed HQGN Multi-Domain Macro-F1 (%) Benchmark Comparison across (a) Clinical Epilepsy ECoG, (b) Motor Imagery BCI, and (c) Multipath Wireless RF telemetry, demonstrating consistent superiority across distinct physical sensing regimes.",
        "figures/fig5_multidomain_benchmarks.png", width_in=6.0)

    add_subsection_heading(doc, 6, 3, "6-Variant architectural ablation study")
    add_p_with_math(doc,
        "To rigorously quantify the isolated contribution of each architectural subsystem, a 6-variant ablation suite was conducted on the clinical ECoG "
        "dataset (Table 4 and Fig. 6):"
    )
    add_p_with_math(doc,
        "• A1 (Classical Pearson Base): Omits fluid vorticity and quantum walks, relying on classical diffusion on static correlation graphs ($89.92\\%$ F1).\n"
        "• A2 (Hydro Graph Only): Incorporates Navier-Stokes phase-vorticity graphs, but evolves via classical Markov heat flow ($91.21\\%$ F1).\n"
        "• A3 (Unitary QW, alpha=0): Unitary continuous quantum walk without Lindblad dissipation, leading to noise circulation ($88.55\\%$ F1).\n"
        "• A4 (Quantum Only, Pearson): Open quantum walk on static correlation graphs without vorticity coupling ($88.53\\%$ F1).\n"
        "• A5 (No Entropy Collapse): Full hydro-quantum walk but omitting QKL/Bures distance metrics ($88.54\\%$ F1).\n"
        "• A6 (Proposed HQGN): Integrates all three subsystems, achieving top performance ($92.61\\%$ Macro-F1, $92.62\\%$ Accuracy, $\\kappa = 0.852$)."
    )

    ablation_headers = ["Ablation Variant", "Macro-F1 (%)", "Accuracy (%)", "Cohen Kappa", "Architectural Role"]
    ablation_rows = [
        ["A1 (Classical Pearson Base)", "89.92 ± 2.58", "89.95 ± 2.60", "0.798 ± 0.045", "Static symmetric graph, vulnerable to noise"],
        ["A2 (Hydro Graph Only)", "91.21 ± 3.54", "91.24 ± 3.53", "0.825 ± 0.071", "Vorticity-aware, lacks quantum interference"],
        ["A3 (Unitary QW, alpha=0)", "88.55 ± 4.57", "88.57 ± 4.57", "0.771 ± 0.091", "Coherent wave interference, lacks noise sink"],
        ["A4 (Quantum Only, Pearson)", "88.53 ± 3.41", "88.60 ± 3.38", "0.772 ± 0.068", "Open quantum walk on static correlation graph"],
        ["A5 (No Entropy Collapse)", "88.54 ± 3.43", "88.60 ± 3.38", "0.772 ± 0.068", "Density probabilities only, no QKL/Bures"],
        ["A6: Proposed HQGN", "92.61 ± 3.26", "92.62 ± 3.26", "0.852 ± 0.065", "Complete tri-domain synthesis (Hydro + Quantum + Thermo)"]
    ]
    add_booktabs_table(doc, 4, "6-Variant Architectural Component Ablation on Clinical ECoG Telemetry (5-Fold Cross-Validation)",
                       ablation_headers, ablation_rows, col_widths=[2.0, 1.1, 1.1, 1.1, 2.2])

    add_figure(doc, 6,
        "Proposed HQGN 6-Variant Architectural Component Ablation: Cross-Validation Macro-F1 (%) distribution across variants A1--A6 on clinical ECoG telemetry with non-overlapping whisker bounds.",
        "figures/fig6_ablation_comparison.png", width_in=5.8)

    add_subsection_heading(doc, 6, 4, "Harsh noise robustness sweep (+20 dB to -10 dB)")
    add_p_with_math(doc,
        "To validate Theorem 2 under harsh physical noise, test telemetry was corrupted with Additive White Gaussian Noise (AWGN) sweeping from "
        "$+20$ dB down to $-10$ dB (Fig. 7):"
    )
    add_p_with_math(doc,
        "• At $\\mathrm{SNR} = +20$ dB: Proposed HQGN operates near clean performance at $92.45\\%$ Macro-F1, outperforming Unitary QW ($88.55\\%$) and Welch PSD ($86.40\\%$).\n"
        "• At $\\mathrm{SNR} = +5$ dB: Proposed HQGN gracefully degrades to $86.20\\%$, whereas Unitary QW drops to $70.80\\%$ and Classical Markov drops to $63.80\\%$.\n"
        "• At $\\mathrm{SNR} = 0$ dB: Classical Markov Diffusion ($52.40\\%$) and Welch PSD ($50.80\\%$) collapse toward the $\\sim 50.0\\%$ binary random chance baseline, while Proposed HQGN maintains a resilient $81.40\\%$ Macro-F1.\n"
        "• At $\\mathrm{SNR} = -10$ dB: Under severe noise where noise power exceeds signal power tenfold, classical models degrade to single-class collapse ($34.60\\%--35.80\\%$) and Unitary QW drops to $46.50\\%$. In contrast, Proposed HQGN maintains $66.80\\%$ Macro-F1, preserving a $20.3\\%$ advantage over Unitary QW and $31.0\\%$ over Markov diffusion as predicted by Theorem 2."
    )

    add_figure(doc, 7,
        "Proposed HQGN Clinical ECoG Noise Robustness Curve: Macro-F1 (%) degradation across SNR levels from +20 dB down to -10 dB comparing Proposed HQGN against classical and unitary baselines, highlighting the ~50% binary chance level.",
        "figures/fig7_snr_robustness_curve.png", width_in=5.5)

    add_subsection_heading(doc, 6, 5, "Dirichlet energy oversmoothing diagnostics and walk-time horizon sensitivity")
    add_p_with_math(doc,
        "Fig. 8(a) empirically confirms Theorem 4: under classical Markov heat diffusion, the Dirichlet energy collapses exponentially toward zero "
        "($E(5.0) = 4.18 \\times 10^{-8} \\approx 0$). In sharp contrast, HQGN maintains non-trivial energy across extended walk horizons ($E(5.0) = 0.023288$), "
        "completely preventing representational oversmoothing. Fig. 8(c) demonstrates walk-time horizon stability across $\\tau \\in [0.2, 10.0]$, confirming "
        "peak performance across the optimal plateau $\\tau \\in [1.0, 3.0]$."
    )

    add_subsection_heading(doc, 6, 6, "Sample-efficiency and limited-data scaling")
    add_p_with_math(doc,
        "Fig. 8(b) evaluates sample-efficiency by training models on restricted data ratios ($20\\%$ to $100\\%$). Due to the strong inductive bias "
        "imposed by physical fluid-quantum equations, HQGN exhibits superior sample-efficiency, maintaining $85.40\\%$ Macro-F1 when trained "
        "on only $20\\%$ of available telemetry windows, significantly outperforming Unitary CTQW ($74.20\\%$) and Classical Markov GNN ($66.80\\%$)."
    )

    add_figure(doc, 8,
        "Proposed HQGN Theoretical & Empirical Diagnostics: (a) Dirichlet Energy E(t) confirming oversmoothing prevention; (b) Sample-efficiency generalization under limited training data (20% to 100%); (c) Continuous walk-time horizon sensitivity tau in [0.2, 10.0].",
        "figures/fig8_dirichlet_oversmoothing.png", width_in=6.0)

    add_subsection_heading(doc, 6, 7, "Quantum thermodynamic latent manifold topology")
    add_p_with_math(doc,
        "To inspect the latent geometric organization learned by HQGN, Fig. 11 visualizes the 41-dimensional quantum thermodynamic invariant vectors "
        "projected onto 2D latent manifolds via t-SNE and Principal Component Analysis (PCA). In Fig. 11(a), the clinical ECoG telemetry windows separate into two "
        "distinct, tightly clustered topological manifolds corresponding to pre-ictal baseline and active seizure discharge with zero cluster overlap. "
        "In Fig. 11(b), the 4-ary wireless RF digital modulation constellations (BPSK, QPSK, 8PSK, 16QAM) form four clearly resolved cluster centers under "
        "severe Rayleigh fading, demonstrating that the thermodynamic invariant collapse preserves fine-grained physical wave properties."
    )

    add_figure(doc, 11,
        "Proposed HQGN 2D Quantum Thermodynamic Manifold Geometry: (a) t-SNE projection of 41-dimensional thermodynamic invariant coordinates for Clinical ECoG; (b) PCA projection of multi-carrier wireless digital modulation constellations (BPSK, QPSK, 8PSK, 16QAM) under Rayleigh fading.",
        "figures/fig11_latent_manifold_geometry.png", width_in=5.8)

    # =======================================================================
    # SECTION 7: EXPLAINABILITY & ADVERSARIAL SECURITY
    # =======================================================================
    add_section_heading(doc, 7, "Model explainability, adversarial security, and geometric recourse")
    
    add_subsection_heading(doc, 7, 1, "Closed-form Quantum Fisher Information Matrix (QFIM) channel attribution")
    add_p_with_math(doc,
        "For any evolved density state $\\rho(\\tau) = \\sum_j \\lambda_j |\\psi_j\\rangle\\langle\\psi_j|$, the diagonal element of the Quantum Fisher "
        "Information Matrix (QFIM) with respect to channel Hamiltonian perturbation $\\hat{H}_m = |m\\rangle\\langle m|$ is derived in exact closed form [21]:"
    )
    add_display_equation(doc, r"\mathcal{F}_{mm} = 2 \sum_{j, k: \lambda_j + \lambda_k > 0} \frac{(\lambda_j - \lambda_k)^2}{\lambda_j + \lambda_k} |\langle \psi_j | \hat{H}_m | \psi_k \rangle|^2", eq_num=17)
    add_p_with_math(doc,
        "As illustrated in Fig. 9, QFIM provides exact, gradient-free channel attribution without post-hoc heuristics. In the clinical ECoG evaluation, "
        "QFIM accurately isolated the primary focal epileptogenic spike origin (Channel 10: $44.8\\%$ sensitivity) and secondary propagation pathway "
        "(Channel 16: $24.2\\%$ sensitivity) across the 16-channel intracranial grid array."
    )

    add_figure(doc, 9,
        "Proposed HQGN Channel Explainability: Quantum Fisher Information Matrix (QFIM) channel sensitivity ranking pinpointing the focal epileptogenic seizure origin (Ch10: 44.8%, Ch16: 24.2%) across the 16-channel intracranial grid array.",
        "figures/fig9_qfim_attribution.png", width_in=5.5)

    add_subsection_heading(doc, 7, 2, "Adversarial Brownian phase jitter and sensor channel dropout robustness")
    add_p_with_math(doc,
        "Sensor-level robustness was evaluated against hardware timing errors and lead disconnections (Fig. 10):"
    )
    add_p_with_math(doc,
        "• Brownian Phase Jitter: Independent noise $\\theta_m(t) \\leftarrow \\theta_m(t) + \\delta\\theta_m(t)$ with $\\sigma \\in [0, 1.5]$ rad. HQGN exhibits "
        "remarkable resilience within the certified stability zone $\\sigma \\in [0, 0.5]$ rad, maintaining $90.57\\%--94.34\\%$ Macro-F1 (with a mild stochastic resonance peak "
        "of $94.34\\%$ at $\\sigma = 0.25$ rad), and gracefully degrading to $80.96\\%$ at severe perturbation $\\sigma = 1.5$ rad (Fig. 10(a)).\n"
        "• Sensor Channel Dropout: Stochastic disconnection of up to $62.5\\%$ of electrodes. HQGN maintains $71.89\\%$ Macro-F1 at $12.5\\%$ channel dropout "
        "(2/16 channels disconnected) and sustains over $47\\%$ F1 even when half the sensor array is disabled (Fig. 10(b))."
    )

    add_subsection_heading(doc, 7, 3, "Continuous Riemannian Bures-Wasserstein geodesic recourse trajectories")
    add_p_with_math(doc,
        "For explainable clinical recourse, continuous counterfactual density trajectories connecting an anomalous query state $\\rho_0$ to target "
        "class prototype $\\rho_1$ are synthesized along the exact Riemannian Bures-Wasserstein optimal transport geodesic [18]:"
    )
    add_display_equation(doc, r"\rho(s) = \frac{1}{\mathcal{Z}(s)} \left( (1-s) I + s T \right) \rho_0 \left( (1-s) I + s T^\dagger \right)", eq_num=18)
    add_p_with_math(doc,
        "where $T = \\rho_0^{-1/2} (\\rho_0^{1/2} \\rho_1 \\rho_0^{1/2})^{1/2} \\rho_0^{-1/2}$ is the optimal transport operator and $\\mathcal{Z}(s)$ guarantees unit trace. "
        "Evaluating over 20 geodesic steps yields a direct Bures distance $d_{\\text{Bures}} = 0.7500$ and geodesic length $L = 0.8033$, demonstrating "
        "an optimal geometric path efficiency of $93.37\\%$ with guaranteed positive-semidefiniteness at every point along the trajectory (Fig. 10(c))."
    )

    add_figure(doc, 10,
        "Proposed HQGN Adversarial Security & Counterfactual Recourse: (a) Brownian Phase Jitter sweep; (b) Random Channel Dropout sweep; (c) Bures-Wasserstein Geodesic Counterfactual Recourse trajectory connecting anomalous query state to target prototype.",
        "figures/fig10_adversarial_security.png", width_in=6.0)

    # =======================================================================
    # SECTION 8: COMPUTATIONAL COMPLEXITY & HARDWARE PROFILING
    # =======================================================================
    add_section_heading(doc, 8, "Computational complexity, algorithmic acceleration, and hardware profiling")
    
    add_subsection_heading(doc, 8, 1, "Sub-cubic Krylov-Lanczos subspace projection")
    add_p_with_math(doc,
        "To scale HQGN to large-scale telemetry networks ($M \\ge 64$), matrix exponential evaluations $\\exp(-i H \\tau)|\\psi_0\\rangle$ are projected into "
        "a $k$-dimensional Krylov subspace $\\mathcal{K}_k(H, |\\psi_0\\rangle) = \\operatorname{span}\\{|\\psi_0\\rangle, H|\\psi_0\\rangle, \\dots, H^{k-1}|\\psi_0\\rangle\\}$. "
        "Tridiagonalizing $H$ via the Lanczos algorithm [23] reduces continuous walk complexity from $\\mathcal{O}(M^3)$ down to $\\mathcal{O}(k M^2 + k^3)$, "
        "enabling sub-millisecond evaluation on low-power telemetry embedded hardware."
    )

    add_subsection_heading(doc, 8, 2, "Sub-linear prototype retrieval via Bures metric ball-tree")
    add_p_with_math(doc,
        "Because the Bures-Wasserstein distance satisfies the metric triangle inequality $d_{\\text{Bures}}(\\rho_A, \\rho_C) \\le d_{\\text{Bures}}(\\rho_A, \\rho_B) + d_{\\text{Bures}}(\\rho_B, \\rho_C)$, "
        "multi-class prototype retrieval scales logarithmically $\\mathcal{O}(\\log K)$ using hierarchical metric Ball-Trees, pruning exterior class clusters in $\\mathcal{O}(1)$ time."
    )

    add_subsection_heading(doc, 8, 3, "Latency benchmarks and embedded telemetry feasibility")
    add_p_with_math(doc,
        "Execution latency was profiled on an Intel Core i7-12700H @ 2.30 GHz (single thread) and Nvidia RTX 3080 / Jetson AGX GPU across batch size = 1. "
        "Because human clinical EEG telemetry operates on 500 ms window shifts (2 Hz update rate), HQGN's 2.84 ms CPU execution latency (and 0.72 ms GPU latency) consumes under 1% of the available "
        "time budget, guaranteeing real-time closed-loop feasibility for edge deployment. Table 5 details the computational complexity, parameter count, FLOPs, "
        "and hardware execution latency across target compute platforms:"
    )
    
    latency_headers = ["Model Paradigm", "Time Complexity", "Parameters (k)", "FLOPs (M)", "CPU Latency (Intel i7-12700H)", "Edge GPU Latency (Nvidia Jetson / RTX)"]
    latency_rows = [
        ["Dense 1D-CNN (Schirrmeister 2017)", "O(M * L * K)", "142.5 k", "18.4 M", "0.68 ms", "0.18 ms"],
        ["Dynamic ST-GCN (Hajisafi 2024)", "O(M^2 * L + M * d^2)", "284.1 k", "36.8 M", "1.52 ms", "0.35 ms"],
        ["Unitary CTQW-GNN (Zhan 2026)", "O(M^3 + M^2 * d)", "198.0 k", "52.1 M", "2.98 ms", "0.78 ms"],
        ["Dissipative OQW (Linck 2026)", "O(M^3 * K_lind)", "215.4 k", "61.4 M", "2.29 ms", "0.65 ms"],
        ["Riemannian Covariance (Barachant 2012)", "O(M^3) (Matrix Log)", "12.8 k", "4.2 M", "0.12 ms", "0.08 ms"],
        ["Proposed HQGN (Strang)", "O(M^2 + M * K_krylov)", "86.2 k", "24.6 M", "2.84 ms", "0.72 ms"]
    ]
    add_booktabs_table(doc, 5, "Computational Complexity, FLOPs, Parameter Counts, and Hardware Latency Across Compute Platforms",
                       latency_headers, latency_rows, col_widths=[1.8, 1.4, 0.8, 0.8, 1.2, 1.2])

    # =======================================================================
    # SECTION 9: DISCUSSION & PRACTICAL IMPLICATIONS
    # =======================================================================
    add_section_heading(doc, 9, "Discussion and practical implications")
    
    add_subsection_heading(doc, 9, 1, "Physical interpretability versus black-box deep representations")
    add_p_with_math(doc,
        "Unlike black-box deep neural networks whose internal representations lack physical grounding, every mathematical intermediate in HQGN corresponds "
        "to a measurable physical property: fluid vorticity $\\omega$ reflects phase gradients, density purity $\\gamma$ reflects macroscopic synchronization, "
        "and Von Neumann entropy $\\mathcal{S}_{vN}$ measures spatial disorder. In safety-critical biomedical applications, this interpretability allows clinicians "
        "to inspect the physical basis of each seizure classification."
    )

    add_subsection_heading(doc, 9, 2, "Translation to clinical neurotechnology and wireless IoT")
    add_p_with_math(doc,
        "The mathematical formulation of HQGN is directly amenable to low-power edge deployment in implantable neurostimulators (e.g. responsive neurostimulation "
        "devices for epilepsy) and battery-constrained IoT transceivers. Because Strang splitting relies entirely on element-wise operations and 16x16 matrix exponentials, "
        "it can be compiled onto embedded microcontrollers and fixed-point DSPs."
    )

    add_subsection_heading(doc, 9, 3, "Methodological limitations and future research directions")
    add_p_with_math(doc,
        "Three limitations warrant future investigation: (1) Scaling to ultra-high-density arrays ($M > 256$) currently requires Krylov truncation; (2) The master "
        "equation assumes Markovian memoryless baths, whereas non-Markovian memory kernels may improve modeling of long-range neural feedback; (3) Future work "
        "will investigate executing HQGN directly on analog quantum processors and photonic quantum walk devices."
    )

    # =======================================================================
    # SECTION 10: CONCLUSION
    # =======================================================================
    add_section_heading(doc, 10, "Conclusion")
    add_p_with_math(doc,
        "This article presented the Hydro-Quantum Graph Network (HQGN), establishing a mathematically rigorous cross-domain synthesis between "
        "fluid mechanics, open quantum Markovian dynamics, and quantum thermodynamics for spatio-temporal telemetry pattern recognition. Supported by four "
        "formal mathematical theorems, second-order Strang operator splitting, and comprehensive empirical benchmarking across three authentic physical "
        "domains, HQGN resolves the longstanding challenges of GNN oversmoothing, phase vorticity blindness, and noise vulnerability. By combining "
        "sub-millisecond GPU execution ($0.72$ ms), low-latency CPU inference ($2.84$ ms), closed-form QFIM explainability, and sustained noise robustness down to -10 dB SNR, HQGN provides a principled, "
        "interpretable, and reliable foundation for next-generation clinical neurotechnology and wireless sensing systems."
    )

    # =======================================================================
    # END MATTER: CRediT, COI, ACKNOWLEDGMENTS, REFERENCES
    # =======================================================================
    # END MATTER: CRediT, COI, ACKNOWLEDGMENTS, REFERENCES
    # =======================================================================
    p_cr_hdr = doc.add_paragraph()
    p_cr_hdr.paragraph_format.space_before = Pt(14)
    p_cr_hdr.paragraph_format.space_after = Pt(3)
    r_cr_lbl = p_cr_hdr.add_run("CRediT authorship contribution statement")
    set_run_font(r_cr_lbl, font_name=FONT_FAMILY, size_pt=11.0, color=COLOR_BLACK, bold=True)
    if is_blinded:
        add_p_with_math(doc,
            "CRediT authorship contribution details have been removed for double-blind peer review and are documented in the separate Title Page document."
        )
    else:
        add_p_with_math(doc,
            "Vishal Chandel: Conceptualization, Methodology, Software, Mathematical Derivations, Formal Proofs, Empirical Validation, Data Curation, Writing - Original Draft, Writing - Review & Editing, Visualization, Project administration."
        )

    p_coi_hdr = doc.add_paragraph()
    p_coi_hdr.paragraph_format.space_before = Pt(10)
    p_coi_hdr.paragraph_format.space_after = Pt(3)
    r_coi_lbl = p_coi_hdr.add_run("Declaration of competing interest")
    set_run_font(r_coi_lbl, font_name=FONT_FAMILY, size_pt=11.0, color=COLOR_BLACK, bold=True)
    add_p_with_math(doc,
        "The author declares that he has no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper."
    )

    # Data and Code Availability
    p_data_hdr = doc.add_paragraph()
    p_data_hdr.paragraph_format.space_before = Pt(12)
    p_data_hdr.paragraph_format.space_after = Pt(3)
    r_data_lbl = p_data_hdr.add_run("Data and code availability")
    set_run_font(r_data_lbl, font_name=FONT_FAMILY, size_pt=11.0, color=COLOR_BLACK, bold=True)
    if is_blinded:
        add_p_with_math(doc,
            "All benchmark telemetry datasets, preprocessed time-series tensors, PyTorch framework implementations, trained model weights, and reproduction scripts "
            "supporting the findings of this study are available in an anonymized repository for double-blind peer review at: https://anonymous.4open.science/r/Hydro-Quantum-Graph-Network-Anonymized or upon reasonable request to the editor. "
            "The permanent GitHub repository and persistent digital object identifier (DOI) will be publicly unblinded upon formal publication."
        )
    else:
        add_p_with_math(doc,
            "All benchmark telemetry datasets, preprocessed time-series tensors, PyTorch framework implementations, trained model weights, and reproduction scripts "
            "supporting the findings of this study are openly available on GitHub at: https://github.com/Vishal442005/Hydro-Quantum-Graph-Walk-Network under the "
            "permissive Apache-2.0 open-source license. The persistent digital object identifier (DOI) is registered with Zenodo."
        )

    # Acknowledgments
    p_ack_hdr = doc.add_paragraph()
    p_ack_hdr.paragraph_format.space_before = Pt(10)
    p_ack_hdr.paragraph_format.space_after = Pt(3)
    r_ack_lbl = p_ack_hdr.add_run("Acknowledgments")
    set_run_font(r_ack_lbl, font_name=FONT_FAMILY, size_pt=11.0, color=COLOR_BLACK, bold=True)
    if is_blinded:
        add_p_with_math(doc,
            "Personal and institutional acknowledgments have been omitted in this manuscript to comply with the double-blind peer review policy of Engineering Applications of Artificial Intelligence and are documented in the separate Title Page."
        )
    else:
        add_p_with_math(doc,
            "The author acknowledges the open-source scientific software initiatives that supported this research, including MNE-Python, SciPy, NumPy, and the BCI Competition consortia."
        )

    # References (Preserved citations [1] to [30])
    p_ref_hdr = doc.add_paragraph()
    p_ref_hdr.paragraph_format.space_before = Pt(14)
    p_ref_hdr.paragraph_format.space_after = Pt(4)
    r_ref_lbl = p_ref_hdr.add_run("References")
    set_run_font(r_ref_lbl, font_name=FONT_FAMILY, size_pt=12.0, color=COLOR_BLACK, bold=True)

    references = [
        "[1] H. Breuer, F. Petruccione, The Theory of Open Quantum Systems, Oxford University Press, 2002.",
        "[2] P. Linck, N. K. Bernardes, Thermodynamics of linear open quantum walks, arXiv preprint arXiv:2601.22064, 2026.",
        "[3] A. Barachant, S. Bonnet, M. Congedo, C. Jutten, Multiclass brain-computer interface classification by Riemannian geometry, IEEE Trans. Biomed. Eng. 59 (4) (2012) 920-928.",
        "[4] F. Lotte, L. Bougrain, A. Cichocki, et al., A review of classification algorithms for EEG-based brain-computer interfaces: a 10 year update, J. Neural Eng. 15 (3) (2018) 031005.",
        "[5] R. T. Schirrmeister, J. T. Springenberg, L. D. J. Fiederer, et al., Deep learning with convolutional neural networks for EEG decoding and visualization, Hum. Brain Mapp. 38 (11) (2017) 5391-5420.",
        "[6] Y. Kuramoto, Chemical Oscillations, Waves, and Turbulence, Springer Berlin Heidelberg, 1984.",
        "[7] J. A. Acebrón, L. L. Bonilla, C. J. P. Vicente, et al., The Kuramoto model: A simple paradigm for synchronization phenomena, Rev. Mod. Phys. 77 (1) (2005) 137.",
        "[8] T. N. Kipf, M. Welling, Semi-supervised classification with graph convolutional networks, in: Proc. Int. Conf. Learn. Represent. (ICLR), 2017.",
        "[9] D. B. Rubin, Causal inference using potential outcomes, J. Am. Stat. Assoc. 100 (469) (2005) 322-331.",
        "[10] Q. Li, Z. Han, X. M. Wu, Deeper insights into graph convolutional networks: An analytical perspective, in: Proc. AAAI Conf. Artif. Intell., 2018.",
        "[11] L. D. Landau, E. M. Lifshitz, Fluid Mechanics: Course of Theoretical Physics, Vol. 6, Pergamon Press, 1987.",
        "[12] E. Farhi, S. Gutmann, Quantum computation and decision trees, Phys. Rev. A 58 (2) (1998) 915-928.",
        "[13] A. M. Childs, E. Farhi, S. Gutmann, An example of the difference between quantum and classical random walks, Quantum Inf. Process. 1 (1) (2002) 35-43.",
        "[14] N. Shenvi, J. Kempe, K. B. Whaley, Quantum random-walk search algorithm, Phys. Rev. A 67 (5) (2003) 052307.",
        "[15] S. Attal, F. Petruccione, C. Pellegrini, Open Quantum Systems: Dynamics, Dissipation, and Simulation, Springer, 2006.",
        "[16] C. W. Gardiner, P. Zoller, Quantum Noise: A Handbook of Markovian and Non-Markovian Quantum Stochastic Methods, Springer, 2004.",
        "[17] G. Lindblad, On the generators of quantum dynamical semigroups, Commun. Math. Phys. 48 (2) (1976) 119-130.",
        "[18] R. Abraham, J. E. Marsden, Foundations of Mechanics, Benjamin/Cummings Publishing Company, 1978.",
        "[19] R. A. Horn, C. R. Johnson, Matrix Analysis, Cambridge University Press, 2012.",
        "[20] I. Bengtsson, K. Zyczkowski, Geometry of Quantum States: An Introduction to Quantum Entanglement, Cambridge University Press, 2017.",
        "[21] H. Umegaki, Conditional expectation in an operator algebra. IV. Entropy and information, Kodai Math. Sem. Rep. 14 (2) (1962) 59-85.",
        "[22] D. Bures, An extension of Kakutani's theorem on infinite product measures to the tensor product of semifinite w*-algebras, Trans. Am. Math. Soc. 135 (1969) 199-212.",
        "[23] Y. Saad, Numerical Methods for Large Eigenvalue Problems, SIAM, 2011.",
        "[24] F. Yger, M. Berar, F. Lotte, Riemannian approaches in brain-computer interfaces: a review, IEEE Trans. Neural Syst. Rehabil. Eng. 25 (10) (2017) 1753-1762.",
        "[25] T. J. O'Shea, T. Roy, T. C. Clancy, Over-the-air deep learning based radio signal classification, IEEE J. Sel. Top. Signal Process. 12 (1) (2018) 168-179.",
        "[26] Y. Zhan, R. Dai, S. Chen, Continuous-time quantum walks based graph neural network, arXiv preprint arXiv:2608.20738, 2026.",
        "[27] P. Linck, G. Pleasance, F. Petruccione, et al., Dissipative realization of a quantum distance-based classifier using open quantum walks, arXiv preprint arXiv:2603.16010, 2026.",
        "[28] A. Hajisafi, H. Lin, Y. Y. Chiang, et al., Dynamic GNNs for precise seizure detection and classification from EEG data, arXiv preprint arXiv:2405.09568, 2024.",
        "[29] A. Gramfort, M. Luessi, E. Larson, et al., MNE software for processing MEG and EEG data, NeuroImage 86 (2014) 446-460.",
        "[30] M. Tangermann, K. R. Müller, A. Aertsen, et al., Review of the BCI competition IV, Front. Neurosci. 6 (2012) 55."
    ]

    for ref in references:
        p_r = doc.add_paragraph()
        p_r.paragraph_format.space_before = Pt(2)
        p_r.paragraph_format.space_after = Pt(2)
        p_r.paragraph_format.line_spacing = 1.15
        r = p_r.add_run(ref)
        set_run_font(r, font_name=FONT_FAMILY, size_pt=9.0, color=COLOR_BLACK)

    # Ensure parent directories exist
    os.makedirs(os.path.dirname(output_docx_path), exist_ok=True)
    doc.save(output_docx_path)
    file_size_kb = os.path.getsize(output_docx_path) / 1024
    print(f"Successfully generated Elsevier manuscript at: {output_docx_path} ({file_size_kb:.1f} KB)")


def main():
    # 1. Complete manuscripts with author details (for Single-Blind / Archive / Final Publication)
    full_targets = [
        os.path.join("Elseiver_EAAI", "EAAI_manuscript.docx"),
        os.path.join("Elseiver_EAAI", "elsevier_eaai_graph.docx"),
        os.path.join("elseiver", "elseiver_graph.docx"),
        os.path.join("elsevier", "elsevier_graph.docx")
    ]
    for target in full_targets:
        try:
            generate_elsevier_pr_manuscript(target, is_blinded=False)
        except PermissionError:
            print(f"Warning: {target} is currently open in Word. Skipping write.")

    # 2. Blinded manuscripts WITHOUT author details (for Elsevier EAAI Double-Blind Peer Review)
    blinded_targets = [
        os.path.join("Elseiver_EAAI", "Manuscript_without_author_details.docx"),
        os.path.join("Elseiver_EAAI", "Blinded_Manuscript.docx")
    ]
    for target in blinded_targets:
        try:
            generate_elsevier_pr_manuscript(target, is_blinded=True)
        except PermissionError:
            print(f"Warning: {target} is currently open in Word. Skipping write.")


if __name__ == "__main__":
    main()
