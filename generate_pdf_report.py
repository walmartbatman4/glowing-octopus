"""
Script to generate the project report PDF following the exact instructor template.
Includes mathematical formula rendering via matplotlib mathtext and HTML/Unicode formatting.
"""

import os
os.environ['MPLCONFIGDIR'] = os.path.join(os.environ.get('TEMP', '.'), 'mpl_cache')

import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    HRFlowable,
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page count."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        
        # Header (on pages > 1)
        if self._pageNumber > 1:
            self.drawString(
                36, 810, 
                "cINN Unbinned Detector Unfolding — Z+jets Delphes Benchmark Report"
            )
            self.drawRightString(A4[0] - 36, 810, "IITH — Project Report")
            self.setStrokeColor(colors.HexColor("#D0D0D0"))
            self.setLineWidth(0.5)
            self.line(36, 804, A4[0] - 36, 804)

        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(A4[0] - 36, 25, footer_text)
        self.drawString(36, 25, "Alfien Jessurun P (ep23btech11001) & Boopesh D (ep23btech11007)")
        self.setStrokeColor(colors.HexColor("#D0D0D0"))
        self.setLineWidth(0.5)
        self.line(36, 35, A4[0] - 36, 35)
        self.restoreState()


def render_display_equation(tex_str, filename, fontsize=13):
    """Render a LaTeX display equation to a transparent high-DPI image."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig = plt.figure(figsize=(7.0, 0.9))
    fig.text(0.5, 0.5, f"${tex_str}$", ha='center', va='center', fontsize=fontsize)
    plt.savefig(filename, bbox_inches='tight', dpi=300, transparent=True, pad_inches=0.05)
    plt.close(fig)


def build_pdf(filename="Project_Report_cINN_Unfolding.pdf"):
    # Target margins: 0.5 in (~36 pt) to maximize content fit within 4-5 pages
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=40,
        bottomMargin=40,
    )
    
    # Pre-render standalone display equations using matplotlib mathtext
    eq_dir = "results/equations"
    eq1_path = os.path.join(eq_dir, "eq_nf.png")
    eq2_path = os.path.join(eq_dir, "eq_loss.png")
    eq3_path = os.path.join(eq_dir, "eq_w1.png")
    
    render_display_equation(
        r"\log p_X(x \mid y) = \log p_Z(f(x; y)) + \sum_{k=1}^K \log \left| \det J_{f_k}(x_{k-1}; y) \right|",
        eq1_path, fontsize=13
    )
    render_display_equation(
        r"\mathcal{L}_{\mathrm{NLL}}(\theta) = - \frac{1}{N} \sum_{i=1}^N \left[ \log p_Z(f_\theta(x_i; y_i)) + \log \left| \det \frac{\partial f_\theta(x_i; y_i)}{\partial x} \right| \right]",
        eq2_path, fontsize=13
    )
    render_display_equation(
        r"W_1(P, Q) = \int_{-\infty}^{\infty} \left| F_P(t) - F_Q(t) \right| \, dt",
        eq3_path, fontsize=13
    )

    styles = getSampleStyleSheet()

    # Custom styles adhering strictly to font size 10-11 pt
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        alignment=1, # Center
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        alignment=1,
        textColor=colors.HexColor('#334155'),
        spaceAfter=10,
    )
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=14.5,
        textColor=colors.HexColor('#0F2942'),
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True,
    )
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=5,
        spaceAfter=2,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        'Body10',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12.2,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=3.5,
        alignment=4, # Justified
    )
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=10.5,
        alignment=1,
        textColor=colors.HexColor('#475569'),
        spaceAfter=4,
    )
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        alignment=0,
    )
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        alignment=0,
    )

    story = []

    # Title & Metadata Header
    story.append(Paragraph("Unbinned Multidimensional Detector Unfolding with Conditional Invertible Neural Networks (cINNs)", title_style))
    story.append(Paragraph(
        "<b>Alfien Jessurun P</b> (ep23btech11001) &nbsp;|&nbsp; <b>Boopesh D</b> (ep23btech11007)<br/>"
        "Department of Physics, Indian Institute of Technology Hyderabad<br/>"
        "Emails: {ep23btech11001, ep23btech11007}@iith.ac.in",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=6))

    # 1. Problem Statement
    story.append(Paragraph("1. Problem Statement", h1_style))
    p1 = (
        "In experimental high-energy physics (HEP), particle collisions inside detectors such as the CMS experiment "
        "at the Large Hadron Collider (LHC) produce collimated sprays of energetic hadrons known as jets. While theoretical "
        "calculations based on Quantum Chromodynamics (QCD) predict particle-level cross sections (the true <b>Gen</b> level, "
        "<i>x</i> &isin; &real;<sup><i>D</i></sup>), physical measurements collected by detectors (the reconstructed <b>Reco</b> "
        "level, <i>y</i> &isin; &real;<sup><i>M</i></sup>) are severely corrupted by finite tracking resolution, energy smearing, "
        "calorimeter non-compensation, and acceptance selection cuts. <b>Unfolding</b> is the ill-posed inverse problem of estimating "
        "the underlying true multidimensional differential cross sections from distorted, noisy detector observations."
    )
    story.append(Paragraph(p1, body_style))

    p2 = (
        "<b>Limitations of Traditional Methods:</b> Conventional unfolding algorithms, most notably Iterative Bayesian "
        "Unfolding (IBU / D'Agostini), require discretizing physical observables into predefined histogram bins. In <i>D</i> "
        "dimensions, binning suffers from the severe <i>curse of dimensionality</i>: a <i>D</i>-dimensional grid with <i>B</i> "
        "bins per axis scales exponentially as <i>B</i><sup><i>D</i></sup>. For 6 jet substructure observables with 20 bins each, "
        "this demands 20<sup>6</sup> = 6.4 &times; 10<sup>7</sup> cause bins and a transfer matrix with over 4 &times; 10<sup>15</sup> "
        "entries. In practice, binning introduces arbitrary boundary choices, erases intra-bin kinematic correlations, and induces "
        "prohibitive covariance matrix explosion (scaling as <i>O</i>(<i>B</i><sup>4</sup>) in memory).<br/>"
        "<b>The cINN Approach:</b> This project addresses these challenges by formulating unbinned, multidimensional unfolding "
        "using Conditional Invertible Neural Networks (cINNs) based on Rational-Quadratic Neural Spline Flows (RQ-NSF). Rather "
        "than inverting binned transfer matrices, the network directly models the continuous posterior density "
        "<i>p</i>(<i>x</i><sub>gen</sub> | <i>y</i><sub>sim</sub>), enabling exact, event-by-event unbinned sampling in arbitrary "
        "dimensions while strictly preserving detector physical boundary conditions."
    )
    story.append(Paragraph(p2, body_style))

    # 2. Methodology
    story.append(Paragraph("2. Methodology", h1_style))
    
    # 2.1 Dataset
    story.append(Paragraph("2.1 Dataset & Physics Preprocessing", h2_style))
    p_data = (
        "We utilize the public LHC benchmark <i>Z</i>+jets dataset from the OmniFold paper (Zenodo record 3548091). "
        "Events simulate <i>pp</i> &rarr; <i>Z</i> + jets collisions at &radic;<i>s</i> = 14 TeV with a hard <i>Z</i> boson "
        "(<i>p</i><sub>T</sub><sup><i>Z</i></sup> &gt; 200 GeV). Simulations are generated using <b>Pythia 8.243</b> (tune 26, nominal "
        "training set of 1,600,000 events) and <b>Herwig 7.1.5</b> (used strictly for out-of-distribution prior-shift stress tests). "
        "Fast detector simulation is performed with Delphes 3.4.2 modeling CMS. Hardest jets (anti-<i>k</i><sub>T</sub>, <i>R</i> = 0.4) "
        "are matched between truth and detector level using spatial distance &Delta;<i>R</i>(<i>j</i><sub>gen</sub>, <i>j</i><sub>sim</sub>) "
        "&le; 0.4 and <i>p</i><sub>T</sub> &gt; 0, yielding 97.95% matched pairs (unmatched 2.05% fakes and misses are recorded for acceptance corrections)."
    )
    story.append(Paragraph(p_data, body_style))

    p_trans = (
        "<b>Invertible Transform Pipeline:</b> Because physical observables exhibit diverse mathematical supports (positive-definite, "
        "compact intervals, discrete integers, and zero-inflated mass points), normalizing flows require exact, invertible bijectors "
        "<i>x</i> &harr; <i>u</i> with closed-form Jacobian log-determinants log |det <i>J</i>|:<br/>"
        "&bull; <b>Positive Heavy-Tailed Observables (Mass <i>m</i>, Groomed Mass <i>m</i><sub>SD</sub>, <i>p</i><sub>T</sub>):</b> "
        "Clamped to &ge; 0 and mapped via <i>u</i> = log(<i>x</i> + &epsilon;) with log |&part;<i>u</i> / &part;<i>x</i>| = -log(<i>x</i> + &epsilon;). "
        "Inverted exactly via <i>x</i> = max(0, exp(<i>u</i>) - &epsilon;).<br/>"
        "&bull; <b>Bounded Observables (Jet Width, <i>N</i>-subjettiness &tau;<sub>2</sub> &isin; [0, 1]):</b> Scaled into (&delta;, 1 - &delta;) "
        "and mapped via logit(<i>v</i>) = log(<i>v</i> / (1 - <i>v</i>)) with exact sigmoid inverse.<br/>"
        "&bull; <b>Discrete Multiplicity (<i>M</i> &isin; &integers;<sup>+</sup>):</b> Dequantized during training by adding uniform noise "
        "<i>U</i>(0, 1) to establish continuous support. At inference, exact non-zero integers are recovered via <i>M</i> = &lfloor;<i>u</i>&rfloor;.<br/>"
        "&bull; <b>Groomed Momentum Fraction (<i>z</i><sub>g</sub> &isin; [0, 0.5]):</b> Soft Drop Soft-Collinear failures produce an exact point mass "
        "at <i>z</i><sub>g</sub> = 0. We implement a zero-inflated transform adding uniform noise <i>U</i>(0, 10<sup>-4</sup>) to exact zeros during "
        "training, and threshold-snapping values back to 0.0 at inference.<br/>"
        "&bull; <b>Standardization:</b> Mean and standard deviation are fitted <i>strictly on the Pythia training split</i>, avoiding data leakage."
    )
    story.append(Paragraph(p_trans, body_style))

    # 2.2 Architecture & Algorithms
    story.append(Paragraph("2.2 Deep Learning Architecture & Algorithmic Design", h2_style))
    p_arch = (
        "<b>Rational-Quadratic Neural Spline Flows (RQ-NSF):</b> Normalizing flows transform a tractable base density "
        "<i>p</i><sub><i>Z</i></sub>(<i>z</i>) = <i>N</i>(0, <i>I</i>) into the complex target distribution "
        "<i>p</i><sub><i>X</i></sub>(<i>x</i> | <i>y</i>) through a chain of <i>K</i> invertible, differentiable bijections "
        "<i>f</i> = <i>f</i><sub><i>K</i></sub> &compfn; &middot;&middot;&middot; &compfn; <i>f</i><sub>1</sub>. "
        "By the multidimensional change-of-variables theorem, the exact conditional log-likelihood is given by:"
    )
    story.append(Paragraph(p_arch, body_style))

    # Display Equation 1: Change of variables
    if os.path.exists(eq1_path):
        story.append(Spacer(1, 3))
        story.append(Image(eq1_path, width=5.0 * inch, height=0.55 * inch))
        story.append(Spacer(1, 3))

    p_arch_spline = (
        "We implement monotonic rational-quadratic splines (Durkan et al., NeurIPS 2019 via Zuko). Each spline transformation "
        "partitions a compact interval [-<i>B</i>, <i>B</i>] into <i>K</i><sub>b</sub> bins. Inside each bin, the mapping is "
        "parameterized as the ratio of two quadratic polynomials, completely determined by bin widths, bin heights, and internal boundary "
        "derivatives produced by an underlying neural network. RQ-splines provide superior non-linear modeling capacity compared to "
        "standard affine coupling layers (e.g. RealNVP), which often struggle with asymmetric and multi-modal particle distributions.<br/>"
        "<b>Model Architectures:</b><br/>"
        "&bull; <b>2D Target (<i>m</i>, <i>M</i>) with 4D Conditioning (<i>m</i><sub>sim</sub>, <i>M</i><sub>sim</sub>, <i>p</i><sub>T,sim</sub>, &eta;<sub>sim</sub>):</b> "
        "6 spline transforms, 8 bins per dimension, hidden layer size 128 &times; 3, and a context MLP (4 &rarr; 64 &rarr; 64).<br/>"
        "&bull; <b>6D Target (Full Jet Substructure) with 8D Conditioning:</b> 8 spline transforms, 10 bins, hidden size 256 &times; 3, "
        "and a context MLP (8 &rarr; 128 &rarr; 128). Autoregressive ordering alternates forward [0, 1, ..., 5] and reverse [5, 4, ..., 0] "
        "across consecutive blocks to ensure complete multi-feature coupling.<br/>"
        "&bull; <b>Unconditional Control Flow <i>p</i>(<i>x</i>):</b> An architectural replica with identical capacity but with context dimension "
        "set to 0. Serves as a vital negative control to demonstrate that validation metrics are genuinely sensitive to detector conditioning.<br/>"
        "<b>Objective & Optimization:</b> The network parameters &theta; are trained by minimizing the Negative Log-Likelihood (NLL):"
    )
    story.append(Paragraph(p_arch_spline, body_style))

    # Display Equation 2: NLL Loss
    if os.path.exists(eq2_path):
        story.append(Spacer(1, 3))
        story.append(Image(eq2_path, width=5.2 * inch, height=0.60 * inch))
        story.append(Spacer(1, 3))

    p_opt = (
        "Optimization utilizes the AdamW optimizer (learning rate &eta; = 10<sup>-3</sup>, weight decay &lambda; = 10<sup>-5</sup>), "
        "a cosine annealing schedule down to 10<sup>-6</sup>, gradient norm clipping at 1.0, and early stopping (patience = 10 epochs) "
        "evaluated over 3 independent random seeds."
    )
    story.append(Paragraph(p_opt, body_style))

    # 2.3 Evaluation Metrics
    story.append(Paragraph("2.3 Statistical Validation Gates & Metrics", h2_style))
    p_eval = (
        "To rigorously validate unbinned unfolding without relying solely on binned visual comparisons, we establish a multi-tiered validation battery:<br/>"
        "&bull; <b>Classifier Two-Sample Test (Pair-C2ST):</b> A LightGBM gradient-boosted decision tree classifier is trained on a 50/50 split "
        "to distinguish true pairs [<i>x</i><sub>true</sub>, <i>y</i>] from unfolded pairs [<i>x</i><sub>unf</sub>, <i>y</i>]. We construct an empirical null "
        "distribution by splitting genuine truth pairs into two equal halves (20 independent shuffles) to obtain null mean &mu;<sub>null</sub> and "
        "standard deviation &sigma;<sub>null</sub>. A model <b>passes the gate</b> if AUC<sub>model</sub> &le; &mu;<sub>null</sub> + 2&sigma;<sub>null</sub>.<br/>"
        "&bull; <b>1D Marginals:</b> Evaluated using the two-sample Kolmogorov-Smirnov (KS) statistic and the Wasserstein-1 (Earth Mover's) distance:"
    )
    story.append(Paragraph(p_eval, body_style))

    # Display Equation 3: W1 Distance
    if os.path.exists(eq3_path):
        story.append(Spacer(1, 3))
        story.append(Image(eq3_path, width=4.0 * inch, height=0.50 * inch))
        story.append(Spacer(1, 3))

    p_eval2 = (
        "&bull; <b>Simulation-Based Calibration (SBC):</b> For 1,000 test events, we draw <i>K</i> = 100 posterior samples and compute the rank "
        "of the true generator value within the generated posterior ensemble. A well-calibrated posterior produces rank statistics that are "
        "strictly uniform, validated via a &chi;<sup>2</sup> goodness-of-fit test (passing if <i>p</i> &gt; 0.01).<br/>"
        "&bull; <b>Latent Normality:</b> Test pairs (<i>x</i>, <i>y</i>) mapped into the latent base space <i>z</i> = <i>f</i>(<i>x</i>; <i>y</i>) "
        "must satisfy &mu;<sub><i>z</i></sub> &isin; [-0.05, 0.05], &sigma;<sub><i>z</i></sub> &isin; [0.95, 1.05], and off-diagonal covariance |<i>r</i><sub><i>ij</i></sub>| &lt; 0.05.<br/>"
        "&bull; <b>Quantile Slices:</b> Conditioning space <i>y</i> (sim mass) is partitioned into 5 quantiles to confirm conditional closure inside localized kinematic intervals.<br/>"
        "&bull; <b>Prior-Shift Robustness:</b> Models trained strictly on Pythia are evaluated on unseen Herwig events without re-training or re-weighting."
    )
    story.append(Paragraph(p_eval2, body_style))

    # 3. Results
    story.append(Paragraph("3. Results & Benchmarking", h1_style))
    p_res1 = (
        "<b>Milestones & Concrete Deliverables:</b> The project developed a fully automated, reproducible codebase: "
        "14 automated pytest unit tests, deterministic split indices (70/15/15), 12 trained neural flow checkpoints (3 seeds for 2D/6D cINN and Controls), "
        "raw JSON metrics, and 24 high-resolution publication-quality PNG/PDF figures."
    )
    story.append(Paragraph(p_res1, body_style))

    # Primary Benchmark Table
    story.append(Paragraph("<b>Table 1: Comprehensive Unfolding Benchmark Comparison across Pythia and Herwig</b>", h2_style))
    
    table_data = [
        [
            Paragraph("<b>Model / Method</b>", table_cell_bold),
            Paragraph("<b>Pythia <i>W</i><sub>1</sub><br/>(<i>m</i> / <i>M</i>)</b>", table_cell_bold),
            Paragraph("<b>Pythia KS<br/>(<i>m</i> / <i>M</i>)</b>", table_cell_bold),
            Paragraph("<b>Pair-C2ST AUC<br/>[Null Gate]</b>", table_cell_bold),
            Paragraph("<b>Herwig <i>W</i><sub>1</sub><br/>(<i>m</i> / <i>M</i>)</b>", table_cell_bold),
            Paragraph("<b>Herwig KS<br/>(<i>m</i> / <i>M</i>)</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>Reco-as-answer</b> (No unfold)", table_cell_style),
            Paragraph("3.602 / 7.086", table_cell_style),
            Paragraph("0.208 / 0.329", table_cell_style),
            Paragraph("&mdash;", table_cell_style),
            Paragraph("2.325 / 7.501", table_cell_style),
            Paragraph("0.115 / 0.295", table_cell_style),
        ],
        [
            Paragraph("<b>Prior-only</b> (Pythia truth)", table_cell_style),
            Paragraph("0.443 / 0.722", table_cell_style),
            Paragraph("0.034 / 0.036", table_cell_style),
            Paragraph("0.8176 [0.524] <i>(Fail)</i>", table_cell_style),
            Paragraph("1.141 / 2.550", table_cell_style),
            Paragraph("0.055 / 0.098", table_cell_style),
        ],
        [
            Paragraph("<b>Control Flow <i>p</i>(<i>x</i>)</b> (2D)", table_cell_style),
            Paragraph("0.217 / 0.239", table_cell_style),
            Paragraph("0.021 / 0.020", table_cell_style),
            Paragraph("0.8277 [0.524] <i>(Fail)</i>", table_cell_style),
            Paragraph("&mdash;", table_cell_style),
            Paragraph("&mdash;", table_cell_style),
        ],
        [
            Paragraph("<b>IBU (PyUnfold 2D)</b>", table_cell_style),
            Paragraph("0.412 / 0.589", table_cell_style),
            Paragraph("0.028 / 0.031", table_cell_style),
            Paragraph("&mdash;", table_cell_style),
            Paragraph("1.250 / 2.810", table_cell_style),
            Paragraph("0.062 / 0.112", table_cell_style),
        ],
        [
            Paragraph("<b>cINN 2D (Ours)</b>", table_cell_bold),
            Paragraph("<b>0.254 / 0.241</b>", table_cell_bold),
            Paragraph("<b>0.019 / 0.016</b>", table_cell_bold),
            Paragraph("<b>0.4231 &plusmn; 0.0109</b> [0.524] <b>(PASS)</b>", table_cell_bold),
            Paragraph("<b>0.848 / 0.204</b>", table_cell_bold),
            Paragraph("<b>0.055 / 0.010</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>cINN 6D (Ours)</b>", table_cell_bold),
            Paragraph("<b>0.302 / 0.223</b>", table_cell_bold),
            Paragraph("<b>0.018 / 0.016</b>", table_cell_bold),
            Paragraph("<b>0.4502 &plusmn; 0.0195</b> [0.526] <b>(PASS)</b>", table_cell_bold),
            Paragraph("<b>0.696 / 0.255</b>", table_cell_bold),
            Paragraph("<b>0.044 / 0.017</b>", table_cell_bold),
        ],
    ]

    t1 = Table(table_data, colWidths=[120, 80, 80, 95, 75, 73])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('BACKGROUND', (0, 5), (-1, 5), colors.HexColor('#ECFDF5')), # Highlight 2D cINN
        ('BACKGROUND', (0, 6), (-1, 6), colors.HexColor('#F0FDF4')), # Highlight 6D cINN
    ]))
    story.append(t1)
    story.append(Spacer(1, 4))

    # Figure embedding 1: 2D Marginal Comparison and Contour Overlay
    img_mass_path = "results/figures/benchmark_2d_mass.png"
    img_contour_path = "results/figures/contour_2d_mass_mult.png"
    if os.path.exists(img_mass_path) and os.path.exists(img_contour_path):
        img_table = [
            [
                Image(img_mass_path, width=2.55 * inch, height=2.3 * inch),
                Image(img_contour_path, width=2.55 * inch, height=2.3 * inch),
            ],
            [
                Paragraph("<b>Figure 1:</b> 2D Jet Mass unfolding benchmark comparing Truth (black), Reco (blue dashed), cINN (red), and IBU with ratio to Truth.", caption_style),
                Paragraph("<b>Figure 2:</b> 2D density contour overlay: True Gen-level (black solid) vs. cINN Unfolded (red dashed) joint distribution.", caption_style),
            ]
        ]
        t_img = Table(img_table, colWidths=[260, 260])
        t_img.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 1),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(t_img)
        story.append(Spacer(1, 4))

    p_6d_desc = (
        "<b>6D Full Observable Benchmark:</b> When scaled to all six OmniFold observables, cINN achieves dramatic smearing removal: "
        "Jet Width <i>W</i><sub>1</sub> drops from 0.0164 &rarr; 0.0015; Groomed Mass (<i>m</i><sub>SD</sub>) <i>W</i><sub>1</sub> drops from 1.0864 &rarr; 0.3840; and &tau;<sub>2</sub> <i>W</i><sub>1</sub> drops from "
        "0.0082 &rarr; 0.0010. The 15-pairwise corner correlation matrix difference yields a Frobenius norm of ||<i>R</i><sub>diff</sub>||<sub>F</sub> = 0.7695 "
        "and maximum absolute difference of 0.334, capturing complex non-linear phase space correlations without binning."
    )
    story.append(Paragraph(p_6d_desc, body_style))

    # Figure embedding 2: SBC Ranks and Herwig Prior Shift
    img_sbc_path = "results/figures/sbc_ranks_2d.png"
    img_herwig_path = "results/figures/herwig_shift_2d_mult.png"
    if os.path.exists(img_sbc_path) and os.path.exists(img_herwig_path):
        img_table2 = [
            [
                Image(img_sbc_path, width=2.55 * inch, height=1.95 * inch),
                Image(img_herwig_path, width=2.55 * inch, height=1.95 * inch),
            ],
            [
                Paragraph("<b>Figure 3:</b> Simulation-Based Calibration (SBC) rank histograms confirming posterior uniformity (<i>p</i><sub>mass</sub> = 0.020, <i>p</i><sub>mult</sub> = 0.311).", caption_style),
                Paragraph("<b>Figure 4:</b> Prior-shift evaluation on unseen Herwig: detector-level smear of -7.5 counts is restored to within +0.04 counts by cINN.", caption_style),
            ]
        ]
        t_img2 = Table(img_table2, colWidths=[260, 260])
        t_img2.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 1),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(t_img2)
        story.append(Spacer(1, 4))

    # 4. LLM / Agentic AI Usage Disclosure
    story.append(Paragraph("4. LLM / Agentic AI Usage Disclosure", h1_style))
    p_llm = (
        "In accordance with project guidelines, AI assistance was utilized throughout this project. "
        "Specifically, <b>Google Antigravity</b> (powered by Gemini agentic models) was used as an autonomous coding agent to assist in: "
        "(1) diagnosing and resolving Windows C++ build wheel incompatibilities for HEP packages (<code>wasserstein</code> / <code>energyflow</code>), "
        "(2) configuring PyTorch CUDA 13.0 bindings for the NVIDIA GeForce RTX 5070 Ti Laptop GPU (<code>sm_120</code>), "
        "(3) implementing unit tests for analytical Jacobian log-determinants vs. autograd, and "
        "(4) automating batch execution scripts and ReportLab document compilation. "
        "All physics logic, transform mathematical derivations, statistical metric gates, and scientific validation checks were verified "
        "and guided by the authors."
    )
    story.append(Paragraph(p_llm, body_style))

    # 5. Workload Distribution (Option C)
    story.append(Paragraph("5. Workload Distribution", h1_style))
    p_work_intro = (
        "The project was executed by a two-member team with clean division between core generative deep learning modeling "
        "and physics-grounded evaluation and benchmarking (Option C):"
    )
    story.append(Paragraph(p_work_intro, body_style))

    p_work_alfien = (
        "&bull; <b>Student 1: Alfien Jessurun P (Roll: ep23btech11001) — Core Generative Modeling & Architecture Lead</b><br/>"
        "&nbsp;&nbsp;&ndash; Formulated and implemented the conditional Normalizing Flow (cINN) architecture using Rational-Quadratic Neural Spline Flows (Zuko RQ-NSF).<br/>"
        "&nbsp;&nbsp;&ndash; Designed the context embedding Multi-Layer Perceptron (MLP) for detector feature conditioning (<i>y</i> &rarr; <i>c</i>).<br/>"
        "&nbsp;&nbsp;&ndash; Built the invertible preprocessing transforms (<code>LogTransform</code>, <code>Logit01Transform</code>, <code>ZeroInflatedLogitTransform</code>, <code>DequantizeTransform</code>) "
        "and derived analytical Jacobian log-determinants.<br/>"
        "&nbsp;&nbsp;&ndash; Implemented the unconditional Control Flow model <i>p</i>(<i>x</i>) for negative baseline calibration.<br/>"
        "&nbsp;&nbsp;&ndash; Built the multi-seed GPU training loop (<code>train.py</code>, <code>03_train.py</code>) with cosine annealing and early stopping."
    )
    story.append(Paragraph(p_work_alfien, body_style))

    p_work_boopesh = (
        "&bull; <b>Student 2: Boopesh D (Roll: ep23btech11007) — Physics Preprocessing, Baselines & Statistical Evaluation Lead</b><br/>"
        "&nbsp;&nbsp;&ndash; Handled dataset ingestion pipeline from Zenodo (<code>energyflow</code>), Delta-R jet matching (&Delta;<i>R</i> &le; 0.4), and split leakage prevention (70/15/15).<br/>"
        "&nbsp;&nbsp;&ndash; Implemented the classical Iterative Bayesian Unfolding (IBU) baseline via PyUnfold with exact vectorized D'Agostini algorithm.<br/>"
        "&nbsp;&nbsp;&ndash; Designed and conducted the Classifier Two-Sample Tests (Pair-C2ST and Marginal-C2ST) with calibrated null distributions using LightGBM.<br/>"
        "&nbsp;&nbsp;&ndash; Built the Simulation-Based Calibration (SBC) rank uniformity test, latent space diagnostics, and quantile slice tests.<br/>"
        "&nbsp;&nbsp;&ndash; Executed out-of-distribution prior-shift stress tests on unseen Herwig events and synthesized evaluation reports."
    )
    story.append(Paragraph(p_work_boopesh, body_style))

    # 6. Summary & Outlook
    story.append(Paragraph("6. Summary & Outlook", h1_style))
    p_summary = (
        "<b>Summary:</b> This work demonstrates that conditional Invertible Neural Networks (cINNs) provide a robust, mathematically sound, "
        "and scalable framework for multidimensional, unbinned detector unfolding in high-energy physics. On the LHC <i>Z</i>+jets benchmark, "
        "cINN achieves near-perfect same-generator closure (<i>W</i><sub>1</sub> distances reduced by &gt;90% over Reco; Pair-C2ST AUC = 0.423, passing the null gate). "
        "Furthermore, cINN scales seamlessly to 6 continuous observables without suffering from the combinatorial matrix explosion of classical IBU. "
        "Under prior shift on Herwig, the cINN substantially mitigates detector bias, bringing reconstructed multiplicity bias from -7.50 down to +0.04 counts.<br/>"
        "<b>Future Outlook:</b> (1) Incorporating <i>Iterative Reweighting (Iterative cINN)</i> via classifier density ratios to eliminate the residual Herwig prior bias; "
        "(2) Enforcing strict hierarchical physical constraints (<i>m</i><sub>SD</sub> &le; <i>m</i>) directly within the spline latent prior; and "
        "(3) Applying the trained flow to full LHC Run 2 / Run 3 CMS Open Data."
    )
    story.append(Paragraph(p_summary, body_style))

    # References
    story.append(Paragraph("References", h1_style))
    refs = [
        "[1] A. Andreassen, P. T. Komiske, E. M. Metodiev, B. Nachman, and J. Thaler, <i>OmniFold: A Method to Simultaneously Unfold All Observables</i>, Phys. Rev. Lett. 124, 112002 (2020), arXiv:1911.09107.",
        "[2] M. Bellagente, M. Haußmann, M. Luchmann, and T. Plehn, <i>Understanding Intermediate Machine Learning Predictions — A Case for Invertible Networks</i>, SciPost Phys. 9, 074 (2020), arXiv:1911.01429.",
        "[3] A. Backes, A. Butter, M. Dunford, and B. Malaescu, <i>Unbinned Detector Unfolding with Conditional Invertible Neural Networks</i>, SciPost Phys. Core 6, 065 (2023), arXiv:2212.08674.",
        "[4] C. Durkan, A. Bekasov, I. Murray, and G. Papamakarios, <i>Neural Spline Flows</i>, Advances in Neural Information Processing Systems (NeurIPS 32), 2019, arXiv:1906.04032.",
        "[5] G. D'Agostini, <i>A Multidimensional unfolding method based on Bayes' theorem</i>, Nucl. Instrum. Meth. A 362, 487-498 (1995).",
        "[6] S. Talts, M. Betancourt, D. Simpson, A. Vehtari, and A. Gelman, <i>Validating Bayesian Inference Algorithms with Simulation-Based Calibration</i>, arXiv:1804.06788 (2018).",
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle('RefStyle', parent=body_style, fontSize=8, leading=10.5, spaceAfter=2)))

    # Build document with NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Report successfully compiled to {filename}")

if __name__ == "__main__":
    build_pdf()
