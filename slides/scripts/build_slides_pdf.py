"""
Compiles the IITH Beamer-style presentation PDF (16:9, 720×405pt) using ReportLab.

Architecture:
  - IITHSlideCanvas draws the full-width orange header band + title text, plus footer.
    Content slides supply their title via TitleMarker() flowable.
  - No separate make_header() orange Table → zero double-band issue.
  - Equations are pre-rendered PNGs (matplotlib mathtext) in slides/images/equations/.
  - Slide images are in slides/images/own/ and slides/images/web/.
"""

import os
import re
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
    PageBreak,
    HRFlowable,
    Flowable,
)
from reportlab.pdfgen import canvas

# ---------------------------------------------------------------------------
# Page geometry
# ---------------------------------------------------------------------------
PAGE_W = 720.0
PAGE_H = 405.0
LEFT_MARGIN  = 25
RIGHT_MARGIN = 25
HEADER_H     = 48   # orange header band height (pt) — title lives here
YELLOW_H     = 4    # yellow accent strip height below orange band
TOP_MARGIN   = HEADER_H + YELLOW_H + 6   # 58 pt — clears header + a little breathing room
BOT_MARGIN   = 26   # footer height

# ---------------------------------------------------------------------------
# IITH Colour Palette
# ---------------------------------------------------------------------------
COLOR_ORANGE = colors.HexColor("#F38221")
COLOR_YELLOW = colors.HexColor("#FECB0A")
COLOR_NAVY   = colors.HexColor("#0F2942")
COLOR_DARK   = colors.HexColor("#1E293B")
COLOR_MUTED  = colors.HexColor("#64748B")
COLOR_BG     = colors.HexColor("#F8FAFC")
COLOR_BORDER = colors.HexColor("#CBD5E1")


# ---------------------------------------------------------------------------
# Shared title registry — TitleMarker populates, canvas decorator reads it
# ---------------------------------------------------------------------------
_slide_titles: dict[int, str] = {}


class TitleMarker(Flowable):
    """Zero-height zero-width flowable that records page → title mapping."""
    def __init__(self, title: str):
        super().__init__()
        self.title = title
        self._width = 0
        self._height = 0

    def wrap(self, aW, aH):
        return 0, 0

    def draw(self):
        page = self.canv.getPageNumber()
        _slide_titles[page] = self.title


# ---------------------------------------------------------------------------
# Canvas Decorator — draws IITH chrome on every page at save() time
# ---------------------------------------------------------------------------
class IITHSlideCanvas(canvas.Canvas):
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
            self._draw_chrome(num_pages)
            super().showPage()
        super().save()

    def _draw_chrome(self, page_count: int):
        self.saveState()
        pn = self._pageNumber

        if pn == 1:
            # ---- Title slide: thin top + bottom accent only ----
            self.setFillColor(COLOR_ORANGE)
            self.rect(0, PAGE_H - 14, PAGE_W, 14, fill=True, stroke=False)
            self.setFillColor(COLOR_YELLOW)
            self.rect(0, PAGE_H - 18, PAGE_W, 4, fill=True, stroke=False)
            self.setFillColor(COLOR_ORANGE)
            self.rect(0, 0, PAGE_W, 14, fill=True, stroke=False)
            self.setFillColor(COLOR_YELLOW)
            self.rect(0, 14, PAGE_W, 4, fill=True, stroke=False)

        elif pn == page_count:
            # ---- Thank You slide: same thin accents ----
            self.setFillColor(COLOR_ORANGE)
            self.rect(0, PAGE_H - 14, PAGE_W, 14, fill=True, stroke=False)
            self.setFillColor(COLOR_YELLOW)
            self.rect(0, PAGE_H - 18, PAGE_W, 4, fill=True, stroke=False)
            self.setFillColor(COLOR_ORANGE)
            self.rect(0, 0, PAGE_W, 14, fill=True, stroke=False)
            self.setFillColor(COLOR_YELLOW)
            self.rect(0, 14, PAGE_W, 4, fill=True, stroke=False)

        else:
            # ---- Content slides: full IITH chrome ----
            # --- Top orange header band ---
            self.setFillColor(COLOR_ORANGE)
            self.rect(0, PAGE_H - HEADER_H, PAGE_W, HEADER_H, fill=True, stroke=False)
            # Yellow accent strip below orange
            self.setFillColor(COLOR_YELLOW)
            self.rect(0, PAGE_H - HEADER_H - YELLOW_H, PAGE_W, YELLOW_H,
                      fill=True, stroke=False)

            # Slide title drawn inside the orange band
            title = _slide_titles.get(pn, "")
            self.setFont("Helvetica-Bold", 15)
            self.setFillColor(colors.white)
            # Vertically centre in the orange band
            title_y = PAGE_H - HEADER_H + (HEADER_H - 15) / 2
            self.drawString(16, title_y, title)

            # --- Bottom footer bar ---
            self.setFillColor(COLOR_NAVY)
            self.rect(0, 0, PAGE_W * 0.46, BOT_MARGIN, fill=True, stroke=False)
            self.rect(PAGE_W * 0.46, 0, PAGE_W * 0.34, BOT_MARGIN,
                      fill=True, stroke=False)
            self.setFillColor(COLOR_ORANGE)
            self.rect(PAGE_W * 0.80, 0, PAGE_W * 0.20, BOT_MARGIN, fill=True, stroke=False)

            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.white)
            self.drawString(14, 8, "Alfien Jessurun P & Boopesh D  |  IIT Hyderabad")
            self.drawCentredString(PAGE_W * 0.63, 8, "cINN Multidimensional Detector Unfolding")
            self.drawRightString(PAGE_W - 14, 8, f"{pn} / {page_count}")

        self.restoreState()


# ---------------------------------------------------------------------------
# Paragraph styles
# ---------------------------------------------------------------------------
def _make_styles():
    base = getSampleStyleSheet()

    title_main = ParagraphStyle(
        "TitleMain", parent=base["Normal"],
        fontName="Helvetica-Bold", fontSize=22, leading=29,
        alignment=1, textColor=COLOR_NAVY, spaceAfter=10,
    )
    title_sub = ParagraphStyle(
        "TitleSub", parent=base["Normal"],
        fontName="Helvetica-Bold", fontSize=14, leading=19,
        alignment=1, textColor=COLOR_ORANGE, spaceAfter=16,
    )
    title_authors = ParagraphStyle(
        "TitleAuthors", parent=base["Normal"],
        fontName="Helvetica", fontSize=12, leading=17,
        alignment=1, textColor=COLOR_DARK,
    )
    body = ParagraphStyle(
        "SlideBody", parent=base["Normal"],
        fontName="Helvetica", fontSize=11.5, leading=15.5,
        textColor=COLOR_DARK, spaceAfter=4,
    )
    bullet = ParagraphStyle(
        "SlideBullet", parent=body,
        leftIndent=16, firstLineIndent=-11, spaceAfter=5,
    )
    sub_bullet = ParagraphStyle(
        "SlideSubBullet", parent=body,
        fontName="Helvetica", fontSize=10.5, leading=14,
        leftIndent=30, firstLineIndent=-11, spaceAfter=4,
    )
    caption = ParagraphStyle(
        "Caption", parent=base["Normal"],
        fontName="Helvetica-Oblique", fontSize=9, leading=12,
        alignment=1, textColor=COLOR_MUTED,
    )
    tc = ParagraphStyle(
        "TableCell", parent=base["Normal"],
        fontName="Helvetica", fontSize=9.5, leading=12.5,
        alignment=1,
    )
    tc_bold = ParagraphStyle(
        "TableCellBold", parent=tc, fontName="Helvetica-Bold",
    )
    tc_left = ParagraphStyle(
        "TableCellLeft", parent=tc, alignment=0,
    )
    return dict(
        title_main=title_main, title_sub=title_sub,
        title_authors=title_authors,
        body=body, bullet=bullet, sub_bullet=sub_bullet,
        caption=caption, tc=tc, tc_bold=tc_bold, tc_left=tc_left,
    )


# ---------------------------------------------------------------------------
# Helper: safe image path (absolute)
# ---------------------------------------------------------------------------
def _img(rel: str) -> str:
    """Return absolute path relative to the repo root (parent of slides/)."""
    # __file__ = slides/scripts/build_slides_pdf.py
    # dirname x2 = slides/, dirname x3 = repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(repo_root, rel.replace("/", os.sep))


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build_iith_slides(filename: str = "slides/main.pdf"):
    _slide_titles.clear()

    doc = SimpleDocTemplate(
        filename,
        pagesize=(PAGE_W, PAGE_H),
        leftMargin=LEFT_MARGIN,
        rightMargin=RIGHT_MARGIN,
        topMargin=TOP_MARGIN,
        bottomMargin=BOT_MARGIN + 4,
    )

    S = _make_styles()
    story = []

    # ══════════════════════════════════════════════════════════════════════
    # SLIDE 1 — Title Frame
    # ══════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 32))
    story.append(Paragraph(
        "Unbinned Multidimensional Detector Unfolding<br/>"
        "with Conditional Invertible Neural Networks (cINNs)",
        S["title_main"]
    ))
    story.append(Paragraph(
        "OmniFold <i>Z</i>+jets Benchmark on Delphes Fast Simulation",
        S["title_sub"]
    ))
    story.append(HRFlowable(width="50%", thickness=1.5, color=COLOR_YELLOW, spaceAfter=14))
    story.append(Paragraph(
        "<b>Alfien Jessurun P</b> (ep23btech11001) &nbsp;&nbsp;|&nbsp;&nbsp; "
        "<b>Boopesh D</b> (ep23btech11007)<br/>"
        "Department of Physics<br/>"
        "<b>Indian Institute of Technology Hyderabad</b>",
        S["title_authors"]
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # SLIDE 2 — Problem Statement
    # ══════════════════════════════════════════════════════════════════════
    story.append(TitleMarker("1. Problem Statement: The Detector Inverse Problem"))
    story.append(Spacer(1, 8))

    col_left_2 = [
        Paragraph("&bull; <b>What is Unfolding?</b>", S["body"]),
        Paragraph(
            "&ndash; Particle collisions produce <i>true</i> physics events at the hadron level.",
            S["sub_bullet"]
        ),
        Paragraph(
            "&ndash; Detectors smear, absorb, and distort these events — creating <i>reconstructed</i> data.",
            S["sub_bullet"]
        ),
        Paragraph(
            "&ndash; <b>Unfolding</b> is the process of correcting back to truth-level distributions.",
            S["sub_bullet"]
        ),
        Spacer(1, 4),
        Paragraph("&bull; <b>Why is it Hard?</b>", S["body"]),
        Paragraph(
            "&ndash; Many truth events map to the same detector reading — it's a many-to-one problem.",
            S["sub_bullet"]
        ),
        Paragraph(
            "&ndash; Classical methods (IBU) require binning: impossible beyond 2–3 dimensions.",
            S["sub_bullet"]
        ),
        Spacer(1, 4),
        Paragraph("&bull; <b>Our cINN Solution:</b>", S["body"]),
        Paragraph(
            "&ndash; Learns the full conditional probability <i>p</i>(<i>x</i><sub>truth</sub> | <i>y</i><sub>reco</sub>) continuously, without any bins.",
            S["sub_bullet"]
        ),
        Paragraph(
            "&ndash; Scales to 6 simultaneous jet substructure observables.",
            S["sub_bullet"]
        ),
    ]

    lhc_img_path = _img("slides/images/web/LHC.jpg")
    col_right_2 = [
        Image(lhc_img_path, width=3.5 * inch, height=1.79 * inch),
        Spacer(1, 4),
        Paragraph(
            "LHC proton collisions produce jets. Detectors smear the kinematics — unfolding recovers the truth.",
            S["caption"]
        ),
    ]

    t2 = Table([[col_left_2, col_right_2]], colWidths=[390, 270])
    t2.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(t2)
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # SLIDE 3 — cINN Architecture Overview
    # ══════════════════════════════════════════════════════════════════════
    story.append(TitleMarker("2. cINN Architecture: How It Works"))
    story.append(Spacer(1, 6))

    col_left_3 = [
        Paragraph("&bull; <b>Two Interacting Pathways:</b>", S["body"]),
        Paragraph(
            "&ndash; <b>Conditioning path:</b> The detector-level event <i>y</i> is fed into a small "
            "MLP (<i>h</i>(<i>y</i>)) that extracts rich context features <i>c</i>. "
            "This path has no invertibility constraints.",
            S["sub_bullet"]
        ),
        Paragraph(
            "&ndash; <b>Normalizing flow:</b> The truth-level event <i>x</i> passes through "
            "a chain of K invertible coupling blocks, each conditioned on <i>c</i>.",
            S["sub_bullet"]
        ),
        Spacer(1, 4),
        Paragraph("&bull; <b>Training (Forward Pass):</b>", S["body"]),
        Paragraph(
            "&ndash; Maps <i>x</i> &rarr; <i>z</i> (Gaussian noise). "
            "Minimises the exact negative log-likelihood — no approximations.",
            S["sub_bullet"]
        ),
        Spacer(1, 4),
        Paragraph("&bull; <b>Inference (Inverse Pass):</b>", S["body"]),
        Paragraph(
            "&ndash; Sample <i>z</i> ~ N(0, I), then invert: <i>x</i><sub>unf</sub> = <i>f</i><sup>-1</sup>(<i>z</i>; <i>h</i>(<i>y</i>)).",
            S["sub_bullet"]
        ),
        Paragraph(
            "&ndash; Sampling 10<sup>6</sup> unfolded events takes only seconds.",
            S["sub_bullet"]
        ),
        Spacer(1, 4),
        Paragraph("&bull; <b>Rational-Quadratic Splines (Zuko RQ-NSF):</b>", S["body"]),
        Paragraph(
            "&ndash; Each coupling block uses a flexible monotone spline instead of a simple affine shift — better at modelling sharp peaks and tails.",
            S["sub_bullet"]
        ),
    ]

    cinn_img_path = _img("slides/images/web/cINN.png")
    col_right_3 = [
        Image(cinn_img_path, width=3.2 * inch, height=2.35 * inch),
        Spacer(1, 4),
        Paragraph(
            "cINN: truth <i>x</i> &harr; latent <i>z</i>, conditioned on detector context <i>c</i> = <i>h</i>(<i>y</i>)",
            S["caption"]
        ),
    ]

    t3 = Table([[col_left_3, col_right_3]], colWidths=[390, 270])
    t3.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(t3)
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # SLIDE 4 — Substructure: Coupling Block Detail
    # ══════════════════════════════════════════════════════════════════════
    story.append(TitleMarker("3. Architecture Details: Hyperparameters"))
    story.append(Spacer(1, 6))

    col_left_4 = [
        Paragraph("&bull; <b>What is a Coupling Block?</b>", S["body"]),
        Paragraph(
            "&ndash; At each block, features are split into two halves.",
            S["sub_bullet"]
        ),
        Paragraph(
            "&ndash; The first half is transformed using a spline parameterised by the second half + detector context <i>c</i>.",
            S["sub_bullet"]
        ),
        Paragraph(
            "&ndash; Halves are swapped and the process repeats — ensuring all dimensions are transformed.",
            S["sub_bullet"]
        ),
        Spacer(1, 4),
        Paragraph("&bull; <b>Why Splines (RQ-NSF)?</b>", S["body"]),
        Paragraph(
            "&ndash; More expressive than simple scaling+shift (affine). "
            "Models complex, multi-modal distributions with fewer blocks.",
            S["sub_bullet"]
        ),
        Paragraph(
            "&ndash; Exact analytic inverse — no iterative solving needed.",
            S["sub_bullet"]
        ),
    ]

    table_arch_data = [
        [
            Paragraph("<b>Hyperparameter</b>", S["tc_bold"]),
            Paragraph("<b>2D Case</b>", S["tc_bold"]),
            Paragraph("<b>6D Case</b>", S["tc_bold"]),
        ],
        [Paragraph("Target observables (<i>x</i>)", S["tc"]),
         Paragraph("2 (<i>m</i>, <i>M</i>)", S["tc"]),
         Paragraph("6 (<i>m</i>, <i>M</i>, <i>w</i>, <i>m</i><sub>SD</sub>, <i>z</i><sub>g</sub>, &tau;<sub>2</sub>)", S["tc"])],
        [Paragraph("Conditioning inputs (<i>y</i>)", S["tc"]),
         Paragraph("4", S["tc"]),
         Paragraph("8", S["tc"])],
        [Paragraph("Coupling Blocks (<i>K</i>)", S["tc"]),
         Paragraph("6 blocks", S["tc"]),
         Paragraph("8 blocks", S["tc"])],
        [Paragraph("Spline Bins", S["tc"]),
         Paragraph("8 bins", S["tc"]),
         Paragraph("10 bins", S["tc"])],
        [Paragraph("Conditioner MLP", S["tc"]),
         Paragraph("3 &times; 128", S["tc"]),
         Paragraph("3 &times; 256", S["tc"])],
        [Paragraph("Context MLP <i>h</i>(<i>y</i>)", S["tc"]),
         Paragraph("4 &rarr; 64 &rarr; 64", S["tc"]),
         Paragraph("8 &rarr; 128 &rarr; 128", S["tc"])],
    ]
    t_spec = Table(table_arch_data, colWidths=[145, 120, 145])
    t_spec.setStyle(TableStyle([
        ("GRID",       (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ("BACKGROUND", (0, 0), (-1,  0), colors.HexColor("#FEF3C7")),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))

    t4 = Table([[col_left_4, t_spec]], colWidths=[290, 380])
    t4.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(t4)
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # SLIDE 5 — Preprocessing & Training Engine  (NO equation image placeholder)
    # ══════════════════════════════════════════════════════════════════════
    story.append(TitleMarker("4. Preprocessing & Training Engine"))
    story.append(Spacer(1, 6))

    col_left_5 = [
        Paragraph("&bull; <b>Why Preprocess?</b>", S["body"]),
        Paragraph(
            "&ndash; Raw jet observables have very different shapes: "
            "heavy tails, bounded ranges [0,1], discrete counts, and spikes at zero.",
            S["sub_bullet"]
        ),
        Paragraph(
            "&ndash; The flow works best when inputs look roughly Gaussian — "
            "so each variable is mapped to a smooth, unbounded space before training.",
            S["sub_bullet"]
        ),
        Spacer(1, 4),
        Paragraph("&bull; <b>Invertible Transforms Applied:</b>", S["body"]),
        Paragraph("&ndash; <b>Heavy tails</b> (<i>m</i>, <i>m</i><sub>SD</sub>): log transform.", S["sub_bullet"]),
        Paragraph("&ndash; <b>Bounded variables</b> (<i>w</i>, &tau;<sub>2</sub> &isin; [0,1]): logit transform.", S["sub_bullet"]),
        Paragraph("&ndash; <b>Discrete multiplicity</b> (<i>M</i>): uniform dequantization (adds U(0,1) noise during training; snaps to integer at inference).", S["sub_bullet"]),
        Paragraph("&ndash; <b>Zero-spike</b> (<i>z</i><sub>g</sub> = 0): small training noise; threshold at inference.", S["sub_bullet"]),
        Spacer(1, 4),
        Paragraph("&bull; <b>No Leakage:</b> All transforms fitted only on the Pythia training split (70/15/15).", S["bullet"]),
    ]

    col_right_5 = [
        Paragraph("<b>Training Setup</b>", ParagraphStyle(
            "RH5", parent=S["tc_bold"], alignment=0, fontSize=12, spaceAfter=6,
        )),
        Paragraph("&bull; <b>Objective:</b> Exact Negative Log-Likelihood (NLL). "
                  "Measures how well the network's density matches the training data — no approximations unlike VAEs.", S["bullet"]),
        Paragraph("&bull; <b>Optimizer:</b> AdamW (lr = 10<sup>-3</sup>, weight decay = 10<sup>-5</sup>)", S["bullet"]),
        Paragraph("&bull; <b>Schedule:</b> Cosine Annealing (down to 10<sup>-6</sup>)", S["bullet"]),
        Paragraph("&bull; <b>Regularization:</b> Gradient clipping (1.0), early stopping (patience = 10 epochs)", S["bullet"]),
        Paragraph("&bull; <b>Multi-Seed:</b> Trained across 3 independent seeds for robustness.", S["bullet"]),
    ]

    t5 = Table([[col_left_5, col_right_5]], colWidths=[345, 315])
    t5.setStyle(TableStyle([
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (1, 0), (1,  0), COLOR_BG),
        ("BOX",        (1, 0), (1,  0), 0.5, COLOR_BORDER),
        ("LEFTPADDING",  (1, 0), (1, 0), 10),
        ("RIGHTPADDING", (1, 0), (1, 0), 8),
        ("TOPPADDING",   (1, 0), (1, 0), 8),
        ("BOTTOMPADDING",(1, 0), (1, 0), 8),
    ]))
    story.append(t5)
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # SLIDE 6 — Metrics: Marginal Distances
    # ══════════════════════════════════════════════════════════════════════
    story.append(TitleMarker("5. Evaluation (1/2): Marginal Distribution Metrics"))
    story.append(Spacer(1, 6))

    eq_w1_path = _img("slides/images/equations/eq_w1.png")
    # eq_w1.png: 1980×191 px → at width=4.4in → height = 4.4*(191/1980) in = 0.424 in
    w1_w, w1_h = 4.4 * inch, 4.4 * (191 / 1980) * inch

    col_left_6 = [
        Paragraph("&bull; <b>Wasserstein-1 Distance (W<sub>1</sub>):</b>", S["body"]),
        Paragraph(
            "&ndash; Measures the minimum 'work' needed to reshape one distribution into another — "
            "intuitive physical units (GeV for mass, counts for multiplicity).",
            S["sub_bullet"]
        ),
        Paragraph(
            "&ndash; A W<sub>1</sub> close to zero means unfolded events have the same distribution as truth.",
            S["sub_bullet"]
        ),
        Spacer(1, 3),
        Image(eq_w1_path, width=w1_w, height=w1_h),
        Spacer(1, 6),
        Paragraph("&bull; <b>Kolmogorov-Smirnov (KS) Statistic:</b>", S["body"]),
        Paragraph(
            "&ndash; Maximum gap between two empirical CDFs — purely shape-sensitive, no bin choice needed.",
            S["sub_bullet"]
        ),
        Paragraph(
            "&ndash; KS = 0 is perfect; KS &lt; 0.05 considered excellent agreement.",
            S["sub_bullet"]
        ),
        Spacer(1, 4),
        Paragraph(
            "&bull; <b>Key Result:</b> cINN reduces jet mass W<sub>1</sub> from 3.60 &rarr; 0.25 GeV "
            "(&gt;90% reduction), outperforming classical IBU.",
            S["bullet"]
        ),
    ]

    col_right_6 = [
        Image(_img("slides/images/own/benchmark_2d_mass.png"),
              width=3.2 * inch, height=2.6 * inch),
        Spacer(1, 4),
        Paragraph("2D Jet Mass: Truth (black), Reco (blue), cINN (red), IBU (green)", S["caption"]),
    ]

    t6 = Table([[col_left_6, col_right_6]], colWidths=[385, 275])
    t6.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(t6)
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # SLIDE 7 — Metrics: Multidimensional Validation Gates
    # ══════════════════════════════════════════════════════════════════════
    story.append(TitleMarker("6. Evaluation (2/2): Multidimensional Validation"))
    story.append(Spacer(1, 6))

    col_left_7 = [
        Paragraph("&bull; <b>Pair-C2ST (Classifier Two-Sample Test):</b>", S["body"]),
        Paragraph(
            "&ndash; Trains a LightGBM classifier to distinguish <i>joint</i> pairs "
            "(<i>x</i><sub>truth</sub>, <i>y</i><sub>reco</sub>) from "
            "(<i>x</i><sub>unfolded</sub>, <i>y</i><sub>reco</sub>).",
            S["sub_bullet"]
        ),
        Paragraph(
            "&ndash; If the cINN is perfect, the classifier cannot tell them apart → AUC = 0.5.",
            S["sub_bullet"]
        ),
        Paragraph(
            "&ndash; A null threshold is calibrated using held-out truth events, giving a pass/fail gate.",
            S["sub_bullet"]
        ),
        Paragraph(
            "&ndash; <b>cINN 6D: AUC = 0.4502 &le; 0.5274 (PASS)</b> — indistinguishable from truth.",
            S["sub_bullet"]
        ),
        Spacer(1, 6),
        Paragraph("&bull; <b>Simulation-Based Calibration (SBC):</b>", S["body"]),
        Paragraph(
            "&ndash; Tests whether the <i>posterior uncertainty</i> is correctly quantified, "
            "not just whether the mean is right.",
            S["sub_bullet"]
        ),
        Paragraph(
            "&ndash; Draws many (truth, reco) pairs, runs inference, checks where truth falls "
            "in the ranked posterior samples. If the network is calibrated, ranks are uniformly distributed.",
            S["sub_bullet"]
        ),
        Paragraph(
            "&ndash; Flat rank histograms (p &gt; 0.01) confirm the cINN's uncertainty estimates are reliable.",
            S["sub_bullet"]
        ),
    ]

    col_right_7 = [
        Image(_img("slides/images/own/sbc_ranks_2d.png"),
              width=3.2 * inch, height=2.6 * inch),
        Spacer(1, 4),
        Paragraph("SBC Rank Histograms: Uniform = well-calibrated posterior", S["caption"]),
    ]

    t7 = Table([[col_left_7, col_right_7]], colWidths=[385, 275])
    t7.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(t7)
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # SLIDE 8 — Results Table
    # ══════════════════════════════════════════════════════════════════════
    story.append(TitleMarker("7. Results: Benchmark Across Pythia and Herwig"))
    story.append(Spacer(1, 6))

    bench_data = [
        [
            Paragraph("<b>Method</b>", S["tc_bold"]),
            Paragraph("<b>Pythia W<sub>1</sub><br/>(<i>m</i> / <i>M</i>)</b>", S["tc_bold"]),
            Paragraph("<b>Pythia KS<br/>(<i>m</i> / <i>M</i>)</b>", S["tc_bold"]),
            Paragraph("<b>Pair-C2ST<br/>AUC</b>", S["tc_bold"]),
            Paragraph("<b>Herwig W<sub>1</sub><br/>(<i>m</i> / <i>M</i>)</b>", S["tc_bold"]),
            Paragraph("<b>Herwig KS<br/>(<i>m</i> / <i>M</i>)</b>", S["tc_bold"]),
        ],
        [Paragraph("Reco-as-answer", S["tc"]),
         Paragraph("3.60 / 7.09", S["tc"]), Paragraph("0.208 / 0.329", S["tc"]),
         Paragraph("—", S["tc"]),
         Paragraph("2.33 / 7.50", S["tc"]), Paragraph("0.115 / 0.295", S["tc"])],
        [Paragraph("Prior-only", S["tc"]),
         Paragraph("0.443 / 0.722", S["tc"]), Paragraph("0.034 / 0.036", S["tc"]),
         Paragraph("0.818 (Fail)", S["tc"]),
         Paragraph("1.14 / 2.55", S["tc"]), Paragraph("0.055 / 0.098", S["tc"])],
        [Paragraph("IBU (2D only)", S["tc"]),
         Paragraph("0.412 / 0.589", S["tc"]), Paragraph("0.028 / 0.031", S["tc"]),
         Paragraph("—", S["tc"]),
         Paragraph("1.25 / 2.81", S["tc"]), Paragraph("0.062 / 0.112", S["tc"])],
        [Paragraph("<b>cINN 2D (Ours)</b>", S["tc_bold"]),
         Paragraph("<b>0.254 / 0.241</b>", S["tc_bold"]),
         Paragraph("<b>0.019 / 0.016</b>", S["tc_bold"]),
         Paragraph("<b>0.423 (PASS)</b>", S["tc_bold"]),
         Paragraph("<b>0.85 / 0.20</b>", S["tc_bold"]),
         Paragraph("<b>0.055 / 0.010</b>", S["tc_bold"])],
        [Paragraph("<b>cINN 6D (Ours)</b>", S["tc_bold"]),
         Paragraph("<b>0.302 / 0.223</b>", S["tc_bold"]),
         Paragraph("<b>0.018 / 0.016</b>", S["tc_bold"]),
         Paragraph("<b>0.450 (PASS)</b>", S["tc_bold"]),
         Paragraph("<b>0.70 / 0.26</b>", S["tc_bold"]),
         Paragraph("<b>0.044 / 0.017</b>", S["tc_bold"])],
    ]
    t_bench = Table(bench_data, colWidths=[108, 108, 102, 108, 108, 126])
    t_bench.setStyle(TableStyle([
        ("GRID",          (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ("BACKGROUND",    (0, 0), (-1,  0), colors.HexColor("#FEF3C7")),
        ("BACKGROUND",    (0, 4), (-1,  4), colors.HexColor("#ECFDF5")),
        ("BACKGROUND",    (0, 5), (-1,  5), colors.HexColor("#D1FAE5")),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_bench)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "&bull; <b>Pythia Closure:</b> cINN achieves state-of-the-art results, passing all Pair-C2ST gates.",
        S["bullet"]
    ))
    story.append(Paragraph(
        "&bull; <b>Herwig Generalisation:</b> Strong cross-generator performance — cINN trained on Pythia still unfolds Herwig accurately.",
        S["bullet"]
    ))
    story.append(Paragraph(
        "&bull; <b>Dimensional Scalability:</b> 6D cINN succeeds where classical IBU (memory explosion) is infeasible.",
        S["bullet"]
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # SLIDE 9 — Credits  (was Slide 10 — Slide 9 removed)
    # ══════════════════════════════════════════════════════════════════════
    story.append(TitleMarker("References & Credits"))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "&bull; <b>Methodology &amp; Architecture References:</b>",
        S["body"]
    ))
    story.append(Paragraph(
        "&nbsp;&nbsp;[1] A. Andreassen et al., <i>OmniFold: A Method to Simultaneously Unfold All Observables</i>, "
        "Phys. Rev. Lett. 124, 112002 (2020).",
        S["sub_bullet"]
    ))
    story.append(Paragraph(
        "&nbsp;&nbsp;[2] A. Backes et al., <i>Unbinned Detector Unfolding with Conditional Invertible Neural Networks</i>, "
        "SciPost Phys. Core 6, 065 (2023).",
        S["sub_bullet"]
    ))
    story.append(Paragraph(
        "&nbsp;&nbsp;[3] C. Durkan et al., <i>Neural Spline Flows</i>, NeurIPS 32 (2019).",
        S["sub_bullet"]
    ))
    story.append(Paragraph(
        "&nbsp;&nbsp;[4] G. D'Agostini, <i>A Multidimensional unfolding method based on Bayes' theorem</i>, "
        "NIM A 362, 487–498 (1995).",
        S["sub_bullet"]
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "&bull; <b>All benchmark figures</b> generated directly from our project pipeline "
        "on Delphes fast-simulation data; authored by Alfien Jessurun P and Boopesh D.",
        S["bullet"]
    ))
    story.append(Paragraph(
        "&bull; <b>LHC image</b> (Slide 2): CERN / CMS Collaboration, CC-BY-4.0.",
        S["bullet"]
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # SLIDE 10 — Thank You
    # ══════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 48))
    story.append(Paragraph("Thank You!", ParagraphStyle(
        "ThankTitle", parent=_make_styles()["title_main"],
        fontSize=36, leading=44, textColor=COLOR_ORANGE,
    )))
    story.append(Paragraph("Questions &amp; Discussion", ParagraphStyle(
        "ThankSub", parent=_make_styles()["title_sub"],
        fontSize=16, leading=21, textColor=COLOR_NAVY,
    )))
    story.append(HRFlowable(width="35%", thickness=2, color=COLOR_YELLOW, spaceAfter=18))
    story.append(Paragraph(
        "<b>Alfien Jessurun P</b> (<font color='#F38221'>ep23btech11001@iith.ac.in</font>)<br/>"
        "<b>Boopesh D</b> (<font color='#F38221'>ep23btech11007@iith.ac.in</font>)<br/><br/>"
        "Department of Physics<br/>"
        "<b>Indian Institute of Technology Hyderabad</b>",
        ParagraphStyle(
            "ThankContact", parent=_make_styles()["title_authors"],
            fontSize=12, leading=17,
        )
    ))

    # ---- Build ----
    doc.build(story, canvasmaker=IITHSlideCanvas)
    print(f"✓ Compiled {filename}")


if __name__ == "__main__":
    build_iith_slides("slides/main.pdf")
