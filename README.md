# Unbinned Multidimensional Detector Unfolding with Conditional Invertible Neural Networks (cINNs)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Authors:**  
- **Alfien Jessurun P** (`ep23btech11001@iith.ac.in`)  
- **Boopesh D** (`ep23btech11007@iith.ac.in`)  
*Department of Physics, Indian Institute of Technology Hyderabad (IITH)*

---

## 📌 Overview

In high-energy collider physics (e.g., at the Large Hadron Collider, CMS / ATLAS), particle interactions produce theoretical hadron-level final states $x \in \mathbb{R}^D$ that undergo detector distortions, tracking inefficiencies, and calorimeter energy smearing, resulting in reconstructed observations $y \in \mathbb{R}^M$.

**Unfolding** is the inverse problem of recovering truth-level differential cross sections from detector-level data. Classical Iterative Bayesian Unfolding (IBU / D'Agostini) suffers from the curse of dimensionality ($B^D$ bins), failing beyond 2–3 observables.

This repository implements a fully continuous, unbinned unfolding pipeline using **Conditional Invertible Neural Networks (cINNs)** powered by **Rational-Quadratic Neural Spline Flows (RQ-NSF)** via Zuko. The architecture models the complete conditional posterior $p(x_{\text{truth}} \mid y_{\text{reco}})$ continuously, scaling seamlessly from 2D jet targets to full 6D substructure observables on the Delphes fast-simulation $Z+\text{jets}$ benchmark (Zenodo 3548091).

---

## 📊 Benchmark Results

Evaluated on held-out **Pythia8** (nominal closure) and cross-generator stress tests on **Herwig7** (simulating unseen physics priors):

| Method / Model | Pythia $W_1$ ($m$ / $M$) | Pythia KS ($m$ / $M$) | Pair-C2ST AUC [Null Threshold] | Herwig $W_1$ ($m$ / $M$) | Herwig KS ($m$ / $M$) |
|---|---|---|---|---|---|
| **Reco-as-answer** | 3.602 / 7.086 | 0.208 / 0.329 | — | 2.325 / 7.501 | 0.115 / 0.295 |
| **Prior-only (Pythia truth)** | 0.443 / 0.722 | 0.034 / 0.036 | 0.8176 [0.524] *(Fail)* | 1.141 / 2.550 | 0.055 / 0.098 |
| **Control Flow $p(x)$ (2D)** | 0.217 / 0.239 | 0.021 / 0.020 | 0.8277 [0.524] *(Fail)* | — | — |
| **IBU (PyUnfold 2D)** | 0.412 / 0.589 | 0.028 / 0.031 | — | 1.250 / 2.810 | 0.062 / 0.112 |
| **cINN 2D (Ours)** | **0.254 / 0.241** | **0.019 / 0.016** | **0.4231 $\pm$ 0.0109 [0.524]** *(PASS)* | **0.848 / 0.204** | **0.055 / 0.010** |
| **cINN 6D (Ours)** | **0.302 / 0.223** | **0.018 / 0.016** | **0.4502 $\pm$ 0.0195 [0.526]** *(PASS)* | **0.696 / 0.255** | **0.044 / 0.017** |

### Key Findings
1. **Smearing Reduction:** cINN slashes mass $W_1$ transport distance from 3.602 GeV down to 0.254 GeV (>90% smearing reduction).
2. **Prior Invariance (Herwig):** The -7.50 count detector smear in multiplicity is restored back to within +0.04 counts.
3. **Multidimensional Conditioning Gate:** cINN passes the LightGBM Pair-C2ST test (AUC $\le$ calibrated null threshold), proving true joint-distribution compatibility, while prior-only and unconditional flows fail decisively (AUC > 0.81).
4. **Posterior Calibration:** Verified via Simulation-Based Calibration (SBC) rank uniformity tests ($p > 0.01$).

---

## 🏗 Architecture & Design

```
Conditioning Path:
  Detector Jet y ──▶ [ Context MLP h(y) ] ──▶ Conditioning Embedding c
                                                        │
Flow Chain:                                             ▼
  Truth Jet x ──▶ [ RQ-Spline Block 1 ] ──▶ ... ──▶ [ RQ-Spline Block K ] ──▶ Latent z ~ N(0, I)
```

- **Forward (Training):** Maps $x \to z$ conditioned on $c=h(y)$ under exact Negative Log-Likelihood (NLL).
- **Inverse (Inference):** Samples $z \sim \mathcal{N}(0, I)$ and evaluates $x = f^{-1}(z; h(y))$ to generate millions of unbinned unfolded events in seconds.
- **Invertible Preprocessing:** Custom pipeline handling heavy tails ($\log$), bounded domains (logit), discrete multiplicities (uniform dequantization), and zero-inflation.

---

## 📁 Repository Structure

```
.
├── configs/                  # Experiment configurations (YAML)
│   ├── cinn_2d.yaml          # 2D cINN & control model hyperparameters
│   ├── cinn_6d.yaml          # 6D cINN hyperparameters
│   ├── data.yaml             # Dataset splits, matching thresholds, preprocessing
│   └── eval.yaml             # Evaluation gates, SBC, Pair-C2ST, and bin settings
├── reports/                  # Detailed markdown technical reports
│   ├── data_inspection.md    # Dataset distributions & kinematic checks
│   ├── stage1_report.md      # Data pipeline acceptance
│   ├── stage2_report.md      # 2D cINN acceptance & benchmarks
│   ├── stage3_report.md      # 6D cINN acceptance & scalability
│   └── limitations.md        # Methodological scope, boundary conditions & limits
├── results/
│   ├── figures/              # Publication figures (PNG & PDF)
│   └── metrics/              # Raw evaluation JSON files
├── scripts/                  # Reproducible workflow pipeline
│   ├── 01_download_inspect.py
│   ├── 02_preprocess.py
│   ├── 03_train.py
│   ├── 04_evaluate.py
│   └── 05_prior_shift.py
├── slides/                   # 16:9 Beamer-style Presentation
│   ├── main.pdf              # Generated 10-slide high-res PDF presentation
│   ├── main.tex              # LaTeX Beamer source
│   ├── scripts/              # Automated PDF slide compiler (ReportLab)
│   └── images/               # Slide illustrations & rendered equations
├── src/unfold/               # Core Python library
│   ├── acceptance.py         # Detector efficiency & purity corrections
│   ├── baselines.py          # Reco-as-answer, Prior-only, PyUnfold IBU
│   ├── datasets.py           # PyTorch dataset & data loaders
│   ├── features.py           # Observable extraction
│   ├── io.py                 # File I/O & download routines
│   ├── metrics.py            # W1, KS, Chi2, Pair-C2ST, SBC
│   ├── models.py             # Zuko NSF flow architectures
│   ├── plotting.py           # Matplotlib publication plots & error bands
│   ├── sample.py             # Conditional sampling & inverse transforms
│   ├── train.py              # Flow training loops & scheduler
│   └── transforms.py         # Analytical invertible transforms & Jacobians
├── tests/                    # Pytest verification suite (14 unit tests)
├── Makefile                  # Automated make targets
├── Project_Report_cINN_Unfolding.pdf # 4-page formal technical project report
├── pyproject.toml            # Packaging metadata
└── requirements.txt          # Python dependencies
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/walmartbatman4/glowing-octopus.git
cd glowing-octopus

# Setup virtual environment
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
# source .venv/bin/activate

# Install dependencies and editable package
pip install -r requirements.txt
pip install -e . --no-build-isolation --no-deps
```

### 2. Verify with Unit Tests
```bash
pytest tests/ -v
```

### 3. Pipeline Execution (`Makefile`)
```bash
# Run quick verification smoke mode (< 3 mins)
make smoke

# Full reproduction from scratch:
make data     # Stage 1: Download & preprocess Zenodo Delphes dataset
make stage2   # Stage 2: Train & evaluate 2D cINN + Baselines + Prior-shift
make stage3   # Stage 3: Train & evaluate 6D cINN + Pairwise Corner + Prior-shift
make all      # Complete automated run
```

---

## 📑 Deliverables

- **Formal Report:** [`Project_Report_cINN_Unfolding.pdf`](Project_Report_cINN_Unfolding.pdf) (4-page publication-format summary)
- **Slide Presentation:** [`slides/main.pdf`](slides/main.pdf) (10-slide 16:9 IITH-themed presentation)
- **Technical Checklists:** Detailed reports in [`reports/`](reports/)

---

## 📜 References
- A. Andreassen et al., *OmniFold: A Method to Simultaneously Unfold All Observables*, Phys. Rev. Lett. 124, 112002 (2020).
- A. Backes et al., *Unbinned Detector Unfolding with Conditional Invertible Neural Networks*, SciPost Phys. Core 6, 065 (2023).
- C. Durkan et al., *Neural Spline Flows*, Advances in Neural Information Processing Systems (NeurIPS 32), 2019.
- G. D'Agostini, *A Multidimensional unfolding method based on Bayes' theorem*, NIM A 362, 487-498 (1995).
