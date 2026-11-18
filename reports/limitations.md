# Limitations and Methodological Analysis

This document outlines the operational boundaries, systematic considerations, and known limitations of unbinned detector unfolding using conditional Invertible Neural Networks (cINNs) on the Z+jets Delphes benchmark.

---

## 1. Prior Dependence and Mis-modelling Shifts

### Observation
While cINN unfolding conditions directly on reconstructed events $y$ rather than relying on global cause-effect inversion, the network's internal mapping is fundamentally trained on the joint density $p_{\text{MC}}(x, y)$ of the nominal simulation (here, Pythia 8.243 tune 26). When evaluating on a distinct physics generator (Herwig 7.1.5 with alternate hadronization and shower models):
- **W1 Distances**: cINN reduces Wasserstein-1 distances significantly compared to Reco-as-answer (e.g. $m$: $0.696$ vs $2.325$; $M$: $0.255$ vs $7.501$).
- **Residual Bias**: A residual bias remains in the mean observables (e.g., $+0.68$ GeV on jet mass) because the response mapping learned in the high-density regions of Pythia does not perfectly span the phase space populated by Herwig.
- **Pair-C2ST under Prior Shift**: The Pair-C2ST classifier on Herwig yields an AUC of $0.5643$ (above the null threshold $0.5110$), showing that while marginals close well, the joint correlation structure exhibits detectable prior-shift residual differences.

### Remediation
For severe prior divergence, the iterative reweighting procedure (Backes et al., arXiv:2212.08674) can be employed, using a classifier ratio $w(x) = p_{\text{unfold}}(x) / p_{\text{prior}}(x)$ to iteratively re-weight training truth events.

---

## 2. Hard Boundaries, Point Masses, and Physical Constraints

### Point Masses ($z_g = 0$)
The groomed momentum fraction $z_g$ from Soft Drop exhibits a discrete delta peak at $0.0$ whenever a jet does not have two resolvable subjets.
- **Handling**: In this pipeline, exact zeros are smoothed with uniform noise $U(0, 10^{-4})$ during training to preserve continuous density modeling in the rational-quadratic spline flow, and snapped back to $0.0$ at inference.
- **Limitation**: Extreme point masses require careful tuning of the noise threshold to avoid inflating the continuous bulk.

### Hierarchical Constraints ($m_{SD} \le m$)
Because Soft Drop removes wide-angle soft radiation, the groomed mass $m_{SD}$ must be strictly less than or equal to the ungroomed jet mass $m$ for every physical jet.
- A standard unconstrained flow does not strictly enforce joint inequality constraints $x_i \le x_j$.
- In our 6D model, $96.16\%$ of events automatically satisfy $m_{SD} \le m$. The remaining $3.84\%$ violate the constraint slightly due to independent spline couplings in the tail, which require kinematic post-processing / clamping.

### Discrete Counts (Multiplicity $M$)
Multiplicity is an integer count $\ge 1$. Handled by dequantization ($+ U(0, 1)$ in training, $\lfloor \cdot \rfloor$ at inference), which guarantees 100% integer recovery with 0 non-integer or zero values.

---

## 3. Matched-Only Training vs. Fakes and Misses

### Impact
- In the Z+jets Delphes dataset, hardest jets at gen and detector level have an unmatched fraction of $2.05\%$ in Pythia26 and $1.85\%$ in Herwig (at $\Delta R > 0.4$).
- Training the cINN exclusively on matched jet pairs $j_{\text{gen}} \leftrightarrow j_{\text{sim}}$ assumes that detector-level jets correspond to an underlying truth jet.
- If uncorrected, detector fakes (jets formed from pileup or split clusters) and misses (truth jets that fail reconstruction) cause acceptance bias.

### Acceptance & Efficiency Framework
- In `src/unfold/acceptance.py`, we implement classifier-based efficiency weights $\epsilon(x) = P(\text{matched} \mid x)$ and purity $1 - p_{\text{fake}}(y)$ to re-weight unfolded events by $1 / \epsilon(x)$.
- Because the unmatched fraction is relatively small ($\sim 2\%$) in this hard $p_T^Z > 200$ GeV sample, the matched-only bulk unfolds accurately. In higher pileup or wider phase spaces, acceptance reweighting is strictly mandatory.

---

## 4. Computational and Binning Scaling

- **cINN**: Scales gracefully to arbitrary dimensions (from 2D to 6D and beyond). Inverting the flow requires only a single forward pass per event ($O(N)$), enabling unbinned analysis and fast event generation.
- **IBU**: Classical binned unfolding requires discretizing the joint response matrix. In 2D with $20 \times 20 = 400$ cause and effect bins, PyUnfold's covariance calculations scale as $O(B^4) = 160,000^2$ elements (requiring $> 190$ GB of RAM). Scaling IBU beyond 2 dimensions is computationally infeasible.
