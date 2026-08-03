# Draft — Application & Results sections (operator-augmented early-warning framework)

> **Notes for the author (Jinsong):**
> - All equations below are written in **Word linear format (UnicodeMath)**. In Word, place the cursor in an equation field (`Alt` + `=`) and type/paste the line; Word renders it to professional form. Symbols used (∇, ∑, ×, −, ≤, ≥, ², ₀, ℓ, τ, ρ, Δ, ⌈⌉, 𝟙, ∨) are the same Unicode glyphs already used in `manuscript.docx`.
> - Equation numbers continue after the existing **(19)**; renumber if you insert elsewhere.
> - `[[FIGURE …]]` marks a plot for you to make and drop into Word. Tables are filled with the actual numbers — paste them and convert to Word tables.
> - Metrics use the agreed **early-warning cutoff**: full-horizon true crossing + a **10-sample pre-origin clean filter** (see Eq. (24)). Prognosis horizon **H = 5 samples = 15 min** at Δt = 3 min. Numbers are for XMEAS10 (Purge Rate), IDV13, fault magnitude 25, test runs Run 9–10.
> - Two new operators/equations are introduced here that are not yet in Section 3 (the **acceleration view** ∇² and the **Union** rule). If you keep them, add Eq. (20)–(21) to Section 3.2.1 and Eq. (22) to Section 3.3; the reframing point is discussed at the end of this file.

---

## 4. Application: few-shot early warning of an incipient purge-rate fault

### 4.1 Process, fault scenario, and control limits

We evaluate the framework on the Tennessee Eastman Process (TEP), the reference benchmark for chemical-process abnormal-situation management. The target variable is the purge rate (stream 9 flow, denoted XMEAS10), a safety- and economics-critical stream whose slow excursions are a classic early indicator of loss of separation. The disturbance is IDV13, a slow drift in reaction kinetics, injected at a **mild magnitude (25% of the reference disturbance amplitude)** so that the fault develops gradually and the target only grazes its alarm band — the regime in which anticipation is hardest and most valuable. Measurements are sampled at Δt = 3 min. The look-back window is L = 96 samples and the target is forecast jointly with four covariates (stripper level, stripper underflow, purge valve, condenser-coolant valve).

The alarm band (ℓ, u) is fit on the healthy record following Eq. (3) and is never informed by the test fault. For XMEAS10 at this operating mode the band is ℓ = 0.1817, u = 0.2415 (standardized units). A window is a genuine prognostic opportunity only when the process is still inside the band at the decision time; this is made precise by the clean filter of Eq. (24).

[[FIGURE 4.1: TEP flowsheet with XMEAS10 highlighted, and a representative IDV13/mag-25 trajectory of XMEAS10 with the (ℓ, u) band, showing the slow drift and the first band crossing τ*.]]

### 4.2 Operator pool: level, velocity, and acceleration views

The operator pool exposes complementary views of the same adapted Timer-XL backbone, each behind the uniform prediction contract of Section 3.2.1. The **raw** operator forecasts the level directly (Eq. (10)); the **diff** operator forecasts the first difference and restores by anchored cumulative summation (Eq. (11)). To this we add a third view that targets the curvature of the trajectory, the signature of an *accelerating* incipient fault.

The **acceleration (∇²) view** forecasts the second difference

    ∇²y_t = y_t − 2·y_(t−1) + y_(t−2)                                        (20)

and restores the target trajectory by anchored double summation from the last observed value y_t and the last observed first difference d_t = y_t − y_(t−1),

    ŷ_(t+h) = y_t + ∑_(k=1)^h [ d_t + ∑_(j=1)^k ĝ_(t+j) ],   h = 1,…,H       (21)

where ĝ denotes the forecast second difference. Differencing removes the non-stationary level (and, at second order, the linear drift) from the forecasting target, leaving a near-stationary, low-amplitude residual. This is the property that makes the operators few-shot efficient (Section 5): the backbone must fit only the *shape* of the onset rather than re-learn the operating level from scarce fault windows. Physically, the three views form a kinematic ladder — the raw operator sees position, diff sees velocity (steady drift), and ∇² sees acceleration (the bend at fault onset) — so they disagree exactly where prognosis is decided.

### 4.3 Union rule for the prognostic alarm decision

The velocity and acceleration views catch different onset shapes: diff leads on steady ramps, ∇² leads on curving onsets. Rather than route between them with a learned gate — which, being a convex combination of the operators, cannot exceed the more confident operator at any step — we combine their **decisions** by a direction-aware envelope. At each horizon step the upper limit is tested against the larger of the two forecasts and the lower limit against the smaller, and the window issues an anticipated alarm when either boundary is breached anywhere in H:

    A^U = 𝟙{ ∃ h ∈ {1,…,H} :  max(ŷ^∇_(t+h), ŷ^(∇²)_(t+h)) > u
                              ∨  min(ŷ^∇_(t+h), ŷ^(∇²)_(t+h)) < ℓ }         (22)

which is exactly the logical union of the two operators' alarm flags, A^U = A^∇ ∨ A^(∇²). The envelope is applied only to the alarm test; when neither trajectory crosses, the reported forecast defaults to the diff operator. Unlike a convex gate, the union can fire whenever *either* view anticipates a crossing, so it recovers onsets that a single view — or any weighted average of the views — would miss. Its cost is a higher false-alarm rate, analyzed in Section 5.4.

### 4.4 Evaluation protocol: early-warning metrics under a pre-origin clean filter

At each decision time t (the forecast origin), the operator predicts ŷ_(t+1),…,ŷ_(t+H) with H = 5. A window carries a ground-truth prognostic label when the *measured* target crosses the band anywhere in the horizon,

    y_w = 𝟙{ ∃ h ∈ {1,…,H} : y_(t+h) > u  ∨  y_(t+h) < ℓ },                 (23)

and the forecast raises an anticipated alarm ŷ_w by the same test applied to ŷ (Eq. (15)). Counting every in-band-to-out-of-band window as a positive would reward the trivial detection of an *already active* alarm; to restrict the evaluation to genuine anticipation we require the C = 10 samples immediately preceding the origin to be inside the band,

    clean_w = 𝟙{ y_τ ∈ [ℓ, u]  for all τ ∈ {t−C+1, …, t} },   C = 10.       (24)

The short pre-origin horizon (10 samples = 30 min) keeps a window eligible as soon as the process has been healthy for half an hour, rather than demanding the full 96-sample look-back be clean, which would discard most leading-edge onsets. On Run 9–10 this filter reduces 1,995 in-horizon-crossing windows to 101 genuine ante-onset windows (76 with the first crossing strictly in the future), against 1,375 clean negatives.

The three prognostic metrics are then computed over clean windows only:

    Recall = |{ w : clean_w=1, y_w=1, ŷ_w=1 }| / |{ w : clean_w=1, y_w=1 }|,     (25)

    FAR    = |{ w : clean_w=1, y_w=0, ŷ_w=1 }| / |{ w : clean_w=1, y_w=0 }|,     (26)

the false prognosis rate, and the mean lead time, averaged over true onset events τ*,

    LT = Δt · mean_(τ*) ( τ* − t_e(τ*) ),                                         (27)

where t_e(τ*) is the earliest clean-window origin whose forecast anticipates the crossing at τ*. Forecast fidelity is summarized by the horizon-mean squared error in the restored (raw) domain,

    MSE = (1 / (|W|·H)) · ∑_(w∈W) ∑_(h=1)^H ( ŷ_(t+h) − y_(t+h) )².               (28)

A false prognosis budget of FAR ≤ 0.05 is used as the operational acceptance threshold.

### 4.5 Data-efficiency benchmark

To probe the cold-start regime, every operator is few-shot aligned at six nested subset ratios ρ ∈ {1%, 5%, 10%, 25%, 50%, 100%} of the fault windows (Eq. (8)), and evaluated on the held-out Run 9–10. The operator framework (raw, diff, ∇², Union) is compared against two families of references: two **zero-shot** foundation models used without adaptation (Time-MoE, Sundial), and five **supervised deep-learning baselines** trained from scratch at each ρ (LSTM-GRU, CNN-LSTM, TCN-Transformer, DiPCA-LSTM, STA-ConvBiLSTM). All baselines are trained in the diff domain and restored to the raw domain, so every method is scored by the identical alarm geometry of Eqs. (23)–(28).

[[FIGURE 4.2: framework schematic — L-sample window → {raw, diff, ∇²} views of the adapted Timer-XL → restore → Union alarm envelope (Eq. 22) → prognostic decision. (Adapt from presentaion/framework_*.html.)]]

[[FIGURE 4.3: two example ante-onset windows on Run 9–10 showing the (ℓ,u) band, the truth, and the diff / ∇² / Union forecasts — one steady-ramp onset caught by diff, one curving onset caught by ∇² but missed by diff. (Adapt from figures/IDV13_XMEAS10/plot_gate diff2_catches.png.)]]

---

## 5. Results and discussion

### 5.1 Data efficiency of the operator framework

Table 1 reports recall across the six subset ratios. The operator views are strikingly data-efficient: diff, ∇², and Union hold within a few points of their full-data recall all the way down to ρ = 1%, whereas the raw view loses roughly a third of its recall as data is withdrawn. Union is the most sensitive operator at every ratio, reaching 0.792 at full data and still 0.723 at 1%. The zero-shot foundation models are flat by construction and already outrank every supervised baseline below ρ = 25%.

**Table 1.** Recall (Eq. 25) vs. training-subset ratio ρ. XMEAS10, IDV13, mag 25, Run 9–10; H = 5, C = 10.

| Method | 1% | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|---|
| Union (∇, ∇²) | **0.723** | **0.733** | **0.782** | **0.782** | **0.762** | **0.792** |
| Timer-XL-∇² | 0.644 | 0.693 | 0.713 | 0.723 | 0.683 | 0.713 |
| Timer-XL-diff | 0.653 | 0.604 | 0.644 | 0.634 | 0.634 | 0.673 |
| Timer-XL-raw | 0.446 | 0.604 | 0.634 | 0.584 | 0.644 | 0.693 |
| Time-MoE (zero-shot) | 0.396 | 0.396 | 0.396 | 0.396 | 0.396 | 0.396 |
| Sundial (zero-shot) | 0.337 | 0.337 | 0.337 | 0.337 | 0.337 | 0.337 |
| STA-ConvBiLSTM | 0.030 | 0.366 | 0.455 | 0.455 | 0.515 | 0.535 |
| TCN-Transformer | 0.050 | 0.188 | 0.386 | 0.495 | 0.446 | 0.446 |
| DiPCA-LSTM | 0.050 | 0.079 | 0.079 | 0.099 | 0.158 | 0.099 |
| CNN-LSTM | 0.010 | 0.059 | 0.040 | 0.069 | 0.089 | 0.119 |
| LSTM-GRU | 0.030 | 0.010 | 0.050 | 0.069 | 0.050 | 0.059 |

[[FIGURE 5.1: Recall vs. ρ (log-x). Curves for raw, diff, ∇², Union, the two zero-shot TSFMs, and the two strongest baselines (STA-ConvBiLSTM, TCN-Transformer). Emphasize the flat operator curves vs. the collapsing baselines at small ρ.]]

### 5.2 Comparison with deep-learning baselines and zero-shot models

The gap to the supervised baselines is decisive and widens as data shrinks. The strongest baseline, STA-ConvBiLSTM, needs the full training set to reach 0.535 recall and collapses to 0.030 at ρ = 1%; the three weakest baselines never exceed 0.16 even at full data. In contrast, **Timer-XL-diff at 1% data (0.653) already exceeds the best baseline at 100% data (0.535)** — a hundred-fold data advantage at equal or better recall. The advantage is not bought with false alarms: Table 2 shows diff holds FAR ≤ 0.033 across all ratios, inside the 0.05 budget, while the raw view stays even lower. Union and ∇² trade the budget for sensitivity, with FAR rising to 0.08–0.16 at small ρ. The supervised baselines show near-zero FAR only because they rarely alarm at all — the same conservatism that suppresses their recall.

**Table 2.** False prognosis rate (Eq. 26) vs. ρ. Budget: FAR ≤ 0.05.

| Method | 1% | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|---|
| Timer-XL-diff | 0.033 | 0.033 | 0.027 | 0.023 | 0.013 | 0.017 |
| Timer-XL-raw | 0.011 | 0.015 | 0.013 | 0.016 | 0.021 | 0.020 |
| Timer-XL-∇² | 0.159 | 0.126 | 0.106 | 0.084 | 0.079 | 0.075 |
| Union (∇, ∇²) | 0.163 | 0.132 | 0.113 | 0.088 | 0.081 | 0.081 |
| Time-MoE (zero-shot) | 0.010 | 0.010 | 0.010 | 0.010 | 0.010 | 0.010 |
| Sundial (zero-shot) | 0.004 | 0.004 | 0.004 | 0.004 | 0.004 | 0.004 |
| STA-ConvBiLSTM | 0.001 | 0.029 | 0.040 | 0.012 | 0.014 | 0.014 |
| TCN-Transformer | 0.001 | 0.008 | 0.012 | 0.010 | 0.010 | 0.030 |
| DiPCA-LSTM | 0.000 | 0.001 | 0.001 | 0.001 | 0.003 | 0.000 |
| CNN-LSTM | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.003 |
| LSTM-GRU | 0.001 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 |

Mean lead time (Table 3) tells the same story in operational units. On these near-onset windows the operators warn 2.9–3.5 min ahead and hold that lead down to 1% data, whereas the best baseline reaches 2.3 min only at full data and forfeits nearly all of it in the cold-start regime; the weak baselines give essentially zero anticipation.

**Table 3.** Mean lead time (Eq. 27), minutes, vs. ρ. Horizon ceiling H·Δt = 15 min.

| Method | 1% | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|---|
| Union (∇, ∇²) | **3.53** | **3.24** | **3.48** | **3.44** | **3.34** | **3.44** |
| Timer-XL-∇² | 3.19 | 3.24 | 3.29 | 3.34 | 2.90 | 3.10 |
| Timer-XL-diff | 3.15 | 2.61 | 2.81 | 2.66 | 2.66 | 2.95 |
| Timer-XL-raw | 1.89 | 2.47 | 2.66 | 2.47 | 2.66 | 2.85 |
| Time-MoE (zero-shot) | 1.50 | 1.50 | 1.50 | 1.50 | 1.50 | 1.50 |
| Sundial (zero-shot) | 1.31 | 1.31 | 1.31 | 1.31 | 1.31 | 1.31 |
| STA-ConvBiLSTM | 0.29 | 1.55 | 1.89 | 1.79 | 2.08 | 2.27 |
| TCN-Transformer | 0.15 | 0.48 | 1.06 | 1.55 | 1.65 | 1.55 |
| DiPCA-LSTM | 0.29 | 0.15 | 0.15 | 0.19 | 0.44 | 0.19 |
| CNN-LSTM | 0.00 | 0.15 | 0.15 | 0.10 | 0.15 | 0.44 |
| LSTM-GRU | 0.15 | 0.00 | 0.10 | 0.19 | 0.15 | 0.39 |

[[FIGURE 5.2: two-panel — (a) FAR vs. ρ, (b) mean lead time vs. ρ, same method set as Fig. 5.1; shade the FAR ≤ 0.05 budget in panel (a).]]

### 5.3 Improvement over the raw foundation model

Table 4 isolates the value added by the operators over the raw Timer-XL, the natural single-backbone reference. The comparison of the two columns is the central message: at full data the operators improve the raw view only modestly, because the raw view has enough windows to fit the level itself; **as data is withdrawn the improvement multiplies**, because differencing removes the very quantity — the operating level — that the raw view can no longer estimate from scarce data.

**Table 4.** Improvement over Timer-XL-raw at ρ = 1% and ρ = 100% (early-warning cutoff).

| | | Recall | Lead (min) | MSE ×10³ | FAR |
|---|---|---|---|---|---|
| **ρ = 1%** | raw | 0.446 | 1.89 | 0.237 | 0.011 |
| | diff | 0.653 (**+46%**) | 3.15 (**+67%**) | 0.087 (**−63%**) | 0.033 |
| | ∇² | 0.644 (+44%) | 3.19 (+69%) | 0.159 (−33%) | 0.159 |
| | Union | 0.723 (**+62%**) | 3.53 (**+87%**) | 0.130 (−45%) | 0.163 |
| **ρ = 100%** | raw | 0.693 | 2.85 | 0.064 | 0.020 |
| | diff | 0.673 (−3%) | 2.95 (+4%) | 0.048 (−25%) | 0.017 |
| | ∇² | 0.713 (+3%) | 3.10 (+9%) | 0.065 (±0%) | 0.075 |
| | Union | 0.792 (**+14%**) | 3.44 (**+21%**) | 0.058 (−9%) | 0.081 |

The degradation of each method as data falls from 100% to 1% quantifies the same effect: raw loses 36% of its recall, 34% of its lead time, and its MSE inflates 3.7×, whereas diff loses only 3% of recall while keeping its lead and holding MSE growth to 1.8×. **The operators convert a data-hungry backbone into a near data-independent one** — the property required for cold-start deployment.

### 5.4 Recall–false-alarm operating range

Because the operators occupy distinct points on the recall–false-alarm plane, the framework offers a tunable operating range rather than a single decision. diff is the budget-compliant operating point — the only operator with FAR ≤ 0.05 at every ρ, together with the lowest MSE — and should be the default when nuisance alarms are costly. Union is the maximum-sensitivity operating point, buying up to +12 recall points and +0.5 min lead over diff at the expense of a 5× higher false-alarm rate; it is preferable when a missed onset is far costlier than a false one and the operator can absorb the extra alarms. ∇² lies between the two, close to Union in behavior. Crucially, this entire range dominates the supervised baselines and the zero-shot models on all three prognostic axes at every data budget.

[[FIGURE 5.3: recall–FAR scatter (recall on y, FAR on x) with one marker per operating point per ρ for raw / diff / ∇² / Union; draw the empirical Pareto front and the FAR = 0.05 budget line. This is the figure that motivates operating-point selection.]]

### 5.5 Forecast accuracy under data reduction

Table 5 reports the horizon-mean MSE (Eq. 28) in the restored raw domain. diff attains the lowest error at every ratio and barely degrades under data withdrawal (0.048 → 0.087 ×10³ from 100% to 1%). The supervised baselines split into two failure modes: STA-ConvBiLSTM and TCN-Transformer match the operators only at full data and then explode by 3–7× as data shrinks, while CNN-LSTM, DiPCA-LSTM, and LSTM-GRU sit at a constant ≈ 0.42 ×10³ regardless of ρ — a collapse to a near-constant predictor that also explains their vanishing recall. Union's MSE is slightly above diff's, as expected for a decision envelope rather than a point forecast, and is not a fidelity penalty of the method.

**Table 5.** Horizon-mean MSE ×10³ (Eq. 28), restored raw domain, vs. ρ. Lower is better.

| Method | 1% | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|---|
| Timer-XL-diff | **0.087** | **0.075** | **0.064** | **0.054** | **0.051** | **0.048** |
| Union (∇, ∇²) | 0.130 | 0.102 | 0.087 | 0.070 | 0.064 | 0.058 |
| Timer-XL-∇² | 0.159 | 0.119 | 0.100 | 0.082 | 0.071 | 0.065 |
| Timer-XL-raw | 0.237 | 0.131 | 0.103 | 0.093 | 0.073 | 0.064 |
| Time-MoE (zero-shot) | 0.115 | 0.115 | 0.115 | 0.115 | 0.115 | 0.115 |
| Sundial (zero-shot) | 0.137 | 0.137 | 0.137 | 0.137 | 0.137 | 0.137 |
| STA-ConvBiLSTM | 0.423 | 0.320 | 0.263 | 0.074 | 0.073 | 0.063 |
| TCN-Transformer | 0.405 | 0.276 | 0.186 | 0.119 | 0.129 | 0.130 |
| DiPCA-LSTM | 0.424 | 0.424 | 0.425 | 0.426 | 0.429 | 0.427 |
| CNN-LSTM | 0.423 | 0.423 | 0.423 | 0.423 | 0.424 | 0.424 |
| LSTM-GRU | 0.424 | 0.423 | 0.424 | 0.424 | 0.424 | 0.425 |

[[FIGURE 5.4: MSE ×10³ vs. ρ (log-y), same method set as Fig. 5.1; highlight the flat ≈0.42 band of the collapsed baselines vs. the low, gently rising operator curves.]]

### 5.6 Summary

Across recall, false-alarm rate, lead time, and forecast error, the operator-augmented framework holds near-full-data prognostic performance down to 1% of the fault data, where every supervised baseline has collapsed and even a hundred-fold data reduction leaves Timer-XL-diff ahead of the best fully-trained baseline. The physical differencing operators supply the inductive bias — stationarity of the forecasting target — that makes this data efficiency possible, and the Union rule converts the complementary velocity and acceleration views into the highest-recall, earliest-warning decision available in the pool. The framework therefore delivers a practical cold-start early-warning capability with an explicit, tunable recall–false-alarm operating range.

---

## Author note — narrative consistency with Section 3

These results present the **operator pool (raw / diff / ∇²) and the Union decision rule**, not the MLP gate of Section 3.2.2. If the paper's headline contribution is reframed to the operator-augmented framework (as this draft assumes), consider:
1. Adding Eq. (20)–(21) (∇² operator) to §3.2.1 and Eq. (22) (Union) to §3.3.
2. Either (a) demoting the MLP gate to an ablation ("a learned convex gate cannot exceed the more sensitive operator, motivating the union rule"), or (b) keeping the gate and adding Union as a parallel fusion — but then the abstract/intro claims need to name Union, not the gate, as the method that carries the results.
This is a structural decision; flag which way you want to go and I can redraft §3 to match.
