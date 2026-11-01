# Phase 2 – when does stacking pay off, and how should a stacked score become a decision?

Phase 1 (`../pipeline`) reproduced the stacked ensemble. Phase 2 tests four hypotheses on top of it. All scripts reuse the Phase 1 split, cleaning and IG-top-10 features; the test set is never used for tuning, calibration, conformal quantiles or threshold choice.

| Module | Script | Purpose |
|---|---|---|
| F1 | `p2_step1_tuned_weighted_stack.py` | Cost-consistent, CV-tuned stack; ablation A (Phase 1) → B (weighted GB member) → C (+ tuned) |
| F2 | `p2_step2_calibration.py` | Calibration; rule fixed in advance: Platt unless isotonic improves 5-fold Brier by > 0.001 |
| F3 | `p2_step3_bootstrap.py` | 1,000 paired bootstrap resamples; intervals for AUC / PR-AUC and differences |
| F4 | `p2_step4_cost_threshold.py` | Cost-based threshold t* = 1/(1+r), cost-ratio sensitivity |
| N1 (H2) | `p2_step5_diversity.py` | Add logistic regression and MLP to the stack; rank correlation; paired gain |
| N2 (H1) | `p2_step6_size_study.py` | Stack vs XGBoost vs logistic regression at 1k–30k training rows (GMSC) |
| N3 (H3) | `p2_step7_conformal_referral.py` | Class-conditional split-conformal referral vs Chow's cost-optimal rule at matched rate |
| N4 (H4) | `p2_step8_age_fairness.py` | Calibration, decisions, referral and coverage by age band |

Planned: N5 German / Australian / Taiwan datasets (H1), N6 reason codes and a scoring demo, a cost-weighted conformity score.

## Run
```bash
make run                      # Phase 1 pipeline (creates ./artifacts)
make run-phase2               # F1-F4 only; run step2, then steps 5-8, from phase2/ (order: 1,2,3,4,7,8,5,6)
```
Steps 7 and 8 need the arrays saved by step 2; step 5 needs the Phase 1 LightGBM model (`pipeline/step5_lightgbm.py`).

## Findings (GMSC; test n = 37,500; 2,507 defaulters)
- **Calibration:** mean predicted default 0.327 → 0.067 (observed 0.067); Brier 0.146 → 0.049; ECE 0.260 → 0.006; ranking unchanged.
- **Stack fix and tuning:** AUC 0.8647 → 0.8650 → 0.8657; PR-AUC flat; the paired difference C − A is not significant.
- **H2:** adding logistic regression and MLP changes stack AUC by +0.0002 (95% CI −0.0001 to +0.0006). Supported.
- **H1 (one dataset):** stack − XGBoost AUC shrinks from +0.017 (1k rows) to +0.001 (30k); a regularised logistic regression beats the tree stack up to ~3k rows and ties at 10k. Few repeats at large sizes; multi-dataset runs pending.
- **H3:** coverage guarantee holds on test (e.g. 90.2% at α = 0.10), but the band is wide (30% referred) and costs up to 2.5× more than a cost-optimal band at the same rate; fewer defaulters are auto-approved (11.9% vs 17.5%). Cost claim rejected.
- **H4:** calibration holds in every age band (ECE ≤ 0.021); defaulter coverage falls from 97.1% (<30) to 75.0% (70+) against a 90% target. An audit of outcomes, not a fairness claim.
- **Cost threshold:** vs the Phase 1 default (0.5 on raw scores), cost per 1,000 falls 51% at r=2, 21% at r=5, 3% at r=10, 2% at r=20. Cost ratios are assumptions.

Tuning used a stratified 40,000-row subsample of the training split.
