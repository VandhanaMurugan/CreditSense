# Phase 2 – when does stacking pay off, and how should a stacked score become a decision?

Review 1 asked the question; Review 2 builds the answer: the **CreditSense decision layer** (`../creditsense`), a calibrated, cost- and capacity-aware approve / refer / decline policy with a finite-sample guarantee, exact reason codes, a scoring service and a monitoring loop.

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
| N5 (H1) | `p2_step9_multidataset.py` | German / Australian / Taiwan (UCI): repeated stratified CV, LR / RF / GB / XGB / stacks, Nadeau–Bengio corrected t-tests |
| N6 | `p2_step10_protocol_audit.py` | Evaluation-protocol audit: SMOTE before CV, whole-data scaling / feature selection vs leak-safe |
| N7 | `p2_step11_stack_protocol_audit.py` | Protocol audit on the stack, plus a control: SMOTE inside training folds |
| N8 | `p2_step12_decision_policy.py` | **Modified algorithm**: CreditSense triage (`creditsense/policy.py`) vs Phase I, Chow, budgeted Chow, symmetric conformal; price of the guarantee; validity over 500 re-splits; label shift; age bands |
| N9 | `p2_step13_build_bundle.py` | Deployable bundle (`artifacts/creditsense_bundle.joblib`) for `creditsense/serve.py` and the FastAPI app |
| N10 | `p2_step14_reason_codes.py` | Exact reason codes for the calibrated stack (`creditsense/explain.py`): additivity, stability, shortcut fidelity |
| N11 | `p2_step15_monitoring.py` | Drift scenarios, PSI vs outcome monitoring, recalibration with a pre-stated promotion rule |
| N13 (part) | `p2_step16_scorecard_baseline.py` | Industry baseline: weight-of-evidence logistic scorecard, with and without the CreditSense triage |

Remaining for Review 3: user-interface demo on the API (N12) and the final comparison write-up (N13).

## Run
```bash
make run                      # Phase 1 pipeline (creates ./artifacts)
make run-phase2               # F1-F4 only; run step2, then steps 5-8, from phase2/ (order: 1,2,3,4,7,8,5,6)
make run-uci                  # steps 9-10 (needs data/uci/)
make review2                  # steps 11-16 (decision policy, bundle, reason codes, monitoring, scorecard)
make test                     # 14 tests
make serve                    # FastAPI at http://127.0.0.1:8000/docs
```
Steps 7 and 8 need the arrays saved by step 2; step 5 needs the Phase 1 LightGBM model (`pipeline/step5_lightgbm.py`).

## Findings (GMSC; test n = 37,500; 2,507 defaulters)
- **Calibration:** mean predicted default 0.327 → 0.067 (observed 0.067); Brier 0.146 → 0.049; ECE 0.260 → 0.006; ranking unchanged.
- **Stack fix and tuning:** AUC 0.8647 → 0.8650 → 0.8657; PR-AUC flat; the paired difference C − A is not significant.
- **H2:** adding logistic regression and MLP changes stack AUC by +0.0002 (95% CI −0.0001 to +0.0006). Supported.
- **H1 (one dataset):** stack − XGBoost AUC shrinks from +0.017 (1k rows) to +0.001 (30k); a regularised logistic regression beats the tree stack up to ~3k rows and ties at 10k. Few repeats at large sizes.
- **H3:** coverage guarantee holds on test (e.g. 90.2% at α = 0.10), but the band is wide (30% referred) and costs up to 2.5× more than a cost-optimal band at the same rate; fewer defaulters are auto-approved (11.9% vs 17.5%). Cost claim rejected.
- **H4:** calibration holds in every age band (ECE ≤ 0.021); defaulter coverage falls from 97.1% (<30) to 75.0% (70+) against a 90% target. An audit of outcomes, not a fairness claim.
- **Cost threshold:** vs the Phase 1 default (0.5 on raw scores), cost per 1,000 falls 51% at r=2, 21% at r=5, 3% at r=10, 2% at r=20. Cost ratios are assumptions.

- **H1 (UCI datasets, leak-safe repeated CV, fixed untuned settings):** the stack is never significantly better than its best tree model (German +0.003, Australian +0.001, Taiwan +0.002 AUC). Logistic regression is within noise of the stack on the two small sets but 0.061 behind on Taiwan. The base paper's Australian AUC is matched; German (0.797 vs 0.944) and Taiwan (0.785 vs 0.870) are not.
- **Protocol audit:** oversampling the whole dataset before CV raises German RF AUC from 0.79 to 0.92 and Taiwan XGB from 0.78 to 0.875. Consistent with, not proof of, the base paper's protocol (which is not described). Taiwan used 1 repeat.

UCI files go in `data/uci/` (see its README; not committed). Run steps 9–10 from `phase2/`.

Tuning used a stratified 40,000-row subsample of the training split.

## Review 2 findings (main policy: r = 5, c = 0.2, referral budget 20%, cap alpha = 10%; costs are assumptions)
- **Decision rule:** CreditSense auto-approves 9.8% of defaulters vs 22.2% for the Phase I rule, at cost 257.2 vs 268.2 per 1,000 applicants. The same budget without a cap costs 130.9 but auto-approves 27.3% of defaulters: the cap has a price, shown in `p2_price_of_guarantee.png`.
- **Guarantee validity:** over 500 re-splits the share of defaulters auto-approved averages 9.94% (95th pct 11.6%); the 95% PAC version averages 8.70%. The cap survives halving or doubling the default rate, while calibration does not.
- **Age bands:** the global cap misses 25.0% of defaulters aged 70+; a per-band cap fixes that but raises their decline rate from 5.8% to 27.4% (a negative factor for elderly applicants under Reg B 1002.6(b)(2)). Reported, not deployed.
- **Reason codes:** exact (max additivity error 1.4e-06); top reason stable in 87%-93% of cases across SHAP backgrounds; age is a top-4 driver for 77% of referred/declined applicants and is never given as a reason.
- **Monitoring:** PSI flags a shift in who applies but misses a downturn (PD 7.9% vs observed 13.4%); re-fitting calibration on 3,000 outcomes cuts ECE 0.052 -> 0.019 and is promoted by the pre-stated rule.
- **Protocol audit on the stack:** German 0.795 (leak-safe) -> 0.922 (SMOTE before CV) vs 0.793 (SMOTE inside folds); Taiwan 0.785 -> 0.933 vs 0.754. The inflation comes from the leak, not SMOTE.
- **Scorecard baseline:** WoE logistic scorecard AUC 0.858 (stack 0.8654); with the same triage it costs 264.1 per 1,000 (stack 257.2). The decision layer is model-agnostic.
