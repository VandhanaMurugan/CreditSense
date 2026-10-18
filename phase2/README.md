# Phase 2 – calibration, validation and cost-based decisions

Phase 1 (`../pipeline`) reproduced the stacked ensemble. Phase 2 turns it into a decision-ready model. All scripts reuse the Phase 1 split, cleaning and IG-top-10 features and never use the test set for tuning, calibration or threshold choice.

| Module | Script | Question answered |
|---|---|---|
| M1 | `p2_step1_tuned_weighted_stack.py` | Is the stack fairly built? Ablation A (Phase 1) → B (cost-weighted GB member) → C (+ CV-tuned learners) |
| M2 | `p2_step2_calibration.py` | Are scores probabilities? Platt vs isotonic, chosen by 5-fold Brier on a 20% train hold-out |
| M3 | `p2_step3_bootstrap.py` | Do models differ beyond noise? 1,000 paired bootstrap resamples |
| M4 | `p2_step4_cost_threshold.py` | Where should the cut-off be? Expected cost, t* = 1/(1+r), cost-ratio sensitivity |

Planned: M5 multi-dataset replication and ablations, M6 fairness by age group, M7 per-borrower reason codes, M8–M10 scoring API, drift monitoring (PSI), final evaluation.

## Run
```bash
make run                      # Phase 1 pipeline first (creates ./artifacts)
cd phase2 && for s in p2_step1_tuned_weighted_stack p2_step2_calibration p2_step3_bootstrap p2_step4_cost_threshold; do python $s.py; done
```
Outputs (JSON, PNG) go to `phase2/outputs/`. Run time is dominated by M1 (about 4–5 minutes on 2 CPUs).

## Findings so far (test n = 37,500; 2,507 defaulters)
- **Stack fix and tuning:** AUC 0.8647 → 0.8650 → 0.8657; PR-AUC flat (0.4007 / 0.3977 / 0.4002). The paired difference C − A is not significant (95% CI −0.0002 to +0.0021).
- **Calibration:** mean predicted default 0.327 → 0.067 (observed 0.067); Brier 0.146 → 0.049; ECE 0.260 → 0.003. Isotonic creates ties and lowers PR-AUC (0.399 → 0.383); Platt preserves ranking (ECE 0.006). Final choice still open.
- **Significance:** boosted models, LightGBM and the stack are practically equivalent; Random Forest is genuinely weaker (XGBoost − RF AUC +0.0038, CI 0.0021 to 0.0056).
- **Cost-based threshold:** with calibrated probabilities, 1/(1+r) lands within ~0.01 of the empirically best cut-off. Expected cost per 1,000 borrowers vs the Phase 1 default (0.5 on raw scores): −51% at r=2, −21% at r=5, −3% at r=10, −2% at r=20. The F1-best threshold (0.82) costs more than the default at r ≥ 10.

Cost ratios are assumptions for sensitivity analysis, not lender data. Tuning used a stratified 40,000-row subsample of the training split.
