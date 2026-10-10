# CreditSense

**Explainable stacked-ensemble credit risk engine** – probability-of-default modelling on imbalanced retail credit data, with leakage-safe preprocessing, calibrated risk tiers and SHAP-based explanations.

Built as the M.Tech (Data Science & AI, PES University) Phase 1 project; extends the stacking approach of Emmanuel, Sun & Wang (2024).

## Headline results
Held-out test set: 37,500 borrowers, 2,507 defaulters (stratified 75/25 split, `random_state=42`).

| Model | AUC | PR-AUC | Recall | F1 |
|---|---|---|---|---|
| Random Forest | 0.8605 | 0.3830 | 0.7284 | 0.3529 |
| Gradient Boosting | 0.8649 | 0.3955 | 0.7770 | 0.3366 |
| XGBoost | 0.8643 | 0.3993 | 0.7718 | 0.3391 |
| **Stacked (RF+GB+XGB → LR)** | **0.8647** | **0.4007** | 0.7718 | 0.3400 |
| LightGBM | 0.8648 | 0.4012 | 0.7738 | 0.3383 |

Stacked-model KS = 0.575. Metrics are at the default 0.5 decision threshold on cost-weighted models; threshold sensitivity (0.20–0.50) is in `results/step3_results.json`.

**Risk tiering** (5 equal-population bands on predicted PD) separates borrowers cleanly: realised default rate rises 0.61% → 0.87% → 2.37% → 5.36% → 24.21% from lowest to highest tier.

**Reading the numbers honestly:** boosted models, the stack and LightGBM are within ~0.001 AUC of each other, which is inside noise for a test set of this size. The stack's value here is stability and a single interpretable meta-learner, not a large lift. On PR-AUC, which matters more at a 6.7% base rate, gains over a single XGBoost are marginal.

## Approach
1. **Split first, then clean.** Stratified 75/25 split; every imputation value and cap is fitted on train only (see `cleaning_params` in `results/step1_results.json`).
2. **Data quality handling**, driven by EDA (`docs/eda/`):
   - Median imputation for `MonthlyIncome` (19.8% missing) and `NumberOfDependents`.
   - Sentinel codes 96/98 in the three late-payment columns replaced by the largest genuine train value below 90.
   - 99.5th-percentile caps on utilisation and debt ratio; one `age = 0` record dropped.
3. **Feature engineering** (6 features): total past-due events, past-due flag, income per dependent, debt-to-income proxy, credit lines per open loan, utilisation bucket.
4. **Feature selection** – Information Gain (`mutual_info_classif`), top 10 of 16 features, computed on train only.
5. **Imbalance** handled with cost-sensitive learning (`class_weight`, `sample_weight`, `scale_pos_weight`) rather than SMOTE, so predicted probabilities stay on the real data distribution.
6. **Models** – Random Forest, Gradient Boosting, XGBoost; stacked with a logistic-regression meta-learner (`StackingClassifier`, 3-fold internal CV). LightGBM as an external comparator.
7. **Explainability** – SHAP `TreeExplainer` on XGBoost (5,000-row sample). Total past-due events (39.0%) and revolving utilisation (22.2%) account for ~61% of attributed importance; age is protective.

## Repository layout
```
pipeline/    step1 … step5 – training, evaluation, threshold tuning, SHAP, LightGBM + risk tiers
notebooks/   executed end-to-end notebook (EDA → models → explainability)
results/     metrics (JSON/CSV) and figures from the last verified run
docs/eda/    exploratory plots that motivated the preprocessing choices
data/        Give Me Some Credit CSV (see data/README.md)
```

## Reproduce
```bash
pip install -r requirements.txt
make run            # runs steps 1-5; outputs go to ./artifacts (git-ignored)
```
Step 1 is the slowest (stacked model with internal CV). Random Forest and stacked-model metrics can move in the 3rd–4th decimal across library versions; gradient-boosting results, tiers and SHAP shares reproduced identically on the pinned versions.

## Known limitations
- The Gradient Boosting member inside the stack is fitted without `sample_weight` (scikit-learn's `StackingClassifier` does not forward it here), so it is not cost-weighted like the standalone GB. Fixing this by wrapping the members in a weighted pipeline is a planned improvement.
- Single train/test split; no repeated CV or confidence intervals on test metrics yet.
- Probabilities are not explicitly calibrated; tiers rely on ranking, not absolute PD.
- Single dataset; no out-of-time validation or fairness audit.

## Planned (Phase 2)
Probability calibration (Platt/isotonic) and reliability curves, repeated stratified CV with bootstrap CIs, weighted stacking fix, cost-based threshold selection using an explicit false-negative/false-positive cost matrix, and monitoring for drift (PSI).

## Reference
Emmanuel, Sun & Wang (2024), stacked-ensemble credit risk modelling.
