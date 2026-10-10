"""
STEP 4 — FEATURE IMPORTANCE & LEAKAGE ANALYSIS (SHAP)
========================================================
SHAP run on XGBoost -- chosen because it's a single, well-defined tree
model with a fast, exact TreeExplainer. The Stacked Ensemble's prediction
is a meta-learner combination of 3 models' outputs; SHAP on the ensemble
directly would explain the meta-learner's weighting of model OUTPUTS, not
original features, so explaining the strongest/most representative base
model (XGBoost, PR-AUC 0.399, among the top base estimators from Step 2)
against the original features is the standard, more interpretable choice.
"""
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "artifacts")  # models/results written here (git-ignored)
os.makedirs(OUT, exist_ok=True)
data = joblib.load(f"{OUT}/step1_data.joblib")
models = joblib.load(f"{OUT}/step1_models.joblib")

X_test_sel = data["X_test_sel"]
xgb_model = models["xgb"]

explainer = shap.TreeExplainer(xgb_model)
sample = X_test_sel.sample(5000, random_state=42)
shap_values = explainer.shap_values(sample)

# Summary plot
plt.figure()
shap.summary_plot(shap_values, sample, show=False, plot_size=(9, 6))
plt.tight_layout()
plt.savefig(f"{OUT}/step4_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# Mean |SHAP| per feature -- overall importance ranking
mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=sample.columns).sort_values(ascending=False)

# Mean SIGNED SHAP per feature (direction of typical effect) -- correlate
# feature value with its SHAP value to determine if high values push risk
# up or down, on average
direction = {}
for col in sample.columns:
    corr = np.corrcoef(sample[col], shap_values[:, sample.columns.get_loc(col)])[0, 1]
    direction[col] = round(float(corr), 3)

print("=== SHAP: Mean |SHAP value| per feature (overall importance) ===")
print(mean_abs_shap.round(4))

print("\n=== SHAP: Correlation of feature value with its SHAP contribution ===")
print("(positive = higher feature value pushes prediction toward DEFAULT; negative = pushes toward NO DEFAULT)")
for k, v in sorted(direction.items(), key=lambda x: -abs(x[1])):
    print(f"  {k}: {v}")

# Redundancy check -- correlation between the top features themselves
corr_matrix = sample.corr()
print("\n=== Feature-feature correlation (redundancy check, |r| > 0.7 flagged) ===")
redundant_pairs = []
cols = sample.columns.tolist()
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.7:
            redundant_pairs.append((cols[i], cols[j], round(float(r), 3)))
            print(f"  {cols[i]} <-> {cols[j]}: r={r:.3f}")

if not redundant_pairs:
    print("  None found above |r| > 0.7 threshold among the 10 IG-selected features.")

# Leakage suspicion check -- any feature with SUSPICIOUSLY dominant SHAP
# share (a single feature explaining a disproportionate fraction of total
# importance can indicate target leakage)
total_importance = mean_abs_shap.sum()
importance_share = (mean_abs_shap / total_importance * 100).round(2)
print("\n=== Leakage suspicion check (share of total SHAP importance) ===")
print(importance_share)
top_share = importance_share.iloc[0]
leakage_flag = "SUSPICIOUS -- investigate further" if top_share > 50 else "No single feature dominates -- no leakage signal from importance concentration"
print(f"\nTop feature ({importance_share.index[0]}) accounts for {top_share}% of total SHAP importance: {leakage_flag}")

results = {
    "mean_abs_shap": mean_abs_shap.round(4).to_dict(),
    "direction_correlation": direction,
    "redundant_pairs": redundant_pairs,
    "importance_share_pct": importance_share.to_dict(),
    "leakage_check": leakage_flag,
}
with open(f"{OUT}/step4_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nStep 4 complete.")
