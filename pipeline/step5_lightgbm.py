"""
STEP 5 — ADD LIGHTGBM
========================
Add LightGBM as an additional model, trained with the SAME imbalance
handling and evaluated with the SAME framework (confusion matrix, P/R/F1,
PR-AUC) and threshold-tuning approach as Steps 2-3, compared against
Random Forest, Gradient Boosting, XGBoost, and the Stacked Ensemble.
"""
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, average_precision_score,
                              precision_recall_curve)

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "artifacts")  # models/results written here (git-ignored)
os.makedirs(OUT, exist_ok=True)
data = joblib.load(f"{OUT}/step1_data.joblib")
models = joblib.load(f"{OUT}/step1_models.joblib")

X_train_sel, y_train = data["X_train_sel"], data["y_train"]
X_test_sel, y_test = data["X_test_sel"], data["y_test"]
scale_pos_weight = data["scale_pos_weight"]

# Train LightGBM with the SAME imbalance-handling philosophy as the other models
lgbm = lgb.LGBMClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1, verbose=-1
)
lgbm.fit(X_train_sel, y_train)

with open(f"{OUT}/step1_results.json") as f:
    step1 = json.load(f)
with open(f"{OUT}/step2_results.json") as f:
    step2 = json.load(f)

# Evaluate LightGBM with the same full framework
probs = lgbm.predict_proba(X_test_sel)[:, 1]
preds = lgbm.predict(X_test_sel)
cm = confusion_matrix(y_test, preds)
tn, fp, fn, tp = cm.ravel()

lgbm_results = {
    "AUC": round(roc_auc_score(y_test, probs), 4),
    "Accuracy": round(accuracy_score(y_test, preds), 4),
    "F1": round(f1_score(y_test, preds), 4),
    "precision": round(precision_score(y_test, preds), 4),
    "recall": round(recall_score(y_test, preds), 4),
    "pr_auc": round(average_precision_score(y_test, probs), 4),
    "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
}
print("LightGBM results:", lgbm_results)

# Threshold sweep for LightGBM (same as Step 3)
thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
rows = []
for t in thresholds:
    preds_t = (probs >= t).astype(int)
    p = precision_score(y_test, preds_t, zero_division=0)
    r = recall_score(y_test, preds_t, zero_division=0)
    f1 = f1_score(y_test, preds_t, zero_division=0)
    rows.append({"threshold": t, "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)})
lgbm_threshold_df = pd.DataFrame(rows)
best_row = lgbm_threshold_df.loc[lgbm_threshold_df["f1"].idxmax()]
print("\nLightGBM threshold sweep:")
print(lgbm_threshold_df.to_string(index=False))
print(f"Best F1 at threshold={best_row['threshold']}: F1={best_row['f1']:.4f}")

# -------------------------------------------------------------------
# FULL 5-MODEL COMPARISON
# -------------------------------------------------------------------
comparison = {
    "Random Forest": {**step1["model_results"]["RandomForest"], **{k: step2["Random Forest"][k] for k in ["precision", "recall", "pr_auc"]}},
    "Gradient Boosting": {**step1["model_results"]["GradientBoosting"], **{k: step2["Gradient Boosting"][k] for k in ["precision", "recall", "pr_auc"]}},
    "XGBoost": {**step1["model_results"]["XGBoost"], **{k: step2["XGBoost"][k] for k in ["precision", "recall", "pr_auc"]}},
    "Stacked Ensemble": {**step1["model_results"]["StackedEnsemble"], **{k: step2["Stacked Ensemble"][k] for k in ["precision", "recall", "pr_auc"]}},
    "LightGBM": {"AUC": lgbm_results["AUC"], "Accuracy": lgbm_results["Accuracy"], "F1": lgbm_results["F1"],
                 "precision": lgbm_results["precision"], "recall": lgbm_results["recall"], "pr_auc": lgbm_results["pr_auc"]},
}

comp_df = pd.DataFrame(comparison).T
comp_df = comp_df[["AUC", "Accuracy", "F1", "precision", "recall", "pr_auc"]]
print("\n=== FULL 5-MODEL COMPARISON ===")
print(comp_df.to_string())
comp_df.to_csv(f"{OUT}/step5_full_comparison.csv")

# Bar chart: AUC and F1 across all 5 models
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
names = list(comparison.keys())
aucs = [comparison[n]["AUC"] for n in names]
f1s = [comparison[n]["F1"] for n in names]
colors = ["#8AB6D6", "#8AB6D6", "#8AB6D6", "#1F7A6C", "#C97A2B"]

axes[0].bar(names, aucs, color=colors)
axes[0].set_ylabel("AUC-ROC")
axes[0].set_title("AUC-ROC Across All 5 Models")
axes[0].set_ylim(0, 1)
for i, v in enumerate(aucs):
    axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
plt.setp(axes[0].get_xticklabels(), rotation=20, ha="right")

axes[1].bar(names, f1s, color=colors)
axes[1].set_ylabel("F1-Score")
axes[1].set_title("F1-Score Across All 5 Models")
axes[1].set_ylim(0, 0.5)
for i, v in enumerate(f1s):
    axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
plt.setp(axes[1].get_xticklabels(), rotation=20, ha="right")

plt.tight_layout()
plt.savefig(f"{OUT}/step5_model_comparison.png", dpi=150)
plt.close()

with open(f"{OUT}/step5_results.json", "w") as f:
    json.dump({
        "lightgbm_results": lgbm_results,
        "lightgbm_threshold_sweep": lgbm_threshold_df.to_dict(orient="records"),
        "lightgbm_best_threshold": {"threshold": float(best_row["threshold"]), "f1": float(best_row["f1"])},
        "full_5model_comparison": comparison,
    }, f, indent=2)

joblib.dump(lgbm, f"{OUT}/step5_lgbm_model.joblib")

# -------------------------------------------------------------------
# RISK TIERING -- actionable output using the Stacked Ensemble's
# probabilities from this leakage-fixed pipeline
# -------------------------------------------------------------------
stacked_probs_final = models["stacked"].predict_proba(X_test_sel)[:, 1]
tier_df = pd.DataFrame({"y_true": y_test.values, "risk_score": stacked_probs_final})
tier_df["risk_tier"] = pd.qcut(
    tier_df["risk_score"], 5,
    labels=["Tier 1 (Lowest Risk)", "Tier 2", "Tier 3", "Tier 4", "Tier 5 (Highest Risk)"]
)
tier_summary = tier_df.groupby("risk_tier", observed=True).agg(
    n_borrowers=("y_true", "count"), actual_default_rate=("y_true", "mean")
).reset_index()
tier_summary["actual_default_rate"] = (tier_summary["actual_default_rate"] * 100).round(2)
tier_summary.to_csv(f"{OUT}/step5_risk_tier_summary.csv", index=False)
print("\n=== RISK TIER SUMMARY (leakage-fixed pipeline) ===")
print(tier_summary)

print("\nStep 5 complete. Full comparison and plots saved.")
