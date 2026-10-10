"""
STEP 2 — DETAILED PERFORMANCE ANALYSIS
========================================
For every model (Random Forest, Gradient Boosting, XGBoost, Stacked Ensemble):
Confusion Matrix, Precision, Recall, F1-Score, Precision-Recall Curve, PR-AUC.
"""
import joblib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                              f1_score, precision_recall_curve, average_precision_score)

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "artifacts")  # models/results written here (git-ignored)
os.makedirs(OUT, exist_ok=True)
data = joblib.load(f"{OUT}/step1_data.joblib")
models = joblib.load(f"{OUT}/step1_models.joblib")

X_test_sel, y_test = data["X_test_sel"], data["y_test"]

model_list = [
    ("Random Forest", models["rf"]),
    ("Gradient Boosting", models["gb"]),
    ("XGBoost", models["xgb"]),
    ("Stacked Ensemble", models["stacked"]),
]

results = {}
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

for i, (name, model) in enumerate(model_list):
    probs = model.predict_proba(X_test_sel)[:, 1]
    preds = model.predict(X_test_sel)

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    pr_auc = average_precision_score(y_test, probs)

    results[name] = {
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "pr_auc": round(pr_auc, 4),
    }
    print(f"\n{name}:")
    print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"  Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}  PR-AUC={pr_auc:.4f}")

    # PR curve subplot
    prec, rec, _ = precision_recall_curve(y_test, probs)
    axes[i].plot(rec, prec, color="#1F7A6C", linewidth=2)
    axes[i].set_title(f"{name}\nPR-AUC={pr_auc:.3f}", fontsize=11)
    axes[i].set_xlabel("Recall")
    axes[i].set_ylabel("Precision" if i == 0 else "")
    axes[i].set_ylim(0, 1)
    axes[i].axhline(y_test.mean(), color="gray", linestyle="--", alpha=0.5, label="baseline")

plt.tight_layout()
plt.savefig(f"{OUT}/step2_pr_curves.png", dpi=150)
plt.close()

# Confusion matrix heatmaps grid
fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4))
for i, (name, model) in enumerate(model_list):
    preds = model.predict(X_test_sel)
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes2[i],
                xticklabels=["Pred: No Default", "Pred: Default"],
                yticklabels=["Actual: No Default", "Actual: Default"], cbar=False)
    axes2[i].set_title(name, fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/step2_confusion_matrices.png", dpi=150)
plt.close()

with open(f"{OUT}/step2_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nStep 2 complete. Plots and results saved.")
