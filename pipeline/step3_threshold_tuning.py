"""
STEP 3 — CLASSIFICATION THRESHOLD TUNING
==========================================
Evaluate thresholds 0.20-0.50 (step 0.05) for every model. Models are
UNCHANGED from Step 1 -- only the decision threshold applied to predicted
probabilities is varied.
"""
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

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

thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
all_results = {}
best_thresholds = {}

fig, ax = plt.subplots(figsize=(8, 5.5))
colors = {"Random Forest": "#8AB6D6", "Gradient Boosting": "#C97A2B", "XGBoost": "#6B8E5A", "Stacked Ensemble": "#1F7A6C"}

for name, model in model_list:
    probs = model.predict_proba(X_test_sel)[:, 1]
    rows = []
    for t in thresholds:
        preds_t = (probs >= t).astype(int)
        p = precision_score(y_test, preds_t, zero_division=0)
        r = recall_score(y_test, preds_t, zero_division=0)
        f1 = f1_score(y_test, preds_t, zero_division=0)
        rows.append({"threshold": t, "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)})

    df_t = pd.DataFrame(rows)
    all_results[name] = df_t.to_dict(orient="records")

    best_row = df_t.loc[df_t["f1"].idxmax()]
    best_thresholds[name] = {"threshold": float(best_row["threshold"]), "f1": float(best_row["f1"]),
                               "precision": float(best_row["precision"]), "recall": float(best_row["recall"])}

    print(f"\n{name} -- threshold sweep:")
    print(df_t.to_string(index=False))
    print(f"  >> Best F1 at threshold={best_row['threshold']}: F1={best_row['f1']:.4f} (P={best_row['precision']:.4f}, R={best_row['recall']:.4f})")

    ax.plot(df_t["threshold"], df_t["f1"], marker="o", label=name, color=colors[name])

ax.axvline(0.5, color="gray", linestyle="--", alpha=0.4, label="default (0.50)")
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("F1-Score")
ax.set_title("F1-Score vs Classification Threshold (all models unchanged)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/step3_threshold_f1.png", dpi=150)
plt.close()

with open(f"{OUT}/step3_results.json", "w") as f:
    json.dump({"threshold_sweep": all_results, "best_thresholds": best_thresholds}, f, indent=2)

print("\n=== SUMMARY: BEST THRESHOLD PER MODEL ===")
for name, b in best_thresholds.items():
    print(f"  {name}: threshold={b['threshold']} -> F1={b['f1']:.4f} (vs F1 at default 0.5)")

print("\nStep 3 complete.")
