"""Phase 2 / Module 3 - statistical rigour: paired bootstrap on the held-out test set.

1,000 resamples (same row indices for every model => paired). Reports 95% percentile CIs for AUC / PR-AUC and the
paired difference vs the reference model, so 'model X is better' is tested rather than asserted.
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from p2_common import *

X_tr, X_te, y_tr, y_te, _ = load_phase1(); y = y_te.values
m1 = joblib.load(os.path.join(ART, "step1_models.joblib")); lg = joblib.load(os.path.join(ART, "step5_lgbm_model.joblib"))
P = {"Random Forest": m1["rf"].predict_proba(X_te)[:, 1], "Gradient Boosting": m1["gb"].predict_proba(X_te)[:, 1],
     "XGBoost": m1["xgb"].predict_proba(X_te)[:, 1], "LightGBM": lg.predict_proba(X_te)[:, 1],
     "Stack A (Phase 1)": np.load(os.path.join(ART, "p2_probs_stackA.npy")),
     "Stack C (weighted + tuned)": np.load(os.path.join(ART, "p2_probs_stackC.npy"))}
def auc_fast(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
B = 1000; rng = np.random.default_rng(SEED); n = len(y)
auc = {k: np.empty(B) for k in P}; ap = {k: np.empty(B) for k in P}
for b in range(B):
    i = rng.integers(0, n, n); yb = y[i]
    for k, s in P.items(): sb = s[i]; auc[k][b] = auc_fast(yb, sb); ap[k][b] = average_precision_score(yb, sb)
    if b % 200 == 0: print("bootstrap", b, flush=True)
ci = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
res = {"n_bootstrap": B, "models": {k: {"AUC": round(auc_fast(y, P[k]), 4), "AUC_CI95": ci(auc[k]),
        "PR_AUC": round(average_precision_score(y, P[k]), 4), "PR_AUC_CI95": ci(ap[k])} for k in P}, "paired_differences": {}}
for ref, cmp in [("XGBoost", "Stack C (weighted + tuned)"), ("XGBoost", "Stack A (Phase 1)"), ("LightGBM", "Stack C (weighted + tuned)"),
                 ("Stack A (Phase 1)", "Stack C (weighted + tuned)"), ("Random Forest", "XGBoost")]:
    for met, d in (("AUC", auc), ("PR_AUC", ap)):
        diff = d[cmp] - d[ref]
        res["paired_differences"][f"{cmp} minus {ref} [{met}]"] = {"mean_diff": round(float(diff.mean()), 5), "CI95": [round(float(np.percentile(diff, 2.5)), 5), round(float(np.percentile(diff, 97.5)), 5)],
            "CI_excludes_zero": bool(np.percentile(diff, 2.5) > 0 or np.percentile(diff, 97.5) < 0)}
save_json(res, "p2_step3_bootstrap.json"); print(json.dumps(res, indent=1))
fig, ax = plt.subplots(figsize=(7.5, 3.8)); ks = list(P)
for j, k in enumerate(ks):
    lo, hi = res["models"][k]["AUC_CI95"]; ax.plot([lo, hi], [j, j], lw=6, color="#2A7F8E", alpha=.55); ax.plot(res["models"][k]["AUC"], j, "o", color="#1F3A5F")
ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks); ax.invert_yaxis(); ax.set_xlabel("Test AUC with 95% bootstrap CI"); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "p2_bootstrap_auc_ci.png"), dpi=150)
