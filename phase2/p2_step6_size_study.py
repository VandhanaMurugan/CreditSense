"""H1 (proxy on one dataset) - does the stacking gain depend on training-set size?
Random subsamples of the GMSC training split (sizes 1k-30k, several repeats); fixed test set. Stack (trees, cost-weighted GB) vs best single
learner (XGBoost) and vs logistic regression. Paired difference per repeat. Multi-dataset version (German, Australian, Taiwan) is the next module."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from p2_common import *

X_tr, X_te, y_tr, y_te, _ = load_phase1()
plan = [(1000, 20), (3000, 12), (10000, 6), (30000, 3)]
res = {}
for n, reps in plan:
    rows = []
    for r in range(reps):
        Xs, _, ys, _ = train_test_split(X_tr, y_tr, train_size=n, stratify=y_tr, random_state=1000 + r)
        w = (ys == 0).sum() / (ys == 1).sum()
        mk_x = lambda: xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, scale_pos_weight=w, eval_metric="auc", random_state=SEED, n_jobs=2)
        mk_r = lambda: RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced_subsample", random_state=SEED, n_jobs=2)
        st = StackingClassifier([("rf", mk_r()), ("gb", WeightedGB(n_estimators=100)), ("xgb", mk_x())], final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"), cv=3, n_jobs=1).fit(Xs, ys)
        xg = mk_x().fit(Xs, ys)
        lrm = make_pipeline(QuantileTransformer(n_quantiles=min(500, n), output_distribution="normal", random_state=SEED), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(Xs, ys)
        p = {"stack": st.predict_proba(X_te)[:, 1], "xgb": xg.predict_proba(X_te)[:, 1], "lr": lrm.predict_proba(X_te)[:, 1]}
        rows.append({k: (roc_auc_score(y_te, v), average_precision_score(y_te, v)) for k, v in p.items()})
    a = lambda k, j: np.array([r[k][j] for r in rows])
    res[n] = {"repeats": reps, "AUC": {k: [round(float(a(k, 0).mean()), 4), round(float(a(k, 0).std()), 4)] for k in ("stack", "xgb", "lr")},
              "stack_minus_xgb_AUC": [round(float((a("stack", 0) - a("xgb", 0)).mean()), 4), round(float((a("stack", 0) - a("xgb", 0)).std()), 4)],
              "stack_minus_lr_AUC": [round(float((a("stack", 0) - a("lr", 0)).mean()), 4), round(float((a("stack", 0) - a("lr", 0)).std()), 4)],
              "stack_minus_xgb_PR": [round(float((a("stack", 1) - a("xgb", 1)).mean()), 4), round(float((a("stack", 1) - a("xgb", 1)).std()), 4)],
              "n_defaulters_in_train_mean": round(float(n * y_tr.mean()), 1)}
    print(n, json.dumps(res[n]), flush=True)
save_json(res, "p2_step6_size_study.json")
ns = list(res); fig, ax = plt.subplots(figsize=(7.5, 4))
for key, lab, c in [("stack_minus_xgb_AUC", "Stack − XGBoost", "#2A7F8E"), ("stack_minus_lr_AUC", "Stack − Logistic regression", "#C8641E")]:
    m = [res[n][key][0] for n in ns]; s = [res[n][key][1] for n in ns]
    ax.errorbar(range(len(ns)), m, yerr=s, marker="o", capsize=4, label=lab, color=c)
ax.axhline(0, color="k", lw=1, ls="--"); ax.set_xticks(range(len(ns))); ax.set_xticklabels([f"{n:,}" for n in ns]); ax.set_xlabel("Training rows"); ax.set_ylabel("Test AUC difference (mean ± sd over repeats)")
ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False); plt.tight_layout(); plt.savefig(os.path.join(OUT, "p2_size_study.png"), dpi=150)
