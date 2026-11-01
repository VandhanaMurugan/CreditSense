"""H2 - does model diversity explain stacking gain?  Adds structurally different learners (logistic regression, small MLP)
to the tree stack; reports rank correlation between learners and paired-bootstrap gain of each stack vs the tree-only stack."""
import time, numpy as np
import xgboost as xgb
from scipy.stats import spearmanr, rankdata
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import average_precision_score
from p2_common import *

X_tr, X_te, y_tr, y_te, spw = load_phase1(); y = y_te.values
m1 = joblib.load(os.path.join(ART, "step1_models.joblib")); lg = joblib.load(os.path.join(ART, "step5_lgbm_model.joblib"))
lr = lambda: make_pipeline(QuantileTransformer(output_distribution="normal", random_state=SEED), LogisticRegression(max_iter=2000, class_weight="balanced"))
mlp = lambda: make_pipeline(QuantileTransformer(output_distribution="normal", random_state=SEED), MLPClassifier((32, 16), early_stopping=True, max_iter=200, random_state=SEED))
L = lr().fit(X_tr, y_tr); M = mlp().fit(X_tr, y_tr)
P = {"RF": m1["rf"].predict_proba(X_te)[:, 1], "GB": m1["gb"].predict_proba(X_te)[:, 1], "XGB": m1["xgb"].predict_proba(X_te)[:, 1],
     "LGBM": lg.predict_proba(X_te)[:, 1], "LR": L.predict_proba(X_te)[:, 1], "MLP": M.predict_proba(X_te)[:, 1]}
names = list(P); out = {"single_models": {k: {"AUC": round(roc_auc_score(y, v), 4), "PR_AUC": round(average_precision_score(y, v), 4)} for k, v in P.items()},
    "spearman_rank_corr": {a: {b: round(float(spearmanr(P[a], P[b])[0]), 3) for b in names} for a in names}}
def trees(): return [("rf", RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced_subsample", random_state=SEED, n_jobs=2)), ("gb", WeightedGB()),
    ("xgb", xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, scale_pos_weight=spw, eval_metric="auc", random_state=SEED, n_jobs=2))]
stacks = {"S1 trees only": trees(), "S2 trees + LR": trees() + [("lr", lr())], "S3 trees + LR + MLP": trees() + [("lr", lr()), ("mlp", mlp())]}
SP = {}
for k, mem in stacks.items():
    s = StackingClassifier(mem, final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"), cv=3, n_jobs=1).fit(X_tr, y_tr)
    SP[k] = s.predict_proba(X_te)[:, 1]; out.setdefault("stacks", {})[k] = summarise(y, SP[k]); print(k, out["stacks"][k]["AUC"], flush=True)
def auc_fast(yy, s): r = rankdata(s); n1 = yy.sum(); return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(yy) - n1))
rng = np.random.default_rng(SEED); n = len(y); B = 500; d = {k: {"AUC": [], "PR_AUC": []} for k in SP if k != "S1 trees only"}
for b in range(B):
    i = rng.integers(0, n, n); yb = y[i]; b0 = SP["S1 trees only"][i]; a0 = auc_fast(yb, b0); p0 = average_precision_score(yb, b0)
    for k in d: s = SP[k][i]; d[k]["AUC"].append(auc_fast(yb, s) - a0); d[k]["PR_AUC"].append(average_precision_score(yb, s) - p0)
out["paired_vs_trees_only"] = {k: {m: {"mean": round(float(np.mean(v)), 5), "CI95": [round(float(np.percentile(v, 2.5)), 5), round(float(np.percentile(v, 97.5)), 5)]} for m, v in dd.items()} for k, dd in d.items()}
save_json(out, "p2_step5_diversity.json"); print("done")
