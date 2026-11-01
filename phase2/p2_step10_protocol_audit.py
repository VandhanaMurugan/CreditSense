"""Protocol audit: how much can evaluation-protocol choices inflate AUC on the base paper's datasets?
The base paper does not specify its split, resampling or selection order (Phase 1 survey). Here the SAME models are scored under four protocols
on German and Taiwan (the two datasets whose reported AUC leak-safe CV does not reproduce):
  P0  leak-safe: everything fitted inside each training fold (baseline, as in step 9)
  P1  SMOTE applied to the whole dataset before cross-validation (synthetic neighbours of test rows end up in training)
  P2  normalisation + Information Gain top-10 selection fitted on the whole dataset before cross-validation
  P3  P1 and P2 together
This shows what such protocols *would* produce; it does not claim the base paper used them.
"""
import sys, time, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from imblearn.over_sampling import SMOTE
sys.argv = [sys.argv[0]]
import importlib.util, os
spec = importlib.util.spec_from_file_location("md", os.path.join(os.path.dirname(os.path.abspath(__file__)), "p2_step9_multidataset.py"))
src = open(spec.origin).read().split("plan = {")[0]; ns = {"__file__": spec.origin}; exec(compile(src, spec.origin, "exec"), ns)
load, prep, SEED, save_json = ns["load"], ns["prep"], ns["SEED"], ns["save_json"]

def models(big): return {"RF": RandomForestClassifier(n_estimators=150 if big else 300, min_samples_leaf=3, random_state=SEED, n_jobs=2),
    "XGB": xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="auc", random_state=SEED, n_jobs=2),
    "LR": LogisticRegression(max_iter=2000)}
res = {}
for name in ("German", "Taiwan"):
    X, y, num, cat = load(name); big = len(y) > 5000; skf = StratifiedKFold(5, shuffle=True, random_state=SEED); out = {}
    Xall = prep(num, cat).fit_transform(X)                                   # whole-data normalisation/encoding (used by P1-P3)
    k = 10 if Xall.shape[1] > 10 else Xall.shape[1]
    sel = np.argsort(-mutual_info_classif(Xall, y, random_state=SEED))[:k]  # whole-data IG selection (P2, P3)
    for proto in ("P0 leak-safe", "P1 SMOTE before CV", "P2 select+scale before CV", "P3 both"):
        sc = {m: [] for m in models(big)}
        for tr, te in skf.split(X, y):
            if proto == "P0 leak-safe":
                pp = prep(num, cat).fit(X.iloc[tr]); Xtr, Xte, ytr, yte = pp.transform(X.iloc[tr]), pp.transform(X.iloc[te]), y[tr], y[te]
            else:
                Xs, ys = Xall, y
                if "P2" in proto or "P3" in proto: Xs = Xs[:, sel]
                if "P1" in proto or "P3" in proto: Xs, ys = SMOTE(random_state=SEED).fit_resample(Xs, ys)   # whole data, then CV on the resampled set
                idx = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xs, ys)); 
                tr, te = idx[len(sc["RF"])]; Xtr, Xte, ytr, yte = Xs[tr], Xs[te], ys[tr], ys[te]
            for m, est in models(big).items(): sc[m].append(roc_auc_score(yte, est.fit(Xtr, ytr).predict_proba(Xte)[:, 1]))
        out[proto] = {m: round(float(np.mean(v)), 4) for m, v in sc.items()}; print(name, proto, out[proto], flush=True)
    res[name] = out
save_json(res, "p2_step10_protocol_audit.json")
