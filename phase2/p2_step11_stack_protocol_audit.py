"""Protocol audit extended to the stacked model (Review 2), plus a control: SMOTE applied correctly inside each training fold.
Datasets: German (5x5 CV) and Taiwan (1x5 CV). Same members and settings as step 9.
  P0  leak-safe (everything fitted inside the training fold)
  P1  SMOTE on the whole dataset before CV (the leaky protocol)
  P4  SMOTE inside each training fold only (correct use of SMOTE)
If P4 ~ P0 while P1 >> P0, the inflation comes from the leak, not from SMOTE itself.
"""
import sys, os, numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.over_sampling import SMOTE
sys.argv = [sys.argv[0]]
here = os.path.dirname(os.path.abspath(__file__)); f9 = os.path.join(here, "p2_step9_multidataset.py")
ns = {"__file__": f9}; exec(compile(open(f9).read().split("plan = {")[0], f9, "exec"), ns)
load, prep, members, SEED, save_json = ns["load"], ns["prep"], ns["members"], ns["SEED"], ns["save_json"]

def stack(big):
    m = members(big); base = [(k, m[k]) for k in ("RF", "GB", "XGB")]
    return StackingClassifier(base, final_estimator=LogisticRegression(max_iter=1000), cv=3, n_jobs=1)
def best_tree(big, name): return members(big)["RF" if name == "German" else "XGB"]

res = {}
for name, reps in (("German", 5), ("Taiwan", 1)):
    X, y, num, cat = load(name); big = len(y) > 5000
    Xall = prep(num, cat).fit_transform(X); Xs, ys = SMOTE(random_state=SEED).fit_resample(Xall, y)
    out = {}
    for proto in ("P0 leak-safe", "P1 SMOTE before CV", "P4 SMOTE inside training folds"):
        sc = {"Stack": [], "Best tree": []}
        XX, YY = (Xs, ys) if proto.startswith("P1") else (X, y)
        for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=SEED).split(XX, YY):
            if proto.startswith("P1"):
                Xtr, Xte, ytr, yte = XX[tr], XX[te], YY[tr], YY[te]
            else:
                pp = prep(num, cat).fit(X.iloc[tr]); Xtr, Xte, ytr, yte = pp.transform(X.iloc[tr]), pp.transform(X.iloc[te]), y[tr], y[te]
                if proto.startswith("P4"): Xtr, ytr = SMOTE(random_state=SEED).fit_resample(Xtr, ytr)
            sc["Stack"].append(roc_auc_score(yte, stack(big).fit(Xtr, ytr).predict_proba(Xte)[:, 1]))
            sc["Best tree"].append(roc_auc_score(yte, best_tree(big, name).fit(Xtr, ytr).predict_proba(Xte)[:, 1]))
        out[proto] = {k: {"mean_AUC": round(float(np.mean(v)), 4), "sd": round(float(np.std(v, ddof=1)), 4), "folds": len(v)} for k, v in sc.items()}
        print(name, proto, out[proto], flush=True)
    out["best_tree_model"] = "RF" if name == "German" else "XGB"
    res[name] = out
res["base_paper_reported_AUC"] = {"German": 0.944, "Taiwan": 0.870}
save_json(res, "p2_step11_stack_protocol_audit.json")
