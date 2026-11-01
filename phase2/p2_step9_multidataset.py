"""H1 on the base paper's datasets: German (1,000), Australian (690), Taiwan (30,000).
Leak-safe repeated stratified 5-fold CV (all preprocessing inside the fold). Same fixed hyper-parameters on every dataset (no tuning).
Models: regularised logistic regression (inner-CV C), RF, GB, XGBoost, stack of trees (LR meta), stack of trees + LR.
Stack = out-of-fold (3-fold) base predictions -> logistic-regression meta-learner (same construction as StackingClassifier, base fits shared across variants).
Comparison: paired fold differences with the Nadeau-Bengio corrected resampled t-test (accounts for overlapping training sets).
No class weighting: ranking (AUC) does not need it, and the question here is model comparison, not thresholds.
"""
import sys, time, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from p2_common import ROOT, OUT, SEED, save_json
D = os.path.join(ROOT, "data", "uci") if (os := __import__("os")) else None

def load(name):
    if name == "German":
        d = pd.read_csv(os.path.join(D, "german.data"), sep=r"\s+", header=None)
        num = [1, 4, 7, 10, 12, 15, 17]; cat = [c for c in range(20) if c not in num]
        return d.iloc[:, :20], (d[20] == 2).astype(int).values, num, cat          # class 2 = bad = positive
    if name == "Australian":
        d = pd.read_csv(os.path.join(D, "australian.dat"), sep=r"\s+", header=None)
        num = [1, 2, 6, 9, 12, 13]; cat = [0, 3, 4, 5, 7, 8, 10, 11]
        return d.iloc[:, :14], d[14].astype(int).values, num, cat                  # class 1 = positive (orientation irrelevant for AUC)
    d = pd.read_excel(os.path.join(D, "taiwan_default_of_credit_card_clients.xls"), header=1)
    y = d["default payment next month"].astype(int).values; X = d.drop(columns=["ID", "default payment next month"])
    cat = [c for c in ("SEX", "EDUCATION", "MARRIAGE") if c in X.columns]; num = [c for c in X.columns if c not in cat]
    return X, y, num, cat

def prep(num, cat):
    return ColumnTransformer([("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                              ("c", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cat)], sparse_threshold=0)
def members(big):
    return {"RF": RandomForestClassifier(n_estimators=150 if big else 300, min_samples_leaf=3, random_state=SEED, n_jobs=2),
            "GB": GradientBoostingClassifier(n_estimators=100 if big else 150, max_depth=3, learning_rate=0.05, random_state=SEED),
            "XGB": xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="auc", random_state=SEED, n_jobs=2),
            "LR": LogisticRegressionCV(Cs=8, cv=3, scoring="roc_auc", max_iter=2000)}
def nb_test(diff, n_te, n_tr):
    K = len(diff); m = diff.mean(); s2 = diff.var(ddof=1); se = np.sqrt((1 / K + n_te / n_tr) * s2); t = m / se; p = 2 * stats.t.sf(abs(t), K - 1); h = stats.t.ppf(0.975, K - 1) * se
    return {"mean_diff": round(float(m), 4), "CI95": [round(float(m - h), 4), round(float(m + h), 4)], "p_value": round(float(p), 4)}

plan = {"German": 5, "Australian": 5, "Taiwan": 1}
which = sys.argv[1:] or list(plan)
res = {}
for name in which:
    X, y, num, cat = load(name); big = len(y) > 5000; t0 = time.time()
    rs = RepeatedStratifiedKFold(n_splits=5, n_repeats=plan[name], random_state=SEED); rows = []
    for k, (tr, te) in enumerate(rs.split(X, y)):
        pp = prep(num, cat).fit(X.iloc[tr]); Xtr = pp.transform(X.iloc[tr]); Xte = pp.transform(X.iloc[te]); ytr, yte = y[tr], y[te]
        base_oof, base_te = {}, {}
        for m, est in members(big).items():
            base_oof[m] = cross_val_predict(est, Xtr, ytr, cv=StratifiedKFold(3, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
            base_te[m] = est.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        for lab, keys in (("Stack_trees", ["RF", "GB", "XGB"]), ("Stack_trees_LR", ["RF", "GB", "XGB", "LR"])):
            meta = LogisticRegression(max_iter=1000).fit(np.c_[[base_oof[q] for q in keys]].T, ytr); base_te[lab] = meta.predict_proba(np.c_[[base_te[q] for q in keys]].T)[:, 1]
        rows.append({m: (roc_auc_score(yte, p), average_precision_score(yte, p)) for m, p in base_te.items()} | {"_n": (len(tr), len(te))})
        print(name, "fold", k + 1, f"{time.time()-t0:.0f}s", flush=True)
    ntr, nte = rows[0]["_n"]; models = [m for m in rows[0] if m != "_n"]
    A = {m: np.array([r[m][0] for r in rows]) for m in models}; PR = {m: np.array([r[m][1] for r in rows]) for m in models}
    best_tree = max(("RF", "GB", "XGB"), key=lambda m: A[m].mean())
    out = {"n": int(len(y)), "positive_rate": round(float(y.mean()), 4), "folds": len(rows), "best_single_tree": best_tree,
           "mean_AUC": {m: round(float(A[m].mean()), 4) for m in models}, "sd_AUC_over_folds": {m: round(float(A[m].std(ddof=1)), 4) for m in models}, "mean_PR_AUC": {m: round(float(PR[m].mean()), 4) for m in models},
           "tests_AUC_nadeau_bengio": {f"Stack_trees - {best_tree}": nb_test(A["Stack_trees"] - A[best_tree], nte, ntr), "Stack_trees - LR": nb_test(A["Stack_trees"] - A["LR"], nte, ntr),
                                       f"{best_tree} - LR": nb_test(A[best_tree] - A["LR"], nte, ntr), "Stack_trees_LR - Stack_trees": nb_test(A["Stack_trees_LR"] - A["Stack_trees"], nte, ntr),
                                       "Stack_trees_LR - LR": nb_test(A["Stack_trees_LR"] - A["LR"], nte, ntr)}}
    res[name] = out; print(name, json.dumps(out), flush=True) if (json := __import__("json")) else None
    save_json(res, f"p2_step9_multidataset_{'_'.join(which)}.json")
