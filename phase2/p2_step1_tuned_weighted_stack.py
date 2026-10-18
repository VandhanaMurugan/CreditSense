"""Phase 2 / Module 1 - fair, tuned stack.

Three controlled variants (ablation) so each change is attributable:
  A  Phase 1 stack            (GB member unweighted, default hyper-parameters)
  B  + consistent weighting   (GB member cost-weighted inside the stack)
  C  + tuned base learners    (B with hyper-parameters from stratified-CV random search)
Tuning uses a stratified 40,000-row subsample of the TRAIN split only (compute budget); final fits use all train rows.
The test set is never touched by tuning.
"""
import time, numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from p2_common import *

X_tr, X_te, y_tr, y_te, spw = load_phase1()
cv3 = StratifiedKFold(3, shuffle=True, random_state=SEED)
Xs, _, ys, _ = train_test_split(X_tr, y_tr, train_size=40000, stratify=y_tr, random_state=SEED)

def search(est, space, n_iter, name):
    t = time.time()
    rs = RandomizedSearchCV(est, space, n_iter=n_iter, scoring="average_precision", cv=cv3, random_state=SEED, n_jobs=1)
    rs.fit(Xs, ys)
    print(f"{name}: best PR-AUC(cv)={rs.best_score_:.4f}  params={rs.best_params_}  ({time.time()-t:.0f}s)", flush=True)
    return rs.best_params_, round(rs.best_score_, 4)

best = {}
best["xgb"] = search(xgb.XGBClassifier(scale_pos_weight=spw, eval_metric="auc", random_state=SEED, n_jobs=2),
    {"n_estimators": [200, 300, 400], "max_depth": [3, 4, 5, 6], "learning_rate": [0.02, 0.03, 0.05, 0.08],
     "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.7, 0.85, 1.0], "min_child_weight": [1, 5, 10]}, 10, "XGB")
best["rf"] = search(RandomForestClassifier(class_weight="balanced_subsample", random_state=SEED, n_jobs=2),
    {"n_estimators": [200, 300], "max_depth": [6, 8, 10, 12], "min_samples_leaf": [5, 20, 50, 100]}, 6, "RF")
best["gb"] = search(WeightedGB(),
    {"n_estimators": [100, 150, 200], "max_depth": [3, 4, 5], "learning_rate": [0.03, 0.05, 0.08],
     "subsample": [0.7, 0.85, 1.0], "min_samples_leaf": [10, 50, 100]}, 4, "GB")
save_json({k: {"params": v[0], "cv_pr_auc_subsample": v[1]} for k, v in best.items()}, "p2_tuning_best_params.json")

def build_stack(variant):
    if variant == "A":
        members = [("rf", RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced_subsample", random_state=SEED, n_jobs=2)),
                   ("gb", GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=SEED)),
                   ("xgb", xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, scale_pos_weight=spw, eval_metric="auc", random_state=SEED, n_jobs=2))]
    elif variant == "B":
        members = [("rf", RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced_subsample", random_state=SEED, n_jobs=2)),
                   ("gb", WeightedGB()),
                   ("xgb", xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, scale_pos_weight=spw, eval_metric="auc", random_state=SEED, n_jobs=2))]
    else:
        members = [("rf", RandomForestClassifier(class_weight="balanced_subsample", random_state=SEED, n_jobs=2, **best["rf"][0])),
                   ("gb", WeightedGB(**best["gb"][0])),
                   ("xgb", xgb.XGBClassifier(scale_pos_weight=spw, eval_metric="auc", random_state=SEED, n_jobs=2, **best["xgb"][0]))]
    return StackingClassifier(members, final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"), cv=3, n_jobs=1)

res, probs = {}, {}
for v, label in [("A", "A_phase1_stack"), ("B", "B_weighted_gb_in_stack"), ("C", "C_weighted_and_tuned")]:
    t = time.time(); m = build_stack(v); m.fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]; probs[label] = p
    res[label] = summarise(y_te.values, p); print(label, res[label], f"({time.time()-t:.0f}s)", flush=True)
    if v == "C": joblib.dump(m, os.path.join(ART, "p2_stack_C.joblib"))
save_json(res, "p2_step1_stack_ablation.json")
np.save(os.path.join(ART, "p2_probs_stackA.npy"), probs["A_phase1_stack"])
np.save(os.path.join(ART, "p2_probs_stackB.npy"), probs["B_weighted_gb_in_stack"])
np.save(os.path.join(ART, "p2_probs_stackC.npy"), probs["C_weighted_and_tuned"])
