"""Phase 2 / Module 2 - probability calibration.

Class weighting makes raw scores over-estimate default probability (they rank well but are not PDs).
Protocol (leak-safe): split TRAIN 80/20 -> fit the tuned, weighted stack on 80% -> learn the calibration map on
the 20% hold-out -> report on the untouched TEST set. Method rule (fixed before looking at test): Platt unless isotonic
improves 5-fold Brier on the calibration hold-out by more than 0.001.
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import calibration_curve
from p2_common import *

X_tr, X_te, y_tr, y_te, spw = load_phase1()
bp = json.load(open(os.path.join(OUT, "p2_tuning_best_params.json")))
X_fit, X_cal, y_fit, y_cal = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=SEED)
spw_fit = (y_fit == 0).sum() / (y_fit == 1).sum()
members = [("rf", RandomForestClassifier(class_weight="balanced_subsample", random_state=SEED, n_jobs=2, **bp["rf"]["params"])),
           ("gb", WeightedGB(**bp["gb"]["params"])),
           ("xgb", xgb.XGBClassifier(scale_pos_weight=spw_fit, eval_metric="auc", random_state=SEED, n_jobs=2, **bp["xgb"]["params"]))]
stack = StackingClassifier(members, final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"), cv=3, n_jobs=1)
stack.fit(X_fit, y_fit)
raw_cal = stack.predict_proba(X_cal)[:, 1]; raw_te = stack.predict_proba(X_te)[:, 1]
yc, yt = y_cal.values, y_te.values

def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p)).reshape(-1, 1)
def fit_platt(s, y): m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(s), y); return lambda q: m.predict_proba(logit(q))[:, 1]
def fit_iso(s, y): m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s, y); return lambda q: m.predict(q)

cv_brier = {}
for name, fitter in [("platt", fit_platt), ("isotonic", fit_iso)]:
    sc = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(raw_cal, yc):
        sc.append(brier_score_loss(yc[b], fitter(raw_cal[a], yc[a])(raw_cal[b])))
    cv_brier[name] = round(float(np.mean(sc)), 5)
# Pre-stated rule: prefer Platt (smooth, preserves ranking) unless isotonic improves Brier by more than 0.001
chosen = "isotonic" if cv_brier["platt"] - cv_brier["isotonic"] > 0.001 else "platt"
print("5-fold Brier on calibration hold-out:", cv_brier, "-> chosen:", chosen, flush=True)
fns = {"platt": fit_platt(raw_cal, yc), "isotonic": fit_iso(raw_cal, yc)}
cal_cal, cal_te = fns[chosen](raw_cal), fns[chosen](raw_te)

out = {"split": {"fit_rows": int(len(y_fit)), "calibration_rows": int(len(y_cal)), "test_rows": int(len(yt))},
       "calibration_method_cv_brier": cv_brier, "chosen_method": chosen,
       "base_rate_test": round(float(yt.mean()), 4),
       "mean_predicted_pd_raw": round(float(raw_te.mean()), 4), "mean_predicted_pd_calibrated": round(float(cal_te.mean()), 4),
       "test_raw": summarise(yt, raw_te), "test_calibrated": summarise(yt, cal_te)}
for k in ("platt", "isotonic"): out[f"test_{k}"] = summarise(yt, fns[k](raw_te))
save_json(out, "p2_step2_calibration.json"); print(json.dumps(out, indent=1), flush=True)
np.save(os.path.join(ART, "p2_cal_probs.npy"), np.c_[cal_cal]); np.save(os.path.join(ART, "p2_cal_y.npy"), yc)
np.save(os.path.join(ART, "p2_raw_cal.npy"), raw_cal); np.save(os.path.join(ART, "p2_raw_te.npy"), raw_te); np.save(os.path.join(ART, "p2_calibrated_te.npy"), cal_te)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for p, lab, c in [(raw_te, "Raw stack score", "#C8641E"), (cal_te, f"Calibrated ({chosen})", "#2A7F8E")]:
    f, m = calibration_curve(yt, p, n_bins=10, strategy="quantile"); ax[0].plot(m, f, "o-", label=lab, color=c)
mx = 0.6; ax[0].plot([0, mx], [0, mx], "k--", lw=1, label="Perfect calibration")
ax[0].set_xlabel("Mean predicted probability"); ax[0].set_ylabel("Observed default rate"); ax[0].set_title("Reliability diagram (test)"); ax[0].legend(frameon=False)
ax[1].hist(raw_te, bins=40, alpha=.7, color="#C8641E", label="Raw"); ax[1].hist(cal_te, bins=40, alpha=.7, color="#2A7F8E", label="Calibrated")
ax[1].set_yscale("log"); ax[1].set_xlabel("Predicted probability"); ax[1].set_title("Score distribution (test)"); ax[1].legend(frameon=False)
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "p2_calibration.png"), dpi=150); print("saved")
