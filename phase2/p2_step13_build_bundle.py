"""Build the deployable CreditSense bundle (artifacts/creditsense_bundle.joblib).

Refits exactly the step-2 model (same split, seeds and tuned settings), checks it reproduces the step-2 scores, and packs
everything the scoring service needs: cleaning parameters, features, stack, Platt map, calibration scores for the policy
(caps and capacity cuts), input ranges for the out-of-range guard, PSI reference bins, SHAP background, risk tiers and
model-card metrics.
"""
import sys, os, json, datetime, numpy as np, pandas as pd, joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from creditsense import __version__
from creditsense.features import fit_cleaning, to_model_frame, RAW_INPUTS
from creditsense.policy import conformal_cap, decide_triage, referral_gain_cut, evaluate
from p2_common import ROOT, ART, OUT, SEED, WeightedGB, load_phase1, summarise, save_json

TARGET = "SeriousDlqin2yrs"
raw = pd.read_csv(os.path.join(ROOT, "data", "GiveMeSomeCredit_dataset.csv"), index_col=0); raw = raw[raw["age"] > 0].copy()
train_raw, test_raw = train_test_split(raw, test_size=0.25, random_state=42, stratify=raw[TARGET])
cleaning = fit_cleaning(train_raw)
X_tr, X_te, y_tr, y_te, _ = load_phase1(); FEATS = list(X_tr.columns)
assert np.allclose(to_model_frame(test_raw, cleaning, FEATS).values, X_te.values), "feature rebuild differs from step 1"

bp = json.load(open(os.path.join(OUT, "p2_tuning_best_params.json")))
X_fit, X_cal, y_fit, y_cal = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=SEED)
spw_fit = (y_fit == 0).sum() / (y_fit == 1).sum()
members = [("rf", RandomForestClassifier(class_weight="balanced_subsample", random_state=SEED, n_jobs=2, **bp["rf"]["params"])),
           ("gb", WeightedGB(**bp["gb"]["params"])),
           ("xgb", xgb.XGBClassifier(scale_pos_weight=spw_fit, eval_metric="auc", random_state=SEED, n_jobs=2, **bp["xgb"]["params"]))]
stack = StackingClassifier(members, final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"), cv=3, n_jobs=1).fit(X_fit, y_fit)
raw_cal = stack.predict_proba(X_cal)[:, 1]; raw_te = stack.predict_proba(X_te)[:, 1]
step2_te = np.load(os.path.join(ART, "p2_raw_te.npy"))
max_diff = float(np.abs(raw_te - step2_te).max()); print("max |bundle - step2| raw score:", max_diff)

logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
pl = LogisticRegression(C=1e6, max_iter=1000).fit(logit(raw_cal).reshape(-1, 1), y_cal.values)
A, B = float(pl.coef_[0, 0]), float(pl.intercept_[0])
cal = lambda r: 1 / (1 + np.exp(-(A * logit(r) + B)))
p_cal, p_te = cal(raw_cal), cal(raw_te); yc, yt = y_cal.values, y_te.values

TIERS = [0.01, 0.03, 0.08, 0.20]              # calibrated PD cut-offs -> tiers 1..5
tier_te = np.digitize(p_te, TIERS) + 1
tiers = {int(t): {"pd_range": [([0] + TIERS)[t - 1], (TIERS + [1])[t - 1]], "share_of_applicants": round(float((tier_te == t).mean()), 4),
                  "observed_default_rate": round(float(yt[tier_te == t].mean()), 4), "mean_predicted_pd": round(float(p_te[tier_te == t].mean()), 4)}
         for t in range(1, 6)}

fit_raw = train_raw.loc[X_fit.index]
ranges = {c: {"min": float(fit_raw[c].min()), "p001": float(fit_raw[c].quantile(0.001)), "p999": float(fit_raw[c].quantile(0.999)),
              "max": float(fit_raw[c].max())} for c in RAW_INPUTS}
def edges(v, k=10):
    e = np.unique(np.quantile(v, np.linspace(0, 1, k + 1))); e[0], e[-1] = -np.inf, np.inf; return e.tolist()
psi_ref = {"score": edges(p_cal), **{f: edges(X_cal[f].values) for f in FEATS}}
psi_ref_counts = {k: np.histogram(p_cal if k == "score" else X_cal[k].values, bins=v)[0].tolist() for k, v in psi_ref.items()}

bg = X_fit.sample(200, random_state=SEED).iloc[:100]   # 100 rows: shap subsamples larger backgrounds, which breaks determinism
policy_default = {"r": 5.0, "c": 0.2, "rho": 0.20, "alpha": 0.10, "delta": None}
cap = conformal_cap(p_cal[yc == 1], 0.10); cut = referral_gain_cut(p_cal, 5, 0.2, cap, 0.20)
card = {"test": summarise(yt, p_te), "test_n": int(len(yt)), "test_defaulters": int(yt.sum()),
        "main_policy_test": evaluate(decide_triage(p_te, 5, 0.2, cap, cut), yt, 5, 0.2)}
bundle = {"version": __version__, "built": datetime.date.today().isoformat(), "features": FEATS, "raw_inputs": RAW_INPUTS,
          "cleaning": cleaning, "stack": stack, "platt": (A, B), "p_cal": p_cal, "y_cal": yc, "tiers": TIERS, "tier_table": tiers,
          "ranges": ranges, "psi_edges": psi_ref, "psi_ref_counts": psi_ref_counts, "shap_background": bg,
          "policy_default": policy_default, "model_card": card, "p_test": p_te, "y_test": yt,
          "data": {"source": "Kaggle Give Me Some Credit", "fit_rows": int(len(X_fit)), "calibration_rows": int(len(X_cal)), "test_rows": int(len(X_te))},
          "reproduces_step2_max_abs_diff": max_diff}
joblib.dump(bundle, os.path.join(ART, "creditsense_bundle.joblib"), compress=3)
np.save(os.path.join(ART, "bundle_p_te.npy"), p_te)
save_json({"version": __version__, "platt_A_B": [A, B], "reproduces_step2_max_abs_diff": max_diff, "tiers": tiers, "model_card": card,
           "policy_default": policy_default, "cap_alpha_0.10": cap, "capacity_cut": cut}, "p2_step13_bundle_summary.json")
print(json.dumps(tiers, indent=1)); print(json.dumps(card["main_policy_test"]))
print("bundle MB:", round(os.path.getsize(os.path.join(ART, "creditsense_bundle.joblib")) / 1e6, 1))
