"""N16 (roadmap P1.1): what happens if age is removed from the model?

Why: age is a top-4 driver for 77% of referred or declined applicants (step 14), the main policy declines 47% of under-30s
and auto-approves 25% of defaulters aged 70+ (step 12). A lender's first governance question is "what do we lose without it?".

Design: identical to the released v2.0.0 model (same data split, seeds, tuned settings, Platt step, policy code) with the single
change that the `age` column is dropped. The with-age side is the released bundle itself (its stored calibration and test scores),
so nothing about v2.0.0 is refitted or changed. The tuned settings were chosen with age present and are not re-tuned.

PROMOTION RULE - written before the first run. The no-age model becomes the v2.1.0 CANDIDATE (for the guide to decide on) only if all hold:
  R1  test AUC lower by at most 0.005
  R2  test Brier higher by at most 0.001
  R3  cost per 1,000 under the main policy (r 5, c 0.2, budget 20%, cap 10%) higher by at most 5%
  R4  share of defaulters auto-approved on the test set at most 11%
The bundle is NOT replaced by this script whatever the result.
"""
import os, sys, json, time, numpy as np, pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from p2_common import ROOT, OUT, SEED, WeightedGB, ks_stat, ece, save_json
from creditsense.features import fit_cleaning, to_model_frame
from creditsense.policy import conformal_cap, referral_gain_cut, decide_triage, evaluate, APPROVE, DECLINE
from creditsense.serve import load_bundle, _CANDIDATES, AGE_BANDS

t0 = time.time(); rng = np.random.default_rng(SEED); TARGET = "SeriousDlqin2yrs"
RULE = {"R1_auc_drop_max": 0.005, "R2_brier_rise_max": 0.001, "R3_cost_rise_max": 0.05, "R4_miss_max": 0.11}
b = load_bundle(_CANDIDATES[-1]); FEATS = b["features"]; pol = b["policy_default"]
raw = pd.read_csv(os.path.join(ROOT, "data", "GiveMeSomeCredit_dataset.csv"), index_col=0); raw = raw[raw["age"] > 0].copy()
train_raw, test_raw = train_test_split(raw, test_size=0.25, random_state=42, stratify=raw[TARGET])
X_tr = to_model_frame(train_raw, b["cleaning"], FEATS); X_te = to_model_frame(test_raw, b["cleaning"], FEATS)
y_tr, y_te = train_raw[TARGET], test_raw[TARGET].values
X_fit, X_cal, y_fit, y_cal = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=SEED)
assert np.array_equal(y_cal.values, b["y_cal"]) and np.array_equal(y_te, b["y_test"]), "split differs from the released bundle"
age_cal, age_te = train_raw.loc[X_cal.index, "age"].values, test_raw["age"].values

NOAGE = [f for f in FEATS if f != "age"]
bp = json.load(open(os.path.join(OUT, "p2_tuning_best_params.json"))); spw = (y_fit == 0).sum() / (y_fit == 1).sum()
members = [("rf", RandomForestClassifier(class_weight="balanced_subsample", random_state=SEED, n_jobs=2, **bp["rf"]["params"])),
           ("gb", WeightedGB(**bp["gb"]["params"])),
           ("xgb", xgb.XGBClassifier(scale_pos_weight=spw, eval_metric="auc", random_state=SEED, n_jobs=2, **bp["xgb"]["params"]))]
stack = StackingClassifier(members, final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"), cv=3, n_jobs=1).fit(X_fit[NOAGE], y_fit)
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
rc, rt = stack.predict_proba(X_cal[NOAGE])[:, 1], stack.predict_proba(X_te[NOAGE])[:, 1]
pl = LogisticRegression(C=1e6, max_iter=1000).fit(logit(rc).reshape(-1, 1), y_cal.values)
cal = lambda r: pl.predict_proba(logit(r).reshape(-1, 1))[:, 1]
S = {"with age (v2.0.0)": (b["p_cal"], b["p_test"]), "without age": (cal(rc), cal(rt))}
yc = y_cal.values
print(f"no-age stack fitted on {len(NOAGE)} features ({time.time()-t0:.0f}s)")

def block(pc, pt):
    cap = conformal_cap(pc[yc == 1], pol["alpha"]); cut = referral_gain_cut(pc, pol["r"], pol["c"], cap, pol["rho"])
    dec = decide_triage(pt, pol["r"], pol["c"], cap, cut)
    bands = {}
    for lo, hi, name in AGE_BANDS:
        m = (age_te >= lo) & (age_te < hi); d1 = m & (y_te == 1)
        bands[name] = {"n": int(m.sum()), "default_rate": round(float(y_te[m].mean()), 4), "approve_rate": round(float((dec[m] == APPROVE).mean()), 4),
                       "decline_rate": round(float((dec[m] == DECLINE).mean()), 4),
                       "defaulters_auto_approved": round(float(((dec == APPROVE) & d1).sum() / max(d1.sum(), 1)), 4), "ECE": round(ece(y_te[m], pt[m]), 4)}
    v = []
    idx = np.arange(len(pc))
    for _ in range(200):                                              # cap validity: random re-splits of calibration + test defaulters
        pool = np.concatenate([pc[yc == 1], pt[y_te == 1]]); perm = rng.permutation(len(pool)); n1 = int((yc == 1).sum())
        v.append(float((pool[perm[n1:]] < conformal_cap(pool[perm[:n1]], pol["alpha"])).mean()))
    return {"test": {"AUC": round(float(roc_auc_score(y_te, pt)), 4), "KS": round(ks_stat(y_te, pt), 4), "Brier": round(float(brier_score_loss(y_te, pt)), 4), "ECE": round(ece(y_te, pt), 4)},
            "main_policy": {"cap_pd": round(cap, 4), **evaluate(dec, y_te, pol["r"], pol["c"])}, "age_bands": bands,
            "cap_validity_mean_over_200_resplits": round(float(np.mean(v)), 4)}, dec

out = {"rule": RULE, "policy": {k: pol[k] for k in ("r", "c", "rho", "alpha")}, "features_removed": ["age"], "n_features_without_age": len(NOAGE)}
decs = {}
for name, (pc, pt) in S.items(): out[name], decs[name] = block(pc, pt)
pa, pn = S["with age (v2.0.0)"][1], S["without age"][1]; d_auc, d_br = [], []
for _ in range(1000):
    i = rng.integers(0, len(y_te), len(y_te))
    d_auc.append(roc_auc_score(y_te[i], pn[i]) - roc_auc_score(y_te[i], pa[i])); d_br.append(brier_score_loss(y_te[i], pn[i]) - brier_score_loss(y_te[i], pa[i]))
ci = lambda d: {"mean": round(float(np.mean(d)), 4), "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]}
out["difference_without_minus_with"] = {"AUC": ci(d_auc), "Brier": ci(d_br), "bootstrap": "1,000 paired resamples of the test set",
                                        "decisions_changed_share": round(float((decs["without age"] != decs["with age (v2.0.0)"]).mean()), 4)}
A, N = out["with age (v2.0.0)"], out["without age"]
checks = {"R1 AUC drop": {"value": round(A["test"]["AUC"] - N["test"]["AUC"], 4), "limit": "<= 0.005"},
          "R2 Brier rise": {"value": round(N["test"]["Brier"] - A["test"]["Brier"], 4), "limit": "<= 0.001"},
          "R3 cost rise": {"value": round(N["main_policy"]["cost_per_1000"] / A["main_policy"]["cost_per_1000"] - 1, 4), "limit": "<= 0.05"},
          "R4 defaulters auto-approved": {"value": N["main_policy"]["defaulters_auto_approved"], "limit": "<= 0.11"}}
checks["R1 AUC drop"]["met"] = bool(checks["R1 AUC drop"]["value"] <= 0.005); checks["R2 Brier rise"]["met"] = bool(checks["R2 Brier rise"]["value"] <= 0.001)
checks["R3 cost rise"]["met"] = bool(checks["R3 cost rise"]["value"] <= 0.05); checks["R4 defaulters auto-approved"]["met"] = bool(checks["R4 defaulters auto-approved"]["value"] <= 0.11)
out["promotion_checks"] = checks; out["candidate_for_v2_1_0"] = bool(all(c["met"] for c in checks.values())); out["runtime_seconds"] = round(time.time() - t0)
for n in S: print(n, out[n]["test"], out[n]["main_policy"])
for n in S: print(n, {k: (v["decline_rate"], v["defaulters_auto_approved"]) for k, v in out[n]["age_bands"].items()})
print(out["difference_without_minus_with"]); print(checks); print("candidate:", out["candidate_for_v2_1_0"])
save_json(out, "p2_step19_no_age_model.json")
