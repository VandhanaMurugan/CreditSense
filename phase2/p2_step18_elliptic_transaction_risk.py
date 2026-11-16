"""N15: does the CreditSense decision layer carry over to another risk domain? Crypto transaction screening.

Data: the Elliptic data set (public): 203,769 Bitcoin transactions in 49 time steps, about 46,000 of them labelled
licit or illicit, 165 anonymised features per transaction.
Mapping from lending:  applicant -> transaction;  default -> illicit;  approve / refer / decline -> allow / review / block;
"share of defaulters auto-approved" -> "share of illicit transactions allowed without review".
The decision code is creditsense/policy.py, unchanged.

Design (split by time step, leak-safe):
  train      steps  1-29    fit the models
  calibrate  steps 30-34    Platt map, cap, review cut
  test       steps 35-49    never used for fitting
Same fixed, untuned model settings as Phase I.

ASSUMPTIONS (policy inputs, not measurements): missing an illicit transaction costs r = 20 times a wrongly blocked licit one;
a review costs c = 0.2; analysts can review rho = 5% of transactions and are right; cap alpha = 10%.

PASS MARKS - written before the first run:
  Q1 calibration  frozen Platt map: ECE on steps 35-49 <= 0.02
  Q2 guarantee    cap at alpha = 10%: share of illicit transactions allowed without review, steps 35-42, <= 12%
  Q3 stress       steps 43-49 follow a change in illicit activity described for this data set (a dark-market closure):
                  the guarantee is EXPECTED to fail there; the test is whether monitoring raises a flag (score PSI >= 0.10
                  or |predicted - observed illicit rate| > 0.02) in the steps where the miss rate exceeds 12%
  Q4 stacking     H1 expects no meaningful gain: stack - best single model AUC gain < 0.005
Known limits: features are anonymised; only about 23% of transactions are labelled, and labels are not a random sample;
costs are assumptions.
"""
import os, sys, time, numpy as np, pandas as pd
import xgboost as xgb, lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from p2_common import ROOT, SEED, WeightedGB, ks_stat, ece, save_json
from creditsense.policy import conformal_cap, referral_gain_cut, decide_triage, decide_two_way, evaluate, APPROVE

D = os.path.join(ROOT, "data", "elliptic")
if not os.path.exists(os.path.join(D, "elliptic_txs_features.csv")): sys.exit("Elliptic files not found - see data/elliptic/README.md")
t0 = time.time(); rng = np.random.default_rng(SEED)
R_COST, C_COST, RHO, ALPHA = 20.0, 0.2, 0.05, 0.10
PASS_MARKS = {"Q1_ece_max": 0.02, "Q2_miss_max_steps_35_42": 0.12, "Q3_flag_psi": 0.10, "Q3_flag_rate_gap": 0.02, "Q4_stack_gain_max": 0.005}

f = pd.read_csv(os.path.join(D, "elliptic_txs_features.csv"), header=None)
f.columns = ["txId", "step"] + [f"f{i}" for i in range(f.shape[1] - 2)]
c = pd.read_csv(os.path.join(D, "elliptic_txs_classes.csv"), dtype=str); c["txId"] = c["txId"].astype("int64")
d = f.merge(c, on="txId"); n_all = len(d)
d = d[d["class"].isin(["1", "2"])].reset_index(drop=True)
y = (d["class"] == "1").astype(int).values; step = d["step"].values
X = d[[col for col in d.columns if col.startswith("f")]].astype("float32")
tr, ca, te = step <= 29, (step >= 30) & (step <= 34), step >= 35
print(f"{n_all:,} transactions, {len(d):,} labelled ({y.mean():.3f} illicit); train {tr.sum():,} | calibrate {ca.sum():,} | test {te.sum():,}  ({time.time()-t0:.0f}s)")
spw = float((y[tr] == 0).sum() / (y[tr] == 1).sum())

def members():
    return [("rf", RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced_subsample", n_jobs=2, random_state=SEED)),
            ("gb", WeightedGB(n_estimators=200, max_depth=4, learning_rate=0.05)),
            ("xgb", xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, scale_pos_weight=spw, n_jobs=2, random_state=SEED, eval_metric="logloss"))]
def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p)).reshape(-1, 1)
def fit_platt(s, yy): m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(s), yy); return lambda q: m.predict_proba(logit(q))[:, 1]
def scores(yy, p): return {"AUC": round(float(roc_auc_score(yy, p)), 4), "PR_AUC": round(float(average_precision_score(yy, p)), 4), "KS": round(ks_stat(yy, p), 4),
                           "Brier": round(float(brier_score_loss(yy, p)), 4), "ECE": round(ece(yy, p), 4), "mean_predicted": round(float(p.mean()), 4), "observed": round(float(yy.mean()), 4)}

models = {"LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
          "RF": members()[0][1], "GB": members()[1][1], "XGB": members()[2][1],
          "LightGBM": lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, scale_pos_weight=spw, n_jobs=2, random_state=SEED, verbose=-1),
          "Stack": StackingClassifier(members(), final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"), cv=3, n_jobs=1)}
P = {}
for name, m in models.items():
    m.fit(X[tr], y[tr]); s = m.predict_proba(X)[:, 1]; P[name] = fit_platt(s[ca], y[ca])(s)
    print(f"  {name:9s} test AUC {roc_auc_score(y[te], P[name][te]):.4f}  ({time.time()-t0:.0f}s)")

early, late = (step >= 35) & (step <= 42), step >= 43
res = {"data": {"source": "Elliptic data set (public)", "transactions": int(n_all), "labelled": int(len(d)), "illicit_share_of_labelled": round(float(y.mean()), 4),
                "train_steps_1_29": int(tr.sum()), "calibrate_steps_30_34": int(ca.sum()), "test_steps_35_49": int(te.sum()), "n_features": int(X.shape[1])},
       "assumptions": {"r": R_COST, "c": C_COST, "rho": RHO, "alpha": ALPHA, "analyst": "right every time"}, "pass_marks": PASS_MARKS,
       "metrics": {n: {"test_35_49": scores(y[te], p[te]), "test_35_42": scores(y[early], p[early]), "test_43_49": scores(y[late], p[late])} for n, p in P.items()}}
singles = ["LR", "RF", "GB", "XGB", "LightGBM"]; best = max(singles, key=lambda n: res["metrics"][n]["test_35_49"]["AUC"])
it = np.where(te)[0]; dd = []
for _ in range(500):
    i = rng.choice(it, size=len(it), replace=True); dd.append(roc_auc_score(y[i], P["Stack"][i]) - roc_auc_score(y[i], P[best][i]))
res["stacking_H1"] = {"best_single": best, "stack_minus_best_single_auc": {"mean": round(float(np.mean(dd)), 4), "ci95": [round(float(np.percentile(dd, 2.5)), 4), round(float(np.percentile(dd, 97.5)), 4)]}}

p = P["Stack"]
cap = conformal_cap(p[ca][y[ca] == 1], ALPHA); cut = referral_gain_cut(p[ca], R_COST, C_COST, cap, RHO); cut0 = referral_gain_cut(p[ca], R_COST, C_COST, np.inf, RHO)
rules = {"Two-way cost threshold": decide_two_way(p, R_COST), "Same budget, no guarantee": decide_triage(p, R_COST, C_COST, np.inf, cut0),
         "CreditSense alpha 10%": decide_triage(p, R_COST, C_COST, cap, cut)}
res["policy"] = {"cap_probability": round(cap, 5), **{n: {"steps_35_42": evaluate(dec[early], y[early], R_COST, C_COST), "steps_43_49": evaluate(dec[late], y[late], R_COST, C_COST),
                                                          "steps_35_49": evaluate(dec[te], y[te], R_COST, C_COST)} for n, dec in rules.items()}}
in_time = []
ci = np.where(ca)[0]
for _ in range(200):
    perm = rng.permutation(ci); a, b = perm[: len(perm) // 2], perm[len(perm) // 2:]
    if (y[a] == 1).sum() < 20 or (y[b] == 1).sum() < 20: continue
    in_time.append(float((p[b][y[b] == 1] < conformal_cap(p[a][y[a] == 1], ALPHA)).mean()))
edges = np.unique(np.quantile(p[ca], np.linspace(0, 1, 11))); edges[0], edges[-1] = -np.inf, np.inf
ref = np.clip(np.histogram(p[ca], bins=edges)[0] / ca.sum(), 1e-4, None)
def psi(v): cur = np.clip(np.histogram(v, bins=edges)[0] / max(len(v), 1), 1e-4, None); return float(np.sum((cur - ref) * np.log(cur / ref)))
dec = rules["CreditSense alpha 10%"]; by_step = {}
for s in range(35, 50):
    m = step == s; ill = m & (y == 1)
    row = {"n": int(m.sum()), "illicit": int(ill.sum()), "missed_share": round(float(((dec == APPROVE) & ill).sum() / max(ill.sum(), 1)), 4),
           "score_psi": round(psi(p[m]), 4), "predicted_illicit_rate": round(float(p[m].mean()), 4), "observed_illicit_rate": round(float(y[m].mean()), 4)}
    row["monitoring_flag"] = bool(row["score_psi"] >= 0.10 or abs(row["predicted_illicit_rate"] - row["observed_illicit_rate"]) > 0.02)
    by_step[s] = row
miss = lambda mask: round(float(((dec == APPROVE) & mask & (y == 1)).sum() / max((mask & (y == 1)).sum(), 1)), 4)
breach = [s for s, r in by_step.items() if s >= 43 and r["missed_share"] > 0.12]
res["guarantee"] = {"in_time_mean_over_random_halves_of_steps_30_34": round(float(np.mean(in_time)), 4), "steps_35_42": miss(early), "steps_43_49": miss(late), "by_step": by_step,
                    "breach_steps_after_42": breach, "breach_steps_flagged_by_monitoring": [s for s in breach if by_step[s]["monitoring_flag"]]}
v = {"Q1 calibration": {"value": res["metrics"]["Stack"]["test_35_49"]["ECE"], "pass_mark": "<= 0.02"},
     "Q2 guarantee": {"value": res["guarantee"]["steps_35_42"], "pass_mark": "<= 0.12 (steps 35-42)"},
     "Q3 stress": {"value": {"miss_share_steps_43_49": res["guarantee"]["steps_43_49"], "breach_steps": len(breach), "flagged": len(res["guarantee"]["breach_steps_flagged_by_monitoring"])},
                   "pass_mark": "every breach step after 42 is flagged by monitoring"},
     "Q4 stacking": {"value": res["stacking_H1"]["stack_minus_best_single_auc"]["mean"], "pass_mark": "< 0.005"}}
v["Q1 calibration"]["met"] = bool(v["Q1 calibration"]["value"] <= 0.02); v["Q2 guarantee"]["met"] = bool(v["Q2 guarantee"]["value"] <= 0.12)
v["Q3 stress"]["met"] = bool(len(breach) == len(res["guarantee"]["breach_steps_flagged_by_monitoring"])); v["Q4 stacking"]["met"] = bool(v["Q4 stacking"]["value"] < 0.005)
res["verdicts"] = v; res["runtime_seconds"] = round(time.time() - t0)
for k, r in v.items(): print(f"{k:16s} {'MET' if r['met'] else 'NOT MET':8s} {r['value']}  ({r['pass_mark']})")
for n in rules: print(f"  {n:28s} 35-42: {res['policy'][n]['steps_35_42']}\n{'':31s}43-49: {res['policy'][n]['steps_43_49']}")
print({s: (r['illicit'], r['missed_share'], r['score_psi'], r['monitoring_flag']) for s, r in by_step.items()})
save_json(res, "p2_step18_elliptic_transaction_risk.json"); print(f"saved  ({time.time()-t0:.0f}s)")
