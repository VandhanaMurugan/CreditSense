"""Existing-system baseline for the Review 3 comparison: a weight-of-evidence (WoE) logistic scorecard, the industry standard.
Same split as everything else; counts binned 0/1/2/3/4+, continuous attributes in 8 quantile bins; bins and WoE fitted on the 80% fit part of the training split, calibration checked on the
20% hold-out, results on the untouched test set. Inputs: the 10 raw GMSC attributes after the Phase I cleaning (the scorecard
does its own binning, so it may use income, debt ratio and dependents, which IG selection dropped from the stack).
Points scaling: 600 points at odds 50:1 (good:bad), 20 points to double the odds (common convention).
"""
import sys, os, json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from creditsense.features import fit_cleaning, apply_cleaning
from creditsense.policy import conformal_cap, referral_gain_cut, decide_triage, evaluate, APPROVE, DECLINE
from p2_common import ROOT, OUT, SEED, summarise, save_json

T = "SeriousDlqin2yrs"
raw = pd.read_csv(os.path.join(ROOT, "data", "GiveMeSomeCredit_dataset.csv"), index_col=0); raw = raw[raw["age"] > 0]
tr, te = train_test_split(raw, test_size=0.25, random_state=42, stratify=raw[T])
cl = fit_cleaning(tr); tr, te = apply_cleaning(tr, cl), apply_cleaning(te, cl)
fit, cal = train_test_split(tr, test_size=0.2, stratify=tr[T], random_state=SEED)
cols = [c for c in raw.columns if c != T]

COUNTS = ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate",
          "NumberRealEstateLoansOrLines", "NumberOfDependents"]
def bins_for(x, c, k=8):
    if c in COUNTS: return np.array([-np.inf, 0.5, 1.5, 2.5, 3.5, np.inf])          # 0, 1, 2, 3, 4+ (counts need explicit bins)
    e = np.unique(np.quantile(x, np.linspace(0, 1, k + 1))); e[0], e[-1] = -np.inf, np.inf; return e
woe_tab = {}
for c in cols:
    e = bins_for(fit[c].values, c); b = np.digitize(fit[c].values, e[1:-1])
    good = pd.Series(fit[T].values == 0).groupby(b).sum() + 0.5; bad = pd.Series(fit[T].values == 1).groupby(b).sum() + 0.5
    woe = np.log((good / good.sum()) / (bad / bad.sum()))
    woe_tab[c] = (e, woe.to_dict())
def woe_frame(df): return pd.DataFrame({c: [woe_tab[c][1].get(int(i), 0.0) for i in np.digitize(df[c].values, woe_tab[c][0][1:-1])] for c in cols})
Wf, Wc, Wt = woe_frame(fit), woe_frame(cal), woe_frame(te)
lr = LogisticRegression(max_iter=2000).fit(Wf, fit[T].values)
pc, pt = lr.predict_proba(Wc)[:, 1], lr.predict_proba(Wt)[:, 1]; yc, yt = cal[T].values, te[T].values
factor = 20 / np.log(2); offset = 600 - factor * np.log(50)
points = offset + factor * np.log((1 - pt) / pt)
res = {"model": "WoE logistic scorecard (8 quantile bins for continuous attributes, 0/1/2/3/4+ for counts; 10 raw attributes)", "test": summarise(yt, pt),
       "score_points_test": {"p05": round(float(np.quantile(points, 0.05))), "median": round(float(np.median(points))), "p95": round(float(np.quantile(points, 0.95)))},
       "coefficients": dict(zip(cols, np.round(lr.coef_[0], 3).tolist()))}
for r in (5,):
    cap = conformal_cap(pc[yc == 1], 0.10); cut = referral_gain_cut(pc, r, 0.2, cap, 0.20)
    res["main_policy_with_CreditSense_triage"] = evaluate(decide_triage(pt, r, 0.2, cap, cut), yt, r, 0.2)
    res["two_way_cost_threshold"] = evaluate(np.where(pt >= 1 / (1 + r), DECLINE, APPROVE), yt, r, 0.2)
save_json(res, "p2_step16_scorecard_baseline.json"); print(json.dumps(res, indent=1))
