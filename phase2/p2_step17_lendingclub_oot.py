"""N14: out-of-time validation of the CreditSense decision layer on real, dated lending data.

Data: Lending Club accepted loans (public), 36-month individual loans issued 2010-2015 with a final outcome.
Why: Give Me Some Credit has no dates and no money. This dataset has both: an issue date per loan and the cash actually
repaid, so the model can be tested on a LATER period and decisions can be valued in dollars instead of assumed units.

Design (leak-safe, split by issue date):
  train      2010-2012   fit cleaning statistics, category lists and all models
  calibrate  2013        Platt map, guarantee cap, referral cut, money parameters are NOT taken from here (train only)
  test       2014, 2015  never used for any fitting
Features are application-time facts only. The lender's own grade, interest rate and instalment are NOT model inputs;
the grade is kept as the "existing score" baseline, and rate/instalment are used only to value decisions.
Same fixed, untuned model settings as Phase I (no tuning on this dataset).

PASS MARKS - written before the first run, reported as met / not met whatever the result:
  P1 ranking      calibrated stack AUC on 2014-2015 >= AUC of the lender's own sub-grade - 0.01
  P2 calibration  frozen Platt map: ECE on 2014-2015 <= 0.02
  P3 guarantee    cap set on 2013 defaulters at alpha = 10%: share of 2014-2015 defaulters auto-approved <= 12%
  P4 money        CreditSense (alpha 10%, budget 20%, $25 per review, perfect analyst) earns at least as much per
                  1,000 loans on 2014-2015 as approving every loan (what the lender did)
  P5 stacking     H1 expects NO meaningful gain: stack - best single model AUC gain < 0.005
Known limits: every loan here was approved by the lender (no rejected applicants); profit = total paid - amount funded
(no time value, servicing cost or fees); in real time the 2013 outcomes would not all be known when 2014 loans are
scored (label delay) - variant B below respects that delay on a smaller sample.
"""
import os, sys, time, numpy as np, pandas as pd
import xgboost as xgb, lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from p2_common import ROOT, SEED, WeightedGB, ks_stat, ece, save_json
from creditsense.policy import (conformal_cap, referral_gain_cut, decide_triage, bayes_threshold, APPROVE, REFER, DECLINE)

SRC = os.path.join(ROOT, "data", "lendingclub", "accepted_2007_to_2018Q4.csv.gz")
if not os.path.exists(SRC): sys.exit("Lending Club file not found - see data/lendingclub/README.md")
t0 = time.time(); rng = np.random.default_rng(SEED)
PASS_MARKS = {"P1_auc_vs_grade_min": -0.01, "P2_ece_max": 0.02, "P3_miss_max_at_alpha10": 0.12, "P4_profit_vs_approve_all_min": 0.0, "P5_stack_gain_max": 0.005}
REVIEW_COST = 25.0   # dollars per referred case: an ASSUMPTION (sensitivity 10 / 25 / 50 below)
RHO = 0.20           # analyst budget: an ASSUMPTION
NUM = ["loan_amnt", "annual_inc", "dti", "fico_range_low", "emp_years", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
       "revol_bal", "revol_util", "total_acc", "credit_age_months", "mths_since_last_delinq", "no_delinq_on_file",
       "pub_rec_bankruptcies", "loan_to_income"]
CAT = ["home_ownership", "purpose", "verification_status"]
USE = ["loan_amnt", "funded_amnt", "term", "int_rate", "installment", "sub_grade", "emp_length", "home_ownership", "annual_inc",
       "verification_status", "issue_d", "loan_status", "purpose", "dti", "delinq_2yrs", "earliest_cr_line", "fico_range_low",
       "inq_last_6mths", "mths_since_last_delinq", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "total_pymnt",
       "application_type", "pub_rec_bankruptcies"]

# ------------------------------------------------------------------ load (chunked: the file has 2.26 million rows)
parts, seen = [], 0
for ch in pd.read_csv(SRC, usecols=USE, chunksize=200_000, low_memory=False):
    seen += len(ch)
    ch["issue"] = pd.to_datetime(ch["issue_d"], format="%b-%Y", errors="coerce")
    parts.append(ch[(ch["term"].astype(str).str.strip() == "36 months") & ch["issue"].dt.year.between(2010, 2015)])
raw = pd.concat(parts, ignore_index=True); del parts
n_window = len(raw)
raw = raw[(raw["application_type"] == "Individual") & raw["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])].reset_index(drop=True)
raw["y"] = (raw["loan_status"] != "Fully Paid").astype(int)
raw["year"] = raw["issue"].dt.year
raw["net"] = raw["total_pymnt"] - raw["funded_amnt"]                      # dollars earned (+) or lost (-) on the loan
raw["emp_years"] = raw["emp_length"].astype(str).str.extract(r"(\d+)")[0].astype(float)
raw["credit_age_months"] = (raw["issue"] - pd.to_datetime(raw["earliest_cr_line"], format="%b-%Y", errors="coerce")).dt.days / 30.44
raw["no_delinq_on_file"] = raw["mths_since_last_delinq"].isna().astype(int)
raw["loan_to_income"] = raw["loan_amnt"] / raw["annual_inc"].clip(lower=1)
raw["grade_rank"] = raw["sub_grade"].map({f"{g}{i}": k for k, (g, i) in enumerate([(g, i) for g in "ABCDEFG" for i in range(1, 6)], 1)}).astype(float)
print(f"rows in file {seen:,}; 36-month 2010-2015 {n_window:,}; individual with final outcome {len(raw):,}  ({time.time()-t0:.0f}s)")

tr, ca = raw["year"] <= 2012, raw["year"] == 2013
te = raw["year"] >= 2014

# ------------------------------------------------------------------ leak-safe preparation (everything fitted on train years)
def fit_prep(d):
    med = d[NUM].median()
    caps = {c: float(d[c].quantile(0.995)) for c in ["annual_inc", "revol_bal", "dti", "revol_util", "loan_to_income"]}
    cats = {c: [v for v, n in d[c].value_counts().items() if n >= 50] for c in CAT}
    return {"med": med, "caps": caps, "cats": cats}

def apply_prep(d, P):
    X = d[NUM].copy()
    X["mths_since_last_delinq"] = X["mths_since_last_delinq"].fillna(999.0)
    X = X.fillna(P["med"])
    for c, v in P["caps"].items(): X[c] = X[c].clip(upper=v)
    X["annual_inc"] = np.log1p(X["annual_inc"]); X["revol_bal"] = np.log1p(X["revol_bal"])
    for c in CAT:
        for v in P["cats"][c]: X[f"{c}={v}"] = (d[c] == v).astype(int).values
    return X.astype(float)

def members(spw):
    return [("rf", RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced_subsample", n_jobs=2, random_state=SEED)),
            ("gb", WeightedGB(n_estimators=200, max_depth=4, learning_rate=0.05)),
            ("xgb", xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, scale_pos_weight=spw, n_jobs=2, random_state=SEED, eval_metric="logloss"))]

def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p)).reshape(-1, 1)
def fit_platt(s, y): m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(s), y); return lambda q: m.predict_proba(logit(q))[:, 1]
def scores(y, p): return {"AUC": round(float(roc_auc_score(y, p)), 4), "KS": round(ks_stat(y, p), 4), "Brier": round(float(brier_score_loss(y, p)), 4),
                          "ECE": round(ece(np.asarray(y), np.asarray(p)), 4), "mean_predicted": round(float(np.mean(p)), 4), "observed": round(float(np.mean(y)), 4)}

P = fit_prep(raw[tr]); X = apply_prep(raw, P); y = raw["y"].values
Xtr, ytr = X[tr], y[tr]; spw = float((ytr == 0).sum() / (ytr == 1).sum())
print(f"train {tr.sum():,} (default {ytr.mean():.3f}) | calibrate {ca.sum():,} ({y[ca].mean():.3f}) | test {te.sum():,} ({y[te].mean():.3f}); {X.shape[1]} features")

models = {"LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
          "RF": members(spw)[0][1], "GB": members(spw)[1][1], "XGB": members(spw)[2][1],
          "LightGBM": lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, scale_pos_weight=spw, n_jobs=2, random_state=SEED, verbose=-1),
          "Stack": StackingClassifier(members(spw), final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"), cv=3, n_jobs=1)}
rawp, pd_ = {}, {}
for name, m in models.items():
    m.fit(Xtr, ytr); rawp[name] = m.predict_proba(X)[:, 1]
    pd_[name] = fit_platt(rawp[name][ca], y[ca])(rawp[name])            # frozen Platt map learned on 2013
    print(f"  {name:9s} fitted  test AUC {roc_auc_score(y[te], pd_[name][te]):.4f}  ({time.time()-t0:.0f}s)")
# existing score: the lender's own sub-grade, turned into a PD with the same Platt step (rank 1..35 scaled to (0,1))
g = (raw["grade_rank"].fillna(raw["grade_rank"].median()).values) / 36.0
pd_["Lender sub-grade"] = fit_platt(g[ca], y[ca])(g)

res = {"data": {"source": "Lending Club accepted loans 2007-2018Q4 (public)", "rows_in_file": int(seen), "rows_36m_2010_2015": int(n_window),
                "rows_used": int(len(raw)), "train_2010_2012": int(tr.sum()), "calibrate_2013": int(ca.sum()), "test_2014": int((raw.year == 2014).sum()),
                "test_2015": int((raw.year == 2015).sum()), "default_rate_by_year": {int(k): round(float(v), 4) for k, v in raw.groupby("year")["y"].mean().items()},
                "n_features": int(X.shape[1]), "target": "Charged Off or Default (vs Fully Paid)"},
       "pass_marks": PASS_MARKS, "assumptions": {"review_cost_dollars": REVIEW_COST, "analyst_budget": RHO, "analyst_accuracy_main": 1.0}}

# ------------------------------------------------------------------ 1. ranking and probability quality, out of time
res["metrics"] = {n: {"test_2014_2015": scores(y[te], p[te]), "test_2014": scores(y[raw.year == 2014], p[raw.year == 2014]),
                      "test_2015": scores(y[raw.year == 2015], p[raw.year == 2015]), "calibration_year_2013": scores(y[ca], p[ca])} for n, p in pd_.items()}
singles = ["LR", "RF", "GB", "XGB", "LightGBM"]
best = max(singles, key=lambda n: res["metrics"][n]["test_2014_2015"]["AUC"])
idx_te = np.where(te)[0]; sub = rng.choice(idx_te, size=min(120_000, len(idx_te)), replace=False)
def boot_auc_diff(a, b, B=500):
    d = []
    for _ in range(B):
        i = rng.choice(sub, size=len(sub), replace=True)
        d.append(roc_auc_score(y[i], pd_[a][i]) - roc_auc_score(y[i], pd_[b][i]))
    return {"mean": round(float(np.mean(d)), 4), "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]}
res["stacking_H1"] = {"best_single": best, "stack_minus_best_single_auc": boot_auc_diff("Stack", best),
                      "stack_minus_lender_grade_auc": boot_auc_diff("Stack", "Lender sub-grade"), "bootstrap": "500 paired resamples of 120,000 test loans"}
print("stack - best single:", res["stacking_H1"]["stack_minus_best_single_auc"], "| stack - grade:", res["stacking_H1"]["stack_minus_lender_grade_auc"])

# ------------------------------------------------------------------ 2. the guarantee, in time and out of time
p = pd_["Stack"]; raw["quarter"] = raw["issue"].dt.to_period("Q").astype(str)
guar = {}
for alpha in (0.05, 0.10, 0.20):
    cap = conformal_cap(p[ca][y[ca] == 1], alpha)
    in_time = []
    ci = np.where(ca)[0]
    for _ in range(200):                                              # same period, random halves: the guarantee's own assumption
        perm = rng.permutation(ci); a, b = perm[: len(perm) // 2], perm[len(perm) // 2:]
        in_time.append(float((p[b][y[b] == 1] < conformal_cap(p[a][y[a] == 1], alpha)).mean()))
    byq = {q: round(float((p[(raw.quarter == q).values & (y == 1)] < cap).mean()), 4) for q in sorted(raw.quarter[te].unique())}
    guar[str(alpha)] = {"cap_pd": round(cap, 4), "in_time_2013_mean_over_200_splits": round(float(np.mean(in_time)), 4),
                        "in_time_95th_percentile": round(float(np.percentile(in_time, 95)), 4),
                        "out_of_time_2014": round(float((p[(raw.year == 2014).values & (y == 1)] < cap).mean()), 4),
                        "out_of_time_2015": round(float((p[(raw.year == 2015).values & (y == 1)] < cap).mean()), 4),
                        "out_of_time_2014_2015": round(float((p[te.values & (y == 1)] < cap).mean()), 4), "by_quarter": byq}
res["guarantee"] = guar
print("guarantee alpha 10%:", {k: v for k, v in guar["0.1"].items() if k != "by_quarter"})

# ------------------------------------------------------------------ 3. money: costs measured on the train years, decisions valued on the test years
d_tr = raw[tr]
good_margin = float((d_tr.net[d_tr.y == 0] / d_tr.funded_amnt[d_tr.y == 0]).mean())          # earned per dollar lent, good loans
bad_loss = float((-d_tr.net[d_tr.y == 1] / d_tr.funded_amnt[d_tr.y == 1]).mean())            # lost per dollar lent, defaulted loans
r_emp = bad_loss / good_margin
contract = (d_tr.installment * 36 - d_tr.funded_amnt)[d_tr.y == 0]
haircut = float((d_tr.net[d_tr.y == 0] / contract).mean())                                     # good loans repay early: share of contract interest actually earned
unit = float(d_tr.net[d_tr.y == 0].mean())                                                      # dollars in "one good loan": the unit of the cost model
res["money_parameters_from_train_years"] = {"profit_per_dollar_good_loan": round(good_margin, 4), "loss_per_dollar_defaulted_loan": round(bad_loss, 4),
                                            "measured_cost_ratio_r": round(r_emp, 2), "share_of_contract_interest_earned_on_good_loans": round(haircut, 3),
                                            "mean_profit_good_loan_dollars": round(unit, 0)}
print("money parameters:", res["money_parameters_from_train_years"])
r_i = (bad_loss * raw.funded_amnt.values) / np.clip(haircut * (raw.installment.values * 36 - raw.funded_amnt.values), 1.0, None)   # per-loan cost ratio from its own price

def value(dec, idx, q=1.0, c=REVIEW_COST):
    """Dollars per 1,000 loans. approve -> what the loan actually returned; decline -> 0;
    refer -> pay the review, analyst right with probability q (good loans approved, defaulters declined)."""
    net, yy, dd = raw.net.values[idx], y[idx], dec
    v = np.where(dd == APPROVE, net, 0.0)
    v = v + np.where(dd == REFER, -c + np.where(yy == 0, q * net, (1 - q) * net), 0.0)
    return {"profit_per_1000_loans": round(float(v.mean() * 1000), 0), "approve_rate": round(float((dd == APPROVE).mean()), 4),
            "refer_rate": round(float((dd == REFER).mean()), 4), "decline_rate": round(float((dd == DECLINE).mean()), 4),
            "defaulters_auto_approved": round(float(((dd == APPROVE) & (yy == 1)).sum() / max((yy == 1).sum(), 1)), 4),
            "goods_auto_declined": round(float(((dd == DECLINE) & (yy == 0)).sum() / max((yy == 0).sum(), 1)), 4)}, v

def rules(p, c_dollars=REVIEW_COST):
    c = c_dollars / unit; out = {}
    out["Approve all (what the lender did)"] = np.full(len(p), APPROVE)
    out["Two-way cost threshold (measured r)"] = np.where(p >= bayes_threshold(r_emp), DECLINE, APPROVE)
    out["Two-way, price-aware (r per loan)"] = np.where(p >= 1.0 / (1.0 + r_i), DECLINE, APPROVE)
    cut = referral_gain_cut(p[ca], r_emp, c, np.inf, RHO)
    out["Same budget, no guarantee"] = decide_triage(p, r_emp, c, np.inf, cut)
    for alpha in (0.10, 0.20):
        cap = conformal_cap(p[ca][y[ca] == 1], alpha); cut = referral_gain_cut(p[ca], r_emp, c, cap, RHO)
        out[f"CreditSense alpha {int(alpha*100)}%"] = decide_triage(p, r_emp, c, cap, cut)
    return out

def boot_diff(va, vb, B=500):
    d = va - vb; n = len(d); s = [float(d[rng.integers(0, n, n)].mean() * 1000) for _ in range(B)]
    return {"mean": round(float(d.mean() * 1000), 0), "ci95": [round(float(np.percentile(s, 2.5)), 0), round(float(np.percentile(s, 97.5)), 0)]}

money = {}
for score_name in ("Stack", "Lender sub-grade"):
    R = rules(pd_[score_name]); block = {}; vals = {}
    for name, dec in R.items(): block[name], vals[name] = value(dec[idx_te], idx_te)
    block["vs_approve_all_dollars_per_1000"] = {n: boot_diff(vals[n], vals["Approve all (what the lender did)"]) for n in R if not n.startswith("Approve all")}
    by_year = {}
    for yr in (2014, 2015):
        iy = np.where((raw.year == yr).values)[0]; by_year[yr] = {n: value(dec[iy], iy)[0] for n, dec in R.items()}
    block["by_year"] = by_year
    money[score_name] = block
    print(f"-- decisions on the {score_name} PD (test 2014-2015, $ per 1,000 loans)")
    for n in R: print(f"   {n:38s} {block[n]['profit_per_1000_loans']:>12,.0f}  A/R/D {block[n]['approve_rate']:.2f}/{block[n]['refer_rate']:.2f}/{block[n]['decline_rate']:.2f}  missed {block[n]['defaulters_auto_approved']:.3f}")
res["money"] = money
# sensitivity: review cost and analyst accuracy (CreditSense alpha 10% on the stack PD)
sens = {}
for c in (10.0, 25.0, 50.0):
    dec = rules(pd_["Stack"], c)["CreditSense alpha 10%"][idx_te]
    for q in (1.0, 0.9, 0.8): sens[f"review ${int(c)}, analyst right {int(q*100)}%"] = value(dec, idx_te, q, c)[0]["profit_per_1000_loans"]
res["money_sensitivity_creditsense_alpha10"] = sens
print("sensitivity:", sens)

# ------------------------------------------------------------------ 3b. EXPLORATORY - added AFTER the first run, to explain its result (not a pass mark)
# (i) how good must the analyst be for the referral band to pay?  (ii) what does each level of the cap cost in dollars?
dec_ng = rules(pd_["Stack"])["Same budget, no guarantee"][idx_te]
base = value(np.full(len(idx_te), APPROVE), idx_te)[0]["profit_per_1000_loans"]
q_grid = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
by_q = {str(q): value(dec_ng, idx_te, q)[0]["profit_per_1000_loans"] for q in q_grid}
ref = dec_ng == REFER; net_te, y_te = raw.net.values[idx_te], y[idx_te]
G, Bd, n_ref = float(net_te[ref & (y_te == 0)].sum()), float(net_te[ref & (y_te == 1)].sum()), int(ref.sum())
q_even = (G + REVIEW_COST * n_ref) / (G - Bd)        # solves q*G + (1-q)*Bd - c*n = G + Bd
price = {}
for alpha in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80):
    cap = conformal_cap(pd_["Stack"][ca][y[ca] == 1], alpha); cut = referral_gain_cut(pd_["Stack"][ca], r_emp, REVIEW_COST / unit, cap, RHO)
    v = value(decide_triage(pd_["Stack"], r_emp, REVIEW_COST / unit, cap, cut)[idx_te], idx_te)[0]
    price[str(alpha)] = {"profit_per_1000_loans": v["profit_per_1000_loans"], "defaulters_auto_approved": v["defaulters_auto_approved"], "decline_rate": v["decline_rate"]}
res["exploratory_after_first_run"] = {
    "note": "Added after seeing the pass-mark results, to explain them. Not pre-registered.",
    "approve_all_profit_per_1000": base, "same_budget_no_guarantee_profit_by_analyst_accuracy": by_q,
    "analyst_accuracy_needed_to_match_approve_all": round(float(q_even), 3),
    "default_rate_among_referred": round(float(y_te[ref].mean()), 4), "profit_by_cap_alpha": price}
print("exploratory: break-even analyst accuracy", round(q_even, 3), "| referred default rate", round(float(y_te[ref].mean()), 3))
print("exploratory: profit by alpha", {a: (v["profit_per_1000_loans"], v["defaulters_auto_approved"]) for a, v in price.items()})

# ------------------------------------------------------------------ 4. variant B: respect label delay (only outcomes knowable before the test year)
# Loans issued 2010-2011 are fully matured by the end of 2014, so a lender scoring 2015 applications could really have used them.
trB = (raw.issue < "2011-07-01").values; caB = ((raw.issue >= "2011-07-01") & (raw.year == 2011)).values; teB = (raw.year == 2015).values
PB = fit_prep(raw[trB]); XB = apply_prep(raw, PB)
mB = lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, scale_pos_weight=float((y[trB] == 0).sum() / (y[trB] == 1).sum()), n_jobs=2, random_state=SEED, verbose=-1).fit(XB[trB], y[trB])
sB = mB.predict_proba(XB)[:, 1]; pB = fit_platt(sB[caB], y[caB])(sB)
capB = conformal_cap(pB[caB][y[caB] == 1], 0.10)
res["variant_B_label_delay"] = {"train": "issued Jan 2010 - Jun 2011", "calibrate": "issued Jul - Dec 2011", "test": "issued 2015", "model": "LightGBM (single model)",
                                "n_train": int(trB.sum()), "n_calibrate": int(caB.sum()), "n_test": int(teB.sum()),
                                "test_2015": scores(y[teB], pB[teB]), "defaulters_auto_approved_alpha10": round(float((pB[teB & (y == 1)] < capB).mean()), 4)}
print("variant B:", res["variant_B_label_delay"])

# ------------------------------------------------------------------ verdicts against the pass marks
M = res["metrics"]; cs = money["Stack"]
verdict = {
    "P1 ranking": {"value": round(M["Stack"]["test_2014_2015"]["AUC"] - M["Lender sub-grade"]["test_2014_2015"]["AUC"], 4), "pass_mark": ">= -0.01"},
    "P2 calibration": {"value": M["Stack"]["test_2014_2015"]["ECE"], "pass_mark": "<= 0.02"},
    "P3 guarantee": {"value": guar["0.1"]["out_of_time_2014_2015"], "pass_mark": "<= 0.12"},
    "P4 money": {"value": cs["vs_approve_all_dollars_per_1000"]["CreditSense alpha 10%"], "pass_mark": "mean >= 0 (dollars per 1,000 loans vs approving all)"},
    "P5 stacking": {"value": res["stacking_H1"]["stack_minus_best_single_auc"]["mean"], "pass_mark": "< 0.005 (H1: no meaningful gain)"}}
verdict["P1 ranking"]["met"] = bool(verdict["P1 ranking"]["value"] >= -0.01)
verdict["P2 calibration"]["met"] = bool(verdict["P2 calibration"]["value"] <= 0.02)
verdict["P3 guarantee"]["met"] = bool(verdict["P3 guarantee"]["value"] <= 0.12)
verdict["P4 money"]["met"] = bool(verdict["P4 money"]["value"]["mean"] >= 0)
verdict["P5 stacking"]["met"] = bool(verdict["P5 stacking"]["value"] < 0.005)
res["verdicts"] = verdict
for k, v in verdict.items(): print(f"{k:16s} {'MET' if v['met'] else 'NOT MET':8s} value {v['value']}  (pass mark {v['pass_mark']})")
res["runtime_seconds"] = round(time.time() - t0)
save_json(res, "p2_step17_lendingclub_oot.json")
print(f"saved phase2/outputs/p2_step17_lendingclub_oot.json  ({time.time()-t0:.0f}s)")
