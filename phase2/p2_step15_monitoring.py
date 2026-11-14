"""Monitoring and learning loop (Review 2): does the system notice drift, and does recalibration on new outcomes fix it?
Scenarios drawn from the untouched test set (10,000 applicants each):
  S0 no drift (random sample)
  S1 covariate shift: more high-utilisation applicants
  S2 downturn / label shift: default rate doubled
  S3 younger applicant mix (e.g. a new digital channel)
For each: PSI of score and features (thresholds 0.10 / 0.25), predicted vs observed default rate, ECE, main-policy outcome.
Learning step on S2: refit Platt on the first 3,000 labelled outcomes (challenger) and test on the other 7,000.
Promotion rule fixed in advance: promote if challenger Brier is lower by more than 0.001 on the 7,000.
"""
import sys, os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from creditsense.serve import CreditSense
from creditsense.policy import conformal_cap, referral_gain_cut, decide_triage, evaluate
from p2_common import ROOT, OUT, SEED, save_json, ece

cs = CreditSense(); b = cs.b; pol = b["policy_default"]
raw = pd.read_csv(os.path.join(ROOT, "data", "GiveMeSomeCredit_dataset.csv"), index_col=0); raw = raw[raw["age"] > 0]
_, test_raw = train_test_split(raw, test_size=0.25, random_state=42, stratify=raw["SeriousDlqin2yrs"])
y_all = test_raw.pop("SeriousDlqin2yrs").values; rng = np.random.default_rng(SEED); N = 10000
util_rank = test_raw["RevolvingUtilizationOfUnsecuredLines"].rank(pct=True).values
scen = {"S0 no drift": rng.choice(len(y_all), N, replace=False),
        "S1 more high-utilisation applicants": rng.choice(len(y_all), N, replace=False, p=util_rank ** 2 / (util_rank ** 2).sum()),
        "S3 younger applicant mix": rng.choice(len(y_all), N, replace=False, p=(w := np.where(test_raw["age"].values < 35, 4.0, 1.0)) / w.sum())}
i1, i0 = np.where(y_all == 1)[0], np.where(y_all == 0)[0]; n1 = int(2 * y_all.mean() * N)
scen["S2 downturn: default rate doubled"] = np.r_[rng.choice(i1, n1, replace=False), rng.choice(i0, N - n1, replace=False)]

def policy_eval(p, y, p_cal):
    cap = conformal_cap(p_cal[b["y_cal"] == 1], pol["alpha"]); cut = referral_gain_cut(p_cal, pol["r"], pol["c"], cap, pol["rho"])
    return evaluate(decide_triage(p, pol["r"], pol["c"], cap, cut), y, pol["r"], pol["c"])

res = {}
for name, idx in scen.items():
    df, y = test_raw.iloc[idx].reset_index(drop=True), y_all[idx]
    mon = cs.monitoring(df, y); p = cs.pd(df)
    feats = {k: v for k, v in mon["psi"].items() if k != "score"}; worst = max(feats, key=lambda k: feats[k]["value"])
    res[name] = {"psi_score": mon["psi"]["score"], "worst_feature": {worst: feats[worst]}, "n_features_act": sum(v["status"] == "act" for v in feats.values()),
                 "n_features_watch": sum(v["status"] == "watch" for v in feats.values()),
                 "observed_default_rate": mon["observed_default_rate"], "mean_predicted_pd": mon["mean_predicted_pd"],
                 "ECE": round(ece(y, p), 4), "main_policy": policy_eval(p, y, b["p_cal"])}
    print(name, json.dumps(res[name]), flush=True)

# learning step on the downturn stream
idx = scen["S2 downturn: default rate doubled"]; rng.shuffle(idx)
df, y = test_raw.iloc[idx].reset_index(drop=True), y_all[idx]; p_old = cs.pd(df)
A, B = b["platt"]; lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
raw_logit = (lg(p_old) - B) / A                       # back to the stack's raw log-odds
new, hold = slice(0, 3000), slice(3000, None)
m = LogisticRegression(C=1e6, max_iter=1000).fit(raw_logit[new].reshape(-1, 1), y[new])
recal = lambda z: m.predict_proba(z.reshape(-1, 1))[:, 1]
p_new = recal(raw_logit); p_cal_new = recal((lg(b["p_cal"]) - B) / A)
champ_brier, chall_brier = brier_score_loss(y[hold], p_old[hold]), brier_score_loss(y[hold], p_new[hold])
res["learning_step_S2"] = {"recent_labelled_outcomes_used": 3000, "evaluated_on": int(len(y[hold])),
    "observed_default_rate_holdout": round(float(y[hold].mean()), 4),
    "champion": {"mean_pd": round(float(p_old[hold].mean()), 4), "ECE": round(ece(y[hold], p_old[hold]), 4), "Brier": round(champ_brier, 5),
                 "main_policy": policy_eval(p_old[hold], y[hold], b["p_cal"])},
    "challenger_recalibrated": {"mean_pd": round(float(p_new[hold].mean()), 4), "ECE": round(ece(y[hold], p_new[hold]), 4), "Brier": round(chall_brier, 5),
                 "main_policy": policy_eval(p_new[hold], y[hold], p_cal_new)},
    "promote": bool(champ_brier - chall_brier > 0.001),
    "note": "Recalibration fixes probability levels, not ranking. The guarantee cap is rank-based, so it is unaffected by a pure change in default rate."}
save_json(res, "p2_step15_monitoring.json"); print(json.dumps(res["learning_step_S2"], indent=1))

names = sorted(k for k in res if k.startswith("S"))
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4))
ax[0].bar(range(len(names)), [res[k]["psi_score"]["value"] for k in names], color="#2A7F8E", label="Score PSI")
ax[0].bar(range(len(names)), [list(res[k]["worst_feature"].values())[0]["value"] for k in names], color="none", edgecolor="#C8641E", lw=2, label="Worst feature PSI")
for y0, lab in ((0.10, "watch"), (0.25, "act")): ax[0].axhline(y0, color="grey", ls="--", lw=1); ax[0].text(len(names) - 0.6, y0 + 0.01, lab, fontsize=8)
ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels([k.split(" ", 1)[0] for k in names]); ax[0].set_ylabel("Population stability index"); ax[0].legend(frameon=False, fontsize=8)
ax[0].set_title("Drift alarms by scenario", fontsize=10)
L = res["learning_step_S2"]; cats = ["Observed", "Champion", "Recalibrated"]
ax[1].bar(cats, [100 * L["observed_default_rate_holdout"], 100 * L["champion"]["mean_pd"], 100 * L["challenger_recalibrated"]["mean_pd"]], color=["#555", "#C8641E", "#2A7F8E"])
ax[1].set_ylabel("Default rate (%)"); ax[1].set_title("Downturn: recalibration on 3,000 new outcomes", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "p2_monitoring.png"), dpi=150); print("saved")
