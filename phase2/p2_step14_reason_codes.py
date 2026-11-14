"""Exact reason codes for the calibrated stack (creditsense/explain.py) - checks for Review 2.
  1. Additivity: base + sum of contributions = calibrated log-odds (exactness of the decomposition)
  2. Stability: top reason with two other random 100-row SHAP backgrounds
  3. Shortcut fidelity: would explaining only the XGBoost member give the same top reason?
  4. Governance: how often age would be among the top-4 drivers (it is never shown as a reason)
Applicants explained: 500 test applicants NOT auto-approved under the main policy (those who would need reasons).
"""
import sys, os, json, time, numpy as np, pandas as pd, joblib, shap, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from creditsense.explain import StackExplainer, REASONS
from creditsense.policy import decide_triage, APPROVE
from p2_common import ART, OUT, SEED, load_phase1, save_json

b = joblib.load(os.path.join(ART, "creditsense_bundle.joblib"))
X_tr, X_te, _, y_te, _ = load_phase1()
pol = b["policy_default"]; pt = b["p_test"]
from creditsense.policy import conformal_cap, referral_gain_cut
cap = conformal_cap(b["p_cal"][b["y_cal"] == 1], pol["alpha"]); cut = referral_gain_cut(b["p_cal"], pol["r"], pol["c"], cap, pol["rho"])
dec = decide_triage(pt, pol["r"], pol["c"], cap, cut)
rng = np.random.default_rng(SEED); idx = rng.choice(np.where(dec != APPROVE)[0], 500, replace=False)
X = X_te.iloc[idx]; lo = np.log(np.clip(pt[idx], 1e-6, 1 - 1e-6) / (1 - np.clip(pt[idx], 1e-6, 1 - 1e-6)))

def top_codes(ex, phi, k=1):
    return [[r["code"] for r in ex.reasons(row, top=k)] for row in phi]

t0 = time.time(); ex = StackExplainer(b["stack"], b["platt"], b["shap_background"]); phi = ex.shap_logodds(X); secs = (time.time() - t0) / len(X)
add_err = np.abs(ex.base_value + phi.sum(1) - lo)
X_fit_like = X_tr.drop(index=b["shap_background"].index, errors="ignore")
ex2 = StackExplainer(b["stack"], b["platt"], X_fit_like.sample(100, random_state=SEED + 1)); phi2 = ex2.shap_logodds(X)
ex3 = StackExplainer(b["stack"], b["platt"], X_fit_like.sample(100, random_state=SEED + 2)); phi3 = ex3.shap_logodds(X)
t1, t1b, t1c = top_codes(ex, phi), top_codes(ex2, phi2), top_codes(ex3, phi3)
t3, t3b = top_codes(ex, phi, 3), top_codes(ex2, phi2, 3)
xgb_model = dict(b["stack"].named_estimators_)["xgb"]
phx = shap.TreeExplainer(xgb_model, data=b["shap_background"], model_output="probability", feature_perturbation="interventional").shap_values(X)
t1x = top_codes(ex, phx)
def group_abs(row):
    col = {f: i for i, f in enumerate(X.columns)}
    return {k: abs(sum(row[col[f]] for f in feats)) for k, (feats, _, _) in REASONS.items()}
age_top4 = np.mean([("age" in sorted((g := group_abs(row)), key=lambda k: -g[k])[:4]) for row in phi])
freq = pd.Series([t[0] for t in t1]).value_counts(normalize=True).round(4).to_dict()
res = {"n_explained": len(X), "seconds_per_applicant": round(secs, 4),
       "additivity_abs_error": {"max": float(add_err.max()), "mean": float(add_err.mean())},
       "stability_top1_agreement": {"second_100_row_background": round(float(np.mean([a == c for a, c in zip(t1, t1b)])), 4),
                                    "third_100_row_background": round(float(np.mean([a == c for a, c in zip(t1, t1c)])), 4)},
       "stability_top3_jaccard_other_background": round(float(np.mean([len(set(a) & set(c)) / len(set(a) | set(c)) for a, c in zip(t3, t3b)])), 4),
       "xgb_only_shortcut_top1_agreement": round(float(np.mean([a == c for a, c in zip(t1, t1x)])), 4),
       "age_among_top4_drivers_share": round(float(age_top4), 4),
       "top_reason_frequency": freq,
       "example": {"pd": round(float(pt[idx[0]]), 4), "reasons": ex.reasons(phi[0], top=4), "inputs": X.iloc[0].to_dict()}}
save_json(res, "p2_step14_reason_codes.json"); print(json.dumps(res, indent=1))

labels = {"serious_delinquency": "90+ days late", "late_30_59": "30-59 days late", "late_60_89": "60-89 days late",
          "delinquency_history": "Late-payment history", "utilisation": "Credit utilisation", "open_lines": "Open credit lines",
          "real_estate": "Real-estate loans"}
s = pd.Series(freq).sort_values()
fig, ax = plt.subplots(figsize=(6.4, 3.6)); ax.barh([labels.get(k, k) for k in s.index], 100 * s.values, color="#2A7F8E")
ax.set_xlabel("% of referred/declined applicants (n=500)"); ax.set_title("Top reason given (age is never shown)", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "p2_reason_codes.png"), dpi=150)
