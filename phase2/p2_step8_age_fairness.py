"""H4 - calibration, decisions and referral by age band (test set). Age is a model input, so this is an audit of outcomes, not a fairness claim."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from p2_common import *
import importlib.util

_, X_te, _, y_te, _ = load_phase1(); yt = y_te.values; age = X_te["age"].values
yc = np.load(os.path.join(ART, "p2_cal_y.npy")); pc = np.load(os.path.join(ART, "p2_cal_probs.npy")).ravel(); pt = np.load(os.path.join(ART, "p2_calibrated_te.npy"))
alpha = 0.10
q_hat = lambda s, a: np.sort(s)[min(int(np.ceil((len(s) + 1) * (1 - a))), len(s)) - 1]
q0 = q_hat(1 - (1 - pc)[yc == 0], alpha); q1 = q_hat(1 - pc[yc == 1], alpha)
in1 = (1 - pt) <= q1; in0 = pt <= q0; refer = (in0.astype(int) + in1.astype(int)) != 1
bands = [(0, 30, "<30"), (30, 40, "30-39"), (40, 50, "40-49"), (50, 60, "50-59"), (60, 70, "60-69"), (70, 200, "70+")]
res = {"alpha": alpha, "bands": {}}
for lo, hi, lab in bands:
    m = (age >= lo) & (age < hi); y = yt[m]; p = pt[m]
    row = {"n": int(m.sum()), "default_rate": round(float(y.mean()), 4), "mean_predicted_pd": round(float(p.mean()), 4), "ECE_in_band": round(ece(y, p, bins=5), 4),
           "AUC": round(roc_auc_score(y, p), 4) if 0 < y.sum() < len(y) else None, "referral_rate": round(float(refer[m].mean()), 4),
           "defaulter_coverage": round(float(in1[m][y == 1].mean()), 4) if y.sum() else None}
    for r in (5, 10):
        t = 1 / (1 + r); dec = p >= t; row[f"decline_rate_r={r}"] = round(float(dec.mean()), 4); row[f"recall_r={r}"] = round(float((dec & (y == 1)).sum() / max((y == 1).sum(), 1)), 4)
    res["bands"][lab] = row; print(lab, json.dumps(row), flush=True)
save_json(res, "p2_step8_age_fairness.json")
labs = list(res["bands"]); fig, ax = plt.subplots(1, 2, figsize=(11, 4)); x = np.arange(len(labs))
ax[0].bar(x - .2, [res["bands"][l]["default_rate"] * 100 for l in labs], .4, label="Observed", color="#C8641E"); ax[0].bar(x + .2, [res["bands"][l]["mean_predicted_pd"] * 100 for l in labs], .4, label="Predicted (calibrated)", color="#2A7F8E")
ax[0].set_xticks(x); ax[0].set_xticklabels(labs); ax[0].set_ylabel("Default rate (%)"); ax[0].set_title("Calibration by age band"); ax[0].legend(frameon=False)
ax[1].bar(x - .2, [res["bands"][l]["referral_rate"] * 100 for l in labs], .4, label="Referral rate (α=0.10)", color="#1F3A5F"); ax[1].bar(x + .2, [res["bands"][l]["decline_rate_r=5"] * 100 for l in labs], .4, label="Decline rate (r=5)", color="#7A8794")
ax[1].set_xticks(x); ax[1].set_xticklabels(labs); ax[1].set_ylabel("% of borrowers"); ax[1].set_title("Decisions by age band"); ax[1].legend(frameon=False)
for a_ in ax: a_.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "p2_age_fairness.png"), dpi=150)
