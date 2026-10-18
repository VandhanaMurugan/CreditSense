"""Phase 2 / Module 4 - cost-based decision threshold.

A missed default (false negative) costs more than a wrongly declined good borrower (false positive). With an explicit
cost ratio r = cost(FN)/cost(FP), a *calibrated* PD makes the Bayes-optimal cut-off t* = 1/(1+r).
Thresholds are chosen on the calibration hold-out and evaluated on the test set. Cost ratios are ASSUMPTIONS (sensitivity
analysis), not product data.
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from p2_common import *

_, _, _, y_te, _ = load_phase1(); yt = y_te.values
yc = np.load(os.path.join(ART, "p2_cal_y.npy")); raw_cal = np.load(os.path.join(ART, "p2_raw_cal.npy")); cal_cal = np.load(os.path.join(ART, "p2_cal_probs.npy")).ravel()
raw_te = np.load(os.path.join(ART, "p2_raw_te.npy")); cal_te = np.load(os.path.join(ART, "p2_calibrated_te.npy"))
grid = np.round(np.arange(0.01, 0.96, 0.005), 3)
def cost(y, p, t, r):
    pred = p >= t; fn = ((~pred) & (y == 1)).sum(); fp = (pred & (y == 0)).sum()
    return (r * fn + fp) / len(y) * 1000          # cost units per 1,000 borrowers (cost of one false alarm = 1 unit)
def f1(y, p, t):
    pred = p >= t; tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum(); fn = ((~pred) & (y == 1)).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1)
t_f1_raw = grid[np.argmax([f1(yc, raw_cal, t) for t in grid])]
res = {"note": "cost ratio r = cost(missed default)/cost(false alarm); assumption-driven sensitivity analysis", "f1_opt_threshold_raw": float(t_f1_raw), "scenarios": {}}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for r, col in zip([2, 5, 10, 20], ["#1F3A5F", "#2A7F8E", "#C8641E", "#8A3B12"]):
    t_emp = grid[np.argmin([cost(yc, cal_cal, t, r) for t in grid])]; t_theory = 1 / (1 + r)
    row = {"theoretical_threshold_1_over_1_plus_r": round(t_theory, 4), "empirical_cost_opt_threshold_calibration_set": float(t_emp),
           "test_cost_per_1000": {"raw_score_at_0.5 (Phase 1 default)": round(cost(yt, raw_te, 0.5, r), 2),
                                  "raw_score_at_F1_opt_threshold": round(cost(yt, raw_te, t_f1_raw, r), 2),
                                  "calibrated_at_theoretical_threshold": round(cost(yt, cal_te, t_theory, r), 2),
                                  "calibrated_at_empirical_threshold": round(cost(yt, cal_te, t_emp, r), 2)},
           "decline_rate_test_at_empirical": round(float((cal_te >= t_emp).mean()), 4),
           "recall_test_at_empirical": round(float(((cal_te >= t_emp) & (yt == 1)).sum() / (yt == 1).sum()), 4)}
    res["scenarios"][f"r={r}"] = row; print(r, json.dumps(row), flush=True)
    ax[0].plot(grid, [cost(yt, cal_te, t, r) for t in grid], color=col, label=f"r = {r}"); ax[0].axvline(t_theory, color=col, ls=":", lw=1)
ax[0].set_xlabel("Decision threshold on calibrated PD"); ax[0].set_ylabel("Expected cost per 1,000 borrowers (test)"); ax[0].set_title("Cost vs threshold (dotted = 1/(1+r))"); ax[0].legend(frameon=False); ax[0].set_xlim(0, 0.6)
ax[1].plot(grid, [f1(yt, raw_te, t) for t in grid], color="#C8641E", label="F1 on raw score"); ax[1].set_xlabel("Threshold on raw score"); ax[1].set_ylabel("F1 (test)"); ax[1].set_title("F1 peaks near 0.8 but ignores error costs"); ax[1].legend(frameon=False)
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "p2_cost_threshold.png"), dpi=150)
save_json(res, "p2_step4_cost_threshold.json")
