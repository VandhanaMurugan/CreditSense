"""H3 - class-conditional (Mondrian) split-conformal referral on the calibrated stack.

Calibration split = the 20% train hold-out from step 2 (never the test set). For each class y, the nonconformity score is 1 - p(y).
Per-class quantiles give prediction sets; a borrower is REFERRED when the set is not a single label (both labels or empty).
Guarantee: coverage of true defaulters >= 1 - alpha (marginal over exchangeable data).
Baseline: Chow's cost-optimal reject rule at the SAME referral rate (refer the highest min(r*p, 1-p) first) - the strongest cost-only competitor.
Auto-decision rule for non-referred: decline if p >= 1/(1+r). Referred cases are assumed to be resolved correctly by an analyst (assumption; break-even referral cost is reported).
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from p2_common import *

_, _, _, y_te, _ = load_phase1(); yt = y_te.values
yc = np.load(os.path.join(ART, "p2_cal_y.npy")); pc = np.load(os.path.join(ART, "p2_cal_probs.npy")).ravel(); pt = np.load(os.path.join(ART, "p2_calibrated_te.npy"))
def q_hat(s, a): n = len(s); k = min(int(np.ceil((n + 1) * (1 - a))), n); return np.sort(s)[k - 1]
def conformal_sets(alpha):
    q = {c: q_hat(1 - np.where(c == 1, pc, 1 - pc)[yc == c], alpha) for c in (0, 1)}
    in1 = (1 - pt) <= q[1]; in0 = pt <= q[0]          # label 1 in set if 1-p1 <= q1 ; label 0 in set if 1-p0 = p1 <= q0
    return in0, in1, q
def missed_share(refer, r):
    """share of all defaulters that are auto-APPROVED (neither declined nor referred) under the t*=1/(1+r) rule"""
    dec = pt >= 1 / (1 + r); return float(((~dec) & (~refer) & (yt == 1)).sum() / (yt == 1).sum())
def auto_cost(refer, r):
    t = 1 / (1 + r); dec = pt >= t; keep = ~refer
    fn = ((~dec) & (yt == 1) & keep).sum(); fp = (dec & (yt == 0) & keep).sum(); return (r * fn + fp) / len(yt) * 1000
res = {"note": "referred cases assumed resolved correctly; cost unit = one wrongly declined good borrower", "alphas": {}}
for alpha in (0.05, 0.10, 0.20, 0.30):
    in0, in1, q = conformal_sets(alpha); size = in0.astype(int) + in1.astype(int); refer = size != 1; rate = float(refer.mean())
    cov1 = float(in1[yt == 1].mean()); cov0 = float(in0[yt == 0].mean())
    row = {"quantiles": {"class0": round(float(q[0]), 4), "class1": round(float(q[1]), 4)}, "referral_rate": round(rate, 4), "coverage_defaulters": round(cov1, 4), "coverage_non_defaulters": round(cov0, 4),
           "default_rate_in_referred": round(float(yt[refer].mean()), 4) if refer.any() else None, "share_of_defaulters_referred": round(float(refer[yt == 1].sum() / (yt == 1).sum()), 4)}
    for r in (5, 10):
        k = int(round(rate * len(yt))); loss = np.minimum(r * pt, 1 - pt); chow = np.zeros(len(yt), bool); chow[np.argsort(-loss)[:k]] = True
        row[f"r={r}"] = {"cost_no_referral": round(auto_cost(np.zeros(len(yt), bool), r), 2), "cost_conformal_referral": round(auto_cost(refer, r), 2), "cost_chow_same_rate": round(auto_cost(chow, r), 2),
                         "defaulters_auto_approved_share": {"no_referral": round(missed_share(np.zeros(len(yt), bool), r), 4), "conformal": round(missed_share(refer, r), 4), "chow_same_rate": round(missed_share(chow, r), 4)},
                         "break_even_referral_cost_per_case": round((auto_cost(np.zeros(len(yt), bool), r) - auto_cost(refer, r)) / max(rate * 1000, 1e-9), 3)}
    res["alphas"][f"alpha={alpha}"] = row; print(alpha, json.dumps(row), flush=True)
save_json(res, "p2_step7_conformal_referral.json")
al = [0.05, 0.1, 0.2, 0.3]; fig, ax = plt.subplots(1, 2, figsize=(11, 4))
rr = [res["alphas"][f"alpha={a}"]["referral_rate"] * 100 for a in al]
cv = [res["alphas"][f"alpha={a}"]["coverage_defaulters"] * 100 for a in al]
ax[0].scatter(rr, cv, color="#2A7F8E", s=60, zorder=3, label="Observed defaulter coverage"); ax[0].scatter(rr, [(1 - a) * 100 for a in al], marker="_", s=300, color="k", zorder=3, label="Guarantee (1 − α)")
for x_, y_, a_ in zip(rr, cv, al): ax[0].annotate(f"α={a_}", (x_, y_), textcoords="offset points", xytext=(6, -12), fontsize=9); ax[0].set_xlabel("Referral rate (% of borrowers)"); ax[0].set_ylabel("%"); ax[0].set_title("Coverage guarantee holds on test"); ax[0].legend(frameon=False)
for r, col in ((5, "#1F3A5F"), (10, "#C8641E")):
    ax[1].scatter(rr, [res["alphas"][f"alpha={a}"][f"r={r}"]["cost_conformal_referral"] for a in al], marker="o", s=60, color=col, label=f"Conformal, r={r}")
    ax[1].scatter(rr, [res["alphas"][f"alpha={a}"][f"r={r}"]["cost_chow_same_rate"] for a in al], marker="s", s=60, color=col, alpha=.55, label=f"Chow rule, r={r}")
ax[1].set_xlabel("Referral rate (% of borrowers)"); ax[1].set_ylabel("Auto-decision cost per 1,000"); ax[1].set_title("Cost of non-referred decisions"); ax[1].legend(frameon=False, fontsize=8)
for a_ in ax: a_.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "p2_conformal_referral.png"), dpi=150)
