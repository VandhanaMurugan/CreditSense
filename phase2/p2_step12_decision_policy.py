"""Phase II Review 2 - modified algorithm: CreditSense triage (cost-optimal three-way decision + guaranteed cap on
auto-approved defaulters). See creditsense/policy.py for the algorithm.

Data: calibrated stack scores from step 2 (calibration hold-out n=22,500 and untouched test n=37,500).
Costs: approve a defaulter = r, decline a good = 1, refer = c per case (analyst assumed correct). r, c are assumptions.
Experiments
  A  main comparison, r in {5,10}, c in {0.1,0.2,0.3}, referral budget 20% (Phase I rule, two-way cost rule, Chow,
     budgeted Chow, symmetric conformal, CreditSense)
  B  price of the guarantee: cost of CreditSense vs budgeted Chow over alpha and budget
  D  validity: 500 random re-splits of the 60,000 out-of-sample scores (Platt refit each time)
  E  label shift: test default rate x0.5 and x2 (resampled)
  F  age bands: global cap vs per-band caps (group-conditional) - an audit, see legal note in the README
"""
import sys, os, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from creditsense.policy import *
from p2_common import ART, OUT, SEED, load_phase1, save_json, ece

X_tr, X_te, y_tr, y_te, _ = load_phase1(); yt = y_te.values
_, X_cal, _, _ = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=SEED)   # same split as step 2
yc = np.load(os.path.join(ART, "p2_cal_y.npy")); pc = np.load(os.path.join(ART, "p2_cal_probs.npy")).ravel()
pt = np.load(os.path.join(ART, "p2_calibrated_te.npy")); rc = np.load(os.path.join(ART, "p2_raw_cal.npy")); rt = np.load(os.path.join(ART, "p2_raw_te.npy"))
assert len(yc) == len(X_cal)
age_c, age_t = X_cal["age"].values, X_te["age"].values

def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p)).reshape(-1, 1)
def platt(raw_fit, y_fit):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(raw_fit), y_fit); return lambda q: m.predict_proba(logit(q))[:, 1]

C0, RHO = 0.2, 0.20          # main configuration: referral cost 0.2, analysts can review 20% of applicants
res = {"assumptions": {"cost_units": "1 = one wrongly declined good borrower", "c_referral": C0, "referral_budget": RHO,
                       "analyst_accuracy": "referred cases assumed decided correctly"},
       "calibration_defaulters": int(yc.sum())}
cap_of = lambda a, delta=None, mask=None: conformal_cap(pc[(yc == 1) if mask is None else (mask & (yc == 1))], a, delta)
def cs(p, r, c, a, rho, delta=None, p_cal=pc, cap=None):
    cap = cap_of(a, delta) if cap is None else cap
    return decide_triage(p, r, c, cap, referral_gain_cut(p_cal, r, c, cap, rho))
def chow_budget(p, r, c, rho, p_cal=pc):
    return decide_triage(p, r, c, np.inf, referral_gain_cut(p_cal, r, c, np.inf, rho))

# ---------- A: main comparison
A = {}
for r in (5, 10):
  for c in (0.1, 0.2, 0.3):
    rules = {"Phase I rule (raw score >= 0.5)": np.where(rt >= 0.5, DECLINE, APPROVE),
             "Two-way cost threshold 1/(1+r)": decide_two_way(pt, r),
             "Chow, unlimited referrals": decide_chow(pt, r, c),
             "Chow, 20% referral budget": chow_budget(pt, r, c, RHO),
             "Symmetric conformal alpha=0.10 (H3)": conformal_symmetric(pt, pc, yc, 0.10),
             "CreditSense alpha=0.10, unlimited": cs(pt, r, c, 0.10, None),
             "CreditSense alpha=0.10, 20% budget": cs(pt, r, c, 0.10, RHO),
             "CreditSense alpha=0.10 (95% PAC), 20% budget": cs(pt, r, c, 0.10, RHO, delta=0.05),
             "CreditSense alpha=0.05, 20% budget": cs(pt, r, c, 0.05, RHO)}
    A[f"r={r},c={c}"] = {k: evaluate(v, yt, r, c) for k, v in rules.items()}
res["caps"] = {"alpha=0.05": round(cap_of(0.05), 4), "alpha=0.10": round(cap_of(0.10), 4), "alpha=0.10 PAC95": round(cap_of(0.10, 0.05), 4), "alpha=0.20": round(cap_of(0.20), 4)}
res["A_main"] = A

# ---------- B: price of the guarantee at fixed referral budgets (r=5, c=0.2)
alphas = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
B = {}
for rho in (0.10, 0.20, 0.30, None):
    base = evaluate(chow_budget(pt, 5, C0, rho), yt, 5, C0); rows = {}
    for a in alphas:
        e = evaluate(cs(pt, 5, C0, a, rho), yt, 5, C0); e["premium_vs_chow_per_1000"] = round(e["cost_per_1000"] - base["cost_per_1000"], 2); rows[f"{a:.2f}"] = e
    B["unlimited" if rho is None else f"rho={rho}"] = {"chow_same_budget": base, "creditsense_by_alpha": rows}
res["B_price_of_guarantee_r5_c0.2"] = B

# ---------- D: validity over random re-splits (main configuration)
rng = np.random.default_rng(SEED); raw_all = np.r_[rc, rt]; y_all = np.r_[yc, yt]; n_cal = len(yc)
D = {a: {"marginal": [], "pac": [], "chow_budget": []} for a in (0.05, 0.10)}
for b in range(500):
    perm = rng.permutation(len(y_all)); ic, it = perm[:n_cal], perm[n_cal:]
    f = platt(raw_all[ic], y_all[ic]); p_c, p_t = f(raw_all[ic]), f(raw_all[it]); y_c, y_t = y_all[ic], y_all[it]
    chow_miss = evaluate(chow_budget(p_t, 5, C0, RHO, p_cal=p_c), y_t, 5, C0)["defaulters_auto_approved"]
    for a in D:
        for kind, delta in (("marginal", None), ("pac", 0.05)):
            cap = conformal_cap(p_c[y_c == 1], a, delta)
            D[a][kind].append(evaluate(cs(p_t, 5, C0, a, RHO, p_cal=p_c, cap=cap), y_t, 5, C0)["defaulters_auto_approved"])
        D[a]["chow_budget"].append(chow_miss)
res["D_validity_500_splits"] = {f"alpha={a}": {k: {"mean": round(float(np.mean(v)), 4), "p95": round(float(np.quantile(v, 0.95)), 4),
                                                   "share_of_splits_above_alpha": round(float(np.mean(np.array(v) > a)), 3)}
                                               for k, v in D[a].items()} for a in D}
res["D_note"] = ("Each split re-fits Platt on 22,500 rows and tests on 37,500. Test miss rates include binomial noise from ~2,500 "
                 "test defaulters, so the PAC exceedance share can sit slightly above delta=0.05.")

# ---------- E: label shift (resample test to change the base rate)
E = {}; i1, i0 = np.where(yt == 1)[0], np.where(yt == 0)[0]
for mult in (0.5, 1.0, 2.0):
    n = len(yt); n1 = int(round(yt.mean() * mult * n)); n0 = n - n1
    idx = np.r_[rng.choice(i1, n1, replace=n1 > len(i1)), rng.choice(i0, n0, replace=n0 > len(i0))]
    ys, ps = yt[idx], pt[idx]
    E[f"base_rate_x{mult}"] = {"observed_default_rate": round(float(ys.mean()), 4), "mean_predicted_pd": round(float(ps.mean()), 4),
                                "ECE": round(ece(ys, ps), 4),
                                "chow_budget": evaluate(chow_budget(ps, 5, C0, RHO), ys, 5, C0),
                                "creditsense": evaluate(cs(ps, 5, C0, 0.10, RHO), ys, 5, C0)}
res["E_label_shift_main_config"] = E

# ---------- F: age bands, global vs per-band cap (main configuration)
bands = [(0, 30, "<30"), (30, 40, "30-39"), (40, 50, "40-49"), (50, 60, "50-59"), (60, 70, "60-69"), (70, 200, "70+")]
F = {}; dg = cs(pt, 5, C0, 0.10, RHO); dgrp = dg.copy()
for lo, hi, lab in bands:
    mc = (age_c >= lo) & (age_c < hi); mt = (age_t >= lo) & (age_t < hi)
    capb = cap_of(0.10, mask=mc)
    dgrp[mt] = cs(pt[mt], 5, C0, 0.10, RHO, cap=capb)
    F[lab] = {"n_test": int(mt.sum()), "calibration_defaulters": int((mc & (yc == 1)).sum()), "cap_band": round(capb, 4),
              "global_cap": evaluate(dg[mt], yt[mt], 5, C0), "per_band_cap": evaluate(dgrp[mt], yt[mt], 5, C0)}
F["_overall"] = {"global_cap": evaluate(dg, yt, 5, C0), "per_band_cap": evaluate(dgrp, yt, 5, C0)}
res["F_age_bands_main_config"] = F
save_json(res, "p2_step12_decision_policy.json")
for k, v in A["r=5,c=0.2"].items(): print(f"{k:48s}", v)
print(json.dumps(res["D_validity_500_splits"]))

# ---------- figures
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
for key, col, lab in (("rho=0.1", "#9BC4CB", "10% budget"), ("rho=0.2", "#2A7F8E", "20% budget"), ("rho=0.3", "#16454D", "30% budget")):
    rows = B[key]["creditsense_by_alpha"]; xs = [100 * float(a) for a in rows]
    ax[0].plot(xs, [rows[a]["cost_per_1000"] for a in rows], "o-", color=col, label=f"CreditSense, {lab}")
    ax[0].axhline(B[key]["chow_same_budget"]["cost_per_1000"], color=col, ls="--", lw=1)
    ax[1].plot(xs, [100 * rows[a]["defaulters_auto_approved"] for a in rows], "o-", color=col, label=lab)
ax[1].plot([0, 32], [0, 32], "k:", lw=1, label="cap = realised")
ax[0].set_xlabel("Cap on defaulters auto-approved, alpha (%)"); ax[0].set_ylabel("Cost per 1,000 applicants (r=5, c=0.2)")
ax[0].set_title("Price of the guarantee (dashed: no cap, same budget)", fontsize=10); ax[0].legend(frameon=False, fontsize=8)
ax[1].set_xlabel("alpha (%)"); ax[1].set_ylabel("% of test defaulters auto-approved"); ax[1].set_title("The cap holds on test", fontsize=10); ax[1].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "p2_price_of_guarantee.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(6.4, 4))
for k, col, lab in (("chow_budget", "#C8641E", "No cap (Chow, 20% budget)"), ("marginal", "#2A7F8E", "CreditSense, marginal"), ("pac", "#16454D", "CreditSense, 95% PAC")):
    ax.hist(100 * np.array(D[0.10][k]), bins=30, alpha=0.65, color=col, label=lab)
ax.axvline(10, color="k", ls="--", lw=1); ax.text(10.3, ax.get_ylim()[1] * 0.92, "alpha = 10%", fontsize=8)
ax.set_xlabel("% of test defaulters auto-approved"); ax.set_ylabel("Random splits (of 500)"); ax.set_title("Guarantee check over 500 re-splits", fontsize=10)
ax.legend(frameon=False, fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(OUT, "p2_guarantee_validity.png"), dpi=150)
print("saved")
