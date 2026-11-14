"""Decision policies shared by the experiments (phase2/p2_step12) and the scoring service (creditsense/serve.py).

All rules act on a *calibrated* probability of default p. Costs are in units of one wrongly declined good borrower:
  approve a defaulter  -> r        (r = cost ratio, an assumption set by the lender)
  decline a good       -> 1
  refer to an analyst  -> c        (per case; the analyst is assumed to decide correctly)

CreditSense triage (the Phase II modified algorithm)
  1. Cost-optimal three-way rule (Chow 1970): pick the action with the lowest expected cost
       approve: r*p     decline: 1-p     refer: c
  2. Risk-appetite cap with a finite-sample guarantee: approve only if p < q_alpha, where q_alpha is the
     k-th smallest calibrated score among calibration-set defaulters. For an exchangeable new defaulter,
     P(auto-approved) <= k/(n1+1) <= alpha. The cap depends only on ranks, so it holds even if calibration
     is imperfect; with k chosen by the Beta rule it holds with probability >= 1-delta over the calibration draw.
  3. Capacity: analysts can review only a share rho of applicants. Refer the cases with the largest referral gain
     g(p) = (cost of the best allowed automatic action) - c, up to the budget. This greedy rule is optimal for a
     single capacity constraint, and g depends only on p, so each applicant gets a decision on its own.
  4. Everything else gets the cheaper allowed automatic action (approve only below the cap).
"""
from __future__ import annotations
import numpy as np
from scipy import stats

APPROVE, REFER, DECLINE = 0, 1, 2
LABELS = {APPROVE: "approve", REFER: "refer", DECLINE: "decline"}


def bayes_threshold(r: float) -> float:
    """Decline iff p >= 1/(1+r) (two-way, no referral)."""
    return 1.0 / (1.0 + r)


def conformal_cap(p_def_cal: np.ndarray, alpha: float, delta: float | None = None) -> float:
    """Lower score bound for defaulters.

    p_def_cal: calibrated scores of calibration-set defaulters.
    delta=None -> marginal guarantee: k = floor(alpha*(n1+1)), E[miss] <= alpha.
    delta=0.05 -> training-conditional (PAC) guarantee: largest k with P(Beta(k, n1-k+1) <= alpha) >= 1-delta,
                  i.e. miss rate <= alpha with probability >= 1-delta over the calibration sample (Vovk 2012).
    Returns +inf-safe threshold: approve only if p < cap. If k == 0 the cap is 0 (nobody can be auto-approved).
    """
    s = np.sort(np.asarray(p_def_cal)); n1 = len(s)
    if delta is None:
        k = int(np.floor(alpha * (n1 + 1)))
    else:
        ks = np.arange(1, n1 + 1)
        ok = stats.beta.cdf(alpha, ks, n1 - ks + 1) >= 1 - delta
        k = int(ks[ok].max()) if ok.any() else 0
    return 0.0 if k == 0 else float(s[k - 1])


def chow_thresholds(r: float, c: float) -> tuple[float, float]:
    """Approve if p <= a, decline if p >= d, else refer (cost-optimal with referral cost c)."""
    t = bayes_threshold(r)
    a = min(c / r, t); d = max(1.0 - c, t)
    return a, d


def decide_two_way(p, r):
    p = np.asarray(p); return np.where(p >= bayes_threshold(r), DECLINE, APPROVE)


def decide_chow(p, r, c):
    p = np.asarray(p); a, d = chow_thresholds(r, c)
    return np.where(p <= a, APPROVE, np.where(p >= d, DECLINE, REFER))


def _auto_cost(p, r, cap):
    """Expected cost of the best automatic action; approval is only allowed below the cap."""
    p = np.asarray(p)
    approve = np.where(p < cap, r * p, np.inf)
    decline = 1 - p
    return np.minimum(approve, decline), np.where(approve <= decline, APPROVE, DECLINE)


def referral_gain_cut(p_cal, r, c, cap=np.inf, rho=None):
    """Referral gain g(p) = (cost of best automatic action) - c. Refer iff g > cut.
    rho=None -> no capacity limit (cut = 0, i.e. Chow's rule). Otherwise cut is the (1-rho) quantile of g on the
    calibration set, so about rho of applicants are referred (never a negative-gain referral)."""
    g, _ = _auto_cost(p_cal, r, cap); g = g - c
    if rho is None: return 0.0
    if rho <= 0: return np.inf
    return max(0.0, float(np.quantile(g, 1 - rho)))


def decide_triage(p, r, c, cap=np.inf, cut=0.0):
    """CreditSense triage: approve only below the cap; refer when the referral gain >= cut; otherwise the cheaper
    automatic action. cap=inf and cut=0 gives Chow's rule; cap=inf with a capacity cut gives a budgeted Chow band."""
    a, act = _auto_cost(p, r, cap)
    g = a - c
    return np.where(g > cut, REFER, act) if np.isfinite(cut) else act


def conformal_symmetric(p, p_cal, y_cal, alpha):
    """Class-conditional split conformal (Phase II H3): refer when the prediction set is not a single label."""
    p = np.asarray(p); p_cal = np.asarray(p_cal); y_cal = np.asarray(y_cal)
    def q_hat(s, a):
        n = len(s); k = min(int(np.ceil((n + 1) * (1 - a))), n); return np.sort(s)[k - 1]
    q1 = q_hat(1 - p_cal[y_cal == 1], alpha); q0 = q_hat(p_cal[y_cal == 0], alpha)
    in1 = (1 - p) <= q1; in0 = p <= q0
    return np.where(in0 & ~in1, APPROVE, np.where(in1 & ~in0, DECLINE, REFER))


def evaluate(decision, y, r, c):
    """Cost per 1,000 applicants (including referral cost), rates and the share of defaulters auto-approved."""
    decision = np.asarray(decision); y = np.asarray(y); n = len(y)
    fn = ((decision == APPROVE) & (y == 1)).sum(); fp = ((decision == DECLINE) & (y == 0)).sum(); nr = (decision == REFER).sum()
    return {"cost_per_1000": round(float((r * fn + fp + c * nr) / n * 1000), 2),
            "approve_rate": round(float((decision == APPROVE).mean()), 4),
            "refer_rate": round(float((decision == REFER).mean()), 4),
            "decline_rate": round(float((decision == DECLINE).mean()), 4),
            "defaulters_auto_approved": round(float(fn / max((y == 1).sum(), 1)), 4),
            "goods_auto_declined": round(float(fp / max((y == 0).sum(), 1)), 4),
            "default_rate_in_approved": round(float(y[decision == APPROVE].mean()) if (decision == APPROVE).any() else 0.0, 4)}
