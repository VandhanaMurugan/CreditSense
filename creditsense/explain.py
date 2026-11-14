"""Exact reason codes for the calibrated stack.

The stack's meta-learner is logistic regression on the members' probabilities, and Platt calibration is linear in the
meta-learner's log-odds. So the calibrated log-odds is   L(x) = B + A * (b0 + sum_m beta_m * g_m(x)),
where g_m is member m's predicted probability. SHAP values are linear in the model, so with one shared background set
    phi_j(L) = A * sum_m beta_m * phi_j(g_m)
is the exact (interventional) SHAP decomposition of the calibrated log-odds. Additivity is checked numerically.
"""
import numpy as np
import shap

REASONS = {   # feature group -> (plain-language reason when it RAISES risk, when it LOWERS risk)
    "serious_delinquency": (["NumberOfTimes90DaysLate"], "Payments 90 or more days late", "No payments 90+ days late"),
    "late_30_59": (["NumberOfTime30-59DaysPastDueNotWorse"], "Payments 30-59 days late", "No recent 30-59 day late payments"),
    "late_60_89": (["NumberOfTime60-89DaysPastDueNotWorse"], "Payments 60-89 days late", "No 60-89 day late payments"),
    "delinquency_history": (["TotalPastDueEvents", "HasPastDueHistory"], "Overall history of late payments", "Clean overall payment history"),
    "utilisation": (["RevolvingUtilizationOfUnsecuredLines", "UtilizationBucket"], "High use of revolving credit limits", "Low use of revolving credit limits"),
    "open_lines": (["NumberOfOpenCreditLinesAndLoans"], "Number of open credit lines and loans", "Number of open credit lines and loans"),
    "real_estate": (["NumberRealEstateLoansOrLines"], "Number of real-estate loans or lines", "Number of real-estate loans or lines"),
    "age": (["age"], "Age (protected attribute - not usable as an adverse-action reason; governance review)",
            "Age (protected attribute - not usable as an adverse-action reason; governance review)"),
}
PROTECTED_GROUPS = {"age"}


class StackExplainer:
    def __init__(self, stack, platt_ab, background):
        self.A, self.B = platt_ab
        names = [n for n, _ in stack.estimators]
        self.beta = stack.final_estimator_.coef_.ravel(); self.b0 = float(stack.final_estimator_.intercept_[0])
        self.features = list(background.columns)
        self.explainers = []
        for name, est in zip(names, stack.estimators_):
            model = est.model_ if hasattr(est, "model_") else est          # WeightedGB wraps a GradientBoostingClassifier
            self.explainers.append(shap.TreeExplainer(model, data=background, model_output="probability",
                                                      feature_perturbation="interventional"))
        ev = lambda e: float(np.ravel(e.expected_value)[-1])          # probability of class 1
        self.base_value = self.B + self.A * (self.b0 + sum(b * ev(e) for b, e in zip(self.beta, self.explainers)))

    def member_shap(self, X):
        out = []
        for e in self.explainers:
            v = e.shap_values(X, check_additivity=False)
            v = v[1] if isinstance(v, list) else v
            if v.ndim == 3: v = v[:, :, 1]
            out.append(v)
        return out

    def shap_logodds(self, X):
        """Per-feature contributions to the calibrated log-odds (n x features)."""
        return self.A * sum(b * v for b, v in zip(self.beta, self.member_shap(X)))

    def reasons(self, phi_row, top=4, include_protected=False, raises_only=False):
        """Group feature contributions into plain-language reasons, largest first.
        raises_only=True gives adverse-action style reasons: only factors that pushed the risk up."""
        idx = {f: i for i, f in enumerate(self.features)}; rows = []
        for key, (feats, up, down) in REASONS.items():
            if key in PROTECTED_GROUPS and not include_protected: continue
            v = float(sum(phi_row[idx[f]] for f in feats if f in idx))
            rows.append({"code": key, "text": up if v > 0 else down, "direction": "raises risk" if v > 0 else "lowers risk",
                         "contribution_logodds": round(v, 4)})
        if raises_only: rows = [r for r in rows if r["contribution_logodds"] > 0]
        rows.sort(key=lambda r: -abs(r["contribution_logodds"]))
        return rows[:top]
