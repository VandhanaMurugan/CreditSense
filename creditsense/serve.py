"""CreditSense scoring service: one applicant or a batch -> calibrated PD, risk tier, approve/refer/decline under an explicit
policy, reason codes, warnings. Decision support only: a human owns every adverse decision.

    from creditsense.serve import CreditSense
    cs = CreditSense()                      # loads artifacts/creditsense_bundle.joblib
    cs.score({"RevolvingUtilizationOfUnsecuredLines": 0.45, "age": 42, ...}, policy={"r": 5, "c": 0.2, "rho": 0.2, "alpha": 0.1})
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import joblib
from .features import to_model_frame, RAW_INPUTS
from .policy import conformal_cap, referral_gain_cut, decide_triage, evaluate, LABELS, APPROVE, REFER, DECLINE

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CANDIDATES = [os.path.join(_ROOT, "artifacts", "creditsense_bundle.joblib"),       # freshly built with `make bundle`
               os.path.join(_ROOT, "models", "creditsense_bundle_v2.0.0.joblib")]    # committed release bundle
DEFAULT_BUNDLE = next((p for p in _CANDIDATES if os.path.exists(p)), _CANDIDATES[-1])
# The bundle stores XGBoost as a pickled memory snapshot, which is tied to the platform it was written on (the v2.0.0
# bundle was built on Linux and its snapshot does not open on Windows). For each released bundle, identified by the SHA-256
# of its file, models/ holds the same XGBoost model exported in XGBoost's portable format; it is used only if the snapshot fails.
PORTABLE_XGB = {"c7ec80198aca7727f5c91e7e5e19f98f28f2c8458bd8a8a22a205760988c9716": os.path.join(_ROOT, "models", "creditsense_xgb_v2.0.0.ubj")}
GMSC_CSV = os.path.join(_ROOT, "data", "GiveMeSomeCredit_dataset.csv")
AGE_BANDS = [(0, 30, "<30"), (30, 40, "30-39"), (40, 50, "40-49"), (50, 60, "50-59"), (60, 70, "60-69"), (70, 200, "70+")]   # as in phase2/p2_step12
DISCLAIMER = ("Decision support only. Probabilities are estimated from 2000s-era US bureau-style data (Give Me Some Credit) and "
              "are valid only for applicants who resemble that population. Cost ratios are policy assumptions.")


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def load_bundle(path: str) -> dict:
    """Load a model bundle. If XGBoost cannot read its pickled snapshot on this platform, read the same XGBoost model from
    the portable export registered for exactly this bundle file. The bundle file itself is never changed, and a bundle with
    no registered export fails with an explanation instead of loading some other model."""
    from xgboost.core import Booster, XGBoostError
    try:
        return joblib.load(path)
    except XGBoostError as err:
        portable = PORTABLE_XGB.get(_sha256(path))
        if portable is None or not os.path.exists(portable):
            raise RuntimeError(f"XGBoost could not read the model snapshot inside {path} on this platform ({str(err).strip()[:80]}), "
                               "and no portable export is registered for this bundle. Rebuild it here with `make bundle`.") from err
        native = Booster.__setstate__

        def from_portable(self, state):
            state = dict(state); state.pop("handle", None)
            self.__dict__.update(state)
            fresh = Booster(model_file=portable)
            self.handle, fresh.handle = fresh.handle, None

        Booster.__setstate__ = from_portable
        try:
            bundle = joblib.load(path)
        finally:
            Booster.__setstate__ = native
        bundle["xgboost_loaded_from"] = os.path.basename(portable)
        return bundle


class CreditSense:
    def __init__(self, bundle_path: str | None = None):
        self.b = load_bundle(bundle_path or os.environ.get("CREDITSENSE_BUNDLE", DEFAULT_BUNDLE))
        self.A, self.B = self.b["platt"]
        self._explainer = None
        self._test_age = False          # False = not looked up yet; None = unavailable

    # ---------- core
    @property
    def explainer(self):
        if self._explainer is None:
            from .explain import StackExplainer
            self._explainer = StackExplainer(self.b["stack"], self.b["platt"], self.b["shap_background"])
        return self._explainer

    def pd(self, raw: pd.DataFrame) -> np.ndarray:
        X = to_model_frame(raw, self.b["cleaning"], self.b["features"])
        return 1 / (1 + np.exp(-(self.A * _logit(self.b["stack"].predict_proba(X)[:, 1]) + self.B)))

    def policy(self, r=None, c=None, rho=None, alpha=None, delta=None):
        d = dict(self.b["policy_default"])
        for k, v in (("r", r), ("c", c), ("rho", rho), ("alpha", alpha), ("delta", delta)):
            if v is not None: d[k] = v
        if d["r"] <= 0 or d["c"] < 0: raise ValueError("r must be > 0 and c >= 0")
        pc, yc = self.b["p_cal"], self.b["y_cal"]
        cap = conformal_cap(pc[yc == 1], d["alpha"], d["delta"]) if d["alpha"] else np.inf
        cut = referral_gain_cut(pc, d["r"], d["c"], cap, d["rho"])
        grid = np.linspace(0, 1, 20001); dec = decide_triage(grid, d["r"], d["c"], cap, cut)
        bounds = {LABELS[k]: [round(float(grid[dec == k].min()), 4), round(float(grid[dec == k].max()), 4)] for k in (APPROVE, REFER, DECLINE) if (dec == k).any()}
        return {**d, "cap": cap, "cut": cut, "pd_ranges": bounds}

    def _warnings(self, row: dict) -> list[str]:
        w = []
        for c in RAW_INPUTS:
            v = row.get(c)
            if v is None or (isinstance(v, float) and np.isnan(v)): w.append(f"{c} is missing"); continue
            rg = self.b["ranges"][c]
            if "Past" in c or "Late" in c:
                if v >= 90: w.append(f"{c} = {v} is a data-error code in the source data")
            hi = min(rg["p999"], self.b["cleaning"].get(f"{c}_p995", np.inf))   # above the cleaning cap the model cannot tell values apart
            if v < rg["p001"] or v > hi: w.append(f"{c} = {v} is outside the range the model can score reliably ({rg['p001']:.3g} to {hi:.3g})")
        if row.get("age") is not None and row["age"] < 18: w.append("age below 18")
        return w

    def score(self, applicant: dict, policy: dict | None = None, explain: bool = True, top: int = 4) -> dict:
        pol = self.policy(**(policy or {}))
        missing = [c for c in RAW_INPUTS if c not in applicant]
        if missing: raise ValueError(f"missing inputs: {missing}")
        raw = pd.DataFrame([{**applicant}])
        p = float(self.pd(raw)[0])
        dec = int(decide_triage(np.array([p]), pol["r"], pol["c"], pol["cap"], pol["cut"])[0])
        warnings = self._warnings(applicant); rule = self._rule_text(dec, p, pol)
        if warnings and dec != REFER:
            dec, rule = REFER, "Referred: at least one input is outside the model's experience, so the score may not be reliable."
        tier = int(np.digitize([p], self.b["tiers"])[0] + 1)
        out = {"pd": round(p, 4), "risk_tier": tier, "tier_observed_default_rate": self.b["tier_table"][tier]["observed_default_rate"],
               "decision": LABELS[dec], "decision_rule": rule,
               "policy": {k: pol[k] for k in ("r", "c", "rho", "alpha", "delta")}, "pd_ranges": pol["pd_ranges"],
               "guarantee": (f"At most {pol['alpha']:.0%} of applicants who go on to default are auto-approved under this policy "
                             f"({'with 95% confidence' if pol['delta'] else 'on average'}), provided new applicants resemble the "
                             "calibration population." if pol["alpha"] else "No guarantee set."),
               "warnings": warnings, "model_version": self.b["version"], "disclaimer": DISCLAIMER}
        if explain:
            X = to_model_frame(raw, self.b["cleaning"], self.b["features"])
            phi = self.explainer.shap_logodds(X)[0]
            out["reasons"] = self.explainer.reasons(phi, top=top)                       # for the analyst: both directions
            out["adverse_action_reasons"] = (self.explainer.reasons(phi, top=top, raises_only=True)
                                             if out["decision"] != "approve" else [])        # for a notice: risk-raising only
            age_v = float(phi[self.b["features"].index("age")])
            out["protected_attribute_contribution"] = {"age_logodds": round(age_v, 4),
                "note": "Age is a model input. It is never given as a reason; its weight is reported for governance review."}
            out["explanation_check"] = {"base_logodds": round(float(self.explainer.base_value), 4),
                                        "sum_of_contributions": round(float(phi.sum()), 4),
                                        "model_logodds": round(float(_logit(p)), 4)}
        return out

    @staticmethod
    def _rule_text(dec, p, pol):
        if dec == APPROVE:
            return f"Approve: PD {p:.1%} is below both the cost-based approval limit and the guarantee cap ({pol['cap']:.2%})."
        if dec == DECLINE:
            return f"Decline (analyst confirms): expected cost of approving (r x PD = {pol['r'] * p:.2f}) exceeds that of declining ({1 - p:.2f}), and the case is outside the referral band."
        return "Refer to an analyst: the expected cost of any automatic decision exceeds the referral cost, or the guarantee cap blocks auto-approval."

    def score_batch(self, df: pd.DataFrame, policy: dict | None = None) -> pd.DataFrame:
        pol = self.policy(**(policy or {}))
        p = self.pd(df[[c for c in df.columns]])
        dec = decide_triage(p, pol["r"], pol["c"], pol["cap"], pol["cut"])
        warn = [self._warnings(r) for r in df.to_dict("records")]
        dec = np.where([len(w) > 0 for w in warn], REFER, dec)
        out = df.copy(); out["pd"] = np.round(p, 4); out["risk_tier"] = np.digitize(p, self.b["tiers"]) + 1
        out["decision"] = [LABELS[d] for d in dec]; out["warnings"] = ["; ".join(w) for w in warn]
        return out

    # ---------- portfolio views
    def policy_simulation(self, **policy) -> dict:
        """Evaluate a policy on the held-out test set (37,500 applicants) and compare with reference rules."""
        pol = self.policy(**policy); pt, yt = self.b["p_test"], self.b["y_test"]; r, c = pol["r"], pol["c"]
        rules = {"CreditSense": decide_triage(pt, r, c, pol["cap"], pol["cut"]),
                 "Same budget, no guarantee": decide_triage(pt, r, c, np.inf, referral_gain_cut(self.b["p_cal"], r, c, np.inf, pol["rho"])),
                 "Two-way cost threshold": np.where(pt >= 1 / (1 + r), DECLINE, APPROVE)}
        return {"policy": {k: pol[k] for k in ("r", "c", "rho", "alpha", "delta")}, "pd_ranges": pol["pd_ranges"],
                "n_test": int(len(yt)), "results": {k: evaluate(v, yt, r, c) for k, v in rules.items()},
                "group_rates": self._group_rates(rules["CreditSense"], yt, r, c)}

    def test_age(self):
        """Ages of the held-out test applicants, in the order of the bundle's p_test / y_test.
        The bundle stores scores and outcomes only, so the ages are re-derived from the dataset with the same split as
        phase2/p2_step13 (drop age = 0, stratified 75/25, random_state 42). Returns None if the dataset is not present
        or does not line up with the bundle (checked on the outcomes)."""
        if self._test_age is False:
            self._test_age = None
            path = os.environ.get("CREDITSENSE_DATA", GMSC_CSV)
            if os.path.exists(path):
                from sklearn.model_selection import train_test_split
                raw = pd.read_csv(path, index_col=0); raw = raw[raw["age"] > 0]
                _, test = train_test_split(raw, test_size=0.25, random_state=42, stratify=raw["SeriousDlqin2yrs"])
                if len(test) == len(self.b["y_test"]) and np.array_equal(test["SeriousDlqin2yrs"].values, np.asarray(self.b["y_test"])):
                    self._test_age = test["age"].values
        return self._test_age

    def _group_rates(self, decision, y, r, c) -> dict:
        """Decision rates of the CreditSense rule by age band on the test set, so a policy owner sees who a policy
        approves, refers and declines before choosing it. An outcome audit, not a fairness certificate."""
        age = self.test_age()
        note = ("Age bands on the held-out test set under the CreditSense rule. The cap on defaulters auto-approved is "
                "set for the whole population; it is not guaranteed within each band.")
        if age is None:
            return {"attribute": "age", "available": False, "groups": {},
                    "note": "Group rates need data/GiveMeSomeCredit_dataset.csv next to the bundle; it was not found or does not match."}
        y = np.asarray(y); groups = {}
        for lo, hi, lab in AGE_BANDS:
            m = (age >= lo) & (age < hi)
            groups[lab] = {"n": int(m.sum()), "observed_default_rate": round(float(y[m].mean()), 4), **evaluate(decision[m], y[m], r, c)}
        return {"attribute": "age", "available": True, "groups": groups, "note": note}

    def monitoring(self, df: pd.DataFrame, y: np.ndarray | None = None) -> dict:
        """PSI of the score and each feature against the calibration reference; calibration-in-the-large if labels given."""
        X = to_model_frame(df, self.b["cleaning"], self.b["features"]); p = self.pd(df)
        def psi(vals, key):
            e = np.array(self.b["psi_edges"][key]); ref = np.array(self.b["psi_ref_counts"][key], float)
            cur = np.histogram(vals, bins=e)[0].astype(float)
            ref = np.clip(ref / ref.sum(), 1e-4, None); cur = np.clip(cur / max(cur.sum(), 1), 1e-4, None)
            return float(np.sum((cur - ref) * np.log(cur / ref)))
        rows = {"score": psi(p, "score"), **{f: psi(X[f].values, f) for f in self.b["features"]}}
        status = lambda v: "stable" if v < 0.10 else ("watch" if v < 0.25 else "act")
        out = {"n": int(len(df)), "psi": {k: {"value": round(v, 4), "status": status(v)} for k, v in rows.items()},
               "mean_predicted_pd": round(float(p.mean()), 4)}
        if y is not None:
            y = np.asarray(y); out["observed_default_rate"] = round(float(y.mean()), 4)
            out["calibration_in_the_large"] = round(float(y.mean() - p.mean()), 4)
        return out

    def model_card(self) -> dict:
        b = self.b
        return {"version": b["version"], "built": b["built"], "data": b["data"], "features": b["features"],
                "test_metrics": b["model_card"]["test"], "main_policy_on_test": b["model_card"]["main_policy_test"],
                "tiers": b["tier_table"], "policy_default": b["policy_default"], "intended_use": "Teaching and research prototype "
                "of a consumer-credit decision-support system; not validated for production lending.",
                "limitations": ["No dates in the data: no out-of-time validation", "Income and debt ratio not used (dropped by IG selection)",
                                "Age is a model input (fair-lending constraints apply)", "Cost ratio r and referral cost c are assumptions",
                                "Guarantee assumes new applicants resemble the calibration population; recalibrate on drift",
                                "Outcomes of declined applicants are never observed (selection bias when retraining)"],
                "disclaimer": DISCLAIMER}
