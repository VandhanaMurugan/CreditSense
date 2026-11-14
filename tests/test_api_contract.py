"""API contract tests: the shapes the UI in ui/ relies on (context pack section 11.2). If one of these fails,
ui/src/api.ts must change with it."""
import io, os, pandas as pd, pytest
from creditsense.serve import DEFAULT_BUNDLE as BUNDLE
pytestmark = pytest.mark.skipif(not os.path.exists(BUNDLE), reason="no bundle: run `make bundle`")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "data", "demo")
INPUTS = ["RevolvingUtilizationOfUnsecuredLines", "age", "NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse",
          "NumberOfTimes90DaysLate", "NumberOfOpenCreditLinesAndLoans", "NumberRealEstateLoansOrLines"]
APP = dict(zip(INPUTS, [0.45, 42, 1, 0, 0, 8, 1]))
RULE_KEYS = {"cost_per_1000", "approve_rate", "refer_rate", "decline_rate", "defaulters_auto_approved", "goods_auto_declined", "default_rate_in_approved"}
REASON_KEYS = {"code", "text", "direction", "contribution_logodds"}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from creditsense import api
    from creditsense.serve import CreditSense
    api._cs = CreditSense(BUNDLE)
    return TestClient(api.app)


def _csv(name): return {"file": (name, open(os.path.join(DEMO, name), "rb"), "text/csv")}


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_version"]


def test_openapi_describes_the_seven_inputs(client):
    props = client.get("/openapi.json").json()["components"]["schemas"]["Applicant"]["properties"]
    assert list(props) == INPUTS and props["age"]["minimum"] == 18 and props["NumberOfTimes90DaysLate"]["type"] == "integer"


def test_score_contract(client):
    r = client.post("/score", json={"applicant": APP})
    assert r.status_code == 200
    b = r.json()
    assert set(b) >= {"pd", "risk_tier", "tier_observed_default_rate", "decision", "decision_rule", "policy", "pd_ranges", "guarantee",
                      "warnings", "reasons", "adverse_action_reasons", "protected_attribute_contribution", "explanation_check",
                      "model_version", "disclaimer"}
    assert 0 <= b["pd"] <= 1 and b["decision"] in b["pd_ranges"] and set(b["policy"]) == {"r", "c", "rho", "alpha", "delta"}
    assert all(set(x) == REASON_KEYS and x["direction"] in ("raises risk", "lowers risk") for x in b["reasons"] + b["adverse_action_reasons"])
    assert set(b["explanation_check"]) == {"base_logodds", "sum_of_contributions", "model_logodds"}
    for lo, hi in b["pd_ranges"].values(): assert 0 <= lo <= hi <= 1


def test_score_is_deterministic(client):
    assert client.post("/score", json={"applicant": APP}).json() == client.post("/score", json={"applicant": APP}).json()


def test_score_honours_the_policy(client):
    strict = client.post("/score", json={"applicant": APP, "policy": {"alpha": 0.02}}).json()
    loose = client.post("/score", json={"applicant": APP, "policy": {"alpha": 0.3}}).json()
    assert strict["pd"] == loose["pd"] and strict["pd_ranges"]["approve"][1] < loose["pd_ranges"]["approve"][1]
    pac = client.post("/score", json={"applicant": APP, "policy": {"delta": 0.05}}).json()
    assert "95% confidence" in pac["guarantee"]


def test_score_rejects_bad_input_with_a_readable_error(client):
    r = client.post("/score", json={"applicant": {**APP, "age": 12}})
    assert r.status_code == 422 and r.json()["detail"][0]["loc"][-1] == "age"
    assert client.post("/score", json={"applicant": {k: v for k, v in APP.items() if k != "age"}}).status_code == 422


def test_sample_applicants_match_the_documented_outcomes(client):
    """Context pack 11.2: A001 approve, A006 refer, A016 and A020 decline, A021 referred with an out-of-range warning."""
    out = pd.read_csv(io.StringIO(client.post("/score_batch", files=_csv("sample_applicants.csv")).text)).set_index("applicant_id")
    assert {"pd", "risk_tier", "decision", "warnings"} <= set(out.columns) and len(out) == 21
    assert out.loc[["A001", "A006", "A016", "A020", "A021"], "decision"].tolist() == ["approve", "refer", "decline", "decline", "refer"]
    assert "outside the range" in out.loc["A021", "warnings"] and out["warnings"].notna().sum() == 1


def test_batch_and_single_scores_agree(client):
    row = pd.read_csv(os.path.join(DEMO, "sample_applicants.csv")).iloc[5]
    one = client.post("/score", json={"applicant": {k: (float(row[k]) if k in INPUTS[:2] else int(row[k])) for k in INPUTS}}).json()
    out = pd.read_csv(io.StringIO(client.post("/score_batch", files=_csv("sample_applicants.csv")).text))
    assert out.loc[5, "pd"] == one["pd"] and out.loc[5, "decision"] == one["decision"]


def test_policy_simulation_contract(client):
    b = client.get("/policy_simulation", params={"r": 5, "c": 0.2, "rho": 0.2, "alpha": 0.1}).json()
    assert list(b["results"]) == ["CreditSense", "Same budget, no guarantee", "Two-way cost threshold"]
    assert all(set(v) == RULE_KEYS for v in b["results"].values()) and b["n_test"] == 37500
    assert b["results"]["CreditSense"]["defaulters_auto_approved"] <= 0.1 + 0.02      # near the cap on this one test set
    for v in b["results"].values(): assert abs(v["approve_rate"] + v["refer_rate"] + v["decline_rate"] - 1) < 1e-3


def test_policy_simulation_unlimited_budget_is_rho_one(client):
    """The UI sends rho = 1 for 'unlimited referrals'; it must behave as no capacity limit (nobody auto-declined at c = 0.2)."""
    b = client.get("/policy_simulation", params={"rho": 1}).json()["results"]["CreditSense"]
    assert b["decline_rate"] == 0 and b["refer_rate"] > 0.4


def test_monitoring_contract(client):
    drift = client.post("/monitoring", files=_csv("batch_drift.csv")).json()
    assert set(drift) == {"n", "psi", "mean_predicted_pd"} and "score" in drift["psi"]
    assert all(v["status"] in ("stable", "watch", "act") for v in drift["psi"].values()) and drift["psi"]["score"]["status"] == "act"
    down = client.post("/monitoring", files=_csv("batch_downturn_with_outcomes.csv")).json()
    assert down["psi"]["score"]["status"] == "stable"                     # PSI misses a downturn ...
    assert down["calibration_in_the_large"] == pytest.approx(down["observed_default_rate"] - down["mean_predicted_pd"], abs=2e-4)
    assert down["calibration_in_the_large"] > 0.02                        # ... the outcome check catches it


def test_model_card_contract(client):
    b = client.get("/model_card").json()
    assert set(b) >= {"version", "built", "data", "features", "test_metrics", "main_policy_on_test", "tiers", "policy_default", "limitations", "disclaimer"}
    assert set(b["test_metrics"]) >= {"AUC", "PR_AUC", "KS", "Brier", "ECE"} and list(b["tiers"]) == ["1", "2", "3", "4", "5"]
    assert b["main_policy_on_test"] == client.get("/policy_simulation").json()["results"]["CreditSense"]
