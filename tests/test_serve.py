import os, numpy as np, pytest
BUNDLE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "creditsense_bundle.joblib")
pytestmark = pytest.mark.skipif(not os.path.exists(BUNDLE), reason="build the bundle first: make bundle")

APP = {"RevolvingUtilizationOfUnsecuredLines": 0.45, "age": 42, "NumberOfTime30-59DaysPastDueNotWorse": 1,
       "NumberOfTime60-89DaysPastDueNotWorse": 0, "NumberOfTimes90DaysLate": 0, "NumberOfOpenCreditLinesAndLoans": 8,
       "NumberRealEstateLoansOrLines": 1}


@pytest.fixture(scope="module")
def cs():
    from creditsense.serve import CreditSense
    return CreditSense(BUNDLE)


def test_schema_and_determinism(cs):
    a, b = cs.score(APP), cs.score(APP)
    assert a["pd"] == b["pd"] and a["decision"] in {"approve", "refer", "decline"}
    assert 1 <= a["risk_tier"] <= 5 and len(a["reasons"]) == 4


def test_explanation_is_exact(cs):
    chk = cs.score(APP)["explanation_check"]
    assert abs(chk["base_logodds"] + chk["sum_of_contributions"] - chk["model_logodds"]) < 1e-3


def test_age_never_a_reason(cs):
    out = cs.score(APP)
    assert all(r["code"] != "age" for r in out["reasons"] + out["adverse_action_reasons"])


def test_adverse_action_reasons_raise_risk(cs):
    out = cs.score({**APP, "NumberOfTimes90DaysLate": 2})
    assert out["decision"] != "approve" and all(r["direction"] == "raises risk" for r in out["adverse_action_reasons"])


def test_out_of_range_is_referred(cs):
    out = cs.score({**APP, "RevolvingUtilizationOfUnsecuredLines": 5000.0})
    assert out["decision"] == "refer" and out["warnings"]


def test_risk_monotone_in_serious_delinquency(cs):
    lo = cs.score({**APP, "NumberOfTimes90DaysLate": 0}, explain=False)["pd"]
    hi = cs.score({**APP, "NumberOfTimes90DaysLate": 3}, explain=False)["pd"]
    assert hi > lo


def test_api(cs):
    from fastapi.testclient import TestClient
    from creditsense import api
    api._cs = cs; c = TestClient(api.app)
    r = c.post("/score", json={"applicant": APP, "policy": {"r": 5, "c": 0.2, "rho": 0.2, "alpha": 0.1}})
    assert r.status_code == 200 and "pd" in r.json()
    assert c.get("/policy_simulation").status_code == 200
