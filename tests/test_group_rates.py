"""P0.3: /policy_simulation reports decision rates by age band, so the policy owner sees them before choosing a policy."""
import json, os, pytest
from creditsense.serve import DEFAULT_BUNDLE as BUNDLE, GMSC_CSV, AGE_BANDS
pytestmark = pytest.mark.skipif(not (os.path.exists(BUNDLE) and os.path.exists(GMSC_CSV)), reason="needs the bundle and the GMSC dataset")
STEP12 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase2", "outputs", "p2_step12_decision_policy.json")
RATES = ("approve_rate", "refer_rate", "decline_rate", "defaulters_auto_approved", "goods_auto_declined", "cost_per_1000")


@pytest.fixture(scope="module")
def cs():
    from creditsense.serve import CreditSense
    return CreditSense(BUNDLE)


def test_groups_cover_the_test_set(cs):
    sim = cs.policy_simulation()
    g = sim["group_rates"]
    assert g["attribute"] == "age" and g["available"] and list(g["groups"]) == [b[2] for b in AGE_BANDS]
    assert sum(v["n"] for v in g["groups"].values()) == sim["n_test"]
    overall = sim["results"]["CreditSense"]
    for k in ("approve_rate", "refer_rate", "decline_rate"):          # band rates average back to the overall rate
        assert abs(sum(v["n"] * v[k] for v in g["groups"].values()) / sim["n_test"] - overall[k]) < 1e-3


def test_main_policy_matches_step12_evidence(cs):
    """Under the main policy the bands must equal the committed experiment output (global cap), number for number."""
    ref = json.load(open(STEP12))["F_age_bands_main_config"]
    groups = cs.policy_simulation(r=5, c=0.2, rho=0.2, alpha=0.10)["group_rates"]["groups"]
    for band, got in groups.items():
        assert got["n"] == ref[band]["n_test"]
        for k in RATES: assert got[k] == pytest.approx(ref[band]["global_cap"][k], abs=1e-4), (band, k)


def test_groups_follow_the_policy(cs):
    strict = cs.policy_simulation(alpha=0.05)["group_rates"]["groups"]
    loose = cs.policy_simulation(alpha=0.20)["group_rates"]["groups"]
    assert all(strict[b]["approve_rate"] <= loose[b]["approve_rate"] for b in strict)


def test_missing_dataset_is_reported_not_guessed(cs, monkeypatch):
    from creditsense.serve import CreditSense
    monkeypatch.setenv("CREDITSENSE_DATA", "/nonexistent/gmsc.csv")
    g = CreditSense(BUNDLE).policy_simulation()["group_rates"]
    assert g["available"] is False and g["groups"] == {} and "GiveMeSomeCredit" in g["note"]


def test_api_exposes_group_rates(cs):
    from fastapi.testclient import TestClient
    from creditsense import api
    api._cs = cs
    body = TestClient(api.app).get("/policy_simulation", params={"alpha": 0.2}).json()
    assert set(body["group_rates"]["groups"]["70+"]) >= {"n", "observed_default_rate", *RATES}
