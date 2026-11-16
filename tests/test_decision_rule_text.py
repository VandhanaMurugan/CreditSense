"""The sentence that explains a decision must agree with the numbers it prints.
Under the main policy, applicants between the referral band and the cost threshold are declined because of the guarantee cap
and the analyst budget, not because approving costs more. The sentence used to claim the opposite."""
import os, re, pandas as pd, pytest
from creditsense.serve import CreditSense, DEFAULT_BUNDLE as BUNDLE
pytestmark = pytest.mark.skipif(not os.path.exists(BUNDLE), reason="needs the bundle")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def scored():
    cs = CreditSense(); df = pd.read_csv(os.path.join(ROOT, "data", "demo", "sample_applicants.csv"))
    return [(row[df.columns[0]], cs.score({k: row[k] for k in cs.b["raw_inputs"]}, explain=False)) for _, row in df.iterrows()]


def test_cap_forced_decline_says_so(scored):
    r = dict(scored)["A016"]                                   # PD about 10.9%: above the cap, below the cost threshold 1/(1+r)
    assert r["decision"] == "decline" and r["policy"]["r"] * r["pd"] < 1 - r["pd"]
    assert "guarantee cap" in r["decision_rule"] and "price of the guarantee" in r["decision_rule"]
    assert "exceeds" not in r["decision_rule"]


def test_cost_based_decline_keeps_the_cost_sentence(scored):
    r = dict(scored)["A020"]
    assert r["decision"] == "decline" and r["policy"]["r"] * r["pd"] > 1 - r["pd"] and "exceeds" in r["decision_rule"]


def test_no_sentence_contradicts_its_own_numbers(scored):
    for name, r in scored:
        m = re.search(r"approving \(r x PD = ([\d.]+)\) exceeds that of declining \(([\d.]+)\)", r["decision_rule"])
        if m: assert float(m.group(1)) > float(m.group(2)), name
        m = re.search(r"approving \(r x PD = ([\d.]+)\) would be cheaper than declining \(([\d.]+)\)", r["decision_rule"])
        if m: assert float(m.group(1)) <= float(m.group(2)), name
