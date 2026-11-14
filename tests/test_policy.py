import numpy as np
from creditsense.policy import (conformal_cap, chow_thresholds, decide_chow, decide_triage, referral_gain_cut,
                                APPROVE, REFER, DECLINE, evaluate)


def test_chow_thresholds():
    a, d = chow_thresholds(5, 0.2)
    assert np.isclose(a, 0.04) and np.isclose(d, 0.8)


def test_triage_without_cap_equals_chow():
    p = np.linspace(0, 1, 1001)
    assert (decide_triage(p, 5, 0.2) == decide_chow(p, 5, 0.2)).all()


def test_cap_blocks_approval():
    p = np.linspace(0, 1, 1001)
    d = decide_triage(p, 5, 0.2, cap=0.02)
    assert not ((d == APPROVE) & (p >= 0.02)).any()


def test_budget_respected():
    rng = np.random.default_rng(0); p_cal = rng.beta(1, 12, 20000); p = rng.beta(1, 12, 20000)
    cut = referral_gain_cut(p_cal, 5, 0.2, cap=0.03, rho=0.2)
    share = (decide_triage(p, 5, 0.2, 0.03, cut) == REFER).mean()
    assert abs(share - 0.2) < 0.02


def test_marginal_guarantee_on_synthetic_data():
    rng = np.random.default_rng(1); misses = []
    for _ in range(300):
        s_cal, s_new = rng.beta(2, 5, 500), rng.beta(2, 5, 5000)       # scores of defaulters, exchangeable
        cap = conformal_cap(s_cal, 0.1); misses.append((s_new < cap).mean())
    assert np.mean(misses) <= 0.105


def test_pac_cap_is_stricter():
    s = np.random.default_rng(2).beta(2, 5, 1500)
    assert conformal_cap(s, 0.1, delta=0.05) <= conformal_cap(s, 0.1)


def test_evaluate_costs():
    y = np.array([1, 0, 1, 0]); d = np.array([APPROVE, DECLINE, REFER, APPROVE])
    e = evaluate(d, y, r=5, c=0.2)
    assert np.isclose(e["cost_per_1000"], (5 + 1 + 0.2) / 4 * 1000)
