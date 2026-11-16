"""Outputs of the real-data experiments (phase2/p2_step17-19). The pass marks were written before the runs; these tests make sure
every reported verdict follows from its own numbers and pass mark, so a 'met' can never be edited in by hand, and that the
with-age side of the no-age comparison is the released model, not a refit."""
import json, os, pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "phase2", "outputs")
def load(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p): pytest.skip(f"{name} not generated (run `make real-data`)")
    return json.load(open(p))


def test_lending_club_verdicts_follow_from_the_numbers():
    r = load("p2_step17_lendingclub_oot.json"); v, m = r["verdicts"], r["pass_marks"]
    auc = lambda n: r["metrics"][n]["test_2014_2015"]["AUC"]
    assert v["P1 ranking"]["met"] == (round(auc("Stack") - auc("Lender sub-grade"), 4) >= m["P1_auc_vs_grade_min"])
    assert v["P2 calibration"]["met"] == (r["metrics"]["Stack"]["test_2014_2015"]["ECE"] <= m["P2_ece_max"])
    assert v["P3 guarantee"]["met"] == (r["guarantee"]["0.1"]["out_of_time_2014_2015"] <= m["P3_miss_max_at_alpha10"])
    gain = r["money"]["Stack"]["vs_approve_all_dollars_per_1000"]["CreditSense alpha 10%"]["mean"]
    assert v["P4 money"]["met"] == (gain >= m["P4_profit_vs_approve_all_min"])
    assert v["P5 stacking"]["met"] == (r["stacking_H1"]["stack_minus_best_single_auc"]["mean"] < m["P5_stack_gain_max"])


def test_lending_club_split_is_by_time_and_money_is_consistent():
    r = load("p2_step17_lendingclub_oot.json"); d = r["data"]
    assert d["train_2010_2012"] + d["calibrate_2013"] + d["test_2014"] + d["test_2015"] == d["rows_used"]
    s = r["money"]["Stack"]; base = s["Approve all (what the lender did)"]
    assert base["approve_rate"] == 1.0 and base["defaulters_auto_approved"] == 1.0
    diff = s["vs_approve_all_dollars_per_1000"]["CreditSense alpha 10%"]["mean"]
    assert abs(diff - (s["CreditSense alpha 10%"]["profit_per_1000_loans"] - base["profit_per_1000_loans"])) <= 2      # rounding
    assert "Added after" in r["exploratory_after_first_run"]["note"]                                                    # exploratory work is labelled as such


def test_elliptic_verdicts_follow_from_the_numbers():
    r = load("p2_step18_elliptic_transaction_risk.json"); v, m, g = r["verdicts"], r["pass_marks"], r["guarantee"]
    assert v["Q1 calibration"]["met"] == (r["metrics"]["Stack"]["test_35_49"]["ECE"] <= m["Q1_ece_max"])
    assert v["Q2 guarantee"]["met"] == (g["steps_35_42"] <= m["Q2_miss_max_steps_35_42"])
    assert v["Q3 stress"]["met"] == (g["breach_steps_after_42"] == g["breach_steps_flagged_by_monitoring"])
    assert v["Q4 stacking"]["met"] == (r["stacking_H1"]["stack_minus_best_single_auc"]["mean"] < m["Q4_stack_gain_max"])
    assert set(g["by_step"]) == {str(s) for s in range(35, 50)}


def test_no_age_comparison_uses_the_released_model_and_its_own_rule():
    r = load("p2_step19_no_age_model.json"); a, n, c, rule = r["with age (v2.0.0)"], r["without age"], r["promotion_checks"], r["rule"]
    step13 = json.load(open(os.path.join(OUT, "p2_step13_bundle_summary.json")))
    released = json.dumps(step13)
    assert str(a["test"]["AUC"]) in released and str(a["main_policy"]["cost_per_1000"]) in released    # with-age side = the released bundle's numbers
    assert c["R1 AUC drop"]["met"] == (round(a["test"]["AUC"] - n["test"]["AUC"], 4) <= rule["R1_auc_drop_max"])
    assert c["R2 Brier rise"]["met"] == (round(n["test"]["Brier"] - a["test"]["Brier"], 4) <= rule["R2_brier_rise_max"])
    assert c["R3 cost rise"]["met"] == (round(n["main_policy"]["cost_per_1000"] / a["main_policy"]["cost_per_1000"] - 1, 4) <= rule["R3_cost_rise_max"])
    assert c["R4 defaulters auto-approved"]["met"] == (n["main_policy"]["defaulters_auto_approved"] <= rule["R4_miss_max"])
    assert r["candidate_for_v2_1_0"] == all(x["met"] for x in c.values())
