import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { RealDataPanels } from "../screens/RealData";
import { mockFetch } from "./fixtures";

const valued = (profit: number, miss: number) => ({ profit_per_1000_loans: profit, approve_rate: 0.5, refer_rate: 0.2, decline_rate: 0.3, defaulters_auto_approved: miss });
const step17 = {
  data: { rows_used: 1000, train_2010_2012: 100, calibrate_2013: 200, test_2014: 300, test_2015: 400 },
  assumptions: { review_cost_dollars: 25, analyst_budget: 0.2 },
  metrics: {}, money_parameters_from_train_years: { measured_cost_ratio_r: 2.5 },
  guarantee: { "0.1": { cap_pd: 0.07, in_time_2013_mean_over_200_splits: 0.1, out_of_time_2014: 0.101, out_of_time_2015: 0.114 } },
  money: { Stack: { "Approve all (what the lender did)": valued(900000, 1), "Same budget, no guarantee": valued(1100000, 0.65),
    "CreditSense alpha 20%": valued(700000, 0.21), "CreditSense alpha 10%": valued(555000, 0.11) } },
  exploratory_after_first_run: { analyst_accuracy_needed_to_match_approve_all: 0.631, default_rate_among_referred: 0.23 },
  verdicts: { "P3 guarantee": { value: 0.1097, pass_mark: "<= 0.12", met: true },
    "P4 money": { value: { mean: -355000, ci95: [-364000, -346000] }, pass_mark: "mean >= 0", met: false } },
};

afterEach(() => vi.unstubAllGlobals());

describe("Real-data evidence panels", () => {
  it("show each pass mark with its verdict, including the ones that were not met", async () => {
    mockFetch({ "p2_step17_lendingclub_oot.json": step17 });
    render(<RealDataPanels />);
    expect(await screen.findByText("P4 money: mean >= 0")).toBeInTheDocument();
    expect(screen.getByText("Not met")).toBeInTheDocument();
    expect(screen.getByText("Met")).toBeInTheDocument();
    expect(screen.getByText("$555,000")).toBeInTheDocument();
    expect(screen.getByText(/analysts are right at least 63\.1%/)).toBeInTheDocument();
  });
  it("say which output is missing instead of showing numbers", async () => {
    mockFetch({});
    render(<RealDataPanels />);
    expect(await screen.findByText(/The Elliptic output is not in the repository copy/)).toBeInTheDocument();
    expect(screen.getByText(/The no-age experiment output is not in the repository copy/)).toBeInTheDocument();
  });
});
