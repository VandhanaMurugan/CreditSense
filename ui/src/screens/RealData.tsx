import { Table } from "../components/Table";
import { Caveat, Panel } from "../components/ui";
import { int, num, pct } from "../lib/format";
import { repoJson } from "../lib/repo";
import { useAsync } from "../lib/useAsync";

// Outputs of phase2/p2_step17 (Lending Club, out of time), p2_step18 (Elliptic, crypto transactions) and p2_step19 (no-age model).
type Interval = { mean: number; ci95: [number, number] };
type Verdict = { value: number | Interval | Record<string, number>; pass_mark: string; met: boolean };
interface Scores { AUC: number; Brier: number; ECE: number; mean_predicted: number; observed: number }
interface Valued { profit_per_1000_loans: number; approve_rate: number; refer_rate: number; decline_rate: number; defaulters_auto_approved: number }
interface Step17 {
  data: { rows_used: number; train_2010_2012: number; calibrate_2013: number; test_2014: number; test_2015: number };
  assumptions: { review_cost_dollars: number; analyst_budget: number };
  metrics: Record<string, { test_2014_2015: Scores }>;
  guarantee: Record<string, { cap_pd: number; in_time_2013_mean_over_200_splits: number; out_of_time_2014: number; out_of_time_2015: number }>;
  money_parameters_from_train_years: { measured_cost_ratio_r: number };
  money: Record<string, Record<string, Valued>>;
  exploratory_after_first_run: { analyst_accuracy_needed_to_match_approve_all: number; default_rate_among_referred: number };
  verdicts: Record<string, Verdict>;
}
interface Step18 {
  data: { labelled: number; train_steps_1_29: number; calibrate_steps_30_34: number; test_steps_35_49: number };
  assumptions: { r: number; c: number; rho: number; alpha: number };
  guarantee: { steps_35_42: number; steps_43_49: number; by_step: Record<string, { monitoring_flag: boolean }> };
  verdicts: Record<string, Verdict>;
}
interface Side { test: { AUC: number; Brier: number; ECE: number }; main_policy: { cost_per_1000: number; defaulters_auto_approved: number };
  age_bands: Record<string, { decline_rate: number; defaulters_auto_approved: number }> }
interface Step19 { "with age (v2.0.0)": Side; "without age": Side; promotion_checks: Record<string, { value: number; limit: string; met: boolean }>;
  candidate_for_v2_1_0: boolean; difference_without_minus_with: { AUC: Interval; decisions_changed_share: number } }

const isInterval = (v: unknown): v is Interval => typeof v === "object" && v !== null && "mean" in v && "ci95" in v;
function show(v: Verdict["value"]): string {
  if (typeof v === "number") return num(v, 4);
  if (isInterval(v)) return `${int(v.mean)} (95% interval ${int(v.ci95[0])} to ${int(v.ci95[1])})`;
  return Object.entries(v).map(([k, x]) => `${k.replace(/_/g, " ")}: ${x}`).join("; ");
}
const Met = ({ met }: { met: boolean }) => <span className={met ? "font-semibold text-approve-ink" : "font-semibold text-decline-ink"}>{met ? "Met" : "Not met"}</span>;
const verdictColumns = [
  { head: "Pass mark (written before the run)", cell: ([k, v]: [string, Verdict]) => `${k}: ${v.pass_mark}` },
  { head: "Result", cell: ([, v]: [string, Verdict]) => show(v.value) },
  { head: "Verdict", cell: ([, v]: [string, Verdict]) => <Met met={v.met} /> },
];
const MONEY_RULES = ["Approve all (what the lender did)", "Same budget, no guarantee", "CreditSense alpha 20%", "CreditSense alpha 10%"];

/** Evidence from real, dated data. Each block loads on its own; a missing output file is reported, not replaced by a stored number. */
export function RealDataPanels() {
  const lc = useAsync(() => repoJson<Step17>("phase2/p2_step17_lendingclub_oot.json"), []);
  const el = useAsync(() => repoJson<Step18>("phase2/p2_step18_elliptic_transaction_risk.json"), []);
  const na = useAsync(() => repoJson<Step19>("phase2/p2_step19_no_age_model.json"), []);
  const missing = (what: string) => <p className="text-sm text-muted">{what} output is not in the repository copy. Run <code>make real-data</code>, then <code>npm run sync-data</code>.</p>;
  const stack = lc.data?.money.Stack;
  const flaggedAll = el.data ? Object.values(el.data.guarantee.by_step).every((s) => s.monitoring_flag) : false;

  return (
    <>
      <Panel title="Out-of-time test on real lending data (Lending Club, loans issued 2010 to 2015)">
        {!lc.data || !stack ? missing("The Lending Club") : (
          <>
            <p className="num mb-3 max-w-[90ch] text-sm">
              {int(lc.data.data.rows_used)} real 36-month loans with a final outcome. Models were fitted on {int(lc.data.data.train_2010_2012)} loans issued 2010 to 2012,
              calibrated on {int(lc.data.data.calibrate_2013)} issued in 2013, and tested on {int(lc.data.data.test_2014 + lc.data.data.test_2015)} issued in 2014 and 2015.
              Decisions are valued in dollars actually repaid, with a cost ratio measured on the training years (r = {num(lc.data.money_parameters_from_train_years.measured_cost_ratio_r)}).
            </p>
            <Table caption="Lending Club pass marks and results" rows={Object.entries(lc.data.verdicts)} rowKey={([k]) => k} columns={verdictColumns} />
            <h3 className="mt-4 font-semibold">Dollars per 1,000 loans on the 2014 to 2015 test loans</h3>
            <Table caption="Profit by decision rule on the test years" rows={MONEY_RULES.filter((r) => stack[r])} rowKey={(r) => r} highlight={(r) => r.startsWith("CreditSense alpha 10")}
              columns={[
                { head: "Rule", cell: (r) => r },
                { head: "Profit per 1,000 loans", numeric: true, cell: (r) => `$${int(stack[r].profit_per_1000_loans)}` },
                { head: "Approve", numeric: true, cell: (r) => pct(stack[r].approve_rate) },
                { head: "Refer", numeric: true, cell: (r) => pct(stack[r].refer_rate) },
                { head: "Decline", numeric: true, cell: (r) => pct(stack[r].decline_rate) },
                { head: "Defaulters auto-approved", numeric: true, cell: (r) => pct(stack[r].defaulters_auto_approved) },
              ]} />
            <h3 className="mt-4 font-semibold">Does the cap hold on later loans?</h3>
            <Table caption="Share of defaulters auto-approved, by cap" rows={Object.entries(lc.data.guarantee)} rowKey={([a]) => a}
              columns={[
                { head: "Cap α", cell: ([a]) => pct(Number(a), 0) },
                { head: "Approve below PD", numeric: true, cell: ([, g]) => pct(g.cap_pd, 2) },
                { head: "Same period (2013, random halves)", numeric: true, cell: ([, g]) => pct(g.in_time_2013_mean_over_200_splits, 2) },
                { head: "Loans issued 2014", numeric: true, cell: ([, g]) => pct(g.out_of_time_2014, 2) },
                { head: "Loans issued 2015", numeric: true, cell: ([, g]) => pct(g.out_of_time_2015, 2) },
              ]} />
            <div className="mt-3">
              <Caveat>
                On this portfolio the cap costs money: every loan here was approved and priced for its risk by the lender, so declining loans gives up interest.
                The referral band without a cap earns more than approving everything only if analysts are right at least {pct(lc.data.exploratory_after_first_run.analyst_accuracy_needed_to_match_approve_all)} of
                the time (an analysis added after the first run, not a pass mark). Review cost ${int(lc.data.assumptions.review_cost_dollars)} per case
                and a {pct(lc.data.assumptions.analyst_budget, 0)} analyst budget are assumptions. There are no rejected applicants in the data, and profit ignores the time value of money.
              </Caveat>
            </div>
          </>
        )}
      </Panel>

      <Panel title="Same decision layer on crypto transaction screening (Elliptic data set)">
        {!el.data ? missing("The Elliptic") : (
          <>
            <p className="num mb-3 max-w-[90ch] text-sm">
              {int(el.data.data.labelled)} labelled Bitcoin transactions in 49 time steps: fitted on steps 1 to 29, calibrated on 30 to 34, tested on 35 to 49.
              Allow, review or block replaces approve, refer or decline. Assumed costs: r = {el.data.assumptions.r}, c = {el.data.assumptions.c}, review budget {pct(el.data.assumptions.rho, 0)}, cap {pct(el.data.assumptions.alpha, 0)}.
            </p>
            <Table caption="Elliptic pass marks and results" rows={Object.entries(el.data.verdicts)} rowKey={([k]) => k} columns={verdictColumns} />
            <div className="mt-3">
              <Caveat>
                Illicit transactions allowed without review: {pct(el.data.guarantee.steps_35_42)} in steps 35 to 42, then {pct(el.data.guarantee.steps_43_49)} in steps 43 to 49,
                after the change in illicit activity that the data set's authors describe. The cap does not survive a change in the population.
                {flaggedAll ? " The monitoring flag was raised in every test step, including the calm ones, so it would not have told an analyst when the break happened: the flag rule needs a threshold suited to small batches." : ""}
              </Caveat>
            </div>
          </>
        )}
      </Panel>

      <Panel title="A model without age (candidate, not released)">
        {!na.data ? missing("The no-age experiment") : (
          <>
            <Table caption="Released model against the same model without age" rows={(["with age (v2.0.0)", "without age"] as const).map((k) => [k, na.data![k]] as const)} rowKey={([k]) => k}
              columns={[
                { head: "Model", cell: ([k]) => k },
                { head: "AUC", numeric: true, cell: ([, s]) => num(s.test.AUC, 4) },
                { head: "Brier", numeric: true, cell: ([, s]) => num(s.test.Brier, 4) },
                { head: "ECE", numeric: true, cell: ([, s]) => num(s.test.ECE, 4) },
                { head: "Cost per 1,000", numeric: true, cell: ([, s]) => num(s.main_policy.cost_per_1000, 1) },
                { head: "Defaulters auto-approved", numeric: true, cell: ([, s]) => pct(s.main_policy.defaulters_auto_approved) },
                { head: "Declined, under 30", numeric: true, cell: ([, s]) => pct(s.age_bands["<30"].decline_rate) },
                { head: "Defaulters auto-approved, 70+", numeric: true, cell: ([, s]) => pct(s.age_bands["70+"].defaulters_auto_approved) },
              ]} />
            <h3 className="mt-4 font-semibold">Promotion rule (written before the run)</h3>
            <Table caption="Promotion checks for the no-age model" rows={Object.entries(na.data.promotion_checks)} rowKey={([k]) => k}
              columns={[
                { head: "Check", cell: ([k, c]) => `${k} ${c.limit}` },
                { head: "Result", numeric: true, cell: ([, c]) => num(c.value, 4) },
                { head: "Verdict", cell: ([, c]) => <Met met={c.met} /> },
              ]} />
            <p className="num mt-2 text-sm text-muted">
              {na.data.candidate_for_v2_1_0 ? "All checks are met, so the no-age model is a candidate for version 2.1.0." : "Not all checks are met, so the no-age model is not a candidate."}{" "}
              The released bundle is unchanged. Removing age changes {pct(na.data.difference_without_minus_with.decisions_changed_share)} of decisions and lowers AUC
              by {num(-na.data.difference_without_minus_with.AUC.mean, 4)} (95% interval {num(-na.data.difference_without_minus_with.AUC.ci95[1], 4)} to {num(-na.data.difference_without_minus_with.AUC.ci95[0], 4)}).
              Age-band differences shrink but do not disappear, because other inputs carry age-related information.
            </p>
          </>
        )}
      </Panel>
    </>
  );
}
