import { api } from "../api";
import { DecisionLegend, DecisionMixBar } from "../components/RuleBars";
import { Table } from "../components/Table";
import { Caveat, ErrorNote, Loading, PageHeader, Panel, Stat } from "../components/ui";
import { useModelCard } from "../lib/context";
import { AGE_BANDS, evidence, pick } from "../lib/evidence";
import { int, num, pct } from "../lib/format";
import { describePolicy, usePolicy } from "../lib/policy";
import { useAsync } from "../lib/useAsync";

const TRUST = ["High", "High now, decays", "Medium", "High on average", "Partial", "Limited", "Do not claim"] as const;
type Trust = (typeof TRUST)[number];
interface Row { output: string; evidence: string; trust: string; distrust: string; rating: Trust }
const range = (v: number[]) => `${pct(Math.min(...v), 0)} to ${pct(Math.max(...v), 0)}`;

function useMatrix(): { rows?: Row[]; error?: string } {
  const card = useModelCard();
  const all = useAsync(async () => ({
    s3: await evidence.step3(), s9: await evidence.step9(), s11: await evidence.step11(), s12: await evidence.step12(),
    s14: await evidence.step14(), s15: await evidence.step15(),
  }), []);
  if (!all.data) return { error: all.error };
  const { s3, s9, s11, s12, s14, s15 } = all.data;
  const m = card.test_metrics;
  const ci = pick(s3.models, /^Stack C/).AUC_CI95;
  const tiers = Object.values(card.tiers);
  const s1 = pick(s15, /^S1/), s2 = pick(s15, /^S2/);
  const c3 = pick(s12.A_main, /^r=5,c=0\.3$/);
  const csC3 = pick(c3, /^CreditSense alpha=0\.10, 20% budget$/).cost_per_1000, p1C3 = pick(c3, /^Phase I rule/).cost_per_1000;
  const val = pick(s12.D_validity_500_splits, /^alpha=0\.1$/).marginal;
  const old = s12.F_age_bands_main_config["70+"].global_cap.defaulters_auto_approved;
  const pvals = Object.entries(s9).map(([name, d]) => `${name} p = ${num(pick(d.tests_AUC_nadeau_bengio, new RegExp(`^Stack_trees - ${d.best_single_tree}$`)).p_value)}`);
  const other = ["German", "Taiwan"].map((d) => `${d} ${num(s11[d]["P0 leak-safe"].Stack.mean_AUC)}`).join(", ");
  return { rows: [
    { output: "Ranking (AUC)", evidence: `AUC ${num(m.AUC, 4)}; bootstrap 95% interval ${num(ci[0], 4)} to ${num(ci[1], 4)}`,
      trust: "Applicants resemble the Give Me Some Credit population", distrust: `Other products or countries (leak-safe stack AUC: ${other})`, rating: "High" },
    { output: "PD level", evidence: `ECE ${num(m.ECE, 4)}; Brier ${num(m.Brier, 4)}`, trust: "The default rate is stable and the model is recalibrated regularly",
      distrust: `A downturn: predicted ${pct(s2.mean_predicted_pd)} against observed ${pct(s2.observed_default_rate)} until recalibrated`, rating: "High now, decays" },
    { output: "Risk tier", evidence: `Observed default rises tier by tier, ${pct(tiers[0].observed_default_rate)} to ${pct(tiers[tiers.length - 1].observed_default_rate)}`,
      trust: "Used as a band", distrust: "An applicant sits near a cut-off", rating: "High" },
    { output: "Decision", evidence: `Tested on ${Object.keys(s12.A_main).length} combinations of cost ratio and referral cost`, trust: "The costs and analyst budget reflect the lender's reality",
      distrust: `The costs are wrong: at c = 0.3 CreditSense costs ${num(csC3, 0)} against ${num(p1C3, 0)} for the Phase I rule`, rating: "Medium" },
    { output: "Guarantee", evidence: `Mean ${pct(val.mean, 2)} of defaulters auto-approved over 500 re-splits (95th percentile ${pct(val.p95)})`,
      trust: "Applicants are exchangeable with the calibration set, judged over many batches", distrust: `Per group (age 70+: ${pct(old)}), per batch, or after the population changes`, rating: "High on average" },
    { output: "Reason codes", evidence: `Exact: largest additivity error ${s14.additivity_abs_error.max.toExponential(1)}; top reason stable in ${range(Object.values(s14.stability_top1_agreement))} of cases`,
      trust: "Explaining this model's score", distrust: "Read as causes of default", rating: "High" },
    { output: "Drift alarm", evidence: `Score PSI ${num(s1.psi_score.value)} (${s1.psi_score.status}) when the applicant mix shifts`, trust: "The mix of applicants changes",
      distrust: `A downturn: score PSI ${num(s2.psi_score.value, 3)} (${s2.psi_score.status}) while defaults doubled`, rating: "Partial" },
    { output: "“Stacking is better”", evidence: `Stack minus best single tree is not significant: ${pvals.join("; ")}`, trust: "Never", distrust: "Always", rating: "Do not claim" },
    { output: "Fairness", evidence: "Age audit only", trust: "Read as an audit of outcomes by age", distrust: "Read as a fairness certificate: the data has no race, sex or other protected attributes", rating: "Limited" },
  ] };
}

export default function FairnessScreen() {
  const card = useModelCard();
  const { policy } = usePolicy();
  const s12 = useAsync(evidence.step12, []);
  const s8 = useAsync(evidence.step8, []);
  const s14 = useAsync(evidence.step14, []);
  const live = useAsync(() => api.simulate(policy), [policy]);
  const matrix = useMatrix();
  const bands = s12.data ? AGE_BANDS(s12.data.F_age_bands_main_config) : [];
  const overall = s12.data?.F_age_bands_main_config._overall;
  const oldest = bands[bands.length - 1], youngest = bands[0];

  return (
    <>
      <PageHeader title="Fairness and reliability"
        lead="What the age audit found, how far each output can be trusted, and the model card. This is an audit of outcomes by age. It is not a fairness certificate." />
      <div className="space-y-4">
        <Panel title="Age bands under the main policy: one cap for everyone, or one cap per band" aside={<DecisionLegend />}>
          {s12.error && <ErrorNote message="Could not load the age audit." hint={s12.error} />}
          {!s12.data && !s12.error && <Loading what="the age audit" />}
          {s12.data && overall && (
            <>
              <Table caption="Age bands, global cap against per-band cap" rows={bands} rowKey={([b]) => b}
                columns={[
                  { head: "Age band", cell: ([b]) => b },
                  { head: "Test applicants", numeric: true, cell: ([, v]) => int(v.n_test) },
                  { head: "Global cap: decisions", cell: ([, v]) => <div className="min-w-[11rem]"><DecisionMixBar result={v.global_cap} height="h-6" /></div> },
                  { head: "Declined (global)", numeric: true, cell: ([, v]) => pct(v.global_cap.decline_rate) },
                  { head: "Defaulters auto-approved (global)", numeric: true, cell: ([, v]) => pct(v.global_cap.defaulters_auto_approved) },
                  { head: "Per-band cap: decisions", cell: ([, v]) => <div className="min-w-[11rem]"><DecisionMixBar result={v.per_band_cap} height="h-6" /></div> },
                  { head: "Declined (per band)", numeric: true, cell: ([, v]) => pct(v.per_band_cap.decline_rate) },
                  { head: "Defaulters auto-approved (per band)", numeric: true, cell: ([, v]) => pct(v.per_band_cap.defaulters_auto_approved) },
                ]} />
              <p className="num mt-2 text-sm text-muted">
                All applicants: global cap {pct(overall.global_cap.defaulters_auto_approved)} of defaulters auto-approved at cost {num(overall.global_cap.cost_per_1000, 1)} per 1,000;
                per-band cap {pct(overall.per_band_cap.defaulters_auto_approved)} at cost {num(overall.per_band_cap.cost_per_1000, 1)}. Main policy, costs assumed.
              </p>
              <div className="mt-3 space-y-2">
                <p className="max-w-[80ch]">
                  The cap holds for the population as a whole, not inside each band. With one global cap, {pct(oldest[1].global_cap.defaulters_auto_approved)} of
                  defaulters aged {oldest[0]} are auto-approved against {pct(youngest[1].global_cap.defaulters_auto_approved)} of those aged {youngest[0]}. A
                  separate cap per band brings the {oldest[0]} figure to {pct(oldest[1].per_band_cap.defaulters_auto_approved)}, but their declines rise
                  from {pct(oldest[1].global_cap.decline_rate)} to {pct(oldest[1].per_band_cap.decline_rate)}. The global cap also declines{" "}
                  {pct(youngest[1].global_cap.decline_rate)} of applicants aged {youngest[0]}.
                </p>
                <Caveat>
                  Legal note. The per-band cap is reported, not deployed. Treating older applicants more strictly because of their age would assign
                  them a negative factor, which US Regulation B,{" "}
                  <a className="text-accent underline" href="https://www.consumerfinance.gov/rules-policy/regulations/1002/6/" target="_blank" rel="noreferrer">12 CFR 1002.6(b)(2)</a>,
                  does not allow in an empirically derived credit system.
                  Calibration and equal error rates across groups cannot both hold when default rates differ (Kleinberg et al.; Chouldechova).
                  Other jurisdictions have their own rules; in India the RBI framework applies.
                </Caveat>
              </div>
            </>
          )}
        </Panel>

        <div className="grid gap-4 lg:grid-cols-2">
          <Panel title="Calibration by age band">
            {s8.data && (
              <Table caption="Calibration and ranking inside each age band" rows={Object.entries(s8.data.bands)} rowKey={([b]) => b}
                columns={[
                  { head: "Age band", cell: ([b]) => b },
                  { head: "Default rate", numeric: true, cell: ([, v]) => pct(v.default_rate) },
                  { head: "Mean predicted PD", numeric: true, cell: ([, v]) => pct(v.mean_predicted_pd) },
                  { head: "ECE in band", numeric: true, cell: ([, v]) => num(v.ECE_in_band, 4) },
                  { head: "AUC", numeric: true, cell: ([, v]) => num(v.AUC, 3) },
                ]} />
            )}
            <p className="mt-2 text-sm text-muted">Probabilities stay close to observed default rates in every band. The gap in the table above comes from different default rates by age, not from miscalibration.</p>
          </Panel>
          <Panel title="Age inside the score">
            {s14.data && (
              <dl className="grid gap-4 sm:grid-cols-2">
                <Stat label="Adverse cases where age is a top-4 driver" value={pct(s14.data.age_among_top4_drivers_share, 0)} note={`of ${int(s14.data.n_explained)} explained referrals and declines`} />
                <Stat label="Age shown as a reason" value="Never" note="Excluded from reason codes by design (tested)" />
              </dl>
            )}
            <p className="mt-3 text-sm">Age is a model input and it moves many scores. It is never given as a reason, and its contribution is shown for every decision under the governance note. A model without age is on the roadmap and has not been run yet.</p>
          </Panel>
        </div>

        <Panel title="Decisions by age band under the policy now in use" aside={describePolicy(policy)}>
          {live.error && <ErrorNote message="The service could not evaluate the policy." hint={live.error} />}
          {live.data?.group_rates && Object.keys(live.data.group_rates.groups).length > 0 ? (
            <Table caption="Live group rates" rows={Object.entries(live.data.group_rates.groups)} rowKey={([b]) => b}
              columns={[
                { head: "Age band", cell: ([b]) => b },
                { head: "Decisions", cell: ([, v]) => <div className="min-w-[14rem]"><DecisionMixBar result={v} height="h-6" /></div> },
                { head: "Approved", numeric: true, cell: ([, v]) => pct(v.approve_rate) },
                { head: "Referred", numeric: true, cell: ([, v]) => pct(v.refer_rate) },
                { head: "Declined", numeric: true, cell: ([, v]) => pct(v.decline_rate) },
                { head: "Defaulters auto-approved", numeric: true, cell: ([, v]) => pct(v.defaulters_auto_approved) },
              ]} />
          ) : live.data ? <p className="text-muted">{live.data.group_rates?.note ?? "Group rates are not available from this service version."}</p> : null}
        </Panel>

        <Panel title="Reliability matrix: how far each output can be trusted">
          {matrix.error && <ErrorNote message="Could not build the reliability matrix from the experiment outputs." hint={matrix.error} />}
          {matrix.rows && (
            <Table caption="Reliability matrix" rows={matrix.rows} rowKey={(r) => r.output}
              columns={[
                { head: "Output", cell: (r) => r.output },
                { head: "Evidence", cell: (r) => <span className="num">{r.evidence}</span> },
                { head: "Trust it when", cell: (r) => r.trust },
                { head: "Do not trust it when", cell: (r) => <span className="num">{r.distrust}</span> },
                { head: "Rating", cell: (r) => <span className="font-semibold whitespace-nowrap">{r.rating}</span> },
              ]} />
          )}
        </Panel>

        <Panel title="Model card" aside={`Version ${card.version}, built ${card.built}`}>
          <dl className="grid grid-cols-2 gap-4 sm:grid-cols-5">
            {(["AUC", "PR_AUC", "KS", "Brier", "ECE"] as const).map((k) => <Stat key={k} label={k.replace("_", "-")} value={num(card.test_metrics[k], 4)} />)}
          </dl>
          <p className="mt-2 text-sm text-muted">
            AUC, PR-AUC and KS describe ranking. Brier and ECE describe probability quality. Held-out test set, {int(Number(card.data.test_rows))} applicants.
          </p>
          <div className="mt-4 grid gap-4 lg:grid-cols-2">
            <div>
              <h3 className="mb-1 font-semibold">Risk tiers</h3>
              <Table caption="Risk tiers" rows={Object.entries(card.tiers)} rowKey={([t]) => t}
                columns={[
                  { head: "Tier", cell: ([t]) => t },
                  { head: "PD range", numeric: true, cell: ([, v]) => `${pct(v.pd_range[0], 0)} to ${pct(v.pd_range[1], 0)}` },
                  { head: "Share of applicants", numeric: true, cell: ([, v]) => pct(v.share_of_applicants) },
                  { head: "Mean predicted PD", numeric: true, cell: ([, v]) => pct(v.mean_predicted_pd) },
                  { head: "Observed default", numeric: true, cell: ([, v]) => pct(v.observed_default_rate) },
                ]} />
            </div>
            <div className="text-sm">
              <h3 className="mb-1 text-base font-semibold">Data and inputs</h3>
              <p className="num">{String(card.data.source)}: {int(Number(card.data.fit_rows))} rows to fit, {int(Number(card.data.calibration_rows))} to calibrate, {int(Number(card.data.test_rows))} to test.</p>
              <p className="mt-1">Model features: {card.features.join(", ")}.</p>
              <h3 className="mt-3 mb-1 text-base font-semibold">Intended use</h3>
              <p>{card.intended_use}</p>
              <h3 className="mt-3 mb-1 text-base font-semibold">Limitations</h3>
              <ul className="list-disc pl-5">{card.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
            </div>
          </div>
        </Panel>
      </div>
    </>
  );
}
