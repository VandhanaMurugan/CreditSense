import { useState } from "react";
import { api, type MonitoringResponse } from "../api";
import { Table } from "../components/Table";
import { Caveat, ErrorNote, Loading, PageHeader, Panel, Stat } from "../components/ui";
import { int, num, pct, signed } from "../lib/format";
import { repoJson, repoText, repoUrl } from "../lib/repo";
import { useAsync } from "../lib/useAsync";

// Threshold for the recalibration banner, from the build brief (context pack 11.3). PSI status words come from the API.
const RECALIBRATE_ABOVE = 0.02;
const STATUS = {
  stable: { mark: "●", word: "Stable", text: "The mix looks like the reference population.", tone: "text-approve-ink bg-approve-tint" },
  watch: { mark: "▲", word: "Watch", text: "A noticeable shift. Check again on the next batch.", tone: "text-refer-ink bg-refer-tint" },
  act: { mark: "■", word: "Act", text: "A large shift. Review the model before relying on its scores.", tone: "text-decline-ink bg-decline-tint" },
} as const;
const DEMO = [
  { file: "batch_drift.csv", label: "Demo: drifted applicant mix" },
  { file: "batch_downturn_with_outcomes.csv", label: "Demo: downturn, with outcomes" },
];

interface Side { mean_pd: number; ECE: number; Brier: number; main_policy: { cost_per_1000: number; defaulters_auto_approved: number } }
interface Step15 {
  learning_step_S2: { recent_labelled_outcomes_used: number; evaluated_on: number; observed_default_rate_holdout: number;
    champion: Side; challenger_recalibrated: Side; promote: boolean; note: string };
  [scenario: string]: unknown;
}
interface Scenario { psi_score: { value: number; status: keyof typeof STATUS }; worst_feature: Record<string, { value: number; status: keyof typeof STATUS }>;
  observed_default_rate: number; mean_predicted_pd: number; ECE: number }

function StatusTag({ status }: { status: keyof typeof STATUS }) {
  const s = STATUS[status];
  return <span className={`inline-flex items-center gap-1 px-1.5 font-semibold ${s.tone}`}><span aria-hidden="true">{s.mark}</span>{s.word}</span>;
}

function Report({ name, m }: { name: string; m: MonitoringResponse }) {
  const rows = Object.entries(m.psi);
  const citl = m.calibration_in_the_large;
  const worst = rows.reduce((a, b) => (b[1].value > a[1].value ? b : a));
  return (
    <div className="space-y-4" data-testid="monitoring-report">
      <Panel title={`Batch: ${name}`} aside={`${int(m.n)} applicants`}>
        <dl className="grid gap-4 sm:grid-cols-3">
          <Stat label="Mean predicted PD" value={pct(m.mean_predicted_pd)} />
          <Stat label="Observed default rate" value={m.observed_default_rate !== undefined ? pct(m.observed_default_rate) : "No outcomes"}
            note={m.observed_default_rate === undefined ? "Add a SeriousDlqin2yrs column to check calibration." : undefined} />
          <Stat label="Observed minus predicted" value={citl !== undefined ? `${signed(citl * 100, 1)} pts` : "–"} note="Calibration-in-the-large" />
        </dl>
        {citl !== undefined && Math.abs(citl) > RECALIBRATE_ABOVE && (
          <div role="alert" className="mt-4 border-l-4 border-decline bg-decline-tint px-3 py-2 text-decline-ink" data-testid="recalibrate-banner">
            <p className="font-semibold">Recalibration recommended</p>
            <p className="text-sm">
              Predicted and observed default rates differ by {num(Math.abs(citl) * 100, 1)} points, more than the {pct(RECALIBRATE_ABOVE, 0)} tolerance.
              The probabilities are {citl > 0 ? "too low" : "too high"} for this batch.
              {m.psi.score.status === "stable" && " The score PSI is stable, so the drift alarm alone would have missed this."}
            </p>
          </div>
        )}
        {citl !== undefined && Math.abs(citl) <= RECALIBRATE_ABOVE && (
          <p className="mt-4 border-l-4 border-approve bg-approve-tint px-3 py-2 text-approve-ink">Predicted and observed default rates agree within {pct(RECALIBRATE_ABOVE, 0)}.</p>
        )}
      </Panel>
      <Panel title="Population stability (PSI)" aside={<>Largest shift: {worst[0]} ({num(worst[1].value, 3)})</>}>
        <Table caption="PSI of the score and each model feature against the calibration reference" rows={rows} rowKey={([k]) => k}
          highlight={([k]) => k === "score"}
          columns={[
            { head: "Measure", cell: ([k]) => (k === "score" ? "Model score (PD)" : k) },
            { head: "PSI", numeric: true, cell: ([, v]) => num(v.value, 4) },
            { head: "Status", cell: ([, v]) => <StatusTag status={v.status} /> },
            { head: "What it means", cell: ([, v]) => STATUS[v.status].text },
          ]} />
        <p className="mt-2 text-sm text-muted">
          PSI compares this batch with the calibration population. It sees a change in who applies. It cannot see a change in how
          often similar applicants default; only outcomes show that.
        </p>
      </Panel>
    </div>
  );
}

export default function HealthScreen() {
  const [report, setReport] = useState<{ name: string; m: MonitoringResponse }>();
  const [error, setError] = useState<string>();
  const [busy, setBusy] = useState(false);
  const evidence = useAsync(() => repoJson<Step15>("phase2/p2_step15_monitoring.json"), []);

  const check = async (name: string, csv: string | Blob) => {
    setBusy(true); setError(undefined);
    try { setReport({ name, m: await api.monitoring(csv) }); }
    catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };
  const step = evidence.data?.learning_step_S2;
  const scenarios = evidence.data
    ? (Object.entries(evidence.data).filter(([k]) => /^S\d/.test(k)) as [string, Scenario][]).sort(([a], [b]) => a.localeCompare(b))
    : [];

  return (
    <>
      <PageHeader title="Model health"
        lead="Check a new batch of applicants against the population the model was calibrated on. With outcomes, the service also checks whether the predicted default rate still matches reality." />
      <Panel title="Check a batch">
        <div className="flex flex-wrap items-center gap-2">
          {DEMO.map((d) => (
            <button key={d.file} disabled={busy} onClick={async () => check(d.file, await repoText(`demo/${d.file}`))}
              className="border border-ink px-3 py-1.5 font-semibold disabled:opacity-50">{d.label}</button>
          ))}
          <label className="cursor-pointer border border-line px-3 py-1.5">
            Upload a CSV
            <input type="file" accept=".csv,text/csv" className="sr-only" onChange={(e) => { const f = e.target.files?.[0]; if (f) void check(f.name, f); }} />
          </label>
          {busy && <Loading what="the check" />}
        </div>
        <p className="mt-2 text-sm text-muted">
          The file needs the seven applicant columns. Outcomes are optional, in a column named SeriousDlqin2yrs. Demo files are in data/demo.
        </p>
      </Panel>
      <div className="mt-4 space-y-4">
        {error && <ErrorNote message="The service could not check this file." hint={error} />}
        {report && <Report name={report.name} m={report.m} />}

        <Panel title="Recalibration evidence: champion against challenger">
          {evidence.error && <ErrorNote message="Could not load the monitoring experiment." hint={evidence.error} />}
          {step && (
            <>
              <p className="max-w-[75ch]">
                In the simulated downturn, the Platt map was re-fitted on {int(step.recent_labelled_outcomes_used)} recent labelled outcomes and both
                versions were compared on {int(step.evaluated_on)} later applicants (observed default rate {pct(step.observed_default_rate_holdout)}).
              </p>
              <div className="mt-3">
                <Table caption="Champion and recalibrated challenger on the downturn hold-out" rowKey={([k]) => k}
                  rows={[["Champion (current model)", step.champion], ["Challenger (recalibrated)", step.challenger_recalibrated]] as [string, Side][]}
                  columns={[
                    { head: "Version", cell: ([k]) => k },
                    { head: "Mean predicted PD", numeric: true, cell: ([, s]) => pct(s.mean_pd) },
                    { head: "ECE", numeric: true, cell: ([, s]) => num(s.ECE, 4), note: "Expected calibration error: average gap between predicted and observed default rates. Lower is better." },
                    { head: "Brier", numeric: true, cell: ([, s]) => num(s.Brier, 4) },
                    { head: "Cost per 1,000 (main policy)", numeric: true, cell: ([, s]) => num(s.main_policy.cost_per_1000, 1) },
                    { head: "Defaulters auto-approved", numeric: true, cell: ([, s]) => pct(s.main_policy.defaulters_auto_approved) },
                  ]} />
              </div>
              <p className="mt-3 font-semibold">{step.promote ? "The pre-stated rule promotes the challenger." : "The pre-stated rule keeps the champion."}</p>
              <p className="text-sm text-muted">{step.note}</p>
              <div className="mt-3">
                <Caveat>
                  This comparison comes from the experiment file phase2/outputs/p2_step15_monitoring.json. Service v2.0.0 reports drift and
                  calibration; it does not re-fit or promote a model from an upload. GMSC has no dates, so every drift scenario here is simulated.
                </Caveat>
              </div>
            </>
          )}
        </Panel>

        {scenarios.length > 0 && (
          <Panel title="Simulated scenarios from the monitoring experiment">
            <Table caption="Drift scenarios" rows={scenarios} rowKey={([k]) => k}
              columns={[
                { head: "Scenario", cell: ([k]) => k },
                { head: "Score PSI", numeric: true, cell: ([, s]) => num(s.psi_score.value, 3) },
                { head: "Alarm", cell: ([, s]) => <StatusTag status={s.psi_score.status} /> },
                { head: "Largest feature shift", cell: ([, s]) => { const [f, v] = Object.entries(s.worst_feature)[0]; return `${f} (${num(v.value, 3)}, ${STATUS[v.status].word.toLowerCase()})`; } },
                { head: "Predicted PD", numeric: true, cell: ([, s]) => pct(s.mean_predicted_pd) },
                { head: "Observed default", numeric: true, cell: ([, s]) => pct(s.observed_default_rate) },
                { head: "ECE", numeric: true, cell: ([, s]) => num(s.ECE, 4) },
              ]} />
            <img src={repoUrl("phase2/p2_monitoring.png")} alt="Monitoring experiment chart: PSI and calibration for each simulated scenario" className="mt-4 w-full max-w-3xl border border-line" />
          </Panel>
        )}
      </div>
    </>
  );
}
