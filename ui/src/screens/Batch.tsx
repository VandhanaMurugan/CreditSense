import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type Decision } from "../api";
import { DecisionChip } from "../components/decision";
import { DecisionLegend, DecisionMixBar } from "../components/RuleBars";
import { Table } from "../components/Table";
import { Caveat, ErrorNote, Loading, PageHeader, Panel, Stat } from "../components/ui";
import { useModelCard } from "../lib/context";
import { parseCsv } from "../lib/csv";
import { int, pct } from "../lib/format";
import { describePolicy, usePolicy } from "../lib/policy";
import { repoText } from "../lib/repo";

const DEMO = [
  { file: "sample_applicants.csv", label: "Demo: sample applicants" },
  { file: "batch_drift.csv", label: "Demo: drifted batch" },
];
const PREVIEW = 25;
type Row = Record<string, string>;

export default function BatchScreen() {
  const card = useModelCard();
  const { policy } = usePolicy();
  const [out, setOut] = useState<{ name: string; csv: string; rows: Row[] }>();
  const [error, setError] = useState<string>();
  const [busy, setBusy] = useState(false);

  const score = async (name: string, file: string | Blob) => {
    setBusy(true); setError(undefined);
    try { const csv = await api.scoreBatch(file, policy); setOut({ name, csv, rows: parseCsv(csv) }); }
    catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };
  const summary = useMemo(() => {
    if (!out?.rows.length) return undefined;
    const n = out.rows.length;
    const share = (d: Decision) => out.rows.filter((r) => r.decision === d).length / n;
    const tiers = Object.entries(card.tiers).map(([t, v]) => ({
      tier: `Tier ${t}`, range: `PD ${pct(v.pd_range[0], 0)} to ${pct(v.pd_range[1], 0)}`,
      applicants: out.rows.filter((r) => r.risk_tier === t).length, reference: Math.round(v.share_of_applicants * n),
    }));
    return {
      n, tiers, mix: { approve_rate: share("approve"), refer_rate: share("refer"), decline_rate: share("decline") },
      meanPd: out.rows.reduce((a, r) => a + Number(r.pd), 0) / n, warned: out.rows.filter((r) => r.warnings).length,
    };
  }, [out, card]);
  const download = () => {
    if (!out) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([out.csv], { type: "text/csv" }));
    a.download = `scored_${out.name}`; a.click(); URL.revokeObjectURL(a.href);
  };
  const idCol = out?.rows[0] && "applicant_id" in out.rows[0] ? "applicant_id" : undefined;

  return (
    <>
      <PageHeader title="Batch scoring" lead="Score a file of applicants under the policy in use. The service returns each applicant's PD, risk tier, recommendation and any input warnings." />
      <Panel title="Score a file">
        <div className="flex flex-wrap items-center gap-2">
          {DEMO.map((d) => (
            <button key={d.file} disabled={busy} onClick={async () => score(d.file, await repoText(`demo/${d.file}`))}
              className="border border-ink px-3 py-1.5 font-semibold disabled:opacity-50">{d.label}</button>
          ))}
          <label className="cursor-pointer border border-line px-3 py-1.5">
            Upload a CSV
            <input type="file" accept=".csv,text/csv" className="sr-only" onChange={(e) => { const f = e.target.files?.[0]; if (f) void score(f.name, f); }} />
          </label>
          {busy && <Loading what="the scores" />}
        </div>
        <p className="num mt-2 text-sm text-muted">
          Policy in use: {describePolicy(policy)}. <a href="#/policy" className="text-accent underline">Change it in the Policy simulator</a>. Use sample or synthetic data only; do not upload personal data.
        </p>
      </Panel>
      <div className="mt-4 space-y-4">
        {error && <ErrorNote message="The service could not score this file." hint={`${error}. The file needs the seven applicant columns with their exact names.`} />}
        {out && summary && (
          <div className="space-y-4" data-testid="batch-result">
            <Panel title={`Results: ${out.name}`} aside={<button onClick={download} className="border border-ink px-2.5 py-1 font-semibold text-ink">Download scored CSV</button>}>
              <dl className="grid gap-4 sm:grid-cols-3">
                <Stat label="Applicants scored" value={int(summary.n)} />
                <Stat label="Mean predicted PD" value={pct(summary.meanPd)} />
                <Stat label="Referred for an input warning" value={int(summary.warned)} note="Inputs outside the range the model can score" />
              </dl>
              <div className="mt-4 flex items-baseline justify-between"><h3 className="font-semibold">Decision mix</h3><DecisionLegend /></div>
              <DecisionMixBar result={summary.mix} />
              <p className="num mt-1 text-sm text-muted">
                Approve {pct(summary.mix.approve_rate)}, refer {pct(summary.mix.refer_rate)}, decline {pct(summary.mix.decline_rate)}. A person confirms every decline.
              </p>
            </Panel>
            <Panel title="Score distribution by risk tier">
              <div className="h-56" role="img" aria-label="Bar chart of applicants per risk tier. The table below has the same values.">
                <ResponsiveContainer>
                  <BarChart data={summary.tiers} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                    <CartesianGrid stroke="#e6e9ef" vertical={false} />
                    <XAxis dataKey="tier" tick={{ fontSize: 12 }} />
                    <YAxis allowDecimals={false} tick={{ fontSize: 12 }} width={44} />
                    <Tooltip formatter={(v) => int(Number(v))} cursor={{ fill: "#f3f5f8" }} />
                    <Bar dataKey="applicants" name="Applicants in this file" fill="#2a78d6" radius={[4, 4, 0, 0]} maxBarSize={56} isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <Table caption="Applicants per risk tier" rows={summary.tiers} rowKey={(r) => r.tier}
                columns={[
                  { head: "Tier", cell: (r) => r.tier },
                  { head: "PD range", cell: (r) => <span className="num">{r.range}</span> },
                  { head: "Applicants in this file", numeric: true, cell: (r) => int(r.applicants) },
                  { head: "Expected from the test-set mix", numeric: true, cell: (r) => int(r.reference) },
                ]} />
            </Panel>
            <Panel title="Scored applicants" aside={summary.n > PREVIEW ? `First ${PREVIEW} of ${int(summary.n)}; the download has all rows` : undefined}>
              <Table caption="Scored applicants" rows={out.rows.slice(0, PREVIEW).map((r, i) => ({ ...r, _row: String(i + 1) }))} rowKey={(r) => r._row}
                columns={[
                  { head: idCol ? "Applicant" : "Row", cell: (r: Row) => (idCol ? r[idCol] : r._row) },
                  { head: "PD", numeric: true, cell: (r: Row) => pct(Number(r.pd)) },
                  { head: "Tier", numeric: true, cell: (r: Row) => r.risk_tier },
                  { head: "Recommendation", cell: (r: Row) => <DecisionChip decision={r.decision as Decision} /> },
                  { head: "Warnings", cell: (r: Row) => r.warnings || "None" },
                ]} />
            </Panel>
            <Caveat>{card.disclaimer}</Caveat>
          </div>
        )}
      </div>
    </>
  );
}
