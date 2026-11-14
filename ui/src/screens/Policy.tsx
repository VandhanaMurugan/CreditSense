import { useEffect, useState } from "react";
import { api, UNLIMITED_RHO, type GroupRates, type Policy, type RuleResult, type SimulationResponse } from "../api";
import { PdBand } from "../components/PdBand";
import { CapBar, DecisionLegend, DecisionMixBar } from "../components/RuleBars";
import { Table } from "../components/Table";
import { Caveat, ErrorNote, Loading, PageHeader, Panel } from "../components/ui";
import { useModelCard } from "../lib/context";
import { int, num, pct, signed } from "../lib/format";
import { describePolicy, isUnlimited, usePolicy } from "../lib/policy";

// Slider limits are the ranges asked for in the build brief (context pack 11.3); they are controls, not results.
const RANGE = { r: [2, 20, 1], c: [0.05, 0.5, 0.05], rho: [0, 0.5, 0.05], alpha: [0.02, 0.3, 0.01] } as const;
const PAC_DELTA = 0.05;
const RULE_NOTE: Record<string, string> = {
  CreditSense: "Cost-optimal approve / refer / decline with the analyst budget and the cap on defaulters auto-approved.",
  "Same budget, no guarantee": "The same cost rule and analyst budget, without the cap.",
  "Two-way cost threshold": "Approve or decline only, at the cost threshold 1 / (1 + r). No referrals.",
};

function Slider({ id, label, help, value, range, onChange, format, disabled }: {
  id: string; label: string; help: string; value: number; range: readonly [number, number, number]; onChange: (v: number) => void;
  format: (v: number) => string; disabled?: boolean;
}) {
  return (
    <div>
      <label htmlFor={id} className="flex items-baseline justify-between font-semibold">
        <span>{label}</span><output className="num font-serif text-lg">{disabled ? "–" : format(value)}</output>
      </label>
      <input id={id} type="range" min={range[0]} max={range[1]} step={range[2]} value={Math.min(Math.max(value, range[0]), range[1])}
        disabled={disabled} onChange={(e) => onChange(Number(e.target.value))} aria-describedby={`${id}-help`} className="w-full accent-[var(--color-ink)]" />
      <p id={`${id}-help`} className="text-sm text-muted">{help}</p>
    </div>
  );
}

function Rules({ sim }: { sim: SimulationResponse }) {
  const rows = Object.entries(sim.results);
  const maxCost = Math.max(...rows.map(([, r]) => r.cost_per_1000));
  const maxMiss = Math.max(...rows.map(([, r]) => r.defaulters_auto_approved), sim.policy.alpha || 0) * 1.1;
  const cap = sim.policy.alpha || undefined;
  return (
    <div className="grid gap-4 lg:grid-cols-3" data-testid="rules">
      {rows.map(([name, r]) => (
        <section key={name} className={`border p-3 ${name === "CreditSense" ? "border-2 border-ink" : "border-line"}`} aria-label={name}>
          <h3 className="font-semibold">{name}</h3>
          <p className="min-h-[3.75rem] text-sm text-muted">{RULE_NOTE[name] ?? ""}</p>
          <p className="mt-2 text-sm text-muted">Decisions</p>
          <DecisionMixBar result={r} />
          <dl className="mt-3 space-y-3">
            <div>
              <dt className="text-sm text-muted">Expected cost per 1,000 applicants</dt>
              <dd className="num font-serif text-2xl">{num(r.cost_per_1000, 1)}</dd>
              <dd><div className="h-1.5 bg-paper" aria-hidden="true"><div className="h-full bg-muted" style={{ width: `${(r.cost_per_1000 / maxCost) * 100}%` }} /></div></dd>
            </div>
            <div>
              <dt className="text-sm text-muted">Defaulters auto-approved</dt>
              <dd className="num font-serif text-2xl">{pct(r.defaulters_auto_approved)}</dd>
              <dd><CapBar value={r.defaulters_auto_approved} cap={name === "CreditSense" ? cap : undefined} max={maxMiss} /></dd>
            </div>
            <div className="num grid grid-cols-2 gap-x-2 text-sm">
              <dt className="text-muted">Good borrowers auto-declined</dt><dd className="text-right">{pct(r.goods_auto_declined)}</dd>
              <dt className="text-muted">Default rate among approved</dt><dd className="text-right">{pct(r.default_rate_in_approved)}</dd>
            </div>
          </dl>
        </section>
      ))}
    </div>
  );
}

function Groups({ sim }: { sim: SimulationResponse }) {
  const g = sim.group_rates;
  if (!g) return <p className="text-muted">This version of the service does not report group rates.</p>;
  if (!Object.keys(g.groups).length) return <p className="text-muted">{g.note}</p>;
  const rows = Object.entries(g.groups) as [string, GroupRates][];
  const overall = sim.results.CreditSense;
  return (
    <>
      <Table caption="CreditSense decisions by age band under the chosen policy" rowKey={([band]) => band} rows={rows}
        columns={[
          { head: "Age band", cell: ([band]) => band },
          { head: "Test applicants", numeric: true, cell: ([, r]) => int(r.n) },
          { head: "Default rate", numeric: true, cell: ([, r]) => pct(r.observed_default_rate) },
          { head: "Decisions", cell: ([, r]) => <div className="min-w-[14rem]"><DecisionMixBar result={r} height="h-6" /></div> },
          { head: "Declined", numeric: true, cell: ([, r]) => pct(r.decline_rate) },
          { head: "Defaulters auto-approved", numeric: true, cell: ([, r]) => pct(r.defaulters_auto_approved) },
        ]} />
      <p className="num mt-2 text-sm text-muted">
        All applicants: {pct(overall.decline_rate)} declined, {pct(overall.defaulters_auto_approved)} of defaulters auto-approved. {g.note}
      </p>
    </>
  );
}

export default function PolicyScreen() {
  const card = useModelCard();
  const { policy, setPolicy, isDefault } = usePolicy();
  const [sim, setSim] = useState<SimulationResponse>();
  const [error, setError] = useState<string>();
  const [pending, setPending] = useState(true);
  const set = (patch: Partial<Policy>) => setPolicy({ ...policy, ...patch });

  // Sliders fire quickly; wait for a pause, and drop answers that arrive out of order.
  useEffect(() => {
    let live = true; setPending(true);
    const t = setTimeout(() => {
      api.simulate(policy).then(
        (s) => { if (live) { setSim(s); setError(undefined); setPending(false); } },
        (e: Error) => { if (live) { setError(e.message); setPending(false); } },
      );
    }, 200);
    return () => { live = false; clearTimeout(t); };
  }, [policy]);

  const presets: [string, Policy][] = [
    ["Main policy", card.policy_default],
    ["Business-friendly (cap 20%)", { ...card.policy_default, alpha: 0.2 }],
  ];
  const unlimited = isUnlimited(policy.rho);
  const cs: RuleResult | undefined = sim?.results.CreditSense;
  const main = card.main_policy_on_test;

  return (
    <>
      <PageHeader title="Policy simulator"
        lead="Set the lender's costs, analyst capacity and risk appetite. The service re-evaluates three rules on the held-out test applicants, so you see the price of a policy before using it." />
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[22rem_1fr]">
        <Panel title="Policy">
          <div className="mb-4 flex flex-wrap gap-2">
            {presets.map(([name, p]) => {
              const on = JSON.stringify(p) === JSON.stringify(policy);
              return (
                <button key={name} onClick={() => setPolicy(p)} aria-pressed={on}
                  className={`border px-2.5 py-1 text-sm ${on ? "border-ink bg-ink font-semibold text-white" : "border-line bg-surface"}`}>{name}</button>
              );
            })}
          </div>
          <div className="space-y-4">
            <Slider id="r" label="Cost ratio r" value={policy.r} range={RANGE.r} onChange={(r) => set({ r })} format={(v) => `${v} : 1`}
              help="Cost of approving one defaulter, in units of one good borrower wrongly declined. An assumption." />
            <Slider id="c" label="Referral cost c" value={policy.c} range={RANGE.c} onChange={(c) => set({ c })} format={(v) => num(v)}
              help="Cost of one analyst review, same units. An assumption. Analysts are assumed to decide correctly." />
            <div>
              <Slider id="rho" label="Analyst budget ρ" value={unlimited ? RANGE.rho[1] : policy.rho!} range={RANGE.rho} disabled={unlimited}
                onChange={(rho) => set({ rho })} format={(v) => pct(v, 0)} help="Largest share of applicants analysts can review." />
              <label className="mt-1 flex items-center gap-2 text-sm">
                <input type="checkbox" checked={unlimited} onChange={(e) => set({ rho: e.target.checked ? UNLIMITED_RHO : card.policy_default.rho })} />
                Unlimited referrals
              </label>
            </div>
            <div>
              <Slider id="alpha" label="Cap α on defaulters auto-approved" value={policy.alpha || RANGE.alpha[0]} range={RANGE.alpha} disabled={!policy.alpha}
                onChange={(alpha) => set({ alpha })} format={(v) => pct(v, 0)} help="Risk appetite: the largest share of future defaulters the lender accepts being auto-approved." />
              <label className="mt-1 flex items-center gap-2 text-sm">
                <input type="checkbox" checked={!policy.alpha} onChange={(e) => set({ alpha: e.target.checked ? 0 : card.policy_default.alpha, delta: e.target.checked ? null : policy.delta })} />
                No cap (guarantee off)
              </label>
              <label className="mt-1 flex items-center gap-2 text-sm">
                <input type="checkbox" checked={!!policy.delta} disabled={!policy.alpha} onChange={(e) => set({ delta: e.target.checked ? PAC_DELTA : null })} />
                Hold the cap with {pct(1 - PAC_DELTA, 0)} confidence (stricter) instead of on average
              </label>
            </div>
          </div>
          <p className="mt-4 border-t border-line pt-3 text-sm text-muted">The decision and batch screens use this policy.</p>
        </Panel>

        <div className="space-y-4">
          {error && <ErrorNote message="The service could not evaluate this policy." hint={error} />}
          {!sim && !error && <Loading what="the simulation" />}
          {sim && (
            <div className={`space-y-4 ${pending ? "opacity-60" : ""}`} aria-busy={pending}>
              <Caveat>
                Results are for {int(sim.n_test)} held-out test applicants under: {describePolicy(sim.policy)}. Costs use assumed values
                of r and c, in units of one good borrower wrongly declined. They are not currency.
              </Caveat>
              <Panel title="Three rules, same applicants" aside={<DecisionLegend />}>
                <Rules sim={sim} />
                {cs && isDefault && <p className="mt-3 text-sm" data-testid="vs-main">This is the main policy.</p>}
                {cs && !isDefault && (
                  <p className="num mt-3 text-sm" data-testid="vs-main">
                    Compared with the main policy, this policy changes CreditSense cost by {signed(cs.cost_per_1000 - main.cost_per_1000, 1)} per 1,000,
                    approvals by {signed((cs.approve_rate - main.approve_rate) * 100, 1)} points and defaulters auto-approved by{" "}
                    {signed((cs.defaulters_auto_approved - main.defaulters_auto_approved) * 100, 1)} points.
                  </p>
                )}
              </Panel>
              <Panel title="Where the policy draws its lines">
                <PdBand ranges={sim.pd_ranges} tiers={card.tiers} />
              </Panel>
              <Panel title="Who the policy affects: decisions by age band">
                <Groups sim={sim} />
              </Panel>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
