import { useEffect, useMemo, useState, type FormEvent } from "react";
import { api, INPUTS, type Applicant, type InputName, type Reason, type ScoreResponse } from "../api";
import { DecisionChip } from "../components/decision";
import { PdBand } from "../components/PdBand";
import { Caveat, ErrorNote, Loading, PageHeader, Panel } from "../components/ui";
import { useModelCard } from "../lib/context";
import { parseCsv } from "../lib/csv";
import { pct, signed } from "../lib/format";
import { repoText } from "../lib/repo";
import { useAsync } from "../lib/useAsync";

// Wording only. Limits, defaults and every number come from the API schema, the model card or the sample file.
const FIELD: Record<InputName, { label: string; unit: string; help: string }> = {
  RevolvingUtilizationOfUnsecuredLines: { label: "Revolving utilisation", unit: "ratio", help: "Card and credit-line balances divided by their limits. 0.45 means 45% of the limits are used." },
  age: { label: "Age", unit: "years", help: "A model input. It is never given as a reason; see the governance note." },
  "NumberOfTime30-59DaysPastDueNotWorse": { label: "30–59 days late", unit: "times", help: "Times 30 to 59 days past due, and no worse, in the last 2 years." },
  "NumberOfTime60-89DaysPastDueNotWorse": { label: "60–89 days late", unit: "times", help: "Times 60 to 89 days past due, and no worse, in the last 2 years." },
  NumberOfTimes90DaysLate: { label: "90+ days late", unit: "times", help: "Times 90 days or more past due." },
  NumberOfOpenCreditLinesAndLoans: { label: "Open credit lines and loans", unit: "count", help: "Open loans (car, mortgage, instalment) plus credit lines (cards)." },
  NumberRealEstateLoansOrLines: { label: "Real-estate loans or lines", unit: "count", help: "Mortgages and home-equity lines of credit." },
};
type Draft = Record<InputName, string>;
const EMPTY = Object.fromEntries(INPUTS.map((k) => [k, ""])) as Draft;

function ReasonList({ reasons, empty }: { reasons: Reason[]; empty: string }) {
  if (!reasons.length) return <p className="text-muted">{empty}</p>;
  const max = Math.max(...reasons.map((r) => Math.abs(r.contribution_logodds)));
  return (
    <ol className="space-y-2">
      {reasons.map((r) => {
        const up = r.direction === "raises risk";
        return (
          <li key={r.code} className="grid grid-cols-[1fr_auto] items-baseline gap-x-3">
            <span>{r.text}</span>
            <span className="num text-sm whitespace-nowrap">
              <span aria-hidden="true">{up ? "▲" : "▼"}</span> {r.direction} <span className="text-muted">({signed(r.contribution_logodds)})</span>
            </span>
            <span className="col-span-2 mt-1 h-1.5 bg-paper" aria-hidden="true">
              <span className={`block h-full ${up ? "bg-ink" : "border border-ink bg-surface"}`} style={{ width: `${(Math.abs(r.contribution_logodds) / max) * 100}%` }} />
            </span>
          </li>
        );
      })}
    </ol>
  );
}

function Result({ result }: { result: ScoreResponse }) {
  const card = useModelCard();
  const [view, setView] = useState<"analyst" | "notice">("analyst");
  const chk = result.explanation_check;
  const gap = Math.abs(chk.base_logodds + chk.sum_of_contributions - chk.model_logodds);
  return (
    <div className="space-y-4" data-testid="result">
      <Panel>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-sm text-muted">Recommendation</p>
            <DecisionChip decision={result.decision} large />
          </div>
          <dl className="flex gap-6">
            <div>
              <dt className="text-sm text-muted">Probability of default, 2 years</dt>
              <dd className="num font-serif text-4xl" data-testid="pd">{pct(result.pd)}</dd>
            </div>
            <div>
              <dt className="text-sm text-muted">Risk tier</dt>
              <dd className="num font-serif text-4xl">{result.risk_tier}<span className="text-lg text-muted"> of {Object.keys(card.tiers).length}</span></dd>
              <dd className="num text-sm text-muted">{pct(result.tier_observed_default_rate)} of this tier defaulted in the test set</dd>
            </div>
          </dl>
        </div>
        <p className="mt-3 max-w-[75ch]" data-testid="decision-rule">{result.decision_rule}</p>
        {result.warnings.length > 0 && (
          <div role="alert" className="mt-3 border-l-4 border-refer bg-refer-tint px-3 py-2 text-refer-ink">
            <p className="font-semibold">Input warning: referred for manual review</p>
            <ul className="list-disc pl-5 text-sm">{result.warnings.map((w) => <li key={w}>{w}</li>)}</ul>
          </div>
        )}
        <div className="mt-5">
          <PdBand ranges={result.pd_ranges} tiers={card.tiers} pd={result.pd} decision={result.decision} />
        </div>
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Why this score">
          <div role="tablist" aria-label="Reason view" className="mb-3 flex border border-line text-sm">
            {(["analyst", "notice"] as const).map((v) => (
              <button key={v} role="tab" aria-selected={view === v} onClick={() => setView(v)}
                className={`flex-1 px-3 py-1.5 ${view === v ? "bg-ink font-semibold text-white" : "bg-surface"}`}>
                {v === "analyst" ? "Analyst view" : "Adverse-action notice"}
              </button>
            ))}
          </div>
          {view === "analyst" ? (
            <>
              <ReasonList reasons={result.reasons} empty="The service returned no reasons." />
              <p className="mt-3 text-sm text-muted">
                The largest drivers in either direction. Values are contributions to the log-odds of default. They explain this
                model's score; they are not causes of default.
              </p>
            </>
          ) : (
            <>
              <ReasonList reasons={result.adverse_action_reasons} empty="No notice is needed: the recommendation is to approve." />
              <p className="mt-3 text-sm text-muted">Only factors that raised the risk, in the wording for an applicant notice. Age is never listed.</p>
            </>
          )}
          <p className="num mt-3 border-t border-line pt-2 text-sm text-muted">
            Exactness check: base {signed(chk.base_logodds, 4)} + contributions {signed(chk.sum_of_contributions, 4)} = model{" "}
            {signed(chk.model_logodds, 4)} (gap {gap.toFixed(4)}).
          </p>
        </Panel>

        <Panel title="Policy and guarantee">
          <dl className="num grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
            <dt className="text-muted">Cost ratio r (missed defaulter ÷ good declined)</dt><dd>{result.policy.r}</dd>
            <dt className="text-muted">Referral cost c</dt><dd>{result.policy.c}</dd>
            <dt className="text-muted">Analyst budget ρ</dt><dd>{result.policy.rho === null ? "unlimited" : pct(result.policy.rho, 0)}</dd>
            <dt className="text-muted">Cap on defaulters auto-approved α</dt><dd>{result.policy.alpha ? pct(result.policy.alpha, 0) : "off"}</dd>
            <dt className="text-muted">Confidence level</dt><dd>{result.policy.delta ? pct(1 - result.policy.delta, 0) : "on average"}</dd>
          </dl>
          <p className="mt-3 border-l-4 border-accent bg-paper px-3 py-2" data-testid="guarantee">{result.guarantee}</p>
          <p className="mt-2 text-sm text-muted">
            The guarantee is about a share of all defaulters, across many applicants. It says nothing about this one applicant.
            r, c and ρ are assumptions set by the lender.
          </p>
          <details className="mt-3 border-t border-line pt-2">
            <summary className="cursor-pointer font-semibold">Governance note: age</summary>
            <p className="num mt-2 text-sm">
              Age contribution to this score: {signed(result.protected_attribute_contribution.age_logodds, 4)} log-odds.{" "}
              {result.protected_attribute_contribution.note}
            </p>
          </details>
        </Panel>
      </div>
    </div>
  );
}

export default function DecisionScreen() {
  const card = useModelCard();
  const schema = useAsync(api.applicantSchema, []);
  const samples = useAsync(async () => {
    const raw = await repoText("demo/sample_applicants.csv");
    // Score the sample file once so each preset can show its PD and decision; nothing here is typed in by hand.
    const scored = parseCsv(await api.scoreBatch(raw));
    return scored;
  }, []);
  const [draft, setDraft] = useState<Draft>(EMPTY);
  const [preset, setPreset] = useState("");
  const [result, setResult] = useState<ScoreResponse>();
  const [error, setError] = useState<string>();
  const [busy, setBusy] = useState(false);

  const run = async (values: Draft) => {
    setBusy(true); setError(undefined);
    try {
      const applicant = Object.fromEntries(INPUTS.map((k) => [k, Number(values[k])])) as Applicant;
      setResult(await api.score(applicant));
    } catch (e) { setResult(undefined); setError((e as Error).message); }
    finally { setBusy(false); }
  };
  const choose = (id: string) => {
    setPreset(id);
    const row = samples.data?.find((r) => r.applicant_id === id);
    if (!row) return;
    const next = Object.fromEntries(INPUTS.map((k) => [k, row[k]])) as Draft;
    setDraft(next); void run(next);
  };
  // Open on the first sample so the screen is never empty.
  useEffect(() => { if (samples.data?.length && !preset) choose(samples.data[0].applicant_id); /* eslint-disable-next-line */ }, [samples.data]);

  const submit = (e: FormEvent) => { e.preventDefault(); void run(draft); };
  const complete = useMemo(() => INPUTS.every((k) => draft[k] !== ""), [draft]);

  return (
    <>
      <PageHeader title="Applicant decision"
        lead="Enter seven bureau-style attributes. CreditSense returns a calibrated probability of default, a recommendation under the stated policy, and the reasons. A person confirms every decline." />
      <div className="mb-4"><Caveat>{card.disclaimer}</Caveat></div>
      <div className="grid gap-4 xl:grid-cols-[22rem_1fr]">
        <Panel title="Applicant">
          <form onSubmit={submit} className="space-y-3" aria-label="Applicant inputs">
            <label className="block">
              <span className="text-sm font-semibold">Sample applicant</span>
              <select value={preset} onChange={(e) => choose(e.target.value)} className="num mt-1 w-full border border-line bg-surface px-2 py-1.5"
                disabled={!samples.data}>
                <option value="">{samples.error ? "Samples unavailable" : samples.data ? "Custom values" : "Loading samples…"}</option>
                {samples.data?.map((r) => (
                  <option key={r.applicant_id} value={r.applicant_id}>
                    {r.applicant_id}: PD {pct(Number(r.pd))}, {r.decision}{r.warnings ? " (input warning)" : ""}
                  </option>
                ))}
              </select>
              <span className="text-sm text-muted">From data/demo/sample_applicants.csv. No real personal data.</span>
            </label>
            {INPUTS.map((k) => {
              const spec = schema.data?.[k];
              const whole = spec?.type === "integer";
              return (
                <label key={k} className="block">
                  <span className="flex items-baseline justify-between text-sm font-semibold">
                    {FIELD[k].label}<span className="font-normal text-muted">{FIELD[k].unit}</span>
                  </span>
                  <input type="number" inputMode={whole ? "numeric" : "decimal"} required name={k} value={draft[k]}
                    min={spec?.minimum} max={spec?.maximum} step={whole ? 1 : "any"}
                    onChange={(e) => { setPreset(""); setDraft({ ...draft, [k]: e.target.value }); }}
                    aria-describedby={`${k}-help`} className="num mt-1 w-full border border-line px-2 py-1.5" />
                  <span id={`${k}-help`} className="block text-sm text-muted">{FIELD[k].help}</span>
                </label>
              );
            })}
            <button type="submit" disabled={!complete || busy} className="w-full bg-ink px-3 py-2 font-semibold text-white disabled:opacity-50">
              {busy ? "Scoring…" : "Score applicant"}
            </button>
          </form>
        </Panel>
        <div>
          {error && <ErrorNote message="The service could not score this applicant." hint={error} />}
          {!error && !result && <Loading what="the first decision" />}
          {result && <Result result={result} />}
        </div>
      </div>
    </>
  );
}
