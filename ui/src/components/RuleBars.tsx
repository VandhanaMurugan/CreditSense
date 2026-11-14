import type { Decision, RuleResult } from "../api";
import { pct } from "../lib/format";
import { DECISION_FILL, DECISION_SHORT } from "./decision";

type Mix = Pick<RuleResult, "approve_rate" | "refer_rate" | "decline_rate">;
const PARTS: [Decision, keyof Mix][] = [["approve", "approve_rate"], ["refer", "refer_rate"], ["decline", "decline_rate"]];
const INK: Record<Decision, string> = { approve: "text-ink", refer: "text-ink", decline: "text-white" };

/** One stacked bar of approve / refer / decline shares. Each segment carries its word and value; colour is secondary. */
export function DecisionMixBar({ result, height = "h-8" }: { result: Mix; height?: string }) {
  const label = PARTS.map(([d, k]) => `${DECISION_SHORT[d]} ${pct(result[k])}`).join(", ");
  return (
    <div role="img" aria-label={label} className={`flex w-full gap-0.5 ${height}`}>
      {PARTS.filter(([, k]) => result[k] > 0).map(([d, k]) => (
        <div key={d} title={`${DECISION_SHORT[d]} ${pct(result[k])}`} style={{ width: `${result[k] * 100}%`, background: DECISION_FILL[d] }}
          className={`num flex min-w-0 items-center overflow-hidden px-1 text-xs font-semibold whitespace-nowrap ${INK[d]}`}>
          {result[k] >= 0.3 ? `${DECISION_SHORT[d]} ${pct(result[k], 0)}` : result[k] >= 0.1 ? pct(result[k], 0) : ""}
        </div>
      ))}
    </div>
  );
}

export function DecisionLegend() {
  return (
    <ul className="flex flex-wrap gap-x-4 text-sm" aria-label="Legend">
      {PARTS.map(([d]) => (
        <li key={d} className="flex items-center gap-1.5">
          <span aria-hidden="true" className="inline-block h-3 w-3" style={{ background: DECISION_FILL[d] }} />{DECISION_SHORT[d]}
        </li>
      ))}
    </ul>
  );
}

/** A share drawn against a limit line (the cap alpha). Over the limit is stated in words as well. */
export function CapBar({ value, cap, max }: { value: number; cap?: number; max: number }) {
  const over = cap !== undefined && value > cap;
  return (
    <div>
      <div className="relative h-4 bg-paper" role="img" aria-label={`${pct(value)}${cap !== undefined ? `, cap ${pct(cap, 0)}` : ""}`}>
        <div className="h-full bg-ink" style={{ width: `${Math.min(value / max, 1) * 100}%` }} />
        {cap !== undefined && cap <= max && (
          <div className="absolute -top-1 h-6 w-0.5 bg-decline" style={{ left: `${(cap / max) * 100}%` }} aria-hidden="true" />
        )}
      </div>
      {cap !== undefined && (
        <p className="num mt-0.5 text-xs text-muted">
          {over ? `Above the ${pct(cap, 0)} cap on this test set` : `Within the ${pct(cap, 0)} cap`}
          <span aria-hidden="true"> (red line)</span>
        </p>
      )}
    </div>
  );
}
