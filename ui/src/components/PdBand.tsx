import type { Decision, PdRanges, Tier } from "../api";
import { pct } from "../lib/format";
import { DECISION_SHORT, DecisionIcon } from "./decision";

// The interesting thresholds sit in the first few percent, so the axis is compressed (power scale).
// The compression is a display choice only; every position is computed from API values.
export const bandX = (p: number) => Math.pow(Math.min(Math.max(p, 0), 1), 0.4) * 100;

const ORDER: Decision[] = ["approve", "refer", "decline"];
const ZONE: Record<Decision, string> = {
  approve: "bg-approve-tint text-approve-ink border-approve",
  refer: "bg-refer-tint text-refer-ink border-refer",
  decline: "bg-decline-tint text-decline-ink border-decline",
};

interface Props { ranges: PdRanges; tiers: Record<string, Tier>; pd?: number; decision?: Decision }

/** The policy drawn as one band: where approve / refer / decline start, the risk tiers, and this applicant. */
export function PdBand({ ranges, tiers, pd, decision }: Props) {
  const zones = ORDER.filter((d) => ranges[d]);
  const cuts = Object.values(tiers).map((t) => t.pd_range[1]).filter((v) => v < 1);
  const overridden = pd !== undefined && decision !== undefined && zones.length > 0 &&
    !(ranges[decision] && pd >= ranges[decision]![0] && pd <= ranges[decision]![1]);
  return (
    <figure aria-label="Policy band: probability-of-default ranges for each decision">
      <div className="relative pt-9">
        {pd !== undefined && (
          <div data-testid="pd-marker" className="absolute top-0 flex -translate-x-1/2 flex-col items-center" style={{ left: `${bandX(pd)}%` }}>
            <span className="num whitespace-nowrap bg-ink px-1.5 text-sm font-semibold text-white">This applicant {pct(pd)}</span>
            <span className="h-[4.25rem] w-0.5 bg-ink ring-2 ring-white" aria-hidden="true" />
          </div>
        )}
        <div className="flex h-12 w-full gap-0.5">
          {zones.map((d) => {
            const [lo, hi] = ranges[d]!;
            return (
              <div key={d} className={`flex min-w-0 items-center gap-1 overflow-hidden border-t-4 px-1.5 text-sm font-semibold ${ZONE[d]}`}
                style={{ width: `${bandX(hi) - bandX(lo)}%` }}>
                <DecisionIcon decision={d} size={14} />
                <span className="truncate">{DECISION_SHORT[d]}</span>
              </div>
            );
          })}
        </div>
        <div className="relative h-6" aria-hidden="true">
          {cuts.map((c) => (
            <span key={c} className="num absolute top-0 -translate-x-1/2 border-l border-muted pl-1 text-xs text-muted" style={{ left: `${bandX(c)}%` }}>
              {pct(c, 0)}
            </span>
          ))}
        </div>
      </div>
      <figcaption className="text-sm text-muted">
        <ul className="flex flex-wrap gap-x-4">
          {zones.map((d) => (
            <li key={d} className="num">
              <span className="font-semibold text-ink">{DECISION_SHORT[d]}</span> PD {pct(ranges[d]![0], 2)} to {pct(ranges[d]![1], 2)}
            </li>
          ))}
        </ul>
        <p>Ticks mark the risk-tier cut-offs. The axis is compressed so the low-PD thresholds stay readable.</p>
        {overridden && <p className="font-semibold text-ink">The decision differs from the band because an input warning forced a referral.</p>}
      </figcaption>
    </figure>
  );
}
