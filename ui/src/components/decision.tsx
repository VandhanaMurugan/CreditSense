import type { Decision } from "../api";

export const DECISION_LABEL: Record<Decision, string> = { approve: "Approve", refer: "Refer to analyst", decline: "Decline" };
export const DECISION_SHORT: Record<Decision, string> = { approve: "Approve", refer: "Refer", decline: "Decline" };
export const DECISION_FILL: Record<Decision, string> = {
  approve: "var(--color-approve)", refer: "var(--color-refer)", decline: "var(--color-decline)",
};
const TONE: Record<Decision, string> = {
  approve: "bg-approve-tint text-approve-ink border-approve",
  refer: "bg-refer-tint text-refer-ink border-refer",
  decline: "bg-decline-tint text-decline-ink border-decline",
};
const PATH: Record<Decision, string> = {
  approve: "M4 10.5l4 4 8-9",                                   // tick
  refer: "M10 4a3 3 0 110 6 3 3 0 010-6zM4 17c0-3 2.7-5 6-5s6 2 6 5",  // person
  decline: "M5 5l10 10M15 5L5 15",                              // cross
};

export function DecisionIcon({ decision, size = 18 }: { decision: Decision; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="2"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={PATH[decision]} />
    </svg>
  );
}

/** Decision state shown as shape + word + colour, never colour alone. */
export function DecisionChip({ decision, large = false }: { decision: Decision; large?: boolean }) {
  return (
    <span data-testid="decision-chip" data-decision={decision}
      className={`inline-flex items-center gap-2 border-l-4 font-semibold ${TONE[decision]} ${large ? "px-4 py-2 text-xl font-serif" : "px-2 py-0.5 text-sm"}`}>
      <DecisionIcon decision={decision} size={large ? 22 : 14} />
      {large ? DECISION_LABEL[decision] : DECISION_SHORT[decision]}
    </span>
  );
}
