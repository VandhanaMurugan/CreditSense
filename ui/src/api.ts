// Typed client for the CreditSense FastAPI service. Every number the UI shows comes through here
// or through repo.ts (files copied from the repository by scripts/sync-data.mjs).
export const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

export const INPUTS = [
  "RevolvingUtilizationOfUnsecuredLines", "age", "NumberOfTime30-59DaysPastDueNotWorse",
  "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate", "NumberOfOpenCreditLinesAndLoans",
  "NumberRealEstateLoansOrLines",
] as const;
export type InputName = (typeof INPUTS)[number];
export type Applicant = Record<InputName, number>;
export type Decision = "approve" | "refer" | "decline";

export interface Policy { r: number; c: number; rho: number | null; alpha: number; delta: number | null }
export interface Reason { code: string; text: string; direction: "raises risk" | "lowers risk"; contribution_logodds: number }
export type PdRanges = Partial<Record<Decision, [number, number]>>;

export interface ScoreResponse {
  pd: number; risk_tier: number; tier_observed_default_rate: number;
  decision: Decision; decision_rule: string; policy: Policy; pd_ranges: PdRanges;
  guarantee: string; warnings: string[]; model_version: string; disclaimer: string;
  reasons: Reason[]; adverse_action_reasons: Reason[];
  protected_attribute_contribution: { age_logodds: number; note: string };
  explanation_check: { base_logodds: number; sum_of_contributions: number; model_logodds: number };
}
export interface RuleResult {
  cost_per_1000: number; approve_rate: number; refer_rate: number; decline_rate: number;
  defaulters_auto_approved: number; goods_auto_declined: number; default_rate_in_approved: number;
}
export interface GroupRates extends RuleResult { n: number; observed_default_rate: number }
export interface SimulationResponse {
  policy: Policy; pd_ranges: PdRanges; n_test: number; results: Record<string, RuleResult>;
  group_rates?: { attribute: string; note: string; groups: Record<string, GroupRates> };
}
export interface Tier { pd_range: [number, number]; share_of_applicants: number; observed_default_rate: number; mean_predicted_pd: number }
export interface ModelCard {
  version: string; built: string; data: Record<string, string | number>; features: string[];
  test_metrics: Record<string, number>; main_policy_on_test: RuleResult; tiers: Record<string, Tier>;
  policy_default: Policy; intended_use: string; limitations: string[]; disclaimer: string;
}
export interface MonitoringResponse {
  n: number; psi: Record<string, { value: number; status: "stable" | "watch" | "act" }>;
  mean_predicted_pd: number; observed_default_rate?: number; calibration_in_the_large?: number;
}
export interface FieldSpec { type: string; minimum?: number; maximum?: number; description?: string }

async function fail(res: Response): Promise<never> {
  let detail = `${res.status} ${res.statusText}`;
  try {
    const body = await res.json();
    if (typeof body.detail === "string") detail = body.detail;
    else if (Array.isArray(body.detail)) detail = body.detail.map((d: { loc: string[]; msg: string }) => `${d.loc.slice(-1)[0]}: ${d.msg}`).join("; ");
  } catch { /* keep the status text */ }
  throw new Error(detail);
}
async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(API_BASE + path);
  return res.ok ? res.json() : fail(res);
}
/** The service reads a missing rho as its default budget, so "unlimited referrals" is sent as rho = 1
 *  (a budget of every applicant), which gives the same referral cut as no budget in creditsense.policy. */
export const UNLIMITED_RHO = 1;
function policyQuery(p: Partial<Policy>): string {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(p)) if (v !== null && v !== undefined) q.set(k, String(v));
  return q.toString();
}
function csvForm(csv: string | Blob, name = "upload.csv"): FormData {
  const form = new FormData();
  form.append("file", typeof csv === "string" ? new Blob([csv], { type: "text/csv" }) : csv, name);
  return form;
}

/** True if something is listening at the API address. A service that crashes inside a request answers without the
 *  headers a browser needs, so the page sees the same "Failed to fetch" as when nothing is running; this probe (which
 *  needs no readable answer, and asks for a route that does not load the model) tells the two apart. */
export async function serviceReachable(): Promise<boolean> {
  try { await fetch(API_BASE + "/openapi.json", { mode: "no-cors" }); return true; } catch { return false; }
}

export const api = {
  health: () => getJson<{ status: string; model_version: string }>("/health"),
  modelCard: () => getJson<ModelCard>("/model_card"),
  /** Field limits and descriptions, read from the API's own schema so the form cannot drift from the service. */
  applicantSchema: async (): Promise<Record<string, FieldSpec>> =>
    (await getJson<{ components: { schemas: { Applicant: { properties: Record<string, FieldSpec> } } } }>("/openapi.json"))
      .components.schemas.Applicant.properties,
  score: async (applicant: Applicant, policy?: Partial<Policy>): Promise<ScoreResponse> => {
    const res = await fetch(API_BASE + "/score", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ applicant, policy }),
    });
    return res.ok ? res.json() : fail(res);
  },
  scoreBatch: async (csv: string | Blob, policy: Partial<Policy> = {}): Promise<string> => {
    const { delta: _delta, ...rest } = policy; // /score_batch has no delta parameter
    const res = await fetch(`${API_BASE}/score_batch?${policyQuery(rest)}`, { method: "POST", body: csvForm(csv) });
    return res.ok ? res.text() : fail(res);
  },
  simulate: (policy: Partial<Policy>) => getJson<SimulationResponse>(`/policy_simulation?${policyQuery(policy)}`),
  monitoring: async (csv: string | Blob): Promise<MonitoringResponse> => {
    const res = await fetch(API_BASE + "/monitoring", { method: "POST", body: csvForm(csv) });
    return res.ok ? res.json() : fail(res);
  },
};
