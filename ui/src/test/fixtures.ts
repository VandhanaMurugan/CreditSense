import type { ModelCard, ScoreResponse, SimulationResponse, RuleResult } from "../api";

// Deliberately odd numbers: if the UI shows them, it read them from the service and did not hard-code its own.
export const rule = (over: Partial<RuleResult> = {}): RuleResult => ({
  cost_per_1000: 311.7, approve_rate: 0.512, refer_rate: 0.213, decline_rate: 0.275, defaulters_auto_approved: 0.0873,
  goods_auto_declined: 0.2311, default_rate_in_approved: 0.0139, ...over,
});
export const card: ModelCard = {
  version: "9.9.9", built: "2031-01-02", data: { source: "Fixture data", fit_rows: 111, calibration_rows: 22, test_rows: 3333 },
  features: ["age"], test_metrics: { AUC: 0.8123, PR_AUC: 0.3456, KS: 0.5012, Brier: 0.0567, ECE: 0.0089 },
  main_policy_on_test: rule(),
  tiers: {
    "1": { pd_range: [0, 0.015], share_of_applicants: 0.31, observed_default_rate: 0.0061, mean_predicted_pd: 0.007 },
    "2": { pd_range: [0.015, 0.06], share_of_applicants: 0.39, observed_default_rate: 0.0312, mean_predicted_pd: 0.03 },
    "3": { pd_range: [0.06, 1], share_of_applicants: 0.3, observed_default_rate: 0.2222, mean_predicted_pd: 0.21 },
  },
  policy_default: { r: 7, c: 0.15, rho: 0.25, alpha: 0.12, delta: null },
  intended_use: "Fixture use.", limitations: ["Fixture limitation one"], disclaimer: "Fixture disclaimer: decision support only.",
};
export const score = (over: Partial<ScoreResponse> = {}): ScoreResponse => ({
  pd: 0.1337, risk_tier: 3, tier_observed_default_rate: 0.2222, decision: "decline",
  decision_rule: "Decline (analyst confirms): fixture rule text.", policy: card.policy_default,
  pd_ranges: { approve: [0, 0.0314], refer: [0.0314, 0.0927], decline: [0.0927, 1] },
  guarantee: "At most 12% of applicants who go on to default are auto-approved (fixture).", warnings: [],
  model_version: "9.9.9", disclaimer: card.disclaimer,
  reasons: [
    { code: "utilisation", text: "High use of revolving credit limits", direction: "raises risk", contribution_logodds: 1.2345 },
    { code: "open_lines", text: "Number of open credit lines and loans", direction: "lowers risk", contribution_logodds: -0.4321 },
  ],
  adverse_action_reasons: [{ code: "utilisation", text: "High use of revolving credit limits", direction: "raises risk", contribution_logodds: 1.2345 }],
  protected_attribute_contribution: { age_logodds: 0.2718, note: "Age is a model input. Fixture note." },
  explanation_check: { base_logodds: -3.5, sum_of_contributions: 1.6311, model_logodds: -1.8689 },
  ...over,
});
export const simulation = (over: Partial<SimulationResponse> = {}): SimulationResponse => ({
  policy: card.policy_default, pd_ranges: score().pd_ranges, n_test: 3333,
  results: { CreditSense: rule(), "Same budget, no guarantee": rule({ cost_per_1000: 140.4, defaulters_auto_approved: 0.2666 }), "Two-way cost threshold": rule({ cost_per_1000: 222.2, refer_rate: 0, defaulters_auto_approved: 0.4111 }) },
  group_rates: { attribute: "age", note: "Fixture group note.", groups: { "<30": { ...rule({ decline_rate: 0.4567 }), n: 77, observed_default_rate: 0.1299 }, "70+": { ...rule({ defaulters_auto_approved: 0.2468 }), n: 55, observed_default_rate: 0.0211 } } },
  ...over,
});
export const SAMPLE_CSV =
  "applicant_id,RevolvingUtilizationOfUnsecuredLines,age,NumberOfTime30-59DaysPastDueNotWorse,NumberOfTime60-89DaysPastDueNotWorse,NumberOfTimes90DaysLate,NumberOfOpenCreditLinesAndLoans,NumberRealEstateLoansOrLines\n" +
  "Z001,0.25,51,0,0,0,6,1\nZ002,0.97,29,2,1,1,3,0\n";
export const SCORED_CSV =
  "applicant_id,RevolvingUtilizationOfUnsecuredLines,age,NumberOfTime30-59DaysPastDueNotWorse,NumberOfTime60-89DaysPastDueNotWorse,NumberOfTimes90DaysLate,NumberOfOpenCreditLinesAndLoans,NumberRealEstateLoansOrLines,pd,risk_tier,decision,warnings\n" +
  'Z001,0.25,51,0,0,0,6,1,0.0123,1,approve,\nZ002,0.97,29,2,1,1,3,0,0.4242,3,refer,"age = 5 is outside the range, check"\n';
export const SCHEMA = { components: { schemas: { Applicant: { properties: {
  RevolvingUtilizationOfUnsecuredLines: { type: "number", minimum: 0, description: "Revolving balance / credit limits" },
  age: { type: "number", minimum: 18, maximum: 120 },
  "NumberOfTime30-59DaysPastDueNotWorse": { type: "integer", minimum: 0 }, "NumberOfTime60-89DaysPastDueNotWorse": { type: "integer", minimum: 0 },
  NumberOfTimes90DaysLate: { type: "integer", minimum: 0 }, NumberOfOpenCreditLinesAndLoans: { type: "integer", minimum: 0 }, NumberRealEstateLoansOrLines: { type: "integer", minimum: 0 },
} } } } };

export interface Call { url: string; method: string; body?: unknown }
type Handler = (call: Call) => unknown;
/** Replaces fetch with a router over the API paths and the repo copies; returns the list of calls made. */
export function mockFetch(routes: Record<string, Handler | unknown>): Call[] {
  const calls: Call[] = [];
  vi.stubGlobal("fetch", vi.fn(async (input: string, init?: RequestInit) => {
    const url = String(input);
    const call: Call = { url, method: init?.method ?? "GET", body: typeof init?.body === "string" ? JSON.parse(init.body) : undefined };
    calls.push(call);
    const key = Object.keys(routes).find((k) => url.includes(k));
    if (!key) return new Response("not found", { status: 404, statusText: "Not Found" });
    const value = routes[key];
    const out = typeof value === "function" ? (value as Handler)(call) : value;
    if (out instanceof Response) return out;
    return typeof out === "string" ? new Response(out) : new Response(JSON.stringify(out), { headers: { "Content-Type": "application/json" } });
  }));
  return calls;
}
