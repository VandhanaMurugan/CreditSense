// Shapes of the repository's experiment outputs (phase2/outputs/*.json, results/*.json) that the UI reads.
import type { RuleResult } from "../api";
import { repoJson } from "./repo";

export interface AgeBand { n_test: number; calibration_defaulters: number; cap_band: number; global_cap: RuleResult; per_band_cap: RuleResult }
export interface Validity { mean: number; p95: number; share_of_splits_above_alpha: number }
export interface Step12 {
  assumptions: Record<string, string | number>;
  A_main: Record<string, Record<string, RuleResult>>;
  "B_price_of_guarantee_r5_c0.2": Record<string, { chow_same_budget: RuleResult; creditsense_by_alpha: Record<string, RuleResult> }>;
  D_validity_500_splits: Record<string, { marginal: Validity; pac: Validity; chow_budget: Validity }>;
  D_note: string;
  E_label_shift_main_config: Record<string, { observed_default_rate: number; mean_predicted_pd: number; ECE: number; creditsense: RuleResult }>;
  F_age_bands_main_config: Record<string, AgeBand> & { _overall: { global_cap: RuleResult; per_band_cap: RuleResult } };
}
export interface Step8 { alpha: number; bands: Record<string, { n: number; default_rate: number; mean_predicted_pd: number; ECE_in_band: number; AUC: number; defaulter_coverage: number }> }
export interface Metrics { AUC: number; PR_AUC: number; KS: number; Brier: number; ECE: number }
export interface Step1 { A_phase1_stack: Metrics; B_weighted_gb_in_stack: Metrics; C_weighted_and_tuned: Metrics }
export interface Step2 { calibration_method_cv_brier: Record<string, number>; chosen_method: string; base_rate_test: number;
  mean_predicted_pd_raw: number; mean_predicted_pd_calibrated: number; test_raw: Metrics; test_calibrated: Metrics }
export interface Step3 { n_bootstrap: number; models: Record<string, { AUC: number; AUC_CI95: [number, number]; PR_AUC: number; PR_AUC_CI95: [number, number] }>;
  paired_differences: Record<string, { mean_diff: number; CI95: [number, number]; CI_excludes_zero: boolean }> }
export interface Step6Row { repeats: number; AUC: Record<"stack" | "xgb" | "lr", [number, number]>; stack_minus_xgb_AUC: [number, number]; stack_minus_lr_AUC: [number, number] }
export type Step6 = Record<string, Step6Row>;
export interface Dataset { n: number; positive_rate: number; folds: number; best_single_tree: string; mean_AUC: Record<string, number>;
  tests_AUC_nadeau_bengio: Record<string, { mean_diff: number; CI95: [number, number]; p_value: number }> }
export type Step9 = Record<string, Dataset>;
export type Step10 = Record<string, Record<string, Record<string, number>>>;
export interface Fold { mean_AUC: number; sd: number; folds: number }
export type Step11 = Record<string, Record<string, { Stack: Fold; "Best tree": Fold }> & { best_tree_model: string }> & { base_paper_reported_AUC: Record<string, number> };
export interface Step14 { n_explained: number; additivity_abs_error: { max: number; mean: number }; stability_top1_agreement: Record<string, number>;
  stability_top3_jaccard_other_background: number; xgb_only_shortcut_top1_agreement: number; age_among_top4_drivers_share: number; top_reason_frequency: Record<string, number> }
export interface Step15Scenario { psi_score: { value: number; status: string }; observed_default_rate: number; mean_predicted_pd: number; ECE: number }
export interface Step16 { model: string; test: Metrics; score_points_test: Record<string, number>; main_policy_with_CreditSense_triage: RuleResult }
export interface Phase1 { full_5model_comparison: Record<string, { AUC: number; Accuracy: number; F1: number; precision: number; recall: number; pr_auc: number }> }

const p2 = <T,>(name: string) => () => repoJson<T>(`phase2/${name}.json`);
export const evidence = {
  step1: p2<Step1>("p2_step1_stack_ablation"),
  step2: p2<Step2>("p2_step2_calibration"),
  step3: p2<Step3>("p2_step3_bootstrap"),
  step6: p2<Step6>("p2_step6_size_study"),
  step8: p2<Step8>("p2_step8_age_fairness"),
  step9: async (): Promise<Step9> => ({
    ...(await repoJson<Step9>("phase2/p2_step9_multidataset_German_Australian.json")),
    ...(await repoJson<Step9>("phase2/p2_step9_multidataset_Taiwan.json")),
  }),
  step10: p2<Step10>("p2_step10_protocol_audit"),
  step11: p2<Step11>("p2_step11_stack_protocol_audit"),
  step12: p2<Step12>("p2_step12_decision_policy"),
  step14: p2<Step14>("p2_step14_reason_codes"),
  step15: p2<Record<string, Step15Scenario>>("p2_step15_monitoring"),
  step16: p2<Step16>("p2_step16_scorecard_baseline"),
  phase1: () => repoJson<Phase1>("phase1/step5_results.json"),
};
export const AGE_BANDS = (f: Step12["F_age_bands_main_config"]) => Object.entries(f).filter(([k]) => !k.startsWith("_")) as [string, AgeBand][];
/** Finds a key by pattern so wording changes in the experiment labels fail loudly instead of showing a wrong row. */
export function pick<T>(obj: Record<string, T>, pattern: RegExp): T {
  const hit = Object.keys(obj).filter((k) => pattern.test(k));
  if (hit.length !== 1) throw new Error(`Expected one entry matching ${pattern} in the experiment output, found ${hit.length}.`);
  return obj[hit[0]];
}
