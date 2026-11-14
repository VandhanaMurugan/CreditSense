import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Table } from "../components/Table";
import { Caveat, ErrorNote, Loading, PageHeader, Panel } from "../components/ui";
import { useModelCard } from "../lib/context";
import { evidence, pick, type Dataset } from "../lib/evidence";
import { int, num, pct, signed } from "../lib/format";
import { repoUrl } from "../lib/repo";
import { useAsync } from "../lib/useAsync";

const FIGURES: [string, string, string][] = [
  ["p2_price_of_guarantee.png", "Price of the guarantee", "Expected cost as the cap α tightens, for several analyst budgets."],
  ["p2_guarantee_validity.png", "Guarantee validity", "Share of defaulters auto-approved over 500 random re-splits."],
  ["p2_calibration.png", "Calibration", "Raw stack scores against Platt-calibrated probabilities."],
  ["p2_reason_codes.png", "Reason codes", "How often each reason leads, and how stable the explanations are."],
  ["p2_monitoring.png", "Monitoring", "PSI and calibration under the simulated drift scenarios."],
  ["p2_size_study.png", "Size study", "Stack against single models as the training set grows."],
];
// Series colours: first three slots of the validated categorical order (blue, orange, aqua); each line is also labelled.
const SERIES = [["stack", "Stack", "#2a78d6"], ["xgb", "XGBoost", "#eb6834"], ["lr", "Logistic regression", "#1baf7a"]] as const;
const sig = (p: number) => (p < 0.05 ? "significant" : "not significant");

export default function EvidenceScreen() {
  const card = useModelCard();
  const all = useAsync(async () => ({
    s1: await evidence.step1(), s2: await evidence.step2(), s3: await evidence.step3(), s6: await evidence.step6(), s9: await evidence.step9(),
    s10: await evidence.step10(), s11: await evidence.step11(), s12: await evidence.step12(), s16: await evidence.step16(), p1: await evidence.phase1(),
  }), []);

  if (all.error) return <ErrorNote message="Could not load the experiment outputs." hint={all.error} />;
  if (!all.data) return <Loading what="the experiment outputs" />;
  const { s1, s2, s3, s6, s9, s10, s11, s12, s16, p1 } = all.data;

  const main = pick(s12.A_main, /^r=5,c=0\.2$/);
  const phase1Rule = pick(main, /^Phase I rule/);
  const LOOSE = "0.20"; // the second operating point reported in the experiment output
  const alpha20 = pick(s12["B_price_of_guarantee_r5_c0.2"], /^rho=0\.2$/).creditsense_by_alpha[LOOSE];
  const paper = s11.base_paper_reported_AUC;
  const german = s11.German;
  const comparison = [
    { system: `Base paper (reported, German)`, auc: `${num(paper.German, 3)}*`, brier: "–", ece: "–", cost: "–", miss: "–", reasons: "No", guarantee: "No" },
    { system: "Phase I system (stack, 0.5 cut-off on raw score)", auc: num(s1.A_phase1_stack.AUC, 4), brier: num(s1.A_phase1_stack.Brier, 4), ece: num(s1.A_phase1_stack.ECE, 4),
      cost: num(phase1Rule.cost_per_1000, 1), miss: pct(phase1Rule.defaulters_auto_approved), reasons: "Global SHAP on XGBoost", guarantee: "No" },
    { system: "Weight-of-evidence scorecard + CreditSense triage", auc: num(s16.test.AUC, 4), brier: num(s16.test.Brier, 4), ece: num(s16.test.ECE, 4),
      cost: num(s16.main_policy_with_CreditSense_triage.cost_per_1000, 1), miss: pct(s16.main_policy_with_CreditSense_triage.defaulters_auto_approved), reasons: "Scorecard points", guarantee: `Yes (${pct(card.policy_default.alpha, 0)})` },
    { system: "CreditSense (main policy)", auc: num(card.test_metrics.AUC, 4), brier: num(card.test_metrics.Brier, 4), ece: num(card.test_metrics.ECE, 4),
      cost: num(card.main_policy_on_test.cost_per_1000, 1), miss: pct(card.main_policy_on_test.defaulters_auto_approved), reasons: "Exact, per applicant", guarantee: `Yes (${pct(card.policy_default.alpha, 0)})` },
    { system: `CreditSense, cap ${pct(Number(LOOSE), 0)}`, auc: num(card.test_metrics.AUC, 4), brier: num(card.test_metrics.Brier, 4), ece: num(card.test_metrics.ECE, 4),
      cost: num(alpha20.cost_per_1000, 1), miss: pct(alpha20.defaulters_auto_approved), reasons: "Exact, per applicant", guarantee: `Yes (${pct(Number(LOOSE), 0)})` },
  ];
  const uci = Object.entries(s9) as [string, Dataset][];
  const sizes = Object.entries(s6).map(([n, v]) => ({ n: Number(n), stack: v.AUC.stack[0], xgb: v.AUC.xgb[0], lr: v.AUC.lr[0] }));
  const diff = pick(s3.paired_differences, /^Stack C.*minus XGBoost \[AUC\]/);
  const stackC = pick(s3.models, /^Stack C/);
  const audits = ["German", "Taiwan"].flatMap((d) => Object.entries(s11[d]).filter(([k]) => /^P\d/.test(k)).map(([protocol, v]) => ({ d, protocol, v: v as { Stack: { mean_AUC: number; sd: number; folds: number }; "Best tree": { mean_AUC: number } } })));

  return (
    <>
      <PageHeader title="Research evidence"
        lead="For examiners: the results behind the product, read from the repository's experiment outputs. Ranking (AUC), probability quality (Brier, ECE) and decision quality (cost, missed defaulters) are kept apart." />
      <div className="space-y-4">
        <Panel title="Comparison with existing systems (Give Me Some Credit test set)">
          <Table caption="Comparison with existing systems" rows={comparison} rowKey={(r) => r.system} highlight={(r) => r.system.startsWith("CreditSense (main")}
            columns={[
              { head: "System", cell: (r) => r.system },
              { head: "AUC", numeric: true, cell: (r) => r.auc },
              { head: "Brier", numeric: true, cell: (r) => r.brier },
              { head: "ECE", numeric: true, cell: (r) => r.ece },
              { head: "Cost per 1,000", numeric: true, cell: (r) => r.cost },
              { head: "Defaulters auto-approved", numeric: true, cell: (r) => r.miss },
              { head: "Reasons", cell: (r) => r.reasons },
              { head: "Guarantee", cell: (r) => r.guarantee },
            ]} />
          <p className="num mt-2 text-sm text-muted">
            *Reported by the base paper on the German dataset, so not comparable with the rows below it. Under a leak-safe protocol the German stack
            reaches {num(german["P0 leak-safe"].Stack.mean_AUC, 3)}; with SMOTE applied before cross-validation it reaches {num(german["P1 SMOTE before CV"].Stack.mean_AUC, 3)}.
          </p>
          <div className="mt-3">
            <Caveat>
              CreditSense does not claim higher accuracy than the base paper or than the scorecard. Its AUC is about the same as the Phase I stack. The difference
              is in the decision layer. Costs assume r = {card.policy_default.r}, c = {String(s12.assumptions.c_referral)} and
              a {pct(Number(s12.assumptions.referral_budget), 0)} analyst budget; {String(s12.assumptions.analyst_accuracy)}.
            </Caveat>
          </div>
        </Panel>

        <Panel title="Does stacking pay off? Four datasets">
          <Table caption="Stack against single models on the base paper's datasets" rows={uci} rowKey={([d]) => d}
            columns={[
              { head: "Dataset", cell: ([d, v]) => `${d} (${int(v.n)} rows, ${v.folds} folds)` },
              ...(["RF", "GB", "XGB", "LR", "Stack_trees"] as const).map((k) => ({ head: k === "Stack_trees" ? "Stack" : k, numeric: true, cell: ([, v]: [string, Dataset]) => num(v.mean_AUC[k], 4) })),
              { head: "Stack − best tree", numeric: true, cell: ([, v]) => { const t = pick(v.tests_AUC_nadeau_bengio, new RegExp(`^Stack_trees - ${v.best_single_tree}$`)); return `${signed(t.mean_diff, 4)} (p ${num(t.p_value)}, ${sig(t.p_value)})`; } },
              { head: "Stack − LR", numeric: true, cell: ([, v]) => { const t = pick(v.tests_AUC_nadeau_bengio, /^Stack_trees - LR$/); return `${signed(t.mean_diff, 4)} (p ${t.p_value < 0.001 ? "< 0.001" : num(t.p_value)}, ${sig(t.p_value)})`; } },
              { head: "Paper reported", numeric: true, cell: ([d]) => (paper[d] !== undefined ? num(paper[d], 3) : "not in repo outputs") },
            ]} />
          <p className="num mt-2 text-sm text-muted">
            Mean AUC over repeated stratified cross-validation, leak-safe, fixed untuned settings, Nadeau–Bengio corrected t-test. On Give Me Some Credit the
            tuned stack scores {num(stackC.AUC, 4)} (95% interval {num(stackC.AUC_CI95[0], 4)} to {num(stackC.AUC_CI95[1], 4)}, {int(s3.n_bootstrap)} bootstrap
            resamples); stack minus XGBoost is {signed(diff.mean_diff, 4)} ({num(diff.CI95[0], 4)} to {num(diff.CI95[1], 4)}), statistically {diff.CI_excludes_zero ? "detectable but negligible in size" : "not significant"}.
          </p>
          <h3 className="mt-4 font-semibold">Gain against training-set size (Give Me Some Credit)</h3>
          <div className="h-64" role="img" aria-label="Line chart of mean AUC by training rows for the stack, XGBoost and logistic regression. The table below has the same values.">
            <ResponsiveContainer>
              <LineChart data={sizes} margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
                <CartesianGrid stroke="#e6e9ef" vertical={false} />
                <XAxis dataKey="n" scale="log" domain={["dataMin", "dataMax"]} type="number" ticks={sizes.map((s) => s.n)} tickFormatter={int}
                  label={{ value: "Training rows (log scale)", position: "insideBottom", offset: -10, fontSize: 12 }} tick={{ fontSize: 12 }} />
                <YAxis domain={["auto", "auto"]} tickFormatter={(v: number) => num(v)} tick={{ fontSize: 12 }} width={44} />
                <Tooltip formatter={(v) => num(Number(v), 4)} labelFormatter={(n) => `${int(Number(n))} rows`} />
                <Legend verticalAlign="top" height={28} formatter={(v) => <span style={{ color: "#14213d" }}>{v}</span>} />
                {SERIES.map(([key, name, color]) => <Line key={key} dataKey={key} name={name} stroke={color} strokeWidth={2} dot={{ r: 4 }} isAnimationActive={false} />)}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <Table caption="Size study values" rows={Object.entries(s6)} rowKey={([n]) => n}
            columns={[
              { head: "Training rows", cell: ([n]) => int(Number(n)) },
              { head: "Stack AUC", numeric: true, cell: ([, v]) => num(v.AUC.stack[0], 4) },
              { head: "XGBoost AUC", numeric: true, cell: ([, v]) => num(v.AUC.xgb[0], 4) },
              { head: "LR AUC", numeric: true, cell: ([, v]) => num(v.AUC.lr[0], 4) },
              { head: "Stack − XGBoost (sd)", numeric: true, cell: ([, v]) => `${signed(v.stack_minus_xgb_AUC[0], 4)} (${num(v.stack_minus_xgb_AUC[1], 4)})` },
              { head: "Stack − LR (sd)", numeric: true, cell: ([, v]) => `${signed(v.stack_minus_lr_AUC[0], 4)} (${num(v.stack_minus_lr_AUC[1], 4)})` },
            ]} />
        </Panel>

        <Panel title="Evaluation-protocol audit">
          <div className="grid gap-4 lg:grid-cols-2">
            <div>
              <h3 className="mb-1 font-semibold">On the stack</h3>
              <Table caption="Protocol audit on the stack" rows={audits} rowKey={(r) => r.d + r.protocol}
                columns={[
                  { head: "Dataset", cell: (r) => r.d },
                  { head: "Protocol", cell: (r) => r.protocol.replace(/^P\d /, "") },
                  { head: "Stack AUC (sd)", numeric: true, cell: (r) => `${num(r.v.Stack.mean_AUC, 4)} (${num(r.v.Stack.sd, 4)})` },
                  { head: "Best tree AUC", numeric: true, cell: (r) => num(r.v["Best tree"].mean_AUC, 4) },
                  { head: "Paper reported", numeric: true, cell: (r) => num(paper[r.d], 3) },
                ]} />
            </div>
            <div>
              <h3 className="mb-1 font-semibold">On single models</h3>
              <Table caption="Protocol audit on single models" rowKey={(r) => r.d + r.protocol}
                rows={Object.entries(s10).flatMap(([d, ps]) => Object.entries(ps).map(([protocol, v]) => ({ d, protocol, v })))}
                columns={[
                  { head: "Dataset", cell: (r) => r.d },
                  { head: "Protocol", cell: (r) => r.protocol.replace(/^P\d /, "") },
                  ...["RF", "XGB", "LR"].map((k) => ({ head: k, numeric: true, cell: (r: { v: Record<string, number> }) => num(r.v[k], 4) })),
                ]} />
            </div>
          </div>
          <div className="mt-3">
            <Caveat>
              Oversampling the whole dataset with SMOTE before cross-validation leaks synthetic copies of test rows into training and lifts AUC to the
              level the base paper reports. SMOTE applied correctly, inside the training folds, gives no gain. This is consistent with, not proof of,
              the base paper's protocol, which the paper does not describe.
            </Caveat>
          </div>
        </Panel>

        <Panel title="Calibration">
          <Table caption="Raw against calibrated scores" rowKey={([k]) => k} rows={[["Raw stack score", s2.test_raw, s2.mean_predicted_pd_raw], ["Platt-calibrated PD", s2.test_calibrated, s2.mean_predicted_pd_calibrated]] as [string, typeof s2.test_raw, number][]}
            columns={[
              { head: "Score", cell: ([k]) => k },
              { head: "Mean predicted PD", numeric: true, cell: ([, , m]) => pct(m) },
              { head: "AUC", numeric: true, cell: ([, v]) => num(v.AUC, 4) },
              { head: "Brier", numeric: true, cell: ([, v]) => num(v.Brier, 4) },
              { head: "ECE", numeric: true, cell: ([, v]) => num(v.ECE, 4) },
            ]} />
          <p className="num mt-2 text-sm text-muted">
            Observed default rate {pct(s2.base_rate_test)}. Calibration changes probability quality, not ranking. Method chosen by a rule fixed in advance:
            5-fold Brier {Object.entries(s2.calibration_method_cv_brier).map(([k, v]) => `${k} ${num(v, 5)}`).join(", ")}; chosen {s2.chosen_method}.
          </p>
        </Panel>

        <Panel title="Phase I results, for continuity">
          <Table caption="Phase I five-model comparison" rows={Object.entries(p1.full_5model_comparison)} rowKey={([m]) => m} highlight={([m]) => m.startsWith("Stacked")}
            columns={[
              { head: "Model", cell: ([m]) => m },
              { head: "AUC", numeric: true, cell: ([, v]) => num(v.AUC, 4) },
              { head: "PR-AUC", numeric: true, cell: ([, v]) => num(v.pr_auc, 4) },
              { head: "Accuracy", numeric: true, cell: ([, v]) => num(v.Accuracy, 4) },
              { head: "Precision", numeric: true, cell: ([, v]) => num(v.precision, 4) },
              { head: "Recall", numeric: true, cell: ([, v]) => num(v.recall, 4) },
              { head: "F1", numeric: true, cell: ([, v]) => num(v.F1, 4) },
            ]} />
          <p className="mt-2 text-sm text-muted">Threshold 0.5 on raw class-weighted scores, same test set. The five models are within a few thousandths of AUC of each other.</p>
        </Panel>

        <Panel title="Figures from the experiments">
          <div className="grid gap-4 md:grid-cols-2">
            {FIGURES.map(([file, title, text]) => (
              <figure key={file} className="border border-line p-2">
                <a href={repoUrl(`phase2/${file}`)}><img src={repoUrl(`phase2/${file}`)} alt={`${title}. ${text}`} className="w-full" /></a>
                <figcaption className="mt-1 text-sm"><span className="font-semibold">{title}.</span> {text} <span className="text-muted">phase2/outputs/{file}</span></figcaption>
              </figure>
            ))}
          </div>
        </Panel>
      </div>
    </>
  );
}
