import { Caveat, PageHeader, Panel } from "../components/ui";
import { useModelCard } from "../lib/context";
import { evidence, pick } from "../lib/evidence";
import { int, num, pct } from "../lib/format";
import { useAsync } from "../lib/useAsync";

export default function AboutScreen() {
  const card = useModelCard();
  const data = useAsync(async () => ({ s12: await evidence.step12(), s14: await evidence.step14() }), []);
  const s12 = data.data?.s12, s14 = data.data?.s14;
  const c3 = s12 && pick(s12.A_main, /^r=5,c=0\.3$/);
  const main = card.main_policy_on_test;
  const young = s12?.F_age_bands_main_config["<30"].global_cap.decline_rate;
  const limits: [string, string][] = [
    ["The costs are assumptions.", `The cost ratio r, the referral cost c and the analyst budget are set by the lender, not measured. Change them and the best policy changes.${
      c3 ? ` At c = 0.3 the main CreditSense policy costs ${num(pick(c3, /^CreditSense alpha=0\.10, 20% budget$/).cost_per_1000, 0)} per 1,000 against ${num(pick(c3, /^Phase I rule/).cost_per_1000, 0)} for the Phase I rule.` : ""}`],
    ["Analysts are assumed to be right.", "The cost of a referral assumes the analyst decides every referred case correctly. Real analysts make mistakes."],
    ["There is no test over time.", "The data has no dates, so the model was never tested on a later period. Every drift result here is simulated. The data describes US borrowers from around 2011."],
    ["The guarantee is an average over many applicants.", "It limits the share of all defaulters who are auto-approved, on average. It does not hold for one batch, for one group, or for one applicant, and it needs new applicants to resemble the calibration population."],
    ["Age is in the model.", `Age is a model input${s14 ? ` and one of the four largest drivers for ${pct(s14.age_among_top4_drivers_share, 0)} of referred or declined applicants` : ""}. It is never given as a reason. Using age in credit decisions is restricted by law in many places.`],
    ["Income and debt ratio are not used.", "Feature selection dropped them. A lender would expect them, and a scorecard built on all attributes shows they carry signal."],
    ["The main policy is strict.", `It declines ${pct(main.decline_rate)} of applicants overall${young !== undefined ? ` and ${pct(young, 0)} of those under 30` : ""}. Compare it with the 20% cap in the Policy simulator.`],
    ["The comparison with the base paper is indirect.", "The paper does not describe its evaluation protocol. The audit shows a mechanism that can produce its numbers. That is consistent with, not proof of, what the paper did."],
    ["The other datasets were run with fixed settings.", "The German and Australian datasets are small, so their intervals are wide, and the Taiwan run used one repeat."],
  ];
  return (
    <>
      <PageHeader title="About and limitations"
        lead="CreditSense is a research prototype built for an M.Tech project at PES University. It supports a credit decision; it does not make one. A person confirms every decline." />
      <div className="space-y-4">
        <Caveat>{card.disclaimer}</Caveat>
        <Panel title="What it does">
          <ol className="max-w-[75ch] list-decimal space-y-1 pl-5">
            <li>Estimates a calibrated probability that the applicant becomes 90 days past due within two years.</li>
            <li>Places the applicant in one of {Object.keys(card.tiers).length} risk tiers.</li>
            <li>Recommends approve, refer or decline under an explicit policy, with a cap on the share of defaulters auto-approved.</li>
            <li>Gives the reasons for the score, computed exactly for the model that produced it.</li>
          </ol>
          <p className="num mt-3 max-w-[75ch] text-sm text-muted">
            Model v{card.version}, built {card.built} on {String(card.data.source)}: a stacked ensemble of tree models with Platt calibration, tested on{" "}
            {int(Number(card.data.test_rows))} held-out applicants. {card.intended_use}
          </p>
        </Panel>
        <Panel title="Limitations, in plain words">
          <dl className="max-w-[80ch] space-y-3">
            {limits.map(([head, text]) => (
              <div key={head}><dt className="font-semibold">{head}</dt><dd className="num">{text}</dd></div>
            ))}
          </dl>
        </Panel>
        <Panel title="What is never claimed">
          <ul className="max-w-[75ch] list-disc space-y-1 pl-5">
            <li>Higher accuracy than the base paper.</li>
            <li>That stacking beats a single model: the gain was not significant on any of the four datasets.</li>
            <li>A guarantee for an individual applicant.</li>
            <li>A fairness certificate: only age was audited, and the data has no other protected attributes.</li>
          </ul>
        </Panel>
        <Panel title="Project">
          <p className="max-w-[75ch]">
            Vandhana Murugan, M.Tech Data Science and AI, PES University. Guide: Ramesh Prakash Guledgudd. Base paper: Emmanuel, Sun and Wang (2024),
            Journal of Big Data 11:23, <a className="text-accent underline" href="https://doi.org/10.1186/s40537-024-00882-0">doi:10.1186/s40537-024-00882-0</a>.
            Code: <a className="text-accent underline" href="https://github.com/VandhanaMurugan/CreditSense">github.com/VandhanaMurugan/CreditSense</a>.
          </p>
        </Panel>
      </div>
    </>
  );
}
