import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "../App";
import { bandX } from "../components/PdBand";
import { parseCsv } from "../lib/csv";
import { card, mockFetch, rule, SAMPLE_CSV, SCHEMA, score, SCORED_CSV, simulation } from "./fixtures";

const base = (extra: Record<string, unknown> = {}) => ({
  "/model_card": card, "/openapi.json": SCHEMA, "sample_applicants.csv": SAMPLE_CSV, "/score_batch": SCORED_CSV,
  "/score": score(), "/policy_simulation": simulation(), ...extra,
});
const open = (hash: string) => { window.location.hash = `#/${hash}`; return render(<App />); };
afterEach(() => { cleanup(); vi.unstubAllGlobals(); window.location.hash = ""; });

describe("Applicant decision screen", () => {
  it("shows the service's numbers, decision, rule, reasons, guarantee and limitation", async () => {
    mockFetch(base());
    open("decision");
    const result = await screen.findByTestId("result");
    expect(within(result).getByTestId("pd")).toHaveTextContent("13.4%");
    const chip = within(result).getAllByTestId("decision-chip")[0];
    expect(chip).toHaveAttribute("data-decision", "decline");
    expect(chip).toHaveTextContent("Decline");                                   // a word, not colour alone
    expect(within(result).getByTestId("decision-rule")).toHaveTextContent("fixture rule text");
    expect(within(result).getByTestId("guarantee")).toHaveTextContent("At most 12%");
    expect(within(result).getByText(/22\.2% of this tier defaulted/)).toBeInTheDocument();
    expect(within(result).getByText("High use of revolving credit limits")).toBeInTheDocument();
    expect(within(result).getByText(/\+1\.23/)).toBeInTheDocument();
    expect(within(result).getByText(/PD 3\.14% to 9\.27%/)).toBeInTheDocument();    // band limits from pd_ranges
    expect(screen.getAllByText(/Fixture disclaimer/).length).toBeGreaterThan(0);  // limitation on the decision screen
    expect(screen.getByTestId("active-policy")).toHaveTextContent("r 7, c 0.15, analyst budget 25%, cap 12%");
  });

  it("builds presets from the sample file scored by the service, and sends the chosen applicant with the policy", async () => {
    const calls = mockFetch(base());
    open("decision");
    const select = await screen.findByRole("combobox");
    await waitFor(() => expect(within(select).getAllByRole("option")).toHaveLength(3));
    expect(within(select).getByRole("option", { name: /Z002: PD 42\.4%, refer \(input warning\)/ })).toBeInTheDocument();
    await userEvent.selectOptions(select, "Z002");
    await waitFor(() => {
      const posts = calls.filter((c) => c.url.endsWith("/score") && c.method === "POST");
      const last = posts[posts.length - 1].body as { applicant: Record<string, number>; policy: Record<string, number> };
      expect(last.applicant).toMatchObject({ age: 29, RevolvingUtilizationOfUnsecuredLines: 0.97, NumberOfTimes90DaysLate: 1 });
      expect(last.policy).toMatchObject({ r: 7, alpha: 0.12 });
    });
  });

  it("takes field limits from the API schema", async () => {
    mockFetch(base());
    open("decision");
    const age = await screen.findByLabelText(/^Age/);
    await waitFor(() => expect(age).toHaveAttribute("min", "18"));
    expect(age).toHaveAttribute("max", "120");
    expect(screen.getByLabelText(/^90\+ days late/)).toHaveAttribute("step", "1");
  });

  it("shows warnings and switches to adverse-action reasons without ever listing age", async () => {
    mockFetch(base({ "/score": score({ decision: "refer", warnings: ["RevolvingUtilizationOfUnsecuredLines = 5000 is outside the range"] }) }));
    open("decision");
    expect(await screen.findByRole("alert")).toHaveTextContent("5000 is outside the range");
    expect(screen.getAllByTestId("decision-chip")[0]).toHaveTextContent("Refer to analyst");
    await userEvent.click(screen.getByRole("tab", { name: "Adverse-action notice" }));
    const panel = screen.getByText(/Only factors that raised the risk/).closest("section")!;
    expect(within(panel).getAllByRole("listitem")).toHaveLength(1);
    expect(within(panel).queryByText(/^Age/)).not.toBeInTheDocument();
  });

  it("reports a service error instead of a stale or invented decision", async () => {
    mockFetch(base({ "/score": () => new Response(JSON.stringify({ detail: [{ loc: ["body", "applicant", "age"], msg: "Input should be greater than or equal to 18" }] }), { status: 422 }) }));
    open("decision");
    expect(await screen.findByRole("alert")).toHaveTextContent("age: Input should be greater than or equal to 18");
    expect(screen.queryByTestId("result")).not.toBeInTheDocument();
  });
});

describe("Policy simulator", () => {
  it("shows three rules and group rates from the service", async () => {
    mockFetch(base());
    open("policy");
    const rules = await screen.findByTestId("rules");
    expect(within(rules).getAllByRole("heading", { level: 3 }).map((h) => h.textContent)).toEqual(["CreditSense", "Same budget, no guarantee", "Two-way cost threshold"]);
    expect(within(rules).getByText("311.7")).toBeInTheDocument();
    expect(within(rules).getByText("41.1%")).toBeInTheDocument();
    expect(within(rules).getByText(/Within the 12% cap/)).toBeInTheDocument();
    expect(screen.getByText("45.7%")).toBeInTheDocument();          // decline rate of the <30 band
    expect(screen.getByText(/Fixture group note/)).toBeInTheDocument();
    expect(screen.getByText(/3,333 held-out test applicants/)).toBeInTheDocument();
  });

  it("says so when the cap is exceeded on the test set", async () => {
    mockFetch(base({ "/policy_simulation": simulation({ results: { ...simulation().results, CreditSense: rule({ defaulters_auto_approved: 0.131 }) } }) }));
    open("policy");
    expect(await screen.findByText(/Above the 12% cap on this test set/)).toBeInTheDocument();
  });

  it("re-queries the service when the policy changes, and the preset sets the 20% cap", async () => {
    const calls = mockFetch(base());
    open("policy");
    await screen.findByTestId("rules");
    await userEvent.click(screen.getByRole("button", { name: /Business-friendly/ }));
    await waitFor(() => expect(calls.some((c) => c.url.includes("/policy_simulation") && c.url.includes("alpha=0.2"))).toBe(true));
    fireEvent.change(screen.getByRole("slider", { name: /Cost ratio r/ }), { target: { value: "11" } });
    await waitFor(() => expect(calls.some((c) => c.url.includes("r=11"))).toBe(true));
    await userEvent.click(screen.getByLabelText(/Hold the cap with 95% confidence/));
    await waitFor(() => expect(calls.some((c) => c.url.includes("delta=0.05"))).toBe(true));
    await userEvent.click(screen.getByLabelText("Unlimited referrals"));
    await waitFor(() => expect(calls.some((c) => c.url.includes("rho=1&"))).toBe(true));
  });
});

describe("Model health", () => {
  const down = { n: 2000, psi: { score: { value: 0.0174, status: "stable" }, age: { value: 0.3111, status: "act" } }, mean_predicted_pd: 0.0815, observed_default_rate: 0.1335, calibration_in_the_large: 0.052 };
  it("recommends recalibration when outcomes disagree, with status words beside the colours", async () => {
    mockFetch(base({ "/monitoring": down, "batch_downturn_with_outcomes.csv": "a,b\n1,2\n", "p2_step15_monitoring.json": {} }));
    open("health");
    await userEvent.click(await screen.findByRole("button", { name: /downturn/ }));
    const banner = await screen.findByTestId("recalibrate-banner");
    expect(banner).toHaveTextContent("differ by 5.2 points");
    expect(banner).toHaveTextContent("drift alarm alone would have missed this");
    const report = screen.getByTestId("monitoring-report");
    expect(within(report).getByText("Act")).toBeInTheDocument();
    expect(within(report).getByText("0.3111")).toBeInTheDocument();
  });
  it("shows no banner and asks for outcomes when the file has none", async () => {
    mockFetch(base({ "/monitoring": { n: 10, psi: { score: { value: 0.4, status: "act" } }, mean_predicted_pd: 0.1 }, "batch_drift.csv": "a\n1\n", "p2_step15_monitoring.json": {} }));
    open("health");
    await userEvent.click(await screen.findByRole("button", { name: /drifted/ }));
    expect(await screen.findByText("No outcomes")).toBeInTheDocument();
    expect(screen.queryByTestId("recalibrate-banner")).not.toBeInTheDocument();
  });
});

describe("Batch scoring", () => {
  it("summarises the service's scored file", async () => {
    mockFetch(base());
    open("batch");
    await userEvent.click(await screen.findByRole("button", { name: /sample applicants/ }));
    const result = await screen.findByTestId("batch-result");
    expect(within(result).getByText(/Approve 50\.0%, refer 50\.0%, decline 0\.0%/)).toBeInTheDocument();
    expect(within(result).getByText("42.4%")).toBeInTheDocument();
    expect(within(result).getByText("age = 5 is outside the range, check")).toBeInTheDocument();
    expect(within(result).getByRole("button", { name: "Download scored CSV" })).toBeInTheDocument();
  });
});

describe("Evidence-backed screens", () => {
  it("say what is missing instead of showing numbers when a repository output cannot be read", async () => {
    mockFetch(base());                                   // no phase2/*.json routes -> 404
    open("evidence");
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load the experiment outputs");
  });
  it("About shows the model card's disclaimer and limits", async () => {
    mockFetch(base());
    open("about");
    expect(await screen.findByText(/Fixture disclaimer/)).toBeInTheDocument();
    expect(screen.getByText(/declines 27\.5% of applicants overall/)).toBeInTheDocument();
  });
});

describe("helpers", () => {
  it("parses quoted CSV cells", () => {
    expect(parseCsv(SCORED_CSV)[1].warnings).toBe("age = 5 is outside the range, check");
    expect(parseCsv(SCORED_CSV)).toHaveLength(2);
  });
  it("band scale is monotone and bounded", () => {
    expect(bandX(0)).toBe(0); expect(bandX(1)).toBe(100); expect(bandX(0.02)).toBeLessThan(bandX(0.07));
  });
  it("tells the user how to start the service when it is down", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("Failed to fetch"); }));
    open("decision");
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/not running/);
    expect(alert).toHaveTextContent(/uvicorn creditsense\.api:app/);
  });
  it("says the service is running but failed to load the model when it answers without a readable reply", async () => {
    // A crash inside a request reaches the page as a failed fetch; the no-cors probe still gets through.
    vi.stubGlobal("fetch", vi.fn(async (_url: string, init?: RequestInit) => {
      if (init?.mode === "no-cors") return new Response(null);
      throw new TypeError("Failed to fetch");
    }));
    open("decision");
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/is running, but it failed to load the model/);
    expect(alert).toHaveTextContent(/terminal window/);
  });
});
