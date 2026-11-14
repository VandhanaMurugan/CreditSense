import { Component, useEffect, useState, type ComponentType, type ReactNode } from "react";
import { ErrorNote } from "./components/ui";
import { API_BASE } from "./api";
import { ModelCardProvider, useModelCard } from "./lib/context";
import { PolicyProvider } from "./lib/policy";
import AboutScreen from "./screens/About";
import BatchScreen from "./screens/Batch";
import DecisionScreen from "./screens/Decision";
import EvidenceScreen from "./screens/Evidence";
import FairnessScreen from "./screens/Fairness";
import HealthScreen from "./screens/Health";
import PolicyScreen from "./screens/Policy";

interface Screen { id: string; label: string; view: ComponentType }
export const SCREENS: Screen[] = [
  { id: "decision", label: "Applicant decision", view: DecisionScreen },
  { id: "policy", label: "Policy simulator", view: PolicyScreen },
  { id: "batch", label: "Batch scoring", view: BatchScreen },
  { id: "health", label: "Model health", view: HealthScreen },
  { id: "fairness", label: "Fairness and reliability", view: FairnessScreen },
  { id: "evidence", label: "Research evidence", view: EvidenceScreen },
  { id: "about", label: "About and limitations", view: AboutScreen },
];

function useHashRoute(): string {
  const read = () => window.location.hash.replace(/^#\/?/, "") || SCREENS[0].id;
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const on = () => setRoute(read());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

/** A screen that cannot read its evidence says so; it never shows a guessed number. */
class ScreenBoundary extends Component<{ children: ReactNode }, { error?: string }> {
  state: { error?: string } = {};
  static getDerivedStateFromError(e: Error) { return { error: e.message }; }
  render() {
    return this.state.error ? <ErrorNote message="This screen could not be drawn from the repository outputs." hint={this.state.error} /> : this.props.children;
  }
}

function Shell() {
  const card = useModelCard();
  const route = useHashRoute();
  const screen = SCREENS.find((s) => s.id === route) ?? SCREENS[0];
  const View = screen.view;
  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[15rem_1fr]">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:bg-surface focus:p-2">Skip to content</a>
      <aside className="bg-ink text-white lg:sticky lg:top-0 lg:h-screen">
        <div className="px-4 pt-4 pb-2 lg:pb-4">
          <p className="font-serif text-xl">CreditSense</p>
          <p className="text-sm opacity-80">Credit decision support</p>
        </div>
        <nav aria-label="Screens">
          <ul className="flex overflow-x-auto lg:block">
            {SCREENS.map((s) => (
              <li key={s.id}>
                <a href={`#/${s.id}`} aria-current={s.id === screen.id ? "page" : undefined}
                  className={`block border-b-4 px-4 py-2 whitespace-nowrap lg:border-b-0 lg:border-l-4 ${s.id === screen.id ? "border-white bg-white/10 font-semibold" : "border-transparent opacity-85 hover:opacity-100"}`}>
                  {s.label}
                </a>
              </li>
            ))}
          </ul>
        </nav>
        <p className="num hidden px-4 pt-6 text-sm opacity-70 lg:block">Model v{card.version}, built {card.built}<br />API {API_BASE}</p>
      </aside>
      <main id="main" className="mx-auto w-full max-w-[84rem] p-4 md:p-6"><ScreenBoundary key={screen.id}><View /></ScreenBoundary></main>
    </div>
  );
}

export default function App() {
  return <ModelCardProvider><PolicyProvider><Shell /></PolicyProvider></ModelCardProvider>;
}
