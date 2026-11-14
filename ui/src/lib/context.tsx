import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { api, serviceReachable, API_BASE, type ModelCard } from "../api";
import { useAsync } from "./useAsync";
import { ErrorNote, Loading } from "../components/ui";

const Ctx = createContext<ModelCard | null>(null);
export const useModelCard = () => {
  const card = useContext(Ctx);
  if (!card) throw new Error("model card not loaded");
  return card;
};

/** Loads the model card once; every screen reads defaults, tiers and limitations from it. */
export function ModelCardProvider({ children }: { children: ReactNode }) {
  const card = useAsync(api.modelCard, []);
  // Only asked when the model card failed: is the service down, or running and failing?
  const [reachable, setReachable] = useState<boolean>();
  useEffect(() => {
    let live = true;
    setReachable(undefined);
    if (card.error) void serviceReachable().then((ok) => live && setReachable(ok));
    return () => { live = false; };
  }, [card.error]);
  if (card.error) {
    if (reachable === undefined) return <div className="p-6"><Loading what="the service status" /></div>;
    return (
      <div className="p-6">
        {reachable ? (
          <ErrorNote message={`The scoring service at ${API_BASE} is running, but it failed to load the model.`}
            hint={`The reason is in the terminal window that runs the service: look at the last lines of the error there. Usual causes are Python packages at different versions from requirements.txt, or a missing model file. Details: ${card.error}`} />
        ) : (
          <ErrorNote message={`Nothing is answering at ${API_BASE}. The scoring service is not running.`}
            hint={`Start it from the repository root with "python -m uvicorn creditsense.api:app" (or "make serve"), then reload this page. If the service runs at another address, set VITE_API_BASE_URL. Details: ${card.error}`} />
        )}
      </div>
    );
  }
  if (!card.data) return <div className="p-6"><Loading what="the model card" /></div>;
  return <Ctx.Provider value={card.data}>{children}</Ctx.Provider>;
}
