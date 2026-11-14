import { createContext, useContext, type ReactNode } from "react";
import { api, type ModelCard } from "../api";
import { useAsync } from "./useAsync";
import { ErrorNote, Loading } from "../components/ui";
import { API_BASE } from "../api";

const Ctx = createContext<ModelCard | null>(null);
export const useModelCard = () => {
  const card = useContext(Ctx);
  if (!card) throw new Error("model card not loaded");
  return card;
};

/** Loads the model card once; every screen reads defaults, tiers and limitations from it. */
export function ModelCardProvider({ children }: { children: ReactNode }) {
  const card = useAsync(api.modelCard, []);
  if (card.error) {
    return (
      <div className="p-6">
        <ErrorNote message={`The scoring service at ${API_BASE} did not answer.`}
          hint={`Start it with "make serve" in the repository root, or set VITE_API_BASE_URL. Details: ${card.error}`} />
      </div>
    );
  }
  if (!card.data) return <div className="p-6"><Loading what="the model card" /></div>;
  return <Ctx.Provider value={card.data}>{children}</Ctx.Provider>;
}
