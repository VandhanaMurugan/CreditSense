import { createContext, useContext, useState, type ReactNode } from "react";
import { UNLIMITED_RHO, type Policy } from "../api";
import { pct } from "./format";
import { useModelCard } from "./context";

interface PolicyState { policy: Policy; setPolicy: (p: Policy) => void; isDefault: boolean }
const Ctx = createContext<PolicyState | null>(null);
export const usePolicy = () => useContext(Ctx)!;

/** One policy for the whole app: set in the Policy simulator, used by the decision and batch screens. */
export function PolicyProvider({ children }: { children: ReactNode }) {
  const card = useModelCard();
  const [policy, setPolicy] = useState<Policy>(card.policy_default);
  const isDefault = JSON.stringify(policy) === JSON.stringify(card.policy_default);
  return <Ctx.Provider value={{ policy, setPolicy, isDefault }}>{children}</Ctx.Provider>;
}

export const isUnlimited = (rho: number | null) => rho === null || rho >= UNLIMITED_RHO;
export function describePolicy(p: Policy): string {
  return [
    `r ${p.r}`, `c ${p.c}`,
    `analyst budget ${isUnlimited(p.rho) ? "unlimited" : pct(p.rho!, 0)}`,
    `cap ${p.alpha ? pct(p.alpha, 0) : "off"}${p.alpha && p.delta ? ` at ${pct(1 - p.delta, 0)} confidence` : ""}`,
  ].join(", ");
}
