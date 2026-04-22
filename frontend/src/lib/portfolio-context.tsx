"use client";

import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { api, setActivePortfolio as setApiActive, type Portfolio } from "@/lib/api";

interface PortfolioCtx {
  portfolios: Portfolio[];
  activePortfolio: Portfolio | null;
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
  setActive: (portfolio: Portfolio) => void;
}

const PortfolioContext = createContext<PortfolioCtx | null>(null);

const ACTIVE_STORAGE_KEY = "mo.activePortfolioId";

function readStoredActiveId(): number | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(ACTIVE_STORAGE_KEY);
    if (!raw) return null;
    const n = parseInt(raw, 10);
    return Number.isFinite(n) ? n : null;
  } catch {
    return null;
  }
}

function writeStoredActiveId(id: number | null): void {
  if (typeof window === "undefined") return;
  try {
    if (id == null) window.localStorage.removeItem(ACTIVE_STORAGE_KEY);
    else window.localStorage.setItem(ACTIVE_STORAGE_KEY, String(id));
  } catch { /* ignore quota/private-mode errors */ }
}

export function PortfolioProvider({ children }: { children: ReactNode }) {
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
  const [activeId, setActiveId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchPortfolios = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.listPortfolios();
      const list = Array.isArray(data) ? data : [];
      setPortfolios(list);

      // Pick which portfolio is active. Preference order:
      //   1. The id stored in localStorage (if it still exists in the list)
      //   2. First portfolio in the list
      //   3. None, if the list is empty (onboarding gate)
      const storedId = readStoredActiveId();
      const chosen = list.find((p) => p.id === storedId) ?? list[0] ?? null;
      setActiveId(chosen?.id ?? null);
      setApiActive(chosen?.name ?? "");
      writeStoredActiveId(chosen?.id ?? null);

      setError(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPortfolios();
  }, [fetchPortfolios]);

  const activePortfolio = portfolios.find((p) => p.id === activeId) ?? null;

  const setActive = useCallback((portfolio: Portfolio) => {
    setActiveId(portfolio.id);
    setApiActive(portfolio.name);
    writeStoredActiveId(portfolio.id);
    // Hard reload so every component refetches its data scoped to the new
    // active portfolio. Simple + correct for beta; the alternative (threading
    // usePortfolio() into every component's data effects) is a larger refactor.
    if (typeof window !== "undefined") window.location.reload();
  }, []);

  const value: PortfolioCtx = {
    portfolios,
    activePortfolio,
    loading,
    error,
    refetch: fetchPortfolios,
    setActive,
  };

  return <PortfolioContext.Provider value={value}>{children}</PortfolioContext.Provider>;
}

export function usePortfolio(): PortfolioCtx {
  const ctx = useContext(PortfolioContext);
  if (!ctx) throw new Error("usePortfolio must be used inside PortfolioProvider");
  return ctx;
}
