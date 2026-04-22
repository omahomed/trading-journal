"use client";

import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { api, setActivePortfolio, type Portfolio } from "@/lib/api";

interface PortfolioCtx {
  portfolios: Portfolio[];
  activePortfolio: Portfolio | null;
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

const PortfolioContext = createContext<PortfolioCtx | null>(null);

export function PortfolioProvider({ children }: { children: ReactNode }) {
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchPortfolios = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.listPortfolios();
      const list = Array.isArray(data) ? data : [];
      setPortfolios(list);
      // Sync the module-level global that API defaults read from. Empty name
      // when list is empty — caller should never fire API calls before the
      // onboarding gate clears anyway.
      setActivePortfolio(list[0]?.name ?? "");
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

  const value: PortfolioCtx = {
    portfolios,
    activePortfolio: portfolios[0] ?? null,
    loading,
    error,
    refetch: fetchPortfolios,
  };

  return <PortfolioContext.Provider value={value}>{children}</PortfolioContext.Provider>;
}

export function usePortfolio(): PortfolioCtx {
  const ctx = useContext(PortfolioContext);
  if (!ctx) throw new Error("usePortfolio must be used inside PortfolioProvider");
  return ctx;
}
