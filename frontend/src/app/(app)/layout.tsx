"use client";

import { Onboarding } from "@/components/onboarding";
import { AdaptiveShell } from "@/components/mobile/adaptive-shell";
import { PortfolioProvider, usePortfolio } from "@/lib/portfolio-context";

export default function AppGroupLayout({ children }: { children: React.ReactNode }) {
  return (
    <PortfolioProvider>
      <AppGate>{children}</AppGate>
    </PortfolioProvider>
  );
}

function AppGate({ children }: { children: React.ReactNode }) {
  const { portfolios, loading, error } = usePortfolio();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center text-sm text-[var(--ink-3)]"
           style={{ background: "var(--bg)" }}>
        Loading…
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center p-6"
           style={{ background: "var(--bg)" }}>
        <div className="max-w-md text-center">
          <div className="text-[15px] font-semibold mb-2">Couldn&apos;t load your portfolios</div>
          <div className="text-[12px] text-[#e5484d]">{error}</div>
        </div>
      </div>
    );
  }

  if (portfolios.length === 0) return <Onboarding />;

  return <AdaptiveShell>{children}</AdaptiveShell>;
}
