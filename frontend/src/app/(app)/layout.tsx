"use client";

import { useState, useEffect } from "react";
import { usePathname } from "next/navigation";
import { Sidebar } from "@/components/sidebar";
import { CommandPalette } from "@/components/command-palette";
import { Onboarding } from "@/components/onboarding";
import { PortfolioProvider, usePortfolio } from "@/lib/portfolio-context";
import { getGroupForHref, getNavItemForHref } from "@/lib/nav";

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

  return <AppShell>{children}</AppShell>;
}

function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [privacy, setPrivacy] = useState(false);
  const [dark, setDark] = useState(false);
  const [rail, setRail] = useState(false);

  const group = getGroupForHref(pathname);
  const navColor = group?.color || "#6366f1";
  const navItem = getNavItemForHref(pathname);
  const pageLabel = navItem?.label || "";

  // Load saved theme on mount
  useEffect(() => {
    const saved = localStorage.getItem("mo-theme");
    if (saved === "dark") { setDark(true); document.documentElement.classList.add("dark"); }
  }, []);

  const toggleDark = () => {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    localStorage.setItem("mo-theme", next ? "dark" : "light");
  };

  return (
    <div className={`flex h-screen ${privacy ? "privacy" : ""}`}>
      <Sidebar
        privacy={privacy} onTogglePrivacy={() => setPrivacy(!privacy)}
        dark={dark} onToggleDark={toggleDark}
        rail={rail} onToggleRail={() => setRail(!rail)}
      />
      <main className="flex-1 flex flex-col min-w-0" style={{ background: "var(--bg)" }}>
        <header className="h-[56px] flex items-center px-6 gap-5 sticky top-0 z-30"
                style={{ background: "var(--header-bg)", backdropFilter: "saturate(160%) blur(10px)", borderBottom: "1px solid var(--border)" }}>
          <div className="flex items-center gap-2 text-[13px] text-[var(--ink-3)]">
            <span className="font-semibold" style={{ color: navColor }}>{group?.label}</span>
            <span className="text-[#b6bac7]">/</span>
            <span className="text-[var(--ink)] font-semibold">{pageLabel}</span>
          </div>
          <div className="flex-1" />
          {/* Tape status pill */}
          <div className="flex items-center gap-1.5 h-[30px] px-3 rounded-full text-xs font-medium bg-[var(--surface)] border border-[var(--border)] text-[var(--ink-2)]">
            <span className="w-1.5 h-1.5 rounded-full bg-[#08a86b]" style={{ boxShadow: "0 0 0 3px #e5f7ee", animation: "pulse-dot 2s ease-in-out infinite" }} />
            <span>Confirmed Uptrend · since Apr 11</span>
          </div>
          <button className="flex items-center gap-1.5 h-[30px] px-3 rounded-full text-xs font-medium bg-[var(--surface)] border border-[var(--border)] text-[var(--ink-2)] hover:bg-[#eef0f6] transition-colors"
                  onClick={() => document.dispatchEvent(new KeyboardEvent("keydown", { key: "k", metaKey: true }))}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
            Quick jump
            <kbd className="text-[10px] bg-[#eef0f6] border border-[var(--border)] rounded px-1" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>⌘K</kbd>
          </button>
        </header>
        <div className="flex-1 overflow-auto px-7 py-6">
          {children}
        </div>
      </main>
      <CommandPalette />

      <style jsx global>{`
        @keyframes slide-up { from { opacity: 0; transform: translateY(6px); } }
        @keyframes pulse-dot {
          0%, 100% { box-shadow: 0 0 0 3px currentColor; }
          50% { box-shadow: 0 0 0 6px transparent; }
        }
      `}</style>
    </div>
  );
}
