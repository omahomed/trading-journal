"use client";

import { useState, useEffect } from "react";
import { Sidebar } from "@/components/sidebar";
import { CommandPalette } from "@/components/command-palette";
import { Dashboard } from "@/components/dashboard";
import { getGroupForPage } from "@/lib/nav";

// Mock KPI data
const KPIS = [
  { label: "NET LIQ VALUE", value: "$448,382", sub: "+$16,737 (+3.88%)", gradient: "linear-gradient(135deg, #6366f1, #818cf8)" },
  { label: "LTD RETURN", value: "255.34%", sub: "$+265,320", gradient: "linear-gradient(135deg, #ec4899, #f472b6)" },
  { label: "YTD RETURN", value: "44.49%", sub: "SPY: +4.14% | NDX: +5.28%", gradient: "linear-gradient(135deg, #10b981, #34d399)" },
  { label: "LIVE EXPOSURE", value: "191.3%", sub: "22/15 Pos · Risk: 26.89%", gradient: "linear-gradient(135deg, #f97316, #fb923c)" },
  { label: "DRAWDOWN", value: "-0.00%", sub: "Clear", gradient: "linear-gradient(135deg, #1e40af, #3b82f6)" },
];

function KPITile({ label, value, sub, gradient }: { label: string; value: string; sub: string; gradient: string }) {
  return (
    <div
      className="relative overflow-hidden rounded-[14px] p-[14px_16px] text-white flex flex-col justify-between h-[90px] transition-transform duration-150 hover:scale-[1.01]"
      style={{ background: gradient, boxShadow: "0 4px 14px rgba(14,20,38,0.06), 0 0 0 1px rgba(14,20,38,0.04)" }}
    >
      <div className="absolute -right-5 -top-5 w-[100px] h-[100px] rounded-full" style={{ background: "radial-gradient(circle, rgba(255,255,255,0.18), transparent 65%)" }} />
      <div className="relative z-10">
        <div className="text-[9px] font-semibold uppercase tracking-[0.10em] opacity-85" style={{ fontFamily: "Inter, system-ui, sans-serif" }}>{label}</div>
        <div className="text-[22px] font-semibold tracking-tight mt-0.5 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{value}</div>
      </div>
      <div className="relative z-10 text-[10px] font-medium opacity-80">{sub}</div>
    </div>
  );
}

function DashboardPage({ navColor }: { navColor: string }) {
  const hour = new Date().getHours();
  const greeting = hour < 12 ? "Good morning" : hour < 17 ? "Good afternoon" : "Good evening";

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px] border-b border-[#e6e8ef]">
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          {greeting}, <em className="italic" style={{ color: navColor }}>{" MO"}</em>
        </h1>
        <div className="text-[13px] text-[var(--ink-3)] mt-1.5">
          {new Date().toLocaleDateString("en-US", { weekday: "long", year: "numeric", month: "long", day: "numeric" })} · CanSlim
        </div>
      </div>

      <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium mb-5"
           style={{ background: "#f1ecfe", color: "#8b5cf6" }}>
        <span className="w-1.5 h-1.5 rounded-full bg-[#8b5cf6]" style={{ animation: "pulse-dot 2s ease-in-out infinite", boxShadow: "0 0 0 3px #f1ecfe" }} />
        Tape: Powertrend · since Apr 11
      </div>

      <div className="grid grid-cols-5 gap-3.5 mb-6">
        {KPIS.map((kpi) => (
          <KPITile key={kpi.label} {...kpi} />
        ))}
      </div>

      {/* Two-column: Equity Curve + This Month */}
      <div className="grid gap-[18px]" style={{ gridTemplateColumns: "2fr 1fr", alignItems: "stretch" }}>
        {/* Equity Curve */}
        <div className="bg-[var(--surface)] rounded-[14px] border border-[var(--border)] overflow-hidden flex flex-col" style={{ boxShadow: "0 1px 2px rgba(14,20,38,0.04), 0 0 0 1px rgba(14,20,38,0.04)" }}>
          <div className="flex items-center justify-between px-[18px] py-3 border-b border-[#e6e8ef]">
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
              <span className="text-[13px] font-semibold">Equity Curve</span>
              <span className="text-xs text-[var(--ink-4)]">Portfolio vs SPY / NDX · event markers</span>
            </div>
            <div className="flex bg-[#eef0f6] border border-[var(--border)] rounded-[10px] p-0.5 gap-0.5">
              {["1Y", "6M", "3M", "All"].map((t, i) => (
                <button key={t} className="px-3 py-1 rounded-md text-xs font-medium transition-all"
                        style={{ background: i === 0 ? "#fff" : "transparent", color: i === 0 ? "#0f1524" : "#5a6175", boxShadow: i === 0 ? "0 1px 2px rgba(14,20,38,0.04)" : "none" }}>
                  {t}
                </button>
              ))}
            </div>
          </div>
          <div className="flex-1 min-h-[380px] flex items-center justify-center text-[var(--ink-4)] text-sm">
            [Equity curve chart — Recharts, connected to Supabase]
          </div>
        </div>

        {/* This Month at a Glance */}
        <div className="bg-[var(--surface)] rounded-[14px] border border-[var(--border)] overflow-hidden flex flex-col" style={{ boxShadow: "0 1px 2px rgba(14,20,38,0.04), 0 0 0 1px rgba(14,20,38,0.04)" }}>
          <div className="flex items-center gap-2 px-[18px] py-3 border-b border-[#e6e8ef]">
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
            <span className="text-[13px] font-semibold">This Month at a Glance</span>
            <span className="text-xs text-[var(--ink-4)]">April 2026</span>
          </div>
          <div className="flex-1 p-[18px] flex flex-col gap-3.5">
            {/* Stats grid */}
            <div className="grid grid-cols-2 gap-2.5">
              {[
                { k: "MTD Return", v: "+4.89%", s: "$+20,843", color: "#08a86b" },
                { k: "Trades", v: "14", s: "9W · 5L", color: "#0f1524" },
                { k: "Best Day", v: "+2.14%", s: "Apr 11 · FTD", color: "#08a86b" },
                { k: "Worst Day", v: "-0.62%", s: "Apr 03", color: "#e5484d" },
              ].map((stat) => (
                <div key={stat.k} className="p-3 border border-[var(--border)] rounded-[10px]">
                  <div className="text-[10px] uppercase tracking-[0.10em] text-[var(--ink-4)] font-semibold">{stat.k}</div>
                  <div className="text-[18px] font-semibold mt-1 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace", color: stat.color }}>{stat.v}</div>
                  <div className="text-[11px] text-[var(--ink-4)] mt-0.5">{stat.s}</div>
                </div>
              ))}
            </div>

            {/* Daily P&L mini bars */}
            <div>
              <div className="text-[10px] uppercase tracking-[0.10em] text-[var(--ink-4)] font-semibold mb-2">Daily P&L · last 20 sessions</div>
              <div className="flex items-center gap-[3px] h-[80px]">
                {[0.4,-0.2,0.8,1.2,-0.6,0.3,0.9,-0.3,1.4,2.1,-0.4,0.7,1.1,-0.8,0.6,1.8,0.9,1.3,-0.5,1.4].map((v, i) => {
                  const pos = v >= 0;
                  const h = Math.abs(v) * 30 + 4;
                  return (
                    <div key={i} className="flex-1 flex flex-col justify-center items-center h-full">
                      {pos ? (
                        <div className="flex-1 flex items-end justify-center w-full">
                          <div style={{ width: "80%", height: h, background: "#08a86b", borderRadius: "3px 3px 0 0" }} />
                        </div>
                      ) : (
                        <>
                          <div className="flex-1" />
                          <div className="flex items-start justify-center w-full">
                            <div style={{ width: "80%", height: h, background: "#e5484d", borderRadius: "0 0 3px 3px" }} />
                          </div>
                        </>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="h-px bg-[#e6e8ef]" />

            {/* Rule discipline */}
            <div>
              <div className="text-[10px] uppercase tracking-[0.10em] text-[var(--ink-4)] font-semibold mb-2.5">Rule Discipline · April</div>
              <div className="flex flex-col gap-2">
                {[
                  ["Cut all losses ≤ -1%", 96, "#08a86b"],
                  ["Followed buy rule", 88, "#08a86b"],
                  ["Sized within ATR", 92, "#08a86b"],
                  ["Journaled same day", 71, "#f59f00"],
                  ["Screenshots saved", 52, "#e5484d"],
                ].map(([k, v, c]) => (
                  <div key={k as string} className="grid items-center gap-2.5" style={{ gridTemplateColumns: "1fr 60px 40px", fontSize: 12 }}>
                    <span>{k as string}</span>
                    <div className="h-2 bg-[#eef0f6] rounded-full overflow-hidden">
                      <div className="h-full rounded-full" style={{ width: `${v}%`, background: c as string, transition: "width 0.6s ease" }} />
                    </div>
                    <span className="text-right text-[11px] font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace", color: c as string }}>{v}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function StubPage({ title, pageId }: { title: string; pageId: string }) {
  const group = getGroupForPage(pageId);
  const color = group?.color || "#6366f1";
  const words = title.split(" ");
  const last = words.pop();
  const rest = words.join(" ");

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px] border-b border-[#e6e8ef]">
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          {rest}{rest ? " " : ""}<em className="italic" style={{ color }}>{last}</em>
        </h1>
      </div>
      <div className="border-[1.5px] border-dashed border-[#d8dbe5] rounded-[14px] bg-white p-20 text-center">
        <div className="w-14 h-14 rounded-[16px] flex items-center justify-center mx-auto mb-[18px] text-2xl"
             style={{ background: `color-mix(in oklab, ${color} 12%, transparent)`, color }}>
          ✦
        </div>
        <h2 className="text-[26px] font-normal italic m-0 mb-1.5" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>Coming together</h2>
        <p className="text-[var(--ink-3)] text-sm max-w-[480px] mx-auto leading-relaxed">
          This page will be built next. The data layer connects to your existing Supabase PostgreSQL — same database, same trade records, same journal entries.
        </p>
      </div>
    </div>
  );
}

const PAGE_TITLES: Record<string, string> = {
  dashboard: "Dashboard", overview: "Trading Overview",
  campaign: "Active Campaign Summary", import: "Import Trades",
  logbuy: "Log Buy", logsell: "Log Sell", sizer: "Position Sizer",
  journal: "Trade Journal", manager: "Trade Manager",
  earnings: "Earnings Planner", heat: "Portfolio Heat", riskmgr: "Risk Manager",
  djournal: "Daily Journal", report: "Daily Report Card",
  routine: "Daily Routine", retro: "Weekly Retro",
  ibd: "IBD Market School", mfactor: "M Factor",
  cycle: "Market Cycle Tracker", rally: "Rally Context",
  coach: "AI Coach", analytics: "Analytics", audit: "Performance Audit",
  heatmap: "Performance Heat Map", period: "Period Review",
  forensics: "Ticker Forensics", admin: "Admin",
};

export default function Home() {
  const [page, setPage] = useState("dashboard");
  const [privacy, setPrivacy] = useState(false);
  const [dark, setDark] = useState(false);
  const [rail, setRail] = useState(false);
  const group = getGroupForPage(page);
  const navColor = group?.color || "#6366f1";

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
      <Sidebar activePage={page} onNavigate={setPage} privacy={privacy} onTogglePrivacy={() => setPrivacy(!privacy)} dark={dark} onToggleDark={toggleDark} rail={rail} onToggleRail={() => setRail(!rail)} />
      <main className="flex-1 flex flex-col min-w-0" style={{ background: "var(--bg)" }}>
        <header className="h-[56px] flex items-center px-6 gap-5 sticky top-0 z-30"
                style={{ background: "var(--header-bg)", backdropFilter: "saturate(160%) blur(10px)", borderBottom: "1px solid var(--border)" }}>
          <div className="flex items-center gap-2 text-[13px] text-[var(--ink-3)]">
            <span className="font-semibold" style={{ color: navColor }}>{group?.label}</span>
            <span className="text-[#b6bac7]">/</span>
            <span className="text-[var(--ink)] font-semibold">{PAGE_TITLES[page]}</span>
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
          {page === "dashboard" ? <Dashboard navColor={navColor} /> : <StubPage title={PAGE_TITLES[page] || page} pageId={page} />}
        </div>
      </main>
      <CommandPalette onNavigate={setPage} />

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
