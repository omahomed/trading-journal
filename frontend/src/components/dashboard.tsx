"use client";

import { useState, useEffect } from "react";
import { api, type JournalEntry, type JournalHistoryPoint } from "@/lib/api";

function KPITile({ label, value, sub, gradient }: { label: string; value: string; sub: string; gradient: string }) {
  return (
    <div className="relative overflow-hidden rounded-[14px] p-[14px_16px] text-white flex flex-col justify-between h-[90px] transition-transform duration-150 hover:scale-[1.01]"
         style={{ background: gradient, boxShadow: "var(--kpi-shadow)" }}>
      <div className="absolute -right-5 -top-5 w-[100px] h-[100px] rounded-full"
           style={{ background: "radial-gradient(circle, rgba(255,255,255,0.18), transparent 65%)" }} />
      <div className="relative z-10">
        <div className="text-[9px] font-semibold uppercase tracking-[0.10em] opacity-85">{label}</div>
        <div className="text-[22px] font-semibold tracking-tight mt-0.5 privacy-mask"
             style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{value}</div>
      </div>
      <div className="relative z-10 text-[10px] font-medium opacity-80 privacy-mask">{sub}</div>
    </div>
  );
}

export function Dashboard({ navColor }: { navColor: string }) {
  const [latest, setLatest] = useState<JournalEntry | null>(null);
  const [history, setHistory] = useState<JournalHistoryPoint[]>([]);
  const [openCount, setOpenCount] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      api.journalLatest().catch(() => null),
      api.journalHistory("CanSlim", 365).catch(() => []),
      api.tradesOpen().catch(() => []),
    ]).then(([lat, hist, open]) => {
      setLatest(lat as JournalEntry);
      setHistory(hist as JournalHistoryPoint[]);
      setOpenCount((open as any[]).length);
      setLoading(false);
    });
  }, []);

  if (loading) {
    return (
      <div className="animate-pulse">
        <div className="h-8 w-64 rounded-lg mb-6" style={{ background: "var(--bg-2)" }} />
        <div className="grid grid-cols-5 gap-3.5 mb-6">
          {[1,2,3,4,5].map(i => <div key={i} className="h-[90px] rounded-[14px]" style={{ background: "var(--bg-2)" }} />)}
        </div>
        <div className="h-[400px] rounded-[14px]" style={{ background: "var(--bg-2)" }} />
      </div>
    );
  }

  const hour = new Date().getHours();
  const greeting = hour < 12 ? "Good morning" : hour < 17 ? "Good afternoon" : "Good evening";

  // Compute KPI values from real data
  const nlv = latest?.end_nlv || 0;
  const dailyDol = latest?.daily_dollar_change || 0;
  const dailyPct = latest?.daily_pct_change || 0;
  const ltdPct = history.length > 0 ? history[history.length - 1]?.portfolio_ltd || 0 : 0;
  const ltdDol = nlv - (history.length > 0 ? history[0]?.end_nlv || nlv : nlv);
  const exposure = latest?.pct_invested || 0;

  // YTD
  const jan1 = history.find(h => h.day >= `${new Date().getFullYear()}-01-01`);
  const ytdNlv = jan1?.end_nlv || history[0]?.end_nlv || nlv;
  const ytdPct = ytdNlv > 0 ? ((nlv - ytdNlv) / ytdNlv) * 100 : 0;
  const spyLtd = history.length > 0 ? history[history.length - 1]?.spy_ltd || 0 : 0;
  const ndxLtd = history.length > 0 ? history[history.length - 1]?.ndx_ltd || 0 : 0;

  // Drawdown from peak
  const peakNlv = Math.max(...history.map(h => h.end_nlv || 0));
  const ddPct = peakNlv > 0 ? ((nlv - peakNlv) / peakNlv) * 100 : 0;

  const kpis = [
    { label: "NET LIQ VALUE", value: `$${nlv.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, sub: `${dailyDol >= 0 ? "+" : ""}$${dailyDol.toLocaleString(undefined, { maximumFractionDigits: 0 })} (${dailyPct >= 0 ? "+" : ""}${dailyPct.toFixed(2)}%)`, gradient: "linear-gradient(135deg, #6366f1, #818cf8)" },
    { label: "LTD RETURN", value: `${ltdPct.toFixed(2)}%`, sub: `$${ltdDol >= 0 ? "+" : ""}${ltdDol.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, gradient: "linear-gradient(135deg, #ec4899, #f472b6)" },
    { label: "YTD RETURN", value: `${ytdPct.toFixed(2)}%`, sub: `SPY: ${spyLtd >= 0 ? "+" : ""}${spyLtd.toFixed(2)}% | NDX: ${ndxLtd >= 0 ? "+" : ""}${ndxLtd.toFixed(2)}%`, gradient: "linear-gradient(135deg, #10b981, #34d399)" },
    { label: "LIVE EXPOSURE", value: `${exposure.toFixed(1)}%`, sub: `${openCount} positions`, gradient: "linear-gradient(135deg, #f97316, #fb923c)" },
    { label: "DRAWDOWN", value: `${ddPct.toFixed(2)}%`, sub: ddPct >= -0.01 ? "Clear" : `from peak $${peakNlv.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, gradient: "linear-gradient(135deg, #1e40af, #3b82f6)" },
  ];

  // Recent history for mini bar chart
  const recentHistory = history.slice(-20);

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      {/* Header */}
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          {greeting}, <em className="italic" style={{ color: navColor }}>MO</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
          {new Date().toLocaleDateString("en-US", { weekday: "long", year: "numeric", month: "long", day: "numeric" })} · CanSlim
        </div>
      </div>

      {/* Tape pill */}
      <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium mb-5"
           style={{ background: "#f1ecfe", color: "#8b5cf6" }}>
        <span className="w-1.5 h-1.5 rounded-full bg-[#8b5cf6]"
              style={{ animation: "pulse-dot 2s ease-in-out infinite", boxShadow: "0 0 0 3px #f1ecfe" }} />
        Tape: {latest?.market_window || "—"} · {latest?.day || ""}
      </div>

      {/* KPI Strip — REAL DATA */}
      <div className="grid grid-cols-5 gap-3.5 mb-6">
        {kpis.map((kpi) => <KPITile key={kpi.label} {...kpi} />)}
      </div>

      {/* Two-column layout */}
      <div className="grid gap-[18px]" style={{ gridTemplateColumns: "2fr 1fr", alignItems: "stretch" }}>
        {/* Equity Curve */}
        <div className="rounded-[14px] overflow-hidden flex flex-col" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
          <div className="flex items-center justify-between px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
              <span className="text-[13px] font-semibold">Equity Curve</span>
              <span className="text-xs" style={{ color: "var(--ink-4)" }}>Portfolio vs SPY / NDX · {history.length} trading days</span>
            </div>
          </div>
          <div className="flex-1 min-h-[380px] flex items-center justify-center text-sm" style={{ color: "var(--ink-4)" }}>
            [Recharts equity curve — {history.length} data points ready]
          </div>
        </div>

        {/* This Month */}
        <div className="rounded-[14px] overflow-hidden flex flex-col" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
          <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
            <span className="text-[13px] font-semibold">This Month at a Glance</span>
          </div>
          <div className="flex-1 p-[18px] flex flex-col gap-3.5">
            <div className="grid grid-cols-2 gap-2.5">
              <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>MTD Return</div>
                <div className="text-[18px] font-semibold mt-1 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace", color: dailyPct >= 0 ? "#08a86b" : "#e5484d" }}>
                  {dailyPct >= 0 ? "+" : ""}{dailyPct.toFixed(2)}%
                </div>
              </div>
              <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>Positions</div>
                <div className="text-[18px] font-semibold mt-1" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{openCount}</div>
              </div>
            </div>

            {/* Daily P&L bars from real data */}
            <div>
              <div className="text-[10px] uppercase tracking-[0.10em] font-semibold mb-2" style={{ color: "var(--ink-4)" }}>
                Daily P&L · last {recentHistory.length} sessions
              </div>
              <div className="flex items-center gap-[3px] h-[80px]">
                {recentHistory.map((h, i) => {
                  const v = h.daily_pct_change || 0;
                  const pos = v >= 0;
                  const maxAbs = Math.max(...recentHistory.map(r => Math.abs(r.daily_pct_change || 0)), 1);
                  const ht = (Math.abs(v) / maxAbs) * 60 + 4;
                  return (
                    <div key={i} className="flex-1 flex flex-col justify-center items-center h-full">
                      {pos ? (
                        <div className="flex-1 flex items-end justify-center w-full">
                          <div style={{ width: "80%", height: ht, background: "#08a86b", borderRadius: "3px 3px 0 0" }} />
                        </div>
                      ) : (
                        <>
                          <div className="flex-1" />
                          <div className="flex items-start justify-center w-full">
                            <div style={{ width: "80%", height: ht, background: "#e5484d", borderRadius: "0 0 3px 3px" }} />
                          </div>
                        </>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
