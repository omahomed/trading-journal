"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import { api, getActivePortfolio, type JournalEntry, type JournalHistoryPoint, type DashboardMetrics } from "@/lib/api";
import { usePortfolio } from "@/lib/portfolio-context";
import { CaptureSnapshotButton } from "./capture-snapshot";
import {
  ResponsiveContainer, ComposedChart, Line, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ReferenceLine,
} from "recharts";

// KPI tile renders one or two subtext lines. The NLV tile uses `extraSub`
// for the small grey "Live estimate: $X" diagnostic — visually subordinate
// to the main `sub` so the user immediately sees which is "the number"
// (journal/broker) and which is "the estimate" (live yfinance).
function KPITile({ label, value, sub, extraSub, gradient }: {
  label: string;
  value: string;
  sub: string;
  extraSub?: string;
  gradient: string;
}) {
  return (
    <div className="relative overflow-hidden rounded-[14px] p-[14px_16px] text-white flex flex-col justify-between min-h-[90px] transition-transform duration-150 hover:scale-[1.01]"
         style={{ background: gradient, boxShadow: "var(--kpi-shadow)" }}>
      <div className="absolute -right-5 -top-5 w-[100px] h-[100px] rounded-full"
           style={{ background: "radial-gradient(circle, rgba(255,255,255,0.18), transparent 65%)" }} />
      <div className="relative z-10">
        <div className="text-[9px] font-semibold uppercase tracking-[0.10em] opacity-85">{label}</div>
        <div className="text-[22px] font-semibold tracking-tight mt-0.5 privacy-mask"
             style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{value}</div>
      </div>
      <div className="relative z-10 privacy-mask">
        <div className="text-[10px] font-medium opacity-80">{sub}</div>
        {extraSub && (
          <div className="text-[9px] font-medium opacity-65 mt-0.5" data-testid="kpi-extra-sub">
            {extraSub}
          </div>
        )}
      </div>
    </div>
  );
}

export function Dashboard({ navColor }: { navColor: string }) {
  const { activePortfolio } = usePortfolio();
  const [latest, setLatest] = useState<JournalEntry | null>(null);
  const [history, setHistory] = useState<JournalHistoryPoint[]>([]);
  const [openCount, setOpenCount] = useState(0);
  const [closedTrades, setClosedTrades] = useState<any[]>([]);
  const [events, setEvents] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [ecRange, setEcRange] = useState<"1Y" | "6M" | "3M" | "All">("1Y");
  const [ecMaximized, setEcMaximized] = useState(false);
  const [showEvents, setShowEvents] = useState(true);
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);

  const loadData = useCallback(async () => {
    const activeId = activePortfolio?.id;
    const [lat, hist, open, closed, ev, dash] = await Promise.all([
      api.journalLatest().catch(() => null),
      api.journalHistory(getActivePortfolio(), 0).catch(() => []),
      api.tradesOpen().catch(() => []),
      api.tradesClosed(getActivePortfolio(), 5000).catch(() => []),
      api.events().catch(() => []),
      activeId != null ? api.dashboardMetrics(activeId).catch(() => null) : Promise.resolve(null),
    ]);
    // Guard: backend can return {error: "..."} at HTTP 200 when something
    // goes wrong server-side. Don't let that poison the render.
    const safeMetrics = (dash && typeof dash === "object" && !("error" in dash))
      ? (dash as DashboardMetrics)
      : null;
    setMetrics(safeMetrics);
    setLatest(lat as JournalEntry);
    setHistory(hist as JournalHistoryPoint[]);
    const openArr = (open as any[]) || [];
    setOpenCount(openArr.length);
    setClosedTrades(Array.isArray(closed) ? closed : []);
    setEvents(Array.isArray(ev) ? ev : []);
    setLoading(false);
  }, []);

  useEffect(() => { loadData(); }, [loadData, activePortfolio?.id]);

  // Auto-refresh when tab regains focus
  useEffect(() => {
    const onFocus = () => { if (!document.hidden) loadData(); };
    document.addEventListener("visibilitychange", onFocus);
    window.addEventListener("focus", onFocus);
    return () => {
      document.removeEventListener("visibilitychange", onFocus);
      window.removeEventListener("focus", onFocus);
    };
  }, [loadData]);

  const ecData = useMemo(() => {
    if (history.length === 0) return [];

    // Compute SMAs on full history first, then filter range
    const fullData = history.map((h, idx) => {
      const portfolio = h.portfolio_ltd || 0;
      // SMA helper — average of last N portfolio_ltd values
      const sma = (n: number) => {
        if (idx < n - 1) return null;
        let sum = 0;
        for (let j = idx - n + 1; j <= idx; j++) sum += (history[j].portfolio_ltd || 0);
        return sum / n;
      };
      return {
        day: h.day,
        portfolio,
        spy: h.spy_ltd || 0,
        ndx: h.ndx_ltd || 0,
        exposure: h.pct_invested || 0,
        sma10: sma(10),
        sma21: sma(21),
        sma50: sma(50),
      };
    });

    // Filter by time range
    let filtered = fullData;
    if (ecRange !== "All") {
      const now = new Date();
      const months = ecRange === "1Y" ? 12 : ecRange === "6M" ? 6 : 3;
      const cutoff = new Date(now.getFullYear(), now.getMonth() - months, now.getDate());
      const cutoffStr = cutoff.toISOString().slice(0, 10);
      filtered = fullData.filter(h => h.day >= cutoffStr);
    }

    if (filtered.length === 0) return [];

    // Rebase to start of visible range
    const base = {
      portfolio: filtered[0].portfolio,
      spy: filtered[0].spy,
      ndx: filtered[0].ndx,
      sma10: filtered[0].sma10 ?? filtered[0].portfolio,
      sma21: filtered[0].sma21 ?? filtered[0].portfolio,
      sma50: filtered[0].sma50 ?? filtered[0].portfolio,
    };

    return filtered.map(h => ({
      day: h.day,
      date: String(h.day).slice(5),
      portfolio: parseFloat((h.portfolio - base.portfolio).toFixed(2)),
      spy: parseFloat((h.spy - base.spy).toFixed(2)),
      ndx: parseFloat((h.ndx - base.ndx).toFixed(2)),
      exposure: h.exposure,
      sma10: h.sma10 != null ? parseFloat((h.sma10 - base.sma10).toFixed(2)) : null,
      sma21: h.sma21 != null ? parseFloat((h.sma21 - base.sma21).toFixed(2)) : null,
      sma50: h.sma50 != null ? parseFloat((h.sma50 - base.sma50).toFixed(2)) : null,
      // Market regime: portfolio above its own 21 SMA
      regime: h.sma21 != null ? (h.portfolio >= h.sma21 ? 1 : 0) : null,
    }));
  }, [history, ecRange]);

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

  // After this refactor: every KPI value below is sourced from the dashboard-
  // metrics endpoint, which itself reads journal.end_nlv as the single
  // source of truth. The live compute_nlv() result appears ONLY as the NLV
  // tile's "Live estimate: $X" subordinate sub-label — it doesn't feed
  // exposure / drawdown / position sizing / risk math.
  const journalAvailable = metrics?.journal_available ?? false;
  const nlv = metrics?.nlv ?? 0;
  const dailyDol = metrics?.nlv_delta_dollar;
  const dailyPct = metrics?.nlv_delta_pct;
  const ltdPct = metrics?.ltd_pct ?? 0;
  const ytdPct = metrics?.ytd_pct;
  const ytdAvailable = metrics?.ytd_available ?? false;
  const exposure = metrics?.exposure_pct ?? 0;
  const ddPct = metrics?.drawdown_current_pct ?? 0;
  const peakNlv = metrics?.drawdown_peak_nlv ?? 0;
  const portfolioHeat = latest?.portfolio_heat || 0;

  // SPY/NDX YTD benchmarks still come from the journal history (they're
  // index-level, not portfolio-level — no equivalent on dashboard-metrics).
  const lastH = history.length > 0 ? history[history.length - 1] : null;
  const currYear = new Date().getFullYear();
  const currYearStr = `${currYear}`;
  const ytdHistory = history.filter(h => String(h.day).slice(0, 4) === currYearStr);
  const jan1 = ytdHistory.length > 0 ? ytdHistory[0] : null;
  const spySt = jan1?.spy || 0;
  const spyCurr = lastH?.spy || 0;
  const ytdSpy = spySt > 0 ? ((spyCurr / spySt) - 1) * 100 : 0;
  const ndxSt = jan1?.nasdaq || 0;
  const ndxCurr = lastH?.nasdaq || 0;
  const ytdNdx = ndxSt > 0 ? ((ndxCurr / ndxSt) - 1) * 100 : 0;

  // NLV tile sub: daily delta when available; first-entry / no-journal cases
  // yield no sub-line (don't fake "+$0" — the spec's "no comparison" rule).
  const nlvSub = (dailyDol != null && dailyPct != null)
    ? `${dailyDol >= 0 ? "+" : ""}$${dailyDol.toLocaleString(undefined, { maximumFractionDigits: 0 })} (${dailyPct >= 0 ? "+" : ""}${dailyPct.toFixed(2)}%)`
    : (journalAvailable ? "First entry — no prior day" : "Save your first daily routine");

  // NLV tile extraSub — small grey "Live estimate" diagnostic. Three states:
  //   1. compute_nlv unavailable → "Live estimate: unavailable"
  //   2. available but no journal anchor → just the dollar value
  //   3. available with anchor → value + diff in $ and %
  let liveEstimateSub: string | undefined;
  if (metrics?.live_estimate_unavailable) {
    liveEstimateSub = "Live estimate: unavailable";
  } else if (metrics?.live_estimate_nlv != null) {
    const liveStr = `$${metrics.live_estimate_nlv.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
    if (metrics.live_estimate_diff != null && metrics.live_estimate_diff_pct != null) {
      const d = metrics.live_estimate_diff;
      const dPct = metrics.live_estimate_diff_pct;
      liveEstimateSub = `Live estimate: ${liveStr} (${d >= 0 ? "+" : ""}$${d.toLocaleString(undefined, { maximumFractionDigits: 0 })}, ${dPct >= 0 ? "+" : ""}${dPct.toFixed(2)}%)`;
    } else {
      liveEstimateSub = `Live estimate: ${liveStr}`;
    }
  }

  const kpis = [
    {
      label: "NET LIQ VALUE",
      value: journalAvailable ? `$${nlv.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : "—",
      sub: nlvSub,
      extraSub: liveEstimateSub,
      gradient: "linear-gradient(135deg, #6366f1, #818cf8)",
    },
    {
      label: "LTD RETURN",
      value: journalAvailable ? `${ltdPct.toFixed(2)}%` : "—",
      sub: journalAvailable ? "Time-weighted, since reset" : "Save your first daily routine",
      gradient: "linear-gradient(135deg, #ec4899, #f472b6)",
    },
    {
      label: "YTD RETURN",
      value: ytdAvailable && ytdPct != null ? `${ytdPct.toFixed(2)}%` : "—",
      sub: ytdAvailable
        ? `SPY: ${ytdSpy >= 0 ? "+" : ""}${ytdSpy.toFixed(2)}% | NDX: ${ytdNdx >= 0 ? "+" : ""}${ytdNdx.toFixed(2)}%`
        : "Available once current-year EOD entries exist",
      gradient: "linear-gradient(135deg, #10b981, #34d399)",
    },
    {
      label: "LIVE EXPOSURE",
      value: journalAvailable ? `${exposure.toFixed(1)}%` : "—",
      sub: `${openCount}/${15} Pos | Risk: ${portfolioHeat.toFixed(2)}%`,
      gradient: "linear-gradient(135deg, #f97316, #fb923c)",
    },
    {
      label: "DRAWDOWN",
      value: journalAvailable ? `${ddPct.toFixed(2)}%` : "—",
      sub: !journalAvailable
        ? "Save your first daily routine"
        : ddPct >= -0.01
          ? "Clear"
          : `from peak $${peakNlv.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
      gradient: "linear-gradient(135deg, #1e40af, #3b82f6)",
    },
  ];

  // Recent history for mini bar chart
  const recentHistory = history.slice(-20);

  return (
    <div id="dashboard-capture-root" style={{ animation: "slide-up 0.18s ease-out" }}>
      {/* Header */}
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
              {greeting}, <em className="italic" style={{ color: navColor }}>MO</em>
            </h1>
            <div className="text-[13px] mt-1.5 flex items-center gap-2 flex-wrap" style={{ color: "var(--ink-3)" }}>
              <span>
                {new Date().toLocaleDateString("en-US", { weekday: "long", year: "numeric", month: "long", day: "numeric" })}
                {activePortfolio ? ` · ${activePortfolio.name}` : ""}
              </span>
              {/* "As of [date]" badge — clarifies the dashboard reflects
                  the most recent SAVED journal entry, which may lag today
                  if Daily Routine hasn't been submitted yet. Hidden when
                  the journal is empty (the empty-state copy below
                  carries the same signal more loudly). */}
              {metrics?.as_of_date && (
                <span className="text-[10px] font-medium px-1.5 py-0.5 rounded-[4px]"
                      data-testid="dashboard-as-of-badge"
                      style={{
                        background: "var(--bg-2)",
                        color: "var(--ink-4)",
                        border: "1px solid var(--border)",
                      }}>
                  As of {metrics.as_of_date}
                </span>
              )}
            </div>
          </div>
          <CaptureSnapshotButton targetSelector="#dashboard-capture-root" snapshotType="dashboard" label="Capture EOD Snapshot" />
        </div>
      </div>

      {/* Tape pill */}
      <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium mb-5"
           style={{ background: "color-mix(in oklab, #8b5cf6 12%, var(--surface))", color: "#8b5cf6" }}>
        <span className="w-1.5 h-1.5 rounded-full bg-[#8b5cf6]"
              style={{ animation: "pulse-dot 2s ease-in-out infinite", boxShadow: "0 0 0 3px color-mix(in oklab, #8b5cf6 12%, var(--surface))" }} />
        Tape: {latest?.market_window || "—"} · {latest?.day || ""}
      </div>

      {/* Empty-state nudge — when the journal has no rows, every KPI value
          renders as "—" and we explicitly point the user at the action that
          unblocks the dashboard. Spec: "Frontend shows '—' for all journal-
          derived values plus a help message 'Save your first daily routine
          to see metrics'." */}
      {!loading && metrics != null && !metrics.journal_available && (
        <div className="mb-4 px-4 py-2.5 rounded-[10px] text-[12px] font-medium"
             data-testid="dashboard-empty-state"
             style={{
               background: "color-mix(in oklab, #6366f1 10%, var(--surface))",
               color: "#4f46e5",
               border: "1px solid color-mix(in oklab, #6366f1 30%, var(--border))",
             }}>
          Save your first daily routine to see dashboard metrics. Until then, every
          tile shows &ldquo;—&rdquo;.
        </div>
      )}

      {/* KPI Strip — REAL DATA */}
      <div className="grid grid-cols-5 gap-3.5 mb-3">
        {kpis.map((kpi) => <KPITile key={kpi.label} {...kpi} />)}
      </div>

      {/* NLV breakdown: cash + positions, both sourced from the latest
          journal entry (cash = nlv - total_holdings, total_holdings =
          pct_invested × end_nlv / 100). The pre-refactor disclaimer
          ("NLV excludes commissions & margin interest. Reconcile in
          Settings to match your broker.") is gone — journal NLV now
          comes from the broker via IBKR auto-fill, so it already
          includes those. The "Some prices unavailable" warning is also
          gone here (it was a live-snapshot concern; live NLV is now
          only a sub-label on the NLV tile). */}
      {metrics?.journal_available && (
        <div className="flex items-center gap-4 mb-6 text-[11px] flex-wrap"
             data-testid="dashboard-cash-positions-row"
             style={{ color: "var(--ink-4)" }}>
          <div>
            <span className="font-semibold privacy-mask" style={{ color: "var(--ink-3)" }}>
              ${(metrics.cash ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </span> cash
          </div>
          <span>·</span>
          <div>
            <span className="font-semibold privacy-mask" style={{ color: "var(--ink-3)" }}>
              ${(metrics.total_holdings ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </span> positions
          </div>
        </div>
      )}

      {/* Two-column: EC + This Month (or full-width when maximized) */}
      <div className="grid gap-[18px]" style={{ gridTemplateColumns: ecMaximized ? "1fr" : "2fr 1fr", alignItems: "stretch" }}>
      {/* Equity Curve */}
      <div className="rounded-[14px] overflow-hidden flex flex-col" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
        <div className="flex items-center justify-between px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
            <span className="text-[13px] font-semibold">Equity Curve</span>
            <span className="text-xs" style={{ color: "var(--ink-4)" }}>Portfolio vs Benchmark · event markers</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex p-0.5 rounded-[10px] gap-0.5" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
              {(["All", "1Y", "6M", "3M"] as const).map(t => (
                <button key={t} onClick={() => setEcRange(t)}
                        className="px-3 py-1 rounded-md text-xs font-medium transition-all"
                        style={{
                          background: ecRange === t ? "var(--surface)" : "transparent",
                          color: ecRange === t ? "var(--ink)" : "var(--ink-4)",
                          boxShadow: ecRange === t ? "0 1px 2px rgba(14,20,38,0.04)" : "none",
                        }}>
                  {t}
                </button>
              ))}
            </div>
            <button onClick={() => setShowEvents(!showEvents)}
                    className="h-[28px] px-2.5 rounded-[8px] flex items-center gap-1.5 text-[10px] font-semibold transition-colors cursor-pointer"
                    style={{
                      background: showEvents ? "color-mix(in oklab, #dc2626 10%, var(--surface))" : "var(--bg)",
                      border: `1px solid ${showEvents ? "color-mix(in oklab, #dc2626 25%, var(--border))" : "var(--border)"}`,
                      color: showEvents ? "#dc2626" : "var(--ink-4)",
                    }}
                    title="Toggle event markers">
              Events {showEvents ? "ON" : "OFF"}
            </button>
            <button onClick={() => setEcMaximized(!ecMaximized)}
                    className="w-[30px] h-[30px] rounded-[8px] flex items-center justify-center transition-colors hover:brightness-95"
                    style={{ background: "var(--bg)", border: "1px solid var(--border)" }}
                    title={ecMaximized ? "Collapse" : "Maximize"}>
              {ecMaximized ? (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--ink-4)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/><line x1="14" y1="10" x2="21" y2="3"/><line x1="3" y1="21" x2="10" y2="14"/>
                </svg>
              ) : (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--ink-4)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/>
                </svg>
              )}
            </button>
          </div>
        </div>

        {/* Market regime bar */}
        {ecData.length > 0 && (
          <div className="flex h-[6px]">
            {ecData.map((d, i) => (
              <div key={i} className="flex-1" style={{
                background: d.regime === 1 ? "#22c55e" : d.regime === 0 ? "#ef4444" : "#d4d4d8",
                opacity: 0.7,
              }} />
            ))}
          </div>
        )}

        {/* Custom legend */}
        <div className="flex items-center gap-4 px-[18px] pt-3 pb-1 flex-wrap" style={{ fontSize: 11 }}>
          {[
            { label: `SPY (${ecData.length > 0 ? (ecData[ecData.length-1].spy >= 0 ? "+" : "") + ecData[ecData.length-1].spy.toFixed(1) + "%" : ""})`, color: "#9ca3af", dash: false, width: 1.5 },
            { label: `Nasdaq (${ecData.length > 0 ? (ecData[ecData.length-1].ndx >= 0 ? "+" : "") + ecData[ecData.length-1].ndx.toFixed(1) + "%" : ""})`, color: "#22c55e", dash: false, width: 1.5 },
            { label: "50 SMA", color: "#ef4444", dash: false, width: 1.3 },
            { label: "21 SMA", color: "#22c55e", dash: false, width: 1.3 },
            { label: "10 SMA", color: "#a855f6", dash: false, width: 1.3 },
            { label: `Portfolio (${ecData.length > 0 ? (ecData[ecData.length-1].portfolio >= 0 ? "+" : "") + ecData[ecData.length-1].portfolio.toFixed(1) + "%" : ""})`, color: "#1e3a8a", dash: false, width: 2.5 },
            { label: "Exposure %", color: "#f97316", dash: false, width: 0 },
          ].map(item => (
            <div key={item.label} className="flex items-center gap-1.5" style={{ color: "var(--ink-3)" }}>
              {item.width > 0 ? (
                <svg width="18" height="10">
                  <line x1="0" y1="5" x2="18" y2="5" stroke={item.color} strokeWidth={item.width}
                        strokeDasharray={item.dash ? "3 2" : "none"} />
                </svg>
              ) : (
                <svg width="18" height="10">
                  <rect x="0" y="2" width="18" height="6" fill={item.color} opacity={0.3} rx="1" />
                </svg>
              )}
              <span>{item.label}</span>
            </div>
          ))}
        </div>

        <div className="flex-1 px-1 pb-3" style={{ minHeight: ecMaximized ? 600 : 420 }}>
          {ecData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={ecData} margin={{ top: 8, right: 50, left: 0, bottom: 5 }}>
                <defs>
                  <linearGradient id="exposureFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#f97316" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#f97316" stopOpacity={0.03} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.5} vertical={false} />
                <XAxis
                  dataKey="day"
                  tick={{ fontSize: 10, fill: "var(--ink-4)" }}
                  tickLine={false}
                  axisLine={{ stroke: "var(--border)" }}
                  interval={Math.max(Math.floor(ecData.length / 8), 1)}
                  tickFormatter={(v: string) => {
                    const d = new Date(v);
                    return d.toLocaleDateString("en-US", { month: "short", year: "numeric" });
                  }}
                  label={{ value: "Date", position: "insideBottom", offset: -10, fontSize: 11, fill: "var(--ink-4)" }}
                />
                <YAxis
                  yAxisId="left"
                  tick={{ fontSize: 10, fill: "var(--ink-4)" }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v: number) => `${v}%`}
                  width={50}
                  label={{ value: "Return %", angle: -90, position: "insideLeft", offset: 10, fontSize: 11, fill: "var(--ink-4)" }}
                />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  tick={{ fontSize: 10, fill: "var(--ink-4)" }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v: number) => `${v}%`}
                  width={50}
                  domain={[0, 800]}
                  label={{ value: "% Exposure", angle: 90, position: "insideRight", offset: 10, fontSize: 11, fill: "var(--ink-4)" }}
                />
                <Tooltip
                  contentStyle={{
                    background: "var(--surface)",
                    border: "1px solid var(--border)",
                    borderRadius: 10,
                    fontSize: 11,
                    boxShadow: "0 4px 14px rgba(0,0,0,0.08)",
                    fontFamily: "var(--font-jetbrains), monospace",
                    padding: "8px 12px",
                  }}
                  labelStyle={{ fontWeight: 600, marginBottom: 4 }}
                  formatter={(value: any, name: any) => {
                    if (value == null) return [null, null];
                    const labels: Record<string, string> = {
                      portfolio: "Portfolio", spy: "SPY", ndx: "Nasdaq",
                      sma10: "10 SMA", sma21: "21 SMA", sma50: "50 SMA",
                      exposure: "Exposure",
                    };
                    return [`${Number(value).toFixed(2)}%`, labels[String(name)] || String(name)];
                  }}
                  labelFormatter={(label: any) => {
                    const d = new Date(String(label));
                    return d.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric", year: "numeric" });
                  }}
                />

                <ReferenceLine yAxisId="left" y={0} stroke="var(--ink-4)" strokeDasharray="3 3" strokeOpacity={0.3} />

                {/* Exposure area — right axis */}
                <Area yAxisId="right" dataKey="exposure" fill="url(#exposureFill)"
                      stroke="#f97316" strokeWidth={1} strokeOpacity={0.5} type="stepAfter"
                      connectNulls={false} />

                {/* 100% exposure reference */}
                <ReferenceLine yAxisId="right" y={100} stroke="#000" strokeDasharray="4 4" strokeOpacity={0.12} />

                {/* 50 SMA — red solid */}
                <Line yAxisId="left" dataKey="sma50" stroke="#ef4444" strokeWidth={1.3}
                      dot={false} type="monotone" connectNulls />

                {/* 21 SMA — green solid */}
                <Line yAxisId="left" dataKey="sma21" stroke="#22c55e" strokeWidth={1.3}
                      dot={false} type="monotone" connectNulls />

                {/* 10 SMA — purple solid */}
                <Line yAxisId="left" dataKey="sma10" stroke="#a855f6" strokeWidth={1.3}
                      dot={false} type="monotone" connectNulls />

                {/* SPY — gray */}
                <Line yAxisId="left" dataKey="spy" stroke="#9ca3af" strokeWidth={1.5}
                      dot={false} type="monotone" />

                {/* NDX — green */}
                <Line yAxisId="left" dataKey="ndx" stroke="#22c55e" strokeWidth={1.5}
                      dot={false} type="monotone" />

                {/* Portfolio — thick navy, on top */}
                <Line yAxisId="left" dataKey="portfolio" stroke="#1e3a8a" strokeWidth={2.5}
                      dot={false} type="monotone" />

                {/* Event markers */}
                {showEvents && events.map((ev, i) => {
                  const evDate = String(ev.event_date || "").slice(0, 10);
                  // Find the matching data point's full day string
                  const match = ecData.find(d => String(d.day).slice(0, 10) === evDate);
                  if (!match) return null;
                  const color = ev.category === "market" ? "#dc2626" : ev.category === "macro" ? "#9333ea" : "#6b7280";
                  return (
                    <ReferenceLine key={`ev-${i}`} yAxisId="left" x={match.day}
                      stroke={color} strokeWidth={1.5} strokeDasharray="4 3" strokeOpacity={0.7}
                      label={{ value: ev.label, position: "insideTopRight", fontSize: 9, fill: color, fontWeight: 600 }} />
                  );
                })}
              </ComposedChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-sm" style={{ color: "var(--ink-4)" }}>
              No data available for selected range
            </div>
          )}
        </div>
      </div>

        {/* This Month at a Glance — hidden when EC maximized */}
        {!ecMaximized && (() => {
          const now = new Date();
          const monthStr = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}`;
          const monthData = history.filter(h => String(h.day).slice(0, 7) === monthStr);

          // MTD return via TWR (compound daily returns)
          const mtdPct = monthData.length > 0
            ? (monthData.reduce((prod, h) => prod * (1 + (h.daily_pct_change || 0) / 100), 1) - 1) * 100
            : 0;
          const mtdDol = monthData.length > 0
            ? monthData[monthData.length - 1].end_nlv - monthData[0].end_nlv
            : 0;

          // Best and worst day
          const sortedByPct = [...monthData].sort((a, b) => (b.daily_pct_change || 0) - (a.daily_pct_change || 0));
          const bestDay = sortedByPct[0];
          const worstDay = sortedByPct[sortedByPct.length - 1];

          // Win/loss count
          const wins = monthData.filter(h => (h.daily_pct_change || 0) > 0).length;
          const losses = monthData.filter(h => (h.daily_pct_change || 0) < 0).length;

          return (
            <div className="rounded-[14px] overflow-hidden flex flex-col" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
              <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
                <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
                <span className="text-[13px] font-semibold">This Month at a Glance</span>
                <span className="text-xs" style={{ color: "var(--ink-4)" }}>{now.toLocaleString("en-US", { month: "long", year: "numeric" })}</span>
              </div>
              <div className="flex-1 p-[18px] flex flex-col gap-3.5">
                {/* 4 stat tiles */}
                <div className="grid grid-cols-2 gap-2.5">
                  <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                    <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>MTD Return</div>
                    <div className="text-[18px] font-semibold mt-1 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace", color: mtdPct >= 0 ? "#08a86b" : "#e5484d" }}>
                      {mtdPct >= 0 ? "+" : ""}{mtdPct.toFixed(2)}%
                    </div>
                    <div className="text-[11px] mt-0.5 privacy-mask" style={{ color: "var(--ink-4)" }}>${mtdDol >= 0 ? "+" : ""}{mtdDol.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
                  </div>
                  <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                    <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>Trades</div>
                    <div className="text-[18px] font-semibold mt-1" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{monthData.length}</div>
                    <div className="text-[11px] mt-0.5" style={{ color: "var(--ink-4)" }}>{wins}W · {losses}L</div>
                  </div>
                  <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                    <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>Best Day</div>
                    <div className="text-[18px] font-semibold mt-1 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace", color: "#08a86b" }}>
                      +{(bestDay?.daily_pct_change || 0).toFixed(2)}%
                    </div>
                    <div className="text-[11px] mt-0.5" style={{ color: "var(--ink-4)" }}>{bestDay?.day?.slice(5) || ""}</div>
                  </div>
                  <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                    <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>Worst Day</div>
                    <div className="text-[18px] font-semibold mt-1 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace", color: "#e5484d" }}>
                      {(worstDay?.daily_pct_change || 0).toFixed(2)}%
                    </div>
                    <div className="text-[11px] mt-0.5" style={{ color: "var(--ink-4)" }}>{worstDay?.day?.slice(5) || ""}</div>
                  </div>
                </div>

                {/* Daily P&L bars from REAL data */}
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

                <div className="h-px" style={{ background: "var(--border)" }} />

                {/* Monthly pulse — computed from journal data */}
                <div>
                  <div className="text-[10px] uppercase tracking-[0.10em] font-semibold mb-2.5" style={{ color: "var(--ink-4)" }}>
                    Monthly Pulse · {now.toLocaleString("en-US", { month: "long" })}
                  </div>
                  {(() => {
                    const winPct = monthData.length > 0 ? (wins / monthData.length) * 100 : 0;
                    const avgExposure = monthData.length > 0
                      ? monthData.reduce((s, d) => s + (d.pct_invested || 0), 0) / monthData.length
                      : 0;
                    const exposureColor = avgExposure > 100 ? "#e5484d" : avgExposure > 70 ? "#f59f00" : "#08a86b";

                    // Profit Factor from closed trades this month
                    const mtdClosed = closedTrades.filter(t => String(t.closed_date || "").slice(0, 7) === monthStr);
                    const grossWins = mtdClosed.filter(t => parseFloat(String(t.realized_pl || 0)) > 0).reduce((s, t) => s + parseFloat(String(t.realized_pl || 0)), 0);
                    const grossLosses = Math.abs(mtdClosed.filter(t => parseFloat(String(t.realized_pl || 0)) < 0).reduce((s, t) => s + parseFloat(String(t.realized_pl || 0)), 0));
                    const profitFactor = grossLosses > 0 ? grossWins / grossLosses : grossWins > 0 ? 99 : 0;
                    const pfColor = profitFactor >= 2 ? "#08a86b" : profitFactor >= 1 ? "#f59f00" : "#e5484d";
                    const pfBar = Math.min(profitFactor * 25, 100); // Scale: 4x = 100%

                    const metrics = [
                      ["Win Rate", winPct, winPct >= 50 ? "#08a86b" : winPct >= 40 ? "#f59f00" : "#e5484d"],
                      ["Profit Factor", pfBar, pfColor, mtdClosed.length > 0 ? `${profitFactor.toFixed(2)}x` : "—"],
                      ["Avg Exposure", Math.min(avgExposure, 100), exposureColor, `${avgExposure.toFixed(0)}%`],
                    ] as [string, number, string, string?][];
                    return (
                      <div className="flex flex-col gap-2">
                        {metrics.map(([k, v, c, label]) => (
                          <div key={k} className="grid items-center gap-2.5" style={{ gridTemplateColumns: "1fr 60px 40px", fontSize: 12 }}>
                            <span>{k}</span>
                            <div className="h-2 rounded-full overflow-hidden" style={{ background: "var(--bg-2)" }}>
                              <div className="h-full rounded-full" style={{ width: `${Math.min(v, 100)}%`, background: c, transition: "width 0.6s ease" }} />
                            </div>
                            <span className="text-right text-[11px] font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace", color: c }}>{label || `${v.toFixed(0)}%`}</span>
                          </div>
                        ))}
                      </div>
                    );
                  })()}
                </div>
              </div>
            </div>
          );
        })()}
      </div>
    </div>
  );
}
