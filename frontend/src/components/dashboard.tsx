"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import Link from "next/link";
import { api, getActivePortfolio, type JournalEntry, type JournalHistoryPoint, type DashboardMetrics, type TradePosition, type TradeDetail } from "@/lib/api";
import { usePortfolio } from "@/lib/portfolio-context";
import { computeEnrichedPositions } from "@/lib/positions";
import {
  computeWinRate,
  computeProfitFactor,
  computeHoldRatio,
  computeOnePctCompliance,
  computeLast10Stats,
  trailingClosedTrades,
  trailingClosedLosses,
} from "@/lib/analytics-stats";
import { CaptureSnapshotButton } from "./capture-snapshot";
import {
  ResponsiveContainer, ComposedChart, Line, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ReferenceLine,
} from "recharts";

// "$+339,829" / "$-50,000" — explicit sign, whole-dollar precision. Matches
// the visual treatment of the NLV tile's daily-delta line so the dollar
// P&L sub-labels on LTD/YTD read as native to the same family.
function formatSignedDollars(n: number): string {
  const sign = n >= 0 ? "+" : "-";
  const abs = Math.abs(n);
  return `$${sign}${abs.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

// KPI tile renders one or two subtext lines. `extraSub` is an optional
// secondary line rendered in smaller, dimmer type below the main `sub`
// (e.g. the YTD tile uses it for the SPY/NDX benchmark row).
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
  const [openTrades, setOpenTrades] = useState<TradePosition[]>([]);
  const [closedTrades, setClosedTrades] = useState<any[]>([]);
  const [allDetails, setAllDetails] = useState<TradeDetail[]>([]);
  const [livePrices, setLivePrices] = useState<Record<string, number>>({});
  const [pricesStale, setPricesStale] = useState(false);
  const [hoveredSquareIdx, setHoveredSquareIdx] = useState<number | null>(null);
  const [events, setEvents] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [ecRange, setEcRange] = useState<"1Y" | "6M" | "3M" | "All">("1Y");
  const [ecMaximized, setEcMaximized] = useState(false);
  const [showEvents, setShowEvents] = useState(true);
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);

  const openCount = openTrades.length;

  const loadData = useCallback(async () => {
    const activeId = activePortfolio?.id;
    const [lat, hist, open, closed, ev, dash, recent] = await Promise.all([
      api.journalLatest().catch(() => null),
      api.journalHistory(getActivePortfolio(), 0).catch(() => []),
      api.tradesOpen().catch(() => []),
      api.tradesClosed(getActivePortfolio(), 5000).catch(() => []),
      api.events().catch(() => []),
      activeId != null ? api.dashboardMetrics(activeId).catch(() => null) : Promise.resolve(null),
      api.tradesRecent(getActivePortfolio(), 2000).catch(() => ({ details: [], lot_closures: [] })),
    ]);
    // Guard: backend can return {error: "..."} at HTTP 200 when something
    // goes wrong server-side. Don't let that poison the render.
    const safeMetrics = (dash && typeof dash === "object" && !("error" in dash))
      ? (dash as DashboardMetrics)
      : null;
    setMetrics(safeMetrics);
    setLatest(lat as JournalEntry);
    setHistory(hist as JournalHistoryPoint[]);
    const openArr = (open as TradePosition[]) || [];
    setOpenTrades(openArr);
    setClosedTrades(Array.isArray(closed) ? closed : []);
    setAllDetails((recent && (recent as any).details) || []);
    setEvents(Array.isArray(ev) ? ev : []);
    setLoading(false);
  }, []);

  // Live prices for open trades — drives overall_pl on the Last 10 panel.
  // Mirrors the analytics.tsx pattern: fetch once when openTrades changes;
  // on failure, fall back to cost basis with a visible stale caption.
  useEffect(() => {
    if (openTrades.length === 0) return;
    const tickers = [...new Set(openTrades.map(t => t.ticker).filter(Boolean))];
    if (tickers.length === 0) return;
    api.batchPrices(tickers, getActivePortfolio())
      .then(prices => { setLivePrices(prices); setPricesStale(false); })
      .catch(() => setPricesStale(true));
  }, [openTrades]);

  const enrichedById = useMemo(() => {
    const enriched = computeEnrichedPositions(openTrades, allDetails, 0, livePrices);
    return Object.fromEntries(enriched.map(p => [p.trade_id, p])) as Record<string, ReturnType<typeof computeEnrichedPositions>[number]>;
  }, [openTrades, allDetails, livePrices]);

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
  // source of truth.
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

  const kpis = [
    {
      label: "NET LIQ VALUE",
      value: journalAvailable ? `$${nlv.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : "—",
      sub: nlvSub,
      gradient: "linear-gradient(135deg, #6366f1, #818cf8)",
    },
    {
      label: "LTD RETURN",
      value: journalAvailable ? `${ltdPct.toFixed(2)}%` : "—",
      // Sub: dollar P&L when the cash ledger answers; falls back to a
      // descriptive label otherwise (avoids showing "$+0" which would
      // imply break-even when really we just couldn't compute).
      sub: !journalAvailable
        ? "Save your first daily routine"
        : metrics?.ltd_pl_dollar != null
          ? formatSignedDollars(metrics.ltd_pl_dollar)
          : "Time-weighted, since reset",
      gradient: "linear-gradient(135deg, #ec4899, #f472b6)",
    },
    {
      label: "YTD RETURN",
      value: ytdAvailable && ytdPct != null ? `${ytdPct.toFixed(2)}%` : "—",
      // Two-line sub when both YTD dollar P&L and SPY/NDX are available:
      //   primary sub  → "$+124,363" (most informative; matches LTD style)
      //   extraSub     → "SPY +4.50% | NDX +6.89%" (benchmark context)
      sub: !ytdAvailable
        ? "Available once current-year EOD entries exist"
        : metrics?.ytd_pl_dollar != null
          ? formatSignedDollars(metrics.ytd_pl_dollar)
          : `SPY: ${ytdSpy >= 0 ? "+" : ""}${ytdSpy.toFixed(2)}% | NDX: ${ytdNdx >= 0 ? "+" : ""}${ytdNdx.toFixed(2)}%`,
      extraSub: ytdAvailable && metrics?.ytd_pl_dollar != null
        ? `SPY ${ytdSpy >= 0 ? "+" : ""}${ytdSpy.toFixed(2)}% | NDX ${ytdNdx >= 0 ? "+" : ""}${ytdNdx.toFixed(2)}%`
        : undefined,
      gradient: "linear-gradient(135deg, #10b981, #34d399)",
    },
    {
      label: "EOD EXPOSURE",
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

      {/* The page-local tape pill that lived here (showing V10's
          journal.market_window) was a duplicate of the V11
          <TapeStatusPill /> already rendered in the app shell at
          frontend/src/app/(app)/layout.tsx. The V11 pill consumes
          /api/market/rally-prefix's state (POWERTREND / UPTREND / RALLY
          MODE / CORRECTION) and is the canonical surface — no need for
          a second one on the dashboard reading the deprecated V10
          vocabulary. */}

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
      <div className="grid grid-cols-5 gap-3.5 mb-6">
        {kpis.map((kpi) => <KPITile key={kpi.label} {...kpi} />)}
      </div>
      {/* The cash + positions sub-row that lived here was removed once
          journal.end_nlv became the single source of truth. The breakdown
          was a debugging aid for the period when live and journal NLV ran
          in parallel — now it's redundant (the Live Exposure tile encodes
          the position-to-NLV ratio) and slightly misleading (the live
          breakdown wouldn't match the journal-headlined NLV). The
          underlying numbers still live on Trade Journal (positions) and
          Settings → Cash Transactions (cash). */}

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

        {/* Leading-indicator panels — hidden when EC maximized */}
        {!ecMaximized && (() => {
          // ─── Last 10 Trades ────────────────────────────────────────────
          // Combine open + closed; per-trade P&L uses overall_pl for open
          // (unrealized + realized partial closures) and realized_pl for
          // closed. Window of 10 most-recent by open_date.
          const allTradesForLast10 = [
            ...openTrades.map(t => ({
              trade_id: t.trade_id,
              ticker: t.ticker || "",
              status: "OPEN",
              open_date: String(t.open_date || ""),
              pl: enrichedById[t.trade_id]?.overall_pl ?? 0,
              rule: (t as any).rule || "",
            })),
            ...closedTrades.map((t: any) => ({
              trade_id: t.trade_id,
              ticker: t.ticker || "",
              status: "CLOSED",
              open_date: String(t.open_date || ""),
              pl: parseFloat(String(t.realized_pl || 0)),
              rule: t.rule || "",
            })),
          ];
          const ltdWinRate = computeWinRate(closedTrades as TradePosition[]);
          const last10 = computeLast10Stats(allTradesForLast10, ltdWinRate);

          // ─── Discipline Pulse (trailing 30 closed, with LTD baselines) ─
          const closedArr = closedTrades as TradePosition[];
          const trailing30 = trailingClosedTrades(closedArr, 30);
          const trailing30Losses = trailingClosedLosses(closedArr, 30);
          const ltdLosses = closedArr.filter((t: TradePosition) => parseFloat(String(t.realized_pl || 0)) < 0);

          const compliance = computeOnePctCompliance(trailing30Losses, history);
          const complianceLtd = computeOnePctCompliance(ltdLosses, history);
          const hr = computeHoldRatio(trailing30);
          const hrLtd = computeHoldRatio(closedArr);
          const pf30 = computeProfitFactor(trailing30);
          const pfLtd = computeProfitFactor(closedArr);
          // Profit Factor needs at least one losing trade in the window
          // to be meaningful — if all wins (or empty), grossLoss=0 and the
          // function returns 0. Tile renders "—" in that case.
          const pf30Losers = trailing30.filter(t => parseFloat(String(t.realized_pl || 0)) < 0).length;

          const dpHeaderSubtitle =
            trailing30.length === 0 ? "no closed trades yet"
            : trailing30.length < 30 ? `trailing ${trailing30.length} closed`
            : "trailing 30 closed";

          const fmt$ = (n: number) =>
            `$${n >= 0 ? "+" : "-"}${Math.abs(n).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
          const outcomeColor = (o: "win" | "loss" | "be") =>
            o === "win" ? "#08a86b" : o === "loss" ? "#e5484d" : "var(--ink-4)";
          const pctColor = (n: number) => (n > 0 ? "#08a86b" : n < 0 ? "#e5484d" : "var(--ink-3)");

          const winRateDelta = last10.winRate - ltdWinRate;
          const winRateDeltaColor = winRateDelta >= 0 ? "#08a86b" : "#e5484d";

          return (
            <div className="flex flex-col gap-3.5">
              {/* ━━━ PANEL 1 — Last 10 Trades ━━━ */}
              <div className="rounded-[14px] overflow-hidden flex flex-col" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
                <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
                  <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
                  <span className="text-[13px] font-semibold">Last {last10.count > 0 && last10.count < 10 ? last10.count : 10} Trades</span>
                  <span className="text-xs ml-auto" style={{ color: "var(--ink-4)" }}>recent momentum</span>
                </div>
                <div className="flex-1 p-[18px] flex flex-col gap-3.5">
                  {pricesStale && last10.trades.some(t => t.status === "OPEN") && (
                    <div className="text-[11px] px-3 py-1.5 rounded-[8px]"
                         style={{ background: "color-mix(in oklab, #d97706 10%, var(--surface))", color: "#92400e", border: "1px solid color-mix(in oklab, #d97706 20%, var(--border))" }}>
                      ⚠️ Live prices unavailable — open trades use cost basis
                    </div>
                  )}

                  {last10.count === 0 ? (
                    <div className="text-[12px] py-6 text-center" style={{ color: "var(--ink-4)" }}>No trades yet</div>
                  ) : (
                    <>
                      {/* 4 mini tiles */}
                      <div className="grid grid-cols-2 gap-2.5">
                        <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                          <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>Win Rate</div>
                          <div className="text-[18px] font-semibold mt-1" style={{ fontFamily: "var(--font-jetbrains), monospace", color: last10.winRate >= 50 ? "#08a86b" : last10.winRate >= 40 ? "#f59f00" : "#e5484d" }}>
                            {last10.winRate.toFixed(0)}%
                          </div>
                          <div className="text-[10px] mt-0.5" style={{ color: winRateDeltaColor }}>
                            vs LTD {ltdWinRate.toFixed(0)}% ({winRateDelta >= 0 ? "+" : ""}{winRateDelta.toFixed(0)}pp)
                          </div>
                          <div className="text-[9px] mt-0.5" style={{ color: "var(--ink-4)" }}>(open trades by current value)</div>
                        </div>
                        <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                          <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>Net P&L</div>
                          <div className="text-[18px] font-semibold mt-1 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace", color: pctColor(last10.netPl) }}>
                            {fmt$(last10.netPl)}
                          </div>
                          <div className="text-[10px] mt-0.5" style={{ color: "var(--ink-4)" }}>across {last10.count} trades</div>
                        </div>
                        <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                          <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>Avg W / L</div>
                          <div className="text-[14px] font-semibold mt-1 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                            <span style={{ color: "#08a86b" }}>{fmt$(last10.avgWin)}</span>
                            <span style={{ color: "var(--ink-4)" }}> / </span>
                            <span style={{ color: "#e5484d" }}>{fmt$(last10.avgLoss)}</span>
                          </div>
                        </div>
                        <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                          <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>Profit Factor</div>
                          <div className="text-[18px] font-semibold mt-1" style={{ fontFamily: "var(--font-jetbrains), monospace", color: last10.profitFactor >= 2 ? "#08a86b" : last10.profitFactor >= 1 ? "#f59f00" : "#e5484d" }}>
                            {last10.profitFactor > 0 ? `${last10.profitFactor.toFixed(2)}x` : "—"}
                          </div>
                        </div>
                      </div>

                      {/* Trade Sequence Strip — oldest → newest */}
                      <div>
                        <div className="text-[10px] uppercase tracking-[0.10em] font-semibold mb-2" style={{ color: "var(--ink-4)" }}>
                          Sequence · oldest → newest
                        </div>
                        <div className="flex items-center gap-1" data-testid="last10-sequence">
                          {last10.trades.map((t, i) => {
                            const daysHeld = (() => {
                              if (!t.open_date) return null;
                              const open = new Date(t.open_date);
                              if (isNaN(open.getTime())) return null;
                              return Math.max(0, Math.floor((Date.now() - open.getTime()) / 86_400_000));
                            })();
                            const tooltipFlipLeft = i >= 5;
                            return (
                              <Link key={t.trade_id}
                                    href={`/trade-journal?trade_id=${encodeURIComponent(t.trade_id)}`}
                                    onMouseEnter={() => setHoveredSquareIdx(i)}
                                    onMouseLeave={() => setHoveredSquareIdx(null)}
                                    className="relative w-7 h-7 rounded-[4px] cursor-pointer transition-transform hover:scale-110"
                                    style={{ background: outcomeColor(t.outcome) }}
                                    aria-label={`${t.ticker} ${t.status}, P&L ${fmt$(t.pl)}`}>
                                {hoveredSquareIdx === i && (
                                  <div className="absolute z-50 p-2.5 rounded-[8px] text-left whitespace-nowrap pointer-events-none"
                                       data-testid={`last10-tooltip-${i}`}
                                       style={{
                                         bottom: "calc(100% + 6px)",
                                         [tooltipFlipLeft ? "right" : "left"]: 0,
                                         background: "var(--surface)",
                                         border: "1px solid var(--border)",
                                         boxShadow: "var(--card-shadow)",
                                         minWidth: 140,
                                       }}>
                                    <div className="text-[12px] font-bold" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{t.ticker}</div>
                                    <div className="text-[12px] font-bold mt-0.5 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace", color: outcomeColor(t.outcome) }}>{fmt$(t.pl)}</div>
                                    <div className="text-[10px] mt-0.5" style={{ color: "var(--ink-4)" }}>{t.status}</div>
                                    <div className="text-[10px] mt-0.5" style={{ color: "var(--ink-4)" }}>
                                      {String(t.open_date || "").slice(0, 10)}{daysHeld != null ? ` · ${daysHeld}d held` : ""}
                                    </div>
                                    {t.rule && (
                                      <div className="text-[10px] mt-0.5" style={{ color: "var(--ink-3)" }}>{t.rule}</div>
                                    )}
                                  </div>
                                )}
                              </Link>
                            );
                          })}
                        </div>
                      </div>
                    </>
                  )}
                </div>
              </div>

              {/* ━━━ PANEL 2 — Discipline Pulse ━━━ */}
              <div className="rounded-[14px] overflow-hidden flex flex-col" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
                <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
                  <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
                  <span className="text-[13px] font-semibold">Discipline Pulse</span>
                  <span className="text-xs ml-auto" style={{ color: "var(--ink-4)" }}>{dpHeaderSubtitle}</span>
                </div>
                <div className="flex-1 p-[18px] grid grid-cols-1 gap-2.5">
                  {/* 1% Rule — trailing 30 closed losses, with LTD baseline */}
                  <Link href="/analytics?tab=drawdown" className="block p-3 rounded-[10px] transition-colors hover:bg-[var(--surface-2)]" style={{ border: "1px solid var(--border)", textDecoration: "none", color: "inherit" }}>
                    <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>1% Rule Compliance</div>
                    {compliance.totalLosses === 0 ? (
                      <>
                        <div className="text-[18px] font-semibold mt-1" style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink-4)" }}>—</div>
                        <div className="text-[11px] mt-0.5" style={{ color: "var(--ink-4)" }}>no losers in window</div>
                        {complianceLtd.totalLosses > 0 && (
                          <div className="text-[10px] mt-0.5" style={{ color: "var(--ink-4)" }}>
                            LTD: {complianceLtd.passRate.toFixed(0)}% ({complianceLtd.totalLosses} closed losses)
                          </div>
                        )}
                      </>
                    ) : (
                      <>
                        <div className="text-[18px] font-semibold mt-1" style={{ fontFamily: "var(--font-jetbrains), monospace", color: compliance.passRate >= 95 ? "#08a86b" : compliance.passRate >= 85 ? "#d97706" : "#e5484d" }}>
                          {compliance.passRate.toFixed(0)}% within rule
                        </div>
                        <div className="text-[11px] mt-0.5" style={{ color: compliance.breaches > 0 ? "#e5484d" : "var(--ink-4)" }}>
                          {compliance.breaches} {compliance.breaches === 1 ? "breach" : "breaches"} of {compliance.totalLosses}
                        </div>
                        {complianceLtd.totalLosses > 0 && (
                          <div className="text-[10px] mt-0.5" style={{ color: "var(--ink-4)" }}>
                            LTD: {complianceLtd.passRate.toFixed(0)}% ({complianceLtd.totalLosses} closed losses)
                          </div>
                        )}
                      </>
                    )}
                  </Link>

                  {/* Hold Ratio — trailing 30 closed, with LTD baseline */}
                  <Link href="/analytics?tab=overview" className="block p-3 rounded-[10px] transition-colors hover:bg-[var(--surface-2)]" style={{ border: "1px solid var(--border)", textDecoration: "none", color: "inherit" }}>
                    <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>Hold Ratio (W/L)</div>
                    {hr.losersHold === 0 ? (
                      <>
                        <div className="text-[18px] font-semibold mt-1" style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink-4)" }}>—</div>
                        <div className="text-[11px] mt-0.5" style={{ color: "var(--ink-4)" }}>no losers in window</div>
                        {hrLtd.losersHold > 0 && (
                          <div className="text-[10px] mt-0.5" style={{ color: "var(--ink-4)" }}>LTD: {hrLtd.ratio.toFixed(2)}x</div>
                        )}
                      </>
                    ) : (
                      <>
                        <div className="text-[18px] font-semibold mt-1" style={{ fontFamily: "var(--font-jetbrains), monospace", color: hr.ratio >= 1 ? "#08a86b" : "#e5484d" }}>
                          {hr.ratio.toFixed(2)}x
                        </div>
                        <div className="text-[11px] mt-0.5" style={{ color: "var(--ink-4)" }}>
                          {hr.ratio >= 1 ? "letting winners run" : "holding losers too long"}
                        </div>
                        {hrLtd.losersHold > 0 && (
                          <div className="text-[10px] mt-0.5" style={{ color: "var(--ink-4)" }}>LTD: {hrLtd.ratio.toFixed(2)}x</div>
                        )}
                      </>
                    )}
                  </Link>

                  {/* Profit Factor — trailing 30 closed, with LTD baseline */}
                  <Link href="/analytics?tab=overview" className="block p-3 rounded-[10px] transition-colors hover:bg-[var(--surface-2)]" style={{ border: "1px solid var(--border)", textDecoration: "none", color: "inherit" }}>
                    <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>Profit Factor</div>
                    {trailing30.length === 0 ? (
                      <>
                        <div className="text-[18px] font-semibold mt-1" style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink-4)" }}>—</div>
                        <div className="text-[11px] mt-0.5" style={{ color: "var(--ink-4)" }}>no closed trades in window</div>
                        {pfLtd > 0 && (
                          <div className="text-[10px] mt-0.5" style={{ color: "var(--ink-4)" }}>LTD: {pfLtd.toFixed(2)}x</div>
                        )}
                      </>
                    ) : pf30Losers === 0 ? (
                      <>
                        <div className="text-[18px] font-semibold mt-1" style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink-4)" }}>—</div>
                        <div className="text-[11px] mt-0.5" style={{ color: "var(--ink-4)" }}>no losers in window</div>
                        {pfLtd > 0 && (
                          <div className="text-[10px] mt-0.5" style={{ color: "var(--ink-4)" }}>LTD: {pfLtd.toFixed(2)}x</div>
                        )}
                      </>
                    ) : (
                      <>
                        <div className="text-[18px] font-semibold mt-1" style={{ fontFamily: "var(--font-jetbrains), monospace", color: pf30 >= 2 ? "#08a86b" : pf30 >= 1 ? "#f59f00" : "#e5484d" }}>
                          {pf30.toFixed(2)}x
                        </div>
                        <div className="text-[11px] mt-0.5" style={{ color: "var(--ink-4)" }}>{trailing30.length} closed in window</div>
                        {pfLtd > 0 && (
                          <div className="text-[10px] mt-0.5" style={{ color: "var(--ink-4)" }}>LTD: {pfLtd.toFixed(2)}x</div>
                        )}
                      </>
                    )}
                  </Link>
                </div>
              </div>
            </div>
          );
        })()}
      </div>
    </div>
  );
}
