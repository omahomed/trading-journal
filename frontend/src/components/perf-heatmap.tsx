"use client";

import { useState, useEffect, useMemo } from "react";
import { api, getActivePortfolio, type TradePosition, type TradeDetail, type JournalHistoryPoint } from "@/lib/api";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";
import { TradeOverviewSidecar } from "./trade-overview-sidecar";

function lerp(a: number, b: number, t: number) { return a + (b - a) * t; }
function heatColor(val: number, zMin: number, zMax: number): string {
  const mid = 0;
  if (val <= zMin) return "#e5484d";
  if (val >= zMax) return "#08a86b";
  if (val <= mid) {
    const t = (val - zMin) / (mid - zMin);
    return `rgb(${lerp(229, 255, t).toFixed(0)}, ${lerp(72, 255, t).toFixed(0)}, ${lerp(77, 255, t).toFixed(0)})`;
  }
  const t = (val - mid) / (zMax - mid);
  return `rgb(${lerp(255, 8, t).toFixed(0)}, ${lerp(255, 168, t).toFixed(0)}, ${lerp(255, 107, t).toFixed(0)})`;
}

// Date-range presets. "from" is an inclusive lower bound; "to" is always
// today. Open trades pass the window filter unconditionally (they're active
// contributors regardless of when opened); closed trades pass when their
// closed_date >= from. Week convention: Monday as the first day (trading
// week orientation).
type DatePreset = "wtd" | "mtd" | "qtd" | "ytd";

function computePresetRange(preset: DatePreset): { fromISO: string; label: string } {
  const now = new Date();
  const y = now.getFullYear();
  const m = now.getMonth();
  const d = now.getDate();
  const day = now.getDay();              // 0=Sun..6=Sat
  const mondayOffset = (day + 6) % 7;     // 0=Mon..6=Sun
  let from: Date;
  let label: string;
  switch (preset) {
    case "wtd":
      from = new Date(y, m, d - mondayOffset);
      label = "This Week";
      break;
    case "mtd":
      from = new Date(y, m, 1);
      label = "This Month";
      break;
    case "qtd":
      from = new Date(y, Math.floor(m / 3) * 3, 1);
      label = "This Quarter";
      break;
    case "ytd":
    default:
      from = new Date(y, 0, 1);
      label = `${y} YTD`;
  }
  const fromISO = `${from.getFullYear()}-${String(from.getMonth() + 1).padStart(2, "0")}-${String(from.getDate()).padStart(2, "0")}`;
  return { fromISO, label };
}

export function PerfHeatmap({ navColor }: { navColor: string }) {
  const [trades, setTrades] = useState<TradePosition[]>([]);
  const [openTrades, setOpenTrades] = useState<TradePosition[]>([]);
  const [journal, setJournal] = useState<JournalHistoryPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<"all" | "open" | "closed">("all");
  const [metricMode, setMetricMode] = useState<"return" | "rmult" | "impact">("return");
  const [datePreset, setDatePreset] = useState<DatePreset>("ytd");
  // Group 8: Stocks vs Options live on disjoint color scales. Default Stocks
  // (primary instrument, hides premium-on-premium outliers like AMD +196.7%).
  // Persisted because the filter is sticky — a user who works mostly in
  // options shouldn't have to re-select on every page load.
  const [instrumentMode, setInstrumentMode] = useState<"stocks" | "options">(() => {
    try {
      const stored = localStorage.getItem("perf-heatmap-instrument");
      return stored === "options" ? "options" : "stocks";
    } catch { return "stocks"; }
  });
  useEffect(() => {
    try { localStorage.setItem("perf-heatmap-instrument", instrumentMode); } catch {}
  }, [instrumentMode]);
  const [selectedTrade, setSelectedTrade] = useState<string | null>(null);
  const [allDetails, setAllDetails] = useState<TradeDetail[]>([]);

  useEffect(() => {
    Promise.all([
      api.tradesClosed(getActivePortfolio(), 1000).catch((err) => {
        log.error("perf-heatmap", "tradesClosed fetch failed", err);
        return [];
      }),
      api.tradesOpen(getActivePortfolio()).catch((err) => {
        log.error("perf-heatmap", "tradesOpen fetch failed", err);
        return [];
      }),
      api.journalHistory(getActivePortfolio(), 0).catch((err) => {
        log.error("perf-heatmap", "journalHistory fetch failed", err);
        return [];
      }),
      api.tradesRecent(getActivePortfolio(), 2000).catch((err) => {
        log.error("perf-heatmap", "tradesRecent fetch failed", err);
        return { details: [], lot_closures: [] };
      }),
    ]).then(([closed, open, jrnl, details]) => {
      setTrades(closed as TradePosition[]);
      setOpenTrades(open as TradePosition[]);
      setJournal(jrnl as JournalHistoryPoint[]);
      setAllDetails(details.details);
      setLoading(false);
    });
  }, []);

  const dateRange = useMemo(() => computePresetRange(datePreset), [datePreset]);

  const heatData = useMemo(() => {
    // Date-preset window filter. Open trades pass unconditionally (they're
    // active contributors right now regardless of when they opened); closed
    // trades pass when their closed_date falls in [from, today].
    let all = [...openTrades, ...trades].filter(t => {
      const isOpen = (t.status || "").toUpperCase() === "OPEN";
      if (isOpen) return true;
      const cd = String(t.closed_date || "").slice(0, 10);
      return cd >= dateRange.fromISO;
    });

    if (viewMode === "open") all = all.filter(t => (t.status || "").toUpperCase() === "OPEN");
    else if (viewMode === "closed") all = all.filter(t => (t.status || "").toUpperCase() === "CLOSED");

    // Instrument filter (Group 8). Mirrors the canonical isOption pattern
    // from trade-journal.tsx:1095 — instrument_type column first, ticker-
    // shape fallback for any legacy row that pre-dates Migration 016.
    const isOptionRow = (t: TradePosition) =>
      String(t.instrument_type || "").toUpperCase() === "OPTION"
      || /^\S+\s+\d{6}\s+\$[0-9.]+(C|P)$/.test(String(t.ticker || ""));
    all = instrumentMode === "options"
      ? all.filter(isOptionRow)
      : all.filter(t => !isOptionRow(t));

    // Compute metrics
    const jSorted = [...journal].sort((a, b) => String(a.day).localeCompare(String(b.day)));

    return all.map(t => {
      const pl = parseFloat(String(t.realized_pl || 0));
      const retPct = parseFloat(String(t.return_pct || 0));
      const rb = parseFloat(String(t.risk_budget || 0));
      const rMult = rb > 0 ? pl / rb : 0;
      const isOpen = (t.status || "").toUpperCase() === "OPEN";

      // NLV impact
      let impact = 0;
      const od = String(t.open_date || "").slice(0, 10);
      const match = jSorted.filter(h => String(h.day).slice(0, 10) <= od);
      if (match.length > 0) {
        const nlv = match[match.length - 1].end_nlv;
        if (nlv > 0) impact = (pl / nlv) * 100;
      }

      return { ticker: t.ticker, tradeId: t.trade_id, status: isOpen ? "O" : "C", retPct, rMult, impact };
    }).sort((a, b) => {
      const key = metricMode === "return" ? "retPct" : metricMode === "rmult" ? "rMult" : "impact";
      return (b as any)[key] - (a as any)[key];
    });
  }, [trades, openTrades, journal, viewMode, metricMode, instrumentMode, dateRange.fromISO]);

  if (loading) return <div className="animate-pulse"><div className="h-[90px] rounded-[14px]" style={{ background: "var(--bg-2)" }} /></div>;

  // Metric config
  const cfg = metricMode === "return"
    ? { key: "retPct" as const, zMin: -7, zMax: 15, fmt: (v: number) => `${v.toFixed(1)}%`, label: "Return %" }
    : metricMode === "rmult"
    ? { key: "rMult" as const, zMin: -1.2, zMax: 3, fmt: (v: number) => `${v.toFixed(2)}R`, label: "R-Multiple" }
    : { key: "impact" as const, zMin: -1, zMax: 2, fmt: (v: number) => `${v.toFixed(2)}%`, label: "Account Impact %" };

  // Cohort-aware bounds (Group 8). Stocks keep fixed bounds — calibrated
  // for equity-scale returns. Options expand the gradient to fit outliers
  // (premium-on-premium can hit +200%; clamping would paint every winner
  // solid green and wash out within-cohort variance). Fixed bounds remain
  // a FLOOR — the gradient never contracts below the equity-tuned range.
  let zMin = cfg.zMin;
  let zMax = cfg.zMax;
  if (instrumentMode === "options" && heatData.length > 0) {
    const vals = heatData.map(d => (d as any)[cfg.key] as number).filter(v => Number.isFinite(v));
    if (vals.length > 0) {
      zMin = Math.min(cfg.zMin, ...vals);
      zMax = Math.max(cfg.zMax, ...vals);
    }
  }

  const cols = 8;
  const fatalities = heatData.filter(d => d.impact < -1).length;
  const avgImpact = heatData.length > 0 ? heatData.reduce((a, d) => a + d.impact, 0) / heatData.length : 0;
  const worst = heatData.length > 0 ? heatData.reduce((w, d) => (d as any)[cfg.key] < (w as any)[cfg.key] ? d : w) : null;

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          Performance <em className="italic" style={{ color: navColor }}>Heat Map</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>{getActivePortfolio()} · {dateRange.label} · {heatData.length} {heatData.length === 1 ? "trade" : "trades"}</div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-4 mb-5 flex-wrap">
        <div className="flex p-0.5 rounded-[8px] gap-0.5" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
          {([
            { key: "wtd" as const, label: "Week" },
            { key: "mtd" as const, label: "Month" },
            { key: "qtd" as const, label: "Quarter" },
            { key: "ytd" as const, label: "YTD" },
          ]).map(p => (
            <button key={p.key} onClick={() => setDatePreset(p.key)}
                    className="px-3 py-1 rounded-md text-[11px] font-medium transition-all"
                    style={{ background: datePreset === p.key ? "var(--surface)" : "transparent", color: datePreset === p.key ? "var(--ink)" : "var(--ink-4)" }}>
              {p.label}
            </button>
          ))}
        </div>
        <div className="flex p-0.5 rounded-[8px] gap-0.5" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
          {(["all", "open", "closed"] as const).map(m => (
            <button key={m} onClick={() => setViewMode(m)}
                    className="px-3 py-1 rounded-md text-[11px] font-medium transition-all capitalize"
                    style={{ background: viewMode === m ? "var(--surface)" : "transparent", color: viewMode === m ? "var(--ink)" : "var(--ink-4)" }}>
              {m === "all" ? "All" : m === "open" ? "Open Only" : "Closed Only"}
            </button>
          ))}
        </div>
        <div className="flex p-0.5 rounded-[8px] gap-0.5" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
          {([
            { key: "return" as const, label: "Return %" },
            { key: "rmult" as const, label: "R-Multiple" },
            { key: "impact" as const, label: "Impact %" },
          ]).map(m => (
            <button key={m.key} onClick={() => setMetricMode(m.key)}
                    className="px-3 py-1 rounded-md text-[11px] font-medium transition-all"
                    style={{ background: metricMode === m.key ? "var(--surface)" : "transparent", color: metricMode === m.key ? "var(--ink)" : "var(--ink-4)" }}>
              {m.label}
            </button>
          ))}
        </div>
        <div className="flex p-0.5 rounded-[8px] gap-0.5" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
          {(["stocks", "options"] as const).map(m => (
            <button key={m} onClick={() => setInstrumentMode(m)}
                    className="px-3 py-1 rounded-md text-[11px] font-medium transition-all capitalize"
                    style={{ background: instrumentMode === m ? "var(--surface)" : "transparent", color: instrumentMode === m ? "var(--ink)" : "var(--ink-4)" }}>
              {m}
            </button>
          ))}
        </div>
      </div>

      {/* Heatmap grid */}
      {heatData.length > 0 ? (
        <div className="rounded-[14px] overflow-hidden mb-5 p-4" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
          <div className="grid gap-[4px]" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
            {heatData.map((d, i) => {
              const val = (d as any)[cfg.key] as number;
              const bg = heatColor(val, zMin, zMax);
              const textColor = Math.abs(val) > (zMax - zMin) * 0.3 ? "#fff" : "var(--ink)";
              return (
                <div key={i} className="rounded-[8px] p-3 text-center transition-transform duration-150 hover:scale-105 cursor-pointer"
                     style={{ background: bg, minHeight: 70, outline: selectedTrade === d.tradeId ? `2px solid ${navColor}` : "none", outlineOffset: 1 }}
                     onClick={() => setSelectedTrade(selectedTrade === d.tradeId ? null : d.tradeId)}>
                  <div className="text-[11px] font-bold" style={{ color: textColor }}>{d.ticker}</div>
                  <div className="text-[9px] opacity-70" style={{ color: textColor }}>({d.status})</div>
                  <div className="text-[13px] font-extrabold mt-1" style={{ color: textColor }}>{cfg.fmt(val)}</div>
                </div>
              );
            })}
          </div>
        </div>
      ) : (
        <div className="mb-5 text-center py-12 text-sm" style={{ color: "var(--ink-4)" }}>No trades match this view.</div>
      )}

      {/* Audit footer */}
      <div className="grid grid-cols-3 gap-3">
        <div className="p-4 rounded-[12px]" style={{ background: `color-mix(in oklab, #e5484d 6%, var(--surface))`, border: "1px solid var(--border)" }}>
          <div className="text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Fatal Hits ({">"}1% Portfolio)</div>
          <div className="text-[22px] font-extrabold mt-1" style={{ color: "#e5484d" }}>{fatalities} Trades</div>
          <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>Target: 0</div>
        </div>
        <div className="p-4 rounded-[12px]" style={{ border: "1px solid var(--border)" }}>
          <div className="text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Avg Portfolio Impact</div>
          <div className="text-[22px] font-extrabold mt-1" style={{ color: pctColor(avgImpact) }}>{avgImpact.toFixed(2)}%</div>
        </div>
        <div className="p-4 rounded-[12px]" style={{ background: `color-mix(in oklab, #e5484d 6%, var(--surface))`, border: "1px solid var(--border)" }}>
          <div className="text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Worst Impact</div>
          <div className="text-[22px] font-extrabold mt-1" style={{ color: "#e5484d" }}>
            {worst ? `${worst.ticker} (${cfg.fmt((worst as any)[cfg.key])})` : "—"}
          </div>
        </div>
      </div>

      {/* Slide-over panel — extracted to a shared component (2026-07-25)
          so Campaign Review's right-click context menu can render the
          same drill-in. */}
      {selectedTrade && (() => {
        const allCampaigns = [...openTrades, ...trades];
        const trade = allCampaigns.find(t => t.trade_id === selectedTrade);
        if (!trade) return null;
        return (
          <TradeOverviewSidecar
            trade={trade}
            details={allDetails}
            portfolio={getActivePortfolio()}
            onClose={() => setSelectedTrade(null)}
          />
        );
      })()}
    </div>
  );
}

function pctColor(v: number) { return v > 0 ? "#08a86b" : v < 0 ? "#e5484d" : "var(--ink-3)"; }
