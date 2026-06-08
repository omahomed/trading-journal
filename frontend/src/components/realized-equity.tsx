"use client";

// Realized Equity — standalone page showing the closed-positions-only
// equity curve. Sister to the dashboard's Equity Curve panel (mark-to-
// market), but movement is event-driven: the curve only steps on dates
// that had at least one SELL closure. The dashboard's panel is left
// untouched; the chart chrome here is DUPLICATED from dashboard.tsx
// ([dashboard.tsx:404-616]) intentionally so the two stay design-paired
// without a premature shared-component extraction.
//
// Data flow:
//   - /api/realized/curve            → realized series + summary
//   - /api/journal/history (days=0)  → daily date axis + SPY/Nasdaq cumulative %
//   - /api/events                    → dashed event ReferenceLines
//
// The realized series is SPARSE (one point per close date); we forward-
// fill onto the journal's daily axis so the line stair-steps cleanly
// and benchmarks stay co-axial. SPY/Nasdaq are rebased to 0% at
// REALIZED_START so they share the % axis with the realized line.
//
// Range toggle adjusts the visible window only — it does NOT rebase
// the realized line. "Realized since 2026-01-01" is the durable anchor.

import { useState, useEffect, useMemo } from "react";
import {
  api, getActivePortfolio,
  type JournalHistoryPoint, type RealizedCurveResponse,
} from "@/lib/api";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";
import {
  ResponsiveContainer, ComposedChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ReferenceLine,
} from "recharts";

const REALIZED_START = "2026-01-01";

// HeroCard — cloned from dashboard's KPITile pattern. Same gradient
// background + radial-glow + JetBrains-mono value, scoped here so the
// dashboard doesn't have to export anything.
function HeroCard({ label, value, sub, gradient }: {
  label: string; value: string; sub?: string; gradient: string;
}) {
  return (
    <div
      className="relative overflow-hidden rounded-[14px] p-[14px_16px] text-white flex flex-col justify-between min-h-[90px]"
      style={{ background: gradient, boxShadow: "var(--kpi-shadow)" }}
    >
      <div
        className="absolute -right-5 -top-5 w-[100px] h-[100px] rounded-full"
        style={{ background: "radial-gradient(circle, rgba(255,255,255,0.18), transparent 65%)" }}
      />
      <div className="relative z-10">
        <div className="text-[9px] font-semibold uppercase tracking-[0.10em] opacity-85">{label}</div>
        <div
          className="text-[22px] font-semibold tracking-tight mt-0.5 privacy-mask"
          style={{ fontFamily: "var(--font-jetbrains), monospace" }}
        >
          {value}
        </div>
        {sub && <div className="text-[10px] mt-1 opacity-75">{sub}</div>}
      </div>
    </div>
  );
}

type EcRange = "All" | "1Y" | "6M" | "3M";

interface ChartRow {
  day: string;
  realized: number;       // cumulative realized % since REALIZED_START (forward-filled)
  realized_pl: number;    // cumulative $ (for tooltip display, optional future use)
  spy: number;            // rebased to 0% at REALIZED_START
  ndx: number;            // rebased to 0% at REALIZED_START
}

export function RealizedEquity({ navColor }: { navColor: string }) {
  const [history, setHistory] = useState<JournalHistoryPoint[]>([]);
  const [realized, setRealized] = useState<RealizedCurveResponse | null>(null);
  const [events, setEvents] = useState<Array<{ event_date?: string; label?: string; category?: string }>>([]);
  const [loading, setLoading] = useState(true);
  const [ecRange, setEcRange] = useState<EcRange>("All");
  const [showEvents, setShowEvents] = useState(true);

  useEffect(() => {
    Promise.all([
      api.journalHistory(getActivePortfolio(), 0).catch((err) => {
        log.error("realized-equity", "journal history fetch failed", err);
        return [] as JournalHistoryPoint[];
      }),
      api.realizedCurve(getActivePortfolio(), REALIZED_START).catch((err) => {
        log.error("realized-equity", "realized curve fetch failed", err);
        return null;
      }),
      api.events().catch((err) => {
        log.error("realized-equity", "events fetch failed", err);
        return [];
      }),
    ]).then(([hist, real, ev]) => {
      setHistory(hist as JournalHistoryPoint[]);
      if (real && !(typeof real === "object" && "error" in real)) {
        setRealized(real as RealizedCurveResponse);
      } else {
        setRealized(null);
      }
      setEvents(Array.isArray(ev) ? ev : []);
      setLoading(false);
    });
  }, []);

  // Build chart data on the UNION of journal days ∪ realized-series days
  // (both filtered to >= REALIZED_START), iterated chronologically. Forward-
  // fill in both directions:
  //   - Days the journal has but realized doesn't → carry the last realized
  //     value (the common case — non-closure trading day).
  //   - Days the realized series has but the journal doesn't → carry the
  //     last benchmark values. This happens when a close lands on a day
  //     the user hasn't journaled yet (e.g. mid-day today, before the
  //     Daily Routine save) — without the union, the chart's rightmost
  //     point would lag behind the summary's `total_realized_pl`.
  //
  // SPY/Nasdaq cumulative % values are rebased so they start at 0% on the
  // first journal day in the window — same anchor as before.
  const ecData: ChartRow[] = useMemo(() => {
    if (history.length === 0 && (realized?.series?.length ?? 0) === 0) return [];

    const journalByDay: Record<string, JournalHistoryPoint> = {};
    for (const h of history) {
      const k = String(h.day).slice(0, 10);
      if (k >= REALIZED_START) journalByDay[k] = h;
    }

    const realByDay: Record<string, { pl: number; pct: number }> = {};
    for (const p of (realized?.series ?? [])) {
      const k = String(p.day).slice(0, 10);
      if (k >= REALIZED_START) realByDay[k] = { pl: p.cum_realized_pl, pct: p.cum_realized_pct };
    }

    const allDays = Array.from(
      new Set([...Object.keys(journalByDay), ...Object.keys(realByDay)]),
    ).sort();
    if (allDays.length === 0) return [];

    // Rebase baseline = first journal day in the window. If only realized
    // data exists (no journal at all in the window), SPY/Nasdaq stay at 0%.
    const firstJournalDay = Object.keys(journalByDay).sort()[0];
    const baseSpy = firstJournalDay ? (journalByDay[firstJournalDay].spy_ltd || 0) : 0;
    const baseNdx = firstJournalDay ? (journalByDay[firstJournalDay].ndx_ltd || 0) : 0;

    let lastPct = 0;
    let lastPl = 0;
    let lastSpy = 0;
    let lastNdx = 0;
    return allDays.map(day => {
      const j = journalByDay[day];
      if (j) {
        lastSpy = (j.spy_ltd || 0) - baseSpy;
        lastNdx = (j.ndx_ltd || 0) - baseNdx;
      }
      const r = realByDay[day];
      if (r) { lastPct = r.pct; lastPl = r.pl; }
      return {
        day,
        realized: parseFloat(lastPct.toFixed(2)),
        realized_pl: lastPl,
        spy: parseFloat(lastSpy.toFixed(2)),
        ndx: parseFloat(lastNdx.toFixed(2)),
      };
    });
  }, [history, realized]);

  // Visible-window slice for the range toggle. Slicing the array does
  // NOT rebase — realized values stay anchored to REALIZED_START.
  const visible = useMemo(() => {
    if (ecData.length === 0) return [];
    if (ecRange === "All") return ecData;
    const now = new Date();
    const months = ecRange === "1Y" ? 12 : ecRange === "6M" ? 6 : 3;
    const cutoff = new Date(now.getFullYear(), now.getMonth() - months, now.getDate());
    const cutoffStr = cutoff.toISOString().slice(0, 10);
    return ecData.filter(h => h.day >= cutoffStr);
  }, [ecData, ecRange]);

  const summary = realized?.summary ?? null;
  const last = visible.length > 0 ? visible[visible.length - 1] : null;

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }} data-testid="realized-equity-root">
      {/* Page header */}
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1
          className="font-normal text-[32px] tracking-tight m-0"
          style={{ fontFamily: "var(--font-fraunces), Georgia, serif", letterSpacing: "-0.02em" }}
        >
          Realized <em className="italic" style={{ color: navColor }}>Equity</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
          Closed-positions-only curve · cumulative realized return since {REALIZED_START}
        </div>
      </div>

      {/* HeroCards summary */}
      <div className="grid grid-cols-3 gap-3 mb-5" data-testid="realized-equity-hero">
        <HeroCard
          label="Realized Banked"
          value={summary ? formatCurrency(summary.total_realized_pl, { decimals: 0, showSign: true }) : "—"}
          sub={summary ? `since ${summary.start_date}` : ""}
          gradient="linear-gradient(135deg, #10b981, #34d399)"
        />
        <HeroCard
          label="Realized Return"
          value={summary ? `${summary.realized_pct >= 0 ? "+" : ""}${summary.realized_pct.toFixed(2)}%` : "—"}
          sub={summary
            ? `vs ${formatCurrency(summary.start_nlv, { decimals: 0 })} baseline (${summary.baseline_source})`
            : ""}
          gradient="linear-gradient(135deg, #2563eb, #3b82f6)"
        />
        <HeroCard
          label="Closed Trades"
          value={summary ? String(summary.closed_count) : "—"}
          sub={summary ? "lot closures" : ""}
          gradient="linear-gradient(135deg, #8b5cf6, #a78bfa)"
        />
      </div>

      {/* Chart panel — cloned from dashboard.tsx:404-616 minus the
          mark-to-market overlays (SMAs, Exposure area + right axis,
          regime bar). Same Recharts ComposedChart structure. */}
      <div
        data-testid="realized-equity-chart-panel"
        className="rounded-[14px] overflow-hidden flex flex-col"
        style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}
      >
        <div
          className="flex items-center justify-between px-[18px] py-3"
          style={{ borderBottom: "1px solid var(--border)" }}
        >
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
            <span className="text-[13px] font-semibold">Realized Equity</span>
            <span className="text-xs" style={{ color: "var(--ink-4)" }}>
              vs Benchmark · event markers
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div
              className="flex p-0.5 rounded-[10px] gap-0.5"
              style={{ background: "var(--bg)", border: "1px solid var(--border)" }}
            >
              {(["All", "1Y", "6M", "3M"] as const).map(t => (
                <button
                  key={t}
                  onClick={() => setEcRange(t)}
                  data-testid={`re-range-${t}`}
                  className="px-3 py-1 rounded-md text-xs font-medium transition-all"
                  style={{
                    background: ecRange === t ? "var(--surface)" : "transparent",
                    color: ecRange === t ? "var(--ink)" : "var(--ink-4)",
                    boxShadow: ecRange === t ? "0 1px 2px rgba(14,20,38,0.04)" : "none",
                  }}
                >
                  {t}
                </button>
              ))}
            </div>
            <button
              onClick={() => setShowEvents(!showEvents)}
              data-testid="re-toggle-events"
              className="h-[28px] px-2.5 rounded-[8px] flex items-center gap-1.5 text-[10px] font-semibold transition-colors cursor-pointer"
              style={{
                background: showEvents ? "color-mix(in oklab, #dc2626 10%, var(--surface))" : "var(--bg)",
                border: `1px solid ${showEvents ? "color-mix(in oklab, #dc2626 25%, var(--border))" : "var(--border)"}`,
                color: showEvents ? "#dc2626" : "var(--ink-4)",
              }}
              title="Toggle event markers"
            >
              Events {showEvents ? "ON" : "OFF"}
            </button>
          </div>
        </div>

        {/* Custom legend — Realized + benchmarks. SMAs + exposure dropped. */}
        <div
          className="flex items-center gap-4 px-[18px] pt-3 pb-1 flex-wrap"
          style={{ fontSize: 11 }}
        >
          {[
            { key: "realized", label: `Realized (${last ? (last.realized >= 0 ? "+" : "") + last.realized.toFixed(1) + "%" : ""})`, color: "#1e3a8a", width: 2.5 },
            { key: "spy",      label: `SPY (${last ? (last.spy >= 0 ? "+" : "") + last.spy.toFixed(1) + "%" : ""})`, color: "#9ca3af", width: 1.5 },
            { key: "ndx",      label: `Nasdaq (${last ? (last.ndx >= 0 ? "+" : "") + last.ndx.toFixed(1) + "%" : ""})`, color: "#22c55e", width: 1.5 },
          ].map(item => (
            <div key={item.key} className="flex items-center gap-1.5" style={{ color: "var(--ink-3)" }}>
              <svg width="18" height="10">
                <line x1="0" y1="5" x2="18" y2="5" stroke={item.color} strokeWidth={item.width} />
              </svg>
              <span>{item.label}</span>
            </div>
          ))}
        </div>

        <div className="px-1 pb-3" style={{ height: 420 }}>
          {visible.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={visible} margin={{ top: 8, right: 50, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.5} vertical={false} />
                <XAxis
                  dataKey="day"
                  tick={{ fontSize: 10, fill: "var(--ink-4)" }}
                  tickLine={false}
                  axisLine={{ stroke: "var(--border)" }}
                  interval={Math.max(Math.floor(visible.length / 8), 1)}
                  tickFormatter={(v: string) => {
                    const d = new Date(v);
                    return d.toLocaleDateString("en-US", { month: "short", year: "numeric" });
                  }}
                  label={{ value: "Date", position: "insideBottom", offset: -10, fontSize: 11, fill: "var(--ink-4)" }}
                />
                <YAxis
                  tick={{ fontSize: 10, fill: "var(--ink-4)" }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v: number) => `${v}%`}
                  width={50}
                  label={{ value: "Return %", angle: -90, position: "insideLeft", offset: 10, fontSize: 11, fill: "var(--ink-4)" }}
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
                      realized: "Realized", spy: "SPY", ndx: "Nasdaq",
                    };
                    return [`${Number(value).toFixed(2)}%`, labels[String(name)] || String(name)];
                  }}
                  labelFormatter={(label: any) => {
                    const d = new Date(String(label));
                    return d.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric", year: "numeric" });
                  }}
                />
                <ReferenceLine y={0} stroke="var(--ink-4)" strokeDasharray="3 3" strokeOpacity={0.3} />

                {/* SPY — gray, monotone */}
                <Line dataKey="spy" stroke="#9ca3af" strokeWidth={1.5} dot={false} type="monotone" />
                {/* NDX — green, monotone */}
                <Line dataKey="ndx" stroke="#22c55e" strokeWidth={1.5} dot={false} type="monotone" />
                {/* Realized — thick navy, ON TOP, stepAfter so the
                    forward-filled flats render as true stairs (jumps
                    happen exactly on close dates, not interpolated). */}
                <Line dataKey="realized" stroke="#1e3a8a" strokeWidth={2.5} dot={false} type="stepAfter" />

                {/* Event markers — same colors/style as the dashboard panel. */}
                {showEvents && events.map((ev, i) => {
                  const evDate = String(ev.event_date || "").slice(0, 10);
                  const match = visible.find(d => d.day === evDate);
                  if (!match) return null;
                  const color = ev.category === "market" ? "#dc2626"
                              : ev.category === "macro"  ? "#9333ea"
                                                         : "#6b7280";
                  return (
                    <ReferenceLine
                      key={`ev-${i}`}
                      x={match.day}
                      stroke={color}
                      strokeWidth={1.5}
                      strokeDasharray="4 3"
                      strokeOpacity={0.7}
                      label={{ value: ev.label, position: "insideTopRight", fontSize: 9, fill: color, fontWeight: 600 }}
                    />
                  );
                })}
              </ComposedChart>
            </ResponsiveContainer>
          ) : (
            <div
              className="h-full flex items-center justify-center text-sm"
              data-testid="realized-equity-empty"
              style={{ color: "var(--ink-4)" }}
            >
              {loading ? "Loading…" : "No data available for selected range"}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
