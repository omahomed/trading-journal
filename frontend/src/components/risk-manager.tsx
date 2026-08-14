"use client";

import { useState, useEffect, useMemo } from "react";
import {
  api, getActivePortfolio, type JournalHistoryPoint, type TradePosition,
  type TradeDetailsBundle, type RiskLevelsResponse, type RiskLevelKey,
} from "@/lib/api";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";
import { computeEnrichedPositions, type EnrichedPosition } from "@/lib/positions";
import { SR15NudgeBanner } from "./sr15-nudge-banner";
import { SR12FloorNudgeBanner } from "./sr12-floor-nudge-banner";
import {
  ResponsiveContainer, ComposedChart, Line, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, ReferenceLine,
} from "recharts";

function KPITile({ label, value, sub, gradient, extraSub }: {
  label: string; value: string; sub: string; gradient: string;
  extraSub?: string;
}) {
  return (
    <div className="relative overflow-hidden rounded-[14px] p-[14px_16px] text-white flex flex-col justify-between h-[90px] transition-transform duration-150 hover:scale-[1.01]"
         style={{ background: gradient, boxShadow: "var(--kpi-shadow)" }}>
      <div className="absolute -right-5 -top-5 w-[100px] h-[100px] rounded-full" style={{ background: "radial-gradient(circle, rgba(255,255,255,0.18), transparent 65%)" }} />
      <div className="relative z-10">
        <div className="text-[9px] font-semibold uppercase tracking-[0.10em] opacity-85">{label}</div>
        <div className="text-[22px] font-semibold tracking-tight mt-0.5 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{value}</div>
      </div>
      <div className="relative z-10 text-[10px] font-medium opacity-80 privacy-mask">
        {sub}
        {extraSub && <div className="opacity-90">{extraSub}</div>}
      </div>
    </div>
  );
}

// Cap → gradient. Deeper cap = redder tile. CLEAR bakes green; L1
// yellow; L2 orange; L3 deep orange; L4 red. Keeps the "active level"
// tile legible at a glance without a separate colored badge.
function activeLevelGradient(level: RiskLevelKey | null): string {
  if (level === null) return "linear-gradient(135deg, #10b981, #34d399)";  // CLEAR — green
  if (level === "L1") return "linear-gradient(135deg, #f59f00, #fbbf24)";
  if (level === "L2") return "linear-gradient(135deg, #f97316, #fb923c)";
  if (level === "L3") return "linear-gradient(135deg, #ea580c, #f97316)";
  return "linear-gradient(135deg, #dc2626, #ef4444)";                        // L4 — red
}

// Row-level accent per L-key, mirrors gradient hue at 100% opacity.
function levelColor(level: RiskLevelKey): string {
  if (level === "L1") return "#f59f00";
  if (level === "L2") return "#f97316";
  if (level === "L3") return "#ea580c";
  return "#dc2626";
}

// Cap-bucket border color for a level pill.
function statusChipStyle(status: string, key: RiskLevelKey): { background: string; color: string } {
  if (status === "FIRED") return { background: levelColor(key), color: "#fff" };
  if (status === "ARMED") return {
    background: `color-mix(in oklab, ${levelColor(key)} 18%, var(--surface))`,
    color: levelColor(key),
  };
  return {
    background: "color-mix(in oklab, #08a86b 12%, var(--surface))",
    color: "#16a34a",
  };
}

export function RiskManager({ navColor }: { navColor: string }) {
  const [history, setHistory] = useState<JournalHistoryPoint[]>([]);
  const [risk, setRisk] = useState<RiskLevelsResponse | null>(null);
  const [riskError, setRiskError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [chartRange, setChartRange] = useState<"3M" | "6M" | "YTD" | "1Y" | "All">("6M");
  const [heatRange, setHeatRange] = useState<"3M" | "6M" | "YTD" | "1Y" | "All">("6M");
  // Migration 062 — positions loaded here so the SR15 nudge banner can
  // render. Same fetch shape ACS uses; failure is silent (banner just
  // won't render).
  const [nudgePositions, setNudgePositions] = useState<EnrichedPosition[]>([]);

  useEffect(() => {
    const portfolio = getActivePortfolio();
    // Journal history + composed risk-levels response fire in parallel.
    // Risk-levels is the primary read for tile state; history feeds the
    // NLV line on the Hard Deck chart + the Heat Tape.
    Promise.all([
      api.journalHistory(portfolio, 0),
      api.riskLevels(portfolio),
    ]).then(([h, r]) => {
      setHistory(h as JournalHistoryPoint[]);
      if (r && "error" in r) {
        setRiskError(r.error);
      } else if (r) {
        setRisk(r as RiskLevelsResponse);
      }
      setLoading(false);
    }).catch((err) => {
      log.error("risk-manager", "fetch failed", err);
      setLoading(false);
    });
    // Fire-and-forget positions load for the SR15/SR12 nudge. Silent on
    // error — the banners just won't render.
    Promise.all([
      api.tradesOpen(portfolio).catch(() => []),
      api.tradesOpenDetails(portfolio).catch(() => ({ details: [], lot_closures: [] })),
    ]).then(async ([openTrades, detailsBundle]) => {
      const trades = openTrades as TradePosition[];
      if (trades.length === 0) return;
      const tickers = trades.map(t => t.ticker).filter(Boolean);
      let prices: Record<string, number> = {};
      try {
        const result = await api.batchPrices(tickers, portfolio);
        if (result && !("error" in result)) prices = result;
      } catch { /* fall back to entry price */ }
      const enriched = computeEnrichedPositions(
        trades, (detailsBundle as TradeDetailsBundle).details, 0, prices,
      );
      setNudgePositions(enriched);
    }).catch(err => {
      log.error("risk-manager", "SR15 nudge positions fetch failed", err);
    });
  }, []);

  // Chart data — NLV + HWM + cycle_reference ratchet + L1 threshold.
  // L2/L3/L4 are IXIC-structural and can't be plotted on the NLV axis,
  // so they surface as ACTIVE / CLEAR pills below the chart instead of
  // as reference lines. See the note at the bottom of the chart card.
  const chartData = useMemo(() => {
    if (history.length === 0) return [];

    let hwm = 0;
    const flipDate = risk?.cycle_reference?.flip_date ?? null;
    const initialRef = Number(risk?.cycle_reference?.initial_nlv ?? 0);
    // Recompute the cycle_reference ratchet PER DAY (not just today's
    // frozen value) so the chart line is a stepped cummax over the
    // portfolio's post-flip NLVs — matches how the DB row was built.
    let cycleRefRatchet = initialRef;
    const fullData = history.map(h => {
      if (h.end_nlv > hwm) hwm = h.end_nlv;
      const cashChange = parseFloat(String(h.cash_change || 0)) || 0;
      const inCycle = !!flipDate && String(h.day).slice(0, 10) >= flipDate;
      if (inCycle && h.end_nlv > cycleRefRatchet) cycleRefRatchet = h.end_nlv;
      return {
        day: h.day,
        nlv: h.end_nlv,
        hwm,
        cycleRef: inCycle ? cycleRefRatchet : null,
        l1Threshold: inCycle ? cycleRefRatchet * 0.925 : null,
        cashIn: cashChange > 0 ? h.end_nlv : null,
        cashOut: cashChange < 0 ? h.end_nlv : null,
      };
    });

    let filtered = fullData;
    const now = new Date();
    if (chartRange !== "All") {
      let cutoff: Date;
      if (chartRange === "3M") cutoff = new Date(now.getTime() - 90 * 86400000);
      else if (chartRange === "6M") cutoff = new Date(now.getTime() - 180 * 86400000);
      else if (chartRange === "YTD") cutoff = new Date(now.getFullYear(), 0, 1);
      else cutoff = new Date(now.getTime() - 365 * 86400000);
      const cutoffStr = cutoff.toISOString().slice(0, 10);
      filtered = fullData.filter(d => d.day >= cutoffStr);
    }
    return filtered;
  }, [history, chartRange, risk]);

  // Has any heat data ever (drives panel visibility)
  const hasHeatData = useMemo(
    () => history.some(h => (h.portfolio_heat || 0) > 0),
    [history]
  );

  // Heat Tape data (filtered by selected timeframe). Same math as before —
  // this panel is intentionally untouched by the L-series rewrite.
  const heatTapeData = useMemo(() => {
    if (history.length === 0) return [];
    const firstHeatIdx = history.findIndex(h => (h.portfolio_heat || 0) > 0);
    if (firstHeatIdx < 0) return [];
    let subset = history.slice(firstHeatIdx);
    if (heatRange !== "All") {
      const now = new Date();
      let cutoff: Date;
      if (heatRange === "3M") cutoff = new Date(now.getTime() - 90 * 86400000);
      else if (heatRange === "6M") cutoff = new Date(now.getTime() - 180 * 86400000);
      else if (heatRange === "YTD") cutoff = new Date(now.getFullYear(), 0, 1);
      else cutoff = new Date(now.getTime() - 365 * 86400000);
      const cutoffStr = cutoff.toISOString().slice(0, 10);
      subset = subset.filter(d => d.day >= cutoffStr);
    }
    if (subset.length < 2) return [];
    const startNlv = subset[0].end_nlv || 1;
    const startSpy = subset[0].spy || 1;
    return subset.map(h => ({
      day: h.day,
      portPct: parseFloat((((h.end_nlv || startNlv) / startNlv - 1) * 100).toFixed(2)),
      spyPct: startSpy > 0 ? parseFloat((((h.spy || startSpy) / startSpy - 1) * 100).toFixed(2)) : 0,
      heat: h.portfolio_heat || 0,
    }));
  }, [history, heatRange]);

  if (loading) {
    return <div className="animate-pulse"><div className="h-[90px] rounded-[14px]" style={{ background: "var(--bg-2)" }} /></div>;
  }

  const cycleRef = risk?.cycle_reference ?? null;
  const activeLevel = risk?.active_level ?? null;
  const effectiveCap = risk?.effective_cap_pct;
  const excessToSell = risk?.excess_dollars_to_sell ?? 0;
  const cycleDdPct = risk?.current_drawdown_from_cycle_pct ?? 0;
  const athHwm = risk?.ath_hwm ?? 0;
  const athDdPct = risk?.ath_drawdown_pct ?? 0;
  const currentNlv = risk?.current_nlv ?? 0;
  const currentExposure = risk?.current_exposure_pct ?? 0;
  const mFactorPct = risk?.m_factor_suggested_exposure_pct;

  // ── Header tile content ───────────────────────────────────────────
  const cycleRefValue = cycleRef
    ? formatCurrency(Number(cycleRef.ratcheted_nlv), { decimals: 0 })
    : "—";
  const cycleRefSub = cycleRef
    ? `Since ${cycleRef.flip_date}${cycleRef.is_frozen ? " · FROZEN" : ""}`
    : "No active cycle";
  const cycleRefExtra = cycleRef
    ? `DD: ${cycleDdPct.toFixed(2)}%`
    : undefined;

  const activeLevelValue = activeLevel ?? "CLEAR";
  const activeLevelSub = effectiveCap != null
    ? `Cap ${effectiveCap}% gross`
    : (mFactorPct != null ? `M Factor: ${mFactorPct.toFixed(0)}%` : "Trend cycle positive");
  const activeLevelExtra = excessToSell > 0
    ? `Sell ${formatCurrency(excessToSell, { decimals: 0 })}`
    : undefined;

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          Risk <em className="italic" style={{ color: navColor }}>Manager</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>Cycle-anchored L-series governor (L1 NLV · L2/L3/L4 IXIC structural)</div>
      </div>

      {riskError && (
        <div className="px-4 py-3 rounded-[10px] mb-4 text-[13px]"
             style={{
               background: "color-mix(in oklab, #e5484d 8%, var(--surface))",
               border: "1px solid var(--border)",
               color: "#e5484d",
             }}>
          Risk levels unavailable: {riskError}
        </div>
      )}

      {/* Migration 062 — SR15 nudge (informational mirror of the ACS
          banner). Chips non-clickable here; operator jumps back to ACS
          or the broker to act. */}
      <SR15NudgeBanner positions={nudgePositions} />

      {/* Migration 064 — SR12 Ratcheting Profit Floor nudge (same
          informational-mirror rule as SR15 above). */}
      <SR12FloorNudgeBanner positions={nudgePositions} />

      {/* Header tiles — cycle reference primary, active level, ATH
          demoted to secondary styling. */}
      <div className="grid grid-cols-3 gap-3.5 mb-6">
        <KPITile
          label="CYCLE REFERENCE"
          value={cycleRefValue}
          sub={cycleRefSub}
          extraSub={cycleRefExtra}
          gradient="linear-gradient(135deg, #6366f1, #818cf8)"
        />
        <KPITile
          label="ACTIVE LEVEL"
          value={activeLevelValue}
          sub={activeLevelSub}
          extraSub={activeLevelExtra}
          gradient={activeLevelGradient(activeLevel)}
        />
        {/* ATH HWM — muted grey tile per the "informational only, no
            action required" spec. Old ATH-anchored tiles reading
            "REMOVE MARGIN / GO TO CASH" are gone; this stays for
            historical context. */}
        <div className="relative overflow-hidden rounded-[14px] p-[14px_16px] flex flex-col justify-between h-[90px]"
             style={{
               background: "var(--surface)",
               border: "1px solid var(--border)",
               boxShadow: "var(--card-shadow)",
             }}>
          <div>
            <div className="text-[9px] font-semibold uppercase tracking-[0.10em]" style={{ color: "var(--ink-4)" }}>
              ALL-TIME HWM
            </div>
            <div className="text-[22px] font-semibold tracking-tight mt-0.5 privacy-mask"
                 style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink-2)" }}>
              {athHwm > 0 ? formatCurrency(athHwm, { decimals: 0 }) : "—"}
            </div>
          </div>
          <div className="text-[10px] font-medium privacy-mask" style={{ color: "var(--ink-4)" }}>
            {athHwm > 0 ? `${athDdPct.toFixed(2)}% from peak · informational` : "No journal data"}
          </div>
        </div>
      </div>

      {/* Charts: Hard Deck (NLV + cycle ref + L1) + Heat Tape side by side */}
      <div className="grid gap-4 mb-6" style={{ gridTemplateColumns: hasHeatData ? "1fr 1fr" : "1fr" }}>

        {/* Hard Deck Chart */}
        <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
          <div className="flex items-center justify-between px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
              <span className="text-[13px] font-semibold">The Hard Deck</span>
            </div>
            <div className="flex p-0.5 rounded-[8px] gap-0.5" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
              {(["3M", "6M", "YTD", "1Y", "All"] as const).map(t => (
                <button key={t} onClick={() => setChartRange(t)}
                        className="px-2 py-0.5 rounded text-[10px] font-medium transition-all"
                        style={{
                          background: chartRange === t ? "var(--surface)" : "transparent",
                          color: chartRange === t ? "var(--ink)" : "var(--ink-4)",
                        }}>
                  {t}
                </button>
              ))}
            </div>
          </div>
          <div style={{ height: 340 }} className="px-1 py-2">
            {chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData} margin={{ top: 8, right: 12, left: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.5} vertical={false} />
                  <XAxis dataKey="day" tick={{ fontSize: 9, fill: "var(--ink-4)" }} tickLine={false} axisLine={{ stroke: "var(--border)" }}
                         interval={Math.max(Math.floor(chartData.length / 6), 1)}
                         tickFormatter={(v: string) => new Date(v).toLocaleDateString("en-US", { month: "short", year: "2-digit" })} />
                  <YAxis tick={{ fontSize: 9, fill: "var(--ink-4)" }} tickLine={false} axisLine={false}
                         tickFormatter={(v: number) => formatCurrency(v, { compact: true, decimals: 0 })} width={48}
                         domain={[(dm: number) => dm * 0.97, (dm: number) => dm * 1.02]} />
                  <Tooltip
                    contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 10, fontFamily: "var(--font-jetbrains), monospace" }}
                    formatter={(value: unknown, name: unknown) => {
                      if (value == null) return [null, null];
                      const labels: Record<string, string> = {
                        nlv: "NLV",
                        hwm: "ATH Peak",
                        cycleRef: "Cycle Ref",
                        l1Threshold: "L1 Threshold",
                        cashIn: "Cash In",
                        cashOut: "Cash Out",
                      };
                      return [formatCurrency(Number(value), { decimals: 0 }), labels[String(name)] || String(name)];
                    }}
                    labelFormatter={(l: unknown) => new Date(String(l)).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
                  />
                  {/* ATH peak — muted context line. No longer a rule input. */}
                  <Line dataKey="hwm" stroke="var(--ink-4)" strokeWidth={1} strokeDasharray="4 3" dot={false} type="stepAfter" />
                  {/* Cycle reference ratchet — solid green. Steps up on new
                      NLV highs inside the positive trend cycle. */}
                  <Line dataKey="cycleRef" stroke="#10b981" strokeWidth={1.5} dot={false} type="stepAfter" />
                  {/* L1 threshold — dashed yellow. Reference × 0.925.
                      Crossing this from above fires L1. */}
                  <Line dataKey="l1Threshold" stroke="#f59f00" strokeWidth={1.5} strokeDasharray="5 3" dot={false} type="stepAfter" />
                  {/* NLV — primary indigo line, unchanged. */}
                  <Line dataKey="nlv" stroke="#6366f1" strokeWidth={2.5} type="monotone"
                        dot={(props: { cx?: number; cy?: number; payload?: { cashIn?: number | null; cashOut?: number | null } }) => {
                          const { cx = 0, cy = 0, payload } = props;
                          if (payload?.cashIn != null) {
                            return <polygon key={`ci-${cx}`} points={`${cx},${cy - 8} ${cx - 6},${cy + 4} ${cx + 6},${cy + 4}`} fill="#16a34a" stroke="var(--surface)" strokeWidth={1} />;
                          }
                          if (payload?.cashOut != null) {
                            return <polygon key={`co-${cx}`} points={`${cx - 6},${cy - 4} ${cx + 6},${cy - 4} ${cx},${cy + 8}`} fill="#dc2626" stroke="var(--surface)" strokeWidth={1} />;
                          }
                          return <circle key={`empty-${cx}`} r={0} />;
                        }}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-sm" style={{ color: "var(--ink-4)" }}>No data</div>
            )}
          </div>
          <div className="px-[18px] pb-2 text-[10px]" style={{ color: "var(--ink-4)" }}>
            NLV vs cycle reference (green step) + L1 threshold (dashed
            yellow). L2/L3/L4 are IXIC structural — see level rows below.
          </div>
        </div>

        {/* Heat Tape — unchanged. */}
        {hasHeatData && (
          <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
            <div className="flex items-center justify-between px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
                <span className="text-[13px] font-semibold">Heat Tape</span>
              </div>
              <div className="flex p-0.5 rounded-[8px] gap-0.5" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
                {(["3M", "6M", "YTD", "1Y", "All"] as const).map(t => (
                  <button key={t} onClick={() => setHeatRange(t)}
                          className="px-2 py-0.5 rounded text-[10px] font-medium transition-all"
                          style={{
                            background: heatRange === t ? "var(--surface)" : "transparent",
                            color: heatRange === t ? "var(--ink)" : "var(--ink-4)",
                          }}>
                    {t}
                  </button>
                ))}
              </div>
            </div>
            <div className="flex items-center gap-4 px-[18px] pt-2 pb-1" style={{ fontSize: 10 }}>
              {[
                { label: "Portfolio", color: "#1a1d29", width: 2.2 },
                { label: "SPY", color: "#8b7bc0", width: 1.5 },
                { label: "Heat %", color: "#4A90E2", fill: true },
              ].map(item => (
                <div key={item.label} className="flex items-center gap-1.5" style={{ color: "var(--ink-3)" }}>
                  {item.fill ? (
                    <svg width="14" height="8"><rect x="0" y="1" width="14" height="6" fill={item.color} opacity={0.3} rx="1" /><line x1="0" y1="4" x2="14" y2="4" stroke={item.color} strokeWidth={1.5} /></svg>
                  ) : (
                    <svg width="14" height="8"><line x1="0" y1="4" x2="14" y2="4" stroke={item.color} strokeWidth={item.width} /></svg>
                  )}
                  <span>{item.label}</span>
                </div>
              ))}
            </div>
            <div style={{ height: 320 }} className="px-1 py-1">
              {heatTapeData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={heatTapeData} margin={{ top: 8, right: 12, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.5} vertical={false} />
                  <XAxis dataKey="day" tick={{ fontSize: 9, fill: "var(--ink-4)" }} tickLine={false} axisLine={{ stroke: "var(--border)" }}
                         interval={Math.max(Math.floor(heatTapeData.length / 6), 1)}
                         tickFormatter={(v: string) => new Date(v).toLocaleDateString("en-US", { month: "short", year: "2-digit" })} />
                  <YAxis yAxisId="left" tick={{ fontSize: 9, fill: "var(--ink-4)" }} tickLine={false} axisLine={false}
                         tickFormatter={(v: number) => `${v}%`} width={40} />
                  <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 9, fill: "var(--ink-4)" }} tickLine={false} axisLine={false}
                         tickFormatter={(v: number) => `${v}%`} width={40} domain={[0, 60]} />
                  <Tooltip
                    contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 10, fontFamily: "var(--font-jetbrains), monospace" }}
                    formatter={(value: unknown, name: unknown) => {
                      if (value == null) return [null, null];
                      const labels: Record<string, string> = { portPct: "Portfolio", spyPct: "SPY", heat: "Heat" };
                      return [`${Number(value).toFixed(2)}%`, labels[String(name)] || String(name)];
                    }}
                    labelFormatter={(l: unknown) => new Date(String(l)).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
                  />
                  <ReferenceLine yAxisId="left" y={0} stroke="var(--ink-4)" strokeDasharray="3 3" strokeOpacity={0.3} />
                  <ReferenceLine yAxisId="right" y={20} stroke="#dc2626" strokeDasharray="4 3" strokeOpacity={0.5}
                                 label={{ value: "20% Threshold", position: "right", fontSize: 8, fill: "#dc2626" }} />
                  <ReferenceLine yAxisId="right" y={7.5} stroke="#f59f00" strokeDasharray="4 3" strokeOpacity={0.5}
                                 label={{ value: "7.5%", position: "right", fontSize: 8, fill: "#f59f00" }} />
                  <Area yAxisId="right" dataKey="heat" fill="rgba(74,144,226,0.15)" stroke="#4A90E2" strokeWidth={1.5} type="monotone" dot={false} />
                  <Line yAxisId="left" dataKey="spyPct" stroke="#8b7bc0" strokeWidth={1.5} dot={false} type="monotone" />
                  <Line yAxisId="left" dataKey="portPct" stroke="#6366f1" strokeWidth={2.2} dot={false} type="monotone" />
                </ComposedChart>
              </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center text-sm" style={{ color: "var(--ink-4)" }}>No heat data in this range</div>
              )}
            </div>
            <div className="px-[18px] pb-2 text-[10px]" style={{ color: "var(--ink-4)" }}>
              {heatTapeData.length} days · threshold editable in Admin
            </div>
          </div>
        )}
      </div>

      {/* L-series Levels */}
      <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
        <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
          <span className="text-[13px] font-semibold">Exposure Levels</span>
          <span className="text-[11px] ml-2" style={{ color: "var(--ink-4)" }}>
            Current gross exposure: {currentExposure.toFixed(1)}%
            {" · "}
            NLV: {formatCurrency(currentNlv, { decimals: 0 })}
          </span>
        </div>
        <div className="p-5 flex flex-col gap-3">
          {(risk?.levels_state ?? []).map(level => {
            const fired = level.status === "FIRED";
            const armed = level.status === "ARMED";
            const color = levelColor(level.key);
            const isActive = level.key === activeLevel;
            return (
              <div key={level.key} className="flex items-center gap-4 p-3 rounded-[10px]"
                   style={{
                     background: fired || isActive
                       ? `color-mix(in oklab, ${color} 10%, var(--surface))`
                       : (armed ? `color-mix(in oklab, ${color} 4%, var(--surface))` : "var(--bg)"),
                     border: `1px solid ${fired || isActive ? color : "var(--border)"}`,
                   }}>
                <div className="w-2 h-2 rounded-full shrink-0" style={{ background: fired || armed ? color : "var(--ink-4)" }} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-[13px] font-semibold" style={{ color: fired ? color : "var(--ink)" }}>
                      {level.key}
                    </span>
                    <span className="text-[12px]" style={{ color: "var(--ink-3)" }}>{level.action}</span>
                    <span className="text-[11px] px-2 py-0.5 rounded-full"
                          style={{
                            background: "color-mix(in oklab, var(--ink) 6%, var(--surface))",
                            color: "var(--ink-3)",
                            fontFamily: "var(--font-jetbrains), monospace",
                          }}>
                      cap {level.cap_pct}%
                    </span>
                    {level.key === "L1" && level.threshold_nlv != null && (
                      <span className="text-[11px] privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink-4)" }}>
                        threshold {formatCurrency(Number(level.threshold_nlv), { decimals: 0 })}
                      </span>
                    )}
                  </div>
                  <div className="text-[11px] mt-0.5" style={{ color: "var(--ink-4)" }}>
                    <span style={{ color: "var(--ink-3)" }}>{level.trigger}</span>
                    <span className="mx-2">·</span>
                    {level.detail}
                  </div>
                </div>
                <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full"
                      style={statusChipStyle(level.status, level.key)}>
                  {level.status}
                </span>
              </div>
            );
          })}
        </div>
        {activeLevel && excessToSell > 0 && (
          <div className="px-5 py-3 text-[12px]"
               style={{
                 borderTop: "1px solid var(--border)",
                 background: "color-mix(in oklab, #e5484d 4%, var(--surface))",
                 color: "var(--ink-2)",
               }}>
            <strong>Action:</strong> {activeLevel} caps gross exposure at {effectiveCap}%.
            Sell approximately <span style={{ fontFamily: "var(--font-jetbrains), monospace", fontWeight: 600 }}>
              {formatCurrency(excessToSell, { decimals: 0 })}
            </span> to bring exposure inside the cap.
            <span className="text-[10px] ml-2" style={{ color: "var(--ink-4)" }}>
              (Reduction order — leveraged instruments first, then lowest-conviction — is pending spec.)
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
