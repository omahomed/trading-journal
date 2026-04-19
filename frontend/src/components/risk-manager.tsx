"use client";

import { useState, useEffect, useMemo } from "react";
import { api, type JournalHistoryPoint } from "@/lib/api";
import {
  ResponsiveContainer, ComposedChart, Line, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, ReferenceLine,
} from "recharts";

function KPITile({ label, value, sub, gradient }: { label: string; value: string; sub: string; gradient: string }) {
  return (
    <div className="relative overflow-hidden rounded-[14px] p-[14px_16px] text-white flex flex-col justify-between h-[90px] transition-transform duration-150 hover:scale-[1.01]"
         style={{ background: gradient, boxShadow: "var(--kpi-shadow)" }}>
      <div className="absolute -right-5 -top-5 w-[100px] h-[100px] rounded-full" style={{ background: "radial-gradient(circle, rgba(255,255,255,0.18), transparent 65%)" }} />
      <div className="relative z-10">
        <div className="text-[9px] font-semibold uppercase tracking-[0.10em] opacity-85">{label}</div>
        <div className="text-[22px] font-semibold tracking-tight mt-0.5 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{value}</div>
      </div>
      <div className="relative z-10 text-[10px] font-medium opacity-80 privacy-mask">{sub}</div>
    </div>
  );
}

const HARD_DECKS = [
  { key: "L1", pct: 7.5, action: "Remove margin", color: "#f59f00" },
  { key: "L2", pct: 12.5, action: "Max 30% invested", color: "#f97316" },
  { key: "L3", pct: 15.0, action: "Go to cash", color: "#dc2626" },
];

export function RiskManager({ navColor }: { navColor: string }) {
  const [history, setHistory] = useState<JournalHistoryPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [chartRange, setChartRange] = useState<"3M" | "6M" | "YTD" | "1Y" | "All">("6M");

  useEffect(() => {
    api.journalHistory("CanSlim", 0).then(h => {
      setHistory(h as JournalHistoryPoint[]);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  const metrics = useMemo(() => {
    if (history.length === 0) return null;
    const nlvs = history.map(h => h.end_nlv);
    const peak = Math.max(...nlvs);
    const current = nlvs[nlvs.length - 1] || 0;
    const ddDol = peak - current;
    const ddPct = peak > 0 ? (ddDol / peak) * 100 : 0;

    // Decks based on current peak
    const deckL1 = peak * (1 - HARD_DECKS[0].pct / 100);
    const deckL2 = peak * (1 - HARD_DECKS[1].pct / 100);
    const deckL3 = peak * (1 - HARD_DECKS[2].pct / 100);
    const distL1 = current - deckL1;

    // Status
    let status = "ALL CLEAR";
    let statusGradient = "linear-gradient(135deg, #10b981, #34d399)";
    if (ddPct >= HARD_DECKS[2].pct) {
      status = HARD_DECKS[2].action.toUpperCase();
      statusGradient = "linear-gradient(135deg, #dc2626, #ef4444)";
    } else if (ddPct >= HARD_DECKS[1].pct) {
      status = HARD_DECKS[1].action.toUpperCase();
      statusGradient = "linear-gradient(135deg, #f97316, #fb923c)";
    } else if (ddPct >= HARD_DECKS[0].pct) {
      status = HARD_DECKS[0].action.toUpperCase();
      statusGradient = "linear-gradient(135deg, #f59f00, #fbbf24)";
    }

    // Max drawdown (all time)
    let maxDD = 0;
    let runPeak = 0;
    for (const h of history) {
      if (h.end_nlv > runPeak) runPeak = h.end_nlv;
      const dd = runPeak > 0 ? ((h.end_nlv - runPeak) / runPeak) * 100 : 0;
      if (dd < maxDD) maxDD = dd;
    }

    // Drawdown tile color: green < L1, yellow L1-L2, red > L2
    let ddGradient = "linear-gradient(135deg, #10b981, #34d399)"; // green
    if (ddPct >= HARD_DECKS[1].pct) {
      ddGradient = "linear-gradient(135deg, #dc2626, #ef4444)"; // red
    } else if (ddPct >= HARD_DECKS[0].pct) {
      ddGradient = "linear-gradient(135deg, #f59f00, #fbbf24)"; // yellow
    }

    return { peak, current, ddDol, ddPct, deckL1, deckL2, deckL3, distL1, status, statusGradient, ddGradient, maxDD };
  }, [history]);

  // Chart data — NLV + HWM + cash flow markers
  const chartData = useMemo(() => {
    if (history.length === 0) return [];

    let hwm = 0;
    const fullData = history.map(h => {
      if (h.end_nlv > hwm) hwm = h.end_nlv;
      const cashChange = parseFloat(String(h.cash_change || 0)) || 0;
      return { day: h.day, nlv: h.end_nlv, hwm, cashIn: cashChange > 0 ? h.end_nlv : null, cashOut: cashChange < 0 ? h.end_nlv : null };
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
  }, [history, chartRange]);

  // Heat Tape data
  const heatTapeData = useMemo(() => {
    if (history.length === 0) return [];
    // Find first day with heat > 0
    const firstHeatIdx = history.findIndex(h => (h.portfolio_heat || 0) > 0);
    if (firstHeatIdx < 0) return [];
    const subset = history.slice(firstHeatIdx);
    if (subset.length < 2) return [];
    const startNlv = subset[0].end_nlv || 1;
    const startSpy = subset[0].spy || 1;
    return subset.map(h => ({
      day: h.day,
      portPct: parseFloat((((h.end_nlv || startNlv) / startNlv - 1) * 100).toFixed(2)),
      spyPct: startSpy > 0 ? parseFloat((((h.spy || startSpy) / startSpy - 1) * 100).toFixed(2)) : 0,
      heat: h.portfolio_heat || 0,
    }));
  }, [history]);

  if (loading) {
    return <div className="animate-pulse"><div className="h-[90px] rounded-[14px]" style={{ background: "var(--bg-2)" }} /></div>;
  }
  if (!metrics) {
    return <div className="text-center py-16" style={{ color: "var(--ink-4)" }}>No journal data available</div>;
  }

  const m = metrics;

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          Risk <em className="italic" style={{ color: navColor }}>Manager</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>Drawdown tracking with hard deck enforcement</div>
      </div>

      {/* KPI tiles */}
      <div className="grid grid-cols-3 gap-3.5 mb-6">
        <KPITile label="CURRENT PEAK (HWM)" value={`$${m.peak.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} sub="All-Time High Water Mark" gradient="linear-gradient(135deg, #6366f1, #818cf8)" />
        <KPITile label="CURRENT DRAWDOWN" value={`-${m.ddPct.toFixed(2)}%`} sub={`-$${m.ddDol.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} gradient={m.ddGradient} />
        <KPITile label="REQUIRED ACTION" value={m.status} sub={m.ddPct < HARD_DECKS[0].pct ? `Buffer: $${m.distL1.toLocaleString(undefined, { maximumFractionDigits: 0 })} to L1` : ""} gradient={m.statusGradient} />
      </div>

      {/* Charts: Hard Deck + Heat Tape side by side */}
      <div className="grid gap-4 mb-6" style={{ gridTemplateColumns: heatTapeData.length > 0 ? "1fr 1fr" : "1fr" }}>

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
                         tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`} width={48}
                         domain={[(dm: number) => Math.min(dm, m.deckL3) * 0.97, (dm: number) => dm * 1.02]} />
                  <Tooltip
                    contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 10, fontFamily: "var(--font-jetbrains), monospace" }}
                    formatter={(value: any, name: any) => {
                      if (value == null) return [null, null];
                      const labels: Record<string, string> = { nlv: "NLV", hwm: "Peak", cashIn: "Cash In", cashOut: "Cash Out" };
                      return [`$${Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 })}`, labels[String(name)] || String(name)];
                    }}
                    labelFormatter={(l: any) => new Date(String(l)).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
                  />
                  {HARD_DECKS.map(d => (
                    <ReferenceLine key={d.key} y={m.peak * (1 - d.pct / 100)} stroke={d.color} strokeWidth={1.5}
                                   label={{ value: d.key, position: "insideTopLeft", fontSize: 11, fontWeight: 700, fill: d.color }} />
                  ))}
                  <Line dataKey="hwm" stroke="var(--ink-4)" strokeWidth={1} strokeDasharray="4 3" dot={false} type="stepAfter" />
                  <Line dataKey="nlv" stroke="#6366f1" strokeWidth={2.5} type="monotone"
                        dot={(props: any) => {
                          const { cx, cy, payload } = props;
                          if (payload.cashIn != null) {
                            return <polygon key={`ci-${cx}`} points={`${cx},${cy - 8} ${cx - 6},${cy + 4} ${cx + 6},${cy + 4}`} fill="#16a34a" stroke="var(--surface)" strokeWidth={1} />;
                          }
                          if (payload.cashOut != null) {
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
        </div>

        {/* Heat Tape */}
        {heatTapeData.length > 0 && (
          <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
            <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
              <span className="text-[13px] font-semibold">Heat Tape</span>
            </div>
            {/* Legend */}
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
                    formatter={(value: any, name: any) => {
                      if (value == null) return [null, null];
                      const labels: Record<string, string> = { portPct: "Portfolio", spyPct: "SPY", heat: "Heat" };
                      return [`${Number(value).toFixed(2)}%`, labels[String(name)] || String(name)];
                    }}
                    labelFormatter={(l: any) => new Date(String(l)).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
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
            </div>
            <div className="px-[18px] pb-2 text-[10px]" style={{ color: "var(--ink-4)" }}>
              {heatTapeData.length} days · threshold editable in Admin
            </div>
          </div>
        )}
      </div>

      {/* Hard Deck Levels */}
      <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
        <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
          <span className="text-[13px] font-semibold">Hard Deck Levels</span>
        </div>
        <div className="p-5 flex flex-col gap-3">
          {HARD_DECKS.map(deck => {
            const deckVal = m.peak * (1 - deck.pct / 100);
            const breached = m.ddPct >= deck.pct;
            const distance = deckVal - m.current;
            return (
              <div key={deck.key} className="flex items-center gap-4 p-3 rounded-[10px]"
                   style={{ background: breached ? `${deck.color}10` : "var(--bg)", border: `1px solid ${breached ? deck.color : "var(--border)"}` }}>
                <div className="w-2 h-2 rounded-full shrink-0" style={{ background: breached ? deck.color : "var(--ink-4)" }} />
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-[13px] font-semibold" style={{ color: breached ? deck.color : "var(--ink)" }}>-{deck.pct}%</span>
                    <span className="text-[12px]" style={{ color: "var(--ink-3)" }}>{deck.action}</span>
                    <span className="text-[11px] privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink-4)" }}>${deckVal.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
                  </div>
                  <div className="text-[11px] mt-0.5" style={{ color: "var(--ink-4)" }}>
                    {breached ? "ACTIVE — Action required" : `$${Math.abs(distance).toLocaleString(undefined, { maximumFractionDigits: 0 })} buffer`}
                  </div>
                </div>
                <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full"
                      style={{ background: breached ? deck.color : "color-mix(in oklab, #08a86b 12%, var(--surface))", color: breached ? "#fff" : "#16a34a" }}>
                  {breached ? "BREACHED" : "CLEAR"}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
