"use client";

import { useState, useEffect, useMemo } from "react";
import { api } from "@/lib/api";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ReferenceLine, Legend,
} from "recharts";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Persona benchmarks (% gain from FTD close, Day 1-25)
const LIFE_CHANGER = [0.8,1.1,1.5,2.1,2.7,2.9,3.1,4.0,4.4,4.8,5.2,5.5,5.9,6.3,6.7,7.2,7.5,7.8,8.2,8.5,8.7,8.8,9.0,9.2,9.9];
const MONEY_MAKER = [0.5,0.8,1.2,1.5,1.7,2.0,2.2,2.5,2.7,3.0,3.3,3.6,3.9,4.2,4.4,4.7,5.0,5.3,5.6,5.8,6.0,6.1,6.2,6.3,6.4];
const SLOG = [0.3,0.4,0.5,0.7,0.8,0.9,1.0,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.6,1.7,1.9,2.0,2.0,2.0,2.0,2.1,2.1,2.2,2.2];
const WHIPSAW = [0.1,-0.1,-0.3,-0.5,-0.6,-0.8,-0.9,-1.1,-1.3,-1.5,-1.9,-2.2,-2.6,-3.0,-3.4];
const RALLY_2025 = [2.5,5.3,6.6,6.5,7.1,7.0,8.7,10.3,9.5,8.5,8.8,10.0,10.0,14.8,16.6,17.5,17.3,17.9,17.9,17.4,15.8,16.1,14.9,17.8,17.2];

function nearestPersona(pct: number, dayIdx: number): string {
  if (dayIdx < 1 || dayIdx > 25) return "—";
  const i = dayIdx - 1;
  const candidates: [string, number][] = [
    ["Life Changer", LIFE_CHANGER[i]], ["Money Maker", MONEY_MAKER[i]], ["SLOG", SLOG[i]],
  ];
  if (i < WHIPSAW.length) candidates.push(["Whipsaw", WHIPSAW[i]]);
  return candidates.reduce((a, b) => Math.abs(pct - a[1]) < Math.abs(pct - b[1]) ? a : b)[0];
}

export function RallyContext({ navColor }: { navColor: string }) {
  const [rallyData, setRallyData] = useState<any>(null);
  const [rallyPoints, setRallyPoints] = useState<any[]>([]);
  const [day0Close, setDay0Close] = useState(0);
  const [ftdDate, setFtdDate] = useState("");
  const [rallyLow, setRallyLow] = useState(0);
  const [loading, setLoading] = useState(true);
  const [showBenchmarks, setShowBenchmarks] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    // Get FTD from cycle state, then fetch rally data
    api.rallyPrefix().then(async (cycle: any) => {
      const ftd = cycle?.ftd_date;
      if (!ftd) { setError("No active FTD found"); setLoading(false); return; }
      setFtdDate(ftd);
      setRallyLow(cycle?.rally_low || 0);

      // Fetch rally price data from yfinance via API
      try {
        const res = await fetch(`${API_BASE}/api/market/rally-data?ftd_date=${ftd}&index=^IXIC`);
        const data = await res.json();
        if (data.error) { setError(data.error); }
        else {
          setDay0Close(data.day0_close || 0);
          setRallyPoints(data.points || []);
          setRallyData(data);
        }
      } catch (e) { setError("Failed to fetch rally data"); }
      setLoading(false);
    }).catch(() => { setError("Failed to load cycle state"); setLoading(false); });
  }, []);

  // Chart data: merge personas + current rally
  const chartData = useMemo(() => {
    const days = Array.from({ length: 25 }, (_, i) => i + 1);
    return days.map(d => {
      const pt = rallyPoints.find((p: any) => p.day === d);
      return {
        day: d,
        lifeChanger: LIFE_CHANGER[d - 1],
        moneyMaker: MONEY_MAKER[d - 1],
        slog: SLOG[d - 1],
        whipsaw: d <= 15 ? WHIPSAW[d - 1] : null,
        rally2025: RALLY_2025[d - 1],
        current: pt ? pt.pct : null,
      };
    });
  }, [rallyPoints]);

  const currentDay = rallyPoints.length;
  const currentPct = currentDay > 0 ? rallyPoints[currentDay - 1]?.pct || 0 : 0;
  const tracking = currentDay > 0 ? nearestPersona(currentPct, currentDay) : "—";

  // Rally failure check
  const rallyFailed = rallyLow > 0 && rallyPoints.some((p: any) => p.low < rallyLow);
  const failDay = rallyFailed ? rallyPoints.find((p: any) => p.low < rallyLow)?.day : null;

  if (loading) return <div className="animate-pulse"><div className="h-[90px] rounded-[14px]" style={{ background: "var(--bg-2)" }} /></div>;

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          Rally <em className="italic" style={{ color: navColor }}>Context</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>FTD Persona Overlay · First 25 Trading Days</div>
      </div>

      {error && <div className="mb-5 px-4 py-3 rounded-[10px] text-[12px]" style={{ background: "color-mix(in oklab, #f59f00 10%, var(--surface))", color: "#d97706", border: "1px solid color-mix(in oklab, #f59f00 30%, var(--border))" }}>{error}</div>}

      {rallyFailed && (
        <div className="mb-5 px-4 py-3 rounded-[10px] text-[13px] font-medium" style={{ background: "color-mix(in oklab, #e5484d 10%, var(--surface))", color: "#dc2626", border: "1px solid color-mix(in oklab, #e5484d 30%, var(--border))" }}>
          Rally Failed — index low undercut rally low (${rallyLow.toLocaleString()}) on Day {failDay}. This rally attempt is void.
        </div>
      )}

      {/* Two columns: Chart left, Metrics + Breakdown right */}
      <div className="grid gap-5 mb-5" style={{ gridTemplateColumns: "1fr 1fr", alignItems: "start" }}>

      {/* LEFT: Chart */}
      <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
        <div className="px-5 pt-4 pb-1 text-center">
          <div className="text-[14px] font-bold">{ftdDate} MMTS Day Rally — First 25 Trading Days</div>
          <div className="text-[10px]" style={{ color: "var(--ink-4)" }}>Trading Days Post Green</div>
        </div>
        <div style={{ height: 450 }} className="px-2 py-2">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.5} vertical={false} />
              <XAxis dataKey="day" tick={{ fontSize: 10, fill: "var(--ink-4)" }} tickLine={false}
                     label={{ value: "Day Number - Post MMTS Green", position: "insideBottom", offset: -5, fontSize: 11, fill: "var(--ink-4)" }} />
              <YAxis tick={{ fontSize: 10, fill: "var(--ink-4)" }} tickLine={false} axisLine={false}
                     tickFormatter={(v: number) => `${v.toFixed(1)}%`} width={45}
                     domain={['auto', 'auto']} />
              <Tooltip contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 11, fontFamily: "var(--font-jetbrains), monospace" }}
                       formatter={(v: any, name: any) => {
                         if (v == null) return [null, null];
                         const labels: Record<string, string> = { current: "Current Rally", lifeChanger: "Life Changer", moneyMaker: "Money Maker", slog: "SLOG", whipsaw: "Whipsaw", rally2025: "04/22/2025 Green" };
                         return [`${Number(v).toFixed(2)}%`, labels[String(name)] || name];
                       }} />
              <Legend verticalAlign="bottom" align="center" iconType="line" wrapperStyle={{ fontSize: 10, paddingTop: 12 }}
                      formatter={(value: string) => {
                        const labels: Record<string, string> = {
                          lifeChanger: "Life Changers", moneyMaker: "Money Makers",
                          slog: "Small Losses or Gains (SLOGs)", whipsaw: "Whipsaws",
                          rally2025: "04/22/2025 Green",
                          current: `${ftdDate} Green`,
                        };
                        return labels[value] || value;
                      }} />
              <ReferenceLine y={0} stroke="#000" strokeOpacity={0.3} strokeWidth={1} />

              {showBenchmarks && (
                <>
                  <Line dataKey="lifeChanger" stroke="#08a86b" strokeWidth={2} dot={false} type="monotone" connectNulls name="lifeChanger" />
                  <Line dataKey="moneyMaker" stroke="#1a1d29" strokeWidth={2} dot={false} type="monotone" connectNulls name="moneyMaker" />
                  <Line dataKey="slog" stroke="#e67e22" strokeWidth={1.8} dot={false} type="monotone" connectNulls name="slog" />
                  <Line dataKey="whipsaw" stroke="#3b82f6" strokeWidth={1.8} dot={false} type="monotone" connectNulls name="whipsaw" />
                </>
              )}
              <Line dataKey="rally2025" stroke="#dc2626" strokeWidth={2} strokeDasharray="5 3" dot={false} type="monotone" connectNulls name="rally2025" />
              <Line dataKey="current" stroke={rallyFailed ? "#9ca3af" : "#d946ef"} strokeWidth={3}
                    dot={{ r: 3, fill: rallyFailed ? "#9ca3af" : "#d946ef" }} type="monotone" connectNulls name="current"
                    label={{ position: "top", fontSize: 9, fill: rallyFailed ? "#9ca3af" : "#d946ef",
                             formatter: (v: any) => v != null ? `${Number(v).toFixed(1)}%` : "" }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* RIGHT: Metrics + Daily Breakdown */}
      <div className="flex flex-col gap-4">
        {/* Status metrics */}
        <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
          <div className="p-4 flex flex-col gap-3">
            {[
              { k: "Rally Day", v: `Day ${currentDay} of 25` },
              { k: "Current Gain", v: `${currentPct >= 0 ? "+" : ""}${currentPct.toFixed(2)}%`, color: currentPct >= 0 ? "#08a86b" : "#e5484d", sub: `from FTD close $${day0Close.toLocaleString(undefined, { minimumFractionDigits: 2 })}` },
              { k: "Tracking Toward", v: tracking, color: tracking === "Life Changer" ? "#08a86b" : tracking === "Whipsaw" ? "#e5484d" : "var(--ink)" },
              { k: "FTD Date", v: ftdDate },
            ].map(s => (
              <div key={s.k} className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>{s.k}</div>
                <div className="text-[18px] font-semibold mt-0.5" style={{ fontFamily: "var(--font-jetbrains), monospace", color: (s as any).color || "var(--ink)" }}>{s.v}</div>
                {(s as any).sub && <div className="text-[10px] mt-0.5" style={{ color: "var(--ink-4)" }}>{(s as any).sub}</div>}
              </div>
            ))}
          </div>
        </div>

        {/* Benchmark toggle */}
        <label className="flex items-center gap-2 cursor-pointer text-[12px]">
          <input type="checkbox" checked={showBenchmarks} onChange={e => setShowBenchmarks(e.target.checked)} className="rounded" />
          <span className="font-medium">Show persona bands</span>
        </label>

        {/* Daily Breakdown */}
        {rallyPoints.length > 0 && (
          <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
            <div className="px-4 py-2.5 text-[12px] font-semibold" style={{ borderBottom: "1px solid var(--border)" }}>
              Daily Breakdown — {rallyPoints.length} days
            </div>
            <div className="overflow-y-auto" style={{ maxHeight: 300 }}>
              <table className="w-full text-[11px]" style={{ borderCollapse: "collapse" }}>
                <thead>
                  <tr>
                    {["Day", "Close", "% FTD"].map(h => (
                      <th key={h} className="text-left px-3 py-1.5 text-[9px] uppercase tracking-[0.06em] font-semibold sticky top-0"
                          style={{ color: "var(--ink-4)", background: "var(--surface-2)", borderBottom: "1px solid var(--border)" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rallyPoints.map((p: any, i: number) => (
                    <tr key={i} style={{ borderBottom: "1px solid var(--border)" }}>
                      <td className="px-3 py-1.5" style={{ fontFamily: "var(--font-jetbrains), monospace", fontSize: 10 }}>{p.day}</td>
                      <td className="px-3 py-1.5 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace", fontSize: 10 }}>${p.close.toFixed(2)}</td>
                      <td className="px-3 py-1.5 font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace", fontSize: 10, color: p.pct >= 0 ? "#08a86b" : "#e5484d" }}>
                        {p.pct >= 0 ? "+" : ""}{p.pct.toFixed(2)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      </div> {/* end grid */}
    </div>
  );
}
