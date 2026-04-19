"use client";

import { useState, useEffect } from "react";
import { api } from "@/lib/api";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip,
} from "recharts";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// IBD count-based pyramid: count → allocation %
const ALLOC_MAP: Record<number, number> = { 0: 0, 1: 30, 2: 55, 3: 75, 4: 90, 5: 100, 6: 100 };

function KPITile({ label, value, sub, gradient }: { label: string; value: string; sub?: string; gradient: string }) {
  return (
    <div className="relative overflow-hidden rounded-[14px] p-[12px_14px] text-white flex flex-col justify-between h-[80px]"
         style={{ background: gradient }}>
      <div className="text-[9px] font-semibold uppercase tracking-[0.10em] opacity-85">{label}</div>
      <div className="text-[20px] font-semibold tracking-tight" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{value}</div>
      {sub && <div className="text-[10px] font-medium opacity-80">{sub}</div>}
    </div>
  );
}

export function IBDMarketSchool({ navColor }: { navColor: string }) {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [histTab, setHistTab] = useState<"exposure" | "signals">("exposure");

  const loadData = () => {
    setLoading(true);
    fetch(`${API_BASE}/api/market/ibd`).then(r => r.json())
      .then(d => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  };

  useEffect(() => { loadData(); }, []);

  if (loading) return <div className="animate-pulse"><div className="h-[90px] rounded-[14px]" style={{ background: "var(--bg-2)" }} /></div>;
  if (!data || data.error) return <div className="text-center py-16" style={{ color: "var(--ink-4)" }}>{data?.error || "Unable to load IBD data."}</div>;

  const expLevel = data.market_exposure || 0;
  const allocPct = ALLOC_MAP[Math.min(expLevel, 6)] || 0;
  const bsOn = data.buy_switch;
  const ddCount = data.distribution_count || 0;
  const hasBuySigs = data.buy_signals?.length > 0;
  const hasSellSigs = data.sell_signals?.length > 0;

  // Historical data for exposure chart
  const histData = (data.history || []).map((h: any) => ({
    date: h.date,
    exposure: h.market_exposure,
  }));

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px] flex items-center justify-between" style={{ borderBottom: "1px solid var(--border)" }}>
        <div>
          <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
            IBD Market <em className="italic" style={{ color: navColor }}>School</em>
          </h1>
          <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
            Market timing signals · NASDAQ + SPY buy/sell · recommended exposure
          </div>
        </div>
        <button onClick={loadData} className="flex items-center gap-1.5 h-[32px] px-3.5 rounded-[10px] text-xs font-medium"
                style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-2)" }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2"/>
          </svg>
          Refresh
        </button>
      </div>

      {/* ═══ FTD Section ═══ */}
      <div className="grid grid-cols-[2fr_1fr_1fr] gap-4 mb-5">
        <div className="px-4 py-3 rounded-[10px]" style={{ background: "#f8f9fc", borderLeft: "4px solid #6366f1" }}>
          <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Effective FTD</div>
          <div className="text-[16px] font-bold mt-0.5">
            {data.ftd_date || "—"} {data.ftd_source ? `· via ${data.ftd_source}` : ""}
          </div>
          <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>Earliest of NASDAQ or SPY confirms the cycle.</div>
        </div>
        <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
          <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>NASDAQ FTD</div>
          <div className="text-[16px] font-semibold mt-1" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
            {data.nasdaq_ftd || data.ftd_date || "—"}
          </div>
        </div>
        <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
          <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>SPY FTD</div>
          <div className="text-[16px] font-semibold mt-1" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
            {data.spy_ftd || "—"}
          </div>
        </div>
      </div>

      {/* ═══ Current Market Status ═══ */}
      <h3 className="text-[15px] font-semibold mb-3">Current Market Status — NASDAQ (^IXIC)</h3>

      <div className="grid grid-cols-4 gap-3 mb-4">
        <KPITile label="CLOSE" value={`$${(data.close || 0).toLocaleString(undefined, { minimumFractionDigits: 2 })}`}
                 sub={data.daily_change} gradient="linear-gradient(135deg, #6366f1, #818cf8)" />
        <KPITile label="BUY SWITCH" value={bsOn ? "ON" : "OFF"}
                 gradient={bsOn ? "linear-gradient(135deg, #10b981, #34d399)" : "linear-gradient(135deg, #e5484d, #f87171)"} />
        <KPITile label="EXPOSURE LEVEL" value={`${expLevel}/6`}
                 sub={`${allocPct}% allocation`} gradient="linear-gradient(135deg, #7c3aed, #a78bfa)" />
        <KPITile label="DISTRIBUTION DAYS" value={String(ddCount)}
                 gradient={ddCount <= 3 ? "linear-gradient(135deg, #10b981, #34d399)" : ddCount <= 5 ? "linear-gradient(135deg, #f59f00, #fbbf24)" : "linear-gradient(135deg, #e5484d, #f87171)"} />
      </div>

      {/* Signals today */}
      {(hasBuySigs || hasSellSigs) ? (
        <div className="mb-4 flex flex-col gap-2">
          <div className="text-[12px] font-semibold" style={{ color: "var(--ink-3)" }}>Signals Today:</div>
          {hasBuySigs && (
            <div className="px-4 py-2.5 rounded-[10px] text-[13px] font-medium" style={{ background: "color-mix(in oklab, #08a86b 10%, var(--surface))", color: "#16a34a", border: "1px solid color-mix(in oklab, #08a86b 30%, var(--border))" }}>
              BUY: {data.buy_signals.join(", ")}
            </div>
          )}
          {hasSellSigs && (
            <div className="px-4 py-2.5 rounded-[10px] text-[13px] font-medium" style={{ background: "color-mix(in oklab, #e5484d 10%, var(--surface))", color: "#dc2626", border: "1px solid color-mix(in oklab, #e5484d 30%, var(--border))" }}>
              SELL: {data.sell_signals.join(", ")}
            </div>
          )}
        </div>
      ) : (
        <div className="mb-4 px-4 py-2.5 rounded-[10px] text-[12px]" style={{ background: "color-mix(in oklab, #1e40af 10%, var(--surface))", color: "#3b82f6", border: "1px solid color-mix(in oklab, #1e40af 30%, var(--border))" }}>
          No new signals today
        </div>
      )}

      {/* Correction / Uptrend status */}
      {data.in_correction && !bsOn && !data.ftd_date ? (
        <div className="mb-5 px-4 py-3 rounded-[10px]" style={{ background: "color-mix(in oklab, #f59f00 10%, var(--surface))", borderLeft: "5px solid #f59f00" }}>
          <div className="text-[14px] font-bold">MARKET IN CORRECTION</div>
          <div className="text-[12px] mt-1" style={{ color: "var(--ink-3)" }}>
            {data.decline_pct?.toFixed(1)}% from ref high ${data.reference_high?.toLocaleString()}
          </div>
          <div className="text-[12px] font-semibold mt-1">Status: Looking for Follow-Through Day</div>
        </div>
      ) : data.ftd_date ? (
        <div className="mb-5 px-4 py-3 rounded-[10px]" style={{ background: "color-mix(in oklab, #08a86b 10%, var(--surface))", borderLeft: "5px solid #08a86b" }}>
          <div className="text-[14px] font-bold" style={{ color: "#08a86b" }}>CONFIRMED UPTREND — FTD on {data.ftd_date}</div>
          <div className="text-[12px] mt-1" style={{ color: "var(--ink-3)" }}>Reference high: ${data.reference_high?.toLocaleString()}</div>
        </div>
      ) : null}

      {/* Distribution Days Detail */}
      <details className="mb-5 rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <summary className="px-5 py-3 cursor-pointer text-[13px] font-semibold">
          Distribution Days Detail ({ddCount} active)
        </summary>
        <div className="p-4">
          {data.distribution_days?.length > 0 ? (
            <div className="flex flex-col gap-2">
              {data.distribution_days.map((dd: any, i: number) => {
                const daysAgo = Math.floor((Date.now() - new Date(dd.date).getTime()) / 86400000);
                const expiresIn = 25 - daysAgo;
                return (
                  <div key={i} className="flex items-center justify-between p-3 rounded-[8px]" style={{ border: "1px solid var(--border)" }}>
                    <div>
                      <span className="text-[12px] font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{dd.date}</span>
                      <span className="text-[11px] ml-2" style={{ color: "var(--ink-4)" }}>({daysAgo} days ago)</span>
                    </div>
                    <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>
                      {dd.type.toUpperCase()} · {dd.loss.toFixed(2)}% · Expires in {expiresIn} days
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="text-[12px]" style={{ color: "var(--ink-4)" }}>No active distribution days</div>
          )}
        </div>
      </details>

      {/* ═══ Historical Signal Tracking (Last 30 Days) ═══ */}
      <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
        <div className="px-5 py-3 text-[14px] font-semibold" style={{ borderBottom: "1px solid var(--border)" }}>
          Historical Signal Tracking (Last 30 Days)
        </div>

        {/* Tabs */}
        <div className="flex gap-1 px-5 pt-2 pb-0.5" style={{ borderBottom: "2px solid var(--border)" }}>
          {([
            { key: "exposure" as const, label: "Exposure Levels" },
            { key: "signals" as const, label: "Signal History" },
          ]).map(t => (
            <button key={t.key} onClick={() => setHistTab(t.key)}
                    className="px-4 py-2 text-[12px] font-medium transition-all"
                    style={{
                      color: histTab === t.key ? navColor : "var(--ink-4)",
                      borderBottom: histTab === t.key ? `2px solid ${navColor}` : "2px solid transparent",
                      marginBottom: -2,
                    }}>
              {t.label}
            </button>
          ))}
        </div>

        {/* Exposure Chart */}
        {histTab === "exposure" && (
          <div style={{ height: 300 }} className="px-2 py-3">
            {histData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={histData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.5} vertical={false} />
                  <XAxis dataKey="date" tick={{ fontSize: 10, fill: "var(--ink-4)" }} tickLine={false}
                         interval={Math.max(Math.floor(histData.length / 8), 1)} />
                  <YAxis tick={{ fontSize: 10, fill: "var(--ink-4)" }} tickLine={false} axisLine={false}
                         domain={[-0.5, 6.5]} ticks={[0, 1, 2, 3, 4, 5, 6]} width={30} />
                  <Tooltip contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 11 }}
                           formatter={(v: any) => [`${v}/6`, "Exposure"]} />
                  <Line dataKey="exposure" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3, fill: "#3b82f6" }} type="stepAfter" />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-sm" style={{ color: "var(--ink-4)" }}>
                No historical data available. Exposure chart populates from database sync.
              </div>
            )}
          </div>
        )}

        {/* Signal History */}
        {histTab === "signals" && (
          <div className="overflow-x-auto">
            {data.recent_signals?.length > 0 ? (
              <table className="w-full text-[11px]" style={{ borderCollapse: "collapse" }}>
                <thead>
                  <tr>
                    {["Date", "Signal", "Type"].map(h => (
                      <th key={h} className="text-left px-4 py-2.5 text-[9px] uppercase tracking-[0.06em] font-semibold"
                          style={{ color: "var(--ink-4)", background: "var(--surface-2)", borderBottom: "1px solid var(--border)" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.recent_signals.map((s: any, i: number) => {
                    const isBuy = s.signal.startsWith("B");
                    return (
                      <tr key={i} style={{ borderBottom: "1px solid var(--border)" }}
                          className="transition-colors"
                          onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-2)")}
                          onMouseLeave={e => (e.currentTarget.style.background = "transparent")}>
                        <td className="px-4 py-2.5" style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink-4)" }}>{s.date}</td>
                        <td className="px-4 py-2.5 font-semibold">{s.signal}</td>
                        <td className="px-4 py-2.5">
                          <span className="px-2 py-0.5 rounded text-[9px] font-bold"
                                style={{ background: isBuy ? "color-mix(in oklab, #08a86b 12%, var(--surface))" : "color-mix(in oklab, #e5484d 12%, var(--surface))", color: isBuy ? "#16a34a" : "#dc2626" }}>
                            {isBuy ? "BUY" : "SELL"}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            ) : (
              <div className="p-8 text-center text-sm" style={{ color: "var(--ink-4)" }}>No recent signals</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
