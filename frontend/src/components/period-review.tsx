"use client";

import { useState, useEffect, useMemo } from "react";
import { api, type TradePosition } from "@/lib/api";
import {
  ResponsiveContainer, ComposedChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend,
} from "recharts";

/* ── helpers ── */
function fmt$(v: number) { return v >= 0 ? `$${v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : `-$${Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`; }
function fmtPct(v: number) { return `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`; }
function pctColor(v: number) { return v > 0 ? "#08a86b" : v < 0 ? "#e5484d" : "var(--ink-3)"; }

type Tab = "weekly" | "monthly" | "annual";

/* ── Metric tile ── */
function MetricTile({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="p-4 rounded-[12px]" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
      <div className="text-[10px] font-bold uppercase tracking-[0.10em]" style={{ color: "var(--ink-4)" }}>{label}</div>
      <div className="text-[22px] font-extrabold mt-1 privacy-mask" style={{ color: color || "var(--ink)", fontFamily: "var(--font-jetbrains), monospace", lineHeight: 1.2 }}>{value}</div>
      {sub && <div className="text-[11px] mt-1 font-medium" style={{ color: "var(--ink-4)" }}>{sub}</div>}
    </div>
  );
}

/* ── Capital Deployed section ── */
function CapitalDeployed({ data }: { data: any[] }) {
  const stats = useMemo(() => {
    if (!data.length) return null;
    const firstBeg = parseFloat(data[0].beg_nlv || data[0].end_nlv || 0);
    let totalDeposits = 0;
    let totalWithdrawals = 0;
    data.forEach(d => {
      const cf = parseFloat(d.cash_change || 0);
      if (cf > 0) totalDeposits += cf;
      else totalWithdrawals += cf;
    });
    const netInvested = firstBeg + totalDeposits + totalWithdrawals;
    const currentValue = parseFloat(data[data.length - 1].end_nlv || 0);
    const straightPnl = currentValue - netInvested;
    const straightReturn = netInvested > 0 ? (straightPnl / netInvested) * 100 : 0;
    const twrLtd = parseFloat(data[data.length - 1].portfolio_ltd || 0);

    return { firstBeg, totalDeposits, totalWithdrawals: Math.abs(totalWithdrawals), netInvested, currentValue, straightPnl, straightReturn, twrLtd };
  }, [data]);

  if (!stats) return null;

  const [expanded, setExpanded] = useState(true);

  return (
    <div className="rounded-[14px] overflow-hidden mb-6" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
      {/* Header */}
      <button onClick={() => setExpanded(!expanded)} className="w-full flex items-center justify-between p-4 text-left cursor-pointer">
        <div className="flex items-center gap-2">
          <span className="text-[15px] font-bold" style={{ color: "var(--ink)" }}>Capital Deployed & Straight Return (LTD)</span>
        </div>
        <span className="text-[12px]" style={{ color: "var(--ink-4)" }}>{expanded ? "▲" : "▼"}</span>
      </button>

      {expanded && (
        <div className="px-4 pb-4">
          {/* Top row: 5 capital metrics */}
          <div className="grid grid-cols-5 gap-3 mb-4">
            <MetricTile label="Starting Capital" value={`$${stats.firstBeg.toLocaleString("en-US", { maximumFractionDigits: 0 })}`} />
            <MetricTile label="Total Deposited" value={`$${stats.totalDeposits.toLocaleString("en-US", { maximumFractionDigits: 0 })}`} />
            <MetricTile label="Total Withdrawn" value={`$${stats.totalWithdrawals.toLocaleString("en-US", { maximumFractionDigits: 0 })}`} />
            <MetricTile label="Net Capital Deployed" value={`$${stats.netInvested.toLocaleString("en-US", { maximumFractionDigits: 0 })}`} sub="Starting + deposits − withdrawals" />
            <MetricTile label="Current Value" value={`$${stats.currentValue.toLocaleString("en-US", { maximumFractionDigits: 0 })}`} color={stats.currentValue >= stats.netInvested ? "#08a86b" : "#e5484d"} />
          </div>

          {/* Divider */}
          <div className="h-px mb-4" style={{ background: "var(--border)" }} />

          {/* Bottom row: 3 return metrics */}
          <div className="grid grid-cols-3 gap-3 mb-3">
            <MetricTile
              label="Straight P&L"
              value={fmt$(stats.straightPnl)}
              sub={stats.straightPnl >= 0 ? "Profit" : "Loss"}
              color={pctColor(stats.straightPnl)}
            />
            <MetricTile
              label="Straight Return (LTD)"
              value={fmtPct(stats.straightReturn)}
              sub="(Current − Net Deployed) / Net Deployed"
              color={pctColor(stats.straightReturn)}
            />
            <MetricTile
              label="TWR Return (LTD)"
              value={fmtPct(stats.twrLtd)}
              sub="Immune to cash flows"
              color={pctColor(stats.twrLtd)}
            />
          </div>

          {/* Caption */}
          <div className="text-[11px] leading-relaxed px-1" style={{ color: "var(--ink-4)" }}>
            <b>Straight Return</b> answers: &ldquo;I deployed {fmt$(stats.netInvested)} and now have {fmt$(stats.currentValue)} — that&apos;s a {fmtPct(stats.straightReturn)} return on capital.&rdquo;
            <br />
            <b>TWR Return</b> answers: &ldquo;My trading decisions generated {fmtPct(stats.twrLtd)} regardless of deposits/withdrawals.&rdquo;
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Period data aggregation (geometric linking) ── */
interface PeriodRow {
  label: string;
  date: Date;
  begNlv: number;
  endNlv: number;
  cashFlow: number;
  periodPnl: number;
  periodReturn: number;
  portfolioLtd: number;
  spyLtd: number;
  ndxLtd: number;
  dailyDollarChange: number;
}

function aggregatePeriods(data: any[], mode: Tab): PeriodRow[] {
  if (!data.length) return [];

  // Group by period
  const groups = new Map<string, any[]>();
  data.forEach(d => {
    const dt = new Date(d.day);
    let key: string;
    if (mode === "weekly") {
      // Week ending Friday
      const fri = new Date(dt);
      const dow = fri.getDay();
      const diff = dow <= 5 ? 5 - dow : 5 - dow + 7;
      fri.setDate(fri.getDate() + diff);
      key = fri.toISOString().slice(0, 10);
    } else if (mode === "monthly") {
      key = `${dt.getFullYear()}-${String(dt.getMonth() + 1).padStart(2, "0")}`;
    } else {
      key = `${dt.getFullYear()}`;
    }
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(d);
  });

  // Aggregate each period
  const rows: PeriodRow[] = [];
  for (const [key, days] of groups) {
    const sorted = days.sort((a: any, b: any) => String(a.day).localeCompare(String(b.day)));
    const begNlv = parseFloat(sorted[0].beg_nlv || sorted[0].end_nlv || 0);
    const endNlv = parseFloat(sorted[sorted.length - 1].end_nlv || 0);
    const cashFlow = sorted.reduce((s: number, d: any) => s + parseFloat(d.cash_change || 0), 0);
    const dailyDollarChange = sorted.reduce((s: number, d: any) => s + parseFloat(d.daily_dollar_change || 0), 0);

    // Geometric linking of daily returns → Period TWR
    let product = 1;
    sorted.forEach((d: any) => { product *= 1 + parseFloat(d.daily_return || 0); });
    const periodReturn = (product - 1) * 100;

    const periodPnl = endNlv - (begNlv + cashFlow);
    const portfolioLtd = parseFloat(sorted[sorted.length - 1].portfolio_ltd || 0);
    const spyLtd = parseFloat(sorted[sorted.length - 1].spy_ltd || 0);
    const ndxLtd = parseFloat(sorted[sorted.length - 1].ndx_ltd || 0);

    let label: string;
    if (mode === "weekly") {
      label = key; // YYYY-MM-DD (Friday)
    } else if (mode === "monthly") {
      const [y, m] = key.split("-");
      const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
      label = `${monthNames[parseInt(m) - 1]} ${y}`;
    } else {
      label = key;
    }

    rows.push({
      label, date: new Date(sorted[sorted.length - 1].day),
      begNlv, endNlv, cashFlow, periodPnl, periodReturn,
      portfolioLtd, spyLtd, ndxLtd, dailyDollarChange,
    });
  }

  return rows.sort((a, b) => a.date.getTime() - b.date.getTime());
}

/* ── Insights panel ── */
function InsightsPanel({ rows, mode, lastPortLtd, lastSpyLtd, lastNdxLtd }: {
  rows: PeriodRow[]; mode: string; lastPortLtd: number; lastSpyLtd: number; lastNdxLtd: number;
}) {
  const returns = rows.map(r => r.periodReturn);
  const n = returns.length;
  const wins = returns.filter(r => r > 0).length;
  const losses = returns.filter(r => r < 0).length;
  const flat = n - wins - losses;
  const winRate = n > 0 ? (wins / n) * 100 : 0;

  const avgWin = wins > 0 ? returns.filter(r => r > 0).reduce((a, b) => a + b, 0) / wins : 0;
  const avgLoss = losses > 0 ? returns.filter(r => r < 0).reduce((a, b) => a + b, 0) / losses : 0;

  const bestIdx = returns.indexOf(Math.max(...returns));
  const worstIdx = returns.indexOf(Math.min(...returns));
  const best = returns[bestIdx] || 0;
  const worst = returns[worstIdx] || 0;
  const bestLabel = rows[bestIdx]?.label || "";
  const worstLabel = rows[worstIdx]?.label || "";

  // Current streak
  let streak = 0;
  let streakType = "";
  for (let i = returns.length - 1; i >= 0; i--) {
    const w = returns[i] > 0;
    if (streak === 0) { streakType = w ? "W" : "L"; streak = 1; }
    else if ((w && streakType === "W") || (!w && streakType === "L")) streak++;
    else break;
  }

  const alphaSpy = lastPortLtd - lastSpyLtd;
  const alphaNdx = lastPortLtd - lastNdxLtd;

  return (
    <div>
      <div className="text-[14px] font-bold mb-3" style={{ color: "var(--ink)" }}>{mode} Insights</div>

      {/* Win Rate */}
      <div className="p-3 rounded-[10px] mb-2" style={{ background: "var(--bg-2)", border: "1px solid var(--border)" }}>
        <div className="text-[10px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>Win Rate</div>
        <div className="text-[22px] font-extrabold" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{winRate.toFixed(0)}%</div>
        <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>{wins}W / {losses}L / {flat}F out of {n}</div>
      </div>

      {/* Streak */}
      <div className="p-3 rounded-[10px] mb-2" style={{ background: "var(--bg-2)", border: "1px solid var(--border)" }}>
        <div className="text-[10px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>Current Streak</div>
        <div className="text-[18px] font-extrabold" style={{ color: streakType === "W" ? "#08a86b" : "#e5484d", fontFamily: "var(--font-jetbrains), monospace" }}>
          {streakType === "W" ? "●" : "●"} {streak} {streakType === "W" ? "winning" : "losing"}
        </div>
      </div>

      {/* Avg Win / Loss */}
      <div className="grid grid-cols-2 gap-2 mb-2">
        <div className="p-3 rounded-[10px]" style={{ background: "var(--bg-2)", border: "1px solid var(--border)" }}>
          <div className="text-[10px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>Avg Win</div>
          <div className="text-[16px] font-extrabold" style={{ color: "#08a86b", fontFamily: "var(--font-jetbrains), monospace" }}>{fmtPct(avgWin)}</div>
        </div>
        <div className="p-3 rounded-[10px]" style={{ background: "var(--bg-2)", border: "1px solid var(--border)" }}>
          <div className="text-[10px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>Avg Loss</div>
          <div className="text-[16px] font-extrabold" style={{ color: "#e5484d", fontFamily: "var(--font-jetbrains), monospace" }}>{fmtPct(avgLoss)}</div>
        </div>
      </div>

      {/* Best / Worst */}
      <div className="grid grid-cols-2 gap-2 mb-2">
        <div className="p-3 rounded-[10px]" style={{ background: "var(--bg-2)", border: "1px solid var(--border)" }}>
          <div className="text-[10px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>Best Period</div>
          <div className="text-[16px] font-extrabold" style={{ color: "#08a86b", fontFamily: "var(--font-jetbrains), monospace" }}>{fmtPct(best)}</div>
          <div className="text-[10px]" style={{ color: "var(--ink-4)" }}>{bestLabel}</div>
        </div>
        <div className="p-3 rounded-[10px]" style={{ background: "var(--bg-2)", border: "1px solid var(--border)" }}>
          <div className="text-[10px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>Worst Period</div>
          <div className="text-[16px] font-extrabold" style={{ color: "#e5484d", fontFamily: "var(--font-jetbrains), monospace" }}>{fmtPct(worst)}</div>
          <div className="text-[10px]" style={{ color: "var(--ink-4)" }}>{worstLabel}</div>
        </div>
      </div>

      {/* Alpha */}
      <div className="grid grid-cols-2 gap-2">
        <div className="p-3 rounded-[10px]" style={{ background: "var(--bg-2)", border: "1px solid var(--border)" }}>
          <div className="text-[10px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>Alpha vs SPY</div>
          <div className="text-[16px] font-extrabold" style={{ color: pctColor(alphaSpy), fontFamily: "var(--font-jetbrains), monospace" }}>{fmtPct(alphaSpy)}</div>
        </div>
        <div className="p-3 rounded-[10px]" style={{ background: "var(--bg-2)", border: "1px solid var(--border)" }}>
          <div className="text-[10px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>Alpha vs Nasdaq</div>
          <div className="text-[16px] font-extrabold" style={{ color: pctColor(alphaNdx), fontFamily: "var(--font-jetbrains), monospace" }}>{fmtPct(alphaNdx)}</div>
        </div>
      </div>
    </div>
  );
}

/* ── Equity Curve chart ── */
function EquityCurveChart({ rows }: { rows: PeriodRow[] }) {
  const chartData = rows.map(r => ({
    label: r.label,
    portfolio: parseFloat(r.portfolioLtd.toFixed(2)),
    spy: parseFloat(r.spyLtd.toFixed(2)),
    ndx: parseFloat(r.ndxLtd.toFixed(2)),
  }));

  const lastPort = chartData.length ? chartData[chartData.length - 1].portfolio : 0;
  const lastSpy = chartData.length ? chartData[chartData.length - 1].spy : 0;
  const lastNdx = chartData.length ? chartData[chartData.length - 1].ndx : 0;

  return (
    <ResponsiveContainer width="100%" height={400}>
      <ComposedChart data={chartData} margin={{ top: 10, right: 10, bottom: 10, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis
          dataKey="label"
          tick={{ fill: "var(--ink-4)", fontSize: 10 }}
          tickLine={false}
          interval="preserveStartEnd"
          minTickGap={60}
        />
        <YAxis
          tick={{ fill: "var(--ink-4)", fontSize: 10 }}
          tickFormatter={(v: number) => `${v}%`}
          tickLine={false}
          axisLine={false}
        />
        <Tooltip
          contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 12 }}
          formatter={(val: any, name: any) => [`${Number(val).toFixed(2)}%`, name]}
        />
        <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: 12 }} />
        <Line
          type="monotone" dataKey="portfolio" name={`Portfolio (${lastPort >= 0 ? "+" : ""}${lastPort.toFixed(1)}%)`}
          stroke="#1f77b4" strokeWidth={2.5} dot={false}
        />
        <Line
          type="monotone" dataKey="spy" name={`S&P 500 (${lastSpy >= 0 ? "+" : ""}${lastSpy.toFixed(1)}%)`}
          stroke="#888" strokeWidth={1.5} dot={false} opacity={0.7}
        />
        <Line
          type="monotone" dataKey="ndx" name={`Nasdaq (${lastNdx >= 0 ? "+" : ""}${lastNdx.toFixed(1)}%)`}
          stroke="#e67e22" strokeWidth={1.5} dot={false} opacity={0.7}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

/* ── Financial Statement table ── */
const DEFAULT_VISIBLE = 10;

function FinancialTable({ rows, mode }: { rows: PeriodRow[]; mode: Tab }) {
  const sorted = [...rows].reverse(); // Most recent first
  const [showAll, setShowAll] = useState(false);
  const visible = showAll ? sorted : sorted.slice(0, DEFAULT_VISIBLE);
  const hasMore = sorted.length > DEFAULT_VISIBLE;

  return (
    <div>
      <div className="overflow-auto rounded-[12px]" style={{ border: "1px solid var(--border)", maxHeight: showAll ? 600 : undefined }}>
        <table className="w-full text-[12px]" style={{ borderCollapse: "collapse" }}>
          <thead className="sticky top-0 z-10">
            <tr style={{ background: "var(--bg-2)" }}>
              <th className="text-left p-3 font-bold" style={{ color: "var(--ink-3)" }}>{mode === "weekly" ? "Week Ending" : mode === "monthly" ? "Month" : "Year"}</th>
              <th className="text-right p-3 font-bold" style={{ color: "var(--ink-3)" }}>Start Equity</th>
              <th className="text-right p-3 font-bold" style={{ color: "var(--ink-3)" }}>Cash Flow</th>
              <th className="text-right p-3 font-bold" style={{ color: "var(--ink-3)" }}>End Equity</th>
              <th className="text-right p-3 font-bold" style={{ color: "var(--ink-3)" }}>Net P&L ($)</th>
              <th className="text-right p-3 font-bold" style={{ color: "var(--ink-3)" }}>Return % (TWR)</th>
              <th className="text-right p-3 font-bold" style={{ color: "var(--ink-3)" }}>LTD Return %</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((r, i) => (
              <tr key={i} style={{ borderTop: "1px solid var(--border)" }} className="hover:brightness-95 transition-colors">
                <td className="p-3 font-semibold" style={{ color: "var(--ink)" }}>{r.label}</td>
                <td className="text-right p-3 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink-2)" }}>{fmt$(r.begNlv)}</td>
                <td className="text-right p-3 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace", color: r.cashFlow !== 0 ? "#0d6efd" : "var(--ink-3)" }}>{fmt$(r.cashFlow)}</td>
                <td className="text-right p-3 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink-2)" }}>{fmt$(r.endNlv)}</td>
                <td className="text-right p-3 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace", color: pctColor(r.periodPnl) }}>{fmt$(r.periodPnl)}</td>
                <td className="text-right p-3" style={{ fontFamily: "var(--font-jetbrains), monospace", color: pctColor(r.periodReturn) }}>{fmtPct(r.periodReturn)}</td>
                <td className="text-right p-3" style={{ fontFamily: "var(--font-jetbrains), monospace", color: pctColor(r.portfolioLtd) }}>{fmtPct(r.portfolioLtd)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {hasMore && (
        <button onClick={() => setShowAll(!showAll)}
          className="mt-2 text-[12px] font-semibold cursor-pointer px-3 py-1.5 rounded-[8px] transition-colors"
          style={{ color: "var(--ink-3)", background: "var(--bg-2)", border: "1px solid var(--border)" }}>
          {showAll ? `Show latest ${DEFAULT_VISIBLE}` : `Show all ${sorted.length} periods`}
        </button>
      )}
    </div>
  );
}

/* ── Period tab content ── */
function PeriodTab({ data, closedTrades, mode }: { data: any[]; closedTrades: TradePosition[]; mode: Tab }) {
  const rows = useMemo(() => aggregatePeriods(data, mode), [data, mode]);

  if (!rows.length) return <div className="p-8 text-center" style={{ color: "var(--ink-4)" }}>Not enough data for {mode} review.</div>;

  const latest = rows[rows.length - 1];
  const lastPortLtd = latest.portfolioLtd;
  const lastSpyLtd = latest.spyLtd;
  const lastNdxLtd = latest.ndxLtd;

  // Trades closed in latest period
  const periodStart = (() => {
    if (mode === "weekly") {
      const d = new Date(latest.date);
      d.setDate(d.getDate() - 6);
      return d;
    }
    if (mode === "monthly") {
      const d = new Date(latest.date);
      d.setDate(1);
      return d;
    }
    const d = new Date(latest.date);
    d.setMonth(0, 1);
    return d;
  })();

  const periodEnd = latest.date;
  const periodClosed = closedTrades.filter(t => {
    const cd = new Date(t.closed_date || "");
    return !isNaN(cd.getTime()) && cd >= periodStart && cd <= new Date(periodEnd.getTime() + 86400000);
  });
  const tradeCount = periodClosed.length;
  const wins = periodClosed.filter(t => parseFloat(String(t.realized_pl || 0)) > 0).length;
  const winRate = tradeCount > 0 ? (wins / tradeCount) * 100 : 0;

  const modeLabel = mode === "weekly" ? "Weekly" : mode === "monthly" ? "Monthly" : "Annual";

  return (
    <div>
      {/* Top metrics row */}
      <div className="grid grid-cols-4 gap-3 mb-6">
        <MetricTile label={`Latest ${modeLabel} P&L`} value={fmt$(latest.periodPnl)} color={pctColor(latest.periodPnl)} />
        <MetricTile label={`${modeLabel} Return %`} value={fmtPct(latest.periodReturn)} color={pctColor(latest.periodReturn)} />
        <MetricTile label="Trades Closed" value={String(tradeCount)} />
        <MetricTile label="Win Rate" value={`${winRate.toFixed(1)}%`} color={winRate >= 50 ? "#08a86b" : "#e5484d"} />
      </div>

      {/* Chart + Insights split */}
      <div className="text-[14px] font-bold mb-3" style={{ color: "var(--ink)" }}>{modeLabel} Performance vs Benchmark</div>
      <div className="grid grid-cols-[3fr_2fr] gap-4 mb-6">
        <div className="rounded-[14px] p-4" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
          <EquityCurveChart rows={rows} />
        </div>
        <div className="rounded-[14px] p-4" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
          <InsightsPanel rows={rows} mode={modeLabel} lastPortLtd={lastPortLtd} lastSpyLtd={lastSpyLtd} lastNdxLtd={lastNdxLtd} />
        </div>
      </div>

      {/* Financial Statement table */}
      <div className="text-[14px] font-bold mb-3" style={{ color: "var(--ink)" }}>{modeLabel} Financial Statement</div>
      <FinancialTable rows={rows} mode={mode} />
    </div>
  );
}

/* ══════════════════════════════════════════════════════════ */
/* ██ MAIN EXPORT                                          ██ */
/* ══════════════════════════════════════════════════════════ */
export function PeriodReview({ navColor }: { navColor: string }) {
  const [data, setData] = useState<any[]>([]);
  const [closedTrades, setClosedTrades] = useState<TradePosition[]>([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<Tab>("weekly");

  useEffect(() => {
    Promise.all([
      api.journalHistory("CanSlim", 0).catch(() => []),
      api.tradesClosed("CanSlim", 5000).catch(() => []),
    ]).then(([jrnl, trades]) => {
      setData(jrnl as any[]);
      setClosedTrades(trades as TradePosition[]);
      setLoading(false);
    });
  }, []);

  // CAGR (for annual tab)
  const cagr = useMemo(() => {
    if (!data.length) return 0;
    const first = new Date(data[0].day);
    const last = new Date(data[data.length - 1].day);
    const totalDays = (last.getTime() - first.getTime()) / (1000 * 60 * 60 * 24);
    if (totalDays <= 0) return 0;
    const years = totalDays / 365.25;
    // TWR curve final value
    let product = 1;
    data.forEach(d => { product *= 1 + parseFloat(d.daily_return || 0); });
    return years > 0 ? (Math.pow(product, 1 / years) - 1) * 100 : 0;
  }, [data]);

  const cagrYears = useMemo(() => {
    if (!data.length) return 0;
    const first = new Date(data[0].day);
    const last = new Date(data[data.length - 1].day);
    return (last.getTime() - first.getTime()) / (1000 * 60 * 60 * 24 * 365.25);
  }, [data]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-7 h-7 border-[3px] border-t-transparent rounded-full animate-spin" style={{ borderColor: `${navColor} transparent ${navColor} ${navColor}` }} />
      </div>
    );
  }

  if (!data.length) {
    return <div className="p-8 text-center" style={{ color: "var(--ink-4)" }}>Insufficient data to generate review.</div>;
  }

  const tabs: { key: Tab; label: string }[] = [
    { key: "weekly", label: "Weekly Review" },
    { key: "monthly", label: "Monthly Review" },
    { key: "annual", label: "Annual & CAGR" },
  ];

  return (
    <div>
      {/* Capital Deployed */}
      <CapitalDeployed data={data} />

      {/* Divider */}
      <div className="h-px mb-5" style={{ background: "var(--border)" }} />

      {/* Tabs */}
      <div className="flex gap-1 mb-5 p-1 rounded-[10px] w-fit" style={{ background: "var(--bg-2)" }}>
        {tabs.map(t => (
          <button key={t.key} onClick={() => setTab(t.key)}
            className="px-4 py-2 rounded-[8px] text-[12px] font-semibold transition-all cursor-pointer"
            style={{
              background: tab === t.key ? navColor : "transparent",
              color: tab === t.key ? "#fff" : "var(--ink-3)",
            }}>
            {t.label}
          </button>
        ))}
      </div>

      {/* CAGR banner for annual tab */}
      {tab === "annual" && (
        <div className="mb-5 p-4 rounded-[14px] flex items-center gap-4" style={{ background: `color-mix(in oklab, ${navColor} 8%, var(--surface))`, border: "1px solid var(--border)" }}>
          <div>
            <div className="text-[13px] font-bold" style={{ color: "var(--ink)" }}>Compound Annual Growth Rate (CAGR)</div>
            <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>Calculated using Time-Weighted Return over {cagrYears.toFixed(1)} years.</div>
          </div>
          <div className="text-[28px] font-extrabold ml-auto" style={{ color: pctColor(cagr), fontFamily: "var(--font-jetbrains), monospace" }}>{fmtPct(cagr)}</div>
        </div>
      )}

      {/* Period content */}
      <PeriodTab data={data} closedTrades={closedTrades} mode={tab} />
    </div>
  );
}
