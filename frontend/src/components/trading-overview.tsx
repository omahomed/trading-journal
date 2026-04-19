"use client";

import { useState, useEffect, useMemo } from "react";
import { api, type JournalHistoryPoint, type TradePosition } from "@/lib/api";
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis,
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

export function TradingOverview({ navColor }: { navColor: string }) {
  const [history, setHistory] = useState<JournalHistoryPoint[]>([]);
  const [closed, setClosed] = useState<TradePosition[]>([]);
  const [recent, setRecent] = useState<any[]>([]);
  const [openTrades, setOpenTrades] = useState<TradePosition[]>([]);
  const [livePrices, setLivePrices] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      api.journalHistory("CanSlim", 0).catch(() => []),
      api.tradesClosed("CanSlim", 500).catch(() => []),
      api.tradesRecent("CanSlim", 10).catch(() => []),
      api.tradesOpen("CanSlim").catch(() => []),
    ]).then(async ([hist, cl, rec, open]) => {
      setHistory(hist as JournalHistoryPoint[]);
      setClosed(cl as TradePosition[]);
      setRecent(rec as any[]);
      const openArr = open as TradePosition[];
      setOpenTrades(openArr);

      // Fetch live prices for open trades to determine W/L
      const tickers = openArr.map(t => t.ticker).filter(Boolean);
      if (tickers.length > 0) {
        try {
          const prices = await api.batchPrices(tickers);
          if (prices && !("error" in prices)) setLivePrices(prices);
        } catch { /* fall back */ }
      }
      setLoading(false);
    });
  }, []);

  // ── Compute all metrics (matching Streamlit Trading Overview) ──
  const activeCount = openTrades.length;

  const metrics = useMemo(() => {
    const totalDays = history.length;

    // Period return — use portfolio_ltd from API (matches Streamlit TWR exactly)
    const periodReturn = totalDays > 0
      ? history[history.length - 1]?.portfolio_ltd || 0
      : 0;

    // Period P&L (sum of daily dollar changes)
    const periodPl = history.reduce((sum, h) => sum + (h.daily_dollar_change || 0), 0);

    // Realized P&L from closed trades
    const periodRealized = closed.reduce((sum, t) => sum + parseFloat(String(t.realized_pl || 0)), 0);

    // Win/loss from closed trades (matches Streamlit: wins = Realized_PL > 0, losses = rest)
    const closedCount = closed.length;
    const wins = closed.filter(t => parseFloat(String(t.realized_pl || 0)) > 0).length;
    const losses = closedCount - wins;
    const winRate = closedCount > 0 ? (wins / closedCount) * 100 : 0;
    // For display: actual negative trades
    const actualLosses = closed.filter(t => parseFloat(String(t.realized_pl || 0)) < 0).length;

    // Profit Factor
    const totalWinDol = closed
      .filter(t => parseFloat(String(t.realized_pl || 0)) > 0)
      .reduce((s, t) => s + parseFloat(String(t.realized_pl || 0)), 0);
    const totalLossDol = Math.abs(closed
      .filter(t => parseFloat(String(t.realized_pl || 0)) < 0)
      .reduce((s, t) => s + parseFloat(String(t.realized_pl || 0)), 0));
    const profitFactor = totalLossDol > 0 ? totalWinDol / totalLossDol : 0;

    // Avg Win / Avg Loss
    const avgWin = wins > 0 ? totalWinDol / wins : 0;
    const avgLoss = losses > 0 ? -(totalLossDol / losses) : 0;
    const avgWinLossRatio = avgLoss !== 0 ? Math.abs(avgWin / avgLoss) : 0;

    // Max Drawdown from equity curve
    let maxDD = 0;
    if (totalDays > 0) {
      let peak = 0;
      let cumProd = 1;
      for (const h of history) {
        cumProd *= (1 + (h.daily_pct_change || 0) / 100);
        if (cumProd > peak) peak = cumProd;
        const dd = peak > 0 ? ((cumProd - peak) / peak) * 100 : 0;
        if (dd < maxDD) maxDD = dd;
      }
    }

    // Last 10 trades W/L — combines open (unrealized) and closed (realized)
    // Closed: W if realized_pl > 0
    // Open: W if live price > avg_entry (unrealized gain)
    const allTrades: { date: string; win: boolean }[] = [];
    for (const t of closed) {
      allTrades.push({
        date: String(t.closed_date || t.open_date || ""),
        win: parseFloat(String(t.realized_pl || 0)) > 0,
      });
    }
    for (const t of openTrades) {
      const entry = parseFloat(String(t.avg_entry || 0));
      const price = livePrices[t.ticker] || 0;
      // W if we have a live price and it's above entry
      const isWin = price > 0 && entry > 0 ? price > entry : false;
      allTrades.push({
        date: String(t.open_date || ""),
        win: isWin,
      });
    }
    // Sort by date descending, take last 10
    allTrades.sort((a, b) => b.date.localeCompare(a.date));
    const last10 = allTrades.slice(0, 10);
    const last10W = last10.filter(t => t.win).length;
    const last10L = last10.length - last10W;

    // Total trades = closed + active
    const totalTrades = closedCount + activeCount;

    return {
      totalDays, periodReturn, periodPl, periodRealized,
      wins, losses, winRate, profitFactor,
      avgWin, avgLoss, avgWinLossRatio, maxDD,
      last10W, last10L, totalTrades, closedCount,
    };
  }, [history, closed, openTrades, livePrices]);

  // EC chart data — use portfolio_ltd from API (pre-computed TWR)
  const ecData = useMemo(() => {
    if (history.length === 0) return [];
    return history.map(h => ({
      day: h.day,
      date: String(h.day).slice(5, 10),
      return_pct: parseFloat((h.portfolio_ltd || 0).toFixed(2)),
    }));
  }, [history]);

  if (loading) {
    return <div className="animate-pulse"><div className="h-[90px] rounded-[14px] mb-6" style={{ background: "var(--bg-2)" }} /></div>;
  }

  const m = metrics;

  const kpis = [
    {
      label: "RETURN (ALL TIME)",
      value: `${m.periodReturn >= 0 ? "+" : ""}${m.periodReturn.toFixed(2)}%`,
      sub: `${m.totalDays} trading days`,
      gradient: "linear-gradient(135deg, #6366f1, #818cf8)",
    },
    {
      label: "P&L (ALL TIME)",
      value: `$${m.periodPl >= 0 ? "+" : ""}${m.periodPl.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
      sub: `Realized: $${m.periodRealized >= 0 ? "+" : ""}${m.periodRealized.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
      gradient: m.periodPl >= 0 ? "linear-gradient(135deg, #10b981, #34d399)" : "linear-gradient(135deg, #e5484d, #f87171)",
    },
    {
      label: "WIN RATE",
      value: `${m.winRate.toFixed(1)}%`,
      sub: `${m.wins}W / ${m.losses}L | PF: ${m.profitFactor.toFixed(2)}`,
      gradient: "linear-gradient(135deg, #ec4899, #f472b6)",
    },
    {
      label: "MAX DRAWDOWN",
      value: `${m.maxDD.toFixed(1)}%`,
      sub: `Avg W/L: ${m.avgWinLossRatio.toFixed(2)}`,
      gradient: "linear-gradient(135deg, #1e40af, #3b82f6)",
    },
  ];

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          Trading <em className="italic" style={{ color: navColor }}>Overview</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>All-time performance · {m.totalDays} trading days</div>
      </div>

      <div className="grid grid-cols-4 gap-3.5 mb-6">
        {kpis.map(k => <KPITile key={k.label} {...k} />)}
      </div>

      <div className="grid gap-[18px]" style={{ gridTemplateColumns: "3fr 2fr", alignItems: "stretch" }}>
        {/* EC — simple area chart */}
        <div className="rounded-[14px] overflow-hidden flex flex-col" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
          <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
            <span className="text-[13px] font-semibold">Equity Curve</span>
          </div>
          <div className="flex-1 min-h-[320px] px-2 py-3">
            {ecData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={ecData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="ecFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#6366f1" stopOpacity={0.2} />
                      <stop offset="100%" stopColor="#6366f1" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.5} vertical={false} />
                  <XAxis
                    dataKey="day"
                    tick={{ fontSize: 10, fill: "var(--ink-4)" }}
                    tickLine={false}
                    axisLine={{ stroke: "var(--border)" }}
                    interval={Math.max(Math.floor(ecData.length / 6), 1)}
                    tickFormatter={(v: string) => {
                      const d = new Date(v);
                      return d.toLocaleDateString("en-US", { month: "short", year: "2-digit" });
                    }}
                  />
                  <YAxis
                    tick={{ fontSize: 10, fill: "var(--ink-4)" }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v: number) => `${v}%`}
                    width={48}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "var(--surface)", border: "1px solid var(--border)",
                      borderRadius: 10, fontSize: 11, boxShadow: "0 4px 14px rgba(0,0,0,0.08)",
                      fontFamily: "var(--font-jetbrains), monospace",
                    }}
                    formatter={(value: any) => [`${Number(value).toFixed(2)}%`, "Return"]}
                    labelFormatter={(label: any) => {
                      const d = new Date(String(label));
                      return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
                    }}
                  />
                  <ReferenceLine y={0} stroke="var(--ink-4)" strokeDasharray="3 3" strokeOpacity={0.3} />
                  <Area
                    dataKey="return_pct"
                    stroke="#6366f1"
                    strokeWidth={2}
                    fill="url(#ecFill)"
                    type="monotone"
                    dot={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-sm" style={{ color: "var(--ink-4)" }}>
                No data available
              </div>
            )}
          </div>
        </div>

        {/* Trade Stats + Profit Breakdown */}
        <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
          <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
            <span className="text-[13px] font-semibold">Trade Stats</span>
          </div>
          <div className="p-[18px]">
            <div className="grid grid-cols-2 gap-2.5 mb-4">
              {[
                { k: "Total Trades", v: String(m.totalTrades) },
                { k: "Active", v: String(activeCount) },
                { k: "Avg W/L Ratio", v: m.avgWinLossRatio.toFixed(2) },
                { k: "Last 10", v: `${m.last10W}W / ${m.last10L}L` },
              ].map(s => (
                <div key={s.k} className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                  <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>{s.k}</div>
                  <div className="text-[18px] font-semibold mt-1" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{s.v}</div>
                </div>
              ))}
            </div>
            <div className="h-px mb-4" style={{ background: "var(--border)" }} />
            <div className="text-[10px] uppercase tracking-[0.10em] font-semibold mb-2.5" style={{ color: "var(--ink-4)" }}>Profit breakdown</div>
            <div className="grid grid-cols-2 gap-2.5 mb-4">
              <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Avg Win</div>
                <div className="text-[18px] font-semibold mt-1 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace", color: "#08a86b" }}>${m.avgWin.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
              </div>
              <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Avg Loss</div>
                <div className="text-[18px] font-semibold mt-1 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace", color: "#e5484d" }}>${m.avgLoss.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2.5">
              <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Profit Factor</div>
                <div className="text-[18px] font-semibold mt-1" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{m.profitFactor.toFixed(2)}</div>
              </div>
              <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Closed Trades</div>
                <div className="text-[18px] font-semibold mt-1" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{m.closedCount}</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Recent trades table */}
      <div className="mt-6 rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
        <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
          <span className="text-[13px] font-semibold">Recent Trades</span>
          <span className="text-xs" style={{ color: "var(--ink-4)" }}>Last {recent.length} transactions</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[12.5px]" style={{ borderCollapse: "separate", borderSpacing: 0 }}>
            <thead>
              <tr>
                {["Date", "Ticker", "Action", "Shares", "Price", "Value", "Rule"].map(h => (
                  <th key={h} className="text-left text-[10px] uppercase tracking-[0.08em] font-semibold px-3 py-2.5 whitespace-nowrap sticky top-0"
                      style={{ color: "var(--ink-4)", background: "var(--surface-2)", borderBottom: "1px solid var(--border)" }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {recent.map((t: any, i: number) => (
                <tr key={i} className="transition-colors" style={{ borderBottom: i < recent.length - 1 ? "1px solid var(--border)" : "none" }}
                    onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-2)")}
                    onMouseLeave={e => (e.currentTarget.style.background = "transparent")}>
                  <td className="px-3 py-2.5 whitespace-nowrap" style={{ fontFamily: "var(--font-jetbrains), monospace", fontSize: 11, color: "var(--ink-4)" }}>
                    {String(t.date || "").slice(5, 16).replace(" ", " ")}
                  </td>
                  <td className="px-3 py-2.5 font-semibold whitespace-nowrap" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{t.ticker}</td>
                  <td className="px-3 py-2.5">
                    <span className="px-2 py-0.5 rounded-full text-[11px] font-medium"
                          style={{ background: t.action === "BUY" ? "#e5f7ee" : "#fdecec", color: t.action === "BUY" ? "#08a86b" : "#e5484d" }}>
                      {t.action}
                    </span>
                  </td>
                  <td className="px-3 py-2.5 text-right" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{parseInt(t.shares)}</td>
                  <td className="px-3 py-2.5 text-right privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>${parseFloat(t.amount || 0).toFixed(2)}</td>
                  <td className="px-3 py-2.5 text-right privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>${parseFloat(t.value || 0).toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                  <td className="px-3 py-2.5 text-[11px]" style={{ color: "var(--ink-3)" }}>{t.rule || ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
