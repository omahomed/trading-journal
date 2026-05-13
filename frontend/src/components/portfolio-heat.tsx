"use client";

import { useState, useEffect } from "react";
import { api, getActivePortfolio, type TradePosition } from "@/lib/api";
import { formatCurrency } from "@/lib/format";

// Canonical option-row predicate, matching perf-heatmap.tsx:73-75 and the
// instrument_type column (Migration 016). Used to keep Portfolio Heat as a
// stocks-only volatility check — options' yfinance ATR is unavailable and
// their premium-as-total_cost has incommensurate risk semantics anyway.
const isOptionRow = (t: TradePosition): boolean =>
  String((t as any).instrument_type || "").toUpperCase() === "OPTION"
  || /^\S+\s+\d{6}\s+\$[0-9.]+(C|P)$/.test(String(t.ticker || ""));

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

interface HeatRow {
  ticker: string;
  trade_id: string;
  weight_pct: number;
  atr_pct: number;
  heat_contribution: number;
  total_cost: number;
  shares: number;
  rule: string;
  // "ok"     — atr_pct is a valid yfinance result, contributes to total heat
  // "failed" — priceLookup raised or rate-limited; row renders "—" and is
  //            excluded from total heat / avgVol so a transient yfinance
  //            miss doesn't silently deflate the headline number
  atr_status: "ok" | "failed";
}

export function PortfolioHeat({ navColor }: { navColor: string }) {
  const [positions, setPositions] = useState<TradePosition[]>([]);
  const [equity, setEquity] = useState(0);
  const [heatData, setHeatData] = useState<HeatRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [fetching, setFetching] = useState(false);
  const [mode, setMode] = useState<"auto" | "manual">("auto");
  const [manualAtr, setManualAtr] = useState<Record<string, string>>({});
  const HEAT_THRESHOLD = 20; // default from app_config

  useEffect(() => {
    Promise.all([
      api.tradesOpen(getActivePortfolio()).catch(() => []),
      api.journalLatest(getActivePortfolio()).catch(() => ({ end_nlv: 100000 })),
    ]).then(([open, journal]) => {
      const openArr = open as TradePosition[];
      // Stocks-only — options carry no yfinance ATR and skew the position
      // count caption; they're filtered here once and the rest of the page
      // (table, manual-ATR grid, totals) reads `positions` already filtered.
      const stockPositions = openArr.filter(t => !isOptionRow(t));
      setPositions(stockPositions);
      const eq = parseFloat(String((journal as any).end_nlv || 100000));
      setEquity(eq);
      setLoading(false);

      // Auto-fetch ATR for all tickers
      if (stockPositions.length > 0) {
        fetchATR(stockPositions, eq);
      }
    });
  }, []);

  const fetchATR = async (trades: TradePosition[], eq: number) => {
    setFetching(true);
    const results: HeatRow[] = [];

    for (const t of trades) {
      const totalCost = parseFloat(String(t.total_cost || 0));
      const weightPct = eq > 0 ? (totalCost / eq) * 100 : 0;
      let atrPct = 0;
      let atrStatus: "ok" | "failed" = "failed";

      try {
        const data = await api.priceLookup(t.ticker);
        // Backend now raises HTTP 503 on yfinance miss, which fetchJSON
        // converts to a throw caught below. A 200 response with a real
        // payload always has a numeric atr_pct.
        if (data && typeof data.atr_pct === "number") {
          atrPct = data.atr_pct;
          atrStatus = "ok";
        }
      } catch { /* keep atrStatus = "failed" */ }

      results.push({
        ticker: t.ticker,
        trade_id: t.trade_id,
        weight_pct: weightPct,
        atr_pct: atrPct,
        heat_contribution: atrStatus === "ok" ? weightPct * (atrPct / 100) : 0,
        total_cost: totalCost,
        shares: t.shares || 0,
        rule: t.rule || "",
        atr_status: atrStatus,
      });
    }

    setHeatData(results.sort((a, b) => b.heat_contribution - a.heat_contribution));
    setFetching(false);
  };

  const computeManualHeat = () => {
    const results: HeatRow[] = positions.map(t => {
      const totalCost = parseFloat(String(t.total_cost || 0));
      const weightPct = equity > 0 ? (totalCost / equity) * 100 : 0;
      const atrPct = parseFloat(manualAtr[t.ticker] || "5") || 5;
      return {
        ticker: t.ticker,
        trade_id: t.trade_id,
        weight_pct: weightPct,
        atr_pct: atrPct,
        heat_contribution: weightPct * (atrPct / 100),
        total_cost: totalCost,
        shares: t.shares || 0,
        rule: t.rule || "",
        // Manual override always counts — user typed the value themselves.
        atr_status: "ok",
      };
    });
    setHeatData(results.sort((a, b) => b.heat_contribution - a.heat_contribution));
  };

  if (loading) {
    return <div className="animate-pulse"><div className="h-[90px] rounded-[14px]" style={{ background: "var(--bg-2)" }} /></div>;
  }

  // Failed rows (yfinance miss / 503) are excluded from totalHeat and the
  // unweighted avgVol so a partial yfinance outage doesn't silently deflate
  // the headline. The table still renders those rows with "—" so the user
  // can see which positions are missing data and refresh if needed.
  const okRows = heatData.filter(h => h.atr_status === "ok");
  const failedCount = heatData.length - okRows.length;
  const totalHeat = okRows.reduce((a, h) => a + h.heat_contribution, 0);
  const avgVol = okRows.length > 0 ? okRows.reduce((a, h) => a + h.atr_pct, 0) / okRows.length : 0;
  const heatOk = totalHeat < HEAT_THRESHOLD;

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          Portfolio <em className="italic" style={{ color: navColor }}>Heat</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
          Volatility Check · {positions.length} stock positions
          {failedCount > 0 && (
            <span style={{ color: "#d97706" }}>
              {" "}· {failedCount} unavailable (refresh to retry)
            </span>
          )}
        </div>
      </div>

      {/* Mode selector */}
      <div className="flex gap-2 mb-6">
        <button onClick={() => { setMode("auto"); if (positions.length > 0) fetchATR(positions, equity); }}
                className="h-[36px] px-4 rounded-[10px] text-[12px] font-semibold transition-all cursor-pointer"
                style={{
                  background: mode === "auto" ? `color-mix(in oklab, ${navColor} 10%, transparent)` : "var(--bg)",
                  color: mode === "auto" ? navColor : "var(--ink-4)",
                  border: `1.5px solid ${mode === "auto" ? navColor : "var(--border)"}`,
                }}>
          Automated (TradingView Formula)
        </button>
        <button onClick={() => setMode("manual")}
                className="h-[36px] px-4 rounded-[10px] text-[12px] font-semibold transition-all cursor-pointer"
                style={{
                  background: mode === "manual" ? `color-mix(in oklab, ${navColor} 10%, transparent)` : "var(--bg)",
                  color: mode === "manual" ? navColor : "var(--ink-4)",
                  border: `1.5px solid ${mode === "manual" ? navColor : "var(--border)"}`,
                }}>
          Manual Override
        </button>
      </div>

      {/* Manual ATR inputs */}
      {mode === "manual" && (
        <div className="mb-5">
          <div className="text-[12px] mb-3" style={{ color: "var(--ink-3)" }}>
            Enter the ATR% (21S) value directly from your TradingView Table:
          </div>
          <div className="grid grid-cols-4 gap-3 mb-4">
            {positions.map(t => (
              <div key={t.trade_id}>
                <label className="block text-[10px] uppercase tracking-[0.08em] font-semibold mb-1" style={{ color: "var(--ink-4)" }}>
                  {t.ticker} ATR%
                </label>
                <input type="number" value={manualAtr[t.ticker] || "5.0"}
                       onChange={e => setManualAtr({ ...manualAtr, [t.ticker]: e.target.value })}
                       step="0.1"
                       className="w-full h-[38px] px-3 rounded-[10px] text-[12px] outline-none"
                       style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: "var(--font-jetbrains), monospace" }} />
              </div>
            ))}
          </div>
          <button onClick={computeManualHeat}
                  className="h-[38px] px-6 rounded-[10px] text-[12px] font-semibold text-white transition-all hover:brightness-110"
                  style={{ background: "#6366f1" }}>
            Calculate Heat
          </button>
        </div>
      )}

      {fetching && (
        <div className="mb-5 flex items-center gap-2 px-4 py-2.5 rounded-[10px] text-[12px]"
             style={{ background: "color-mix(in oklab, #1e40af 10%, var(--surface))", color: "#3b82f6", border: "1px solid color-mix(in oklab, #1e40af 30%, var(--border))" }}>
          <span className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
          Fetching ATR data for {positions.length} positions...
        </div>
      )}

      {/* KPI tiles */}
      {heatData.length > 0 && (
        <>
          <div className="grid grid-cols-3 gap-3.5 mb-6">
            <KPITile
              label="TOTAL PORTFOLIO HEAT"
              value={`${totalHeat.toFixed(2)}%`}
              sub={`Target < ${HEAT_THRESHOLD}%`}
              gradient={heatOk ? "linear-gradient(135deg, #10b981, #34d399)" : "linear-gradient(135deg, #e5484d, #f87171)"}
            />
            <KPITile
              label="AVG STOCK VOLATILITY"
              value={`${avgVol.toFixed(2)}%`}
              sub={failedCount > 0
                ? `${okRows.length} of ${heatData.length} positions`
                : `${heatData.length} positions`}
              gradient="linear-gradient(135deg, #f97316, #fb923c)"
            />
            <KPITile
              label="EQUITY BASIS"
              value={formatCurrency(equity, { decimals: 0 })}
              sub=""
              gradient="linear-gradient(135deg, #1e40af, #3b82f6)"
            />
          </div>

          {/* Heat table */}
          <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
            <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
              <span className="text-[13px] font-semibold">Heat Breakdown</span>
              <span className="text-xs" style={{ color: "var(--ink-4)" }}>Sorted by heat contribution</span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-[12px]" style={{ borderCollapse: "separate", borderSpacing: 0 }}>
                <thead>
                  <tr>
                    {["Ticker", "Weight (%)", "ATR (21S) %", "Heat Contribution", "Heat Bar"].map(h => (
                      <th key={h} className="text-left text-[10px] uppercase tracking-[0.08em] font-semibold px-3 py-2.5 whitespace-nowrap"
                          style={{ color: "var(--ink-4)", background: "var(--surface-2)", borderBottom: "1px solid var(--border)" }}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {heatData.map((h, i) => {
                    // Heat bar: scale to max contribution for visual
                    const maxHeat = Math.max(...heatData.map(r => r.heat_contribution), 1);
                    const barWidth = (h.heat_contribution / maxHeat) * 100;
                    const barColor = h.heat_contribution > 3 ? "#ef4444" : h.heat_contribution > 1.5 ? "#f97316" : "#22c55e";
                    return (
                      <tr key={h.trade_id} style={{ borderBottom: i < heatData.length - 1 ? "1px solid var(--border)" : "none" }}
                          className="transition-colors"
                          onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-2)")}
                          onMouseLeave={e => (e.currentTarget.style.background = "transparent")}>
                        <td className="px-3 py-2.5 font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                          {h.ticker}
                          {h.atr_status === "failed" && (
                            <span title="Price data unavailable — refresh in a moment"
                                  style={{ marginLeft: 6, color: "#d97706" }}>⚠</span>
                          )}
                        </td>
                        <td className="px-3 py-2.5 text-right privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{h.weight_pct.toFixed(1)}%</td>
                        <td className="px-3 py-2.5 text-right" style={{ fontFamily: "var(--font-jetbrains), monospace", color: h.atr_status === "failed" ? "var(--ink-4)" : undefined }}>
                          {h.atr_status === "ok" ? `${h.atr_pct.toFixed(2)}%` : "—"}
                        </td>
                        <td className="px-3 py-2.5 text-right font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace", color: h.atr_status === "ok" ? barColor : "var(--ink-4)" }}>
                          {h.atr_status === "ok" ? `${h.heat_contribution.toFixed(2)}%` : "—"}
                        </td>
                        <td className="px-3 py-2.5 w-[140px]">
                          <div className="h-2.5 rounded-full overflow-hidden" style={{ background: "var(--bg)" }}>
                            <div className="h-full rounded-full transition-all" style={{ width: `${barWidth}%`, background: barColor }} />
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                  {/* Total row */}
                  <tr style={{ background: "var(--surface-2)" }}>
                    <td className="px-3 py-2.5 font-semibold text-[11px]" style={{ color: "var(--ink-3)" }}>TOTAL</td>
                    <td className="px-3 py-2.5 text-right font-semibold privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                      {heatData.reduce((a, h) => a + h.weight_pct, 0).toFixed(1)}%
                    </td>
                    <td className="px-3 py-2.5 text-right font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                      {avgVol.toFixed(2)}%
                    </td>
                    <td className="px-3 py-2.5 text-right font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace", color: heatOk ? "#08a86b" : "#e5484d" }}>
                      {totalHeat.toFixed(2)}%
                    </td>
                    <td className="px-3 py-2.5" />
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {positions.length === 0 && (
        <div className="text-center py-16 text-sm" style={{ color: "var(--ink-4)" }}>No open positions found to calculate heat.</div>
      )}
    </div>
  );
}
