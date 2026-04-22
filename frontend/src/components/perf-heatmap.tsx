"use client";

import { useState, useEffect, useMemo } from "react";
import { api, getActivePortfolio, type TradePosition, type TradeDetail, type JournalHistoryPoint } from "@/lib/api";

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

export function PerfHeatmap({ navColor }: { navColor: string }) {
  const [trades, setTrades] = useState<TradePosition[]>([]);
  const [openTrades, setOpenTrades] = useState<TradePosition[]>([]);
  const [journal, setJournal] = useState<JournalHistoryPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<"all" | "open" | "closed">("all");
  const [metricMode, setMetricMode] = useState<"return" | "rmult" | "impact">("return");
  const [selectedTrade, setSelectedTrade] = useState<string | null>(null);
  const [allDetails, setAllDetails] = useState<TradeDetail[]>([]);

  useEffect(() => {
    Promise.all([
      api.tradesClosed(getActivePortfolio(), 1000).catch(() => []),
      api.tradesOpen(getActivePortfolio()).catch(() => []),
      api.journalHistory(getActivePortfolio(), 0).catch(() => []),
      api.tradesRecent(getActivePortfolio(), 2000).catch(() => []),
    ]).then(([closed, open, jrnl, details]) => {
      setTrades(closed as TradePosition[]);
      setOpenTrades(open as TradePosition[]);
      setJournal(jrnl as JournalHistoryPoint[]);
      setAllDetails(details as TradeDetail[]);
      setLoading(false);
    });
  }, []);

  const heatData = useMemo(() => {
    // Combine and filter to 2026
    let all = [...openTrades, ...trades].filter(t => {
      const od = String(t.open_date || "").slice(0, 4);
      const cd = String(t.closed_date || "").slice(0, 4);
      const isOpen = (t.status || "").toUpperCase() === "OPEN";
      return od === "2026" || cd === "2026" || isOpen;
    });

    if (viewMode === "open") all = all.filter(t => (t.status || "").toUpperCase() === "OPEN");
    else if (viewMode === "closed") all = all.filter(t => (t.status || "").toUpperCase() === "CLOSED");

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
  }, [trades, openTrades, journal, viewMode, metricMode]);

  if (loading) return <div className="animate-pulse"><div className="h-[90px] rounded-[14px]" style={{ background: "var(--bg-2)" }} /></div>;

  // Metric config
  const cfg = metricMode === "return"
    ? { key: "retPct" as const, zMin: -7, zMax: 15, fmt: (v: number) => `${v.toFixed(1)}%`, label: "Return %" }
    : metricMode === "rmult"
    ? { key: "rMult" as const, zMin: -1.2, zMax: 3, fmt: (v: number) => `${v.toFixed(2)}R`, label: "R-Multiple" }
    : { key: "impact" as const, zMin: -1, zMax: 2, fmt: (v: number) => `${v.toFixed(2)}%`, label: "Account Impact %" };

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
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>CanSlim · 2026 trades</div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-4 mb-5 flex-wrap">
        <div className="flex p-0.5 rounded-[8px] gap-0.5" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
          {(["all", "open", "closed"] as const).map(m => (
            <button key={m} onClick={() => setViewMode(m)}
                    className="px-3 py-1 rounded-md text-[11px] font-medium transition-all capitalize"
                    style={{ background: viewMode === m ? "var(--surface)" : "transparent", color: viewMode === m ? "var(--ink)" : "var(--ink-4)" }}>
              {m === "all" ? "All 2026" : m === "open" ? "Open Only" : "Closed Only"}
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
      </div>

      {/* Heatmap grid */}
      {heatData.length > 0 ? (
        <div className="rounded-[14px] overflow-hidden mb-5 p-4" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
          <div className="grid gap-[4px]" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
            {heatData.map((d, i) => {
              const val = (d as any)[cfg.key] as number;
              const bg = heatColor(val, cfg.zMin, cfg.zMax);
              const textColor = Math.abs(val) > (cfg.zMax - cfg.zMin) * 0.3 ? "#fff" : "var(--ink)";
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

      {/* Slide-over panel */}
      {selectedTrade && (() => {
        const allCampaigns = [...openTrades, ...trades];
        const trade = allCampaigns.find(t => t.trade_id === selectedTrade);
        if (!trade) return null;
        const txns = allDetails.filter(d => d.trade_id === selectedTrade).sort((a, b) => String(a.date || "").localeCompare(String(b.date || "")));
        const buys = txns.filter(d => String(d.action).toUpperCase() === "BUY");
        const sells = txns.filter(d => String(d.action).toUpperCase() === "SELL");
        const pl = parseFloat(String(trade.realized_pl || 0));
        const ret = parseFloat(String(trade.return_pct || 0));
        const rb = parseFloat(String(trade.risk_budget || 0));
        const rMult = rb > 0 ? pl / rb : null;
        const isOpen = (trade.status || "").toUpperCase() === "OPEN";
        const avgEntry = parseFloat(String(trade.avg_entry || 0)) || (buys.length > 0 ? buys.reduce((a, d) => a + parseFloat(String(d.shares || 0)) * parseFloat(String(d.amount || 0)), 0) / buys.reduce((a, d) => a + parseFloat(String(d.shares || 0)), 0) : 0);
        const avgExit = parseFloat(String(trade.avg_exit || 0)) || (sells.length > 0 ? sells.reduce((a, d) => a + parseFloat(String(d.shares || 0)) * parseFloat(String(d.amount || 0)), 0) / sells.reduce((a, d) => a + parseFloat(String(d.shares || 0)), 0) : 0);
        const totalShares = trade.shares || buys.reduce((a, d) => a + parseFloat(String(d.shares || 0)), 0);
        const mono = "var(--font-jetbrains), monospace";

        return (
          <div className="fixed inset-0 z-50 flex justify-end" onClick={() => setSelectedTrade(null)}>
            {/* Backdrop */}
            <div className="absolute inset-0" style={{ background: "rgba(0,0,0,0.3)" }} />
            {/* Panel */}
            <div className="relative w-[480px] h-full overflow-y-auto" style={{ background: "var(--surface)", boxShadow: "-4px 0 20px rgba(0,0,0,0.1)", animation: "slide-in-right 0.2s ease-out" }}
                 onClick={e => e.stopPropagation()}>
              {/* Header */}
              <div className="sticky top-0 z-10 flex items-center justify-between px-6 py-4" style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)" }}>
                <div>
                  <div className="text-[18px] font-bold" style={{ fontFamily: mono }}>{trade.ticker}</div>
                  <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>{trade.trade_id} · {trade.status}</div>
                </div>
                <div className="text-right">
                  <div className="text-[20px] font-extrabold privacy-mask" style={{ fontFamily: mono, color: pctColor(pl) }}>
                    ${pl >= 0 ? "+" : ""}{pl.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                  </div>
                  <button onClick={() => setSelectedTrade(null)} className="text-[11px] mt-1" style={{ color: "var(--ink-4)" }}>Close ×</button>
                </div>
              </div>

              <div className="p-6 flex flex-col gap-5">
                {/* Flight Deck — clean layout */}
                <div>
                  <div className="flex items-baseline justify-between mb-4">
                    <div>
                      <div className="text-[12px] font-medium" style={{ color: "var(--ink-3)" }}>{trade.rule || ""}</div>
                      <div className="text-[12px] font-medium" style={{ color: "var(--ink-3)" }}>
                        {String(trade.open_date || "").slice(0, 10)} → {String(trade.closed_date || "").slice(0, 10) || (isOpen ? "Active" : "—")}
                        {' · '}{totalShares} shares
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-5 gap-4 py-3" style={{ borderTop: "1px solid var(--border)", borderBottom: "1px solid var(--border)" }}>
                    <div>
                      <div className="text-[9px] uppercase font-semibold" style={{ color: "var(--ink-4)" }}>Entry</div>
                      <div className="text-[15px] font-bold mt-0.5 privacy-mask" style={{ fontFamily: mono }}>{avgEntry > 0 ? `$${avgEntry.toFixed(2)}` : "—"}</div>
                    </div>
                    <div>
                      <div className="text-[9px] uppercase font-semibold" style={{ color: "var(--ink-4)" }}>Exit</div>
                      <div className="text-[15px] font-bold mt-0.5 privacy-mask" style={{ fontFamily: mono, color: isOpen ? "#08a86b" : "var(--ink)" }}>{avgExit > 0 ? `$${avgExit.toFixed(2)}` : isOpen ? "Active" : "—"}</div>
                    </div>
                    <div>
                      <div className="text-[9px] uppercase font-semibold" style={{ color: "var(--ink-4)" }}>P&L</div>
                      <div className="text-[15px] font-bold mt-0.5 privacy-mask" style={{ fontFamily: mono, color: pctColor(pl) }}>${pl >= 0 ? "+" : ""}{pl.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
                    </div>
                    <div>
                      <div className="text-[9px] uppercase font-semibold" style={{ color: "var(--ink-4)" }}>Return</div>
                      <div className="text-[15px] font-bold mt-0.5" style={{ fontFamily: mono, color: pctColor(ret || (avgExit - avgEntry)) }}>
                        {ret !== 0 ? `${ret >= 0 ? "+" : ""}${ret.toFixed(1)}%` : avgEntry > 0 && avgExit > 0 ? `${(((avgExit - avgEntry) / avgEntry) * 100).toFixed(1)}%` : "—"}
                      </div>
                    </div>
                    <div>
                      <div className="text-[9px] uppercase font-semibold" style={{ color: "var(--ink-4)" }}>R-Multiple</div>
                      <div className="text-[15px] font-bold mt-0.5" style={{ fontFamily: mono }}>{rMult != null ? `${rMult.toFixed(2)}R` : "—"}</div>
                    </div>
                  </div>
                </div>

                {/* Transaction Trail */}
                {txns.length > 0 && (
                  <div>
                    <div className="text-[12px] font-semibold mb-2">Transaction Trail — {buys.length} buy(s) · {sells.length} sell(s)</div>
                    <div className="rounded-[8px] overflow-hidden" style={{ border: "1px solid var(--border)" }}>
                      <table className="w-full text-[10px]" style={{ borderCollapse: "collapse" }}>
                        <thead><tr>
                          {["Date", "Action", "Shares", "Price", "Value", "Rule"].map(h => (
                            <th key={h} className="text-left px-2.5 py-1.5 text-[9px] uppercase font-semibold"
                                style={{ color: "var(--ink-4)", background: "var(--surface-2)", borderBottom: "1px solid var(--border)" }}>{h}</th>
                          ))}
                        </tr></thead>
                        <tbody>{txns.map((tx, j) => {
                          const isSell = String(tx.action).toUpperCase() === "SELL";
                          const shs = parseFloat(String(tx.shares || 0));
                          const px = parseFloat(String(tx.amount || 0));
                          return (
                            <tr key={j} style={{ borderBottom: "1px solid var(--border)" }}>
                              <td className="px-2.5 py-1.5" style={{ fontFamily: mono, color: "var(--ink-4)", fontSize: 9 }}>{String(tx.date || "").slice(0, 16)}</td>
                              <td className="px-2.5 py-1.5">
                                <span className="px-1.5 py-0.5 rounded text-[9px] font-bold"
                                      style={{ background: `color-mix(in oklab, ${isSell ? "#e5484d" : "#08a86b"} 12%, var(--surface))`, color: isSell ? "#e5484d" : "#08a86b" }}>
                                  {tx.action}
                                </span>
                              </td>
                              <td className="px-2.5 py-1.5" style={{ fontFamily: mono, color: isSell ? "#e5484d" : "var(--ink)" }}>{isSell ? -shs : shs}</td>
                              <td className="px-2.5 py-1.5 privacy-mask" style={{ fontFamily: mono }}>${px.toFixed(2)}</td>
                              <td className="px-2.5 py-1.5 privacy-mask" style={{ fontFamily: mono }}>${(shs * px).toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                              <td className="px-2.5 py-1.5 text-[9px]" style={{ color: "var(--ink-3)" }}>{tx.rule || ""}</td>
                            </tr>
                          );
                        })}</tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Notes */}
                {(trade.buy_notes || (trade as any).sell_notes) && (
                  <div>
                    <div className="text-[12px] font-semibold mb-2">Notes</div>
                    {trade.buy_notes && (
                      <div className="p-3 rounded-[8px] mb-2 text-[11px]" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
                        <span className="font-semibold" style={{ color: "var(--ink-4)" }}>Entry:</span> {trade.buy_notes}
                      </div>
                    )}
                    {(trade as any).sell_notes && (
                      <div className="p-3 rounded-[8px] text-[11px]" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
                        <span className="font-semibold" style={{ color: "var(--ink-4)" }}>Exit:</span> {(trade as any).sell_notes}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        );
      })()}
    </div>
  );
}

function pctColor(v: number) { return v > 0 ? "#08a86b" : v < 0 ? "#e5484d" : "var(--ink-3)"; }
