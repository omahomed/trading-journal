"use client";

// TradeOverviewSidecar — the fixed slide-in trade drill-in panel
// originally built into Performance Heat Map. Now shared with Campaign
// Review's right-click context menu (2026-07-25). Shows:
//   - Header: ticker, trade_id, status, P&L, close button
//   - Flight Deck grid: Entry / Exit / P&L / Return / R-Multiple
//   - Transaction Trail: date / action / shares / price / value / rule
//   - Per-Lot Excursion: MAE/MFE for each BUY lot (equity only, lazy-
//     loaded on mount via /api/trades/{trade_id}/lot-excursions)
//   - Notes: Entry (buy_notes) + Exit (sell_notes) when present
//
// Callers pass the trade summary + full details list (the component
// filters details to this trade_id itself). Options are handled via
// the standard is-option predicate — multiplier folded into value,
// unit label flipped to "Contracts".

import { useEffect, useState } from "react";
import type { TradePosition, TradeDetail, LotExcursion } from "@/lib/api";
import { api } from "@/lib/api";
import { formatCurrency } from "@/lib/format";

const mono = "var(--font-jetbrains), monospace";

function pctColor(v: number): string {
  return v > 0 ? "#08a86b" : v < 0 ? "#e5484d" : "var(--ink-3)";
}

// Per-lot MAE/MFE table. Highlights the worst MAE in red (deepest
// drawdown among the campaign's lots) — visual anchor for "which
// add-on tested you the most?" Rows sorted trx_id-natural (B1 < A1 <
// A2). Error rows collapse to a single "—" placeholder cell so a bad
// lot doesn't hide the healthy ones.
function LotExcursionTable({ rows }: { rows: LotExcursion[] }) {
  const sorted = [...rows].sort((a, b) => (a.trx_id || "").localeCompare(b.trx_id || ""));
  // Worst MAE = most negative pct across success rows. Used to red-
  // highlight the one lot that took the deepest hit from its fill.
  const worstMae = sorted.reduce<number | null>((min, r) => {
    if (r.error !== null || r.mae_pct == null) return min;
    return min === null || r.mae_pct < min ? r.mae_pct : min;
  }, null);

  return (
    <div className="rounded-[8px] overflow-hidden" data-testid="lot-excursion-table"
         style={{ border: "1px solid var(--border)" }}>
      <table className="w-full text-[10px]" style={{ borderCollapse: "collapse" }}>
        <thead><tr>
          {["Lot", "Fill Date", "Fill $", "MAE %", "× ATR", "MFE %", "Days Held"].map(h => (
            <th key={h} className="text-left px-2.5 py-1.5 text-[9px] uppercase font-semibold"
                style={{ color: "var(--ink-4)", background: "var(--surface-2)", borderBottom: "1px solid var(--border)" }}>
              {h}
            </th>
          ))}
        </tr></thead>
        <tbody>{sorted.map((r) => {
          const isB = (r.trx_id || "").toUpperCase().startsWith("B");
          const rowKey = `${r.trade_id}-${r.trx_id}`;
          if (r.error !== null) {
            return (
              <tr key={rowKey} style={{ borderBottom: "1px solid var(--border)" }}
                  data-testid={`lot-excursion-row-${r.trx_id}`}>
                <td className="px-2.5 py-1.5 font-semibold" style={{ fontFamily: mono, color: isB ? "var(--ink)" : "var(--ink-3)" }}>
                  {r.trx_id}
                </td>
                <td className="px-2.5 py-1.5 text-[10px]" style={{ color: "var(--ink-4)" }} colSpan={6}>
                  {r.fill_date ?? "—"} · {r.error}
                </td>
              </tr>
            );
          }
          const isWorstMae = worstMae != null && r.mae_pct === worstMae && r.mae_pct < 0;
          return (
            <tr key={rowKey} style={{ borderBottom: "1px solid var(--border)" }}
                data-testid={`lot-excursion-row-${r.trx_id}`}>
              <td className="px-2.5 py-1.5 font-semibold"
                  style={{ fontFamily: mono, color: isB ? "var(--ink)" : "var(--ink-3)" }}>
                {r.trx_id}
              </td>
              <td className="px-2.5 py-1.5 text-[10px]"
                  style={{ fontFamily: mono, color: "var(--ink-4)" }}>
                {r.fill_date}
              </td>
              <td className="px-2.5 py-1.5 privacy-mask"
                  style={{ fontFamily: mono }}>
                {r.fill_price != null ? formatCurrency(r.fill_price) : "—"}
              </td>
              <td className="px-2.5 py-1.5"
                  style={{
                    fontFamily: mono,
                    color: isWorstMae ? "#e5484d" : (r.mae_pct != null && r.mae_pct < 0 ? "#e5484d" : "var(--ink-3)"),
                    fontWeight: isWorstMae ? 700 : 400,
                  }}>
                {r.mae_pct != null ? `${r.mae_pct.toFixed(2)}%` : "—"}
              </td>
              <td className="px-2.5 py-1.5"
                  style={{ fontFamily: mono, color: "var(--ink-4)" }}>
                {r.mae_atr_multiple != null ? `${r.mae_atr_multiple.toFixed(2)}×` : "—"}
              </td>
              <td className="px-2.5 py-1.5"
                  style={{
                    fontFamily: mono,
                    color: r.mfe_pct != null && r.mfe_pct > 0 ? "#08a86b" : "var(--ink-3)",
                  }}>
                {r.mfe_pct != null ? `+${r.mfe_pct.toFixed(2)}%` : "—"}
              </td>
              <td className="px-2.5 py-1.5"
                  style={{ fontFamily: mono, color: "var(--ink-4)" }}>
                {r.days_held ?? "—"}
              </td>
            </tr>
          );
        })}</tbody>
      </table>
    </div>
  );
}

export function TradeOverviewSidecar({ trade, details, onClose, portfolio }: {
  trade: TradePosition;
  details: TradeDetail[];
  onClose: () => void;
  // Portfolio for the per-lot excursions fetch. Optional — falls
  // back to getActivePortfolio() inside the api client when omitted.
  portfolio?: string;
}) {
  // Per-lot MAE/MFE — fetched on mount, cheap (one yfinance call per
  // campaign, shared across lots). Options campaigns skip the fetch
  // entirely — the API would return empty lots (upstream STOCK-only
  // filter) but no reason to burn a request just to hide the result.
  const [lotExcursions, setLotExcursions] = useState<LotExcursion[] | null>(null);
  const [lotExcursionsErr, setLotExcursionsErr] = useState<string | null>(null);
  const skipLotExcursions = String((trade as { instrument_type?: string }).instrument_type || "").toUpperCase() === "OPTION"
    || /^\S+\s+\d{6}\s+\$[0-9.]+(C|P)$/.test(String(trade.ticker || ""));
  useEffect(() => {
    if (skipLotExcursions) return;
    let cancelled = false;
    setLotExcursions(null);
    setLotExcursionsErr(null);
    api.lotExcursions(trade.trade_id, portfolio)
      .then(resp => { if (!cancelled) setLotExcursions(resp.lots || []); })
      .catch(err => {
        if (!cancelled) setLotExcursionsErr(err?.message || "Failed to load per-lot excursion");
      });
    return () => { cancelled = true; };
  }, [trade.trade_id, portfolio, skipLotExcursions]);
  const txns = details
    .filter(d => d.trade_id === trade.trade_id)
    .sort((a, b) => String(a.date || "").localeCompare(String(b.date || "")));
  const buys = txns.filter(d => String(d.action).toUpperCase() === "BUY");
  const sells = txns.filter(d => String(d.action).toUpperCase() === "SELL");
  const pl = parseFloat(String(trade.realized_pl || 0));
  const ret = parseFloat(String((trade as { return_pct?: number | string }).return_pct || 0));
  const rb = parseFloat(String((trade as { risk_budget?: number | string }).risk_budget || 0));
  const rMult = rb > 0 ? pl / rb : null;
  const isOpen = (trade.status || "").toUpperCase() === "OPEN";
  // Migration 016 option formatting. Notional = shares × price × 100;
  // unit label is "Contracts" not "Shares".
  const isOption = String((trade as { instrument_type?: string }).instrument_type || "").toUpperCase() === "OPTION"
    || /^\S+\s+\d{6}\s+\$[0-9.]+(C|P)$/.test(String(trade.ticker || ""));
  const multiplier = isOption
    ? Math.max(parseFloat(String((trade as { multiplier?: number | string }).multiplier || 0)) || 100, 1)
    : 1;
  const unitLabel = isOption ? "Contracts" : "Shares";
  const avgEntry = parseFloat(String(trade.avg_entry || 0)) || (buys.length > 0
    ? buys.reduce((a, d) => a + parseFloat(String(d.shares || 0)) * parseFloat(String(d.amount || 0)), 0)
        / buys.reduce((a, d) => a + parseFloat(String(d.shares || 0)), 0)
    : 0);
  const avgExit = parseFloat(String((trade as { avg_exit?: number | string }).avg_exit || 0)) || (sells.length > 0
    ? sells.reduce((a, d) => a + parseFloat(String(d.shares || 0)) * parseFloat(String(d.amount || 0)), 0)
        / sells.reduce((a, d) => a + parseFloat(String(d.shares || 0)), 0)
    : 0);
  const totalShares = trade.shares || buys.reduce((a, d) => a + parseFloat(String(d.shares || 0)), 0);
  const buyNotes = (trade as { buy_notes?: string }).buy_notes;
  const sellNotes = (trade as { sell_notes?: string }).sell_notes;

  return (
    <div className="fixed inset-0 z-50 flex justify-end" onClick={onClose}
         data-testid="trade-overview-sidecar">
      {/* Backdrop */}
      <div className="absolute inset-0" style={{ background: "rgba(0,0,0,0.3)" }} />
      {/* Panel */}
      <div className="relative w-[480px] h-full overflow-y-auto"
           style={{ background: "var(--surface)", boxShadow: "-4px 0 20px rgba(0,0,0,0.1)", animation: "slide-in-right 0.2s ease-out" }}
           onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="sticky top-0 z-10 flex items-center justify-between px-6 py-4"
             style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)" }}>
          <div>
            <div className="text-[18px] font-bold" style={{ fontFamily: mono }}>{trade.ticker}</div>
            <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>
              {trade.trade_id} · {trade.status}
            </div>
          </div>
          <div className="text-right">
            <div className="text-[20px] font-extrabold privacy-mask"
                 style={{ fontFamily: mono, color: pctColor(pl) }}>
              {formatCurrency(pl, { showSign: true, decimals: 0 })}
            </div>
            <button onClick={onClose} className="text-[11px] mt-1" style={{ color: "var(--ink-4)" }}>
              Close ×
            </button>
          </div>
        </div>

        <div className="p-6 flex flex-col gap-5">
          {/* Flight Deck */}
          <div>
            <div className="flex items-baseline justify-between mb-4">
              <div>
                <div className="text-[12px] font-medium" style={{ color: "var(--ink-3)" }}>
                  {trade.rule || ""}
                </div>
                <div className="text-[12px] font-medium" style={{ color: "var(--ink-3)" }}>
                  {String(trade.open_date || "").slice(0, 10)} → {String((trade as { closed_date?: string }).closed_date || "").slice(0, 10) || (isOpen ? "Active" : "—")}
                  {' · '}{totalShares} {unitLabel.toLowerCase()}
                </div>
              </div>
            </div>

            <div className="grid grid-cols-5 gap-4 py-3"
                 style={{ borderTop: "1px solid var(--border)", borderBottom: "1px solid var(--border)" }}>
              <div>
                <div className="text-[9px] uppercase font-semibold" style={{ color: "var(--ink-4)" }}>Entry</div>
                <div className="text-[15px] font-bold mt-0.5 privacy-mask" style={{ fontFamily: mono }}>
                  {avgEntry > 0 ? formatCurrency(avgEntry) : "—"}
                </div>
              </div>
              <div>
                <div className="text-[9px] uppercase font-semibold" style={{ color: "var(--ink-4)" }}>Exit</div>
                <div className="text-[15px] font-bold mt-0.5 privacy-mask"
                     style={{ fontFamily: mono, color: isOpen ? "#08a86b" : "var(--ink)" }}>
                  {avgExit > 0 ? formatCurrency(avgExit) : isOpen ? "Active" : "—"}
                </div>
              </div>
              <div>
                <div className="text-[9px] uppercase font-semibold" style={{ color: "var(--ink-4)" }}>P&L</div>
                <div className="text-[15px] font-bold mt-0.5 privacy-mask"
                     style={{ fontFamily: mono, color: pctColor(pl) }}>
                  {formatCurrency(pl, { showSign: true, decimals: 0 })}
                </div>
              </div>
              <div>
                <div className="text-[9px] uppercase font-semibold" style={{ color: "var(--ink-4)" }}>Return</div>
                <div className="text-[15px] font-bold mt-0.5"
                     style={{ fontFamily: mono, color: pctColor(ret || (avgExit - avgEntry)) }}>
                  {ret !== 0
                    ? `${ret >= 0 ? "+" : ""}${ret.toFixed(1)}%`
                    : avgEntry > 0 && avgExit > 0
                      ? `${(((avgExit - avgEntry) / avgEntry) * 100).toFixed(1)}%`
                      : "—"}
                </div>
              </div>
              <div>
                <div className="text-[9px] uppercase font-semibold" style={{ color: "var(--ink-4)" }}>R-Multiple</div>
                <div className="text-[15px] font-bold mt-0.5" style={{ fontFamily: mono }}>
                  {rMult != null ? `${rMult.toFixed(2)}R` : "—"}
                </div>
              </div>
            </div>
          </div>

          {/* Transaction Trail */}
          {txns.length > 0 && (
            <div>
              <div className="text-[12px] font-semibold mb-2">
                Transaction Trail — {buys.length} buy(s) · {sells.length} sell(s)
              </div>
              <div className="rounded-[8px] overflow-hidden" style={{ border: "1px solid var(--border)" }}>
                <table className="w-full text-[10px]" style={{ borderCollapse: "collapse" }}>
                  <thead><tr>
                    {["Date", "Action", unitLabel, "Price", "Value", "Rule"].map(h => (
                      <th key={h} className="text-left px-2.5 py-1.5 text-[9px] uppercase font-semibold"
                          style={{ color: "var(--ink-4)", background: "var(--surface-2)", borderBottom: "1px solid var(--border)" }}>
                        {h}
                      </th>
                    ))}
                  </tr></thead>
                  <tbody>{txns.map((tx, j) => {
                    const isSell = String(tx.action).toUpperCase() === "SELL";
                    const shs = parseFloat(String(tx.shares || 0));
                    const px = parseFloat(String(tx.amount || 0));
                    return (
                      <tr key={j} style={{ borderBottom: "1px solid var(--border)" }}>
                        <td className="px-2.5 py-1.5" style={{ fontFamily: mono, color: "var(--ink-4)", fontSize: 9 }}>
                          {String(tx.date || "").slice(0, 16)}
                        </td>
                        <td className="px-2.5 py-1.5">
                          <span className="px-1.5 py-0.5 rounded text-[9px] font-bold"
                                style={{
                                  background: `color-mix(in oklab, ${isSell ? "#e5484d" : "#08a86b"} 12%, var(--surface))`,
                                  color: isSell ? "#e5484d" : "#08a86b",
                                }}>
                            {tx.action}
                          </span>
                        </td>
                        <td className="px-2.5 py-1.5" style={{ fontFamily: mono, color: isSell ? "#e5484d" : "var(--ink)" }}>
                          {isSell ? -shs : shs}
                        </td>
                        <td className="px-2.5 py-1.5 privacy-mask" style={{ fontFamily: mono }}>
                          {formatCurrency(px)}
                        </td>
                        <td className="px-2.5 py-1.5 privacy-mask" style={{ fontFamily: mono }}>
                          {formatCurrency(shs * px * multiplier, { decimals: 0 })}
                        </td>
                        <td className="px-2.5 py-1.5 text-[9px]" style={{ color: "var(--ink-3)" }}>
                          {tx.rule || ""}
                        </td>
                      </tr>
                    );
                  })}</tbody>
                </table>
              </div>
            </div>
          )}

          {/* Per-Lot Excursion — MAE/MFE anchored to EACH lot's own
              fill. Answers "was this add-on well-timed?" the way the
              campaign-level MAE (anchored to B1) can't. Hidden for
              options campaigns (server returns empty lots). Loading
              state is silent; error state is a discreet one-liner. */}
          {!isOption && (
            <div>
              <div className="text-[12px] font-semibold mb-2">
                Per-Lot Excursion
                <span className="ml-2 text-[10px] font-normal" style={{ color: "var(--ink-4)" }}>
                  MAE / MFE per BUY, anchored to that lot's own fill
                </span>
              </div>
              {lotExcursionsErr ? (
                <div className="text-[11px] px-3 py-2 rounded-[8px]"
                     style={{ color: "var(--ink-4)", background: "var(--bg)", border: "1px solid var(--border)" }}>
                  {lotExcursionsErr}
                </div>
              ) : lotExcursions === null ? (
                <div className="text-[11px] px-3 py-2 rounded-[8px]"
                     style={{ color: "var(--ink-4)", background: "var(--bg)", border: "1px solid var(--border)" }}>
                  Loading …
                </div>
              ) : lotExcursions.length === 0 ? (
                <div className="text-[11px] px-3 py-2 rounded-[8px]"
                     style={{ color: "var(--ink-4)", background: "var(--bg)", border: "1px solid var(--border)" }}>
                  No per-lot excursion data (options-only campaign or no
                  qualifying BUYs).
                </div>
              ) : (
                <LotExcursionTable rows={lotExcursions} />
              )}
            </div>
          )}

          {/* Notes */}
          {(buyNotes || sellNotes) && (
            <div>
              <div className="text-[12px] font-semibold mb-2">Notes</div>
              {buyNotes && (
                <div className="p-3 rounded-[8px] mb-2 text-[11px]"
                     style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
                  <span className="font-semibold" style={{ color: "var(--ink-4)" }}>Entry:</span> {buyNotes}
                </div>
              )}
              {sellNotes && (
                <div className="p-3 rounded-[8px] text-[11px]"
                     style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
                  <span className="font-semibold" style={{ color: "var(--ink-4)" }}>Exit:</span> {sellNotes}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
