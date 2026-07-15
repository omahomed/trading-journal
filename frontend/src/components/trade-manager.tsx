"use client";

import { useState, useEffect, useRef } from "react";
import { api, getActivePortfolio, type TradePosition, type TradeDetail } from "@/lib/api";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";
import { SELL_RULE_LABELS as SELL_RULES } from "@/lib/trade-rules";
import { computeEnrichedPositions } from "@/lib/positions";
import { SR8TrimCalculator } from "./sr8-trim-calculator";

// BUY_RULES imported from @/lib/trade-rules — see file for the ordered
// list. Local alias keeps existing call sites unchanged.
// (SELL_RULE_LABELS as SELL_RULES already imported above.)
import { BUY_RULE_LABELS as BUY_RULES } from "@/lib/trade-rules";

type Tab = "stops" | "edit" | "delete" | "export" | "sr8-trim";

const TABS: { key: Tab; label: string; icon: string }[] = [
  { key: "stops", label: "Stop Loss Adjustment", icon: "🛡️" },
  { key: "edit", label: "Edit Transaction", icon: "📝" },
  { key: "delete", label: "Delete Trade", icon: "🗑️" },
  { key: "export", label: "Export", icon: "📥" },
  { key: "sr8-trim", label: "SR8 Trim", icon: "✂️" },
];

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-[10px] uppercase tracking-[0.10em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>{label}</label>
      {children}
    </div>
  );
}

const inputCls = "w-full h-[42px] px-3.5 rounded-[10px] text-[13px] outline-none";
const inputStyle: React.CSSProperties = {
  background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)",
  fontFamily: "var(--font-jetbrains), monospace",
};

function SearchSelect({ value, onChange, options, placeholder }: {
  value: string; onChange: (v: string) => void; options: readonly string[]; placeholder?: string;
}) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const handler = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);
  const filtered = search ? options.filter(o => o.toLowerCase().includes(search.toLowerCase())) : options;
  return (
    <div ref={ref} className="relative">
      <button type="button" onClick={() => setOpen(!open)} className={inputCls + " flex items-center justify-between text-left cursor-pointer"} style={{ ...inputStyle, fontFamily: "inherit" }}>
        <span style={{ opacity: value ? 1 : 0.5 }}>{value || placeholder || "Select..."}</span>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--ink-4)" strokeWidth="2"><path d="M6 9l6 6 6-6"/></svg>
      </button>
      {open && (
        <div className="absolute z-50 mt-1 w-full rounded-[10px] overflow-hidden shadow-lg" style={{ background: "var(--surface)", border: "1px solid var(--border)", maxHeight: 520 }}>
          <div className="p-2" style={{ borderBottom: "1px solid var(--border)" }}>
            <input type="text" value={search} onChange={e => setSearch(e.target.value)} placeholder="Type to search..." autoFocus
                   className="w-full h-[34px] px-3 rounded-[8px] text-[12px] outline-none" style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }} />
          </div>
          <div className="overflow-y-auto" style={{ maxHeight: 460 }}>
            {filtered.map(o => (
              <button key={o} type="button" onClick={() => { onChange(o); setOpen(false); setSearch(""); }}
                      className="w-full text-left px-3 py-2 text-[12px] transition-colors hover:brightness-95 cursor-pointer"
                      style={{ background: o === value ? "var(--surface-2)" : "transparent", color: "var(--ink)" }}>{o}</button>
            ))}
            {filtered.length === 0 && <div className="px-3 py-3 text-[12px] text-center" style={{ color: "var(--ink-4)" }}>No matches</div>}
          </div>
        </div>
      )}
    </div>
  );
}

export function TradeManager({ navColor, initialTab, onTabConsumed }: { navColor: string; initialTab?: string; onTabConsumed?: () => void }) {
  const [tab, setTab] = useState<Tab>((initialTab as Tab) || "stops");

  useEffect(() => {
    if (initialTab && ["stops", "edit", "delete", "export", "sr8-trim"].includes(initialTab)) {
      setTab(initialTab as Tab);
      onTabConsumed?.();
    }
  }, [initialTab, onTabConsumed]);

  // Live prices for the SR8 Trim tab. Fetched once alongside the main
  // load and reused — the other tabs don't need them, so we keep this
  // separate. computeEnrichedPositions runs inside the SR8 tab body
  // (see below) so live prices feed through to current_price.
  const [livePrices, setLivePrices] = useState<Record<string, number>>({});
  const [openTrades, setOpenTrades] = useState<TradePosition[]>([]);
  const [allTrades, setAllTrades] = useState<TradePosition[]>([]);
  const [details, setDetails] = useState<TradeDetail[]>([]);
  const [loading, setLoading] = useState(true);

  // Stop Loss tab
  const [selectedStop, setSelectedStop] = useState("");
  const [newStop, setNewStop] = useState("");
  const [stopSaving, setStopSaving] = useState(false);
  const [stopResult, setStopResult] = useState<{ ok: boolean; msg: string } | null>(null);
  // Ladder edit — per-leg share counts on the selected trade's B1
  // ladder. Seeded from the persisted ladder when a trade is selected;
  // reset when selection changes. Saving PUTs to /update-ladder.
  const [ladderEdit, setLadderEdit] = useState<[number, number, number] | null>(null);

  // Edit tab
  const [editTicker, setEditTicker] = useState("");
  const [editTradeId, setEditTradeId] = useState("");
  const [editTxIdx, setEditTxIdx] = useState(-1);
  const [editFields, setEditFields] = useState<Record<string, string>>({});
  // Migration 047: confluence rules for the transaction being edited.
  // Populated from tx.rules[1..] when a transaction is selected. Empty
  // for SELL rows (multi-select is buy-only) and for pre-047 rows.
  const [editConfluence, setEditConfluence] = useState<string[]>([]);
  const [editConfluenceQuery, setEditConfluenceQuery] = useState("");
  const [editConfluenceOpen, setEditConfluenceOpen] = useState(false);
  const [editSaving, setEditSaving] = useState(false);
  const [editResult, setEditResult] = useState<{ ok: boolean; msg: string } | null>(null);

  // Delete tab
  const [deleteTradeId, setDeleteTradeId] = useState("");
  const [deleteConfirm, setDeleteConfirm] = useState("");
  const [deleteResult, setDeleteResult] = useState<{ ok: boolean; msg: string } | null>(null);

  // Export tab
  const [exportType, setExportType] = useState<"summary" | "details">("summary");
  const [exportStatus, setExportStatus] = useState<"all" | "open" | "closed">("all");
  const [exportFrom, setExportFrom] = useState("");
  const [exportTo, setExportTo] = useState("");

  useEffect(() => {
    Promise.all([
      api.tradesOpen(getActivePortfolio()).catch((err) => {
        log.error("trade-manager", "tradesOpen fetch failed", err);
        return [];
      }),
      api.tradesClosed(getActivePortfolio(), 500).catch((err) => {
        log.error("trade-manager", "tradesClosed fetch failed", err);
        return [];
      }),
      api.tradesRecent(getActivePortfolio(), 500).catch((err) => {
        log.error("trade-manager", "tradesRecent fetch failed", err);
        return { details: [], lot_closures: [] };
      }),
    ]).then(([open, closed, det]) => {
      setOpenTrades(open as TradePosition[]);
      setAllTrades([...open as TradePosition[], ...closed as TradePosition[]]);
      setDetails(det.details);
      setLoading(false);
      // Fan out live-price fetch for the SR8 Trim tab. Independent
      // from the main load so a flaky prices endpoint doesn't keep
      // the page in the loading state.
      const tickers = (open as TradePosition[]).map((t) => t.ticker).filter(Boolean);
      if (tickers.length > 0) {
        api.batchPrices(tickers, getActivePortfolio())
          .then((prices) => setLivePrices(prices || {}))
          .catch((err) => log.error("trade-manager", "batchPrices fetch failed", err));
      }
    });
  }, []);

  // Seed the ladder-edit state from the selected trade's B1 ladder
  // whenever selection (or the underlying details) changes. Null when
  // B1 has no ladder — the Edit Ladder block hides entirely.
  useEffect(() => {
    const b1 = details.find(d => d.trade_id === selectedStop && String(d.action).toUpperCase() === "BUY");
    const raw = b1 ? (b1 as any).stop_ladder : null;
    if (!raw || typeof raw !== "object") { setLadderEdit(null); return; }
    const legs = (raw as { legs?: unknown }).legs;
    if (!Array.isArray(legs) || legs.length !== 3) { setLadderEdit(null); return; }
    const shares = legs.map(l => Math.max(0, Math.floor(Number((l as any).shares) || 0))) as [number, number, number];
    setLadderEdit(shares);
  }, [selectedStop, details]);

  if (loading) {
    return <div className="animate-pulse"><div className="h-[90px] rounded-[14px]" style={{ background: "var(--bg-2)" }} /></div>;
  }

  // Stop loss tab: details for selected position
  const stopTrade = openTrades.find(t => t.trade_id === selectedStop);
  const stopDetails = details.filter(d => d.trade_id === selectedStop && String(d.action).toUpperCase() === "BUY");
  // B1 = earliest BUY. Ladder is B1-only by convention. Parse defensively
  // — backend returns "" for NULL ladders after fillna in _df_to_records.
  const stopB1 = stopDetails.length > 0 ? stopDetails[0] : null;
  const stopB1Ladder = (() => {
    const raw = stopB1 ? (stopB1 as any).stop_ladder : null;
    if (!raw || typeof raw !== "object") return null;
    const legs = (raw as { legs?: unknown }).legs;
    if (!Array.isArray(legs) || legs.length !== 3) return null;
    const parsed = legs.map(l => {
      const leg = l as { pct?: unknown; shares?: unknown };
      return { pct: Number(leg.pct), shares: Number(leg.shares) };
    });
    if (parsed.some(l => !Number.isFinite(l.pct) || !Number.isFinite(l.shares))) return null;
    return { legs: parsed };
  })();
  const stopB1Price = stopB1 ? parseFloat(String(stopB1.amount || 0)) : 0;
  const stopB1Shares = stopB1 ? parseFloat(String(stopB1.shares || 0)) : 0;

  // Edit tab: filtered trades by ticker
  const editTrades = editTicker
    ? allTrades.filter(t => (t.ticker || "").toUpperCase().includes(editTicker.toUpperCase()))
    : [];
  const editTxns = details.filter(d => d.trade_id === editTradeId);
  const editTx = editTxIdx >= 0 && editTxIdx < editTxns.length ? editTxns[editTxIdx] : null;

  // Delete tab
  const deleteTrade = allTrades.find(t => t.trade_id === deleteTradeId);
  const deleteTxCount = details.filter(d => d.trade_id === deleteTradeId).length;

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          Trade <em className="italic" style={{ color: navColor }}>Manager</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
          Manage positions, update stops, and edit transactions
        </div>
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 mb-6 overflow-x-auto pb-0.5" style={{ borderBottom: "2px solid var(--border)" }}>
        {TABS.map(t => (
          <button key={t.key} onClick={() => setTab(t.key)}
                  className="px-4 py-2 text-[12px] font-medium whitespace-nowrap transition-all"
                  style={{
                    color: tab === t.key ? navColor : "var(--ink-4)",
                    borderBottom: tab === t.key ? `2px solid ${navColor}` : "2px solid transparent",
                    marginBottom: -2,
                  }}>
            {t.icon} {t.label}
          </button>
        ))}
      </div>

      {/* ═══════════ STOP LOSS ADJUSTMENT ═══════════ */}
      {tab === "stops" && (
        <div className="flex flex-col gap-5">
          <div className="text-[13px]" style={{ color: "var(--ink-3)" }}>
            Select a position and set a new stop loss for all open lots.
          </div>

          <Field label="Select Position">
            <SearchSelect
              value={selectedStop ? (() => { const t = openTrades.find(x => x.trade_id === selectedStop); return t ? `${t.ticker} | ${t.trade_id} (${t.shares} shs @ ${formatCurrency(parseFloat(String(t.avg_entry || 0)))})` : ""; })() : ""}
              onChange={(v) => { const id = v.split(" | ")[1]?.split(" (")[0]?.trim() || ""; setSelectedStop(id); setNewStop(""); }}
              options={openTrades.map(t => `${t.ticker} | ${t.trade_id} (${t.shares} shs @ ${formatCurrency(parseFloat(String(t.avg_entry || 0)))})`)}
              placeholder="Search positions..."
            />
          </Field>

          {stopTrade && (
            <>
              {/* Current lots */}
              <div className="rounded-[10px] overflow-hidden" style={{ border: "1px solid var(--border)" }}>
                <div className="px-4 py-2.5 text-[12px] font-semibold" style={{ background: "var(--surface-2)", borderBottom: "1px solid var(--border)" }}>
                  Current Lots — {stopTrade.ticker}
                </div>
                <table className="w-full text-[11px]" style={{ borderCollapse: "collapse" }}>
                  <thead>
                    <tr>
                      {["Trx ID", "Date", "Shares", "Entry", "Current Stop", "Notes"].map(h => (
                        <th key={h} className="text-left px-3 py-2 text-[9px] uppercase tracking-[0.06em] font-semibold"
                            style={{ color: "var(--ink-4)", background: "var(--surface-2)", borderBottom: "1px solid var(--border)" }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {stopDetails.map((d, i) => (
                      <tr key={i} style={{ borderBottom: i < stopDetails.length - 1 ? "1px solid var(--border)" : "none" }}>
                        <td className="px-3 py-2 font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace", fontSize: 10 }}>{d.trx_id || "—"}</td>
                        <td className="px-3 py-2" style={{ fontFamily: "var(--font-jetbrains), monospace", fontSize: 10, color: "var(--ink-4)" }}>{String(d.date || "").slice(0, 16)}</td>
                        <td className="px-3 py-2" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{d.shares}</td>
                        <td className="px-3 py-2 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{formatCurrency(parseFloat(String(d.amount || 0)))}</td>
                        <td className="px-3 py-2" style={{ fontFamily: "var(--font-jetbrains), monospace", color: parseFloat(String(d.stop_loss || 0)) > 0 ? "var(--ink)" : "var(--ink-4)" }}>
                          {parseFloat(String(d.stop_loss || 0)) > 0 ? formatCurrency(parseFloat(String(d.stop_loss))) : "—"}
                        </td>
                        <td className="px-3 py-2 text-[10px]" style={{ color: "var(--ink-4)" }}>{d.notes || ""}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Edit Ladder block — visible only when B1 has a persisted
                  stop_ladder. Percentages are locked (Phase 1 convention);
                  the user edits per-leg share counts. Sum must equal B1
                  shares. Below, the "Promote to Global Stop" flow (existing
                  New Stop Price + Update All Lots button) still works —
                  entering a single stop clears the ladder on save. */}
              {stopB1Ladder && ladderEdit && stopB1Price > 0 && (() => {
                const legSum = ladderEdit[0] + ladderEdit[1] + ladderEdit[2];
                const mismatch = legSum !== Math.floor(stopB1Shares);
                return (
                  <div className="rounded-[10px] overflow-hidden"
                       style={{ border: "1px solid color-mix(in oklab, #d97706 30%, var(--border))", background: "color-mix(in oklab, #d97706 4%, var(--surface))" }}>
                    <div className="px-4 py-2.5 text-[12px] font-semibold flex items-center justify-between"
                         style={{ background: "color-mix(in oklab, #d97706 8%, var(--surface))", borderBottom: "1px solid color-mix(in oklab, #d97706 20%, var(--border))", color: "#d97706" }}>
                      <span>🪜 Edit Scale-Out Ladder (B1 · locked at −3 / −5 / −7 %)</span>
                      <span className="text-[10px] font-normal" style={{ color: "var(--ink-4)" }}>
                        Prices track B1 entry {formatCurrency(stopB1Price)}
                      </span>
                    </div>
                    <div className="px-4 py-3 flex flex-col gap-2">
                      {([3, 5, 7] as const).map((pct, i) => (
                        <div key={pct} className="grid grid-cols-[40px_100px_1fr_1fr] items-center gap-3 text-[12px]"
                             style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                          <span className="font-semibold" style={{ color: "#d97706" }}>−{pct}%</span>
                          <span style={{ color: "var(--ink-3)" }}>{formatCurrency(stopB1Price * (1 - pct / 100))}</span>
                          <input type="number" min="0" step="1" value={ladderEdit[i]}
                                 onChange={e => {
                                   const v = Math.max(0, Math.floor(Number(e.target.value) || 0));
                                   setLadderEdit(prev => {
                                     if (!prev) return prev;
                                     const next: [number, number, number] = [...prev] as [number, number, number];
                                     next[i] = v;
                                     return next;
                                   });
                                 }}
                                 className={inputCls} style={{ ...inputStyle, height: "32px" }} />
                          <span className="text-right text-[11px]" style={{ color: "var(--ink-4)" }}>
                            loss if hit: −{formatCurrency(ladderEdit[i] * stopB1Price * (pct / 100), { decimals: 0 })}
                          </span>
                        </div>
                      ))}
                      <div className="flex items-center justify-between pt-1 text-[11px]"
                           style={{ borderTop: "1px dashed var(--border)", color: mismatch ? "#d97706" : "var(--ink-4)" }}>
                        <span>
                          Legs total <strong style={{ color: mismatch ? "#d97706" : "var(--ink)", fontFamily: "var(--font-jetbrains), monospace" }}>{legSum}</strong> shs
                          {mismatch ? ` (must equal ${Math.floor(stopB1Shares)})` : ""}
                        </span>
                        <button type="button"
                                disabled={stopSaving || mismatch}
                                onClick={async () => {
                                  setStopSaving(true);
                                  setStopResult(null);
                                  const r = await api.updateTradeLadder({
                                    portfolio: getActivePortfolio(),
                                    trade_id: selectedStop,
                                    stop_ladder: {
                                      legs: [
                                        { pct: 3, shares: ladderEdit[0] },
                                        { pct: 5, shares: ladderEdit[1] },
                                        { pct: 7, shares: ladderEdit[2] },
                                      ],
                                    },
                                  });
                                  setStopSaving(false);
                                  if (r.error) {
                                    setStopResult({ ok: false, msg: r.error });
                                  } else {
                                    setStopResult({ ok: true, msg: `Ladder updated on ${stopTrade?.ticker || selectedStop}` });
                                    // Reload so the edited ladder shows on subsequent renders.
                                    const det = await api.tradesRecent(getActivePortfolio(), 500);
                                    setDetails(det.details);
                                  }
                                }}
                                className="h-[30px] px-4 rounded-[8px] text-[11px] font-semibold text-white transition-all hover:brightness-110 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
                                style={{ background: "#d97706" }}>
                          {stopSaving ? "Saving…" : "Save ladder"}
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })()}

              {/* New stop input */}
              <div className="grid grid-cols-2 gap-4">
                <Field label={stopB1Ladder ? "Promote to Global Stop ($)" : "New Stop Price ($)"}>
                  <input type="number" value={newStop} onChange={e => setNewStop(e.target.value)}
                         step="0.01" placeholder="0.00" className={inputCls} style={inputStyle} />
                </Field>
                <div className="flex items-end">
                  <button
                    disabled={stopSaving || !newStop || parseFloat(newStop) <= 0}
                    onClick={async () => {
                      setStopSaving(true);
                      setStopResult(null);
                      const r = await api.updateTradeStops({
                        portfolio: getActivePortfolio(),
                        trade_id: selectedStop,
                        new_stop: parseFloat(newStop),
                      });
                      setStopSaving(false);
                      if (r.error) {
                        setStopResult({ ok: false, msg: r.error });
                      } else {
                        const beMsg = r.be_applied ? " · BE rule flagged" : "";
                        setStopResult({ ok: true, msg: `Stop updated on ${r.updated_lots || 0} lot(s)${beMsg}` });
                        setNewStop("");
                        setSelectedStop("");
                        // Reload so the updated stop shows on subsequent renders
                        const [open, closed, det] = await Promise.all([
                          api.tradesOpen(getActivePortfolio()),
                          api.tradesClosed(getActivePortfolio(), 500),
                          api.tradesRecent(getActivePortfolio(), 500),
                        ]);
                        setOpenTrades(open as TradePosition[]);
                        setAllTrades([...(open as TradePosition[]), ...(closed as TradePosition[])]);
                        setDetails(det.details);
                      }
                    }}
                    className="h-[42px] px-6 rounded-[10px] text-[13px] font-semibold text-white transition-all hover:brightness-110 disabled:opacity-50 disabled:cursor-not-allowed"
                    style={{ background: navColor }}>
                    {stopSaving ? "Saving…" : stopB1Ladder ? "Promote to Global Stop" : "Update All Lots"}
                  </button>
                </div>
              </div>

              {/* BE rule preview: if new stop is within 0.5% of avg_entry AND
                  the user eyeballs the stock as up 10%+, the backend will
                  stamp be_stop_moved_at. We show the preview here. */}
              {newStop && parseFloat(newStop) > 0 && stopTrade && (() => {
                const avgEntry = parseFloat(String(stopTrade.avg_entry || 0));
                const nearBe = avgEntry > 0 && Math.abs(parseFloat(newStop) - avgEntry) / avgEntry <= 0.005;
                return (
                  <div className="px-4 py-2.5 rounded-[10px] text-[12px] font-medium"
                       style={{ background: "color-mix(in oklab, #1e40af 10%, var(--surface))", color: "#3b82f6", border: "1px solid color-mix(in oklab, #1e40af 30%, var(--border))" }}>
                    Will update stop to <strong>{formatCurrency(parseFloat(newStop))}</strong> for all {stopDetails.length} lot(s) of {stopTrade.ticker}
                    {stopB1Ladder && (
                      <div className="mt-1 text-[11px]" style={{ color: "#d97706" }}>
                        🪜 This clears the Scale-Out ladder — the plan will be over and every lot moves to the single stop above.
                      </div>
                    )}
                    {nearBe && (
                      <div className="mt-1 text-[11px]" style={{ color: "#8b5cf6" }}>
                        🎯 Near breakeven ({formatCurrency(avgEntry)}) — if stock is up ≥10% from entry, this will flag the <strong>+10% BE rule</strong>.
                      </div>
                    )}
                  </div>
                );
              })()}

              {stopResult && (
                <div className="px-4 py-2.5 rounded-[10px] text-[12px] font-medium"
                     style={{
                       background: stopResult.ok ? "color-mix(in oklab, #08a86b 10%, var(--surface))" : "color-mix(in oklab, #e5484d 10%, var(--surface))",
                       color: stopResult.ok ? "#16a34a" : "#dc2626",
                       border: `1px solid ${stopResult.ok ? "color-mix(in oklab, #08a86b 30%, var(--border))" : "color-mix(in oklab, #e5484d 30%, var(--border))"}`,
                     }}>
                  {stopResult.ok ? "✓" : "✗"} {stopResult.msg}
                </div>
              )}

              {/* BE rule manual flag — backfill for trades where stop was
                  already moved to BE before the auto-detect was wired up. */}
              <div className="rounded-[10px] px-4 py-3 flex items-center justify-between"
                   style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}>
                <div className="flex flex-col gap-0.5">
                  <div className="text-[12px] font-semibold">🎯 +10% BE Rule Flag</div>
                  <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>
                    {(stopTrade as any)?.be_stop_moved_at
                      ? <>Flagged on <strong style={{ color: "#8b5cf6" }}>{String((stopTrade as any).be_stop_moved_at).slice(0, 10)}</strong></>
                      : <>Not flagged — use this if you already moved stop to BE before the auto-detect was wired up.</>}
                  </div>
                </div>
                <button
                  disabled={stopSaving}
                  onClick={async () => {
                    const currentlyFlagged = !!(stopTrade as any)?.be_stop_moved_at;
                    setStopSaving(true);
                    setStopResult(null);
                    const r = await api.flagBeRule({
                      portfolio: getActivePortfolio(),
                      trade_id: selectedStop,
                      flagged: !currentlyFlagged,
                    });
                    setStopSaving(false);
                    if (r.error) {
                      setStopResult({ ok: false, msg: r.error });
                    } else {
                      setStopResult({ ok: true, msg: r.flagged ? "Flagged as BE rule applied" : "BE flag cleared" });
                      const [open, closed, det] = await Promise.all([
                        api.tradesOpen(getActivePortfolio()),
                        api.tradesClosed(getActivePortfolio(), 500),
                        api.tradesRecent(getActivePortfolio(), 500),
                      ]);
                      setOpenTrades(open as TradePosition[]);
                      setAllTrades([...(open as TradePosition[]), ...(closed as TradePosition[])]);
                      setDetails(det.details);
                    }
                  }}
                  className="h-[32px] px-3 rounded-[8px] text-[11px] font-semibold transition-all hover:brightness-95 disabled:opacity-50"
                  style={{
                    background: (stopTrade as any)?.be_stop_moved_at ? "var(--bg)" : "#8b5cf6",
                    color: (stopTrade as any)?.be_stop_moved_at ? "var(--ink)" : "white",
                    border: (stopTrade as any)?.be_stop_moved_at ? "1px solid var(--border)" : "none",
                  }}>
                  {(stopTrade as any)?.be_stop_moved_at ? "Clear Flag" : "Flag BE Applied"}
                </button>
              </div>
            </>
          )}
        </div>
      )}

      {/* ═══════════ EDIT TRANSACTION ═══════════ */}
      {tab === "edit" && (
        <div className="flex flex-col gap-5">
          <div className="text-[13px]" style={{ color: "var(--ink-3)" }}>
            Search by ticker, select a campaign and transaction to edit.
          </div>

          <div className="grid grid-cols-3 gap-4">
            <Field label="Search Ticker">
              <input type="text" value={editTicker} onChange={e => { setEditTicker(e.target.value.toUpperCase()); setEditTradeId(""); setEditTxIdx(-1); }}
                     placeholder="Type ticker..." className={inputCls} style={inputStyle} />
            </Field>
            <Field label="Select Campaign">
              <select value={editTradeId} onChange={e => { setEditTradeId(e.target.value); setEditTxIdx(-1); setEditFields({}); }}
                      className={inputCls} style={{ ...inputStyle, appearance: "none" as any }}>
                <option value="">Select...</option>
                {editTrades.map(t => (
                  <option key={t.trade_id} value={t.trade_id}>{t.trade_id} — {t.ticker} ({t.status})</option>
                ))}
              </select>
            </Field>
            <Field label="Select Transaction">
              <select value={String(editTxIdx)} onChange={e => {
                const idx = parseInt(e.target.value);
                setEditTxIdx(idx);
                if (idx >= 0 && idx < editTxns.length) {
                  const tx = editTxns[idx];
                  const txRules = Array.isArray((tx as any).rules) ? (tx as any).rules as string[] : [];
                  setEditFields({
                    date: String(tx.date || "").slice(0, 16),
                    rule: txRules[0] || tx.rule || "",
                    trx_id: tx.trx_id || "",
                    stop_loss: String(tx.stop_loss || ""),
                    notes: tx.notes || "",
                    shares: String(tx.shares || ""),
                    amount: String(tx.amount || ""),
                  });
                  // Confluence carries rules[1..]; hydrate on select.
                  setEditConfluence(txRules.slice(1));
                  setEditConfluenceQuery("");
                  setEditConfluenceOpen(false);
                } else {
                  setEditConfluence([]);
                }
              }}
                      className={inputCls} style={{ ...inputStyle, appearance: "none" as any }}>
                <option value="-1">Select...</option>
                {editTxns.map((tx, i) => (
                  <option key={i} value={i}>{tx.trx_id || `#${i + 1}`} — {tx.action} {tx.shares} shs @ {formatCurrency(parseFloat(String(tx.amount || 0)))}</option>
                ))}
              </select>
            </Field>
          </div>

          {editTx && (
            <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
              <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
                <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
                <span className="text-[13px] font-semibold">Edit: {editTx.ticker} — {editTx.trx_id || "Transaction"}</span>
              </div>
              <div className="p-5 flex flex-col gap-4">
                <div className="grid grid-cols-2 gap-4">
                  <Field label="Date / Time">
                    <input type="datetime-local" value={editFields.date || ""} onChange={e => setEditFields({ ...editFields, date: e.target.value })}
                           className={inputCls} style={inputStyle} />
                  </Field>
                  <Field label={editTx && String(editTx.action).toUpperCase() === "SELL" ? "Sell Rule" : "Primary Buy Rule"}>
                    <SearchSelect
                      value={editFields.rule || ""}
                      onChange={v => setEditFields({ ...editFields, rule: v })}
                      options={editTx && String(editTx.action).toUpperCase() === "SELL" ? SELL_RULES : BUY_RULES}
                      placeholder="Select rule..."
                    />
                  </Field>
                </div>

                {/* Migration 047: buy-side confluence chips. Renders
                    only for BUY rows — SELL row multi-select is
                    intentionally out of scope for now. */}
                {editTx && String(editTx.action).toUpperCase() === "BUY" && (
                  <div>
                    <Field label="Confluence Rules (optional)">
                      <div className="flex items-center gap-1.5 flex-wrap min-h-[42px] p-1 rounded-[10px]"
                           style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
                        {editConfluence.map(r => (
                          <span key={r}
                                className="inline-flex items-center gap-1 h-[28px] px-2 rounded-[8px] text-[11px] font-semibold"
                                style={{
                                  background: "color-mix(in oklab, #6366f1 12%, transparent)",
                                  color: "#6366f1",
                                  border: "1px solid color-mix(in oklab, #6366f1 30%, var(--border))",
                                }}>
                            +{r}
                            <button type="button"
                                    onClick={() => setEditConfluence(prev => prev.filter(x => x !== r))}
                                    className="ml-0.5 opacity-60 hover:opacity-100 cursor-pointer"
                                    style={{ background: "none", border: "none", padding: 0, lineHeight: 1, color: "inherit", fontSize: 14 }}>×</button>
                          </span>
                        ))}
                        <div className="relative flex-1 min-w-[180px]">
                          <input type="text"
                                 value={editConfluenceQuery}
                                 placeholder={editConfluence.length > 0 ? "Add another…" : "Type to add confluence rules…"}
                                 onChange={e => { setEditConfluenceQuery(e.target.value); setEditConfluenceOpen(true); }}
                                 onFocus={() => setEditConfluenceOpen(true)}
                                 onBlur={() => setTimeout(() => setEditConfluenceOpen(false), 150)}
                                 onKeyDown={e => {
                                   if (e.key === "Backspace" && !editConfluenceQuery && editConfluence.length > 0) {
                                     setEditConfluence(prev => prev.slice(0, -1));
                                   }
                                 }}
                                 className="w-full h-[34px] px-3 rounded-[8px] text-[12px] outline-none"
                                 style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)",
                                          fontFamily: "var(--font-jetbrains), monospace" }} />
                          {editConfluenceOpen && (() => {
                            const q = editConfluenceQuery.trim().toLowerCase();
                            const available = BUY_RULES
                              .filter(r => r !== editFields.rule && !editConfluence.includes(r))
                              .filter(r => !q || r.toLowerCase().includes(q));
                            return available.length > 0 ? (
                              <div className="absolute z-50 mt-1 w-[280px] rounded-[10px] overflow-hidden shadow-lg"
                                   style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
                                <div className="overflow-y-auto" style={{ maxHeight: 240 }}>
                                  {available.slice(0, 30).map(r => (
                                    <button key={r} type="button"
                                            onMouseDown={e => {
                                              e.preventDefault();
                                              setEditConfluence(prev => [...prev, r]);
                                              setEditConfluenceQuery("");
                                              setEditConfluenceOpen(false);
                                            }}
                                            className="w-full text-left px-3 py-1.5 text-[12px] transition-colors cursor-pointer"
                                            onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-2)")}
                                            onMouseLeave={e => (e.currentTarget.style.background = "transparent")}>
                                      +{r}
                                    </button>
                                  ))}
                                </div>
                              </div>
                            ) : null;
                          })()}
                        </div>
                      </div>
                    </Field>
                  </div>
                )}
                <div className="grid grid-cols-3 gap-4">
                  <Field label="Trx ID">
                    {/* Read-only — server generates trx_ids collision-safely
                        (db_layer.generate_unique_trx_id + migration 018 UNIQUE
                        constraint). Editing here would just be a way to create
                        new collisions. */}
                    <input type="text" value={editFields.trx_id || ""} readOnly
                           className={inputCls}
                           style={{ ...inputStyle, opacity: 0.6, cursor: "not-allowed" }} />
                  </Field>
                  <Field label="Stop Loss ($)">
                    <input type="number" value={editFields.stop_loss || ""} onChange={e => setEditFields({ ...editFields, stop_loss: e.target.value })}
                           step="0.01" className={inputCls} style={inputStyle} />
                  </Field>
                  <Field label="Notes">
                    <input type="text" value={editFields.notes || ""} onChange={e => setEditFields({ ...editFields, notes: e.target.value })}
                           className={inputCls} style={{ ...inputStyle, fontFamily: "inherit" }} />
                  </Field>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <Field label="Shares">
                    <input type="number" value={editFields.shares || ""} onChange={e => setEditFields({ ...editFields, shares: e.target.value })}
                           className={inputCls} style={inputStyle} />
                  </Field>
                  <Field label="Price ($)">
                    <input type="number" value={editFields.amount || ""} onChange={e => setEditFields({ ...editFields, amount: e.target.value })}
                           step="0.01" className={inputCls} style={inputStyle} />
                  </Field>
                </div>
                <div className="flex items-center gap-4">
                  <button disabled={editSaving}
                          onClick={async () => {
                            if (!editTx) return;
                            setEditSaving(true);
                            setEditResult(null);
                            try {
                              // Migration 047: submit the ordered rules
                              // array (primary + confluence) alongside
                              // the legacy `rule` scalar. For SELL rows
                              // editConfluence stays empty (multi-select
                              // is buy-only), so the array collapses to
                              // [rule] — identical to pre-047 behavior.
                              const rulesPayload = [editFields.rule || "", ...editConfluence].filter(Boolean);
                              const res = await api.editTransaction({
                                detail_id: editTx.detail_id,
                                trade_id: editTx.trade_id,
                                ticker: editTx.ticker,
                                action: editTx.action,
                                date: editFields.date || "",
                                shares: parseFloat(editFields.shares || "0"),
                                amount: parseFloat(editFields.amount || "0"),
                                value: parseFloat(editFields.shares || "0") * parseFloat(editFields.amount || "0"),
                                rule: editFields.rule || "",
                                rules: rulesPayload,
                                notes: editFields.notes || "",
                                stop_loss: parseFloat(editFields.stop_loss || "0"),
                                trx_id: editFields.trx_id || "",
                              } as any);
                              if (res.error) {
                                setEditResult({ ok: false, msg: res.error });
                              } else {
                                setEditResult({ ok: true, msg: "Transaction updated successfully" });
                                // Refresh data
                                const [open, closed, det] = await Promise.all([
                                  api.tradesOpen(getActivePortfolio()),
                                  api.tradesClosed(getActivePortfolio(), 500),
                                  api.tradesRecent(getActivePortfolio(), 500),
                                ]);
                                setOpenTrades(open); setAllTrades([...open, ...closed]); setDetails(det.details);
                              }
                            } catch (err: any) {
                              setEditResult({ ok: false, msg: err.message || "Failed to save" });
                            } finally {
                              setEditSaving(false);
                            }
                          }}
                          className="h-[42px] px-6 rounded-[10px] text-[13px] font-semibold text-white transition-all hover:brightness-110 w-fit disabled:opacity-50"
                          style={{ background: navColor }}>
                    {editSaving ? "Saving..." : "Save Changes"}
                  </button>
                  <button disabled={editSaving}
                          onClick={async () => {
                            if (!editTx) return;
                            const label = `${editTx.action} ${editTx.shares} ${editTx.ticker} @ ${formatCurrency(parseFloat(String(editTx.amount || 0)))}`;
                            const confirmed = window.confirm(
                              `Delete this transaction?\n\n${label}\n${editTx.trx_id ? `Trx ID: ${editTx.trx_id}` : ""}\n\nThe campaign's avg entry / realized P&L / status will be recomputed from the remaining transactions. If this is the only transaction, the entire campaign is removed. Soft-delete — recoverable from the DB by clearing deleted_at.`
                            );
                            if (!confirmed) return;
                            setEditSaving(true);
                            setEditResult(null);
                            try {
                              const res = await api.deleteTransaction(
                                editTx.detail_id,
                                editTx.trade_id,
                                editTx.ticker,
                              );
                              if (res.error) {
                                setEditResult({ ok: false, msg: res.error });
                              } else {
                                setEditResult({ ok: true, msg: "Transaction deleted" });
                                // Reset selection so the now-stale row clears.
                                setEditTxIdx(-1);
                                setEditFields({});
                                const [open, closed, det] = await Promise.all([
                                  api.tradesOpen(getActivePortfolio()),
                                  api.tradesClosed(getActivePortfolio(), 500),
                                  api.tradesRecent(getActivePortfolio(), 500),
                                ]);
                                setOpenTrades(open); setAllTrades([...open, ...closed]); setDetails(det.details);
                              }
                            } catch (err: any) {
                              setEditResult({ ok: false, msg: err.message || "Failed to delete" });
                            } finally {
                              setEditSaving(false);
                            }
                          }}
                          className="h-[42px] px-6 rounded-[10px] text-[13px] font-semibold transition-all hover:brightness-110 w-fit disabled:opacity-50"
                          style={{ background: "var(--bg)", color: "#e5484d", border: "1px solid color-mix(in oklab, #e5484d 35%, var(--border))" }}>
                    Delete Transaction
                  </button>
                  {editResult && (
                    <span className="text-[12px] font-medium" style={{ color: editResult.ok ? "#16a34a" : "#e5484d" }}>
                      {editResult.msg}
                    </span>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ═══════════ DELETE TRADE ═══════════ */}
      {tab === "delete" && (
        <div className="flex flex-col gap-5">
          <div className="text-[13px]" style={{ color: "var(--ink-3)" }}>
            Permanently delete an entire campaign and all its transactions.
          </div>

          {deleteResult && (
            <div className="px-4 py-3 rounded-[10px] text-[13px]"
                 style={{ background: deleteResult.ok ? "color-mix(in oklab, #08a86b 10%, var(--surface))" : "color-mix(in oklab, #e5484d 10%, var(--surface))",
                          border: `1px solid ${deleteResult.ok ? "#08a86b30" : "#e5484d30"}`,
                          color: deleteResult.ok ? "#08a86b" : "#e5484d" }}>
              {deleteResult.msg}
            </div>
          )}

          <Field label="Select Trade to Delete">
            <select value={deleteTradeId} onChange={e => { setDeleteTradeId(e.target.value); setDeleteConfirm(""); }}
                    className={inputCls} style={{ ...inputStyle, appearance: "none" as any }}>
              <option value="">Select...</option>
              {[...allTrades].sort((a, b) => String(b.trade_id || "").localeCompare(String(a.trade_id || ""))).map(t => (
                <option key={t.trade_id} value={t.trade_id}>{t.trade_id} — {t.ticker} ({t.status})</option>
              ))}
            </select>
          </Field>

          {deleteTrade && (
            <div className="rounded-[14px] overflow-hidden" style={{ background: "color-mix(in oklab, #e5484d 10%, var(--surface))", border: "1px solid color-mix(in oklab, #e5484d 30%, var(--border))" }}>
              <div className="px-5 py-4">
                <div className="text-[14px] font-semibold mb-2" style={{ color: "#dc2626" }}>
                  Delete Campaign: {deleteTrade.ticker} ({deleteTrade.trade_id})
                </div>
                <div className="grid grid-cols-3 gap-3 mb-4">
                  <div className="text-[12px]"><span style={{ color: "var(--ink-4)" }}>Status:</span> <strong>{deleteTrade.status}</strong></div>
                  <div className="text-[12px]"><span style={{ color: "var(--ink-4)" }}>Shares:</span> <strong>{deleteTrade.shares}</strong></div>
                  <div className="text-[12px]"><span style={{ color: "var(--ink-4)" }}>Transactions:</span> <strong>{deleteTxCount}</strong></div>
                </div>
                <div className="mb-3">
                  <div className="text-[11px] font-semibold mb-1" style={{ color: "#dc2626" }}>Type DELETE to confirm</div>
                  <input type="text" value={deleteConfirm} onChange={e => setDeleteConfirm(e.target.value)}
                         placeholder="DELETE" className={inputCls}
                         style={{ ...inputStyle, borderColor: "color-mix(in oklab, #e5484d 30%, var(--border))" }} />
                </div>
                <button onClick={async () => {
                          if (deleteConfirm !== "DELETE") return;
                          try {
                            const tid = deleteTradeId;
                            const res = await api.deleteTrade(tid);
                            if (res.error) { setDeleteResult({ ok: false, msg: res.error }); return; }
                            setAllTrades(prev => prev.filter(t => t.trade_id !== tid));
                            setDetails(prev => prev.filter(d => d.trade_id !== tid));
                            setDeleteTradeId(""); setDeleteConfirm("");
                            setDeleteResult({ ok: true, msg: `Trade ${tid} permanently deleted.` });
                          } catch (e: any) { setDeleteResult({ ok: false, msg: e.message || "Delete failed" }); }
                        }}
                        disabled={deleteConfirm !== "DELETE"}
                        className="h-[42px] px-6 rounded-[10px] text-[13px] font-semibold text-white transition-all disabled:opacity-40"
                        style={{ background: "#dc2626" }}>
                  Permanently Delete
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ═══════════ EXPORT ═══════════ */}
      {tab === "export" && (
        <div className="flex flex-col gap-5">
          <div className="text-[13px]" style={{ color: "var(--ink-3)" }}>
            Download trade data as CSV for external analysis.
          </div>

          <div className="grid grid-cols-2 gap-4">
            <Field label="Export Level">
              <div className="flex gap-2 mt-1">
                {(["summary", "details"] as const).map(t => (
                  <button key={t} onClick={() => setExportType(t)}
                          className="flex-1 h-[38px] rounded-[10px] text-[12px] font-semibold transition-all capitalize"
                          style={{
                            background: exportType === t ? `color-mix(in oklab, ${navColor} 10%, transparent)` : "var(--bg)",
                            color: exportType === t ? navColor : "var(--ink-4)",
                            border: `1.5px solid ${exportType === t ? navColor : "var(--border)"}`,
                          }}>
                    {t === "summary" ? "Campaign Summary" : "Transaction Details"}
                  </button>
                ))}
              </div>
            </Field>
            <Field label="Status Filter">
              <div className="flex gap-2 mt-1">
                {(["all", "open", "closed"] as const).map(s => (
                  <button key={s} onClick={() => setExportStatus(s)}
                          className="flex-1 h-[38px] rounded-[10px] text-[12px] font-semibold transition-all capitalize"
                          style={{
                            background: exportStatus === s ? `color-mix(in oklab, ${navColor} 10%, transparent)` : "var(--bg)",
                            color: exportStatus === s ? navColor : "var(--ink-4)",
                            border: `1.5px solid ${exportStatus === s ? navColor : "var(--border)"}`,
                          }}>
                    {s}
                  </button>
                ))}
              </div>
            </Field>
          </div>

          {/* Date range presets + custom */}
          <Field label="Date Range">
            <div className="flex gap-1.5 mb-3 flex-wrap">
              {([
                { label: "All Time", from: "", to: "" },
                { label: "YTD", from: `${new Date().getFullYear()}-01-01`, to: "" },
                { label: "MTD", from: `${new Date().getFullYear()}-${String(new Date().getMonth() + 1).padStart(2, "0")}-01`, to: "" },
                { label: "Last 30d", from: new Date(Date.now() - 30 * 86400000).toISOString().slice(0, 10), to: "" },
                { label: "Last 90d", from: new Date(Date.now() - 90 * 86400000).toISOString().slice(0, 10), to: "" },
                { label: "Q1", from: `${new Date().getFullYear()}-01-01`, to: `${new Date().getFullYear()}-03-31` },
                { label: "Q2", from: `${new Date().getFullYear()}-04-01`, to: `${new Date().getFullYear()}-06-30` },
              ]).map(p => {
                const active = exportFrom === p.from && exportTo === p.to;
                return (
                  <button key={p.label} onClick={() => { setExportFrom(p.from); setExportTo(p.to); }}
                          className="h-[30px] px-3 rounded-[8px] text-[11px] font-medium transition-all"
                          style={{
                            background: active ? `color-mix(in oklab, ${navColor} 10%, transparent)` : "var(--bg)",
                            color: active ? navColor : "var(--ink-4)",
                            border: `1px solid ${active ? navColor : "var(--border)"}`,
                          }}>
                    {p.label}
                  </button>
                );
              })}
            </div>
          </Field>
          <div className="grid grid-cols-2 gap-4">
            <Field label="From Date">
              <input type="date" value={exportFrom} onChange={e => setExportFrom(e.target.value)}
                     className={inputCls} style={inputStyle} />
            </Field>
            <Field label="To Date">
              <input type="date" value={exportTo} onChange={e => setExportTo(e.target.value)}
                     className={inputCls} style={inputStyle} />
            </Field>
          </div>

          {/* Preview */}
          {(() => {
            const dateFilter = (dateStr: string) => {
              if (!exportFrom && !exportTo) return true;
              const d = String(dateStr || "").slice(0, 10);
              if (exportFrom && d < exportFrom) return false;
              if (exportTo && d > exportTo) return false;
              return true;
            };
            const filteredSummary = allTrades
              .filter(t => exportStatus === "all" || t.status?.toUpperCase() === exportStatus.toUpperCase())
              .filter(t => dateFilter(t.open_date));
            const filteredDetails = details
              .filter(d => {
                if (exportStatus !== "all") {
                  const trade = allTrades.find(t => t.trade_id === d.trade_id);
                  if (exportStatus === "open" && trade?.status?.toUpperCase() !== "OPEN") return false;
                  if (exportStatus === "closed" && trade?.status?.toUpperCase() !== "CLOSED") return false;
                }
                return dateFilter(d.date);
              });
            const count = exportType === "summary" ? filteredSummary.length : filteredDetails.length;
            return (
              <div className="rounded-[10px] p-4" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
                <div className="text-[12px]" style={{ color: "var(--ink-3)" }}>
                  {count} {exportType === "summary" ? "campaigns" : "transactions"} will be exported
                  {(exportFrom || exportTo) && (
                    <span> · {exportFrom || "start"} → {exportTo || "now"}</span>
                  )}
                </div>
              </div>
            );
          })()}

          <button onClick={() => {
                    const dateFilter = (dateStr: string) => {
                      if (!exportFrom && !exportTo) return true;
                      const d = String(dateStr || "").slice(0, 10);
                      if (exportFrom && d < exportFrom) return false;
                      if (exportTo && d > exportTo) return false;
                      return true;
                    };
                    let data: any[];
                    if (exportType === "summary") {
                      data = allTrades
                        .filter(t => exportStatus === "all" || t.status?.toUpperCase() === exportStatus.toUpperCase())
                        .filter(t => dateFilter(t.open_date));
                    } else {
                      data = details.filter(d => {
                        if (exportStatus !== "all") {
                          const trade = allTrades.find(t => t.trade_id === d.trade_id);
                          if (exportStatus === "open" && trade?.status?.toUpperCase() !== "OPEN") return false;
                          if (exportStatus === "closed" && trade?.status?.toUpperCase() !== "CLOSED") return false;
                        }
                        return dateFilter(d.date);
                      });
                    }
                    if (data.length === 0) return alert("No data to export");
                    const headers = Object.keys(data[0]);
                    const csv = [headers.join(","), ...data.map(row => headers.map(h => {
                      const v = String((row as any)[h] ?? "");
                      return v.includes(",") || v.includes('"') ? `"${v.replace(/"/g, '""')}"` : v;
                    }).join(","))].join("\n");
                    const blob = new Blob([csv], { type: "text/csv" });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url;
                    a.download = `trades_${exportType}_${exportStatus}_${new Date().toISOString().slice(0, 10)}.csv`;
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                  className="h-[48px] px-8 rounded-[12px] text-[14px] font-semibold text-white transition-all hover:brightness-110 w-fit"
                  style={{ background: "#6366f1" }}>
            Download CSV
          </button>
        </div>
      )}

      {/* ═══════════ SR8 TRIM CALCULATOR ═══════════ */}
      {tab === "sr8-trim" && (() => {
        // Enrich on the fly using the same path ACS uses. equity arg is
        // only consumed for pos_size_pct (not surfaced by the calculator),
        // so 0 is fine — NAV is user-input via the calculator's own field.
        const enriched = computeEnrichedPositions(openTrades, details, 0, livePrices);
        const sr8 = enriched.filter((p) => p.sell_rule_tier === "sr8");
        return <SR8TrimCalculator positions={sr8} />;
      })()}
    </div>
  );
}
