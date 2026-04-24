"use client";

import { useState, useEffect, useRef } from "react";
import { api, getActivePortfolio, type TradePosition, type TradeDetail } from "@/lib/api";

const BUY_RULES = [
  "br1.1 Consolidation", "br1.2 Cup w Handle", "br1.3 Cup w/o Handle", "br1.4 Double Bottom",
  "br1.5 IPO Base", "br1.6 Flat Base", "br1.7 Consolidation Pivot", "br1.8 High Tight Flag",
  "br2.1 HVE", "br2.2 HVSI", "br2.3 HV1",
  "br3.1 Reclaim 21e", "br3.2 Reclaim 50s", "br3.3 Reclaim 200s", "br3.4 Reclaim 10W", "br3.5 Reclaim 8e",
  "br4.1 PB 21e", "br4.2 PB 50s", "br4.3 PB 10w", "br4.4 PB 200s", "br4.5 PB 8e", "br4.6 VWAP",
  "br5.1 Undercut & Rally", "br5.2 Upside Reversal",
  "br6.1 Gapper", "br6.2 Continuation Gap Up",
  "br7.1 TQQQ Strategy", "br7.2 New High after Gentle PB", "br7.3 JL Century Mark",
  "br8.1 Daily STL Break", "br8.2 Weekly STL Break", "br8.3 Monthly STL Break",
  "br9.1 21e Strategy",
  "br10.1 Hedging with leverage product",
  "br11.1 Shorting",
  "br12.1 Option Play",
];

const SELL_RULES = [
  "sr1 Capital Protection", "sr2 Trailing Stop", "sr3 Portfolio Management",
  "sr4 Time Stop", "sr5 Climax Top", "sr6 Exhaustion Gap",
  "sr7 200d Moving Avg Break", "sr8 Living Below 50d", "sr9 Failed Breakout",
  "sr10 Scale-Out T1 (-3%)", "sr11 Scale-Out T2 (-5%)", "sr12 Scale-Out T3 (-8%)",
  "sr13 Earnings Exit", "sr14 Market Correction Exit",
  "sr15 BE Stop Out (moved at +10%)",
];

type Tab = "stops" | "edit" | "delete" | "export";

const TABS: { key: Tab; label: string; icon: string }[] = [
  { key: "stops", label: "Stop Loss Adjustment", icon: "🛡️" },
  { key: "edit", label: "Edit Transaction", icon: "📝" },
  { key: "delete", label: "Delete Trade", icon: "🗑️" },
  { key: "export", label: "Export", icon: "📥" },
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
  value: string; onChange: (v: string) => void; options: string[]; placeholder?: string;
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
        <div className="absolute z-50 mt-1 w-full rounded-[10px] overflow-hidden shadow-lg" style={{ background: "var(--surface)", border: "1px solid var(--border)", maxHeight: 280 }}>
          <div className="p-2" style={{ borderBottom: "1px solid var(--border)" }}>
            <input type="text" value={search} onChange={e => setSearch(e.target.value)} placeholder="Type to search..." autoFocus
                   className="w-full h-[34px] px-3 rounded-[8px] text-[12px] outline-none" style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }} />
          </div>
          <div className="overflow-y-auto" style={{ maxHeight: 220 }}>
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
    if (initialTab && ["stops", "edit", "delete", "export"].includes(initialTab)) {
      setTab(initialTab as Tab);
      onTabConsumed?.();
    }
  }, [initialTab, onTabConsumed]);
  const [openTrades, setOpenTrades] = useState<TradePosition[]>([]);
  const [allTrades, setAllTrades] = useState<TradePosition[]>([]);
  const [details, setDetails] = useState<TradeDetail[]>([]);
  const [loading, setLoading] = useState(true);

  // Stop Loss tab
  const [selectedStop, setSelectedStop] = useState("");
  const [newStop, setNewStop] = useState("");
  const [stopSaving, setStopSaving] = useState(false);
  const [stopResult, setStopResult] = useState<{ ok: boolean; msg: string } | null>(null);

  // Edit tab
  const [editTicker, setEditTicker] = useState("");
  const [editTradeId, setEditTradeId] = useState("");
  const [editTxIdx, setEditTxIdx] = useState(-1);
  const [editFields, setEditFields] = useState<Record<string, string>>({});
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
      api.tradesOpen(getActivePortfolio()).catch(() => []),
      api.tradesClosed(getActivePortfolio(), 500).catch(() => []),
      api.tradesRecent(getActivePortfolio(), 500).catch(() => []),
    ]).then(([open, closed, det]) => {
      setOpenTrades(open as TradePosition[]);
      setAllTrades([...open as TradePosition[], ...closed as TradePosition[]]);
      setDetails(det as TradeDetail[]);
      setLoading(false);
    });
  }, []);

  if (loading) {
    return <div className="animate-pulse"><div className="h-[90px] rounded-[14px]" style={{ background: "var(--bg-2)" }} /></div>;
  }

  // Stop loss tab: details for selected position
  const stopTrade = openTrades.find(t => t.trade_id === selectedStop);
  const stopDetails = details.filter(d => d.trade_id === selectedStop && String(d.action).toUpperCase() === "BUY");

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
              value={selectedStop ? (() => { const t = openTrades.find(x => x.trade_id === selectedStop); return t ? `${t.ticker} | ${t.trade_id} (${t.shares} shs @ $${parseFloat(String(t.avg_entry || 0)).toFixed(2)})` : ""; })() : ""}
              onChange={(v) => { const id = v.split(" | ")[1]?.split(" (")[0]?.trim() || ""; setSelectedStop(id); setNewStop(""); }}
              options={openTrades.map(t => `${t.ticker} | ${t.trade_id} (${t.shares} shs @ $${parseFloat(String(t.avg_entry || 0)).toFixed(2)})`)}
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
                        <td className="px-3 py-2 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>${parseFloat(String(d.amount || 0)).toFixed(2)}</td>
                        <td className="px-3 py-2" style={{ fontFamily: "var(--font-jetbrains), monospace", color: parseFloat(String(d.stop_loss || 0)) > 0 ? "var(--ink)" : "var(--ink-4)" }}>
                          {parseFloat(String(d.stop_loss || 0)) > 0 ? `$${parseFloat(String(d.stop_loss)).toFixed(2)}` : "—"}
                        </td>
                        <td className="px-3 py-2 text-[10px]" style={{ color: "var(--ink-4)" }}>{d.notes || ""}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* New stop input */}
              <div className="grid grid-cols-2 gap-4">
                <Field label="New Stop Price ($)">
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
                        setDetails(det as TradeDetail[]);
                      }
                    }}
                    className="h-[42px] px-6 rounded-[10px] text-[13px] font-semibold text-white transition-all hover:brightness-110 disabled:opacity-50 disabled:cursor-not-allowed"
                    style={{ background: navColor }}>
                    {stopSaving ? "Saving…" : "Update All Lots"}
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
                    Will update stop to <strong>${parseFloat(newStop).toFixed(2)}</strong> for all {stopDetails.length} lot(s) of {stopTrade.ticker}
                    {nearBe && (
                      <div className="mt-1 text-[11px]" style={{ color: "#8b5cf6" }}>
                        🎯 Near breakeven (${avgEntry.toFixed(2)}) — if stock is up ≥10% from entry, this will flag the <strong>+10% BE rule</strong>.
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
                  setEditFields({
                    date: String(tx.date || "").slice(0, 16),
                    rule: tx.rule || "",
                    trx_id: tx.trx_id || "",
                    stop_loss: String(tx.stop_loss || ""),
                    notes: tx.notes || "",
                    shares: String(tx.shares || ""),
                    amount: String(tx.amount || ""),
                  });
                }
              }}
                      className={inputCls} style={{ ...inputStyle, appearance: "none" as any }}>
                <option value="-1">Select...</option>
                {editTxns.map((tx, i) => (
                  <option key={i} value={i}>{tx.trx_id || `#${i + 1}`} — {tx.action} {tx.shares} shs @ ${parseFloat(String(tx.amount || 0)).toFixed(2)}</option>
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
                  <Field label="Rule (Strategy)">
                    <SearchSelect
                      value={editFields.rule || ""}
                      onChange={v => setEditFields({ ...editFields, rule: v })}
                      options={editTx && String(editTx.action).toUpperCase() === "SELL" ? SELL_RULES : BUY_RULES}
                      placeholder="Select rule..."
                    />
                  </Field>
                </div>
                <div className="grid grid-cols-3 gap-4">
                  <Field label="Trx ID">
                    <input type="text" value={editFields.trx_id || ""} onChange={e => setEditFields({ ...editFields, trx_id: e.target.value })}
                           className={inputCls} style={inputStyle} />
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
                                notes: editFields.notes || "",
                                stop_loss: parseFloat(editFields.stop_loss || "0"),
                                trx_id: editFields.trx_id || "",
                              });
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
                                setOpenTrades(open); setAllTrades([...open, ...closed]); setDetails(det as TradeDetail[]);
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
    </div>
  );
}
