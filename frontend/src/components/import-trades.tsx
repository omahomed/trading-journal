"use client";

import { useState, useEffect, useMemo } from "react";
import { api, type TradePosition, type TradeDetail } from "@/lib/api";

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
];

type Trade = {
  symbol: string;
  description?: string;
  asset_class?: string;
  action: "BUY" | "SELL" | string;
  quantity: number;
  price: number;
  amount: number;
  commission: number;
  net_cash: number;
  order_time: string;
  trade_date: string;
  put_call?: string;
  strike?: string;
  expiry?: string;
  order_id?: string;
};

function isOption(t: Trade) {
  const ac = (t.asset_class || "").toUpperCase();
  return ac.startsWith("OPT") || !!t.put_call;
}

function optionLabel(t: Trade) {
  const pc = (t.put_call || "").toUpperCase() === "P" ? "Put" : "Call";
  const strike = t.strike ? `$${t.strike}` : "";
  const exp = t.expiry ? t.expiry.slice(5) : "";
  return `${pc} ${strike} ${exp}`.trim();
}

// Build a clean human-readable option ticker: UNDERLYING YYMMDD $STRIKE{C|P}
// IBKR's raw `symbol` for options already includes the OCC serial
// ("ARM   260618C00175000"); strip everything after the first whitespace
// block so we don't end up with both notations smashed together.
function optionTicker(t: Trade) {
  const underlying = (t.symbol || "").trim().split(/\s+/)[0] || t.symbol || "";
  const pc = (t.put_call || "").toUpperCase() === "P" ? "P" : "C";
  const exp = (t.expiry || "").replace(/-/g, "").slice(2); // YYMMDD
  const strike = t.strike || "";
  return `${underlying} ${exp} $${strike}${pc}`;
}

const EXEC_CACHE_KEY = "importTrades.executions.v1";

export function ImportTrades({ navColor, onNavigate }: { navColor: string; onNavigate?: (page: string) => void }) {
  const [pulling, setPulling] = useState(false);
  const [executions, setExecutions] = useState<Trade[]>(() => {
    if (typeof window === "undefined") return [];
    try {
      const raw = window.localStorage.getItem(EXEC_CACHE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed.trades) ? parsed.trades : [];
    } catch { return []; }
  });
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [rawDebug, setRawDebug] = useState<Record<string, any> | null>(null);
  const [showDebug, setShowDebug] = useState(false);
  const [todayDetails, setTodayDetails] = useState<TradeDetail[]>([]);
  const [openTrades, setOpenTrades] = useState<TradePosition[]>([]);
  const [quickLogRow, setQuickLogRow] = useState<number | null>(null);
  const [quickLog, setQuickLog] = useState<{ rule: string; trade_id: string; stop: string; action_type: "new" | "scalein"; notes: string }>({
    rule: "", trade_id: "", stop: "", action_type: "new", notes: "",
  });
  const [quickLogBusy, setQuickLogBusy] = useState(false);
  const [quickLogResult, setQuickLogResult] = useState<{ row: number; ok: boolean; msg: string } | null>(null);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState(false);

  const loadContext = async () => {
    const [details, open] = await Promise.all([
      api.tradesRecent("CanSlim", 500).catch(() => []),
      api.tradesOpen("CanSlim").catch(() => []),
    ]);
    setTodayDetails(details as TradeDetail[]);
    setOpenTrades(open as TradePosition[]);
  };

  useEffect(() => { loadContext(); }, []);

  // Persist pulled executions to localStorage so the table survives page navigation
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      if (executions.length > 0) {
        window.localStorage.setItem(EXEC_CACHE_KEY, JSON.stringify({ trades: executions, cached_at: Date.now() }));
      } else {
        window.localStorage.removeItem(EXEC_CACHE_KEY);
      }
    } catch { /* ignore quota errors */ }
  }, [executions]);

  const handlePull = async () => {
    setPulling(true); setError(""); setMessage("");
    try {
      const result = await api.importTrades();
      setRawDebug(result.debug || null);
      if (result.error) {
        setError(result.error);
      } else {
        setExecutions((result.trades || []) as Trade[]);
        setMessage(result.count === 0 ? (result.message || "No trades found") : `Pulled ${result.count} execution(s) from IBKR`);
      }
    } catch (e: any) {
      setError(e.message || "Failed to connect to IBKR");
    } finally {
      setPulling(false);
    }
    await loadContext();
  };

  // Heuristic duplicate detection: same trade_date + symbol + action + quantity
  // already present in trades_details. Close-enough match on the common case.
  const duplicateMap = useMemo(() => {
    const m = new Map<number, boolean>();
    executions.forEach((t, i) => {
      const tickerToMatch = isOption(t) ? optionTicker(t) : t.symbol;
      const dup = todayDetails.some(d => {
        const dDate = String(d.date || "").slice(0, 10);
        const dAction = String(d.action || "").toUpperCase();
        const dTicker = String(d.ticker || "").toUpperCase();
        const dQty = Math.abs(parseFloat(String(d.shares || 0)));
        return dDate === t.trade_date
          && dAction === (t.action || "").toUpperCase()
          && dTicker === tickerToMatch.toUpperCase()
          && Math.abs(dQty - t.quantity) < 0.01;
      });
      m.set(i, dup);
    });
    return m;
  }, [executions, todayDetails]);

  const sendToLogBuy = (t: Trade) => {
    const ticker = isOption(t) ? optionTicker(t) : t.symbol;
    // If there's already an open campaign for this ticker, flip to scale-in
    // mode so the Log Buy form opens pre-pointed at that campaign instead of
    // creating a duplicate new trade.
    const existing = openTrades.find(o => String(o.ticker || "").toUpperCase() === ticker.toUpperCase());
    localStorage.setItem("ps_prefill", JSON.stringify({
      ticker, shares: t.quantity, price: t.price,
      action: existing ? "scale_in" : "new",
      trade_id: existing?.trade_id || undefined,
      date: t.trade_date || undefined,
      time: t.order_time ? t.order_time.slice(0, 5) : undefined,
    }));
    onNavigate?.("logbuy");
  };

  const sendToLogSell = (t: Trade) => {
    const ticker = isOption(t) ? optionTicker(t) : t.symbol;
    // Find the open campaign for this ticker (exact match) — if multiple, pick first
    const match = openTrades.find(o => String(o.ticker || "").toUpperCase() === ticker.toUpperCase());
    localStorage.setItem("ps_prefill_sell", JSON.stringify({
      ticker, shares: t.quantity, price: t.price,
      trade_id: match?.trade_id || "",
      date: t.trade_date || undefined,
      time: t.order_time ? t.order_time.slice(0, 5) : undefined,
    }));
    onNavigate?.("logsell");
  };

  const openQuickLog = async (t: Trade, idx: number) => {
    setQuickLogResult(null);
    const ticker = isOption(t) ? optionTicker(t) : t.symbol;
    if (t.action === "BUY") {
      // Default to a new campaign; try to grab the next trade_id
      let tradeId = "";
      try {
        const r = await api.nextTradeId("CanSlim", t.trade_date);
        if (!("error" in r)) tradeId = r.trade_id;
      } catch { /* ignore */ }
      // If there's already an open campaign for this ticker, default to scale-in
      const existing = openTrades.find(o => String(o.ticker || "").toUpperCase() === ticker.toUpperCase());
      setQuickLog({
        rule: BUY_RULES[0],
        trade_id: existing ? existing.trade_id : tradeId,
        stop: "",
        action_type: existing ? "scalein" : "new",
        notes: "",
      });
    } else {
      const existing = openTrades.find(o => String(o.ticker || "").toUpperCase() === ticker.toUpperCase());
      setQuickLog({
        rule: SELL_RULES[0],
        trade_id: existing?.trade_id || "",
        stop: "",
        action_type: "new",
        notes: "",
      });
    }
    setQuickLogRow(idx);
  };

  const submitQuickLog = async (t: Trade) => {
    setQuickLogBusy(true);
    try {
      const ticker = isOption(t) ? optionTicker(t) : t.symbol;
      if (t.action === "BUY") {
        if (!quickLog.trade_id) {
          setQuickLogResult({ row: quickLogRow!, ok: false, msg: "trade_id missing" });
          setQuickLogBusy(false);
          return;
        }
        const res = await api.logBuy({
          portfolio: "CanSlim",
          action_type: quickLog.action_type,
          ticker, trade_id: quickLog.trade_id,
          shares: t.quantity, price: t.price,
          stop_loss: parseFloat(quickLog.stop) || 0,
          rule: quickLog.rule,
          notes: quickLog.notes,
          date: t.trade_date,
          time: (t.order_time || "").slice(0, 5),
        });
        if (res.error) throw new Error(res.error);
        setQuickLogResult({ row: quickLogRow!, ok: true, msg: `Logged ${res.trx_id || "buy"} on ${ticker}` });
      } else {
        if (!quickLog.trade_id) {
          setQuickLogResult({ row: quickLogRow!, ok: false, msg: "Pick an open campaign" });
          setQuickLogBusy(false);
          return;
        }
        const res = await api.logSell({
          portfolio: "CanSlim",
          trade_id: quickLog.trade_id,
          shares: t.quantity, price: t.price,
          rule: quickLog.rule,
          notes: quickLog.notes,
          date: t.trade_date,
          time: (t.order_time || "").slice(0, 5),
        });
        if (res.error) throw new Error(res.error);
        setQuickLogResult({ row: quickLogRow!, ok: true, msg: `Logged ${res.trx_id || "sell"} on ${ticker}` });
      }
      setQuickLogRow(null);
      await loadContext();
    } catch (e: any) {
      setQuickLogResult({ row: quickLogRow!, ok: false, msg: e.message || "Failed" });
    }
    setQuickLogBusy(false);
  };

  const deleteToday = async () => {
    if (!deleteConfirm) { setDeleteConfirm(true); setTimeout(() => setDeleteConfirm(false), 3500); return; }
    setDeleteBusy(true);
    try {
      const today = executions[0]?.trade_date || new Date().toISOString().slice(0, 10);
      const res = await api.deleteTransactionsByDate(today, "CanSlim");
      if (res.error) {
        setError(res.error);
      } else {
        setMessage(`Deleted ${res.deleted || 0} transaction(s) from ${today}. Affected campaigns recomputed.`);
        await loadContext();
      }
    } catch (e: any) {
      setError(e.message || "Delete failed");
    }
    setDeleteBusy(false);
    setDeleteConfirm(false);
  };

  const stockRows = executions.map((t, i) => ({ t, i })).filter(({ t }) => !isOption(t));
  const optionRows = executions.map((t, i) => ({ t, i })).filter(({ t }) => isOption(t));

  const fmtDol = (v: number, d = 2) => `$${v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d })}`;

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          Import <em className="italic" style={{ color: navColor }}>Trades</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
          Pull executions from IBKR Flex Web Service
        </div>
      </div>

      {/* Connection status */}
      <div className="flex items-center gap-3 mb-6 p-4 rounded-[14px]" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <div className="w-10 h-10 rounded-[12px] flex items-center justify-center text-lg"
             style={{ background: `color-mix(in oklab, ${navColor} 12%, transparent)` }}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={navColor} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>
            <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>
          </svg>
        </div>
        <div className="flex-1">
          <div className="text-[13px] font-semibold">Interactive Brokers Flex Query</div>
          <div className="text-[12px]" style={{ color: "var(--ink-4)" }}>
            Configured via server-side credentials · Stocks & Options
          </div>
        </div>
        {executions.length > 0 && (
          <button onClick={deleteToday} disabled={deleteBusy}
                  className="h-[36px] px-4 rounded-[10px] text-[12px] font-semibold transition-all hover:brightness-110 disabled:opacity-50"
                  style={{ background: deleteConfirm ? "#e5484d" : "var(--bg)", color: deleteConfirm ? "#fff" : "var(--ink)", border: `1px solid ${deleteConfirm ? "#e5484d" : "var(--border)"}` }}>
            {deleteBusy ? "Deleting..." : deleteConfirm ? "Click again to confirm" : "Undo Today's Imports"}
          </button>
        )}
        <button onClick={handlePull} disabled={pulling}
                className="flex items-center gap-2 h-[36px] px-5 rounded-[10px] text-[13px] font-semibold text-white transition-all disabled:opacity-50"
                style={{ background: navColor }}>
          {pulling ? (
            <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
          ) : (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
          )}
          {pulling ? "Pulling..." : "Pull IBKR Trades"}
        </button>
      </div>

      {error && (
        <div className="mb-5 flex items-center gap-2 px-4 py-2.5 rounded-[10px] text-[12px] font-medium"
             style={{ background: "color-mix(in oklab, #e5484d 10%, var(--surface))", color: "#e5484d", border: "1px solid #e5484d30" }}>
          {error}
        </div>
      )}

      {message && !error && (
        <div className="mb-5 flex items-center gap-2 px-4 py-2.5 rounded-[10px] text-[12px] font-medium"
             style={{ background: "color-mix(in oklab, #08a86b 10%, var(--surface))", color: "#08a86b", border: "1px solid #08a86b30" }}>
          {message}
        </div>
      )}

      {/* Raw IBKR fields inspector — helpful when parser is picking the wrong
          time / date / quantity field for a particular Flex Query config. */}
      {rawDebug && (rawDebug.first || rawDebug.first_opt) && (
        <div className="mb-5 rounded-[10px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
          <button onClick={() => setShowDebug(!showDebug)}
                  className="w-full flex items-center gap-2 px-4 py-2.5 text-left cursor-pointer text-[11px] font-medium"
                  style={{ color: "var(--ink-3)" }}>
            <span className="text-[9px] transition-transform" style={{ transform: showDebug ? "rotate(90deg)" : "none" }}>▶</span>
            Raw IBKR fields (first trade) · {showDebug ? "hide" : "show"}
            <span className="ml-auto text-[10px]" style={{ color: "var(--ink-4)" }}>diagnose parser mismatches</span>
          </button>
          {showDebug && (
            <div className="px-4 pb-3">
              {rawDebug.first && (
                <>
                  <div className="text-[10px] uppercase tracking-[0.08em] font-semibold mt-1 mb-1" style={{ color: "var(--ink-4)" }}>Stock (first row)</div>
                  <pre className="text-[10px] p-2 rounded overflow-x-auto" style={{ background: "var(--bg)", fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink-2)" }}>
{JSON.stringify(rawDebug.first, null, 2)}
                  </pre>
                </>
              )}
              {rawDebug.first_opt && (
                <>
                  <div className="text-[10px] uppercase tracking-[0.08em] font-semibold mt-2 mb-1" style={{ color: "var(--ink-4)" }}>Option (first row)</div>
                  <pre className="text-[10px] p-2 rounded overflow-x-auto" style={{ background: "var(--bg)", fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink-2)" }}>
{JSON.stringify(rawDebug.first_opt, null, 2)}
                  </pre>
                </>
              )}
            </div>
          )}
        </div>
      )}

      {/* Workflow guide when no data */}
      {executions.length === 0 && !pulling && (
        <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
          <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
            <span className="text-[13px] font-semibold">Import Workflow</span>
          </div>
          <div className="p-6">
            <div className="flex flex-col gap-4">
              {[
                { step: "1", title: "Pull Executions", desc: "Fetch today's trade confirmations from IBKR Flex Query" },
                { step: "2", title: "Review & Validate", desc: "Check quantities, prices, and partial fill consolidation. Duplicates flagged automatically." },
                { step: "3", title: "Log to Journal", desc: "Send each row to Log Buy / Log Sell (full form) or use Quick Log (inline, one click)." },
                { step: "4", title: "Mistake?", desc: "\"Undo Today's Imports\" wipes every transaction with today's date and recomputes affected campaigns." },
              ].map(s => (
                <div key={s.step} className="flex items-start gap-3">
                  <div className="w-7 h-7 rounded-full flex items-center justify-center text-[12px] font-semibold text-white shrink-0"
                       style={{ background: navColor }}>
                    {s.step}
                  </div>
                  <div>
                    <div className="text-[13px] font-semibold">{s.title}</div>
                    <div className="text-[12px]" style={{ color: "var(--ink-4)" }}>{s.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Stocks table */}
      {stockRows.length > 0 && (
        <div className="rounded-[14px] overflow-hidden mb-6" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
          <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
            <span className="text-[13px] font-semibold">Stock Executions</span>
            <span className="text-xs" style={{ color: "var(--ink-4)" }}>{stockRows.length} rows</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-[12px]" style={{ borderCollapse: "separate", borderSpacing: 0 }}>
              <thead>
                <tr>
                  {["Time", "Symbol", "Action", "Qty", "Price", "Amount", "Comm.", "Net", "Actions"].map(h => (
                    <th key={h} className="text-left text-[10px] uppercase tracking-[0.08em] font-semibold px-3 py-2.5 whitespace-nowrap"
                        style={{ color: "var(--ink-4)", background: "var(--surface-2)", borderBottom: "1px solid var(--border)" }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {stockRows.map(({ t, i }) => {
                  const dup = duplicateMap.get(i);
                  const rowResult = quickLogResult?.row === i ? quickLogResult : null;
                  const quickOpen = quickLogRow === i;
                  return (
                    <>
                      <tr key={i} style={{ background: dup ? "color-mix(in oklab, #f59f00 8%, var(--surface))" : undefined }}>
                        <td className="px-3 py-2.5" style={{ fontFamily: "var(--font-jetbrains), monospace", fontSize: 11 }}>{t.order_time}</td>
                        <td className="px-3 py-2.5 font-semibold">
                          {t.symbol}
                          {dup && <span className="ml-2 px-1.5 py-0.5 rounded text-[9px] font-semibold" style={{ background: "#f59f00", color: "#fff" }}>DUPLICATE</span>}
                        </td>
                        <td className="px-3 py-2.5">
                          <span className="px-2 py-0.5 rounded-full text-[11px] font-medium"
                                style={{
                                  background: t.action === "BUY"
                                    ? "color-mix(in oklab, #08a86b 14%, var(--surface))"
                                    : "color-mix(in oklab, #e5484d 14%, var(--surface))",
                                  color: t.action === "BUY" ? "#16a34a" : "#ef4444",
                                  border: `1px solid ${t.action === "BUY"
                                    ? "color-mix(in oklab, #08a86b 32%, var(--border))"
                                    : "color-mix(in oklab, #e5484d 32%, var(--border))"}`,
                                }}>
                            {t.action}
                          </span>
                        </td>
                        <td className="px-3 py-2.5 text-right" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{t.quantity}</td>
                        <td className="px-3 py-2.5 text-right privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{fmtDol(t.price)}</td>
                        <td className="px-3 py-2.5 text-right privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{fmtDol(t.amount)}</td>
                        <td className="px-3 py-2.5 text-right" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{fmtDol(t.commission)}</td>
                        <td className="px-3 py-2.5 text-right privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{fmtDol(t.net_cash)}</td>
                        <td className="px-3 py-2.5">
                          <div className="flex gap-1.5">
                            <button onClick={() => t.action === "BUY" ? sendToLogBuy(t) : sendToLogSell(t)}
                                    className="px-2 py-1 rounded text-[10px] font-semibold transition-all hover:brightness-110"
                                    style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }}
                                    title={`Prefill Log ${t.action === "BUY" ? "Buy" : "Sell"} form`}>
                              {t.action === "BUY" ? "→ Log Buy" : "→ Log Sell"}
                            </button>
                            <button onClick={() => openQuickLog(t, i)}
                                    className="px-2 py-1 rounded text-[10px] font-semibold transition-all hover:brightness-110 text-white"
                                    style={{ background: navColor }}
                                    title="Log inline without leaving this page">
                              Quick Log
                            </button>
                          </div>
                        </td>
                      </tr>
                      {rowResult && (
                        <tr>
                          <td colSpan={9} className="px-3 py-2 text-[11px]"
                              style={{ background: rowResult.ok ? "color-mix(in oklab, #08a86b 10%, var(--surface))" : "color-mix(in oklab, #e5484d 10%, var(--surface))",
                                       color: rowResult.ok ? "#08a86b" : "#e5484d" }}>
                            {rowResult.ok ? "✓" : "✗"} {rowResult.msg}
                          </td>
                        </tr>
                      )}
                      {quickOpen && (
                        <tr>
                          <td colSpan={9} style={{ background: "var(--bg)", borderTop: "1px solid var(--border)", borderBottom: "1px solid var(--border)" }}>
                            <div className="p-4 flex flex-wrap items-end gap-3">
                              {t.action === "BUY" && (
                                <div className="flex flex-col gap-1">
                                  <label className="text-[9px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Action</label>
                                  <select value={quickLog.action_type} onChange={e => setQuickLog({ ...quickLog, action_type: e.target.value as any })}
                                          className="h-[32px] px-2 rounded-[8px] text-[12px]"
                                          style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)" }}>
                                    <option value="new">New Campaign</option>
                                    <option value="scalein">Scale-In</option>
                                  </select>
                                </div>
                              )}
                              <div className="flex flex-col gap-1 min-w-[160px]">
                                <label className="text-[9px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>
                                  {t.action === "BUY" ? "Trade ID" : "Open Campaign"}
                                </label>
                                {t.action === "BUY" ? (
                                  <input value={quickLog.trade_id} onChange={e => setQuickLog({ ...quickLog, trade_id: e.target.value })}
                                         className="h-[32px] px-2 rounded-[8px] text-[12px]"
                                         style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: "var(--font-jetbrains), monospace" }} />
                                ) : (
                                  <select value={quickLog.trade_id} onChange={e => setQuickLog({ ...quickLog, trade_id: e.target.value })}
                                          className="h-[32px] px-2 rounded-[8px] text-[12px]"
                                          style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)" }}>
                                    <option value="">— pick campaign —</option>
                                    {openTrades
                                      .filter(o => String(o.ticker || "").toUpperCase() === (isOption(t) ? optionTicker(t) : t.symbol).toUpperCase())
                                      .map(o => <option key={o.trade_id} value={o.trade_id}>{o.trade_id} ({o.shares} shs)</option>)}
                                    {openTrades.length > 0 && <option disabled>─────</option>}
                                    {openTrades.map(o => <option key={`all-${o.trade_id}`} value={o.trade_id}>{o.ticker} · {o.trade_id}</option>)}
                                  </select>
                                )}
                              </div>
                              <div className="flex flex-col gap-1 min-w-[220px]">
                                <label className="text-[9px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Rule</label>
                                <select value={quickLog.rule} onChange={e => setQuickLog({ ...quickLog, rule: e.target.value })}
                                        className="h-[32px] px-2 rounded-[8px] text-[12px]"
                                        style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)" }}>
                                  {(t.action === "BUY" ? BUY_RULES : SELL_RULES).map(r => <option key={r} value={r}>{r}</option>)}
                                </select>
                              </div>
                              {t.action === "BUY" && (
                                <div className="flex flex-col gap-1 w-[100px]">
                                  <label className="text-[9px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Stop $ (opt.)</label>
                                  <input type="number" step="0.01" value={quickLog.stop} onChange={e => setQuickLog({ ...quickLog, stop: e.target.value })}
                                         className="h-[32px] px-2 rounded-[8px] text-[12px]"
                                         style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: "var(--font-jetbrains), monospace" }} />
                                </div>
                              )}
                              <div className="flex flex-col gap-1 flex-1 min-w-[200px]">
                                <label className="text-[9px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Notes</label>
                                <input value={quickLog.notes} onChange={e => setQuickLog({ ...quickLog, notes: e.target.value })}
                                       className="h-[32px] px-2 rounded-[8px] text-[12px]"
                                       style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)" }} />
                              </div>
                              <div className="flex gap-2">
                                <button onClick={() => setQuickLogRow(null)}
                                        className="h-[32px] px-3 rounded-[8px] text-[12px] font-medium"
                                        style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)" }}>
                                  Cancel
                                </button>
                                <button onClick={() => submitQuickLog(t)} disabled={quickLogBusy}
                                        className="h-[32px] px-4 rounded-[8px] text-[12px] font-semibold text-white disabled:opacity-50"
                                        style={{ background: navColor }}>
                                  {quickLogBusy ? "Saving..." : `Log ${t.action}`}
                                </button>
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Options table */}
      {optionRows.length > 0 && (
        <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
          <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: "#f59f00" }} />
            <span className="text-[13px] font-semibold">Options Executions</span>
            <span className="text-xs" style={{ color: "var(--ink-4)" }}>{optionRows.length} rows · contract = qty × 100</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-[12px]" style={{ borderCollapse: "separate", borderSpacing: 0 }}>
              <thead>
                <tr>
                  {["Time", "Underlying", "Contract", "Action", "Contracts", "Price", "Total Cost", "Comm.", "Actions"].map(h => (
                    <th key={h} className="text-left text-[10px] uppercase tracking-[0.08em] font-semibold px-3 py-2.5 whitespace-nowrap"
                        style={{ color: "var(--ink-4)", background: "var(--surface-2)", borderBottom: "1px solid var(--border)" }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {optionRows.map(({ t, i }) => {
                  const dup = duplicateMap.get(i);
                  const totalCost = t.quantity * t.price * 100;
                  return (
                    <tr key={i} style={{ background: dup ? "color-mix(in oklab, #f59f00 8%, var(--surface))" : undefined }}>
                      <td className="px-3 py-2.5" style={{ fontFamily: "var(--font-jetbrains), monospace", fontSize: 11 }}>{t.order_time}</td>
                      <td className="px-3 py-2.5 font-semibold">
                        {t.symbol}
                        {dup && <span className="ml-2 px-1.5 py-0.5 rounded text-[9px] font-semibold" style={{ background: "#f59f00", color: "#fff" }}>DUPLICATE</span>}
                      </td>
                      <td className="px-3 py-2.5 text-[11px]" style={{ color: "var(--ink-3)" }}>{optionLabel(t)}</td>
                      <td className="px-3 py-2.5">
                        <span className="px-2 py-0.5 rounded-full text-[11px] font-medium"
                              style={{
                                background: t.action === "BUY"
                                  ? "color-mix(in oklab, #08a86b 14%, var(--surface))"
                                  : "color-mix(in oklab, #e5484d 14%, var(--surface))",
                                color: t.action === "BUY" ? "#16a34a" : "#ef4444",
                                border: `1px solid ${t.action === "BUY"
                                  ? "color-mix(in oklab, #08a86b 32%, var(--border))"
                                  : "color-mix(in oklab, #e5484d 32%, var(--border))"}`,
                              }}>
                          {t.action}
                        </span>
                      </td>
                      <td className="px-3 py-2.5 text-right" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{t.quantity}</td>
                      <td className="px-3 py-2.5 text-right privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{fmtDol(t.price)}</td>
                      <td className="px-3 py-2.5 text-right privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{fmtDol(totalCost)}</td>
                      <td className="px-3 py-2.5 text-right" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{fmtDol(t.commission)}</td>
                      <td className="px-3 py-2.5">
                        <div className="flex gap-1.5">
                          <button onClick={() => t.action === "BUY" ? sendToLogBuy(t) : sendToLogSell(t)}
                                  className="px-2 py-1 rounded text-[10px] font-semibold transition-all hover:brightness-110"
                                  style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }}>
                            {t.action === "BUY" ? "→ Log Buy" : "→ Log Sell"}
                          </button>
                          <button onClick={() => openQuickLog(t, i)}
                                  className="px-2 py-1 rounded text-[10px] font-semibold transition-all hover:brightness-110 text-white"
                                  style={{ background: navColor }}>
                            Quick Log
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
