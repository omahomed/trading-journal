"use client";

// Campaign Detail page — fill-by-fill ledger across all open stock campaigns.
// Build sequence:
//   Commit 1: route stub + nav rename (merged)
//   Commit 2: page scaffold + KPI strip + data fetch (merged)
//   Commit 3 (this commit): ledger table — 16 sortable cols + Edit col,
//     filter toolbar, sticky totals footer, CSV export, empty state.
//     The Edit pencil is rendered but disabled — Commit 4 wires it to
//     the same separate-window edit flow Trade Journal uses.
//
// Stocks only — option fills (instrument_type='OPTION' or option-shaped
// tickers) are filtered out at the campaign-scope step. Live prices only
// (no manual_price overlay), so batchPrices is called without portfolio.

import { useState, useEffect, useMemo, useCallback } from "react";
import { api, getActivePortfolio, type TradePosition, type TradeDetail, type LotClosure, type TradeDetailsBundle } from "@/lib/api";
import { walkLedger, type LedgerRowInfo } from "@/lib/campaign-detail-walk";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";
import { SELL_RULE_LABELS as SELL_RULES } from "@/lib/trade-rules";

const mono = "var(--font-jetbrains), monospace";

// Buy-rule dropdown list. Mirrors trade-journal.tsx:20-34 — same constant,
// kept inline here to match Trade Journal's edit flow exactly (the user
// confirmed: "exact same as the current trade journal"). When the buy-rule
// taxonomy gets its single-source-of-truth cleanup, both inline copies
// should migrate together to lib/trade-rules.
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

const TILE_GRADIENTS = {
  indigo: "linear-gradient(135deg, #6366f1, #818cf8)",
  blue:   "linear-gradient(135deg, #2563eb, #60a5fa)",
  green:  "linear-gradient(135deg, #10b981, #34d399)",
  pink:   "linear-gradient(135deg, #ec4899, #f472b6)",
  orange: "linear-gradient(135deg, #f97316, #fb923c)",
  red:    "linear-gradient(135deg, #e5484d, #f87171)",
};

function KPITile({ label, value, sub, gradient }: { label: string; value: string; sub: string; gradient: string }) {
  return (
    <div className="relative overflow-hidden rounded-[14px] p-[18px] text-white flex flex-col justify-between min-h-[108px] transition-transform duration-150 hover:scale-[1.01]"
         style={{ background: gradient, boxShadow: "0 4px 14px rgba(0,0,0,0.08)" }}>
      <div className="absolute -right-5 -top-5 w-[100px] h-[100px] rounded-full"
           style={{ background: "radial-gradient(circle, rgba(255,255,255,0.18), transparent 65%)" }} />
      <div className="relative z-10">
        <div className="text-[10px] font-semibold uppercase tracking-[0.12em] opacity-85">{label}</div>
        <div className="text-[28px] font-semibold tracking-tight mt-1 privacy-mask" style={{ fontFamily: mono }}>{value}</div>
      </div>
      <div className="relative z-10 text-[11px] font-medium opacity-80 privacy-mask">{sub}</div>
    </div>
  );
}

function isOptionTicker(t: string): boolean {
  return /^\S+\s+\d{6}\s+\$[0-9.]+(C|P)$/.test(String(t || "").trim());
}

function isStockCampaign(t: TradePosition): boolean {
  const type = String((t as { instrument_type?: string }).instrument_type || "").toUpperCase();
  if (type === "STOCK") return true;
  if (type === "OPTION") return false;
  return !isOptionTicker(t.ticker || "");
}

// ─── Enriched-row shape used by table body + filtering + sorting ────────
// Sells expose blended cost basis (weighted-avg buy_price from
// lot_closures) and the sum of per-pair realized_pl, per the build spec's
// Option B. Buys carry remaining/status from the LIFO walker.
interface LedgerRow {
  detail_id: number;
  trx_id: string;           // "B1" / "A1" / "S1" / …
  trade_id: string;
  ticker: string;
  action: "BUY" | "SELL";
  date: string;             // ISO datetime
  shares: number;
  // Buy: amount; Sell: weighted-avg cost basis from closures.
  cost_basis: number;
  // Buy: null; Sell: detail.amount (the sell price).
  exit_price: number | null;
  // Buy: remaining shares post-LIFO (0..shares); Sell: null.
  remaining: number | null;
  stop_loss: number;
  status: "Open" | "Partial" | "Closed";
  // Buy: null; Sell: sum of lot_closures.realized_pl for this sell_trx_id.
  realized: number | null;
  // Buy: (mark - cost) × remaining × multiplier; Sell: null.
  unrealized: number | null;
  // Buy: open-position return at mark; Sell: realized return on basis.
  ret: number | null;
  // Buy: mark × remaining × multiplier; Sell: shares × exit × multiplier.
  value: number;
  rule: string;
  notes: string;
  multiplier: number;
  mark: number;
}

// Column config — keeps the table body + header in sync.
type ColKey =
  | "trx_id" | "date" | "ticker" | "action" | "status"
  | "shares" | "remaining" | "cost_basis" | "exit_price" | "stop_loss"
  | "value" | "realized" | "unrealized" | "ret" | "rule" | "notes";

interface ColumnDef {
  key: ColKey;
  label: string;
  numeric: boolean;
  align: "left" | "right";
}

const COLUMNS: ColumnDef[] = [
  { key: "trx_id",     label: "TRX ID",         numeric: false, align: "left" },
  { key: "date",       label: "Date",           numeric: false, align: "left" },
  { key: "ticker",     label: "Ticker",         numeric: false, align: "left" },
  { key: "action",     label: "Action",         numeric: false, align: "left" },
  { key: "status",     label: "Status",         numeric: false, align: "left" },
  { key: "shares",     label: "Shares",         numeric: true,  align: "right" },
  { key: "remaining",  label: "Remaining",      numeric: true,  align: "right" },
  { key: "cost_basis", label: "Amount",         numeric: true,  align: "right" },
  { key: "exit_price", label: "Exit Price",     numeric: true,  align: "right" },
  { key: "stop_loss",  label: "Stop Loss",      numeric: true,  align: "right" },
  { key: "value",      label: "Value",          numeric: true,  align: "right" },
  { key: "realized",   label: "Realized P&L",   numeric: true,  align: "right" },
  { key: "unrealized", label: "Unrealized P&L", numeric: true,  align: "right" },
  { key: "ret",        label: "Return %",       numeric: true,  align: "right" },
  { key: "rule",       label: "Rule",           numeric: false, align: "left" },
  { key: "notes",      label: "Notes",          numeric: false, align: "left" },
];
const NUMERIC_KEYS: Set<ColKey> = new Set(COLUMNS.filter(c => c.numeric).map(c => c.key));

// Return-heat chip tier classes — per spec (README §Design Tokens).
function retChipStyle(n: number): { bg: string; color: string } {
  if (n >= 15) return { bg: "#d1fae5", color: "#047857" };
  if (n >= 5)  return { bg: "#ecfdf4", color: "#047857" };
  if (n > 0)   return { bg: "#f0fdf4", color: "#15803d" };
  if (n > -5)  return { bg: "#fef2f2", color: "#b91c1b" };
  return       { bg: "#fde7e8", color: "#991b1b" };
}

function pctColor(n: number) { return n > 0 ? "#08a86b" : n < 0 ? "#e5484d" : "var(--ink-3)"; }

// CSV cell escape — quotes any cell containing comma, quote, or newline.
function csvCell(v: string | number | null): string {
  if (v == null) return "";
  const s = String(v);
  if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

interface Filters {
  q: string;
  series: "all" | "B" | "A";
  action: "all" | "BUY" | "SELL";
  status: "all" | "Open" | "Partial" | "Closed";
  ticker: string;        // ticker or "all"
  rule: string;          // rule or "all"
  pl: "all" | "realized" | "unrealized";
  from: string;          // YYYY-MM-DD
  to: string;            // YYYY-MM-DD
}
const EMPTY_FILTERS: Filters = {
  q: "", series: "all", action: "all", status: "all",
  ticker: "all", rule: "all", pl: "all", from: "", to: "",
};

export function CampaignDetail({ navColor }: { navColor: string }) {
  const [openTrades, setOpenTrades] = useState<TradePosition[]>([]);
  const [details, setDetails] = useState<TradeDetail[]>([]);
  const [closures, setClosures] = useState<LotClosure[]>([]);
  const [livePrices, setLivePrices] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdatedAt, setLastUpdatedAt] = useState<Date | null>(null);
  const [sort, setSort] = useState<{ key: ColKey; dir: "asc" | "desc" }>({ key: "date", dir: "desc" });
  const [filters, setFilters] = useState<Filters>(EMPTY_FILTERS);

  // ── Edit modal state ───────────────────────────────────────────────
  // Same shape Trade Journal uses (trade-journal.tsx:441-452). The
  // user's clarification was explicit: "the edit should be the same
  // function that we have in trade journal. exact same. with open a
  // separate window to edit a transactions." So we mirror the modal
  // chrome + Save/Delete handlers verbatim and add only one extension:
  // a read-only "Lots consumed" subtable for Sell rows (Option B from
  // the audit), wired off lot_closures.
  const [editingTxn, setEditingTxn] = useState<TradeDetail | null>(null);
  const [editForm, setEditForm] = useState<{
    date: string;
    shares: string;
    amount: string;
    stop_loss: string;
    rule: string;
    notes: string;
  }>({ date: "", shares: "", amount: "", stop_loss: "", rule: "", notes: "" });
  const [editError, setEditError] = useState<string | null>(null);
  const [editLoading, setEditLoading] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  const fetchAll = useCallback(async () => {
    setError(null);
    try {
      const [openRaw, bundleRaw] = await Promise.all([
        api.tradesOpen(getActivePortfolio()).catch(err => {
          log.error("campaign-detail", "tradesOpen failed", err);
          return [] as TradePosition[];
        }),
        api.tradesOpenDetails(getActivePortfolio()).catch(err => {
          log.error("campaign-detail", "tradesOpenDetails failed", err);
          return { details: [], lot_closures: [] } as TradeDetailsBundle;
        }),
      ]);

      const allOpen = openRaw as TradePosition[];
      const stockOpen = allOpen.filter(isStockCampaign);
      const stockIds = new Set(stockOpen.map(t => String(t.trade_id || "")));
      const stockDetails = (bundleRaw.details || []).filter(d => stockIds.has(String(d.trade_id || "")));
      const stockClosures = (bundleRaw.lot_closures || []).filter(c => stockIds.has(String(c.trade_id || "")));

      const tickers = Array.from(new Set(stockOpen.map(t => t.ticker).filter(Boolean) as string[]));
      let prices: Record<string, number> = {};
      if (tickers.length > 0) {
        try {
          const r = await api.batchPrices(tickers);
          if (r && !("error" in r)) prices = r as Record<string, number>;
        } catch (e) {
          log.error("campaign-detail", "batchPrices failed", e);
        }
      }

      setOpenTrades(stockOpen);
      setDetails(stockDetails);
      setClosures(stockClosures);
      setLivePrices(prices);
      setLastUpdatedAt(new Date());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchAll().finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [fetchAll]);

  const onRefresh = useCallback(async () => {
    if (refreshing) return;
    setRefreshing(true);
    await fetchAll();
    setRefreshing(false);
  }, [fetchAll, refreshing]);

  // ── Edit modal handlers ────────────────────────────────────────────
  // openEditModal mirrors trade-journal.tsx:454-467: populate editForm
  // from the chosen detail row, clear any prior error / delete-confirm
  // state. The pencil button on each row passes its detail_id; we look
  // up the raw TradeDetail in state.
  const openEditModal = useCallback((detailId: number) => {
    const tx = details.find(d => Number((d as { detail_id?: number }).detail_id) === detailId);
    if (!tx) return;
    setEditingTxn(tx);
    setEditForm({
      date: String(tx.date || "").slice(0, 16),
      shares: String(tx.shares ?? ""),
      amount: String(tx.amount ?? ""),
      stop_loss: String((tx as { stop_loss?: number }).stop_loss ?? ""),
      rule: tx.rule || "",
      notes: String((tx as { notes?: string }).notes ?? ""),
    });
    setEditError(null);
    setConfirmingDelete(false);
    setEditLoading(false);
  }, [details]);

  const closeEditModal = useCallback(() => {
    setEditingTxn(null);
    setEditError(null);
    setConfirmingDelete(false);
    setEditLoading(false);
  }, []);

  // ESC dismisses the modal (gated on !editLoading so an in-flight
  // save/delete can't be cancelled mid-request) — matches trade-
  // journal.tsx:609-616.
  useEffect(() => {
    if (!editingTxn) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !editLoading) closeEditModal();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [editingTxn, editLoading, closeEditModal]);

  const saveEdit = useCallback(async () => {
    if (!editingTxn) return;
    setEditLoading(true);
    setEditError(null);
    try {
      const shares = parseFloat(editForm.shares || "0") || 0;
      const amount = parseFloat(editForm.amount || "0") || 0;
      const res = await api.editTransaction({
        detail_id: Number((editingTxn as { detail_id?: number }).detail_id),
        trade_id: editingTxn.trade_id,
        ticker: editingTxn.ticker,
        action: editingTxn.action,
        date: editForm.date,
        shares,
        amount,
        value: shares * amount,
        rule: editForm.rule,
        notes: editForm.notes,
        stop_loss: parseFloat(editForm.stop_loss || "0") || 0,
        trx_id: String((editingTxn as { trx_id?: string }).trx_id || ""),
      });
      if (res.error) {
        setEditError(res.error);
      } else {
        await fetchAll();
        closeEditModal();
      }
    } catch (err: unknown) {
      setEditError(err instanceof Error ? err.message : String(err));
    } finally {
      setEditLoading(false);
    }
  }, [editingTxn, editForm, fetchAll, closeEditModal]);

  const deleteTxn = useCallback(async () => {
    if (!editingTxn) return;
    // Two-click confirm — same pattern as trade-journal.tsx:580-583.
    if (!confirmingDelete) {
      setConfirmingDelete(true);
      return;
    }
    setEditLoading(true);
    setEditError(null);
    try {
      const res = await api.deleteTransaction(
        Number((editingTxn as { detail_id?: number }).detail_id),
        editingTxn.trade_id,
        editingTxn.ticker,
      );
      if (res && (res as { error?: string }).error) {
        setEditError((res as { error: string }).error);
      } else {
        await fetchAll();
        closeEditModal();
      }
    } catch (err: unknown) {
      setEditError(err instanceof Error ? err.message : String(err));
    } finally {
      setEditLoading(false);
    }
  }, [editingTxn, confirmingDelete, fetchAll, closeEditModal]);

  // Walker — per-detail status + remaining.
  const walked = useMemo(() => walkLedger(details), [details]);

  // Closure aggregation per sell_trx_id (composite with trade_id since
  // trx_ids repeat across campaigns). Used for Option B: a Sell row's
  // cost basis = weighted-avg buy_price; realized = sum across closures.
  const closureBySell = useMemo(() => {
    const map = new Map<string, { sumPl: number; sumShares: number; sumBuyCost: number }>();
    for (const c of closures) {
      const tid = String(c.trade_id || "");
      const sellTrx = String((c as { sell_trx_id?: string }).sell_trx_id || "");
      if (!tid || !sellTrx) continue;
      const key = `${tid}|${sellTrx}`;
      const shares = parseFloat(String(c.shares || 0)) || 0;
      const buyPrice = parseFloat(String((c as { buy_price?: number | string }).buy_price || 0)) || 0;
      const pl = parseFloat(String(c.realized_pl || 0)) || 0;
      const cur = map.get(key) || { sumPl: 0, sumShares: 0, sumBuyCost: 0 };
      cur.sumPl += pl;
      cur.sumShares += shares;
      cur.sumBuyCost += shares * buyPrice;
      map.set(key, cur);
    }
    return map;
  }, [closures]);

  // Closure aggregation per buy_trx_id — the Buy-side mirror. A Buy lot
  // that's been sold off (fully or partially) has its realized P&L
  // attributed via the closures' buy_trx_id field; we use that to surface
  // the realized return on the Buy row itself rather than computing a
  // misleading mark-based return. Without this, a fully-closed Buy lot
  // shows +146% (current price vs entry) when the actual realized return
  // was −2.77% (sell price vs entry).
  const closureByBuy = useMemo(() => {
    const map = new Map<string, { sumPl: number; sumShares: number; sumSellRevenue: number }>();
    for (const c of closures) {
      const tid = String(c.trade_id || "");
      const buyTrx = String((c as { buy_trx_id?: string }).buy_trx_id || "");
      if (!tid || !buyTrx) continue;
      const key = `${tid}|${buyTrx}`;
      const shares = parseFloat(String(c.shares || 0)) || 0;
      const sellPrice = parseFloat(String((c as { sell_price?: number | string }).sell_price || 0)) || 0;
      const pl = parseFloat(String(c.realized_pl || 0)) || 0;
      const cur = map.get(key) || { sumPl: 0, sumShares: 0, sumSellRevenue: 0 };
      cur.sumPl += pl;
      cur.sumShares += shares;
      cur.sumSellRevenue += shares * sellPrice;
      map.set(key, cur);
    }
    return map;
  }, [closures]);

  // Per-trade multiplier lookup (campaigns are stocks so this should
  // always resolve to 1, but the field exists on the row and we honor it).
  const tradeMultiplier = useMemo(() => {
    const m = new Map<string, number>();
    for (const t of openTrades) {
      const v = parseFloat(String((t as { multiplier?: string | number }).multiplier ?? 0));
      m.set(String(t.trade_id), v > 0 ? v : 1);
    }
    return m;
  }, [openTrades]);

  // ───── Enriched rows ────────────────────────────────────────────────
  const enrichedRows = useMemo<LedgerRow[]>(() => {
    return details.map(d => {
      const action = String(d.action || "").toUpperCase() === "SELL" ? "SELL" : "BUY";
      const detailId = Number((d as { detail_id?: number }).detail_id ?? -1);
      const info: LedgerRowInfo | undefined = walked.perDetail.get(detailId);
      const tradeId = String(d.trade_id || "");
      const trxId = String(d.trx_id || "");
      const ticker = String(d.ticker || "");
      const shares = parseFloat(String(d.shares || 0)) || 0;
      const amount = parseFloat(String(d.amount || 0)) || 0;
      const stopLoss = parseFloat(String(d.stop_loss || 0)) || 0;
      const mult = tradeMultiplier.get(tradeId) ?? 1;
      const mark = livePrices[ticker] ?? amount;

      let cost_basis: number;
      let exit_price: number | null;
      let remaining: number | null;
      let realized: number | null;
      let unrealized: number | null;
      let ret: number | null;
      let value: number;

      if (action === "SELL") {
        const key = `${tradeId}|${trxId}`;
        const agg = closureBySell.get(key);
        cost_basis = agg && agg.sumShares > 0 ? agg.sumBuyCost / agg.sumShares : amount;
        exit_price = amount;
        remaining = null;
        realized = agg ? agg.sumPl : 0;
        unrealized = null;
        // Sell rows no longer carry a Return %. The realized $ already
        // captures the outcome of the sale; the % was redundant and the
        // user asked for it removed.
        ret = null;
        value = shares * amount * mult;
      } else {
        cost_basis = amount;
        remaining = info ? info.remaining : shares;
        const rem = remaining ?? 0;
        // Look up the closures attributed to THIS Buy lot. Used both to
        // surface the realized P&L on the row (matching Trade Journal)
        // and to drive the realized-return branch when the lot is fully
        // closed (remaining = 0). Without the latter, a fully-closed Buy
        // would render +146 % (mark vs entry) instead of the actual
        // realized −2.77 % (avg sell vs entry).
        const buyKey = `${tradeId}|${trxId}`;
        const buyAgg = closureByBuy.get(buyKey);
        const avgSellPrice = buyAgg && buyAgg.sumShares > 0
          ? buyAgg.sumSellRevenue / buyAgg.sumShares
          : null;

        realized = buyAgg ? buyAgg.sumPl : 0;
        exit_price = avgSellPrice;
        unrealized = rem > 0 ? (mark - amount) * rem * mult : 0;
        // Value: cost-basis value of the lot (matches Trade Journal's
        // Value column). Previous formula `mark × remaining × mult`
        // collapsed to $0 for fully-closed Buys, hiding the lot's cost
        // basis entirely from the row.
        value = shares * amount * mult;
        // Return %: mark-based while shares remain open; realized when
        // the lot is fully closed by sells.
        if (rem > 0) {
          ret = amount > 0 ? ((mark - amount) / amount) * 100 : 0;
        } else if (avgSellPrice != null && amount > 0) {
          ret = ((avgSellPrice - amount) / amount) * 100;
        } else {
          ret = 0;
        }
      }

      const status: "Open" | "Partial" | "Closed" = info ? info.status : (action === "SELL" ? "Closed" : "Open");

      return {
        detail_id: detailId,
        trx_id: trxId,
        trade_id: tradeId,
        ticker,
        action,
        date: String(d.date || ""),
        shares,
        cost_basis,
        exit_price,
        remaining,
        stop_loss: stopLoss,
        status,
        realized,
        unrealized,
        ret,
        value,
        rule: String(d.rule || ""),
        notes: String(d.notes || ""),
        multiplier: mult,
        mark,
      };
    });
  }, [details, walked, closureBySell, closureByBuy, tradeMultiplier, livePrices]);

  // ───── KPI numbers (whole ledger, unfiltered) ───────────────────────
  const kpis = useMemo(() => {
    let unrealized = 0;
    let marketValue = 0;
    let realized = 0;
    for (const r of enrichedRows) {
      if (r.action === "BUY" && r.remaining && r.remaining > 0) {
        unrealized += r.unrealized ?? 0;
        // Market Value tile keeps its "current market value of OPEN
        // positions" semantic — `mark × remaining × multiplier` — even
        // though the row-level Value column now shows cost basis to
        // match Trade Journal. Computed directly here so the two
        // surfaces stay decoupled.
        marketValue += r.mark * r.remaining * r.multiplier;
      }
      if (r.action === "SELL") {
        // Realized P&L tile sums only Sell-side closures. Sums on Buys
        // would double-count (every closure pair shows up on both
        // sides post-fix).
        realized += r.realized ?? 0;
      }
    }
    return {
      transactionCount: enrichedRows.length,
      activeCampaignCount: openTrades.length,
      openLotCount: walked.openLotCount,
      realized,
      unrealized,
      marketValue,
    };
  }, [enrichedRows, openTrades.length, walked.openLotCount]);

  // ───── Filter dropdown options (derived from full ledger) ───────────
  const tickerOptions = useMemo(
    () => Array.from(new Set(enrichedRows.map(r => r.ticker).filter(Boolean))).sort(),
    [enrichedRows],
  );
  const ruleOptions = useMemo(
    () => Array.from(new Set(enrichedRows.map(r => r.rule).filter(Boolean))).sort(),
    [enrichedRows],
  );

  // ───── Filter ───────────────────────────────────────────────────────
  const filtered = useMemo(() => {
    const q = filters.q.trim().toLowerCase();
    return enrichedRows.filter(r => {
      if (q) {
        const haystack = [r.ticker, r.trx_id, r.rule, r.notes].map(v => v.toLowerCase());
        if (!haystack.some(s => s.includes(q))) return false;
      }
      if (filters.series !== "all" && r.trx_id.charAt(0).toUpperCase() !== filters.series) return false;
      if (filters.action !== "all" && r.action !== filters.action) return false;
      if (filters.status !== "all" && r.status !== filters.status) return false;
      if (filters.ticker !== "all" && r.ticker !== filters.ticker) return false;
      if (filters.rule !== "all" && r.rule !== filters.rule) return false;
      if (filters.pl === "realized" && r.action !== "SELL") return false;
      if (filters.pl === "unrealized" && !(r.action === "BUY" && (r.remaining ?? 0) > 0)) return false;
      const d = r.date.slice(0, 10);
      if (filters.from && d < filters.from) return false;
      if (filters.to && d > filters.to) return false;
      return true;
    });
  }, [enrichedRows, filters]);

  // ───── Sort (nulls last regardless of direction) ────────────────────
  const sorted = useMemo(() => {
    const { key, dir } = sort;
    const numeric = NUMERIC_KEYS.has(key);
    return [...filtered].sort((a, b) => {
      const va = (a as unknown as Record<string, unknown>)[key];
      const vb = (b as unknown as Record<string, unknown>)[key];
      const an = va == null;
      const bn = vb == null;
      if (an && bn) return 0;
      if (an) return 1;
      if (bn) return -1;
      const cmp = numeric
        ? (Number(va) - Number(vb))
        : String(va).localeCompare(String(vb));
      return dir === "asc" ? cmp : -cmp;
    });
  }, [filtered, sort]);

  // ───── Totals (visible filtered+sorted set) ─────────────────────────
  // Realized is summed from Sells only — Buy rows now carry their own
  // attributed realized too, and summing both sides would double-count
  // (each lot_closures row contributes equally to a Buy AND a Sell).
  // Unrealized naturally lives on Buys only (Sells are null).
  const totals = useMemo(() => sorted.reduce((t, r) => {
    t.shares += r.shares;
    t.remaining += r.remaining ?? 0;
    t.value += r.value;
    if (r.action === "SELL") t.realized += r.realized ?? 0;
    t.unrealized += r.unrealized ?? 0;
    return t;
  }, { shares: 0, remaining: 0, value: 0, realized: 0, unrealized: 0 }), [sorted]);

  const onSort = (key: ColKey) => {
    setSort(s => s.key === key
      ? { key, dir: s.dir === "asc" ? "desc" : "asc" }
      : { key, dir: NUMERIC_KEYS.has(key) ? "desc" : "asc" });
  };

  // Reset disabled when no filter is active.
  const filtersDirty = useMemo(() => (
    !!filters.q || filters.series !== "all" || filters.action !== "all"
    || filters.status !== "all" || filters.ticker !== "all" || filters.rule !== "all"
    || filters.pl !== "all" || !!filters.from || !!filters.to
  ), [filters]);
  const resetFilters = () => setFilters(EMPTY_FILTERS);

  // ───── CSV export (filtered + sorted, raw values) ───────────────────
  const onExportCsv = useCallback(() => {
    const header = COLUMNS.map(c => c.label).join(",");
    const rows = sorted.map(r => COLUMNS.map(c => {
      const v: unknown = (r as unknown as Record<string, unknown>)[c.key];
      if (v == null) return "";
      if (typeof v === "number") return String(v);
      return csvCell(String(v));
    }).join(","));
    const csv = [header, ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `campaign-detail-${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [sorted]);

  // ───── UI ───────────────────────────────────────────────────────────
  const lastUpdatedLabel = lastUpdatedAt
    ? lastUpdatedAt.toISOString().slice(0, 16).replace("T", " ")
    : "";
  const uniqueTickerCount = useMemo(
    () => new Set(enrichedRows.map(r => r.ticker).filter(Boolean)).size,
    [enrichedRows],
  );

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }} data-testid="campaign-detail-root">
      {/* Page header */}
      <div className="mb-[22px] pb-[14px] flex items-end justify-between gap-4"
           style={{ borderBottom: "1px solid var(--border)" }}>
        <div>
          <h1 className="font-normal text-[32px] tracking-tight m-0"
              style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
            Active Campaign <em className="italic" style={{ color: navColor }}>Detail</em>
          </h1>
          <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
            Every fill across all open stock campaigns
            {lastUpdatedLabel ? ` · as of ${lastUpdatedLabel}` : ""}
          </div>
        </div>
        <div className="flex gap-2 shrink-0">
          <button type="button" onClick={onExportCsv}
                  disabled={enrichedRows.length === 0}
                  data-testid="export-csv-btn"
                  className="px-3 py-2 rounded-[10px] text-[13px] flex items-center gap-1.5 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-2)" }}>
            ↓ Export CSV
          </button>
          <button type="button" onClick={onRefresh} disabled={refreshing}
                  data-testid="refresh-btn"
                  className="px-3 py-2 rounded-[10px] text-[13px] flex items-center gap-1.5 transition-colors"
                  style={{ background: "var(--surface)", border: "1px solid var(--border)", color: refreshing ? "var(--ink-4)" : "var(--ink-2)" }}>
            ⟳ {refreshing ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 px-4 py-3 rounded-[10px]"
             data-testid="campaign-detail-error"
             style={{ background: "color-mix(in oklab, #e5484d 8%, var(--surface))", border: "1px solid var(--border)", color: "#e5484d" }}>
          Failed to load: {error}
        </div>
      )}

      {/* KPI strip */}
      {loading && !lastUpdatedAt ? (
        <div className="grid grid-cols-5 gap-[14px]" data-testid="kpi-strip-loading">
          {[0, 1, 2, 3, 4].map(i => (
            <div key={i} className="rounded-[14px] animate-pulse min-h-[108px]"
                 style={{ background: "var(--bg-2)" }} />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-5 gap-[14px]" data-testid="kpi-strip">
          <KPITile label="Transactions" value={String(kpis.transactionCount)}
                   sub={`${kpis.activeCampaignCount} active campaigns`} gradient={TILE_GRADIENTS.indigo} />
          <KPITile label="Open Lots" value={String(kpis.openLotCount)}
                   sub="held, unrealized" gradient={TILE_GRADIENTS.blue} />
          <KPITile label="Realized P&L" value={formatCurrency(kpis.realized, { decimals: 0 })}
                   sub="closed trims" gradient={kpis.realized >= 0 ? TILE_GRADIENTS.green : TILE_GRADIENTS.red} />
          <KPITile label="Unrealized P&L" value={formatCurrency(kpis.unrealized, { decimals: 0 })}
                   sub="open at mark" gradient={kpis.unrealized >= 0 ? TILE_GRADIENTS.pink : TILE_GRADIENTS.red} />
          <KPITile label="Market Value" value={formatCurrency(kpis.marketValue, { decimals: 0 })}
                   sub="open positions" gradient={TILE_GRADIENTS.orange} />
        </div>
      )}

      {/* Card: Transaction Ledger */}
      <div className="mt-5 rounded-[14px] overflow-hidden"
           data-testid="ledger-card"
           style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "0 1px 2px rgba(14,20,38,0.04)" }}>
        {/* Card header strip */}
        <div className="px-[18px] py-[14px] flex items-center gap-2"
             style={{ borderBottom: "1px solid var(--border)" }}>
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
          <span className="text-[13px] font-semibold">Transaction Ledger</span>
          <span className="text-[12px]" style={{ color: "var(--ink-4)" }}>
            {enrichedRows.length} fills · {uniqueTickerCount} tickers
          </span>
        </div>

        {/* Filter toolbar */}
        <div className="px-[18px] py-[14px] flex flex-wrap items-end gap-[12px_14px]"
             style={{ background: "var(--bg-2)", borderBottom: "1px solid var(--border)" }}>
          {/* Search */}
          <div className="flex flex-col gap-1" style={{ flex: "1 1 220px", minWidth: 200 }}>
            <span className="text-[9px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>Search</span>
            <div className="relative">
              <input type="text" value={filters.q}
                     onChange={e => setFilters(f => ({ ...f, q: e.target.value }))}
                     placeholder="Ticker, TRX ID, rule or note…"
                     data-testid="filter-q"
                     className="w-full h-[34px] pl-9 pr-8 rounded-[10px] text-[12px]"
                     style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)" }} />
              <span className="absolute left-3 top-1/2 -translate-y-1/2 text-[12px]" style={{ color: "var(--ink-4)" }}>⌕</span>
              {filters.q && (
                <button type="button" onClick={() => setFilters(f => ({ ...f, q: "" }))}
                        className="absolute right-2 top-1/2 -translate-y-1/2 px-1 text-[12px]"
                        style={{ color: "var(--ink-4)" }}>✕</button>
              )}
            </div>
          </div>

          {/* Series */}
          <SegmentedControl
            label="Series"
            value={filters.series}
            onChange={v => setFilters(f => ({ ...f, series: v as Filters["series"] }))}
            options={[{ v: "all", l: "All" }, { v: "B", l: "B · Original" }, { v: "A", l: "A · Add-on" }]}
            testId="filter-series"
          />

          {/* Action */}
          <SegmentedControl
            label="Action"
            value={filters.action}
            onChange={v => setFilters(f => ({ ...f, action: v as Filters["action"] }))}
            options={[{ v: "all", l: "All" }, { v: "BUY", l: "Buy" }, { v: "SELL", l: "Sell" }]}
            testId="filter-action"
          />

          {/* P&L */}
          <SegmentedControl
            label="P&L"
            value={filters.pl}
            onChange={v => setFilters(f => ({ ...f, pl: v as Filters["pl"] }))}
            options={[{ v: "all", l: "All" }, { v: "realized", l: "Realized" }, { v: "unrealized", l: "Unrealized" }]}
            testId="filter-pl"
          />

          {/* Status */}
          <FilterSelect
            label="Status"
            value={filters.status}
            onChange={v => setFilters(f => ({ ...f, status: v as Filters["status"] }))}
            options={[
              { v: "all", l: "All status" },
              { v: "Open", l: "Open" },
              { v: "Partial", l: "Partial" },
              { v: "Closed", l: "Closed" },
            ]}
            testId="filter-status"
          />

          {/* Ticker */}
          <FilterSelect
            label="Ticker"
            value={filters.ticker}
            onChange={v => setFilters(f => ({ ...f, ticker: v }))}
            options={[{ v: "all", l: "All tickers" }, ...tickerOptions.map(t => ({ v: t, l: t }))]}
            testId="filter-ticker"
          />

          {/* Rule */}
          <FilterSelect
            label="Rule"
            value={filters.rule}
            onChange={v => setFilters(f => ({ ...f, rule: v }))}
            options={[{ v: "all", l: "All rules" }, ...ruleOptions.map(r => ({ v: r, l: r }))]}
            testId="filter-rule"
          />

          {/* Date range */}
          <div className="flex flex-col gap-1">
            <span className="text-[9px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>Date range</span>
            <div className="flex items-center gap-2">
              <input type="date" value={filters.from}
                     onChange={e => setFilters(f => ({ ...f, from: e.target.value }))}
                     data-testid="filter-from"
                     className="h-[34px] px-2 rounded-[10px] text-[12px]"
                     style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }} />
              <span className="text-[12px]" style={{ color: "var(--ink-4)" }}>–</span>
              <input type="date" value={filters.to}
                     onChange={e => setFilters(f => ({ ...f, to: e.target.value }))}
                     data-testid="filter-to"
                     className="h-[34px] px-2 rounded-[10px] text-[12px]"
                     style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }} />
            </div>
          </div>

          {/* Tail: match count + Reset */}
          <div className="ml-auto flex items-center gap-3">
            <span className="text-[11px]" style={{ color: "var(--ink-4)", fontFamily: mono }}>
              <b>{sorted.length}</b> of {enrichedRows.length}
            </span>
            <button type="button" onClick={resetFilters} disabled={!filtersDirty}
                    data-testid="filter-reset"
                    className="text-[11px] font-medium disabled:opacity-40 disabled:cursor-not-allowed"
                    style={{ color: filtersDirty ? navColor : "var(--ink-4)" }}>
              ✕ Reset
            </button>
          </div>
        </div>

        {/* Ledger table */}
        <div className="overflow-auto" style={{ maxHeight: "clamp(340px, calc(100vh - 392px), 760px)" }}>
          <table className="w-full text-[11.5px]" data-testid="ledger-table" style={{ borderCollapse: "collapse", whiteSpace: "nowrap" }}>
            <thead>
              <tr>
                {COLUMNS.map(c => {
                  const active = sort.key === c.key;
                  const caret = active ? (sort.dir === "asc" ? "▲" : "▼") : "";
                  return (
                    <th key={c.key}
                        onClick={() => onSort(c.key)}
                        data-testid={`th-${c.key}`}
                        className="px-3 py-2 text-[9.5px] font-bold uppercase tracking-[0.08em] cursor-pointer select-none sticky top-0"
                        style={{
                          background: "var(--surface-2)", zIndex: 3,
                          color: active ? navColor : "var(--ink-4)",
                          textAlign: c.align,
                          borderBottom: "1px solid var(--border)",
                        }}>
                      {c.label}{caret ? ` ${caret}` : ""}
                    </th>
                  );
                })}
                {/* Edit column — header has no sort. */}
                <th className="px-3 py-2 text-[9.5px] font-bold uppercase tracking-[0.08em] sticky top-0"
                    style={{ background: "var(--surface-2)", zIndex: 3, color: "var(--ink-4)", textAlign: "right", borderBottom: "1px solid var(--border)" }}>
                  Edit
                </th>
              </tr>
            </thead>
            <tbody>
              {sorted.length === 0 ? (
                <tr>
                  <td colSpan={COLUMNS.length + 1}
                      data-testid="ledger-empty-state"
                      className="px-3 py-14 text-center text-[12px]"
                      style={{ color: "var(--ink-4)" }}>
                    {enrichedRows.length === 0
                      ? "No fills loaded yet. Adjust the date range or refresh."
                      : `No transactions match these filters. Reset to see all ${enrichedRows.length} fills.`}
                  </td>
                </tr>
              ) : sorted.map(r => (
                <LedgerRowEl key={`${r.detail_id}-${r.trx_id}`}
                             r={r}
                             navColor={navColor}
                             onEdit={openEditModal}
                             isEditing={editingTxn != null && Number((editingTxn as { detail_id?: number }).detail_id) === r.detail_id} />
              ))}
            </tbody>
            {sorted.length > 0 && (
              <tfoot>
                <tr style={{ background: "var(--bg-2)" }}>
                  <td className="px-3 py-2 text-[9.5px] font-bold uppercase tracking-[0.06em] sticky bottom-0"
                      colSpan={5}
                      style={{ background: "var(--bg-2)", color: "var(--ink-3)", borderTop: "2px solid var(--border-2)" }}>
                    Totals · {sorted.length} rows
                  </td>
                  <td className="px-3 py-2 text-right font-bold sticky bottom-0"
                      data-testid="footer-shares"
                      style={{ background: "var(--bg-2)", fontFamily: mono, borderTop: "2px solid var(--border-2)" }}>
                    {Math.round(totals.shares).toLocaleString()}
                  </td>
                  <td className="px-3 py-2 text-right font-bold sticky bottom-0"
                      data-testid="footer-remaining"
                      style={{ background: "var(--bg-2)", fontFamily: mono, borderTop: "2px solid var(--border-2)" }}>
                    {Math.round(totals.remaining).toLocaleString()}
                  </td>
                  <td className="sticky bottom-0" style={{ background: "var(--bg-2)", borderTop: "2px solid var(--border-2)" }} />
                  <td className="sticky bottom-0" style={{ background: "var(--bg-2)", borderTop: "2px solid var(--border-2)" }} />
                  <td className="sticky bottom-0" style={{ background: "var(--bg-2)", borderTop: "2px solid var(--border-2)" }} />
                  <td className="px-3 py-2 text-right font-bold privacy-mask sticky bottom-0"
                      data-testid="footer-value"
                      style={{ background: "var(--bg-2)", fontFamily: mono, borderTop: "2px solid var(--border-2)" }}>
                    {formatCurrency(totals.value, { decimals: 0 })}
                  </td>
                  <td className="px-3 py-2 text-right font-bold privacy-mask sticky bottom-0"
                      data-testid="footer-realized"
                      style={{ background: "var(--bg-2)", fontFamily: mono, color: pctColor(totals.realized), borderTop: "2px solid var(--border-2)" }}>
                    {formatCurrency(totals.realized, { showSign: true, decimals: 0 })}
                  </td>
                  <td className="px-3 py-2 text-right font-bold privacy-mask sticky bottom-0"
                      data-testid="footer-unrealized"
                      style={{ background: "var(--bg-2)", fontFamily: mono, color: pctColor(totals.unrealized), borderTop: "2px solid var(--border-2)" }}>
                    {formatCurrency(totals.unrealized, { showSign: true, decimals: 0 })}
                  </td>
                  <td className="sticky bottom-0" style={{ background: "var(--bg-2)", borderTop: "2px solid var(--border-2)" }} />
                  <td className="sticky bottom-0" style={{ background: "var(--bg-2)", borderTop: "2px solid var(--border-2)" }} />
                  <td className="sticky bottom-0" style={{ background: "var(--bg-2)", borderTop: "2px solid var(--border-2)" }} />
                  <td className="sticky bottom-0" style={{ background: "var(--bg-2)", borderTop: "2px solid var(--border-2)" }} />
                </tr>
              </tfoot>
            )}
          </table>
        </div>
      </div>

      {/* Edit modal — mirrors trade-journal.tsx:1491-1623 chrome + Save/
          Delete handlers, plus the per-Sell "Lots consumed" subtable
          (Option B) directly below the form when editing a SELL row. */}
      {editingTxn && (
        <EditModal
          editingTxn={editingTxn}
          editForm={editForm}
          editError={editError}
          editLoading={editLoading}
          confirmingDelete={confirmingDelete}
          closures={closures}
          navColor={navColor}
          setEditForm={setEditForm}
          onClose={closeEditModal}
          onSave={saveEdit}
          onDelete={deleteTxn}
        />
      )}
    </div>
  );
}

// ─── Sub-components ──────────────────────────────────────────────────

interface EditModalProps {
  editingTxn: TradeDetail;
  editForm: { date: string; shares: string; amount: string; stop_loss: string; rule: string; notes: string };
  editError: string | null;
  editLoading: boolean;
  confirmingDelete: boolean;
  closures: LotClosure[];
  navColor: string;
  setEditForm: React.Dispatch<React.SetStateAction<{ date: string; shares: string; amount: string; stop_loss: string; rule: string; notes: string }>>;
  onClose: () => void;
  onSave: () => void;
  onDelete: () => void;
}

function EditModal({ editingTxn, editForm, editError, editLoading, confirmingDelete, closures, navColor, setEditForm, onClose, onSave, onDelete }: EditModalProps) {
  const isSell = String(editingTxn.action).toUpperCase() === "SELL";
  const trxId = String((editingTxn as { trx_id?: string }).trx_id || "");
  const detailId = (editingTxn as { detail_id?: number }).detail_id;
  // For Option B: when editing a Sell row, surface the lot_closures the
  // matching engine wrote so the trader can audit cost-basis attribution.
  // Read-only — re-running the match is out of scope (and is the
  // explicit guidance in the handoff README).
  const consumedLots = isSell ? closures.filter(c =>
    String(c.trade_id) === editingTxn.trade_id
    && String((c as { sell_trx_id?: string }).sell_trx_id || "") === trxId
  ) : [];

  return (
    <div data-testid="cd-edit-modal-backdrop"
         className="fixed inset-0 z-[100] grid place-items-start justify-center pt-[10vh]"
         style={{ background: "rgba(0,0,0,0.4)", backdropFilter: "blur(4px)" }}
         onClick={() => { if (!editLoading) onClose(); }}>
      <div className="w-[640px] max-w-[92vw] rounded-[14px] overflow-hidden"
           data-testid="cd-edit-modal"
           style={{ background: "var(--surface)", boxShadow: "0 20px 48px rgba(0,0,0,0.2), 0 0 0 1px var(--border)", animation: "cmdk-rise 0.22s cubic-bezier(.2,.9,.3,1.1)" }}
           onClick={e => e.stopPropagation()}>
        <div className="px-[18px] py-3.5 flex items-center" style={{ borderBottom: "1px solid var(--border)" }}>
          <div>
            <div className="text-[14px] font-semibold">
              Edit · {editingTxn.action} · {editingTxn.ticker}
            </div>
            <div className="text-[11px] mt-0.5" style={{ color: "var(--ink-4)" }}>
              Trade {editingTxn.trade_id} · transaction {trxId || `#${detailId}`}
            </div>
          </div>
          <kbd className="ml-auto text-[10px] rounded px-1.5 py-0.5"
               style={{ background: "var(--bg-2)", border: "1px solid var(--border)", color: "var(--ink-4)", fontFamily: mono }}>ESC</kbd>
        </div>
        <div className="p-4 flex flex-col gap-3 max-h-[70vh] overflow-y-auto">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-[10px] uppercase tracking-[0.10em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>Date / Time</label>
              <input type="datetime-local" value={editForm.date}
                     onChange={e => setEditForm(f => ({ ...f, date: e.target.value }))}
                     disabled={editLoading}
                     data-testid="cd-edit-date"
                     className="w-full h-[38px] px-3 rounded-[8px] text-[12px] outline-none"
                     style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }} />
            </div>
            <div>
              <label className="block text-[10px] uppercase tracking-[0.10em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>Trx ID</label>
              <input type="text" value={trxId} readOnly
                     data-testid="cd-edit-trx-id"
                     className="w-full h-[38px] px-3 rounded-[8px] text-[12px] outline-none"
                     style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", opacity: 0.6, cursor: "not-allowed", fontFamily: mono }} />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-[10px] uppercase tracking-[0.10em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>Shares</label>
              <input type="number" step="any" value={editForm.shares}
                     onChange={e => setEditForm(f => ({ ...f, shares: e.target.value }))}
                     disabled={editLoading}
                     data-testid="cd-edit-shares"
                     className="w-full h-[38px] px-3 rounded-[8px] text-[12px] outline-none"
                     style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }} />
            </div>
            <div>
              <label className="block text-[10px] uppercase tracking-[0.10em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>Price ($)</label>
              <input type="number" step="0.01" value={editForm.amount}
                     onChange={e => setEditForm(f => ({ ...f, amount: e.target.value }))}
                     disabled={editLoading}
                     data-testid="cd-edit-amount"
                     className="w-full h-[38px] px-3 rounded-[8px] text-[12px] outline-none"
                     style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }} />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-[10px] uppercase tracking-[0.10em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>Stop Loss ($)</label>
              <input type="number" step="0.01" value={editForm.stop_loss}
                     onChange={e => setEditForm(f => ({ ...f, stop_loss: e.target.value }))}
                     disabled={editLoading}
                     data-testid="cd-edit-stop"
                     className="w-full h-[38px] px-3 rounded-[8px] text-[12px] outline-none"
                     style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }} />
            </div>
            <div>
              <label className="block text-[10px] uppercase tracking-[0.10em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>Rule (Strategy)</label>
              <select value={editForm.rule}
                      onChange={e => setEditForm(f => ({ ...f, rule: e.target.value }))}
                      disabled={editLoading}
                      data-testid="cd-edit-rule"
                      className="w-full h-[38px] px-3 rounded-[8px] text-[12px] outline-none"
                      style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", WebkitAppearance: "none", MozAppearance: "none", appearance: "none" }}>
                <option value="">Select...</option>
                {(isSell ? SELL_RULES : BUY_RULES).map(r => (
                  <option key={r} value={r}>{r}</option>
                ))}
              </select>
            </div>
          </div>
          <div>
            <label className="block text-[10px] uppercase tracking-[0.10em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>Notes</label>
            <textarea value={editForm.notes}
                      onChange={e => setEditForm(f => ({ ...f, notes: e.target.value }))}
                      disabled={editLoading} rows={2}
                      data-testid="cd-edit-notes"
                      className="w-full px-3 py-2 rounded-[8px] text-[12px] outline-none resize-none"
                      style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }} />
          </div>

          {/* Option B: "Lots consumed" subtable for Sell rows. Read-only —
              lot matching is immutable once written by Log Sell. */}
          {isSell && consumedLots.length > 0 && (
            <div data-testid="cd-lots-consumed" className="mt-1">
              <label className="block text-[10px] uppercase tracking-[0.10em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>
                Lots consumed ({consumedLots.length})
              </label>
              <div className="rounded-[8px] overflow-hidden" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
                <table className="w-full text-[11px]" style={{ borderCollapse: "collapse" }}>
                  <thead>
                    <tr style={{ background: "var(--bg-2)", borderBottom: "1px solid var(--border)" }}>
                      <th className="px-2.5 py-1.5 text-left text-[9.5px] uppercase tracking-[0.08em] font-bold" style={{ color: "var(--ink-4)" }}>Lot</th>
                      <th className="px-2.5 py-1.5 text-right text-[9.5px] uppercase tracking-[0.08em] font-bold" style={{ color: "var(--ink-4)" }}>Shares</th>
                      <th className="px-2.5 py-1.5 text-right text-[9.5px] uppercase tracking-[0.08em] font-bold" style={{ color: "var(--ink-4)" }}>Buy Price</th>
                      <th className="px-2.5 py-1.5 text-right text-[9.5px] uppercase tracking-[0.08em] font-bold" style={{ color: "var(--ink-4)" }}>Sell Price</th>
                      <th className="px-2.5 py-1.5 text-right text-[9.5px] uppercase tracking-[0.08em] font-bold" style={{ color: "var(--ink-4)" }}>Realized P&L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {consumedLots.map(c => {
                      const shares = parseFloat(String(c.shares || 0)) || 0;
                      const buyPrice = parseFloat(String((c as { buy_price?: number | string }).buy_price || 0)) || 0;
                      const sellPrice = parseFloat(String((c as { sell_price?: number | string }).sell_price || 0)) || 0;
                      const pl = parseFloat(String(c.realized_pl || 0)) || 0;
                      const buyTrxId = String((c as { buy_trx_id?: string }).buy_trx_id || "—");
                      return (
                        <tr key={buyTrxId + shares + buyPrice} style={{ borderBottom: "1px solid var(--border)" }}>
                          <td className="px-2.5 py-1.5 text-[11px] font-semibold" style={{ fontFamily: mono }}>{buyTrxId}</td>
                          <td className="px-2.5 py-1.5 text-right" style={{ fontFamily: mono }}>{Math.round(shares).toLocaleString()}</td>
                          <td className="px-2.5 py-1.5 text-right" style={{ fontFamily: mono }}>${buyPrice.toFixed(2)}</td>
                          <td className="px-2.5 py-1.5 text-right" style={{ fontFamily: mono }}>${sellPrice.toFixed(2)}</td>
                          <td className="px-2.5 py-1.5 text-right font-bold privacy-mask" style={{ fontFamily: mono, color: pl > 0 ? "#08a86b" : pl < 0 ? "#e5484d" : "var(--ink-3)" }}>
                            {formatCurrency(pl, { showSign: true, decimals: 2 })}
                          </td>
                        </tr>
                      );
                    })}
                    {/* Total row */}
                    <tr style={{ background: "var(--bg-2)" }}>
                      <td className="px-2.5 py-1.5 text-[9.5px] uppercase tracking-[0.06em] font-bold" style={{ color: "var(--ink-3)" }}>Total</td>
                      <td className="px-2.5 py-1.5 text-right font-bold" style={{ fontFamily: mono }}>
                        {Math.round(consumedLots.reduce((s, c) => s + (parseFloat(String(c.shares || 0)) || 0), 0)).toLocaleString()}
                      </td>
                      <td />
                      <td />
                      <td className="px-2.5 py-1.5 text-right font-bold privacy-mask"
                          data-testid="cd-lots-total-realized"
                          style={{ fontFamily: mono, color: (() => { const t = consumedLots.reduce((s, c) => s + (parseFloat(String(c.realized_pl || 0)) || 0), 0); return t > 0 ? "#08a86b" : t < 0 ? "#e5484d" : "var(--ink-3)"; })() }}>
                        {formatCurrency(consumedLots.reduce((s, c) => s + (parseFloat(String(c.realized_pl || 0)) || 0), 0), { showSign: true, decimals: 2 })}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <div className="text-[10px] mt-1.5" style={{ color: "var(--ink-4)" }}>
                Lot matching is set at the time of sale and is read-only here.
              </div>
            </div>
          )}

          {editError && (
            <div data-testid="cd-edit-error"
                 className="px-3 py-2 rounded-[8px] text-[11px] leading-relaxed"
                 style={{
                   background: "color-mix(in oklab, #e5484d 10%, var(--surface))",
                   border: "1px solid color-mix(in oklab, #e5484d 30%, var(--border))",
                   color: "#991b1b",
                 }}>
              {editError}
            </div>
          )}
        </div>
        <div className="px-[18px] py-3 flex items-center gap-2" style={{ borderTop: "1px solid var(--border)" }}>
          <button onClick={onDelete}
                  disabled={editLoading}
                  data-testid="cd-edit-delete"
                  className="h-[32px] px-3 rounded-md text-[12px] font-medium transition-colors hover:brightness-95 disabled:opacity-60 disabled:cursor-wait"
                  style={{
                    background: confirmingDelete ? "#e5484d" : "var(--surface)",
                    color: confirmingDelete ? "white" : "#e5484d",
                    border: confirmingDelete ? "1px solid #e5484d" : "1px solid color-mix(in oklab, #e5484d 35%, var(--border))",
                  }}>
            {confirmingDelete ? "Confirm Delete" : "Delete Transaction"}
          </button>
          <button onClick={onClose}
                  disabled={editLoading}
                  data-testid="cd-edit-cancel"
                  className="ml-auto h-[32px] px-3 rounded-md text-[12px] font-medium hover:brightness-95 disabled:opacity-60 disabled:cursor-wait"
                  style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-2)" }}>
            Cancel
          </button>
          <button onClick={onSave}
                  disabled={editLoading}
                  data-testid="cd-edit-save"
                  className="h-[32px] px-3.5 rounded-md text-[12px] font-medium text-white flex items-center gap-1.5 hover:brightness-95 disabled:opacity-60 disabled:cursor-wait"
                  style={{ background: navColor }}>
            {editLoading && (
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="animate-spin">
                <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2"/>
              </svg>
            )}
            {editLoading ? "Saving…" : "Save Changes"}
          </button>
        </div>
      </div>
      <style jsx global>{`@keyframes cmdk-rise { from { transform: translateY(-10px) scale(0.97); opacity: 0; } }`}</style>
    </div>
  );
}

function SegmentedControl<T extends string>({ label, value, onChange, options, testId }: {
  label: string;
  value: T;
  onChange: (v: T) => void;
  options: { v: T; l: string }[];
  testId: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[9px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>{label}</span>
      <div className="flex p-0.5 rounded-[10px] gap-0.5 h-[34px]" style={{ background: "var(--bg-2)", border: "1px solid var(--border)" }}>
        {options.map(o => (
          <button key={o.v} type="button"
                  onClick={() => onChange(o.v)}
                  data-testid={`${testId}-${o.v}`}
                  className="px-3 rounded-[8px] text-[11px] font-medium transition-all"
                  style={{
                    background: value === o.v ? "var(--surface)" : "transparent",
                    color: value === o.v ? "var(--ink)" : "var(--ink-4)",
                    boxShadow: value === o.v ? "0 1px 2px rgba(14,20,38,0.04)" : "none",
                  }}>
            {o.l}
          </button>
        ))}
      </div>
    </div>
  );
}

function FilterSelect({ label, value, onChange, options, testId }: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { v: string; l: string }[];
  testId: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[9px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>{label}</span>
      <select value={value}
              onChange={e => onChange(e.target.value)}
              data-testid={testId}
              className="h-[34px] px-2.5 rounded-[10px] text-[12px] min-w-[120px]"
              style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", appearance: "none" as never }}>
        {options.map(o => <option key={o.v} value={o.v}>{o.l}</option>)}
      </select>
    </div>
  );
}

function LedgerRowEl({ r, navColor, onEdit, isEditing }: { r: LedgerRow; navColor: string; onEdit: (detailId: number) => void; isEditing: boolean }) {
  const isSell = r.action === "SELL";
  const seriesChar = (r.trx_id || "").charAt(0).toUpperCase();
  // TRX ID pill color by series:
  //   B-prefix (original buys) → ops-green tint
  //   A-prefix (add-ons)       → violet tint (--g-mkt)
  //   S-prefix (sells)         → red tint (--down)
  //   anything else            → muted ink (defensive fallback)
  const seriesTagColor =
    seriesChar === "B" ? "var(--g-ops, #08a86b)" :
    seriesChar === "A" ? "var(--g-mkt, #8b5cf6)" :
    seriesChar === "S" ? "#e5484d" :
    "var(--ink-4)";
  const statusBg =
    r.status === "Open" ? "color-mix(in oklab, #08a86b 14%, var(--surface))" :
    r.status === "Partial" ? "color-mix(in oklab, #f59f00 18%, var(--surface))" :
    "var(--bg-2)";
  const statusColor =
    r.status === "Open" ? "#08a86b" :
    r.status === "Partial" ? "#a87108" :
    "var(--ink-4)";
  const retStyle = r.ret != null ? retChipStyle(r.ret) : null;
  return (
    <tr className="hover:bg-[var(--surface-2)]" data-testid={`row-${r.detail_id}`}
        style={{
          borderBottom: "1px solid var(--border)",
          background: isEditing ? "color-mix(in oklab, " + navColor + " 6%, transparent)" : undefined,
        }}>
      <td className="px-3 py-2.5 text-[11.5px]" data-testid={`row-${r.detail_id}-trx`}>
        {r.trx_id ? (
          <span className="inline-block px-1.5 py-0.5 rounded-md text-[10px] font-bold"
                style={{ background: `color-mix(in oklab, ${seriesTagColor} 14%, var(--surface))`, color: seriesTagColor, fontFamily: mono }}>
            {r.trx_id}
          </span>
        ) : <span style={{ color: "var(--ink-4)" }}>—</span>}
      </td>
      <td className="px-3 py-2.5 text-[11.5px]" style={{ fontFamily: mono, color: "var(--ink-3)" }}>
        {r.date.length >= 16 ? r.date.slice(0, 16).replace("T", " ") : r.date.slice(0, 10)}
      </td>
      <td className="px-3 py-2.5 text-[11.5px] font-semibold" style={{ fontFamily: mono }}>{r.ticker}</td>
      <td className="px-3 py-2.5 text-[11.5px]" data-testid={`row-${r.detail_id}-action`}>
        <span className="inline-flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: isSell ? "#e5484d" : "#08a86b" }} />
          <span style={{ color: isSell ? "#e5484d" : "#08a86b" }}>{isSell ? "Sell" : "Buy"}</span>
        </span>
      </td>
      <td className="px-3 py-2.5 text-[11.5px]">
        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10.5px] font-semibold"
              style={{ background: statusBg, color: statusColor }}>
          <span className="w-1 h-1 rounded-full" style={{ background: statusColor }} />
          {r.status}
        </span>
      </td>
      <td className="px-3 py-2.5 text-right" style={{ fontFamily: mono }}>{r.shares > 0 ? Math.round(r.shares).toLocaleString() : "—"}</td>
      <td className="px-3 py-2.5 text-right" style={{ fontFamily: mono, color: r.remaining == null ? "var(--ink-4)" : "var(--ink)" }}>
        {r.remaining == null ? "—" : Math.round(r.remaining).toLocaleString()}
      </td>
      <td className="px-3 py-2.5 text-right privacy-mask" style={{ fontFamily: mono }}>
        {r.cost_basis > 0 ? `$${r.cost_basis.toFixed(2)}` : "—"}
      </td>
      <td className="px-3 py-2.5 text-right privacy-mask" style={{ fontFamily: mono, color: r.exit_price == null ? "var(--ink-4)" : "var(--ink)" }}>
        {r.exit_price == null ? "—" : `$${r.exit_price.toFixed(2)}`}
      </td>
      <td className="px-3 py-2.5 text-right privacy-mask" style={{ fontFamily: mono }}>
        {r.stop_loss > 0 ? `$${r.stop_loss.toFixed(2)}` : "—"}
      </td>
      <td className="px-3 py-2.5 text-right privacy-mask" style={{ fontFamily: mono }}>
        {r.value !== 0 ? formatCurrency(r.value, { decimals: 0 }) : "—"}
      </td>
      <td className="px-3 py-2.5 text-right privacy-mask" style={{ fontFamily: mono, color: r.realized == null ? "var(--ink-4)" : pctColor(r.realized) }}>
        {r.realized == null ? "—" : formatCurrency(r.realized, { showSign: true, decimals: 2 })}
      </td>
      <td className="px-3 py-2.5 text-right privacy-mask" style={{ fontFamily: mono, color: r.unrealized == null ? "var(--ink-4)" : pctColor(r.unrealized) }}>
        {r.unrealized == null ? "—" : formatCurrency(r.unrealized, { showSign: true, decimals: 2 })}
      </td>
      <td className="px-3 py-2.5 text-right" style={{ fontFamily: mono }}>
        {r.ret == null || retStyle == null ? "—" : (
          <span className="inline-block px-2 py-0.5 rounded-md font-semibold text-[10.5px]"
                style={{ background: retStyle.bg, color: retStyle.color }}>
            {r.ret >= 0 ? "+" : ""}{r.ret.toFixed(1)}%
          </span>
        )}
      </td>
      <td className="px-3 py-2.5 text-[11px]" style={{ color: isSell ? "#e5484d" : "var(--ink-2)", fontFamily: mono }}>
        {r.rule || "—"}
      </td>
      <td className="px-3 py-2.5 text-[11px]" style={{ color: r.notes ? "var(--ink-2)" : "var(--ink-4)", maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis" }}
          title={r.notes || ""}>
        {r.notes || "—"}
      </td>
      <td className="px-3 py-2.5 text-right">
        <button type="button"
                onClick={() => onEdit(r.detail_id)}
                title="Edit this fill"
                data-testid={`edit-${r.detail_id}`}
                className="w-[26px] h-[26px] rounded-[8px] inline-flex items-center justify-center cursor-pointer transition-all hover:brightness-95"
                style={{
                  background: isEditing ? `color-mix(in oklab, ${navColor} 14%, var(--surface))` : "var(--surface-2)",
                  border: `1px solid ${isEditing ? navColor : "var(--border)"}`,
                  color: isEditing ? navColor : "var(--ink-3)",
                }}>
          ✎
        </button>
      </td>
    </tr>
  );
}
