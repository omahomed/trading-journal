"use client";

// Per-trade Entry vs Add performance.
//
// Each row = one campaign with two performance buckets shown side by
// side: the B series (original entry lots) vs the A series (add-on
// lots). Answers "where did my profit come from, the initial entry
// or the scale-ins?" — a cut none of the existing analytics surfaces
// expose this way:
//
//   - Trade Review        : per-trade total P&L, not split by series
//   - Add Effectiveness   : rule-level A-only summary, not per-trade
//   - Campaign Detail     : per-fill ledger (this page is per-trade)
//
// Math reuses the existing data layer entirely:
//   - lot_closures.realized_pl carries per-closure dollars + the
//     buy_trx_id it consumed → realized B/A split is just a groupby
//   - walkLedger (lib/campaign-detail-walk) walks all details once
//     and returns per-detail remaining shares → unrealized B/A is
//     the remaining lots × (mark − lot price) × multiplier
//
// Open positions are included by default with live unrealized,
// because "is my open NBIS profit coming from B or A?" is the most
// actionable question this page answers. Mark price comes from
// /api/prices/batch (same provider Dashboard NLV and Campaign Detail
// use), so the figures here agree with the rest of the app.

import { useState, useEffect, useMemo, useCallback, useRef } from "react";
import {
  api,
  getActivePortfolio,
  type TradePosition,
  type TradeDetail,
  type LotClosure,
} from "@/lib/api";
import { walkLedger } from "@/lib/campaign-detail-walk";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";
import { KPITile, TILE_GRADIENTS, SegmentedControl } from "./campaign-detail";

const mono = "var(--font-jetbrains), monospace";

// Campaign-level status — two states only. "Partial" was meaningful
// on the per-fill Campaign Detail page (some lots open, some closed
// within one trade), but at the campaign-summary level it just
// means "still has open exposure", which is what "Open" already
// covers. Collapsing keeps the filter intuitive.
type SeriesStatus = "Open" | "Closed";
const STATUS_KEYS: readonly SeriesStatus[] = ["Open", "Closed"] as const;

interface TradeRow {
  trade_id: string;
  ticker: string;
  status: SeriesStatus;
  open_date: string;
  closed_date: string | null;
  // Series breakdown. Shares × initial cost basis lets the user see
  // weight; realized + unrealized roll up into pnl; return % is
  // P&L ÷ cost (the natural ROI on capital deployed in that series).
  b_shares: number;
  a_shares: number;
  b_initial_cost: number;
  a_initial_cost: number;
  b_realized: number;
  a_realized: number;
  b_unrealized: number;
  a_unrealized: number;
  b_pnl: number;
  a_pnl: number;
  b_return_pct: number | null;
  a_return_pct: number | null;
  total_pnl: number;
  // Blended ROI on all capital deployed across both series. Null when
  // the trade has no buys (data corruption). Lives alongside the per-
  // series Return % so the user can read combined performance at a
  // glance without summing two columns mentally.
  total_return_pct: number | null;
  rule: string;
  multiplier: number;
  // Migration 046 — excursion metrics passed through from the summary
  // row. Nullable + optional; the daily reconciler stamps them on
  // OPEN equity positions, closed rows keep the last-observed values.
  // ×ATR multiples are derived at render time from atr21_entry_pct.
  mae_pct: number | null;
  mfe_pct: number | null;
  atr21_entry_pct: number | null;
}

type ColKey =
  | "trade_id" | "ticker" | "status" | "open_date" | "closed_date"
  | "b_shares" | "b_pnl" | "b_return_pct"
  | "a_shares" | "a_pnl" | "a_return_pct"
  | "total_pnl" | "total_return_pct" | "rule"
  | "mae_pct" | "mfe_pct";

interface Filters {
  q: string;
  status: SeriesStatus[];  // empty = no filter
  tickers: string[];       // empty = no filter; multi-chip
  rule: string;            // rule or "all"
  pl: "all" | "realized" | "unrealized";
  // Numeric Return-% thresholds. Empty string = no filter. Set to "0"
  // to slice "positive only", or any number for ">= X%". Trades with
  // a null series Return % (e.g. no A lots) fall out of the A-min
  // filter when it's active — the row literally has no A series to
  // compare against.
  b_min_pct: string;
  a_min_pct: string;
  from: string;            // YYYY-MM-DD
  to: string;              // YYYY-MM-DD
}
const EMPTY_FILTERS: Filters = {
  q: "", status: [], tickers: [], rule: "all", pl: "all",
  b_min_pct: "", a_min_pct: "", from: "", to: "",
};

const NUMERIC_KEYS = new Set<ColKey>([
  "b_shares", "b_pnl", "b_return_pct",
  "a_shares", "a_pnl", "a_return_pct",
  "total_pnl", "total_return_pct",
  "mae_pct", "mfe_pct",
]);

// Prefer explicit instrument_type metadata (Migration 016) with a
// ticker-shape fallback for legacy rows. Shared helpers — getMultiplier
// scales cost basis / unrealized into dollar terms for options;
// isOption gates the page-level exclusion. Both predicates match
// positions.ts + log-sell.tsx so the three call sites stay in lockstep.
function isOption(trade: TradePosition): boolean {
  const type = String((trade as { instrument_type?: string }).instrument_type || "").toUpperCase();
  if (type === "OPTION") return true;
  if (type === "STOCK") return false;
  return /^\S+\s+\d{6}\s+\$[0-9.]+(C|P)$/.test(String(trade.ticker || ""));
}
function getMultiplier(trade: TradePosition): number {
  const raw = parseFloat(String((trade as { multiplier?: number | string }).multiplier || 0));
  if (raw > 0) return raw;
  return isOption(trade) ? 100 : 1;
}

function seriesPrefix(trxId: string): "B" | "A" | "" {
  const c = String(trxId || "").charAt(0).toUpperCase();
  if (c === "B") return "B";
  if (c === "A") return "A";
  return "";
}

// Compute the per-trade row from the raw trades + details + closures +
// live prices. All math is dollar-denominated (multiplier folded into
// cost basis and unrealized so the same number works for stocks and
// options).
function computeTradeRows(
  trades: TradePosition[],
  details: TradeDetail[],
  closures: LotClosure[],
  livePrices: Record<string, number>,
): TradeRow[] {
  const detailsByTrade = new Map<string, TradeDetail[]>();
  for (const d of details) {
    const tid = String(d.trade_id || "");
    if (!tid) continue;
    if (!detailsByTrade.has(tid)) detailsByTrade.set(tid, []);
    detailsByTrade.get(tid)!.push(d);
  }
  const closuresByTrade = new Map<string, LotClosure[]>();
  for (const c of closures) {
    const tid = String(c.trade_id || "");
    if (!tid) continue;
    if (!closuresByTrade.has(tid)) closuresByTrade.set(tid, []);
    closuresByTrade.get(tid)!.push(c);
  }
  // walkLedger once over ALL details — its `perDetail` map is keyed
  // by detail.id, stable across all campaigns. Per-trade loops below
  // just look up by id.
  const walk = walkLedger(details);

  return trades.map(trade => {
    const tradeDetails = detailsByTrade.get(trade.trade_id) || [];
    const tradeClosures = closuresByTrade.get(trade.trade_id) || [];
    const multiplier = getMultiplier(trade);

    const buys = tradeDetails.filter(d => String(d.action).toUpperCase() === "BUY");

    // Realized — attribute each closure's realized_pl to the series
    // its buy_trx_id belongs to. The closure row already carries the
    // multiplier-correct dollar amount, so we don't apply it again.
    let b_realized = 0;
    let a_realized = 0;
    for (const c of tradeClosures) {
      const prefix = seriesPrefix(c.buy_trx_id || "");
      if (prefix === "B") b_realized += c.realized_pl || 0;
      else if (prefix === "A") a_realized += c.realized_pl || 0;
    }

    // Initial cost basis + shares per series.
    let b_shares = 0;
    let a_shares = 0;
    let b_initial_cost = 0;
    let a_initial_cost = 0;
    for (const d of buys) {
      const shares = Math.abs(parseFloat(String(d.shares || 0)));
      const price = parseFloat(String(d.amount || 0));
      const prefix = seriesPrefix(d.trx_id || "");
      const cost = shares * price * multiplier;
      if (prefix === "B") {
        b_shares += shares;
        b_initial_cost += cost;
      } else if (prefix === "A") {
        a_shares += shares;
        a_initial_cost += cost;
      }
    }

    // Unrealized — for each remaining buy lot, (mark − lot price) ×
    // remaining × multiplier. Falls back to avg_entry if no live
    // price has landed (matches positions.ts behavior).
    const mark = livePrices[trade.ticker] || parseFloat(String(trade.avg_entry || 0)) || 0;
    let b_unrealized = 0;
    let a_unrealized = 0;
    let remaining_total = 0;
    let total_shares = 0;
    for (const d of buys) {
      const id = (d as { detail_id?: number; id?: number }).detail_id ?? (d as { id?: number }).id ?? -1;
      const remaining = walk.perDetail.get(id)?.remaining ?? 0;
      const shares = Math.abs(parseFloat(String(d.shares || 0)));
      total_shares += shares;
      remaining_total += remaining;
      if (remaining <= 0) continue;
      const lotPrice = parseFloat(String(d.amount || 0));
      const lotUnreal = (mark - lotPrice) * remaining * multiplier;
      const prefix = seriesPrefix(d.trx_id || "");
      if (prefix === "B") b_unrealized += lotUnreal;
      else if (prefix === "A") a_unrealized += lotUnreal;
    }

    const b_pnl = b_realized + b_unrealized;
    const a_pnl = a_realized + a_unrealized;
    const b_return_pct = b_initial_cost > 0 ? (b_pnl / b_initial_cost) * 100 : null;
    const a_return_pct = a_initial_cost > 0 ? (a_pnl / a_initial_cost) * 100 : null;
    const total_cost = b_initial_cost + a_initial_cost;
    const total_pnl = b_pnl + a_pnl;
    const total_return_pct = total_cost > 0 ? (total_pnl / total_cost) * 100 : null;

    // Campaign-level status: any remaining shares = Open, zero =
    // Closed. The intermediate "Partial" tier (some sells fired but
    // not all shares are out) collapses into Open here — the user
    // still has skin in the game, which is what they care about at
    // the summary level.
    const status: SeriesStatus = remaining_total > 0.00001 ? "Open" : "Closed";

    return {
      trade_id: trade.trade_id,
      ticker: String(trade.ticker || ""),
      status,
      open_date: String(trade.open_date || "").slice(0, 10),
      closed_date: trade.closed_date ? String(trade.closed_date).slice(0, 10) : null,
      b_shares, a_shares,
      b_initial_cost, a_initial_cost,
      b_realized, a_realized,
      b_unrealized, a_unrealized,
      b_pnl, a_pnl,
      b_return_pct, a_return_pct,
      total_pnl,
      total_return_pct,
      rule: String((trade as { rule?: string; buy_rule?: string }).rule || (trade as { buy_rule?: string }).buy_rule || ""),
      multiplier,
      mae_pct:  toNumOrNull((trade as { mae_pct?: number | null }).mae_pct),
      mfe_pct:  toNumOrNull((trade as { mfe_pct?: number | null }).mfe_pct),
      atr21_entry_pct: toNumOrNull((trade as { atr21_entry_pct?: number | null }).atr21_entry_pct),
    };
  });
}

// Backend loaders can return NUMERIC columns as strings when they're
// deserialized through psycopg2's default cursor. Coerce here so the
// downstream number-only comparisons stay total.
function toNumOrNull(v: number | string | null | undefined): number | null {
  if (v == null || v === "") return null;
  const n = typeof v === "number" ? v : parseFloat(String(v));
  return Number.isFinite(n) ? n : null;
}

// Multi-select Status pill matches the Campaign Detail StatusMultiSelect.
function StatusMultiSelect({ value, onChange }: {
  value: SeriesStatus[];
  onChange: (next: SeriesStatus[]) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("mousedown", onClick);
    return () => window.removeEventListener("mousedown", onClick);
  }, [open]);

  const toggle = (k: SeriesStatus) => {
    onChange(value.includes(k) ? value.filter(v => v !== k) : [...value, k]);
  };

  const summary = value.length === 0
    ? "All status"
    : value.length <= 2
      ? value.join(", ")
      : `${value.length} selected`;

  return (
    <div className="flex flex-col gap-1" ref={ref}>
      <span className="text-[9px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>Status</span>
      <div className="relative">
        <button type="button" onClick={() => setOpen(o => !o)}
                className="h-[34px] px-2.5 rounded-[10px] text-[12px] min-w-[120px] flex items-center justify-between gap-2"
                style={{ background: "var(--surface)", border: "1px solid var(--border)", color: value.length > 0 ? "var(--ink)" : "var(--ink-3)" }}>
          <span className="truncate">{summary}</span>
          <span style={{ opacity: 0.6 }}>▾</span>
        </button>
        {open && (
          <div className="absolute top-full mt-1 left-0 z-40 rounded-[10px] py-1.5 overflow-hidden"
               style={{
                 minWidth: 160, background: "var(--surface)", border: "1px solid var(--border)",
                 boxShadow: "0 8px 24px rgba(0,0,0,0.16), 0 2px 6px rgba(0,0,0,0.08)",
               }}>
            {STATUS_KEYS.map(k => {
              const checked = value.includes(k);
              return (
                <button key={k} type="button" onClick={() => toggle(k)}
                        className="w-full text-left px-3 py-2 text-[12px] flex items-center gap-2 transition-colors hover:brightness-95"
                        style={{ background: checked ? "var(--surface-2)" : "transparent", color: "var(--ink)" }}>
                  <span className="inline-flex items-center justify-center w-4 h-4 rounded-[4px]"
                        style={{
                          background: checked ? "var(--ink)" : "transparent",
                          border: `1px solid ${checked ? "var(--ink)" : "var(--border)"}`,
                          color: "var(--surface)", fontSize: 10, fontWeight: 700, lineHeight: 1,
                        }}>
                    {checked ? "✓" : ""}
                  </span>
                  {k}
                </button>
              );
            })}
            {value.length > 0 && (
              <button type="button" onClick={() => onChange([])}
                      className="w-full text-left px-3 py-1.5 text-[11px] font-medium"
                      style={{ borderTop: "1px solid var(--border)", color: "var(--ink-3)" }}>
                Clear
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// Multi-ticker chip + autocomplete. Mirrors the trade-journal.tsx
// pattern: chips for picked tickers, an input that filters the
// dropdown to matching tickers from the available set, Enter/Backspace
// keyboard handling. Used in place of the single-select Ticker
// dropdown so the user can pull "MU + AAPL + NBIS" in one go.
function TickerMultiSelect({ value, onChange, tickers, navColor }: {
  value: string[];
  onChange: (next: string[]) => void;
  tickers: string[];
  navColor: string;
}) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const available = useMemo(
    () => tickers.filter(t => !value.includes(t))
                 .filter(t => !query || t.toUpperCase().includes(query.trim().toUpperCase())),
    [tickers, value, query],
  );
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[9px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>Tickers</span>
      <div className="flex items-center gap-1.5 flex-wrap">
        {value.map(t => (
          <span key={t} className="flex items-center gap-1 h-[28px] px-2.5 rounded-[8px] text-[11px] font-semibold"
                style={{ background: `color-mix(in oklab, ${navColor} 10%, transparent)`, color: navColor, border: `1px solid ${navColor}30` }}>
            {t}
            <button onClick={() => onChange(value.filter(x => x !== t))}
                    className="ml-0.5 opacity-60 hover:opacity-100" style={{ lineHeight: 1 }}>×</button>
          </span>
        ))}
        <div className="relative">
          <input type="text" value={query}
                 placeholder={value.length > 0 ? "Add ticker…" : "Search tickers…"}
                 onChange={e => { setQuery(e.target.value.toUpperCase()); setOpen(true); }}
                 onKeyDown={e => {
                   if (e.key === "Enter" && query) {
                     const match = tickers.find(t => t.toUpperCase() === query.trim().toUpperCase());
                     const next = match ?? query.trim().toUpperCase();
                     if (next && !value.includes(next)) onChange([...value, next]);
                     setQuery("");
                     setOpen(false);
                   } else if (e.key === "Backspace" && !query && value.length > 0) {
                     onChange(value.slice(0, -1));
                   }
                 }}
                 onFocus={() => setOpen(true)}
                 onBlur={() => setTimeout(() => setOpen(false), 150)}
                 className="h-[34px] px-3 rounded-[10px] text-[12px] w-[140px]"
                 style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }} />
          {open && available.length > 0 && (
            <div className="absolute z-50 mt-1 w-[180px] rounded-[10px] overflow-hidden shadow-lg"
                 style={{ background: "var(--surface)", border: "1px solid var(--border)", maxHeight: 200 }}>
              <div className="overflow-y-auto" style={{ maxHeight: 200 }}>
                {available.map(t => (
                  <button key={t} type="button"
                          onMouseDown={e => { e.preventDefault(); onChange([...value, t]); setQuery(""); setOpen(false); }}
                          className="w-full text-left px-3 py-1.5 text-[12px] transition-colors"
                          style={{ fontFamily: mono }}
                          onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-2)")}
                          onMouseLeave={e => (e.currentTarget.style.background = "transparent")}>
                    {t}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Single-select dropdown for Rule. Mirrors Campaign Detail's
// FilterSelect shape so the two pages feel like siblings.
function FilterSelect({ label, value, onChange, options }: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { v: string; l: string }[];
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[9px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>{label}</span>
      <select value={value} onChange={e => onChange(e.target.value)}
              className="h-[34px] px-2.5 rounded-[10px] text-[12px] min-w-[120px]"
              style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", appearance: "none" as never }}>
        {options.map(o => <option key={o.v} value={o.v}>{o.l}</option>)}
      </select>
    </div>
  );
}

export function EntryVsAdd({ navColor }: { navColor: string }) {
  const [openTrades, setOpenTrades] = useState<TradePosition[]>([]);
  const [closedTrades, setClosedTrades] = useState<TradePosition[]>([]);
  const [details, setDetails] = useState<TradeDetail[]>([]);
  const [closures, setClosures] = useState<LotClosure[]>([]);
  const [livePrices, setLivePrices] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdatedAt, setLastUpdatedAt] = useState<Date | null>(null);
  const [filters, setFilters] = useState<Filters>(EMPTY_FILTERS);
  const [sort, setSort] = useState<{ key: ColKey; dir: "asc" | "desc" }>({ key: "open_date", dir: "desc" });

  const loadAll = useCallback(async (forRefresh: boolean) => {
    const portfolio = getActivePortfolio();
    try {
      if (forRefresh) setRefreshing(true);
      else setLoading(true);
      setError(null);
      // 10000 transaction cap covers any realistic history; if a
      // user ever exceeds it, add a dedicated /api/trades/all
      // endpoint instead of bumping further. tradesRecent already
      // returns the bundled lot_closures so this is a single fetch
      // for the per-series math.
      const [opens, closeds, bundle] = await Promise.all([
        api.tradesOpen(portfolio).catch(() => [] as TradePosition[]),
        api.tradesClosed(portfolio, 1000).catch(() => [] as TradePosition[]),
        api.tradesRecent(portfolio, 10000).catch(() => ({ details: [], lot_closures: [] })),
      ]);
      setOpenTrades(opens);
      setClosedTrades(closeds);
      setDetails(bundle.details || []);
      setClosures(bundle.lot_closures || []);
      // Live prices only matter for open positions' unrealized.
      const tickers = [...new Set(opens.map(t => t.ticker).filter(Boolean))];
      if (tickers.length > 0) {
        try {
          const prices = await api.batchPrices(tickers, portfolio);
          setLivePrices(prices);
        } catch (e) {
          log.warn("entry-vs-add", "batchPrices failed", e);
        }
      }
      setLastUpdatedAt(new Date());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (forRefresh) setRefreshing(false);
      else setLoading(false);
    }
  }, []);

  useEffect(() => { loadAll(false); }, [loadAll]);

  // Options have no meaningful B-vs-A story in this workflow — they
  // rarely receive scale-ins, and when they do the multiplier mixes
  // weirdly with the per-series ROI math. Hard-exclude them so the
  // page stays focused on stock-campaign attribution.
  const allTrades = useMemo(
    () => [...openTrades, ...closedTrades].filter(t => !isOption(t)),
    [openTrades, closedTrades],
  );
  const rows = useMemo(
    () => computeTradeRows(allTrades, details, closures, livePrices),
    [allTrades, details, closures, livePrices],
  );

  const tickerOptions = useMemo(
    () => [...new Set(rows.map(r => r.ticker).filter(Boolean))].sort(),
    [rows],
  );
  const ruleOptions = useMemo(
    () => [...new Set(rows.map(r => r.rule).filter(Boolean))].sort(),
    [rows],
  );

  const filtered = useMemo(() => {
    return rows.filter(r => {
      const q = filters.q.trim().toLowerCase();
      if (q) {
        const haystack = [r.ticker, r.trade_id, r.rule].map(v => v.toLowerCase());
        if (!haystack.some(s => s.includes(q))) return false;
      }
      if (filters.status.length > 0 && !filters.status.includes(r.status)) return false;
      if (filters.tickers.length > 0 && !filters.tickers.includes(r.ticker)) return false;
      if (filters.rule !== "all" && r.rule !== filters.rule) return false;
      // P&L scope. realized = has any closed shares; unrealized = has
      // any remaining lot. A purely open trade with no closures shows
      // up under unrealized only; a fully closed campaign under
      // realized only.
      if (filters.pl === "realized" && (r.b_realized === 0 && r.a_realized === 0)) return false;
      if (filters.pl === "unrealized" && (r.b_unrealized === 0 && r.a_unrealized === 0)) return false;
      // Per-series Return % thresholds. Strings parsed lazily so the
      // user can clear them by emptying the input. A row with a null
      // Return % (no lots in that series) fails the filter when it's
      // active — there's no series to evaluate against.
      const bMin = parseFloat(filters.b_min_pct);
      if (!isNaN(bMin)) {
        if (r.b_return_pct == null || r.b_return_pct < bMin) return false;
      }
      const aMin = parseFloat(filters.a_min_pct);
      if (!isNaN(aMin)) {
        if (r.a_return_pct == null || r.a_return_pct < aMin) return false;
      }
      const d = r.open_date.slice(0, 10);
      if (filters.from && d < filters.from) return false;
      if (filters.to && d > filters.to) return false;
      return true;
    });
  }, [rows, filters]);

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

  // KPI strip — dynamic on filters. Status split is "Open / Partial /
  // Closed" counts of the visible (filtered) rows so the user can
  // see what they're looking at at a glance.
  const kpis = useMemo(() => {
    const openCount = sorted.filter(r => r.status === "Open").length;
    const closedCount = sorted.filter(r => r.status === "Closed").length;
    const bRealized = sorted.reduce((t, r) => t + r.b_realized, 0);
    const aRealized = sorted.reduce((t, r) => t + r.a_realized, 0);
    const unrealized = sorted.reduce((t, r) => t + r.b_unrealized + r.a_unrealized, 0);
    return {
      trades: sorted.length,
      openCount, closedCount,
      bRealized, aRealized, unrealized,
    };
  }, [sorted]);

  const onSort = (key: ColKey) => {
    setSort(s => s.key === key
      ? { key, dir: s.dir === "asc" ? "desc" : "asc" }
      : { key, dir: NUMERIC_KEYS.has(key) ? "desc" : "asc" });
  };

  const filtersDirty = useMemo(() => (
    !!filters.q || filters.status.length > 0 || filters.tickers.length > 0
    || filters.rule !== "all" || filters.pl !== "all"
    || !!filters.b_min_pct || !!filters.a_min_pct
    || !!filters.from || !!filters.to
  ), [filters]);
  const resetFilters = () => setFilters(EMPTY_FILTERS);

  const onExportCsv = useCallback(() => {
    const header = [
      "Trade ID", "Ticker", "Status", "Open", "Close",
      "B Shares", "B Cost", "B Realized", "B Unrealized", "B P&L", "B Return %",
      "A Shares", "A Cost", "A Realized", "A Unrealized", "A P&L", "A Return %",
      "Total P&L", "Total Return %", "Rule",
      "MAE %", "MFE %", "MAE ATR", "MFE ATR", "ATR21 Entry %",
    ].join(",");
    const escape = (v: unknown) => {
      const s = v == null ? "" : String(v);
      return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const lines = sorted.map(r => {
      // Derive ATR multiples at export time so the CSV and the on-screen
      // secondary line stay in lockstep (single source: atr21_entry_pct).
      const maeAtr = r.mae_pct != null && r.atr21_entry_pct != null && r.atr21_entry_pct > 0
        ? Math.abs(r.mae_pct) / r.atr21_entry_pct : null;
      const mfeAtr = r.mfe_pct != null && r.atr21_entry_pct != null && r.atr21_entry_pct > 0
        ? r.mfe_pct / r.atr21_entry_pct : null;
      return [
        r.trade_id, r.ticker, r.status, r.open_date, r.closed_date ?? "",
        r.b_shares, r.b_initial_cost.toFixed(2),
        r.b_realized.toFixed(2), r.b_unrealized.toFixed(2),
        r.b_pnl.toFixed(2), r.b_return_pct?.toFixed(2) ?? "",
        r.a_shares, r.a_initial_cost.toFixed(2),
        r.a_realized.toFixed(2), r.a_unrealized.toFixed(2),
        r.a_pnl.toFixed(2), r.a_return_pct?.toFixed(2) ?? "",
        r.total_pnl.toFixed(2), r.total_return_pct?.toFixed(2) ?? "",
        r.rule,
        r.mae_pct?.toFixed(2) ?? "",
        r.mfe_pct?.toFixed(2) ?? "",
        maeAtr?.toFixed(2) ?? "",
        mfeAtr?.toFixed(2) ?? "",
        r.atr21_entry_pct?.toFixed(2) ?? "",
      ].map(escape).join(",");
    });
    const csv = [header, ...lines].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `entry-vs-add-${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [sorted]);

  const lastUpdatedLabel = lastUpdatedAt
    ? `${lastUpdatedAt.toISOString().slice(0, 10)} ${String(lastUpdatedAt.getHours()).padStart(2, "0")}:${String(lastUpdatedAt.getMinutes()).padStart(2, "0")}`
    : "";

  const statusSplitLabel = `${kpis.openCount} O · ${kpis.closedCount} C`;

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }} data-testid="entry-vs-add-root">
      {/* Page header */}
      <div className="mb-[22px] pb-[14px] flex items-end justify-between gap-4"
           style={{ borderBottom: "1px solid var(--border)" }}>
        <div>
          <h1 className="font-normal text-[32px] tracking-tight m-0"
              style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
            Entry vs <em className="italic" style={{ color: navColor }}>Add</em>
          </h1>
          <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
            Per-stock-trade performance split by series — original entry (B) vs add-ons (A) · options excluded
            {lastUpdatedLabel ? ` · as of ${lastUpdatedLabel}` : ""}
          </div>
        </div>
        <div className="flex gap-2 shrink-0">
          <button type="button" onClick={onExportCsv}
                  disabled={sorted.length === 0}
                  className="px-3 py-2 rounded-[10px] text-[13px] flex items-center gap-1.5 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-2)" }}>
            ↓ Export CSV
          </button>
          <button type="button" onClick={() => loadAll(true)} disabled={refreshing}
                  className="px-3 py-2 rounded-[10px] text-[13px] flex items-center gap-1.5 transition-colors"
                  style={{ background: "var(--surface)", border: "1px solid var(--border)", color: refreshing ? "var(--ink-4)" : "var(--ink-2)" }}>
            ⟳ {refreshing ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 px-4 py-3 rounded-[10px]"
             style={{ background: "color-mix(in oklab, #e5484d 8%, var(--surface))", border: "1px solid var(--border)", color: "#e5484d" }}>
          Failed to load: {error}
        </div>
      )}

      {/* KPI strip */}
      {loading && !lastUpdatedAt ? (
        <div className="grid grid-cols-5 gap-[14px]">
          {[0, 1, 2, 3, 4].map(i => (
            <div key={i} className="rounded-[14px] animate-pulse min-h-[108px]" style={{ background: "var(--bg-2)" }} />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-5 gap-[14px]">
          <KPITile label="Trades" value={String(kpis.trades)} sub={`${rows.length} total`} gradient={TILE_GRADIENTS.indigo} />
          <KPITile label="Status Mix" value={statusSplitLabel} sub="Open · Closed" gradient={TILE_GRADIENTS.blue} />
          <KPITile label="B Realized" value={formatCurrency(kpis.bRealized, { decimals: 0 })}
                   sub="entry-lot closures" gradient={kpis.bRealized >= 0 ? TILE_GRADIENTS.green : TILE_GRADIENTS.red} />
          <KPITile label="A Realized" value={formatCurrency(kpis.aRealized, { decimals: 0 })}
                   sub="add-on closures" gradient={kpis.aRealized >= 0 ? TILE_GRADIENTS.pink : TILE_GRADIENTS.red} />
          <KPITile label="Unrealized" value={formatCurrency(kpis.unrealized, { decimals: 0 })}
                   sub="B + A open at mark" gradient={kpis.unrealized >= 0 ? TILE_GRADIENTS.orange : TILE_GRADIENTS.red} />
        </div>
      )}

      {/* Card: Trade Ledger */}
      <div className="mt-5 rounded-[14px] overflow-hidden"
           style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "0 1px 2px rgba(14,20,38,0.04)" }}>
        <div className="px-[18px] py-[14px] flex items-center gap-2"
             style={{ borderBottom: "1px solid var(--border)" }}>
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
          <span className="text-[13px] font-semibold">Trade Ledger</span>
          <span className="text-[12px]" style={{ color: "var(--ink-4)" }}>
            {sorted.length} trades · {new Set(sorted.map(r => r.ticker)).size} tickers
          </span>
        </div>

        {/* Filter toolbar */}
        <div className="px-[18px] py-[14px] flex flex-wrap items-end gap-[12px_14px]"
             style={{ background: "var(--bg-2)", borderBottom: "1px solid var(--border)" }}>
          <div className="flex flex-col gap-1" style={{ flex: "1 1 220px", minWidth: 200 }}>
            <span className="text-[9px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>Search</span>
            <div className="relative">
              <input type="text" value={filters.q}
                     onChange={e => setFilters(f => ({ ...f, q: e.target.value }))}
                     placeholder="Ticker, trade ID, or rule…"
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

          <SegmentedControl label="P&L"
            value={filters.pl}
            onChange={v => setFilters(f => ({ ...f, pl: v as Filters["pl"] }))}
            options={[{ v: "all", l: "All" }, { v: "realized", l: "Realized" }, { v: "unrealized", l: "Unrealized" }]}
            testId="filter-pl"
          />

          <StatusMultiSelect
            value={filters.status}
            onChange={next => setFilters(f => ({ ...f, status: next }))}
          />

          <TickerMultiSelect
            value={filters.tickers}
            onChange={next => setFilters(f => ({ ...f, tickers: next }))}
            tickers={tickerOptions}
            navColor={navColor}
          />

          <FilterSelect label="Rule"
            value={filters.rule}
            onChange={v => setFilters(f => ({ ...f, rule: v }))}
            options={[{ v: "all", l: "All rules" }, ...ruleOptions.map(r => ({ v: r, l: r }))]}
          />

          {/* Per-series Return % min thresholds. Empty input = no
              filter. Type "0" to slice positives only; type "50"
              for ">= 50%". Negative numbers work too (e.g. "-20"
              to surface trades with B% <= -20%? Actually no — this
              is a MIN, so "-20" means ">= -20%"). For a max bound,
              add later if asked. */}
          <div className="flex flex-col gap-1">
            <span className="text-[9px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>B % Min</span>
            <input type="number" inputMode="decimal" step="1" value={filters.b_min_pct}
                   onChange={e => setFilters(f => ({ ...f, b_min_pct: e.target.value }))}
                   placeholder="≥ %"
                   className="h-[34px] px-2.5 rounded-[10px] text-[12px] w-[80px]"
                   style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }} />
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-[9px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>A % Min</span>
            <input type="number" inputMode="decimal" step="1" value={filters.a_min_pct}
                   onChange={e => setFilters(f => ({ ...f, a_min_pct: e.target.value }))}
                   placeholder="≥ %"
                   className="h-[34px] px-2.5 rounded-[10px] text-[12px] w-[80px]"
                   style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }} />
          </div>

          <div className="flex flex-col gap-1">
            <span className="text-[9px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>Date Range</span>
            <div className="flex gap-1 items-center">
              <input type="date" value={filters.from}
                     onChange={e => setFilters(f => ({ ...f, from: e.target.value }))}
                     className="h-[34px] px-2 rounded-[10px] text-[12px]"
                     style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)" }} />
              <span style={{ color: "var(--ink-4)" }}>–</span>
              <input type="date" value={filters.to}
                     onChange={e => setFilters(f => ({ ...f, to: e.target.value }))}
                     className="h-[34px] px-2 rounded-[10px] text-[12px]"
                     style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)" }} />
            </div>
          </div>

          <div className="ml-auto flex items-end gap-3">
            <span className="text-[11px]" style={{ color: "var(--ink-4)" }}>{sorted.length} of {rows.length}</span>
            {filtersDirty && (
              <button type="button" onClick={resetFilters}
                      className="h-[34px] px-3 rounded-[10px] text-[11px] font-medium"
                      style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-3)" }}>
                ✕ Reset
              </button>
            )}
          </div>
        </div>

        {/* Ledger table */}
        <div className="overflow-x-auto">
          <table className="w-full text-[12px]" style={{ borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ background: "var(--surface-2)" }}>
                {([
                  { k: "trade_id", l: "Trade ID", align: "left" },
                  { k: "ticker", l: "Ticker", align: "left" },
                  { k: "status", l: "Status", align: "left" },
                  { k: "open_date", l: "Open", align: "left" },
                  { k: "closed_date", l: "Close", align: "left" },
                  { k: "b_shares", l: "B Shares", align: "right" },
                  { k: "b_pnl", l: "B P&L", align: "right" },
                  { k: "b_return_pct", l: "B Return %", align: "right" },
                  { k: "a_shares", l: "A Shares", align: "right" },
                  { k: "a_pnl", l: "A P&L", align: "right" },
                  { k: "a_return_pct", l: "A Return %", align: "right" },
                  { k: "total_pnl", l: "Total P&L", align: "right" },
                  { k: "total_return_pct", l: "Total %", align: "right" },
                  { k: "mae_pct", l: "MAE %", align: "right" },
                  { k: "mfe_pct", l: "MFE %", align: "right" },
                  { k: "rule", l: "Rule", align: "left" },
                ] as { k: ColKey; l: string; align: "left" | "right" }[]).map(c => (
                  <th key={c.k} onClick={() => onSort(c.k)}
                      className="px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.04em] cursor-pointer select-none"
                      style={{
                        color: "var(--ink-4)",
                        borderBottom: "1px solid var(--border)",
                        textAlign: c.align,
                        whiteSpace: "nowrap",
                      }}>
                    {c.l}{sort.key === c.k ? (sort.dir === "asc" ? " ▲" : " ▼") : ""}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.length === 0 ? (
                <tr>
                  <td colSpan={16} className="px-3 py-8 text-center text-[12px]" style={{ color: "var(--ink-4)" }}>
                    {loading ? "Loading…" : "No trades match the current filters."}
                  </td>
                </tr>
              ) : sorted.map(r => (
                <EntryVsAddRow key={r.trade_id} row={r} />
              ))}
            </tbody>
            {sorted.length > 0 && (
              <tfoot>
                <tr style={{ background: "var(--bg-2)" }}>
                  <td colSpan={5} className="px-3 py-2 text-[10px] font-bold uppercase" style={{ color: "var(--ink-4)" }}>
                    Totals · {sorted.length} rows
                  </td>
                  <td className="px-3 py-2 text-right font-bold" style={{ fontFamily: mono }}>
                    {sorted.reduce((t, r) => t + r.b_shares, 0).toLocaleString()}
                  </td>
                  <td className="px-3 py-2 text-right font-bold"
                      style={{ fontFamily: mono, color: kpis.bRealized + sorted.reduce((t, r) => t + r.b_unrealized, 0) >= 0 ? "#08a86b" : "#e5484d" }}>
                    {formatCurrency(kpis.bRealized + sorted.reduce((t, r) => t + r.b_unrealized, 0), { decimals: 0 })}
                  </td>
                  <td className="px-3 py-2" />
                  <td className="px-3 py-2 text-right font-bold" style={{ fontFamily: mono }}>
                    {sorted.reduce((t, r) => t + r.a_shares, 0).toLocaleString()}
                  </td>
                  <td className="px-3 py-2 text-right font-bold"
                      style={{ fontFamily: mono, color: kpis.aRealized + sorted.reduce((t, r) => t + r.a_unrealized, 0) >= 0 ? "#08a86b" : "#e5484d" }}>
                    {formatCurrency(kpis.aRealized + sorted.reduce((t, r) => t + r.a_unrealized, 0), { decimals: 0 })}
                  </td>
                  <td className="px-3 py-2" />
                  <td className="px-3 py-2 text-right font-bold"
                      style={{ fontFamily: mono, color: sorted.reduce((t, r) => t + r.total_pnl, 0) >= 0 ? "#08a86b" : "#e5484d" }}>
                    {formatCurrency(sorted.reduce((t, r) => t + r.total_pnl, 0), { decimals: 0 })}
                  </td>
                  <td className="px-3 py-2 text-right font-bold" style={{ fontFamily: mono, color: "var(--ink-3)" }}>
                    {(() => {
                      // Blended Total % across the filtered set: Σ pnl ÷ Σ cost.
                      // Not a simple mean of per-row %s — capital-weighted is
                      // what you want for a portfolio-level read.
                      const sumPnl = sorted.reduce((t, r) => t + r.total_pnl, 0);
                      const sumCost = sorted.reduce((t, r) => t + r.b_initial_cost + r.a_initial_cost, 0);
                      if (sumCost <= 0) return "—";
                      const pct = (sumPnl / sumCost) * 100;
                      return <span style={{ color: pct >= 0 ? "#08a86b" : "#e5484d" }}>{pct.toFixed(1)}%</span>;
                    })()}
                  </td>
                  {/* MAE / MFE / Rule columns don't aggregate meaningfully;
                      leave empty in the totals row rather than displaying
                      a misleading average across trades. */}
                  <td className="px-3 py-2" />
                  <td className="px-3 py-2" />
                  <td className="px-3 py-2" />
                </tr>
              </tfoot>
            )}
          </table>
        </div>
      </div>
    </div>
  );
}

function EntryVsAddRow({ row: r }: { row: TradeRow }) {
  const statusBg = r.status === "Open"
    ? "color-mix(in oklab, #08a86b 14%, var(--surface))"
    : "color-mix(in oklab, #64748b 14%, var(--surface))";
  const statusColor = r.status === "Open" ? "#08a86b" : "var(--ink-3)";

  const fmtPnl = (v: number) => (
    <span style={{ color: v > 0 ? "#08a86b" : v < 0 ? "#e5484d" : "var(--ink-3)" }}>
      {formatCurrency(v, { decimals: 0 })}
    </span>
  );
  const fmtPct = (v: number | null) => v == null
    ? <span style={{ color: "var(--ink-4)" }}>—</span>
    : <span style={{ color: v > 0 ? "#08a86b" : v < 0 ? "#e5484d" : "var(--ink-3)" }}>{v.toFixed(1)}%</span>;
  const fmtShares = (n: number) => n === 0
    ? <span style={{ color: "var(--ink-4)" }}>—</span>
    : <span>{n.toLocaleString()}</span>;

  return (
    <tr style={{ borderBottom: "1px solid var(--border)" }}
        onMouseEnter={e => (e.currentTarget.style.background = "var(--bg-2)")}
        onMouseLeave={e => (e.currentTarget.style.background = "transparent")}>
      <td className="px-3 py-2" style={{ fontFamily: mono, color: "var(--ink-3)" }}>{r.trade_id}</td>
      <td className="px-3 py-2 font-semibold" style={{ fontFamily: mono }}>{r.ticker}</td>
      <td className="px-3 py-2">
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold"
              style={{ background: statusBg, color: statusColor }}>
          <span className="w-1 h-1 rounded-full" style={{ background: statusColor }} />
          {r.status}
        </span>
      </td>
      <td className="px-3 py-2" style={{ color: "var(--ink-4)", fontFamily: mono, fontSize: 11 }}>{r.open_date}</td>
      <td className="px-3 py-2" style={{ color: "var(--ink-4)", fontFamily: mono, fontSize: 11 }}>{r.closed_date || "—"}</td>
      <td className="px-3 py-2 text-right" style={{ fontFamily: mono }}>{fmtShares(r.b_shares)}</td>
      <td className="px-3 py-2 text-right font-semibold" style={{ fontFamily: mono }}>
        {r.b_shares > 0 ? fmtPnl(r.b_pnl) : <span style={{ color: "var(--ink-4)" }}>—</span>}
      </td>
      <td className="px-3 py-2 text-right" style={{ fontFamily: mono }}>{fmtPct(r.b_return_pct)}</td>
      <td className="px-3 py-2 text-right" style={{ fontFamily: mono }}>{fmtShares(r.a_shares)}</td>
      <td className="px-3 py-2 text-right font-semibold" style={{ fontFamily: mono }}>
        {r.a_shares > 0 ? fmtPnl(r.a_pnl) : <span style={{ color: "var(--ink-4)" }}>—</span>}
      </td>
      <td className="px-3 py-2 text-right" style={{ fontFamily: mono }}>{fmtPct(r.a_return_pct)}</td>
      <td className="px-3 py-2 text-right font-bold" style={{ fontFamily: mono }}>{fmtPnl(r.total_pnl)}</td>
      <td className="px-3 py-2 text-right" style={{ fontFamily: mono }}>{fmtPct(r.total_return_pct)}</td>
      <td className="px-3 py-2 text-right" style={{ fontFamily: mono }}>
        {fmtExcursion(r.mae_pct, r.atr21_entry_pct, "adverse")}
      </td>
      <td className="px-3 py-2 text-right" style={{ fontFamily: mono }}>
        {fmtExcursion(r.mfe_pct, r.atr21_entry_pct, "favorable")}
      </td>
      <td className="px-3 py-2 text-[11px]" style={{ color: "var(--ink-3)" }}>{r.rule}</td>
    </tr>
  );
}

// Dual-value cell: primary line = signed excursion %; secondary line
// = |excursion| ÷ atr21_entry_pct (×ATR multiple). Both null → em-dash.
// Direction ("adverse"|"favorable") only affects the color rule so a
// -0.0 MAE reads neutral, not red.
function fmtExcursion(
  pct: number | null,
  atr21: number | null,
  direction: "adverse" | "favorable",
) {
  if (pct == null) return <span style={{ color: "var(--ink-4)" }}>—</span>;
  const atrMult = atr21 != null && atr21 > 0 ? Math.abs(pct) / atr21 : null;
  // MAE is signed ≤ 0 (red when non-zero); MFE is signed ≥ 0 (green).
  // A zero excursion is displayed neutral.
  const color = pct === 0
    ? "var(--ink-3)"
    : direction === "adverse" ? "#e5484d" : "#08a86b";
  return (
    <span className="inline-flex flex-col items-end leading-tight">
      <span style={{ color }}>{pct.toFixed(1)}%</span>
      {atrMult != null && (
        <span className="text-[10px]" style={{ color: "var(--ink-4)" }}>
          {atrMult.toFixed(2)}× ATR
        </span>
      )}
    </span>
  );
}
