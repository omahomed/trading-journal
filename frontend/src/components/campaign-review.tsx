"use client";

// Campaign Review — per-campaign performance table with B/A series split,
// lesson tagging, and post-mortem drill-in. Superseded the old Entry-vs-Add
// page (2026-07-25) by absorbing the lesson features from the prior
// Campaign Review page and dropping grading (unused).
//
// Each row = one campaign with two performance buckets side by side: the
// B series (original entry lots) vs the A series (add-on lots). Answers
// "where did my profit come from, the initial entry or the scale-ins?"
// while also carrying the lesson chips + expandable editor for
// post-mortem notes.
//
// Math reuses the existing data layer entirely:
//   - lot_closures.realized_pl carries per-closure dollars + the
//     buy_trx_id it consumed → realized B/A split is just a groupby
//   - walkLedger (lib/campaign-detail-walk) walks all details once
//     and returns per-detail remaining shares → unrealized B/A is
//     the remaining lots × (mark − lot price) × multiplier
//   - r_multiple derived client-side: realized_pl / risk_budget for
//     CLOSED trades only (blank on OPEN)
//   - lesson_category + lesson_note fetched via api.getTradeLessons
//     and merged by trade_id

import { Fragment, useState, useEffect, useMemo, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
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
import { LESSON_CATEGORIES, CAT_COLORS, CAT_FALLBACK } from "@/lib/lesson-categories";
import { KPITile, TILE_GRADIENTS, SegmentedControl } from "./campaign-detail";
import { TradeOverviewSidecar } from "./trade-overview-sidecar";

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
  // Sell rule from the closing SELL row. Blank for OPEN campaigns
  // (nothing sold yet). Rendered as its own column so the exit reason
  // sits next to the buy setup.
  sell_rule: string;
  // R-multiple = realized_pl / risk_budget. Only computed for CLOSED
  // campaigns and only when risk_budget > 0 — an OPEN trade doesn't
  // have a final realized number worth normalizing yet, and a legacy
  // row without a stored risk budget can't produce a meaningful R.
  r_multiple: number | null;
  // Lesson chips + free-text note from trade_lessons (joined via
  // api.getTradeLessons). Categories are pipe-separated so a trade
  // can wear multiple tags. Both fields are "" when nothing tagged.
  lesson_category: string;
  lesson_note: string;
  multiplier: number;
  // Instrument classification snapshot — drives the Instrument filter
  // and any downstream option-specific rendering. Derived from
  // instrument_type + ticker shape at row-build time so the filter
  // predicate is a cheap boolean check.
  is_option: boolean;
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
  | "b_pnl" | "b_return_pct"
  | "a_pnl" | "a_return_pct"
  | "total_pnl" | "total_return_pct"
  | "r_multiple"
  | "mae_pct" | "mfe_pct"
  | "rule" | "sell_rule";

// Date range presets — Week / Month / YTD / Custom, mirroring the
// Trend Cycle Review page so filter UX stays consistent across the
// Deep Dive group. "All" is the everything-since-forever default.
// Custom pairs with the from/to inputs which only render when picked.
type DateRangeKey = "all" | "week" | "month" | "ytd" | "custom";

type InstrumentKey = "all" | "stocks" | "options";
// Rank filter — direction + absolute Top/Bottom N buckets. N is a
// fixed count (5/10/20), not a percentile — much easier to reason
// about ("show me my 10 biggest wins") than a %. Top/Bottom sort by
// total_pnl; ties broken by trade_id for determinism. "all" no-ops.
type RankKey = "all" | "winners" | "losers" | "top_5" | "top_10" | "top_20" | "bottom_5" | "bottom_10" | "bottom_20";

interface Filters {
  q: string;
  status: SeriesStatus[];  // empty = no filter
  tickers: string[];       // empty = no filter; multi-chip
  rule: string;            // rule or "all"
  pl: "all" | "realized" | "unrealized";
  rank: RankKey;           // direction / percentile bucket over total_pnl
  instrument: InstrumentKey;  // "all" | "stocks" | "options"
  lesson: string;          // "all" | "none" | category name
  // Numeric Return-% thresholds. Empty string = no filter. Set to "0"
  // to slice "positive only", or any number for ">= X%". Trades with
  // a null series Return % (e.g. no A lots) fall out of the A-min
  // filter when it's active — the row literally has no A series to
  // compare against.
  b_min_pct: string;
  a_min_pct: string;
  // Preset segmented control gates the date filter; from/to are only
  // consulted when dateRange === "custom".
  dateRange: DateRangeKey;
  from: string;            // YYYY-MM-DD (custom range)
  to: string;              // YYYY-MM-DD (custom range)
}
const EMPTY_FILTERS: Filters = {
  q: "", status: [], tickers: [], rule: "all", pl: "all", rank: "all",
  instrument: "stocks", lesson: "all",
  b_min_pct: "", a_min_pct: "",
  dateRange: "all", from: "", to: "",
};

const NUMERIC_KEYS = new Set<ColKey>([
  "b_pnl", "b_return_pct",
  "a_pnl", "a_return_pct",
  "total_pnl", "total_return_pct",
  "r_multiple",
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

// Date-preset predicate. YTD = current-year rows onward, Month =
// current calendar month, Week = Monday-anchored current week. Custom
// hands off to the from/to inputs. "all" always passes. Invalid date
// input passes for all/custom, fails for date-bounded presets so a
// row missing dates doesn't spuriously appear in "This Week".
function dateFilterPasses(dateStr: string, f: Filters): boolean {
  if (f.dateRange === "all") return true;
  const d = dateStr ? new Date(dateStr) : null;
  if (!d || isNaN(d.getTime())) return f.dateRange === "custom";
  const now = new Date();
  if (f.dateRange === "ytd") return d.getFullYear() === now.getFullYear();
  if (f.dateRange === "month") {
    return d.getFullYear() === now.getFullYear() && d.getMonth() === now.getMonth();
  }
  if (f.dateRange === "week") {
    const day = now.getDay();
    const daysSinceMon = (day + 6) % 7;
    const monday = new Date(now);
    monday.setHours(0, 0, 0, 0);
    monday.setDate(now.getDate() - daysSinceMon);
    return d >= monday;
  }
  // custom
  if (f.from) {
    const from = new Date(f.from);
    if (!isNaN(from.getTime()) && d < from) return false;
  }
  if (f.to) {
    const to = new Date(f.to);
    if (!isNaN(to.getTime()) && d > to) return false;
  }
  return true;
}

// Compute the per-trade row from the raw trades + details + closures +
// live prices. All math is dollar-denominated (multiplier folded into
// cost basis and unrealized so the same number works for stocks and
// options). Lesson data is intentionally NOT merged here — that
// happens post-fetch in the component so this pure function stays
// testable without a lesson-loading side dependency.
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

    // R-multiple: realized_pl / risk_budget. Closed-only — an open
    // campaign hasn't finalized its realized number, so the divisor
    // is comparing apples to a moving target. Null when either
    // component is missing or zero (legacy rows without risk_budget
    // populated → no R possible).
    const riskBudget = toNumOrNull((trade as { risk_budget?: number | null }).risk_budget);
    const realizedPl = toNumOrNull((trade as { realized_pl?: number | null }).realized_pl);
    const r_multiple = status === "Closed" && riskBudget != null && riskBudget > 0 && realizedPl != null
      ? realizedPl / riskBudget
      : null;

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
      // Sell rule only makes sense on closed campaigns (open trades
      // haven't fired an exit yet). Empty string on OPEN so the
      // column renders blank without producing a misleading "—".
      sell_rule: status === "Closed"
        ? String((trade as { sell_rule?: string }).sell_rule || "")
        : "",
      r_multiple,
      // Lesson fields land empty here; the component merges from
      // api.getTradeLessons once that fetch resolves.
      lesson_category: "",
      lesson_note: "",
      multiplier,
      is_option: isOption(trade),
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

export function CampaignReview({ navColor }: { navColor: string }) {
  const [openTrades, setOpenTrades] = useState<TradePosition[]>([]);
  const [closedTrades, setClosedTrades] = useState<TradePosition[]>([]);
  const [details, setDetails] = useState<TradeDetail[]>([]);
  const [closures, setClosures] = useState<LotClosure[]>([]);
  const [livePrices, setLivePrices] = useState<Record<string, number>>({});
  // Lessons keyed by trade_id — merged into rows post-fetch. Stored as
  // a map rather than baked directly into the row snapshot so an
  // optimistic edit doesn't require reissuing the whole trade fetch.
  const [lessons, setLessons] = useState<Record<string, { note: string; category: string }>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdatedAt, setLastUpdatedAt] = useState<Date | null>(null);
  const [filters, setFilters] = useState<Filters>(EMPTY_FILTERS);
  const [sort, setSort] = useState<{ key: ColKey; dir: "asc" | "desc" }>({ key: "open_date", dir: "desc" });
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [savingLessonId, setSavingLessonId] = useState<string | null>(null);
  // Right-click context menu — cursor-anchored. Mirrors the ACS pattern:
  // right-click on a row opens the menu; clicking anywhere else closes.
  const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number; row: TradeRow } | null>(null);
  // Sidecar overview — same shared component PHM uses. Null = closed.
  const [overviewTradeId, setOverviewTradeId] = useState<string | null>(null);
  const router = useRouter();

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
      const [opens, closeds, bundle, lessonBundle] = await Promise.all([
        api.tradesOpen(portfolio).catch(() => [] as TradePosition[]),
        api.tradesClosed(portfolio, 1000).catch(() => [] as TradePosition[]),
        api.tradesRecent(portfolio, 10000).catch(() => ({ details: [], lot_closures: [] })),
        api.getTradeLessons(portfolio).catch(() => ({ lessons: {} })),
      ]);
      setOpenTrades(opens);
      setClosedTrades(closeds);
      setDetails(bundle.details || []);
      setClosures(bundle.lot_closures || []);
      setLessons(lessonBundle?.lessons || {});
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

  // Context menu close — click anywhere OR press Esc.
  useEffect(() => {
    if (!ctxMenu) return;
    const close = () => setCtxMenu(null);
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") close(); };
    window.addEventListener("click", close);
    window.addEventListener("keydown", onKey);
    return () => { window.removeEventListener("click", close); window.removeEventListener("keydown", onKey); };
  }, [ctxMenu]);

  // Sidecar close — Esc dismisses. Backdrop click is handled by the
  // sidecar component itself.
  useEffect(() => {
    if (!overviewTradeId) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOverviewTradeId(null); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [overviewTradeId]);

  // Both open + closed trades feed the row set. Options + stocks flow
  // through the same math (multiplier folded into cost basis and
  // unrealized). Whether options are visible is controlled by the
  // Instrument filter below — the default is "stocks" since the B-vs-A
  // ROI framing is most meaningful for equity campaigns, but the user
  // can flip to "all" or "options" to reconcile against portfolio-
  // wide totals (e.g. Straight P&L vs Trades P&L cross-check).
  const allTrades = useMemo(
    () => [...openTrades, ...closedTrades],
    [openTrades, closedTrades],
  );
  const rows = useMemo(
    () => {
      const base = computeTradeRows(allTrades, details, closures, livePrices);
      // Merge lesson data by trade_id. Empty strings when a trade has
      // no lesson row yet, which is the default from computeTradeRows.
      return base.map(r => {
        const l = lessons[r.trade_id];
        if (!l) return r;
        return {
          ...r,
          lesson_category: l.category || "",
          lesson_note: l.note || "",
        };
      });
    },
    [allTrades, details, closures, livePrices, lessons],
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
    // Pass 1: apply every filter EXCEPT rank. Rank's percentile buckets
    // need the otherwise-filtered set as their reference population
    // (Top 10% of NVDA trades ≠ Top 10% of all trades where I picked NVDA).
    const pass1 = rows.filter(r => {
      const q = filters.q.trim().toLowerCase();
      if (q) {
        const haystack = [r.ticker, r.trade_id, r.rule].map(v => v.toLowerCase());
        if (!haystack.some(s => s.includes(q))) return false;
      }
      if (filters.status.length > 0 && !filters.status.includes(r.status)) return false;
      if (filters.tickers.length > 0 && !filters.tickers.includes(r.ticker)) return false;
      if (filters.rule !== "all" && r.rule !== filters.rule) return false;
      // Instrument filter — default is "stocks" (equity-only, matches
      // the page's original intent). "all" = drop the gate entirely;
      // "options" narrows to option contracts.
      if (filters.instrument === "stocks" && r.is_option) return false;
      if (filters.instrument === "options" && !r.is_option) return false;
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
      // Lesson filter. "none" = show only untagged trades. Otherwise
      // match against pipe-separated categories (a trade can wear
      // multiple tags — "Discipline|Sizing").
      if (filters.lesson === "none") {
        if (r.lesson_category.trim() !== "") return false;
      } else if (filters.lesson !== "all") {
        const cats = r.lesson_category.split("|").map(s => s.trim()).filter(Boolean);
        if (!cats.includes(filters.lesson)) return false;
      }
      // Date-preset filter. Uses close_date when set (post-mortem
      // semantics: "trades I closed in the last N"); falls back to
      // open_date for open positions so they don't get dropped from
      // presets. Custom pulls from filters.from / filters.to.
      const d = (r.closed_date || r.open_date).slice(0, 10);
      if (!dateFilterPasses(d, filters)) return false;
      return true;
    });

    // Pass 2: rank filter. Winners / losers gate on total_pnl sign
    // (includes unrealized so open positions with mark-to-market gain
    // count as winners). Top/Bottom N buckets sort pass1 by total_pnl
    // and slice — capped at pass1.length so "Top 20" with 7 trades
    // returns all 7 (no wraparound / no error).
    if (filters.rank === "all") return pass1;
    if (filters.rank === "winners") return pass1.filter(r => r.total_pnl > 0);
    if (filters.rank === "losers") return pass1.filter(r => r.total_pnl < 0);
    const n = pass1.length;
    if (n === 0) return pass1;
    const takeCount = filters.rank === "top_5" || filters.rank === "bottom_5" ? 5
      : filters.rank === "top_10" || filters.rank === "bottom_10" ? 10
      : 20;  // top_20 / bottom_20
    const isTop = filters.rank === "top_5" || filters.rank === "top_10" || filters.rank === "top_20";
    const rankedSorted = [...pass1].sort((a, b) =>
      isTop ? (b.total_pnl - a.total_pnl) : (a.total_pnl - b.total_pnl),
    );
    const keep = new Set(rankedSorted.slice(0, takeCount).map(r => r.trade_id));
    return pass1.filter(r => keep.has(r.trade_id));
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

  // KPI strip — all totals scoped to the FILTERED set (matches what
  // the user sees in the table). Win % is over decided closed
  // campaigns; ties (realized === 0) fall out of both W and L.
  // "Trades P&L" is capital-based on B + A across the whole visible
  // set — realized + unrealized. Not to be confused with Period
  // Review's Straight P&L which is capital-deployment based.
  const kpis = useMemo(() => {
    const openCount = sorted.filter(r => r.status === "Open").length;
    const closedCount = sorted.filter(r => r.status === "Closed").length;
    const closedRows = sorted.filter(r => r.status === "Closed");
    let winners = 0, losers = 0;
    for (const r of closedRows) {
      const pl = (r.b_realized + r.a_realized);
      if (pl > 0) winners += 1;
      else if (pl < 0) losers += 1;
    }
    const decided = winners + losers;
    const winRate = decided > 0 ? winners / decided : null;
    const bRealized = sorted.reduce((t, r) => t + r.b_realized, 0);
    const aRealized = sorted.reduce((t, r) => t + r.a_realized, 0);
    const bUnrealized = sorted.reduce((t, r) => t + r.b_unrealized, 0);
    const aUnrealized = sorted.reduce((t, r) => t + r.a_unrealized, 0);
    const bPnl = bRealized + bUnrealized;
    const aPnl = aRealized + aUnrealized;
    const totalPnl = bPnl + aPnl;
    const tradesPnlPerTrade = sorted.length > 0 ? totalPnl / sorted.length : 0;
    // Add-on activity: how many A-series lots exist and across how
    // many distinct campaigns. Tells the user whether their add
    // pattern is broad or concentrated.
    let addOnLots = 0;
    let campaignsWithAdds = 0;
    for (const r of sorted) {
      if (r.a_shares > 0) {
        campaignsWithAdds += 1;
        // a_shares is total shares from A* trx_ids; not a lot count.
        // We approximate by dividing by the average A lot size seen
        // — but that's noise. Simpler: count details post-hoc.
        addOnLots += r.a_shares > 0 ? 1 : 0;  // per-campaign at-least-one
      }
    }
    return {
      trades: sorted.length,
      openCount, closedCount,
      winners, losers, decided, winRate,
      bRealized, aRealized, bUnrealized, aUnrealized,
      bPnl, aPnl, totalPnl,
      tradesPnlPerTrade,
      campaignsWithAdds,
    };
  }, [sorted]);

  // Rule-level rollup for the Setup Performance expander. Computed
  // over the FILTERED set so date presets etc. cascade — "which setups
  // are working THIS MONTH" is a legitimate question and the answer
  // needs to respect the same date lens as the ledger below.
  //
  // Cutoff: trades opened before 2026-01-01 are excluded — the rule/setup
  // tagging system landed in Jan 2026, so anything opened prior carries
  // legacy "History" tags that would pollute the setup-performance signal.
  // The ledger table still shows those trades (unaffected); only this
  // aggregate is cutoff-scoped.
  const SETUP_CUTOFF = "2026-01-01";
  const setupRollup = useMemo(() => {
    type Bucket = {
      rule: string;
      trades: number;
      winners: number;
      losers: number;
      total_pnl: number;
      best_pnl: number;
      worst_pnl: number;
    };
    const buckets = new Map<string, Bucket>();
    for (const r of sorted) {
      if (r.open_date < SETUP_CUTOFF) continue;
      const key = r.rule.trim() || "(untagged)";
      const b = buckets.get(key) || {
        rule: key, trades: 0, winners: 0, losers: 0,
        total_pnl: 0, best_pnl: -Infinity, worst_pnl: Infinity,
      };
      b.trades += 1;
      if (r.total_pnl > 0) b.winners += 1;
      else if (r.total_pnl < 0) b.losers += 1;
      b.total_pnl += r.total_pnl;
      if (r.total_pnl > b.best_pnl) b.best_pnl = r.total_pnl;
      if (r.total_pnl < b.worst_pnl) b.worst_pnl = r.total_pnl;
      buckets.set(key, b);
    }
    return Array.from(buckets.values())
      .map(b => ({
        ...b,
        avg_pnl: b.trades > 0 ? b.total_pnl / b.trades : 0,
        win_rate: (b.winners + b.losers) > 0 ? b.winners / (b.winners + b.losers) : null,
      }))
      .sort((a, b) => b.total_pnl - a.total_pnl);
  }, [sorted]);

  // Count of untagged CLOSED trades in the full row set (ignores
  // current filters — the point of the chip is "what needs tagging
  // in your book overall", not "what needs tagging inside the view
  // I've already narrowed to"). Ignores open trades since lesson
  // tagging is post-mortem semantics.
  const untaggedClosedCount = useMemo(
    () => rows.filter(r => r.status === "Closed" && r.lesson_category.trim() === "").length,
    [rows],
  );

  // Distinct add-on lot count across the filtered set (walks details
  // once). Kept separate from kpis to avoid re-computing when the
  // details array itself hasn't changed. Used for the A P&L sub-label.
  const addOnLotCount = useMemo(() => {
    const filteredIds = new Set(sorted.map(r => r.trade_id));
    let n = 0;
    for (const d of details) {
      if (!filteredIds.has(String(d.trade_id || ""))) continue;
      if (String(d.action).toUpperCase() !== "BUY") continue;
      if (String(d.trx_id || "").toUpperCase().startsWith("A")) n += 1;
    }
    return n;
  }, [details, sorted]);

  const onSort = (key: ColKey) => {
    setSort(s => s.key === key
      ? { key, dir: s.dir === "asc" ? "desc" : "asc" }
      : { key, dir: NUMERIC_KEYS.has(key) ? "desc" : "asc" });
  };

  // Optimistic lesson-category toggle. Categories are pipe-separated
  // in the DB (see LESSON_CATEGORIES catalog + Log Sell). Failure
  // rolls back by re-fetching the whole lesson bundle.
  const toggleCategory = useCallback(async (trade_id: string, cat: string) => {
    const current = lessons[trade_id]?.category || "";
    const currentList = current.split("|").map(s => s.trim()).filter(Boolean);
    const nextList = currentList.includes(cat)
      ? currentList.filter(c => c !== cat)
      : [...currentList, cat];
    const nextCategory = nextList.join("|");
    const prevNote = lessons[trade_id]?.note || "";
    setLessons(prev => ({ ...prev, [trade_id]: { note: prevNote, category: nextCategory } }));
    setSavingLessonId(trade_id);
    try {
      await api.saveTradeLessons({
        portfolio: getActivePortfolio(),
        trade_id,
        note: prevNote,
        category: nextCategory,
      });
    } catch (e) {
      log.warn("campaign-review", "toggleCategory failed", e);
      // Re-fetch to reconcile on failure.
      try {
        const fresh = await api.getTradeLessons(getActivePortfolio());
        setLessons(fresh?.lessons || {});
      } catch { /* leave optimistic state; user retry will re-attempt */ }
    }
    setSavingLessonId(null);
  }, [lessons]);

  // Save note on blur. Debounce is overkill for a post-mortem page —
  // users rarely type in this box more than a couple of times per
  // session, and every write is a small POST.
  const saveNote = useCallback(async (trade_id: string, note: string) => {
    const prevNote = lessons[trade_id]?.note || "";
    if (note === prevNote) return;
    const prevCategory = lessons[trade_id]?.category || "";
    setLessons(prev => ({ ...prev, [trade_id]: { note, category: prevCategory } }));
    setSavingLessonId(trade_id);
    try {
      await api.saveTradeLessons({
        portfolio: getActivePortfolio(),
        trade_id,
        note,
        category: prevCategory,
      });
    } catch (e) {
      log.warn("campaign-review", "saveNote failed", e);
      try {
        const fresh = await api.getTradeLessons(getActivePortfolio());
        setLessons(fresh?.lessons || {});
      } catch { /* fall through */ }
    }
    setSavingLessonId(null);
  }, [lessons]);

  const filtersDirty = useMemo(() => (
    !!filters.q || filters.status.length > 0 || filters.tickers.length > 0
    || filters.rule !== "all" || filters.pl !== "all" || filters.rank !== "all"
    || filters.instrument !== "stocks" || filters.lesson !== "all"
    || !!filters.b_min_pct || !!filters.a_min_pct
    || filters.dateRange !== "all" || !!filters.from || !!filters.to
  ), [filters]);
  const resetFilters = () => setFilters(EMPTY_FILTERS);

  const onExportCsv = useCallback(() => {
    const header = [
      "Trade ID", "Ticker", "Status", "Open", "Close",
      "B Cost", "B Realized", "B Unrealized", "B P&L", "B Return %",
      "A Cost", "A Realized", "A Unrealized", "A P&L", "A Return %",
      "Total P&L", "Total Return %", "R",
      "MAE %", "MFE %", "MAE ATR", "MFE ATR", "ATR21 Entry %",
      "Buy Rule", "Sell Rule", "Lesson Categories", "Lesson Note",
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
        r.b_initial_cost.toFixed(2),
        r.b_realized.toFixed(2), r.b_unrealized.toFixed(2),
        r.b_pnl.toFixed(2), r.b_return_pct?.toFixed(2) ?? "",
        r.a_initial_cost.toFixed(2),
        r.a_realized.toFixed(2), r.a_unrealized.toFixed(2),
        r.a_pnl.toFixed(2), r.a_return_pct?.toFixed(2) ?? "",
        r.total_pnl.toFixed(2), r.total_return_pct?.toFixed(2) ?? "",
        r.r_multiple?.toFixed(2) ?? "",
        r.mae_pct?.toFixed(2) ?? "",
        r.mfe_pct?.toFixed(2) ?? "",
        maeAtr?.toFixed(2) ?? "",
        mfeAtr?.toFixed(2) ?? "",
        r.atr21_entry_pct?.toFixed(2) ?? "",
        r.rule, r.sell_rule,
        r.lesson_category, r.lesson_note,
      ].map(escape).join(",");
    });
    const csv = [header, ...lines].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `campaign-review-${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [sorted]);

  const lastUpdatedLabel = lastUpdatedAt
    ? `${lastUpdatedAt.toISOString().slice(0, 10)} ${String(lastUpdatedAt.getHours()).padStart(2, "0")}:${String(lastUpdatedAt.getMinutes()).padStart(2, "0")}`
    : "";

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }} data-testid="campaign-review-root">
      {/* Page header */}
      <div className="mb-[22px] pb-[14px] flex items-end justify-between gap-4"
           style={{ borderBottom: "1px solid var(--border)" }}>
        <div>
          <h1 className="font-normal text-[32px] tracking-tight m-0"
              style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
            Campaign <em className="italic" style={{ color: navColor }}>Review</em>
          </h1>
          <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
            Per-campaign performance with entry-vs-add series split and post-mortem lesson tagging
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
        <>
        <div className="grid grid-cols-5 gap-[14px]">
          <KPITile
            label="Trades"
            value={String(kpis.trades)}
            sub={`${kpis.openCount} open · ${kpis.closedCount} closed`}
            gradient={TILE_GRADIENTS.indigo}
          />
          <KPITile
            label="Win %"
            value={kpis.winRate == null ? "—" : `${(kpis.winRate * 100).toFixed(0)}%`}
            sub={kpis.decided > 0
              ? `${kpis.winners} of ${kpis.decided} closed profitable`
              : "no closed trades in view"}
            gradient={kpis.winRate == null
              ? TILE_GRADIENTS.blue
              : kpis.winRate >= 0.5 ? TILE_GRADIENTS.green : TILE_GRADIENTS.orange}
          />
          <KPITile
            label="B P&L"
            value={formatCurrency(kpis.bPnl, { decimals: 0 })}
            sub={`${formatCurrency(kpis.bRealized, { decimals: 0 })} realized · ${formatCurrency(kpis.bUnrealized, { decimals: 0 })} unrealized`}
            gradient={kpis.bPnl >= 0 ? TILE_GRADIENTS.green : TILE_GRADIENTS.red}
          />
          <KPITile
            label="A P&L"
            value={formatCurrency(kpis.aPnl, { decimals: 0 })}
            sub={addOnLotCount === 0
              ? "no add-ons in view"
              : `${addOnLotCount} add-on${addOnLotCount === 1 ? "" : "s"} across ${kpis.campaignsWithAdds} trade${kpis.campaignsWithAdds === 1 ? "" : "s"}`}
            gradient={kpis.aPnl >= 0 ? TILE_GRADIENTS.pink : TILE_GRADIENTS.red}
          />
          <KPITile
            label="Trades P&L"
            value={formatCurrency(kpis.totalPnl, { decimals: 0 })}
            sub={kpis.trades > 0
              ? `avg ${formatCurrency(kpis.tradesPnlPerTrade, { decimals: 0 })}/trade · realized + unrealized`
              : "no trades in view"}
            gradient={kpis.totalPnl >= 0 ? TILE_GRADIENTS.orange : TILE_GRADIENTS.red}
          />
        </div>

        {/* Quick-actions chip row — one-click filters for common
            questions. Only renders when there's something worth
            surfacing (e.g. no chip when everything is tagged). */}
        {untaggedClosedCount > 0 && (
          <div className="flex gap-2 mt-3">
            <button
              type="button"
              onClick={() => setFilters(f => ({ ...f, lesson: "none", status: ["Closed"] }))}
              className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-semibold cursor-pointer transition-all hover:brightness-95"
              style={{
                background: "color-mix(in oklab, #d97706 10%, var(--surface))",
                color: "#d97706",
                border: "1px solid color-mix(in oklab, #d97706 30%, var(--border))",
              }}
              data-testid="cr-untagged-chip"
              title="Click to filter to untagged closed trades"
            >
              🎓 {untaggedClosedCount} untagged closed trade{untaggedClosedCount === 1 ? "" : "s"}
              <span style={{ opacity: 0.7 }}>→</span>
            </button>
          </div>
        )}
        </>
      )}

      {/* Setup Performance expander — rule-level rollup over the
          filtered set. Closed by default so daily use isn't cluttered.
          Click a rule row to filter the ledger below to that rule. */}
      {sorted.length > 0 && (
        <details className="mt-5 rounded-[14px] overflow-hidden"
                 style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
          <summary className="px-[18px] py-[12px] flex items-center gap-2 cursor-pointer text-[13px] font-semibold list-none"
                   style={{ borderBottom: "1px solid transparent" }}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
            Setup Performance
            <span className="text-[12px] font-normal" style={{ color: "var(--ink-4)" }}>
              · {setupRollup.length} rule{setupRollup.length === 1 ? "" : "s"} · opened ≥ {SETUP_CUTOFF}
              {setupRollup.length > 0 && setupRollup[0].total_pnl > 0 && (
                <> · best: <b style={{ color: "var(--ink-3)" }}>{setupRollup[0].rule}</b>{" "}
                <span style={{ color: "#08a86b" }}>{formatCurrency(setupRollup[0].total_pnl, { decimals: 0 })}</span></>
              )}
            </span>
            <span className="ml-auto text-[11px]" style={{ color: "var(--ink-4)" }}>▾</span>
          </summary>
          <div className="overflow-x-auto" style={{ borderTop: "1px solid var(--border)" }}>
            <table className="w-full text-[12px]" style={{ borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ background: "var(--surface-2)" }}>
                  {[
                    { l: "Rule", align: "left" },
                    { l: "Trades", align: "right" },
                    { l: "Win %", align: "right" },
                    { l: "Avg $", align: "right" },
                    { l: "Total $", align: "right" },
                    { l: "Best", align: "right" },
                    { l: "Worst", align: "right" },
                  ].map(c => (
                    <th key={c.l}
                        className="px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.04em]"
                        style={{
                          color: "var(--ink-4)",
                          borderBottom: "1px solid var(--border)",
                          textAlign: c.align as "left" | "right",
                          whiteSpace: "nowrap",
                        }}>
                      {c.l}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {setupRollup.map(b => {
                  const isSelected = filters.rule === b.rule;
                  const clickable = b.rule !== "(untagged)";
                  return (
                    <tr
                      key={b.rule}
                      onClick={() => {
                        if (!clickable) return;
                        // Toggle: click again to clear the filter.
                        setFilters(f => ({ ...f, rule: isSelected ? "all" : b.rule }));
                      }}
                      className={clickable ? "cursor-pointer" : ""}
                      style={{
                        borderBottom: "1px solid var(--border)",
                        background: isSelected ? "color-mix(in oklab, " + navColor + " 8%, transparent)" : "transparent",
                      }}
                      onMouseEnter={e => { if (clickable && !isSelected) e.currentTarget.style.background = "var(--bg-2)"; }}
                      onMouseLeave={e => { if (clickable && !isSelected) e.currentTarget.style.background = "transparent"; }}
                      title={clickable ? (isSelected ? "Click to clear rule filter" : `Click to filter ledger to "${b.rule}"`) : undefined}
                    >
                      <td className="px-3 py-2 font-medium">
                        {b.rule === "(untagged)"
                          ? <span style={{ color: "var(--ink-4)", fontStyle: "italic" }}>{b.rule}</span>
                          : <span style={{ color: "var(--ink-2)" }}>{b.rule}</span>}
                      </td>
                      <td className="px-3 py-2 text-right" style={{ fontFamily: mono }}>{b.trades}</td>
                      <td className="px-3 py-2 text-right" style={{ fontFamily: mono, color: b.win_rate == null ? "var(--ink-4)" : b.win_rate >= 0.5 ? "#08a86b" : "#d97706" }}>
                        {b.win_rate == null ? "—" : `${(b.win_rate * 100).toFixed(0)}%`}
                      </td>
                      <td className="px-3 py-2 text-right" style={{ fontFamily: mono, color: b.avg_pnl > 0 ? "#08a86b" : b.avg_pnl < 0 ? "#e5484d" : "var(--ink-3)" }}>
                        {formatCurrency(b.avg_pnl, { decimals: 0 })}
                      </td>
                      <td className="px-3 py-2 text-right font-semibold" style={{ fontFamily: mono, color: b.total_pnl > 0 ? "#08a86b" : b.total_pnl < 0 ? "#e5484d" : "var(--ink-3)" }}>
                        {formatCurrency(b.total_pnl, { decimals: 0 })}
                      </td>
                      <td className="px-3 py-2 text-right" style={{ fontFamily: mono, color: "#08a86b" }}>
                        {formatCurrency(b.best_pnl, { decimals: 0 })}
                      </td>
                      <td className="px-3 py-2 text-right" style={{ fontFamily: mono, color: b.worst_pnl < 0 ? "#e5484d" : "var(--ink-3)" }}>
                        {formatCurrency(b.worst_pnl, { decimals: 0 })}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </details>
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

        {/* Filter toolbar — collapsed to two rows so the eye doesn't
            traverse three near-empty bands:
              Row 1 (text-heavy): Search grows to fill · Tickers · Rule · Lesson · Rank
              Row 2 (compact controls): Status · P&L · Instrument · Date (+ custom range) · B% · A% · counter+reset
            Row 1's controls sit AFTER a flex-grow Search so the row
            fills width regardless of how many chips the user has in
            Tickers. */}
        <div className="px-[18px] py-[14px] flex flex-col gap-[12px]"
             style={{ background: "var(--bg-2)", borderBottom: "1px solid var(--border)" }}>

          {/* Row 1 — text / high-cardinality dropdowns. Search takes the
              remaining horizontal slack via flex-1. */}
          <div className="flex flex-wrap items-end gap-[12px_14px]">
            <div className="flex flex-col gap-1 flex-1" style={{ minWidth: 240 }}>
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

            <FilterSelect label="Lesson"
              value={filters.lesson}
              onChange={v => setFilters(f => ({ ...f, lesson: v }))}
              options={[
                { v: "all", l: "All lessons" },
                { v: "none", l: "— untagged" },
                ...LESSON_CATEGORIES.map(c => ({ v: c, l: c })),
              ]}
            />

            <FilterSelect label="Rank"
              value={filters.rank}
              onChange={v => setFilters(f => ({ ...f, rank: v as RankKey }))}
              options={[
                { v: "all", l: "All ranks" },
                { v: "winners", l: "Winners only" },
                { v: "losers", l: "Losers only" },
                { v: "top_5", l: "Top 5 by P&L" },
                { v: "top_10", l: "Top 10 by P&L" },
                { v: "top_20", l: "Top 20 by P&L" },
                { v: "bottom_5", l: "Bottom 5 by P&L" },
                { v: "bottom_10", l: "Bottom 10 by P&L" },
                { v: "bottom_20", l: "Bottom 20 by P&L" },
              ]}
            />
          </div>

          {/* Row 2 — compact segments + numeric + counter/reset (right-anchored). */}
          <div className="flex flex-wrap items-end gap-[12px_14px]">
            <StatusMultiSelect
              value={filters.status}
              onChange={next => setFilters(f => ({ ...f, status: next }))}
            />

            <SegmentedControl label="P&L"
              value={filters.pl}
              onChange={v => setFilters(f => ({ ...f, pl: v as Filters["pl"] }))}
              options={[{ v: "all", l: "All" }, { v: "realized", l: "Realized" }, { v: "unrealized", l: "Unrealized" }]}
              testId="filter-pl"
            />

            <SegmentedControl label="Instrument"
              value={filters.instrument}
              onChange={v => setFilters(f => ({ ...f, instrument: v as InstrumentKey }))}
              options={[{ v: "all", l: "All" }, { v: "stocks", l: "Stocks" }, { v: "options", l: "Options" }]}
              testId="filter-instrument"
            />

            <SegmentedControl label="Date"
              value={filters.dateRange}
              onChange={v => setFilters(f => ({ ...f, dateRange: v as DateRangeKey }))}
              options={[
                { v: "all", l: "All" },
                { v: "week", l: "Week" },
                { v: "month", l: "Month" },
                { v: "ytd", l: "YTD" },
                { v: "custom", l: "Custom" },
              ]}
              testId="filter-date-range"
            />
            {filters.dateRange === "custom" && (
              <div className="flex flex-col gap-1">
                <span className="text-[9px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>Range</span>
                <div className="flex gap-1 items-center">
                  <input type="date" value={filters.from}
                         onChange={e => setFilters(f => ({ ...f, from: e.target.value }))}
                         className="h-[34px] px-2 rounded-[10px] text-[12px]"
                         style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }} />
                  <span style={{ color: "var(--ink-4)" }}>–</span>
                  <input type="date" value={filters.to}
                         onChange={e => setFilters(f => ({ ...f, to: e.target.value }))}
                         className="h-[34px] px-2 rounded-[10px] text-[12px]"
                         style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }} />
                </div>
              </div>
            )}

            {/* Per-series Return % min thresholds. Empty input = no
                filter. Type "0" for positives only, "50" for ">= 50%". */}
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
                  { k: "b_pnl", l: "B P&L", align: "right" },
                  { k: "b_return_pct", l: "B Return %", align: "right" },
                  { k: "a_pnl", l: "A P&L", align: "right" },
                  { k: "a_return_pct", l: "A Return %", align: "right" },
                  { k: "total_pnl", l: "Total P&L", align: "right" },
                  { k: "total_return_pct", l: "Total %", align: "right" },
                  {
                    k: "r_multiple", l: "R", align: "right",
                    tip: "R-multiple = realized P&L / initial risk budget. Blank on OPEN campaigns (final realized isn't in yet) and on legacy rows without risk_budget populated.",
                  },
                  {
                    k: "mae_pct", l: "MAE %", align: "right",
                    tip: "Maximum Adverse Excursion. The worst % below your B1 entry price the trade ever printed on any daily bar after entry. Bar 0 (entry day) is skipped unless there was a same-day sell — the reversal-candle low often prints BEFORE your entry and doesn't reflect anything you actually held through. Sub-line shows the ratio to ATR21 at entry (how many typical daily ranges the drawdown covered).",
                  },
                  {
                    k: "mfe_pct", l: "MFE %", align: "right",
                    tip: "Maximum Favorable Excursion. The best % above your B1 entry price the trade ever printed on any daily bar after entry. Bar 0 (entry day) is skipped unless there was a same-day sell above entry. Sub-line shows the ratio to ATR21 at entry.",
                  },
                  { k: "rule", l: "Buy Rule", align: "left" },
                  { k: "sell_rule", l: "Sell Rule", align: "left" },
                ] as { k: ColKey; l: string; align: "left" | "right"; tip?: string }[]).map(c => (
                  <th key={c.k} onClick={() => onSort(c.k)}
                      title={c.tip}
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
                {/* Non-sortable Lesson column + expand caret. Chip
                    content sort is not meaningful; caret is a click
                    target, not data. */}
                <th className="px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.04em] select-none"
                    style={{ color: "var(--ink-4)", borderBottom: "1px solid var(--border)", textAlign: "left", whiteSpace: "nowrap" }}>
                  Lesson
                </th>
                <th className="px-3 py-2 text-[10px]" style={{ borderBottom: "1px solid var(--border)" }} />
              </tr>
            </thead>
            <tbody>
              {sorted.length === 0 ? (
                <tr>
                  <td colSpan={17} className="px-3 py-8 text-center text-[12px]" style={{ color: "var(--ink-4)" }}>
                    {loading ? "Loading…" : "No trades match the current filters."}
                  </td>
                </tr>
              ) : sorted.map(r => (
                <Fragment key={r.trade_id}>
                  <CampaignReviewRow
                    row={r}
                    expanded={expandedId === r.trade_id}
                    onToggleExpand={() => setExpandedId(expandedId === r.trade_id ? null : r.trade_id)}
                    onContextMenu={e => {
                      e.preventDefault();
                      setCtxMenu({ x: e.clientX, y: e.clientY, row: r });
                    }}
                  />
                  {expandedId === r.trade_id && (
                    <tr style={{ background: "var(--bg-2)" }}>
                      <td colSpan={17} className="px-4 py-4">
                        <LessonEditor
                          row={r}
                          saving={savingLessonId === r.trade_id}
                          onToggleCategory={cat => toggleCategory(r.trade_id, cat)}
                          onSaveNote={note => saveNote(r.trade_id, note)}
                        />
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
            </tbody>
            {sorted.length > 0 && (
              <tfoot>
                <tr style={{ background: "var(--bg-2)" }}>
                  <td colSpan={5} className="px-3 py-2 text-[10px] font-bold uppercase" style={{ color: "var(--ink-4)" }}>
                    Totals · {sorted.length} rows
                  </td>
                  <td className="px-3 py-2 text-right font-bold"
                      style={{ fontFamily: mono, color: kpis.bPnl >= 0 ? "#08a86b" : "#e5484d" }}>
                    {formatCurrency(kpis.bPnl, { decimals: 0 })}
                  </td>
                  <td className="px-3 py-2" />
                  <td className="px-3 py-2 text-right font-bold"
                      style={{ fontFamily: mono, color: kpis.aPnl >= 0 ? "#08a86b" : "#e5484d" }}>
                    {formatCurrency(kpis.aPnl, { decimals: 0 })}
                  </td>
                  <td className="px-3 py-2" />
                  <td className="px-3 py-2 text-right font-bold"
                      style={{ fontFamily: mono, color: kpis.totalPnl >= 0 ? "#08a86b" : "#e5484d" }}>
                    {formatCurrency(kpis.totalPnl, { decimals: 0 })}
                  </td>
                  <td className="px-3 py-2 text-right font-bold" style={{ fontFamily: mono, color: "var(--ink-3)" }}>
                    {(() => {
                      // Blended Total % across the filtered set: Σ pnl ÷ Σ cost.
                      // Capital-weighted, not a simple mean of per-row %s.
                      const sumCost = sorted.reduce((t, r) => t + r.b_initial_cost + r.a_initial_cost, 0);
                      if (sumCost <= 0) return "—";
                      const pct = (kpis.totalPnl / sumCost) * 100;
                      return <span style={{ color: pct >= 0 ? "#08a86b" : "#e5484d" }}>{pct.toFixed(1)}%</span>;
                    })()}
                  </td>
                  {/* R / MAE / MFE / Buy Rule / Sell Rule / Lesson /
                      caret columns don't aggregate meaningfully; leave
                      empty in the totals row rather than displaying
                      a misleading average across trades. */}
                  <td className="px-3 py-2" />
                  <td className="px-3 py-2" />
                  <td className="px-3 py-2" />
                  <td className="px-3 py-2" />
                  <td className="px-3 py-2" />
                  <td className="px-3 py-2" />
                  <td className="px-3 py-2" />
                </tr>
              </tfoot>
            )}
          </table>
        </div>
      </div>

      {/* Right-click context menu — cursor-anchored. Two actions:
          overview (opens the shared sidecar) + view in trade journal
          (deep-links via ?trade_id=). Menu itself dismisses on any
          window click or Esc — handled by the effect at top. */}
      {ctxMenu && (
        <div className="fixed z-50 rounded-[10px] py-1.5 min-w-[200px] overflow-hidden"
             style={{
               left: ctxMenu.x,
               top: ctxMenu.y,
               background: "var(--surface)",
               border: "1px solid var(--border)",
               boxShadow: "0 8px 24px rgba(0,0,0,0.16), 0 2px 6px rgba(0,0,0,0.08)",
             }}
             data-testid="cr-context-menu"
             onClick={e => e.stopPropagation()}>
          <div className="px-3 py-1.5 text-[10px] uppercase tracking-[0.08em] font-semibold"
               style={{ color: "var(--ink-4)" }}>
            {ctxMenu.row.ticker} · {ctxMenu.row.trade_id}
          </div>
          <button
            type="button"
            className="w-full text-left px-3 py-2 text-[12px] font-medium flex items-center gap-2 transition-colors hover:brightness-95"
            style={{ color: "var(--ink)" }}
            data-testid="cr-ctx-overview"
            onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-2)")}
            onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
            onClick={() => { setOverviewTradeId(ctxMenu.row.trade_id); setCtxMenu(null); }}
          >
            <span style={{ color: "var(--ink-4)" }}>&#x1F50D;</span> Overview
          </button>
          <button
            type="button"
            className="w-full text-left px-3 py-2 text-[12px] font-medium flex items-center gap-2 transition-colors hover:brightness-95"
            style={{ color: "var(--ink)" }}
            data-testid="cr-ctx-journal"
            onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-2)")}
            onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
            onClick={() => {
              router.push(`/trade-journal?trade_id=${encodeURIComponent(ctxMenu.row.trade_id)}`);
              setCtxMenu(null);
            }}
          >
            <span style={{ color: "var(--ink-4)" }}>&#x1F4CB;</span> View in Trade Journal
          </button>
        </div>
      )}

      {/* Sidecar mount — shared component. Renders when overviewTradeId
          is set; nothing when null. Sourced trade + details are looked
          up from state; if the row disappears while the sidecar is open
          (unlikely — refresh would need to strip it), we quietly bail. */}
      {overviewTradeId && (() => {
        const trade = [...openTrades, ...closedTrades].find(t => t.trade_id === overviewTradeId);
        if (!trade) return null;
        return (
          <TradeOverviewSidecar
            trade={trade}
            details={details}
            onClose={() => setOverviewTradeId(null)}
          />
        );
      })()}
    </div>
  );
}

function CampaignReviewRow({ row: r, expanded, onToggleExpand, onContextMenu }: {
  row: TradeRow;
  expanded: boolean;
  onToggleExpand: () => void;
  onContextMenu: (e: React.MouseEvent<HTMLTableRowElement>) => void;
}) {
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

  const cats = r.lesson_category.split("|").map(s => s.trim()).filter(Boolean);

  // R-multiple color banding: ≥2R deep green, 0..2R green, -1..0 amber,
  // <-1R red. Same shape CR used so users don't have to relearn the
  // signal after the merge.
  const rColor = r.r_multiple == null
    ? "var(--ink-3)"
    : r.r_multiple >= 2 ? "#08a86b"
    : r.r_multiple >= 0 ? "#16a34a"
    : r.r_multiple >= -1 ? "#d97706"
    : "#e5484d";

  return (
    <tr onClick={onToggleExpand}
        onContextMenu={onContextMenu}
        className="cursor-pointer"
        style={{
          borderBottom: "1px solid var(--border)",
          background: expanded ? "var(--surface-2)" : "transparent",
        }}
        onMouseEnter={e => { if (!expanded) e.currentTarget.style.background = "var(--bg-2)"; }}
        onMouseLeave={e => { if (!expanded) e.currentTarget.style.background = "transparent"; }}>
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
      <td className="px-3 py-2 text-right font-semibold" style={{ fontFamily: mono }}>
        {r.b_shares > 0 ? fmtPnl(r.b_pnl) : <span style={{ color: "var(--ink-4)" }}>—</span>}
      </td>
      <td className="px-3 py-2 text-right" style={{ fontFamily: mono }}>{fmtPct(r.b_return_pct)}</td>
      <td className="px-3 py-2 text-right font-semibold" style={{ fontFamily: mono }}>
        {r.a_shares > 0 ? fmtPnl(r.a_pnl) : <span style={{ color: "var(--ink-4)" }}>—</span>}
      </td>
      <td className="px-3 py-2 text-right" style={{ fontFamily: mono }}>{fmtPct(r.a_return_pct)}</td>
      <td className="px-3 py-2 text-right font-bold" style={{ fontFamily: mono }}>{fmtPnl(r.total_pnl)}</td>
      <td className="px-3 py-2 text-right" style={{ fontFamily: mono }}>{fmtPct(r.total_return_pct)}</td>
      <td className="px-3 py-2 text-right font-semibold" style={{ fontFamily: mono, color: rColor }}>
        {r.r_multiple == null ? "—" : `${r.r_multiple >= 0 ? "+" : ""}${r.r_multiple.toFixed(2)}R`}
      </td>
      <td className="px-3 py-2 text-right" style={{ fontFamily: mono }}>
        {fmtExcursion(r.mae_pct, r.atr21_entry_pct, "adverse")}
      </td>
      <td className="px-3 py-2 text-right" style={{ fontFamily: mono }}>
        {fmtExcursion(r.mfe_pct, r.atr21_entry_pct, "favorable")}
      </td>
      <td className="px-3 py-2 text-[11px]" style={{ color: "var(--ink-3)" }}>{r.rule}</td>
      <td className="px-3 py-2 text-[11px]" style={{ color: "var(--ink-3)" }}>
        {r.sell_rule || <span style={{ color: "var(--ink-4)" }}>—</span>}
      </td>
      <td className="px-3 py-2">
        {cats.length > 0 ? (
          <div className="flex flex-wrap gap-1">
            {cats.slice(0, 3).map(c => {
              const cc = CAT_COLORS[c] || CAT_FALLBACK;
              return (
                <span key={c} className="text-[10px] font-semibold px-1.5 py-0.5 rounded-full"
                      style={{ background: cc.bg, color: cc.fg }}>
                  {c}
                </span>
              );
            })}
            {cats.length > 3 && (
              <span className="text-[10px]" style={{ color: "var(--ink-4)" }}>+{cats.length - 3}</span>
            )}
          </div>
        ) : (
          <span className="text-[11px] italic" style={{ color: "var(--ink-4)" }}>untagged</span>
        )}
      </td>
      <td className="px-3 py-2 text-right" style={{ color: "var(--ink-4)" }}>
        {expanded ? "▲" : "▼"}
      </td>
    </tr>
  );
}

// Lesson editor rendered inside the expanded row. Same pattern the
// old Campaign Review page used (category chips + note textarea that
// saves on blur). Kept local rather than extracting a shared module —
// this page is the only surface that surfaces lesson editing today.
function LessonEditor({ row, saving, onToggleCategory, onSaveNote }: {
  row: TradeRow;
  saving: boolean;
  onToggleCategory: (cat: string) => void;
  onSaveNote: (note: string) => void;
}) {
  const [note, setNote] = useState(row.lesson_note);
  // Re-sync when the parent's optimistic update flows in (e.g. user
  // toggles a category which triggers a lessons state update).
  useEffect(() => { setNote(row.lesson_note); }, [row.lesson_note]);
  const selected = new Set(row.lesson_category.split("|").map(s => s.trim()).filter(Boolean));

  return (
    <div className="flex flex-col gap-3" data-testid="lesson-editor">
      <div className="flex items-center justify-between">
        <div className="text-[12px] font-semibold flex items-center gap-2">
          <span>🎓</span>
          Lesson · <span style={{ color: "var(--ink-3)" }}>{row.ticker} — {row.trade_id}</span>
          {saving && <span className="text-[10px]" style={{ color: "var(--ink-4)" }}>saving…</span>}
        </div>
        <div className="text-[10px]" style={{ color: "var(--ink-4)" }}>
          Buy: <b style={{ color: "var(--ink-3)" }}>{row.rule || "—"}</b>
          {row.sell_rule && <> · Sell: <b style={{ color: "var(--ink-3)" }}>{row.sell_rule}</b></>}
        </div>
      </div>

      {/* Category chips — click to toggle. Same catalog as Log Sell. */}
      <div className="flex flex-wrap gap-1.5">
        {LESSON_CATEGORIES.map(cat => {
          const isSel = selected.has(cat);
          const cc = CAT_COLORS[cat] || CAT_FALLBACK;
          return (
            <button
              key={cat}
              type="button"
              onClick={e => { e.stopPropagation(); onToggleCategory(cat); }}
              className="text-[11px] font-semibold px-2.5 py-1 rounded-full cursor-pointer transition-all"
              style={{
                background: isSel ? cc.bg : "var(--surface)",
                color: isSel ? cc.fg : "var(--ink-3)",
                border: `1px solid ${isSel ? cc.bg : "var(--border)"}`,
              }}
            >
              {isSel ? "✓ " : ""}{cat}
            </button>
          );
        })}
      </div>

      {/* Note textarea — saves on blur. */}
      <textarea
        value={note}
        onChange={e => setNote(e.target.value)}
        onBlur={() => onSaveNote(note)}
        onClick={e => e.stopPropagation()}
        placeholder="What did you learn from this trade?"
        rows={3}
        className="w-full rounded-[10px] px-3 py-2 text-[12px] resize-vertical"
        style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)" }}
      />
    </div>
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
