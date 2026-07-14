"use client";

// Campaign Review — post-mortem for closed campaigns since 2026-01-01.
// Sits under "Deep Dive" and complements the Analytics → Trade Review tab
// (which is a "top winners / worst losers" analytic lens). This page is
// the operational grind-through: filter, grade, and tag lessons on every
// closed campaign. Reuses:
//   - trades_summary.grade for inline star grading
//   - trade_lessons (one row per trade_id) for lesson category + note
//   - LESSON_CATEGORIES shared catalog (same picker as Log Sell)
// R multiple is computed server-side from the B1 detail row.

import { Fragment, useState, useEffect, useMemo, useCallback } from "react";
import { usePathname } from "next/navigation";
import { api, getActivePortfolio, type CampaignReviewRow } from "@/lib/api";
import { formatCurrency } from "@/lib/format";
import { LESSON_CATEGORIES, CAT_COLORS, CAT_FALLBACK } from "@/lib/lesson-categories";
import { getGroupForHref } from "@/lib/nav";

const mono = "var(--font-jetbrains), monospace";

type SeriesKey = "all" | "original" | "add_on";
type InstrumentKey = "all" | "stocks" | "options";
type DateRangeKey = "ytd" | "month" | "week" | "custom";
type GradeKey = "all" | "unrated" | "1" | "2" | "3" | "4" | "5";
// Which date column gates all the date-range filters. "close" is the
// original / default behavior — "trades I closed this month". "open"
// flips the semantics to "trades I opened this month". A trade that
// runs 3 months lands in different weeks depending on which basis you
// pick, so the toggle is genuinely load-bearing for periodic reviews.
type DateBasisKey = "open" | "close";
// P&L filter. Direction (winners/losers) + percentile buckets in a
// single dropdown. Percentile options ("top_10" etc.) are relative to
// the currently-otherwise-filtered set, not the global population —
// so combining Ticker=NVDA + P&L=Top 10% gives you the top 10% of your
// NVDA trades, which is what you'd intuitively expect.
type PLKey = "all" | "winners" | "losers" | "top_10" | "top_25" | "bottom_10" | "bottom_25";

// Sortable table columns. "lesson" and the expand caret intentionally
// left out — sorting by chip content isn't meaningful.
type SortKey = "ticker" | "open_date" | "closed_date" | "realized_pl" | "return_pct" | "r_multiple" | "grade";
type SortDir = "asc" | "desc";

interface Filters {
  q: string;
  series: SeriesKey;
  tickers: string[];  // multi-select; empty = no ticker filter
  rule: string;
  instrument: InstrumentKey;
  lesson: string;  // "all" | "none" | category name
  dateRange: DateRangeKey;
  dateBasis: DateBasisKey;  // which date column drives the range
  from: string;
  to: string;
  grade: GradeKey;
  pl: PLKey;
}

const DEFAULT_FILTERS: Filters = {
  q: "",
  series: "all",
  tickers: [],
  rule: "all",
  instrument: "all",
  lesson: "all",
  dateRange: "ytd",
  dateBasis: "close",
  from: "",
  to: "",
  grade: "all",
  pl: "all",
};

// True when any filter diverges from the defaults. Used to enable /
// dim the Reset button. Sort state is intentionally excluded — sort
// is a viewing preference, not a filter, and reset shouldn't touch it.
function hasActiveFilters(f: Filters): boolean {
  return f.q !== ""
    || f.series !== "all"
    || f.tickers.length > 0
    || f.rule !== "all"
    || f.instrument !== "all"
    || f.lesson !== "all"
    || f.dateRange !== "ytd"
    || f.dateBasis !== "close"
    || f.from !== ""
    || f.to !== ""
    || f.grade !== "all"
    || f.pl !== "all";
}

// Collapse an OCC option ticker ("ALAB 260717 $195C") to its underlying
// ("ALAB"). Equity tickers pass through unchanged. Used to dedupe the
// Ticker filter list and to match option rows when the user picks the
// underlying symbol.
function underlyingSymbol(ticker: string): string {
  const t = (ticker || "").trim();
  if (!t) return t;
  if (/^\S+\s+\d{6}\s+\$[0-9.]+(C|P)$/.test(t)) return t.split(/\s+/)[0];
  return t;
}

// Filter → date-range predicate. YTD = "everything since 2026-01-01"
// (server-side base cut for the "closed" case; for the "open" basis,
// YTD narrows client-side to trades opened in 2026 which strips out
// the 2025 opens that closed in 2026 — different intent, different
// result). Month = current calendar month. Week = Monday of current
// week. Custom uses the from/to inputs.
function dateFilterPasses(row: CampaignReviewRow, f: Filters): boolean {
  const raw = f.dateBasis === "open" ? row.open_date : row.closed_date;
  const d = raw ? new Date(raw) : null;
  if (!d || isNaN(d.getTime())) return f.dateRange === "ytd" || f.dateRange === "custom";
  const now = new Date();
  if (f.dateRange === "ytd") {
    if (f.dateBasis === "open") {
      // 2026 cut is baked in at the server for closed-basis. For open-
      // basis, we need to enforce the same year gate client-side so
      // "YTD by open date" doesn't spuriously include a 2025 open that
      // happened to close in 2026.
      return d.getFullYear() >= 2026;
    }
    return true;
  }
  if (f.dateRange === "month") {
    return d.getFullYear() === now.getFullYear() && d.getMonth() === now.getMonth();
  }
  if (f.dateRange === "week") {
    // Monday-anchored week. getDay(): 0=Sun..6=Sat. Shift so Mon=0.
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

// Inline star clicker. Clicking a star sets the grade to that number.
// Clicking the currently-set top star clears to null. Optimistic update.
function GradeStars({ value, onChange }: {
  value: number | null;
  onChange: (next: number | null) => void;
}) {
  const [hover, setHover] = useState<number | null>(null);
  return (
    <div className="inline-flex items-center gap-0.5" onMouseLeave={() => setHover(null)}>
      {[1, 2, 3, 4, 5].map(n => {
        const filled = hover != null ? n <= hover : (value != null && n <= value);
        return (
          <button key={n} type="button"
                  onMouseEnter={() => setHover(n)}
                  onClick={e => {
                    e.stopPropagation();
                    // Clicking the currently-set top star clears the grade
                    onChange(value === n ? null : n);
                  }}
                  className="text-[14px] leading-none px-0.5 py-0.5 cursor-pointer transition-transform hover:scale-125"
                  style={{ color: filled ? "#f59f00" : "var(--ink-4)", opacity: filled ? 1 : 0.5, background: "transparent", border: "none" }}
                  aria-label={`Grade ${n}`}>
            {filled ? "★" : "☆"}
          </button>
        );
      })}
    </div>
  );
}

// Duplicated locally rather than imported from campaign-detail (avoids
// review→detail coupling). Two tiny controls; cheaper than the shared
// extraction refactor.
function SegmentedControl<T extends string>({ label, value, onChange, options }: {
  label: string;
  value: T;
  onChange: (v: T) => void;
  options: { v: T; l: string }[];
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[9px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>{label}</span>
      <div className="flex p-0.5 rounded-[10px] gap-0.5 h-[34px]" style={{ background: "var(--bg-2)", border: "1px solid var(--border)" }}>
        {options.map(o => (
          <button key={o.v} type="button" onClick={() => onChange(o.v)}
                  className="px-3 rounded-[8px] text-[11px] font-medium transition-all cursor-pointer"
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

// Sortable table header. Click to toggle asc/desc; clicking a different
// column switches to that column with a sensible default direction
// (numeric/date desc first, text asc first — handled by toggleSort).
function SortHeader({ label, sortKey, activeKey, dir, onToggle, align }: {
  label: string;
  sortKey: SortKey;
  activeKey: SortKey;
  dir: SortDir;
  onToggle: (k: SortKey) => void;
  align: "left" | "right";
}) {
  const isActive = activeKey === sortKey;
  const arrow = isActive ? (dir === "asc" ? "▲" : "▼") : "";
  return (
    <th className="px-3 py-2.5 font-semibold uppercase tracking-[0.05em] text-[10px] cursor-pointer select-none"
        style={{ textAlign: align }}
        onClick={() => onToggle(sortKey)}>
      <span className="inline-flex items-center gap-1"
            style={{ color: isActive ? "var(--ink)" : "var(--ink-4)" }}>
        {label}
        <span className="text-[8px] opacity-70" style={{ width: 8, display: "inline-block" }}>{arrow}</span>
      </span>
    </th>
  );
}

export function CampaignReview() {
  const pathname = usePathname();
  const navColor = getGroupForHref(pathname)?.color || "#0d6efd";

  const [rows, setRows] = useState<CampaignReviewRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<Filters>(DEFAULT_FILTERS);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [saving, setSaving] = useState<string | null>(null);
  // Sort state — default matches the server return order (closed_date desc).
  const [sortKey, setSortKey] = useState<SortKey>("closed_date");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  // Ticker multi-select typeahead state
  const [tickerQuery, setTickerQuery] = useState("");
  const [tickerDropdownOpen, setTickerDropdownOpen] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.campaignsReview(getActivePortfolio(), "2026-01-01");
      setRows(Array.isArray(data) ? data : []);
    } catch (e: any) {
      setError(e?.message || "Failed to load campaigns");
      setRows([]);
    }
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  // Derived filter option lists. Options tickers arrive as OCC strings
  // ("ALAB 260717 $195C") — collapse them to the underlying so the
  // dropdown shows "ALAB" once and picking it matches both the equity
  // rows and every option row on that underlying.
  const tickerOptions = useMemo(() => {
    const s = new Set<string>();
    rows.forEach(r => { if (r.ticker) s.add(underlyingSymbol(r.ticker)); });
    return Array.from(s).sort();
  }, [rows]);

  const ruleOptions = useMemo(() => {
    const s = new Set<string>();
    rows.forEach(r => { if (r.rule) s.add(r.rule); });
    return Array.from(s).sort();
  }, [rows]);

  const filtered = useMemo(() => {
    // Pass 1: apply all non-P&L filters. P&L direction (winners/losers)
    // could be baked in here, but percentile buckets need the
    // otherwise-filtered set as their reference population, so we do
    // both P&L variants in a second pass for consistency.
    const pass1 = rows.filter(r => {
      // Search
      if (filters.q) {
        const q = filters.q.toLowerCase();
        const hay = `${r.ticker} ${r.trade_id} ${r.rule} ${r.sell_rule} ${r.lesson_note}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      // Series (original = no add-ons; add_on = had add-ons)
      if (filters.series === "original" && r.has_add_ons) return false;
      if (filters.series === "add_on" && !r.has_add_ons) return false;
      // Ticker (multi-select; empty array = no filter). Match on the
      // underlying so picking "ALAB" catches both the equity row and
      // any "ALAB 260717 $195C" option rows.
      if (filters.tickers.length > 0 && !filters.tickers.includes(underlyingSymbol(r.ticker))) return false;
      // Rule
      if (filters.rule !== "all" && r.rule !== filters.rule) return false;
      // Instrument
      if (filters.instrument !== "all") {
        const isOpt = String(r.instrument_type).toUpperCase() === "OPTION"
          || /^\S+\s+\d{6}\s+\$[0-9.]+(C|P)$/.test(r.ticker || "");
        if (filters.instrument === "options" && !isOpt) return false;
        if (filters.instrument === "stocks" && isOpt) return false;
      }
      // Lesson
      if (filters.lesson === "none") {
        if (r.lesson_category && r.lesson_category.trim() !== "") return false;
      } else if (filters.lesson !== "all") {
        const cats = (r.lesson_category || "").split("|").map(s => s.trim()).filter(Boolean);
        if (!cats.includes(filters.lesson)) return false;
      }
      // Grade
      if (filters.grade === "unrated") {
        if (r.grade != null) return false;
      } else if (filters.grade !== "all") {
        if (r.grade !== parseInt(filters.grade, 10)) return false;
      }
      // Date range
      if (!dateFilterPasses(r, filters)) return false;
      return true;
    });

    // Pass 2: P&L direction + percentile. Percentile is relative to
    // pass1 — so "Top 10%" with Ticker=NVDA means top 10% of NVDA
    // trades, not top 10% of the global population.
    if (filters.pl === "all") return pass1;
    if (filters.pl === "winners") return pass1.filter(r => (r.realized_pl || 0) > 0);
    if (filters.pl === "losers") return pass1.filter(r => (r.realized_pl || 0) < 0);
    // Percentile buckets. Ceil so "Top 10%" of 7 trades still returns
    // the single best trade rather than 0 rows.
    const pct = filters.pl === "top_10" || filters.pl === "bottom_10" ? 0.10 : 0.25;
    const isTop = filters.pl === "top_10" || filters.pl === "top_25";
    const n = pass1.length;
    if (n === 0) return pass1;
    const takeCount = Math.max(1, Math.ceil(n * pct));
    const sorted = [...pass1].sort((a, b) =>
      isTop ? (b.realized_pl || 0) - (a.realized_pl || 0) : (a.realized_pl || 0) - (b.realized_pl || 0),
    );
    const keep = new Set(sorted.slice(0, takeCount).map(r => r.trade_id));
    return pass1.filter(r => keep.has(r.trade_id));
  }, [rows, filters]);

  // Summary stats over the filtered set (calculated before sort — order
  // doesn't affect aggregates).
  //
  // Definitions used:
  //   winner   = realized_pl > 0
  //   loser    = realized_pl < 0
  //   scratches (== 0) are counted in n but excluded from both W and L
  //   win rate = winCount / (winCount + lossCount)
  //   expectancy = avgWin × winRate − avgLoss × lossRate
  //     (avgLoss is absolute value; sign is applied in the formula so a
  //     positive expectancy means +EV per trade)
  //   top R    = avg R for grades 4-5 (four or five stars)
  //   bot R    = avg R for grades 1-2 (one or two stars)
  //     grade 3 is middle/undecided, excluded from the comparison so the
  //     top/bot signal isn't diluted
  const stats = useMemo(() => {
    const n = filtered.length;
    let totalPl = 0, sumR = 0, rCount = 0, sumGrade = 0, gradeCount = 0;
    let winCount = 0, lossCount = 0, sumWin = 0, sumLoss = 0;
    let topRSum = 0, topRCount = 0, botRSum = 0, botRCount = 0;
    for (const r of filtered) {
      const pl = r.realized_pl || 0;
      totalPl += pl;
      if (pl > 0) { winCount += 1; sumWin += pl; }
      else if (pl < 0) { lossCount += 1; sumLoss += Math.abs(pl); }
      if (r.r_multiple != null) {
        sumR += r.r_multiple;
        rCount += 1;
        if (r.grade != null) {
          if (r.grade >= 4) { topRSum += r.r_multiple; topRCount += 1; }
          else if (r.grade <= 2) { botRSum += r.r_multiple; botRCount += 1; }
        }
      }
      if (r.grade != null) { sumGrade += r.grade; gradeCount += 1; }
    }
    const decidedTrades = winCount + lossCount;
    const winRate = decidedTrades > 0 ? winCount / decidedTrades : null;
    const avgWin = winCount > 0 ? sumWin / winCount : 0;
    const avgLoss = lossCount > 0 ? sumLoss / lossCount : 0;
    const expectancy = winRate != null
      ? avgWin * winRate - avgLoss * (1 - winRate)
      : null;
    return {
      n,
      totalPl,
      avgR: rCount > 0 ? sumR / rCount : null,
      avgGrade: gradeCount > 0 ? sumGrade / gradeCount : null,
      unratedCount: filtered.filter(r => r.grade == null).length,
      winCount, lossCount, winRate, expectancy,
      topR: topRCount > 0 ? topRSum / topRCount : null,
      botR: botRCount > 0 ? botRSum / botRCount : null,
    };
  }, [filtered]);

  // Sort the filtered rows. Nulls sort to the bottom regardless of dir
  // so an unrated row doesn't jump to the top when sorting by grade asc.
  const sorted = useMemo(() => {
    const out = [...filtered];
    const key = sortKey;
    const mult = sortDir === "asc" ? 1 : -1;
    out.sort((a, b) => {
      const av = (a as any)[key];
      const bv = (b as any)[key];
      const aNull = av == null || av === "";
      const bNull = bv == null || bv === "";
      if (aNull && bNull) return 0;
      if (aNull) return 1;   // nulls always last
      if (bNull) return -1;
      if (typeof av === "number" && typeof bv === "number") return (av - bv) * mult;
      // Dates are ISO strings — string compare matches chronological order
      return String(av).localeCompare(String(bv)) * mult;
    });
    return out;
  }, [filtered, sortKey, sortDir]);

  const toggleSort = useCallback((key: SortKey) => {
    if (sortKey === key) {
      setSortDir(d => d === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      // Default direction per column type: dates/numbers desc first
      // (latest/largest usually more interesting), ticker asc first.
      setSortDir(key === "ticker" ? "asc" : "desc");
    }
  }, [sortKey]);

  // Optimistic grade update
  const setGrade = useCallback(async (trade_id: string, grade: number | null) => {
    setRows(prev => prev.map(r => r.trade_id === trade_id ? { ...r, grade } : r));
    setSaving(trade_id);
    try {
      await api.setTradeGrade({ portfolio: getActivePortfolio(), trade_id, grade });
    } catch {
      // Reload to reconcile if the save failed
      load();
    }
    setSaving(null);
  }, [load]);

  // Toggle a category in the lesson_category pipe-separated field
  const toggleCategory = useCallback(async (trade_id: string, cat: string) => {
    const row = rows.find(r => r.trade_id === trade_id);
    if (!row) return;
    const current = (row.lesson_category || "").split("|").map(s => s.trim()).filter(Boolean);
    const next = current.includes(cat)
      ? current.filter(c => c !== cat)
      : [...current, cat];
    const nextStr = next.join("|");
    setRows(prev => prev.map(r => r.trade_id === trade_id ? { ...r, lesson_category: nextStr } : r));
    setSaving(trade_id);
    try {
      await api.saveTradeLessons({
        portfolio: getActivePortfolio(),
        trade_id,
        note: row.lesson_note || "",
        category: nextStr,
      });
    } catch {
      load();
    }
    setSaving(null);
  }, [rows, load]);

  // Save note on blur (debounce is overkill for a review page)
  const saveNote = useCallback(async (trade_id: string, note: string) => {
    const row = rows.find(r => r.trade_id === trade_id);
    if (!row) return;
    if (note === row.lesson_note) return;
    setRows(prev => prev.map(r => r.trade_id === trade_id ? { ...r, lesson_note: note } : r));
    setSaving(trade_id);
    try {
      await api.saveTradeLessons({
        portfolio: getActivePortfolio(),
        trade_id,
        note,
        category: row.lesson_category || "",
      });
    } catch {
      load();
    }
    setSaving(null);
  }, [rows, load]);

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      {/* Header */}
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          Campaign <em className="italic" style={{ color: navColor }}>Review</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
          Post-mortem on every closed campaign since 2026-01-01. Grade the trade, tag the lesson.
        </div>
      </div>

      {/* Filter bar */}
      <div className="rounded-[14px] overflow-hidden mb-4" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <div className="px-[18px] py-[14px] flex flex-wrap items-end gap-[12px_14px]"
             style={{ background: "var(--bg-2)", borderBottom: "1px solid var(--border)" }}>
          {/* Ticker multi-select. Same chip + typeahead pattern as Trade
              Journal — chips for what's picked, typeahead for adding
              more, backspace on empty removes the last chip. Placed
              first because ticker is the most common way into the page. */}
          <div className="flex flex-col gap-1" style={{ minWidth: 200 }}>
            <span className="text-[9px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>Ticker</span>
            <div className="flex items-center gap-1.5 flex-wrap min-h-[34px]">
              {filters.tickers.map(t => (
                <span key={t} className="inline-flex items-center gap-1 h-[26px] px-2 rounded-[8px] text-[11px] font-semibold"
                      style={{ background: `color-mix(in oklab, ${navColor} 10%, transparent)`, color: navColor, border: `1px solid color-mix(in oklab, ${navColor} 30%, var(--border))` }}>
                  {t}
                  <button type="button"
                          onClick={() => setFilters(f => ({ ...f, tickers: f.tickers.filter(x => x !== t) }))}
                          className="ml-0.5 opacity-60 hover:opacity-100 cursor-pointer" style={{ lineHeight: 1 }}>×</button>
                </span>
              ))}
              <div className="relative">
                <input type="text" value={tickerQuery}
                       placeholder={filters.tickers.length > 0 ? "Add ticker…" : "Search tickers…"}
                       onChange={e => { setTickerQuery(e.target.value.toUpperCase()); setTickerDropdownOpen(true); }}
                       onKeyDown={e => {
                         if (e.key === "Enter" && tickerQuery) {
                           const match = tickerOptions.find(t => t.toUpperCase() === tickerQuery.trim());
                           const ticker = match ?? tickerQuery.trim().toUpperCase();
                           if (ticker && !filters.tickers.includes(ticker)) {
                             setFilters(f => ({ ...f, tickers: [...f.tickers, ticker] }));
                           }
                           setTickerQuery(""); setTickerDropdownOpen(false);
                         }
                         if (e.key === "Backspace" && !tickerQuery && filters.tickers.length > 0) {
                           setFilters(f => ({ ...f, tickers: f.tickers.slice(0, -1) }));
                         }
                       }}
                       onFocus={() => setTickerDropdownOpen(true)}
                       onBlur={() => setTimeout(() => setTickerDropdownOpen(false), 150)}
                       className="h-[34px] px-3 rounded-[10px] text-[12px] w-[140px]"
                       style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }} />
                {tickerDropdownOpen && (() => {
                  const available = tickerOptions
                    .filter(t => !filters.tickers.includes(t))
                    .filter(t => !tickerQuery || t.toUpperCase().includes(tickerQuery.trim()));
                  return available.length > 0 ? (
                    <div className="absolute z-50 mt-1 w-[180px] rounded-[10px] overflow-hidden shadow-lg"
                         style={{ background: "var(--surface)", border: "1px solid var(--border)", maxHeight: 200 }}>
                      <div className="overflow-y-auto" style={{ maxHeight: 200 }}>
                        {available.slice(0, 50).map(t => (
                          <button key={t} type="button"
                                  onMouseDown={e => {
                                    e.preventDefault();
                                    setFilters(f => ({ ...f, tickers: [...f.tickers, t] }));
                                    setTickerQuery(""); setTickerDropdownOpen(false);
                                  }}
                                  className="w-full text-left px-3 py-1.5 text-[12px] transition-colors"
                                  style={{ fontFamily: mono }}
                                  onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-2)")}
                                  onMouseLeave={e => (e.currentTarget.style.background = "transparent")}>
                            {t}
                          </button>
                        ))}
                      </div>
                    </div>
                  ) : null;
                })()}
              </div>
            </div>
          </div>

          {/* Search */}
          <div className="flex flex-col gap-1" style={{ flex: "1 1 220px", minWidth: 200 }}>
            <span className="text-[9px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>Search</span>
            <div className="relative">
              <input type="text" value={filters.q}
                     onChange={e => setFilters(f => ({ ...f, q: e.target.value }))}
                     placeholder="Ticker, trade ID, rule or note…"
                     className="w-full h-[34px] pl-9 pr-8 rounded-[10px] text-[12px]"
                     style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)" }} />
              <span className="absolute left-3 top-1/2 -translate-y-1/2 text-[12px]" style={{ color: "var(--ink-4)" }}>⌕</span>
              {filters.q && (
                <button type="button" onClick={() => setFilters(f => ({ ...f, q: "" }))}
                        className="absolute right-2 top-1/2 -translate-y-1/2 px-1 text-[12px] cursor-pointer"
                        style={{ color: "var(--ink-4)" }}>✕</button>
              )}
            </div>
          </div>

          <SegmentedControl<SeriesKey>
            label="Series"
            value={filters.series}
            onChange={v => setFilters(f => ({ ...f, series: v }))}
            options={[{ v: "all", l: "All" }, { v: "original", l: "B · Original" }, { v: "add_on", l: "A · Add-on" }]}
          />

          <SegmentedControl<InstrumentKey>
            label="Instrument"
            value={filters.instrument}
            onChange={v => setFilters(f => ({ ...f, instrument: v }))}
            options={[{ v: "all", l: "All" }, { v: "stocks", l: "Stocks" }, { v: "options", l: "Options" }]}
          />

          <FilterSelect
            label="Rule"
            value={filters.rule}
            onChange={v => setFilters(f => ({ ...f, rule: v }))}
            options={[{ v: "all", l: "All rules" }, ...ruleOptions.map(r => ({ v: r, l: r }))]}
          />

          <FilterSelect
            label="Lesson"
            value={filters.lesson}
            onChange={v => setFilters(f => ({ ...f, lesson: v }))}
            options={[
              { v: "all", l: "All lessons" },
              { v: "none", l: "— untagged" },
              ...LESSON_CATEGORIES.map(c => ({ v: c, l: c })),
            ]}
          />

          <FilterSelect
            label="Grade"
            value={filters.grade}
            onChange={v => setFilters(f => ({ ...f, grade: v as GradeKey }))}
            options={[
              { v: "all", l: "All grades" },
              { v: "unrated", l: "Unrated" },
              { v: "1", l: "★" },
              { v: "2", l: "★★" },
              { v: "3", l: "★★★" },
              { v: "4", l: "★★★★" },
              { v: "5", l: "★★★★★" },
            ]}
          />

          <FilterSelect
            label="P&L"
            value={filters.pl}
            onChange={v => setFilters(f => ({ ...f, pl: v as PLKey }))}
            options={[
              { v: "all", l: "All P&L" },
              { v: "winners", l: "Winners only" },
              { v: "losers", l: "Losers only" },
              { v: "top_10", l: "Top 10% by P&L" },
              { v: "top_25", l: "Top 25% by P&L" },
              { v: "bottom_10", l: "Bottom 10% by P&L" },
              { v: "bottom_25", l: "Bottom 25% by P&L" },
            ]}
          />

          <SegmentedControl<DateRangeKey>
            label="Date"
            value={filters.dateRange}
            onChange={v => setFilters(f => ({ ...f, dateRange: v }))}
            options={[
              { v: "week", l: "Week" },
              { v: "month", l: "Month" },
              { v: "ytd", l: "YTD" },
              { v: "custom", l: "Custom" },
            ]}
          />

          {/* Which date column gates the range above. Applies to all
              presets (Week / Month / YTD / Custom) — an asymmetric
              behavior where only Custom respected the basis would be
              confusing when flipping between presets. */}
          <SegmentedControl<DateBasisKey>
            label="Basis"
            value={filters.dateBasis}
            onChange={v => setFilters(f => ({ ...f, dateBasis: v }))}
            options={[
              { v: "close", l: "Close" },
              { v: "open", l: "Open" },
            ]}
          />

          {filters.dateRange === "custom" && (
            <div className="flex flex-col gap-1">
              <span className="text-[9px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>Range</span>
              <div className="flex items-center gap-2">
                <input type="date" value={filters.from} onChange={e => setFilters(f => ({ ...f, from: e.target.value }))}
                       className="h-[34px] px-2 rounded-[10px] text-[12px]"
                       style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }} />
                <span className="text-[12px]" style={{ color: "var(--ink-4)" }}>–</span>
                <input type="date" value={filters.to} onChange={e => setFilters(f => ({ ...f, to: e.target.value }))}
                       className="h-[34px] px-2 rounded-[10px] text-[12px]"
                       style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }} />
              </div>
            </div>
          )}

          {/* Reset filters. Sort is preserved — that's a viewing
              preference, not a filter, and re-picking a sort after every
              reset would be annoying. Button dims to a subtle "hint"
              state when no filters diverge from defaults, so it's
              visible for discoverability without shouting. */}
          {(() => {
            const active = hasActiveFilters(filters);
            return (
              <div className="flex flex-col gap-1 ml-auto">
                <span className="text-[9px] font-bold uppercase tracking-[0.08em]" style={{ color: "transparent" }}>Reset</span>
                <button type="button"
                        onClick={() => { setFilters(DEFAULT_FILTERS); setTickerQuery(""); setTickerDropdownOpen(false); }}
                        disabled={!active}
                        className="h-[34px] px-3 rounded-[10px] text-[11px] font-semibold transition-all cursor-pointer disabled:cursor-not-allowed"
                        style={{
                          background: active ? "color-mix(in oklab, #e5484d 8%, var(--surface))" : "var(--bg-2)",
                          color: active ? "#dc2626" : "var(--ink-4)",
                          border: `1px solid ${active ? "color-mix(in oklab, #e5484d 25%, var(--border))" : "var(--border)"}`,
                          opacity: active ? 1 : 0.6,
                        }}>
                  ↺ Reset filters
                </button>
              </div>
            );
          })()}
        </div>

        {/* Summary strip */}
        <div className="px-[18px] py-[10px] flex flex-wrap items-center gap-[16px] text-[12px]"
             style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)", color: "var(--ink-3)" }}>
          <span><b style={{ color: "var(--ink)" }}>{stats.n}</b> campaign{stats.n === 1 ? "" : "s"}</span>
          <span>
            Win rate{" "}
            <b style={{ color: "var(--ink)", fontFamily: mono }}>
              {stats.winCount}W / {stats.lossCount}L
            </b>
            {stats.winRate != null && (
              <> · <b style={{ color: stats.winRate >= 0.5 ? "#08a86b" : "#d97706", fontFamily: mono }}>
                {(stats.winRate * 100).toFixed(0)}%
              </b></>
            )}
          </span>
          <span>P&L <b style={{ color: stats.totalPl >= 0 ? "#08a86b" : "#e5484d", fontFamily: mono }}>{formatCurrency(stats.totalPl, { decimals: 0 })}</b></span>
          <span title="avg win × win rate − avg loss × loss rate">
            Expectancy{" "}
            <b style={{ color: stats.expectancy == null ? "var(--ink-3)" : stats.expectancy >= 0 ? "#08a86b" : "#e5484d", fontFamily: mono }}>
              {stats.expectancy == null ? "—" : `${formatCurrency(stats.expectancy, { decimals: 0 })}/trade`}
            </b>
          </span>
          <span>Avg R <b style={{ color: "var(--ink)", fontFamily: mono }}>{stats.avgR == null ? "—" : `${stats.avgR.toFixed(2)}R`}</b></span>
          <span title="Average R multiple for trades graded 4-5★ vs 1-2★ (grade 3 excluded so the discriminator isn't diluted)">
            R by grade{" "}
            <b style={{ color: "var(--ink)", fontFamily: mono }}>
              {stats.topR == null ? "—" : `${stats.topR >= 0 ? "+" : ""}${stats.topR.toFixed(2)}R`}
            </b>
            <span style={{ color: "var(--ink-4)" }}> @4-5★ · </span>
            <b style={{ color: "var(--ink)", fontFamily: mono }}>
              {stats.botR == null ? "—" : `${stats.botR >= 0 ? "+" : ""}${stats.botR.toFixed(2)}R`}
            </b>
            <span style={{ color: "var(--ink-4)" }}> @1-2★</span>
          </span>
          <span>Avg Grade <b style={{ color: "var(--ink)", fontFamily: mono }}>{stats.avgGrade == null ? "—" : stats.avgGrade.toFixed(2)}</b></span>
          <span>Unrated <b style={{ color: stats.unratedCount > 0 ? "#d97706" : "var(--ink-3)" }}>{stats.unratedCount}</b></span>
        </div>

        {/* Table */}
        {loading ? (
          <div className="px-[18px] py-8 text-center text-[12px]" style={{ color: "var(--ink-3)" }}>Loading closed campaigns…</div>
        ) : error ? (
          <div className="px-[18px] py-8 text-center text-[12px]" style={{ color: "#e5484d" }}>Failed: {error}</div>
        ) : filtered.length === 0 ? (
          <div className="px-[18px] py-8 text-center text-[12px]" style={{ color: "var(--ink-3)" }}>
            {rows.length === 0 ? "No closed campaigns since 2026-01-01." : "No campaigns match the current filters."}
          </div>
        ) : (
          <div className="overflow-auto">
            <table className="w-full text-[12px]" style={{ borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ background: "var(--surface-2)", color: "var(--ink-4)" }}>
                  <SortHeader label="Ticker"  sortKey="ticker"       activeKey={sortKey} dir={sortDir} onToggle={toggleSort} align="left" />
                  <SortHeader label="Open"    sortKey="open_date"    activeKey={sortKey} dir={sortDir} onToggle={toggleSort} align="left" />
                  <SortHeader label="Close"   sortKey="closed_date"  activeKey={sortKey} dir={sortDir} onToggle={toggleSort} align="left" />
                  <SortHeader label="P&L"     sortKey="realized_pl"  activeKey={sortKey} dir={sortDir} onToggle={toggleSort} align="right" />
                  <SortHeader label="Return"  sortKey="return_pct"   activeKey={sortKey} dir={sortDir} onToggle={toggleSort} align="right" />
                  <SortHeader label="R"       sortKey="r_multiple"   activeKey={sortKey} dir={sortDir} onToggle={toggleSort} align="right" />
                  <SortHeader label="Grade"   sortKey="grade"        activeKey={sortKey} dir={sortDir} onToggle={toggleSort} align="left" />
                  <th className="text-left px-3 py-2.5 font-semibold uppercase tracking-[0.05em] text-[10px]">Lesson</th>
                  <th className="px-3 py-2.5" />
                </tr>
              </thead>
              <tbody>
                {sorted.map(r => {
                  const cats = (r.lesson_category || "").split("|").map(s => s.trim()).filter(Boolean);
                  const isOpen = expandedId === r.trade_id;
                  const isOpt = String(r.instrument_type).toUpperCase() === "OPTION"
                    || /^\S+\s+\d{6}\s+\$[0-9.]+(C|P)$/.test(r.ticker || "");
                  const shortOpen = r.open_date ? r.open_date.slice(0, 10) : "—";
                  const shortClose = r.closed_date ? r.closed_date.slice(0, 10) : "—";
                  const rColor = r.r_multiple == null ? "var(--ink-3)"
                    : r.r_multiple >= 2 ? "#08a86b"
                    : r.r_multiple >= 0 ? "#16a34a"
                    : r.r_multiple >= -1 ? "#d97706"
                    : "#e5484d";
                  return (
                    <Fragment key={r.trade_id}>
                      <tr onClick={() => setExpandedId(isOpen ? null : r.trade_id)}
                          className="cursor-pointer transition-colors hover:brightness-95"
                          style={{ borderBottom: "1px solid var(--border)", background: isOpen ? "var(--surface-2)" : "transparent" }}>
                        <td className="px-3 py-2.5">
                          <div className="flex items-center gap-1.5">
                            <span className="font-semibold" style={{ color: "var(--ink)" }}>{r.ticker}</span>
                            {isOpt && (
                              <span className="text-[8px] font-bold px-1 py-0.5 rounded"
                                    style={{ background: "color-mix(in oklab, #6d28d9 12%, var(--surface))", color: "#6d28d9" }}>
                                OPT
                              </span>
                            )}
                            {r.has_add_ons && (
                              <span className="text-[8px] font-semibold px-1 py-0.5 rounded"
                                    style={{ background: "var(--bg-2)", color: "var(--ink-4)" }} title="Had add-ons">
                                +A
                              </span>
                            )}
                          </div>
                          <div className="text-[10px]" style={{ color: "var(--ink-4)", fontFamily: mono }}>{r.trade_id}</div>
                        </td>
                        <td className="px-3 py-2.5" style={{ fontFamily: mono, color: "var(--ink-3)" }}>
                          {shortOpen}
                        </td>
                        <td className="px-3 py-2.5" style={{ fontFamily: mono, color: "var(--ink-3)" }}>
                          {shortClose}
                        </td>
                        <td className="px-3 py-2.5 text-right font-semibold"
                            style={{ fontFamily: mono, color: r.realized_pl >= 0 ? "#08a86b" : "#e5484d" }}>
                          {formatCurrency(r.realized_pl, { decimals: 0 })}
                        </td>
                        <td className="px-3 py-2.5 text-right"
                            style={{ fontFamily: mono, color: r.return_pct >= 0 ? "#08a86b" : "#e5484d" }}>
                          {(r.return_pct || 0).toFixed(1)}%
                        </td>
                        <td className="px-3 py-2.5 text-right font-semibold"
                            style={{ fontFamily: mono, color: rColor }}>
                          {r.r_multiple == null ? "—" : `${r.r_multiple >= 0 ? "+" : ""}${r.r_multiple.toFixed(2)}R`}
                        </td>
                        <td className="px-3 py-2.5" onClick={e => e.stopPropagation()}>
                          <GradeStars value={r.grade} onChange={next => setGrade(r.trade_id, next)} />
                        </td>
                        <td className="px-3 py-2.5">
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
                              {cats.length > 3 && <span className="text-[10px]" style={{ color: "var(--ink-4)" }}>+{cats.length - 3}</span>}
                            </div>
                          ) : (
                            <span className="text-[11px] italic" style={{ color: "var(--ink-4)" }}>untagged</span>
                          )}
                        </td>
                        <td className="px-3 py-2.5 text-right" style={{ color: "var(--ink-4)" }}>
                          {isOpen ? "▲" : "▼"}
                        </td>
                      </tr>
                      {isOpen && (
                        <tr style={{ background: "var(--bg-2)" }}>
                          <td colSpan={9} className="px-4 py-4">
                            <ExcursionPanel row={r} />
                            <LessonEditor
                              row={r}
                              savingId={saving}
                              onToggleCategory={cat => toggleCategory(r.trade_id, cat)}
                              onSaveNote={note => saveNote(r.trade_id, note)}
                            />
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

// Lesson editor for the expanded row. Same picker UX as Log Sell's Exit
// Lesson block (LESSON_CATEGORIES chips, click to toggle) + note textarea
// that saves on blur.
function LessonEditor({ row, savingId, onToggleCategory, onSaveNote }: {
  row: CampaignReviewRow;
  savingId: string | null;
  onToggleCategory: (cat: string) => void;
  onSaveNote: (note: string) => void;
}) {
  const [note, setNote] = useState(row.lesson_note || "");
  useEffect(() => { setNote(row.lesson_note || ""); }, [row.lesson_note]);
  const selected = new Set((row.lesson_category || "").split("|").map(s => s.trim()).filter(Boolean));
  const isSaving = savingId === row.trade_id;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div className="text-[12px] font-semibold flex items-center gap-2">
          <span>🎓</span> Lesson · <span style={{ color: "var(--ink-3)" }}>{row.ticker} — {row.trade_id}</span>
          {isSaving && <span className="text-[10px]" style={{ color: "var(--ink-4)" }}>saving…</span>}
        </div>
        <div className="text-[10px]" style={{ color: "var(--ink-4)" }}>
          Rule: <b style={{ color: "var(--ink-3)" }}>{row.rule || "—"}</b>
          {row.sell_rule && <> · Exit: <b style={{ color: "var(--ink-3)" }}>{row.sell_rule}</b></>}
        </div>
      </div>

      {/* Category chips */}
      <div className="flex flex-wrap gap-1.5">
        {LESSON_CATEGORIES.map(cat => {
          const isSel = selected.has(cat);
          const cc = CAT_COLORS[cat] || CAT_FALLBACK;
          return (
            <button key={cat} type="button" onClick={() => onToggleCategory(cat)}
                    className="text-[11px] font-semibold px-2.5 py-1 rounded-full cursor-pointer transition-all"
                    style={{
                      background: isSel ? cc.bg : "var(--surface)",
                      color: isSel ? cc.fg : "var(--ink-3)",
                      border: `1px solid ${isSel ? cc.bg : "var(--border)"}`,
                    }}>
              {isSel ? "✓ " : ""}{cat}
            </button>
          );
        })}
      </div>

      {/* Note */}
      <textarea value={note} onChange={e => setNote(e.target.value)}
                onBlur={() => onSaveNote(note)}
                placeholder="What did you learn from this trade?"
                rows={3}
                className="w-full rounded-[10px] px-3 py-2 text-[12px] resize-vertical"
                style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)" }} />
    </div>
  );
}


// Excursion metrics on the closed-campaign row expander (migration 046).
// Renders MAE / MFE / max retrace as % + ×ATR + days-to-each, plus a
// capture ratio (exit price ÷ MFE price) — how much of the run the trader
// captured. Null-safe: any field missing → "—", the whole panel collapses
// to a "not populated" note when nothing is present at all.
function ExcursionPanel({ row }: { row: CampaignReviewRow }) {
  const mae = row.mae_pct ?? null;
  const mfe = row.mfe_pct ?? null;
  const retrace = row.max_retrace_pct ?? null;
  const atr21 = row.atr21_entry_pct ?? null;
  const dMae = row.days_to_mae ?? null;
  const dMfe = row.days_to_mfe ?? null;

  const anyValue = mae != null || mfe != null || retrace != null;
  if (!anyValue) {
    return (
      <div className="mb-3 text-[11px] italic"
           style={{ color: "var(--ink-4)" }}
           data-testid="campaign-review-excursion-empty">
        Excursion metrics not yet populated for this campaign.
      </div>
    );
  }

  const atrMult = (pct: number | null) =>
    pct != null && atr21 != null && atr21 > 0
      ? Math.abs(pct) / atr21
      : null;

  // Capture ratio = actual exit price / theoretical MFE price. Requires
  // avg_exit and MFE. Values > 1 don't happen (you can't exit above the
  // max high); values close to 1 mean the trader captured most of the run.
  const mfePrice = mfe != null && row.avg_entry > 0
    ? row.avg_entry * (1 + mfe / 100)
    : null;
  const captureRatio = mfePrice != null && mfePrice > 0 && row.avg_exit > 0
    ? row.avg_exit / mfePrice
    : null;

  const cell = (
    label: string,
    pct: number | null,
    days: number | null,
    testId: string,
    tone: "adverse" | "favorable" | "neutral",
  ) => {
    const mult = atrMult(pct);
    const color = pct == null || pct === 0
      ? "var(--ink-3)"
      : tone === "adverse" ? "#e5484d"
      : tone === "favorable" ? "#08a86b"
      : "var(--ink)";
    return (
      <div className="flex flex-col gap-0.5" data-testid={testId}>
        <span className="text-[9px] uppercase tracking-[0.08em] font-semibold"
              style={{ color: "var(--ink-4)" }}>{label}</span>
        <span className="text-[15px] font-semibold" style={{ color, fontFamily: mono }}>
          {pct == null ? "—" : `${pct.toFixed(2)}%`}
        </span>
        <span className="text-[10px]" style={{ color: "var(--ink-4)", fontFamily: mono }}>
          {mult != null ? `${mult.toFixed(2)}× ATR` : "—"}
          {days != null ? ` · day ${days}` : ""}
        </span>
      </div>
    );
  };

  return (
    <div className="mb-4 pb-3" style={{ borderBottom: "1px dashed var(--border)" }}
         data-testid="campaign-review-excursion-panel">
      <div className="text-[10px] uppercase tracking-[0.08em] font-semibold mb-2"
           style={{ color: "var(--ink-4)" }}>
        Excursion
      </div>
      <div className="grid gap-4" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
        {cell("MAE",         mae,     dMae, "campaign-review-mae",     "adverse")}
        {cell("MFE",         mfe,     dMfe, "campaign-review-mfe",     "favorable")}
        {cell("Max retrace", retrace, null, "campaign-review-retrace", "adverse")}
        <div className="flex flex-col gap-0.5" data-testid="campaign-review-capture">
          <span className="text-[9px] uppercase tracking-[0.08em] font-semibold"
                style={{ color: "var(--ink-4)" }}>Capture ratio</span>
          <span className="text-[15px] font-semibold" style={{ color: "var(--ink)", fontFamily: mono }}>
            {captureRatio == null ? "—" : `${(captureRatio * 100).toFixed(0)}%`}
          </span>
          <span className="text-[10px]" style={{ color: "var(--ink-4)", fontFamily: mono }}>
            {captureRatio == null ? "—" : "exit ÷ MFE price"}
          </span>
        </div>
      </div>
      {atr21 != null && (
        <div className="text-[10px] mt-2" style={{ color: "var(--ink-4)" }}>
          ATR21 at entry: <strong style={{ fontFamily: mono, color: "var(--ink-3)" }}>{atr21.toFixed(2)}%</strong>
          {" (frozen snapshot; ×ATR multiples derived from this)"}
        </div>
      )}
    </div>
  );
}
