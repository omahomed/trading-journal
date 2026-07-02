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

interface Filters {
  q: string;
  series: SeriesKey;
  ticker: string;
  rule: string;
  instrument: InstrumentKey;
  lesson: string;  // "all" | "none" | category name
  dateRange: DateRangeKey;
  from: string;
  to: string;
  grade: GradeKey;
}

const DEFAULT_FILTERS: Filters = {
  q: "",
  series: "all",
  ticker: "all",
  rule: "all",
  instrument: "all",
  lesson: "all",
  dateRange: "ytd",
  from: "",
  to: "",
  grade: "all",
};

// Filter → date-range predicate. YTD = "everything since 2026-01-01"
// (the same population the server returns). Month = current calendar month.
// Week = Monday of current week. Custom uses the from/to inputs.
function dateFilterPasses(row: CampaignReviewRow, f: Filters): boolean {
  const d = row.closed_date ? new Date(row.closed_date) : null;
  if (!d || isNaN(d.getTime())) return f.dateRange === "ytd" || f.dateRange === "custom";
  const now = new Date();
  if (f.dateRange === "ytd") return true;
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

export function CampaignReview() {
  const pathname = usePathname();
  const navColor = getGroupForHref(pathname)?.color || "#0d6efd";

  const [rows, setRows] = useState<CampaignReviewRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<Filters>(DEFAULT_FILTERS);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [saving, setSaving] = useState<string | null>(null);

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

  // Derived filter option lists
  const tickerOptions = useMemo(() => {
    const s = new Set<string>();
    rows.forEach(r => { if (r.ticker) s.add(r.ticker); });
    return Array.from(s).sort();
  }, [rows]);

  const ruleOptions = useMemo(() => {
    const s = new Set<string>();
    rows.forEach(r => { if (r.rule) s.add(r.rule); });
    return Array.from(s).sort();
  }, [rows]);

  const filtered = useMemo(() => {
    return rows.filter(r => {
      // Search
      if (filters.q) {
        const q = filters.q.toLowerCase();
        const hay = `${r.ticker} ${r.trade_id} ${r.rule} ${r.sell_rule} ${r.lesson_note}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      // Series (original = no add-ons; add_on = had add-ons)
      if (filters.series === "original" && r.has_add_ons) return false;
      if (filters.series === "add_on" && !r.has_add_ons) return false;
      // Ticker
      if (filters.ticker !== "all" && r.ticker !== filters.ticker) return false;
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
  }, [rows, filters]);

  // Summary stats over the filtered set
  const stats = useMemo(() => {
    const n = filtered.length;
    let totalPl = 0, sumR = 0, rCount = 0, sumGrade = 0, gradeCount = 0;
    for (const r of filtered) {
      totalPl += r.realized_pl || 0;
      if (r.r_multiple != null) { sumR += r.r_multiple; rCount += 1; }
      if (r.grade != null) { sumGrade += r.grade; gradeCount += 1; }
    }
    return {
      n,
      totalPl,
      avgR: rCount > 0 ? sumR / rCount : null,
      avgGrade: gradeCount > 0 ? sumGrade / gradeCount : null,
      unratedCount: filtered.filter(r => r.grade == null).length,
    };
  }, [filtered]);

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
            label="Ticker"
            value={filters.ticker}
            onChange={v => setFilters(f => ({ ...f, ticker: v }))}
            options={[{ v: "all", l: "All tickers" }, ...tickerOptions.map(t => ({ v: t, l: t }))]}
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
        </div>

        {/* Summary strip */}
        <div className="px-[18px] py-[10px] flex flex-wrap items-center gap-[16px] text-[12px]"
             style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)", color: "var(--ink-3)" }}>
          <span><b style={{ color: "var(--ink)" }}>{stats.n}</b> campaign{stats.n === 1 ? "" : "s"}</span>
          <span>P&L <b style={{ color: stats.totalPl >= 0 ? "#08a86b" : "#e5484d", fontFamily: mono }}>{formatCurrency(stats.totalPl, { decimals: 0 })}</b></span>
          <span>Avg R <b style={{ color: "var(--ink)", fontFamily: mono }}>{stats.avgR == null ? "—" : `${stats.avgR.toFixed(2)}R`}</b></span>
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
                  <th className="text-left px-3 py-2.5 font-semibold uppercase tracking-[0.05em] text-[10px]">Ticker</th>
                  <th className="text-left px-3 py-2.5 font-semibold uppercase tracking-[0.05em] text-[10px]">Open → Close</th>
                  <th className="text-right px-3 py-2.5 font-semibold uppercase tracking-[0.05em] text-[10px]">P&L</th>
                  <th className="text-right px-3 py-2.5 font-semibold uppercase tracking-[0.05em] text-[10px]">Return</th>
                  <th className="text-right px-3 py-2.5 font-semibold uppercase tracking-[0.05em] text-[10px]">R</th>
                  <th className="text-left px-3 py-2.5 font-semibold uppercase tracking-[0.05em] text-[10px]">Grade</th>
                  <th className="text-left px-3 py-2.5 font-semibold uppercase tracking-[0.05em] text-[10px]">Lesson</th>
                  <th className="px-3 py-2.5" />
                </tr>
              </thead>
              <tbody>
                {filtered.map(r => {
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
                          {shortOpen} → {shortClose}
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
                          <td colSpan={8} className="px-4 py-4">
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
