"use client";

// Trend Cycle Review — per-leg performance analysis using the trend_count
// column stamped by the engine's Phase 11. Sits under "Deep Dive" next
// to Campaign Review with the same chrome + color scheme. Purely
// client-side: reads journal history via api.journalHistory (which now
// includes trend_count), groups rows by sign, and computes leg metrics
// via lib/trend-cycles. No new backend endpoint, no persisted state.

import { useState, useEffect, useMemo, useCallback } from "react";
import { usePathname } from "next/navigation";
import { api, getActivePortfolio, type JournalHistoryPoint } from "@/lib/api";
import { formatCurrency } from "@/lib/format";
import { getGroupForHref } from "@/lib/nav";
import { computeTrendCycles, type TrendCycleLeg } from "@/lib/trend-cycles";

const mono = "var(--font-jetbrains), monospace";

type SignKey = "all" | "positive" | "negative";
type DateRangeKey = "all" | "ytd" | "month" | "week" | "custom";
type SortKey = "end_date" | "duration_days" | "return_pct" | "return_dollars"
  | "alpha_pct" | "ndx_return_pct" | "spy_return_pct"
  | "max_drawdown_pct" | "avg_pct_invested";
type SortDir = "asc" | "desc";

interface Filters {
  sign: SignKey;
  dateRange: DateRangeKey;
  from: string;
  to: string;
}

const DEFAULT_FILTERS: Filters = {
  sign: "all",
  dateRange: "all",
  from: "",
  to: "",
};

function hasActiveFilters(f: Filters): boolean {
  return f.sign !== "all"
    || f.dateRange !== "all"
    || f.from !== ""
    || f.to !== "";
}

// Filter → date-range predicate. Uses the leg's END date (the leg is
// "closed" on that date, which is what the review is about).
function dateFilterPasses(leg: TrendCycleLeg, f: Filters): boolean {
  const d = leg.end_date ? new Date(leg.end_date) : null;
  if (!d || isNaN(d.getTime())) return f.dateRange === "all" || f.dateRange === "custom";
  const now = new Date();
  if (f.dateRange === "all") return true;
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

// Duplicated from Campaign Review for consistency without cross-module
// coupling. Tiny controls; a shared extraction isn't worth the refactor.
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

export function TrendCycleReview() {
  const pathname = usePathname();
  const navColor = getGroupForHref(pathname)?.color || "#0d6efd";

  const [rows, setRows] = useState<JournalHistoryPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<Filters>(DEFAULT_FILTERS);
  const [sortKey, setSortKey] = useState<SortKey>("end_date");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      // days=0 returns full history; the endpoint already includes
      // trend_count in the whitelist after the Half-2 fix.
      const data = await api.journalHistory(getActivePortfolio(), 0);
      setRows(Array.isArray(data) ? data : []);
    } catch (e: any) {
      setError(e?.message || "Failed to load journal history");
      setRows([]);
    }
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  // Compute all legs from journal history (client-side, pure derivation).
  // `baselineAggregates` reflects UNFILTERED history — used by the Cycle
  // Anatomy card so its "typical shape" baselines don't move when you
  // filter. The summary strip further down uses filteredAggregates.
  const { legs, aggregates: baselineAggregates } = useMemo(() => computeTrendCycles(rows), [rows]);

  // Filter
  const filtered = useMemo(() => {
    return legs.filter(l => {
      if (filters.sign === "positive" && l.sign !== 1) return false;
      if (filters.sign === "negative" && l.sign !== -1) return false;
      if (!dateFilterPasses(l, filters)) return false;
      return true;
    });
  }, [legs, filters]);

  // Recompute the aggregates over the FILTERED set — makes the strip
  // reflect the current slice. Same pattern as Campaign Review.
  const filteredAggregates = useMemo(() => {
    return computeTrendCyclesFromLegs(filtered);
  }, [filtered]);

  // Sort
  const sorted = useMemo(() => {
    const out = [...filtered];
    const mult = sortDir === "asc" ? 1 : -1;
    out.sort((a, b) => {
      const av = (a as any)[sortKey];
      const bv = (b as any)[sortKey];
      const aNull = av == null;
      const bNull = bv == null;
      if (aNull && bNull) return 0;
      if (aNull) return 1;
      if (bNull) return -1;
      if (typeof av === "number" && typeof bv === "number") return (av - bv) * mult;
      return String(av).localeCompare(String(bv)) * mult;
    });
    return out;
  }, [filtered, sortKey, sortDir]);

  const toggleSort = useCallback((key: SortKey) => {
    if (sortKey === key) {
      setSortDir(d => d === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      // Dates/numbers desc first (latest/biggest matter more); non-date default asc.
      setSortDir(key === "end_date" ? "desc" : "desc");
    }
  }, [sortKey]);

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      {/* Header */}
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          Trend Cycle <em className="italic" style={{ color: navColor }}>Review</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
          Per-leg performance — return, alpha vs NASDAQ, drawdown, and exposure discipline for every 21e trend cycle you've journaled.
        </div>
      </div>

      {/* Filter bar */}
      <div className="rounded-[14px] overflow-hidden mb-4" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <div className="px-[18px] py-[14px] flex flex-wrap items-end gap-[12px_14px]"
             style={{ background: "var(--bg-2)", borderBottom: "1px solid var(--border)" }}>
          <SegmentedControl<SignKey>
            label="Sign"
            value={filters.sign}
            onChange={v => setFilters(f => ({ ...f, sign: v }))}
            options={[
              { v: "all", l: "All" },
              { v: "positive", l: "▲ Positive" },
              { v: "negative", l: "▼ Negative" },
            ]}
          />

          <SegmentedControl<DateRangeKey>
            label="End date"
            value={filters.dateRange}
            onChange={v => setFilters(f => ({ ...f, dateRange: v }))}
            options={[
              { v: "week", l: "Week" },
              { v: "month", l: "Month" },
              { v: "ytd", l: "YTD" },
              { v: "all", l: "All" },
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

          {(() => {
            const active = hasActiveFilters(filters);
            return (
              <div className="flex flex-col gap-1 ml-auto">
                <span className="text-[9px] font-bold uppercase tracking-[0.08em]" style={{ color: "transparent" }}>Reset</span>
                <button type="button"
                        onClick={() => setFilters(DEFAULT_FILTERS)}
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

        {/* Cycle Anatomy — historical baselines over ALL legs (never
            filtered). Reference for "is this leg abnormal?" — the
            summary strip below reflects the current filtered slice. */}
        {legs.length > 0 && (
          <div className="px-[18px] py-[10px] flex flex-col gap-[6px] text-[11px]"
               style={{ background: "color-mix(in oklab, var(--surface-2) 60%, var(--surface))", borderBottom: "1px solid var(--border)" }}>
            <div className="text-[9px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>
              Cycle anatomy · over {baselineAggregates.total_legs} legs of history
            </div>
            {/* Positive baselines */}
            {baselineAggregates.positive_legs > 0 && (
              <div className="flex flex-wrap items-center gap-x-[14px] gap-y-[2px]"
                   style={{ color: "var(--ink-3)" }}>
                <span className="font-bold" style={{ color: "#08a86b" }}>▲ POSITIVE</span>
                <span>avg <b style={{ color: "var(--ink)", fontFamily: mono }}>{baselineAggregates.avg_positive_duration == null ? "—" : `${baselineAggregates.avg_positive_duration.toFixed(1)}d`}</b></span>
                <span>median <b style={{ color: "var(--ink)", fontFamily: mono }}>{baselineAggregates.median_positive_duration == null ? "—" : `${baselineAggregates.median_positive_duration.toFixed(0)}d`}</b></span>
                <span>range <b style={{ color: "var(--ink)", fontFamily: mono }}>{baselineAggregates.shortest_positive_duration == null ? "—" : `${baselineAggregates.shortest_positive_duration}-${baselineAggregates.longest_positive_days}d`}</b></span>
                <span>avg return <b style={{ color: (baselineAggregates.avg_positive_return_pct ?? 0) >= 0 ? "#08a86b" : "#e5484d", fontFamily: mono }}>{baselineAggregates.avg_positive_return_pct == null ? "—" : `${baselineAggregates.avg_positive_return_pct >= 0 ? "+" : ""}${baselineAggregates.avg_positive_return_pct.toFixed(2)}%`}</b></span>
                <span>avg DD <b style={{ color: "#e5484d", fontFamily: mono }}>{baselineAggregates.avg_positive_dd_pct == null ? "—" : `${baselineAggregates.avg_positive_dd_pct.toFixed(2)}%`}</b></span>
              </div>
            )}
            {/* Negative baselines */}
            {baselineAggregates.negative_legs > 0 && (
              <div className="flex flex-wrap items-center gap-x-[14px] gap-y-[2px]"
                   style={{ color: "var(--ink-3)" }}>
                <span className="font-bold" style={{ color: "#e5484d" }}>▼ NEGATIVE</span>
                <span>avg <b style={{ color: "var(--ink)", fontFamily: mono }}>{baselineAggregates.avg_negative_duration == null ? "—" : `${baselineAggregates.avg_negative_duration.toFixed(1)}d`}</b></span>
                <span>median <b style={{ color: "var(--ink)", fontFamily: mono }}>{baselineAggregates.median_negative_duration == null ? "—" : `${baselineAggregates.median_negative_duration.toFixed(0)}d`}</b></span>
                <span>range <b style={{ color: "var(--ink)", fontFamily: mono }}>{baselineAggregates.shortest_negative_duration == null ? "—" : `${baselineAggregates.shortest_negative_duration}-${baselineAggregates.longest_negative_days}d`}</b></span>
                <span>avg return <b style={{ color: (baselineAggregates.avg_negative_return_pct ?? 0) >= 0 ? "#08a86b" : "#e5484d", fontFamily: mono }}>{baselineAggregates.avg_negative_return_pct == null ? "—" : `${baselineAggregates.avg_negative_return_pct >= 0 ? "+" : ""}${baselineAggregates.avg_negative_return_pct.toFixed(2)}%`}</b></span>
                <span>avg DD <b style={{ color: "#e5484d", fontFamily: mono }}>{baselineAggregates.avg_negative_dd_pct == null ? "—" : `${baselineAggregates.avg_negative_dd_pct.toFixed(2)}%`}</b></span>
              </div>
            )}
          </div>
        )}

        {/* Summary strip */}
        <div className="px-[18px] py-[10px] flex flex-wrap items-center gap-[16px] text-[12px]"
             style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)", color: "var(--ink-3)" }}>
          <span><b style={{ color: "var(--ink)" }}>{filteredAggregates.total_legs}</b> leg{filteredAggregates.total_legs === 1 ? "" : "s"} <span style={{ color: "var(--ink-4)" }}>({filteredAggregates.positive_legs}▲ · {filteredAggregates.negative_legs}▼)</span></span>
          <span>
            Win rate{" "}
            <b style={{ color: (filteredAggregates.win_rate ?? 0) >= 0.5 ? "#08a86b" : "#d97706", fontFamily: mono }}>
              {filteredAggregates.win_rate == null ? "—" : `${(filteredAggregates.win_rate * 100).toFixed(0)}%`}
            </b>
          </span>
          <span>
            Avg ▲ leg{" "}
            <b style={{ color: (filteredAggregates.avg_positive_return_pct ?? 0) >= 0 ? "#08a86b" : "#e5484d", fontFamily: mono }}>
              {filteredAggregates.avg_positive_return_pct == null ? "—" : `${filteredAggregates.avg_positive_return_pct >= 0 ? "+" : ""}${filteredAggregates.avg_positive_return_pct.toFixed(2)}%`}
            </b>
          </span>
          <span>
            Avg ▼ leg{" "}
            <b style={{ color: (filteredAggregates.avg_negative_return_pct ?? 0) >= 0 ? "#08a86b" : "#e5484d", fontFamily: mono }}>
              {filteredAggregates.avg_negative_return_pct == null ? "—" : `${filteredAggregates.avg_negative_return_pct >= 0 ? "+" : ""}${filteredAggregates.avg_negative_return_pct.toFixed(2)}%`}
            </b>
          </span>
          <span title="avg win × win rate − avg loss × loss rate, per leg (%)">
            Expectancy{" "}
            <b style={{ color: (filteredAggregates.expectancy_pct ?? 0) >= 0 ? "#08a86b" : "#e5484d", fontFamily: mono }}>
              {filteredAggregates.expectancy_pct == null ? "—" : `${filteredAggregates.expectancy_pct >= 0 ? "+" : ""}${filteredAggregates.expectancy_pct.toFixed(2)}%/leg`}
            </b>
          </span>
          <span title="Sum of per-leg alpha (portfolio return − SPY return)">
            Cum α vs SPY{" "}
            <b style={{ color: (filteredAggregates.cumulative_alpha_pct ?? 0) >= 0 ? "#08a86b" : "#e5484d", fontFamily: mono }}>
              {filteredAggregates.cumulative_alpha_pct == null ? "—" : `${filteredAggregates.cumulative_alpha_pct >= 0 ? "+" : ""}${filteredAggregates.cumulative_alpha_pct.toFixed(2)}%`}
            </b>
          </span>
          <span title="Avg % invested during positive vs negative legs — proxy for exposure discipline. Values are stored as percentages already (e.g. 106% = using margin), not fractions.">
            % Inv{" "}
            <b style={{ color: "#08a86b", fontFamily: mono }}>
              {filteredAggregates.avg_pct_invested_positive == null ? "—" : `${filteredAggregates.avg_pct_invested_positive.toFixed(0)}%`}
            </b>
            <span style={{ color: "var(--ink-4)" }}> ▲ · </span>
            <b style={{ color: "#e5484d", fontFamily: mono }}>
              {filteredAggregates.avg_pct_invested_negative == null ? "—" : `${filteredAggregates.avg_pct_invested_negative.toFixed(0)}%`}
            </b>
            <span style={{ color: "var(--ink-4)" }}> ▼</span>
          </span>
        </div>

        {/* Table */}
        {loading ? (
          <div className="px-[18px] py-8 text-center text-[12px]" style={{ color: "var(--ink-3)" }}>Loading journal history…</div>
        ) : error ? (
          <div className="px-[18px] py-8 text-center text-[12px]" style={{ color: "#e5484d" }}>Failed: {error}</div>
        ) : filtered.length === 0 ? (
          <div className="px-[18px] py-8 text-center text-[12px]" style={{ color: "var(--ink-3)" }}>
            {legs.length === 0 ? "No legs detected. Journal needs at least one row with a non-null, non-zero trend_count." : "No legs match the current filters."}
          </div>
        ) : (
          <div className="overflow-auto">
            <table className="w-full text-[12px]" style={{ borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ background: "var(--surface-2)", color: "var(--ink-4)" }}>
                  <SortHeader label="Cycle"      sortKey="end_date"         activeKey={sortKey} dir={sortDir} onToggle={toggleSort} align="left" />
                  <SortHeader label="Duration"   sortKey="duration_days"    activeKey={sortKey} dir={sortDir} onToggle={toggleSort} align="right" />
                  <SortHeader label="P&L $"      sortKey="return_dollars"   activeKey={sortKey} dir={sortDir} onToggle={toggleSort} align="right" />
                  <SortHeader label="P&L %"      sortKey="return_pct"       activeKey={sortKey} dir={sortDir} onToggle={toggleSort} align="right" />
                  <SortHeader label="NDX %"      sortKey="ndx_return_pct"   activeKey={sortKey} dir={sortDir} onToggle={toggleSort} align="right" />
                  <SortHeader label="SPY %"      sortKey="spy_return_pct"   activeKey={sortKey} dir={sortDir} onToggle={toggleSort} align="right" />
                  <SortHeader label="α vs SPY"   sortKey="alpha_pct"        activeKey={sortKey} dir={sortDir} onToggle={toggleSort} align="right" />
                  <SortHeader label="Max DD"     sortKey="max_drawdown_pct" activeKey={sortKey} dir={sortDir} onToggle={toggleSort} align="right" />
                  <SortHeader label="Avg % Inv"  sortKey="avg_pct_invested" activeKey={sortKey} dir={sortDir} onToggle={toggleSort} align="right" />
                </tr>
              </thead>
              <tbody>
                {sorted.map((l, i) => {
                  const signColor = l.sign === 1 ? "#08a86b" : "#e5484d";
                  const returnColor = l.return_pct == null ? "var(--ink-3)"
                    : l.return_pct > 0 ? "#08a86b"
                    : l.return_pct < 0 ? "#e5484d"
                    : "var(--ink)";
                  const alphaColor = l.alpha_pct == null ? "var(--ink-3)"
                    : l.alpha_pct > 0 ? "#08a86b"
                    : l.alpha_pct < 0 ? "#e5484d"
                    : "var(--ink)";
                  return (
                    <tr key={i}
                        className="transition-colors hover:brightness-95"
                        style={{ borderBottom: "1px solid var(--border)" }}>
                      <td className="px-3 py-2.5">
                        <div className="flex items-center gap-2">
                          <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded"
                                style={{ background: "var(--bg-2)", color: "var(--ink-4)", fontFamily: mono }}>
                            #{l.cycle_number}
                          </span>
                          <span className="text-[16px] leading-none" style={{ color: signColor }}>
                            {l.sign === 1 ? "▲" : "▼"}
                          </span>
                          <span className="text-[13px] font-semibold" style={{ color: signColor, fontFamily: mono }}>
                            {l.sign === 1 ? `+${l.duration_days}` : `-${l.duration_days}`}
                          </span>
                        </div>
                        <div className="text-[10px] mt-1" style={{ color: "var(--ink-4)", fontFamily: mono }}>
                          {l.start_date} → {l.end_date}
                        </div>
                      </td>
                      <td className="px-3 py-2.5 text-right" style={{ fontFamily: mono, color: "var(--ink)" }}>
                        {l.duration_days}d
                      </td>
                      <td className="px-3 py-2.5 text-right font-semibold"
                          style={{ fontFamily: mono, color: returnColor }}>
                        {l.start_nlv == null
                          ? <span style={{ color: "var(--ink-5)" }}>—</span>
                          : formatCurrency(l.return_dollars, { decimals: 0, showSign: true })}
                      </td>
                      <td className="px-3 py-2.5 text-right font-semibold"
                          style={{ fontFamily: mono, color: returnColor }}>
                        {l.return_pct == null ? "—" : `${l.return_pct >= 0 ? "+" : ""}${l.return_pct.toFixed(2)}%`}
                      </td>
                      <td className="px-3 py-2.5 text-right"
                          style={{ fontFamily: mono, color: (l.ndx_return_pct ?? 0) >= 0 ? "#08a86b" : "#e5484d" }}>
                        {l.ndx_return_pct == null ? "—" : `${l.ndx_return_pct >= 0 ? "+" : ""}${l.ndx_return_pct.toFixed(2)}%`}
                      </td>
                      <td className="px-3 py-2.5 text-right"
                          style={{ fontFamily: mono, color: (l.spy_return_pct ?? 0) >= 0 ? "#08a86b" : "#e5484d" }}>
                        {l.spy_return_pct == null ? "—" : `${l.spy_return_pct >= 0 ? "+" : ""}${l.spy_return_pct.toFixed(2)}%`}
                      </td>
                      <td className="px-3 py-2.5 text-right font-semibold"
                          style={{ fontFamily: mono, color: alphaColor }}>
                        {l.alpha_pct == null ? "—" : `${l.alpha_pct >= 0 ? "+" : ""}${l.alpha_pct.toFixed(2)}%`}
                      </td>
                      <td className="px-3 py-2.5 text-right"
                          style={{ fontFamily: mono, color: l.max_drawdown_pct < -5 ? "#e5484d" : "var(--ink-3)" }}>
                        {l.max_drawdown_pct === 0 ? "—" : `${l.max_drawdown_pct.toFixed(2)}%`}
                      </td>
                      <td className="px-3 py-2.5 text-right" style={{ fontFamily: mono, color: "var(--ink-3)" }}>
                        {`${l.avg_pct_invested.toFixed(0)}%`}
                      </td>
                    </tr>
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

// Small helper: re-run buildAggregates when the visible leg set changes.
// Cheaper than recomputing legs from raw rows on every filter change,
// which computeTrendCycles would do — legs never change once computed
// for a given data set. Duplicated here for encapsulation, but really
// just calls buildAggregates internally via computeTrendCyclesFromLegs.
function computeTrendCyclesFromLegs(legs: TrendCycleLeg[]): ReturnType<typeof computeTrendCycles>["aggregates"] {
  // Recompute aggregates given a pre-filtered leg set. Same math as
  // buildAggregates in lib/trend-cycles.ts — inlined here to avoid
  // exporting the private helper. Cheap: legs are ≤ ~50 in a typical
  // year, so the O(n) reduce runs in microseconds.
  const positive = legs.filter(l => l.sign === 1);
  const negative = legs.filter(l => l.sign === -1);
  const legsWithReturn = legs.filter(l => l.return_pct != null);
  const winners = legsWithReturn.filter(l => (l.return_pct as number) > 0);
  const losers = legsWithReturn.filter(l => (l.return_pct as number) < 0);
  const avg = (arr: number[]) => arr.length > 0 ? arr.reduce((s, v) => s + v, 0) / arr.length : null;
  const avgPosReturn = positive.length > 0 ? avg(positive.map(l => l.return_pct ?? 0)) : null;
  const avgNegReturn = negative.length > 0 ? avg(negative.map(l => l.return_pct ?? 0)) : null;
  const decidedTrades = winners.length + losers.length;
  const winRate = decidedTrades > 0 ? winners.length / decidedTrades : null;
  const avgWinPct = winners.length > 0 ? winners.reduce((s, l) => s + (l.return_pct as number), 0) / winners.length : 0;
  const avgLossPct = losers.length > 0 ? Math.abs(losers.reduce((s, l) => s + (l.return_pct as number), 0)) / losers.length : 0;
  const expectancy_pct = winRate != null ? avgWinPct * winRate - avgLossPct * (1 - winRate) : null;
  const avgWinDollars = winners.length > 0 ? winners.reduce((s, l) => s + l.return_dollars, 0) / winners.length : 0;
  const avgLossDollars = losers.length > 0 ? Math.abs(losers.reduce((s, l) => s + l.return_dollars, 0)) / losers.length : 0;
  const expectancy_dollars = winRate != null ? avgWinDollars * winRate - avgLossDollars * (1 - winRate) : null;
  const legsWithAlpha = legs.filter(l => l.alpha_pct != null);
  const cumulative_alpha_pct = legsWithAlpha.length > 0
    ? legsWithAlpha.reduce((s, l) => s + (l.alpha_pct as number), 0)
    : null;
  const median = (arr: number[]): number | null => {
    if (arr.length === 0) return null;
    const s = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(s.length / 2);
    return s.length % 2 === 0 ? (s[mid - 1] + s[mid]) / 2 : s[mid];
  };
  const min = (arr: number[]): number | null =>
    arr.length === 0 ? null : Math.min(...arr);
  const posDurations = positive.map(l => l.duration_days);
  const negDurations = negative.map(l => l.duration_days);
  return {
    total_legs: legs.length,
    positive_legs: positive.length,
    negative_legs: negative.length,
    win_rate: winRate,
    avg_positive_return_pct: avgPosReturn,
    avg_negative_return_pct: avgNegReturn,
    expectancy_pct,
    expectancy_dollars,
    cumulative_alpha_pct,
    avg_pct_invested_positive: avg(positive.map(l => l.avg_pct_invested)),
    avg_pct_invested_negative: avg(negative.map(l => l.avg_pct_invested)),
    longest_positive_days: positive.reduce((m, l) => Math.max(m, l.duration_days), 0),
    longest_negative_days: negative.reduce((m, l) => Math.max(m, l.duration_days), 0),
    avg_positive_duration: avg(posDurations),
    median_positive_duration: median(posDurations),
    shortest_positive_duration: min(posDurations),
    avg_positive_dd_pct: avg(positive.map(l => l.max_drawdown_pct)),
    avg_negative_duration: avg(negDurations),
    median_negative_duration: median(negDurations),
    shortest_negative_duration: min(negDurations),
    avg_negative_dd_pct: avg(negative.map(l => l.max_drawdown_pct)),
  };
}
