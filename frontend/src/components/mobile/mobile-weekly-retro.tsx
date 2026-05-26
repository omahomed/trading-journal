"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import { Calendar, ChevronDown, ChevronRight, Pin, RotateCcw, X } from "lucide-react";
import { api, getActivePortfolio, type NotesRailItem, type NotesRailItemTag } from "@/lib/api";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";
import { usePortfolio } from "@/lib/portfolio-context";
import { TAG_PALETTE, type TagTone } from "@/lib/tag-palette";

/**
 * Mobile Weekly Retro — Phase 2 T2-2. Read-only chronological list of
 * weekly retrospectives. Replaces the desktop 706-LOC monolith (rail +
 * tiles + form + modal) with a per-week card stack scoped to the active
 * portfolio. Tap a card → /weekly-retro?week=YYYY-MM-DD; the desktop
 * detail view honors that query param via the page.tsx searchParams
 * passthrough (mirrors the daily-report ?date= pattern).
 *
 * Data source: `api.weeklyRetroList(portfolio)` returns the rail
 * envelope { weeks, ytd_stats } where weeks includes synthetic empty
 * rows (id: null, has_content: false) for Mondays without a retro so
 * the desktop sparkline grid stays continuous. The mobile list filters
 * those out — only real retros (has_content === true) belong on a list
 * view.
 *
 * Filter model is two-dimensional: { year, month }. Defaults to current
 * year + current month on mount so the page lands on ~3-5 cards. User
 * widens via the two pill pickers; the Reset chip snaps both back.
 * Cross-month weeks (Apr 28 – May 2) filter by week_start month (April).
 *
 * LTD% and YTD% are computed client-side from chained weekly_return_pct
 * since the backend doesn't expose running cumulatives on the list
 * endpoint. LTD chains oldest → newest across all years; YTD resets at
 * each January 1.
 */

type YearFilter = number | "all";
type MonthFilter = number | "all"; // 0..11 like Date#getMonth

// ── Pure helpers ────────────────────────────────────────────────────

function fmtWeekSpan(weekStart: string, weekEnd: string): string {
  // "2026-05-19", "2026-05-23" → "May 19 – 23" (same month) or
  // "2026-04-28", "2026-05-02" → "Apr 28 – May 2" (cross-month).
  const [sy, sm, sd] = weekStart.split("-").map(Number);
  const [ey, em, ed] = weekEnd.split("-").map(Number);
  if (!sy || !sm || !sd || !ey || !em || !ed) return `${weekStart} – ${weekEnd}`;
  const start = new Date(sy, sm - 1, sd);
  const end = new Date(ey, em - 1, ed);
  const startMonth = start.toLocaleDateString("en-US", { month: "short" });
  if (sm === em) {
    return `${startMonth} ${sd} – ${ed}`;
  }
  const endMonth = end.toLocaleDateString("en-US", { month: "short" });
  return `${startMonth} ${sd} – ${endMonth} ${ed}`;
}

function fmtMonthHeader(year: number, month: number): string {
  // month is 0..11. "MAY 2026"
  const d = new Date(year, month, 1);
  return d.toLocaleDateString("en-US", { month: "long", year: "numeric" }).toUpperCase();
}

const MONTH_NAMES = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

function pctClass(v: number): string {
  if (v > 0) return "text-m-accent";
  if (v < 0) return "text-m-down";
  return "text-m-text-dim";
}

function signed(v: number, decimals = 2): string {
  return `${v >= 0 ? "+" : ""}${v.toFixed(decimals)}%`;
}

// 5-bucket grade tier from the desktop letter ("A+", "B-", "C+", ...).
// A+/A → high, B → mid, C/D/F → low, "" / null → none. Mirrors directive
// §A2 and matches daily-journal's high/mid/low palette so the visual
// language is consistent across list views.
type GradeTier = "high" | "mid" | "low" | "none";

function gradeTier(letter: string | null | undefined): GradeTier {
  if (!letter) return "none";
  const first = letter.trim().charAt(0).toUpperCase();
  if (first === "A") return "high";
  if (first === "B") return "mid";
  if (first === "C" || first === "D" || first === "F") return "low";
  return "none";
}

/**
 * Compute LTD% and YTD% per row from chained weekly_return_pct.
 * Input is in any order; we re-sort oldest → newest internally, chain,
 * and return a Map keyed by week_start. LTD chains across all years.
 * YTD resets at each calendar year boundary (by week_start year). Null
 * weekly_return_pct values are treated as 0% for the chain (logged once).
 */
function computeLtdYtd(weeks: NotesRailItem[]): Map<string, { ltdPct: number; ytdPct: number }> {
  const sorted = [...weeks].sort((a, b) => a.week_start.localeCompare(b.week_start));
  const out = new Map<string, { ltdPct: number; ytdPct: number }>();
  let ltdMult = 1;
  let ytdMult = 1;
  let currentYear: number | null = null;
  let nullWarned = false;
  for (const w of sorted) {
    const year = Number(w.week_start.slice(0, 4));
    if (currentYear === null || year !== currentYear) {
      ytdMult = 1;
      currentYear = year;
    }
    const raw = w.sparkline_value;
    if (raw == null) {
      if (!nullWarned) {
        log.warn("mobile-weekly-retro", "missing sparkline_value, treating as 0%", { week: w.week_start });
        nullWarned = true;
      }
    }
    const r = typeof raw === "number" && Number.isFinite(raw) ? raw : 0;
    ltdMult *= 1 + r / 100;
    ytdMult *= 1 + r / 100;
    out.set(w.week_start, {
      ltdPct: (ltdMult - 1) * 100,
      ytdPct: (ytdMult - 1) * 100,
    });
  }
  return out;
}

// ── Main component ─────────────────────────────────────────────────

export function MobileWeeklyRetro() {
  const router = useRouter();
  const { activePortfolio } = usePortfolio();
  const [weeks, setWeeks] = useState<NotesRailItem[]>([]);
  const [loading, setLoading] = useState(true);

  // Default filter — current year + current month so the list lands on
  // a focused set rather than the full history.
  const now = useMemo(() => new Date(), []);
  const [year, setYear] = useState<YearFilter>(now.getFullYear());
  const [month, setMonth] = useState<MonthFilter>(now.getMonth());

  useEffect(() => {
    let cancelled = false;
    api
      .weeklyRetroList(getActivePortfolio())
      .then((res) => {
        if (cancelled) return;
        if (res && typeof res === "object" && "weeks" in res && Array.isArray(res.weeks)) {
          // Filter out synthetic empty rows (id: null OR has_content false).
          // Desktop rail needs continuous Mondays for the sparkline grid;
          // the mobile list shows only real retros.
          const real = res.weeks.filter((w) => w.has_content && w.id != null);
          setWeeks(real);
        } else {
          setWeeks([]);
        }
        setLoading(false);
      })
      .catch((err) => {
        log.error("mobile-weekly-retro", "weeklyRetroList fetch failed", err);
        if (!cancelled) {
          setWeeks([]);
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Chained LTD / YTD computed once per data set. Stable map keyed by
  // week_start so per-row lookup is O(1) during render.
  const ltdYtd = useMemo(() => computeLtdYtd(weeks), [weeks]);

  // Year selector options: descending unique years across the data.
  const yearOptions = useMemo(() => {
    const years = new Set<number>();
    for (const w of weeks) years.add(Number(w.week_start.slice(0, 4)));
    return Array.from(years).sort((a, b) => b - a);
  }, [weeks]);

  // Apply year + month predicate.
  const filtered = useMemo(() => {
    if (weeks.length === 0) return [];
    return weeks.filter((w) => {
      const wy = Number(w.week_start.slice(0, 4));
      const wm = Number(w.week_start.slice(5, 7)) - 1; // 0..11
      if (year !== "all" && wy !== year) return false;
      if (month !== "all" && wm !== month) return false;
      return true;
    });
  }, [weeks, year, month]);

  // Group filtered weeks by YYYY-MM (sticky section header per month).
  // Backend returns newest-first; preserve that within groups.
  const grouped = useMemo(() => {
    const m = new Map<string, NotesRailItem[]>();
    for (const w of filtered) {
      const key = w.week_start.slice(0, 7);
      const list = m.get(key);
      if (list) list.push(w);
      else m.set(key, [w]);
    }
    return Array.from(m.entries()); // newest-first thanks to upstream sort
  }, [filtered]);

  const isFilteredDefault = year === now.getFullYear() && month === now.getMonth();
  const resetFilters = () => {
    setYear(now.getFullYear());
    setMonth(now.getMonth());
  };

  const handleTap = (weekStart: string) => {
    router.push(`/weekly-retro?week=${weekStart}`);
  };

  if (loading) {
    return (
      <div className="flex flex-col gap-3 pt-2">
        <div className="h-9 animate-pulse rounded-m-md bg-m-surface-2" />
        <div className="h-9 animate-pulse rounded-m-md bg-m-surface-2" />
        <div className="h-24 animate-pulse rounded-m-lg bg-m-surface-2" />
        <div className="h-24 animate-pulse rounded-m-lg bg-m-surface-2" />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 pt-2">
      <Header
        entryCount={weeks.length}
        portfolioName={activePortfolio?.name ?? ""}
      />

      <FilterRow
        year={year}
        month={month}
        yearOptions={yearOptions}
        onYearChange={setYear}
        onMonthChange={setMonth}
        isDefault={isFilteredDefault}
        onReset={resetFilters}
      />

      {filtered.length === 0 ? (
        <EmptyState
          year={year}
          month={month}
          portfolioName={activePortfolio?.name ?? "this portfolio"}
        />
      ) : (
        <div className="flex flex-col gap-3">
          {grouped.map(([yyyyMm, rows]) => {
            const [yStr, mStr] = yyyyMm.split("-");
            const yNum = Number(yStr);
            const mNum = Number(mStr) - 1;
            return (
              <section key={yyyyMm} className="flex flex-col gap-2">
                <h2
                  data-testid={`month-header-${yyyyMm}`}
                  className="sticky top-0 z-10 -mx-5 flex items-baseline gap-1.5 px-5 py-1.5 text-[10px] font-semibold tracking-[0.10em] text-m-text-dim"
                  style={{ background: "var(--m-bg)" }}
                >
                  <span>{fmtMonthHeader(yNum, mNum)}</span>
                  <span className="text-m-text-faint">
                    · {rows.length} {rows.length === 1 ? "retro" : "retros"}
                  </span>
                </h2>
                <div className="flex flex-col gap-2">
                  {rows.map((w) => (
                    <WeekCard
                      key={w.week_start}
                      week={w}
                      ltdYtd={ltdYtd.get(w.week_start)}
                      onTap={() => handleTap(w.week_start)}
                    />
                  ))}
                </div>
              </section>
            );
          })}

          <EndOfListFooter year={year} month={month} />
        </div>
      )}
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────────

function Header({
  entryCount,
  portfolioName,
}: {
  entryCount: number;
  portfolioName: string;
}) {
  // Wordmark dropped — the shell's MobilePageHeader owns the page title.
  return (
    <div className="pt-1 text-[12px] text-m-text-dim">
      {entryCount} {entryCount === 1 ? "retro" : "retros"}
      {portfolioName && <> · {portfolioName}</>}
    </div>
  );
}

function FilterRow({
  year,
  month,
  yearOptions,
  onYearChange,
  onMonthChange,
  isDefault,
  onReset,
}: {
  year: YearFilter;
  month: MonthFilter;
  yearOptions: number[];
  onYearChange: (y: YearFilter) => void;
  onMonthChange: (m: MonthFilter) => void;
  isDefault: boolean;
  onReset: () => void;
}) {
  const yearLabel = year === "all" ? "All time" : String(year);
  const monthLabel = month === "all" ? "All months" : MONTH_NAMES[month] ?? "—";

  return (
    <div className="-mx-5 flex items-center gap-2 overflow-x-auto whitespace-nowrap px-5">
      <PillSelectSheet
        triggerLabel={yearLabel}
        sheetTitle="Filter by year"
        active={year !== "all"}
        leadingIcon={<Calendar size={12} strokeWidth={1.75} aria-hidden="true" />}
      >
        {(close) => (
          <PillSheetOptionList
            options={[
              { key: "all", label: "All time", value: "all" as const },
              ...yearOptions.map((y) => ({ key: String(y), label: String(y), value: y })),
            ]}
            currentValue={year}
            onSelect={(v) => {
              onYearChange(v);
              close();
            }}
          />
        )}
      </PillSelectSheet>

      <PillSelectSheet
        triggerLabel={monthLabel}
        sheetTitle="Filter by month"
        active={month !== "all"}
      >
        {(close) => (
          <PillSheetOptionList
            options={[
              { key: "all", label: "All months", value: "all" as const },
              ...MONTH_NAMES.map((name, i) => ({
                key: String(i),
                label: name,
                value: i,
              })),
            ]}
            currentValue={month}
            onSelect={(v) => {
              onMonthChange(v);
              close();
            }}
          />
        )}
      </PillSelectSheet>

      <button
        type="button"
        data-testid="filter-reset"
        onClick={onReset}
        aria-label="Reset filters to current month"
        className={`ml-auto inline-flex min-h-[32px] items-center gap-1 rounded-m-pill border-[0.5px] border-m-border px-3 py-1 text-[11px] ${
          isDefault ? "text-m-text-faint" : "text-m-text-dim"
        }`}
      >
        <RotateCcw size={11} strokeWidth={1.75} aria-hidden="true" />
        Reset
      </button>
    </div>
  );
}

// Pill-shaped trigger + bottom sheet. Specialized to the filter row; not
// extracted to a shared primitive yet (MobileSelectSheet is tile-shaped
// with a label-over-value layout that doesn't fit this row).
function PillSelectSheet({
  triggerLabel,
  sheetTitle,
  active,
  leadingIcon,
  children,
}: {
  triggerLabel: string;
  sheetTitle: string;
  active: boolean;
  leadingIcon?: ReactNode;
  children: (close: () => void) => ReactNode;
}) {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [open]);

  const triggerClass = active
    ? "inline-flex min-h-[32px] items-center gap-1.5 rounded-m-pill bg-m-accent px-3 py-1 text-[12px] font-medium text-m-accent-text-on"
    : "inline-flex min-h-[32px] items-center gap-1.5 rounded-m-pill border-[0.5px] border-m-border bg-m-surface px-3 py-1 text-[12px] text-m-text";

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label={`${sheetTitle}: ${triggerLabel}. Tap to change.`}
        className={triggerClass}
      >
        {leadingIcon}
        <span className="font-m-num tabular-nums">{triggerLabel}</span>
        <ChevronDown size={11} strokeWidth={1.5} aria-hidden="true" />
      </button>

      {open && (
        <>
          <button
            type="button"
            aria-label={`Close ${sheetTitle}`}
            onClick={() => setOpen(false)}
            className="fixed inset-0 z-40 bg-black/50"
            style={{ animation: "m-backdrop-enter var(--m-duration-tap) ease-out" }}
          />
          <div
            role="dialog"
            aria-modal="true"
            aria-label={sheetTitle}
            className="fixed inset-x-0 bottom-0 z-50 flex max-h-[75vh] flex-col border-t-[0.5px] border-m-border bg-m-bg"
            style={{
              borderTopLeftRadius: "var(--m-radius-xl)",
              borderTopRightRadius: "var(--m-radius-xl)",
              animation: "m-sheet-enter var(--m-duration-sheet) var(--m-ease-spring)",
            }}
          >
            <div className="flex shrink-0 items-center justify-between border-b-[0.5px] border-m-border px-5 pt-4 pb-3">
              <h2 className="text-base font-medium text-m-text">{sheetTitle}</h2>
              <button
                type="button"
                onClick={() => setOpen(false)}
                aria-label="Close"
                className="flex h-8 w-8 items-center justify-center text-m-text-dim"
              >
                <X size={20} strokeWidth={1.5} aria-hidden="true" />
              </button>
            </div>
            <div
              role="listbox"
              aria-label={sheetTitle}
              className="min-h-0 flex-1 overflow-y-auto"
              style={{
                paddingBottom: "max(1.5rem, env(safe-area-inset-bottom))",
                WebkitOverflowScrolling: "touch",
              }}
            >
              {children(() => setOpen(false))}
            </div>
          </div>
        </>
      )}
    </>
  );
}

function PillSheetOptionList<T extends string | number>({
  options,
  currentValue,
  onSelect,
}: {
  options: Array<{ key: string; label: string; value: T }>;
  currentValue: T;
  onSelect: (v: T) => void;
}) {
  return (
    <>
      {options.map((opt) => {
        const isActive = opt.value === currentValue;
        return (
          <button
            key={opt.key}
            type="button"
            role="option"
            aria-selected={isActive}
            onClick={() => onSelect(opt.value)}
            className={`flex min-h-[48px] w-full items-center justify-between border-b-[0.5px] border-m-border px-5 py-3 text-left last:border-b-0 ${
              isActive ? "bg-m-surface" : ""
            }`}
          >
            <span className={`text-base ${isActive ? "font-medium text-m-text" : "text-m-text-dim"}`}>
              {opt.label}
            </span>
            {isActive && (
              <span aria-hidden="true" className="text-[11px] font-medium text-m-accent">
                Selected
              </span>
            )}
          </button>
        );
      })}
    </>
  );
}

function WeekCard({
  week,
  ltdYtd,
  onTap,
}: {
  week: NotesRailItem;
  ltdYtd: { ltdPct: number; ytdPct: number } | undefined;
  onTap: () => void;
}) {
  const weeklyPct = typeof week.sparkline_value === "number" ? week.sparkline_value : 0;
  const weeklyPnl = typeof week.weekly_pnl === "number" ? week.weekly_pnl : 0;
  const ltdPct = ltdYtd?.ltdPct ?? 0;
  const ytdPct = ltdYtd?.ytdPct ?? 0;
  const tier = gradeTier(week.week_grade);
  const span = fmtWeekSpan(week.week_start, week.week_end);

  return (
    <button
      type="button"
      data-testid={`week-card-${week.week_start}`}
      onClick={onTap}
      aria-label={`Open weekly retro for ${span}`}
      className="flex flex-col gap-1.5 rounded-m-lg border-[0.5px] border-m-border bg-m-surface px-[14px] py-3 text-left active:opacity-80"
    >
      {/* Row 1: [pin] · week span · grade pill · chevron */}
      <div className="flex items-baseline justify-between gap-2">
        <div className="flex min-w-0 items-baseline gap-1.5">
          {week.pinned && (
            <Pin
              data-testid={`week-card-pin-${week.week_start}`}
              size={13}
              strokeWidth={1.75}
              className="text-m-purple shrink-0 self-center"
              aria-label="Pinned"
            />
          )}
          <span className="font-m-num text-[13px] font-medium tabular-nums text-m-text">
            {span}
          </span>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          <GradePill label={week.week_grade} tier={tier} />
          <ChevronRight size={14} strokeWidth={1.5} className="text-m-text-faint" aria-hidden="true" />
        </div>
      </div>

      {/* Row 2: Weekly return % (large) · Weekly P&L $ (small muted) */}
      <div className="flex items-baseline justify-between gap-2">
        <span className={`font-m-num text-[22px] font-medium tabular-nums ${pctClass(weeklyPct)}`}>
          {signed(weeklyPct)}
        </span>
        <span className="font-m-num text-[13px] tabular-nums text-m-text-dim privacy-mask">
          {weeklyPnl >= 0 ? "+" : "−"}
          {formatCurrency(Math.abs(weeklyPnl), { decimals: 0 })}
        </span>
      </div>

      {/* Sub-row: N trades · LTD% · YTD% */}
      <div
        data-testid={`week-card-sub-${week.week_start}`}
        className="flex items-baseline gap-3 font-m-num text-[11px] tabular-nums text-m-text-dim"
      >
        <span data-testid={`sub-trades-${week.week_start}`}>
          {week.trades_count} {week.trades_count === 1 ? "trade" : "trades"}
        </span>
        <span data-testid={`sub-ltd-${week.week_start}`}>
          <span className={pctClass(ltdPct)}>{signed(ltdPct, 1)}</span> LTD
        </span>
        <span data-testid={`sub-ytd-${week.week_start}`}>
          <span className={pctClass(ytdPct)}>{signed(ytdPct, 1)}</span> YTD
        </span>
      </div>

      {/* Conditional tag row — entirely omitted when no tags. */}
      {week.tags.length > 0 && (
        <div
          data-testid={`week-card-tags-${week.week_start}`}
          className="mt-0.5 flex flex-wrap items-center gap-1.5"
        >
          {week.tags.map((t) => (
            <TagPill key={t.name} tag={t} />
          ))}
        </div>
      )}
    </button>
  );
}

function GradePill({
  label,
  tier,
}: {
  label: string | null;
  tier: GradeTier;
}) {
  if (!label || tier === "none") {
    return (
      <span
        data-testid="grade-pill-none"
        className="font-m-num text-[10px] tabular-nums text-m-text-faint"
      >
        —
      </span>
    );
  }
  const className =
    tier === "high"
      ? "rounded-m-pill bg-m-accent-tint px-2 py-px text-[10px] font-semibold text-m-accent border-[0.5px] border-m-accent-border"
      : tier === "mid"
        ? "rounded-m-pill bg-m-warn-tint px-2 py-px text-[10px] font-semibold text-m-warn border-[0.5px] border-m-warn-border-soft"
        : "rounded-m-pill px-2 py-px text-[10px] font-semibold text-m-down";
  const style =
    tier === "low"
      ? {
          background: "color-mix(in oklab, var(--m-down) 14%, var(--m-surface))",
          border: "0.5px solid color-mix(in oklab, var(--m-down) 30%, var(--m-border))",
        }
      : undefined;
  return (
    <span data-testid={`grade-pill-${tier}`} className={className} style={style}>
      {label}
    </span>
  );
}

function TagPill({ tag }: { tag: NotesRailItemTag }) {
  // Look up the user-stored palette key (rose | amber | emerald | sky |
  // violet) in TAG_PALETTE. Fall back to a neutral muted style if the
  // backend ever sends an unknown color key (e.g., schema additions not
  // yet shipped to the frontend).
  const palette = TAG_PALETTE[tag.color as TagTone];
  if (!palette) {
    return (
      <span
        data-testid={`tag-pill-${tag.name}`}
        className="rounded-m-pill border-[0.5px] border-m-border bg-m-surface-2 px-2 py-px text-[10px] font-medium text-m-text-dim"
      >
        {tag.name}
      </span>
    );
  }
  return (
    <span
      data-testid={`tag-pill-${tag.name}`}
      className="rounded-m-pill px-2 py-px text-[10px] font-medium"
      style={{
        background: palette.body,
        color: palette.text,
        border: `0.5px solid ${palette.ring}`,
      }}
    >
      {tag.name}
    </span>
  );
}

function EmptyState({
  year,
  month,
  portfolioName,
}: {
  year: YearFilter;
  month: MonthFilter;
  portfolioName: string;
}) {
  const yearText = year === "all" ? "" : ` ${year}`;
  const monthText = month === "all" ? "" : ` ${MONTH_NAMES[month] ?? ""}`;
  const filterText = `${monthText}${yearText}`.trim();
  return (
    <div
      data-testid="empty-state"
      className="rounded-m-xl border-[0.5px] border-m-border bg-m-surface px-5 py-10 text-center"
    >
      <div className="text-[14px] font-medium text-m-text">
        No weekly retros{filterText ? ` for ${filterText}` : ""}
      </div>
      <div className="mt-1.5 text-[12px] text-m-text-dim">
        Save weekly retros from desktop in {portfolioName}.
      </div>
    </div>
  );
}

function EndOfListFooter({ year, month }: { year: YearFilter; month: MonthFilter }) {
  if (year === "all" && month === "all") {
    return (
      <div
        data-testid="end-of-list-footer"
        className="py-3 text-center text-[11px] text-m-text-faint"
      >
        End of history
      </div>
    );
  }
  let text = "";
  if (year !== "all" && month !== "all") {
    text = `End of ${MONTH_NAMES[month] ?? ""} ${year}`;
  } else if (year !== "all") {
    text = `End of ${year}`;
  } else if (month !== "all") {
    text = `End of ${MONTH_NAMES[month] ?? ""}`;
  }
  return (
    <div
      data-testid="end-of-list-footer"
      className="py-3 text-center text-[11px] text-m-text-faint"
    >
      {text}
    </div>
  );
}
