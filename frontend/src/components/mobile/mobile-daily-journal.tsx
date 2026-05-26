"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { ChevronRight } from "lucide-react";
import { api, getActivePortfolio, type JournalHistoryPoint } from "@/lib/api";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";
import { usePortfolio } from "@/lib/portfolio-context";

/**
 * Mobile Daily Journal — Phase 2 T2-1. Replaces the desktop 19-
 * column horizontal-scroll table with a chronological per-day
 * card stack. Read-only: tap a card → /daily-report?date=… (the
 * existing desktop detail view, which gets its own mobile-fit in
 * T2-4).
 *
 * Data source mirrors desktop daily-journal.tsx: single
 * `api.journalHistory(portfolio, 0)` call on mount returning the
 * full history for the active portfolio. No pagination, no pull-
 * to-refresh, no CSV export — desktop affordances that don't map
 * to mobile workflow.
 *
 * Sticky month headers group rows chronologically. Filter chip
 * row mirrors desktop's Week / Month / All toggle:
 *   - Week  → trailing 7-day rolling window (matches desktop)
 *   - Month → current calendar month only (mobile simplification
 *             — desktop's month-picker browsing is a follow-up)
 *   - All   → full history, no filter
 */

type ViewFilter = "week" | "month" | "all";

type MctStateName = "POWERTREND" | "UPTREND" | "RALLY MODE" | "CORRECTION";

type MctState = { state: MctStateName; display_day_num: number | null };

// Mobile-flavored MCT state colors. Mirrors desktop's state →
// color intent (purple / green / amber / red) but pulled from the
// mobile-tokens.css palette so the badge stays consistent with the
// rest of the mobile chrome. Desktop component is at daily-journal
// .tsx:23 with pure-RGB hex; mobile uses CSS vars.
const MCT_TONES: Record<MctStateName, { bg: string; fg: string }> = {
  POWERTREND:   { bg: "var(--m-purple)", fg: "var(--m-bg)" },
  UPTREND:      { bg: "var(--m-accent)", fg: "var(--m-accent-text-on)" },
  "RALLY MODE": { bg: "var(--m-warn)",   fg: "var(--m-bg)" },
  CORRECTION:   { bg: "var(--m-down)",   fg: "var(--m-bg)" },
};

function MobileMctBadge({ s }: { s: MctState | undefined }) {
  if (!s) {
    return (
      <span className="font-m-num text-[10px] tabular-nums text-m-text-faint">—</span>
    );
  }
  const tone = MCT_TONES[s.state];
  const showDay = typeof s.display_day_num === "number" && s.display_day_num > 0;
  return (
    <span
      className="inline-flex items-center gap-1 rounded-[4px] px-1.5 py-px text-[10px] font-semibold tracking-wider"
      style={{ background: tone.bg, color: tone.fg }}
      title={s.state}
    >
      {s.state}
      {showDay ? ` D${s.display_day_num}` : ""}
    </span>
  );
}

function mctFromRow(h: JournalHistoryPoint): MctState | undefined {
  const raw = String((h as Record<string, unknown>).market_cycle ?? "")
    .toUpperCase()
    .trim();
  if (
    raw !== "POWERTREND" &&
    raw !== "UPTREND" &&
    raw !== "RALLY MODE" &&
    raw !== "CORRECTION"
  ) {
    return undefined;
  }
  const dayRaw = (h as Record<string, unknown>).mct_display_day_num;
  const dayNum =
    typeof dayRaw === "number" && Number.isFinite(dayRaw) && dayRaw > 0
      ? Math.round(dayRaw)
      : null;
  return { state: raw as MctStateName, display_day_num: dayNum };
}

// 5-bucket grade label mirrors desktop daily-journal.tsx:338.
// Distinct from Daily Routine's 11-bucket letterGrade — different
// surfaces, different mappings.
function gradeLabel(score: number): string {
  if (score >= 5) return "A+";
  if (score >= 4) return "A";
  if (score >= 3) return "B";
  if (score >= 2) return "C";
  if (score > 0) return "D";
  return "";
}

type GradeTier = "high" | "mid" | "low" | "none";

function gradeTier(score: number): GradeTier {
  if (score >= 4) return "high";   // A+, A
  if (score >= 3) return "mid";    // B
  if (score > 0) return "low";     // C, D
  return "none";
}

function fmtDayLabel(day: string): string {
  // YYYY-MM-DD → "Mon May 25"
  const [y, m, d] = day.split("-").map(Number);
  if (!y || !m || !d) return day;
  return new Date(y, m - 1, d).toLocaleDateString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
  });
}

function fmtMonthHeader(yyyyMm: string): string {
  // YYYY-MM → "MAY 2026"
  const [y, m] = yyyyMm.split("-").map(Number);
  if (!y || !m) return yyyyMm;
  return new Date(y, m - 1, 1).toLocaleDateString("en-US", {
    month: "long",
    year: "numeric",
  }).toUpperCase();
}

function pctClass(v: number): string {
  if (v > 0) return "text-m-accent";
  if (v < 0) return "text-m-down";
  return "text-m-text-dim";
}

export function MobileDailyJournal() {
  const router = useRouter();
  const { activePortfolio } = usePortfolio();
  const [history, setHistory] = useState<JournalHistoryPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<ViewFilter>("week");

  // Mount fetch — full history, active portfolio, sorted newest-
  // first. Single call (matches desktop pattern).
  useEffect(() => {
    let cancelled = false;
    api
      .journalHistory(getActivePortfolio(), 0)
      .then((h) => {
        if (cancelled) return;
        const arr = (Array.isArray(h) ? h : []) as JournalHistoryPoint[];
        arr.sort((a, b) => String(b.day).localeCompare(String(a.day)));
        setHistory(arr);
        setLoading(false);
      })
      .catch((err) => {
        log.error("mobile-daily-journal", "journalHistory fetch failed", err);
        if (!cancelled) {
          setHistory([]);
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const currentMonth = useMemo(() => {
    const n = new Date();
    return `${n.getFullYear()}-${String(n.getMonth() + 1).padStart(2, "0")}`;
  }, []);

  // Predicates per audit §4. Week = trailing 7-day rolling window
  // (matches desktop); Month = current calendar month only (mobile
  // simplification — no month-picker browsing); All = no filter.
  const filtered = useMemo(() => {
    if (history.length === 0) return [];
    if (filter === "week") {
      const weekAgo = new Date(Date.now() - 7 * 86400000).toISOString().slice(0, 10);
      return history.filter((h) => String(h.day).slice(0, 10) >= weekAgo);
    }
    if (filter === "month") {
      return history.filter((h) => String(h.day).slice(0, 7) === currentMonth);
    }
    return history;
  }, [history, filter, currentMonth]);

  // Group filtered entries by month so the sticky header renders
  // once per chronological grouping. Map keyed by YYYY-MM, values
  // ordered newest-first within the group (matches outer sort).
  const grouped = useMemo(() => {
    const m = new Map<string, JournalHistoryPoint[]>();
    for (const h of filtered) {
      const key = String(h.day).slice(0, 7);
      const list = m.get(key);
      if (list) list.push(h);
      else m.set(key, [h]);
    }
    return Array.from(m.entries()); // already in newest-first order
  }, [filtered]);

  const handleTap = (day: string) => {
    router.push(`/daily-report?date=${day}`);
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
        entryCount={history.length}
        portfolioName={activePortfolio?.name ?? ""}
      />

      <FilterChips active={filter} onChange={setFilter} />

      {filtered.length === 0 ? (
        <EmptyState
          filter={filter}
          portfolioName={activePortfolio?.name ?? "this portfolio"}
          hasAnyHistory={history.length > 0}
          onSaveRoutine={() => router.push("/daily-routine")}
        />
      ) : (
        <div className="flex flex-col gap-3">
          {grouped.map(([yyyyMm, rows]) => (
            <section key={yyyyMm} className="flex flex-col gap-2">
              <h2
                data-testid={`month-header-${yyyyMm}`}
                className="sticky top-0 z-10 -mx-5 px-5 py-1.5 text-[10px] font-semibold tracking-[0.10em] text-m-text-dim"
                style={{ background: "var(--m-bg)" }}
              >
                {fmtMonthHeader(yyyyMm)}
              </h2>
              <div className="flex flex-col gap-2">
                {rows.map((h) => (
                  <DayCard
                    key={String(h.day)}
                    entry={h}
                    onTap={() => handleTap(String(h.day).slice(0, 10))}
                  />
                ))}
              </div>
            </section>
          ))}

          {filter !== "all" && (
            <EndOfListFooter filter={filter} />
          )}
        </div>
      )}
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────

function Header({
  entryCount,
  portfolioName,
}: {
  entryCount: number;
  portfolioName: string;
}) {
  // Wordmark dropped — the shell's MobilePageHeader owns the page
  // title. This is the per-page metadata strip only.
  return (
    <div className="pt-1 text-[12px] text-m-text-dim">
      {entryCount} {entryCount === 1 ? "entry" : "entries"}
      {portfolioName && <> · {portfolioName}</>}
    </div>
  );
}

function FilterChips({
  active,
  onChange,
}: {
  active: ViewFilter;
  onChange: (f: ViewFilter) => void;
}) {
  const chips: { key: ViewFilter; label: string }[] = [
    { key: "week", label: "Week" },
    { key: "month", label: "Month" },
    { key: "all", label: "All" },
  ];
  return (
    <div
      role="radiogroup"
      aria-label="Date range filter"
      className="-mx-5 overflow-x-auto whitespace-nowrap px-5"
    >
      {chips.map((c) => {
        const isActive = c.key === active;
        const className = isActive
          ? "mr-1.5 inline-block rounded-m-pill bg-m-accent px-[14px] py-1.5 text-xs font-medium text-m-accent-text-on"
          : "mr-1.5 inline-block rounded-m-pill border-[0.5px] border-m-border bg-m-surface px-[14px] py-1.5 text-xs text-m-text-muted";
        return (
          <button
            key={c.key}
            type="button"
            role="radio"
            aria-checked={isActive}
            onClick={() => onChange(c.key)}
            className={className}
          >
            {c.label}
          </button>
        );
      })}
    </div>
  );
}

function DayCard({
  entry,
  onTap,
}: {
  entry: JournalHistoryPoint;
  onTap: () => void;
}) {
  const day = String(entry.day).slice(0, 10);
  const dailyPct = Number(entry.daily_pct_change ?? 0) || 0;
  const ltdPct = Number((entry as Record<string, unknown>).portfolio_ltd ?? 0) || 0;
  const pctInv = Number(entry.pct_invested ?? 0) || 0;
  const ndxPct = Number((entry as Record<string, unknown>).ndx_daily_pct ?? 0) || 0;
  const spyPct = Number((entry as Record<string, unknown>).spy_daily_pct ?? 0) || 0;
  const score = Number((entry as Record<string, unknown>).score ?? 0) || 0;
  const grade = gradeLabel(score);
  const tier = gradeTier(score);
  const mct = mctFromRow(entry);

  return (
    <button
      type="button"
      data-testid={`day-card-${day}`}
      onClick={onTap}
      aria-label={`Open daily report for ${fmtDayLabel(day)}`}
      className="flex flex-col gap-1.5 rounded-m-lg border-[0.5px] border-m-border bg-m-surface px-[14px] py-3 text-left active:opacity-80"
    >
      {/* Row 1: Date · MCT badge · Grade pill · chevron */}
      <div className="flex items-baseline justify-between gap-2">
        <div className="flex min-w-0 items-baseline gap-1.5">
          <span className="font-m-num text-[13px] font-medium tabular-nums text-m-text">
            {fmtDayLabel(day)}
          </span>
          <MobileMctBadge s={mct} />
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          <GradePill label={grade} tier={tier} />
          <ChevronRight size={14} strokeWidth={1.5} className="text-m-text-faint" aria-hidden="true" />
        </div>
      </div>

      {/* Row 2: End NLV (large) · Daily % (signed, colored) */}
      <div className="flex items-baseline justify-between gap-2">
        <span className="font-m-num text-[17px] font-medium tabular-nums text-m-text privacy-mask">
          {formatCurrency(Number(entry.end_nlv ?? 0), { decimals: 0 })}
        </span>
        <span
          className={`font-m-num text-[15px] font-medium tabular-nums ${pctClass(dailyPct)}`}
        >
          {dailyPct >= 0 ? "+" : ""}
          {dailyPct.toFixed(2)}%
        </span>
      </div>

      {/* Sub-row: LTD · % invested · NDX · SPY (4 items — fits on
          380px+ viewports per user on-device verification of T2-1.
          Heat remains deferred to T2-4 detail view per scope plan.) */}
      <div
        data-testid={`day-card-sub-${day}`}
        className="flex items-baseline gap-3 font-m-num text-[11px] tabular-nums text-m-text-dim"
      >
        <span data-testid={`sub-ltd-${day}`}>
          <span className={pctClass(ltdPct)}>
            {ltdPct >= 0 ? "+" : ""}
            {ltdPct.toFixed(2)}%
          </span>{" "}
          LTD
        </span>
        <span data-testid={`sub-inv-${day}`}>{pctInv.toFixed(1)}% inv</span>
        <span data-testid={`sub-ndx-${day}`}>
          <span className={pctClass(ndxPct)}>
            {ndxPct >= 0 ? "+" : ""}
            {ndxPct.toFixed(2)}%
          </span>{" "}
          NDX
        </span>
        <span data-testid={`sub-spy-${day}`}>
          <span className={pctClass(spyPct)}>
            {spyPct >= 0 ? "+" : ""}
            {spyPct.toFixed(2)}%
          </span>{" "}
          SPY
        </span>
      </div>
    </button>
  );
}

function GradePill({ label, tier }: { label: string; tier: GradeTier }) {
  if (!label) {
    return (
      <span className="font-m-num text-[10px] tabular-nums text-m-text-faint">—</span>
    );
  }
  // Tier → token mapping. "low" tier has no Tailwind tint utility
  // (no --m-down-tint token exists), so the background falls back
  // to inline color-mix per the convention used in mobile-position-
  // sizer / mobile-trade-journal / mobile-score-selector.
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

function EmptyState({
  filter,
  portfolioName,
  hasAnyHistory,
  onSaveRoutine,
}: {
  filter: ViewFilter;
  portfolioName: string;
  hasAnyHistory: boolean;
  onSaveRoutine: () => void;
}) {
  const rangeLabel =
    filter === "week" ? "this week" : filter === "month" ? "this month" : "any time";

  return (
    <div
      data-testid="empty-state"
      className="rounded-m-xl border-[0.5px] border-m-border bg-m-surface px-5 py-10 text-center"
    >
      <div className="text-[14px] font-medium text-m-text">
        {hasAnyHistory ? `No entries for ${rangeLabel}` : "No entries yet"}
      </div>
      <div className="mt-1.5 text-[12px] text-m-text-dim">
        {hasAnyHistory
          ? `Nothing saved for ${portfolioName} ${rangeLabel === "any time" ? "" : `in ${rangeLabel}`}.`
          : `Start your journal by saving today's daily routine.`}
      </div>
      <button
        type="button"
        onClick={onSaveRoutine}
        className="mt-3 inline-flex h-9 items-center justify-center rounded-m-pill bg-m-accent px-4 text-[12px] font-medium text-m-accent-text-on"
      >
        Save daily routine →
      </button>
    </div>
  );
}

function EndOfListFooter({ filter }: { filter: ViewFilter }) {
  const text =
    filter === "week"
      ? "End of week · Tap Month for more history"
      : "End of month · Tap All for more history";
  return (
    <div
      data-testid="end-of-list-footer"
      className="py-3 text-center text-[11px] text-m-text-faint"
    >
      {text}
    </div>
  );
}
