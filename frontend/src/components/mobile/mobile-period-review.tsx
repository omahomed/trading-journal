"use client";

import { useEffect, useMemo, useState } from "react";
import { api, getActivePortfolio } from "@/lib/api";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";

type Tab = "weekly" | "monthly" | "annual";

interface JournalRow {
  day: string;
  beg_nlv?: string | number | null;
  end_nlv?: string | number | null;
  cash_change?: string | number | null;
  daily_return?: string | number | null;
  daily_dollar_change?: string | number | null;
  portfolio_ltd?: string | number | null;
  spy_ltd?: string | number | null;
  ndx_ltd?: string | number | null;
}

interface PeriodRow {
  label: string;
  date: Date;
  begNlv: number;
  endNlv: number;
  cashFlow: number;
  periodPnl: number;
  periodReturn: number;
}

/**
 * Mobile Period Review — compact vertical list of weekly / monthly /
 * annual TWR-linked periods. Same aggregation logic as the desktop
 * (period-review.tsx aggregatePeriods) so the two surfaces agree on
 * every row.
 *
 * Desktop shows: equity curve chart + insights panel + full financial
 * table with 8+ columns. Mobile shows: tab picker + headline totals +
 * a scrollable list of periods (most-recent first) with return, P&L,
 * begin/end NLV. Everything else -- benchmarks, capital deployed
 * chart, insights -- lives on desktop.
 */
export function MobilePeriodReview() {
  const [data, setData] = useState<JournalRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("weekly");

  useEffect(() => {
    api.journalHistory(getActivePortfolio(), 0)
      .then((rows) => setData((rows as unknown) as JournalRow[]))
      .catch((err) => {
        log.error("mobile-period-review", "journalHistory failed", err);
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => setLoading(false));
  }, []);

  const rows = useMemo(() => aggregatePeriods(data, tab), [data, tab]);
  const reversed = useMemo(() => [...rows].reverse(), [rows]);
  const summary = useMemo(() => buildSummary(rows), [rows]);

  return (
    <div className="pb-4 flex flex-col gap-3" data-testid="mobile-period-review-root">
      {/* Tab picker */}
      <div className="rounded-m-sm p-1 grid grid-cols-3 gap-1"
           style={{
             background: "var(--m-surface)",
             border: "0.5px solid var(--m-border)",
           }}>
        {(["weekly", "monthly", "annual"] as const).map((t) => {
          const active = t === tab;
          return (
            <button key={t} type="button" onClick={() => setTab(t)}
                    aria-pressed={active}
                    className="rounded-m-sm text-[12px] font-semibold py-2 uppercase tracking-[0.06em]"
                    style={{
                      background: active
                        ? "color-mix(in oklab, var(--m-accent) 14%, transparent)"
                        : "transparent",
                      color: active ? "var(--m-accent)" : "var(--m-text-dim)",
                      minHeight: 44,
                    }}>
              {t}
            </button>
          );
        })}
      </div>

      {error && (
        <div className="px-4 py-3 rounded-m-sm text-[12px]"
             style={{
               background: "color-mix(in oklab, var(--m-down) 12%, var(--m-surface))",
               border: "1px solid var(--m-warn-border-soft)",
               color: "var(--m-down)",
             }}>
          Failed to load: {error}
        </div>
      )}

      {/* Summary strip */}
      <div className="rounded-m-md p-4"
           style={{
             background: "var(--m-surface)",
             border: "0.5px solid var(--m-border)",
           }}>
        <div className="grid grid-cols-3 gap-3">
          <SummaryStat label={tab === "weekly" ? "Weeks" : tab === "monthly" ? "Months" : "Years"}
                       main={String(rows.length)} />
          <SummaryStat label="Best"
                       main={summary.best ? `${summary.best.periodReturn >= 0 ? "+" : ""}${summary.best.periodReturn.toFixed(1)}%` : "—"}
                       sub={summary.best?.label}
                       color="var(--m-accent)" />
          <SummaryStat label="Worst"
                       main={summary.worst ? `${summary.worst.periodReturn >= 0 ? "+" : ""}${summary.worst.periodReturn.toFixed(1)}%` : "—"}
                       sub={summary.worst?.label}
                       color="var(--m-down)" />
        </div>
      </div>

      {loading ? (
        <>
          {[0, 1, 2, 3].map(i => (
            <div key={i} className="rounded-m-md animate-pulse h-[68px]"
                 style={{ background: "var(--m-surface)" }} />
          ))}
        </>
      ) : reversed.length === 0 ? (
        <div className="rounded-m-md p-8 text-center text-[13px]"
             style={{
               background: "var(--m-surface)",
               border: "0.5px solid var(--m-border)",
               color: "var(--m-text-muted)",
             }}>
          No {tab} periods yet. Log an NLV to seed history.
        </div>
      ) : (
        <div className="rounded-m-md overflow-hidden"
             style={{
               background: "var(--m-surface)",
               border: "0.5px solid var(--m-border)",
             }}>
          {reversed.map((r, idx) => (
            <PeriodListRow key={r.label} row={r} isLast={idx === reversed.length - 1} />
          ))}
        </div>
      )}
    </div>
  );
}

// ── UI ─────────────────────────────────────────────────────────────

function SummaryStat({ label, main, sub, color }: {
  label: string;
  main: string;
  sub?: string;
  color?: string;
}) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-[0.06em] font-semibold text-m-text-dim">
        {label}
      </div>
      <div className="mt-0.5 text-[16px] font-semibold privacy-mask"
           style={{ color: color ?? "var(--m-text)", fontFamily: "var(--font-jetbrains), monospace" }}>
        {main}
      </div>
      {sub && (
        <div className="mt-0.5 text-[10px] text-m-text-faint truncate"
             style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
          {sub}
        </div>
      )}
    </div>
  );
}

function PeriodListRow({ row, isLast }: { row: PeriodRow; isLast: boolean }) {
  const retColor = row.periodReturn > 0 ? "var(--m-accent)"
                : row.periodReturn < 0 ? "var(--m-down)"
                : "var(--m-text-muted)";
  const plColor = row.periodPnl > 0 ? "var(--m-accent)"
                : row.periodPnl < 0 ? "var(--m-down)"
                : "var(--m-text-muted)";

  return (
    <div className="px-4 py-3 flex items-center justify-between gap-3"
         style={{ borderBottom: isLast ? "none" : "0.5px solid var(--m-border)" }}>
      <div className="min-w-0 flex-1">
        <div className="text-[13px] font-semibold text-m-text truncate"
             style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
          {row.label}
        </div>
        <div className="mt-0.5 text-[10px] text-m-text-faint privacy-mask"
             style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
          {formatCurrency(row.begNlv, { decimals: 0 })} → {formatCurrency(row.endNlv, { decimals: 0 })}
        </div>
      </div>
      <div className="text-right shrink-0">
        <div className="text-[14px] font-semibold"
             style={{ color: retColor, fontFamily: "var(--font-jetbrains), monospace" }}>
          {row.periodReturn >= 0 ? "+" : ""}{row.periodReturn.toFixed(2)}%
        </div>
        <div className="mt-0.5 text-[10px] privacy-mask"
             style={{ color: plColor, fontFamily: "var(--font-jetbrains), monospace" }}>
          {formatCurrency(row.periodPnl, { decimals: 0, showSign: true })}
        </div>
      </div>
    </div>
  );
}

// ── Aggregation (mirrors period-review.tsx:127-189) ───────────────

function aggregatePeriods(data: JournalRow[], mode: Tab): PeriodRow[] {
  if (!data.length) return [];

  const groups = new Map<string, JournalRow[]>();
  data.forEach((d) => {
    const dt = new Date(d.day);
    let key: string;
    if (mode === "weekly") {
      const fri = new Date(dt);
      const dow = fri.getDay();
      const diff = dow <= 5 ? 5 - dow : 5 - dow + 7;
      fri.setDate(fri.getDate() + diff);
      key = fri.toISOString().slice(0, 10);
    } else if (mode === "monthly") {
      key = `${dt.getFullYear()}-${String(dt.getMonth() + 1).padStart(2, "0")}`;
    } else {
      key = `${dt.getFullYear()}`;
    }
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(d);
  });

  const rows: PeriodRow[] = [];
  for (const [key, days] of groups) {
    const sorted = days.sort((a, b) => String(a.day).localeCompare(String(b.day)));
    const begNlv = num(sorted[0].beg_nlv ?? sorted[0].end_nlv);
    const endNlv = num(sorted[sorted.length - 1].end_nlv);
    const cashFlow = sorted.reduce((s, d) => s + num(d.cash_change), 0);

    let product = 1;
    sorted.forEach((d) => { product *= 1 + num(d.daily_return); });
    const periodReturn = (product - 1) * 100;
    const periodPnl = endNlv - (begNlv + cashFlow);

    let label: string;
    if (mode === "weekly") {
      label = key;
    } else if (mode === "monthly") {
      const [y, m] = key.split("-");
      const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
      label = `${monthNames[parseInt(m) - 1]} ${y}`;
    } else {
      label = key;
    }

    rows.push({
      label,
      date: new Date(sorted[sorted.length - 1].day),
      begNlv, endNlv, cashFlow, periodPnl, periodReturn,
    });
  }

  return rows.sort((a, b) => a.date.getTime() - b.date.getTime());
}

function num(v: string | number | null | undefined): number {
  const n = parseFloat(String(v ?? 0));
  return Number.isFinite(n) ? n : 0;
}

interface Summary {
  best: PeriodRow | null;
  worst: PeriodRow | null;
}

function buildSummary(rows: PeriodRow[]): Summary {
  if (!rows.length) return { best: null, worst: null };
  let best = rows[0];
  let worst = rows[0];
  for (const r of rows) {
    if (r.periodReturn > best.periodReturn) best = r;
    if (r.periodReturn < worst.periodReturn) worst = r;
  }
  return { best, worst };
}
