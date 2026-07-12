"use client";

// Trader Mindset — behavior-pattern deep view.
//
// Sits under "Deep Dive" next to Trend Cycle Review and Campaign Review
// with the same chrome + navColor. Reads /api/mindset/traps once per
// (portfolio, weeks) pair and renders three views over the same data:
//
//   1. Summary tiles — total tags fired, most-frequent trap, positive:
//      negative ratio.
//   2. Heat map — tags x weeks grid, cell shade = fire count.
//   3. Drill-through list — trades that contributed to the selected
//      tag (comes from URL ?tag= or a cell/row click).
//
// Design lift: mirrors Trend Cycle Review's Fraunces italic header +
// Deep Dive blue (#0d6efd). No new backend endpoint — Phase 2's
// aggregation already returns everything the page needs.

import { useState, useEffect, useMemo, useCallback } from "react";
import { usePathname, useSearchParams, useRouter } from "next/navigation";
import Link from "next/link";
import { api, getActivePortfolio, type MindsetTrap, type MindsetTrapTrade } from "@/lib/api";
import { getGroupForHref } from "@/lib/nav";

const mono = "var(--font-jetbrains), monospace";

const WEEK_OPTIONS = [4, 8, 13, 26, 52] as const;
type WeekWindow = typeof WEEK_OPTIONS[number];

const POSITIVE_TAG = "Followed Plan";

// Short "May 11" style label for heat map column headers.
function shortDate(iso: string): string {
  const [y, m, d] = iso.split("-").map(n => parseInt(n, 10));
  if (!y || !m || !d) return iso;
  const monthShort = new Date(y, m - 1, d).toLocaleString("en-US", { month: "short" });
  return `${monthShort} ${d}`;
}

// Cell fill intensity: 0 count → surface, higher counts → progressively
// deeper accent. Cap at 4+ so a rogue high-count week doesn't wash out
// the rest of the grid.
function cellStyle(count: number, positive: boolean): React.CSSProperties {
  if (count === 0) {
    return {
      background: "var(--surface)",
      color: "var(--ink-4)",
      border: "1px solid var(--border)",
    };
  }
  const accent = positive ? "#08a86b" : "#e5484d";
  const intensity = Math.min(count, 4);
  const bgPct = 12 + intensity * 14; // 26 / 40 / 54 / 68
  return {
    background: `color-mix(in oklab, ${accent} ${bgPct}%, var(--surface))`,
    color: bgPct >= 50 ? "#fff" : accent,
    border: `1px solid color-mix(in oklab, ${accent} 30%, var(--border))`,
    fontWeight: 600,
  };
}

export function TraderMindset() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const router = useRouter();
  const navColor = getGroupForHref(pathname)?.color || "#0d6efd";

  const [portfolio] = useState<string>(getActivePortfolio());
  const [weeks, setWeeks] = useState<WeekWindow>(8);
  const [traps, setTraps] = useState<MindsetTrap[] | null>(null);
  const [weeksIncluded, setWeeksIncluded] = useState<Array<{ week_start: string }>>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Selected tag drives the drill-through list. Sourced from ?tag=
  // on mount so deep-links from the Recurring Traps strip land on a
  // pre-filtered view.
  const initialTag = searchParams.get("tag") || "";
  const [selectedTag, setSelectedTag] = useState<string>(initialTag);

  useEffect(() => {
    if (!portfolio) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    api.mindsetTraps(portfolio, weeks)
      .then(res => {
        if (cancelled) return;
        if (res && !res.error) {
          setTraps(res.traps || []);
          setWeeksIncluded(res.weeks_included || []);
        } else {
          setTraps([]);
          setWeeksIncluded([]);
          setError(res?.error || "No data");
        }
      })
      .catch(err => {
        if (cancelled) return;
        setError(String(err));
        setTraps([]);
      })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [portfolio, weeks]);

  const totalCount = useMemo(
    () => (traps || []).reduce((a, t) => a + t.total_count, 0),
    [traps],
  );

  const positiveCount = useMemo(
    () => (traps || []).filter(t => t.tag === POSITIVE_TAG).reduce((a, t) => a + t.total_count, 0),
    [traps],
  );
  const negativeCount = totalCount - positiveCount;
  const topTrap = (traps || [])[0] || null;
  // First non-positive trap for the "most-frequent MISTAKE" tile —
  // Followed Plan can top the sort with a healthy week, which shouldn't
  // hijack the "top mistake" answer.
  const topMistake = useMemo(
    () => (traps || []).find(t => t.tag !== POSITIVE_TAG) || null,
    [traps],
  );

  const setTagAndUrl = useCallback((tag: string) => {
    setSelectedTag(tag);
    // Reflect selection into ?tag= so the URL is shareable and back/forward work.
    const params = new URLSearchParams(searchParams.toString());
    if (tag) params.set("tag", tag);
    else params.delete("tag");
    router.replace(`${pathname}${params.toString() ? `?${params.toString()}` : ""}`, { scroll: false });
  }, [pathname, router, searchParams]);

  const drilldownTrades: MindsetTrapTrade[] = useMemo(() => {
    if (!selectedTag) return [];
    const trap = (traps || []).find(t => t.tag === selectedTag);
    return trap ? trap.trades : [];
  }, [selectedTag, traps]);

  const isEmpty = !loading && (traps || []).length === 0;

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      {/* Header */}
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          Trader <em className="italic" style={{ color: navColor }}>Mindset</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
          Behavior patterns from Weekly Retro · {portfolio} · last {weeks} weeks
        </div>
      </div>

      {/* Controls: week window switch + reset filter */}
      <div className="rounded-[14px] overflow-hidden mb-4" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <div className="px-[18px] py-[14px] flex flex-wrap items-center gap-3"
             style={{ background: "var(--bg-2)", borderBottom: "1px solid var(--border)" }}>
          <span className="text-[9px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>Window</span>
          <div className="flex gap-1.5">
            {WEEK_OPTIONS.map(w => {
              const active = weeks === w;
              return (
                <button
                  key={w}
                  type="button"
                  onClick={() => setWeeks(w)}
                  className="h-[32px] px-3 rounded-[8px] text-[11px] font-semibold cursor-pointer"
                  style={{
                    background: active ? `color-mix(in oklab, ${navColor} 12%, transparent)` : "var(--surface)",
                    color: active ? navColor : "var(--ink-3)",
                    border: `1px solid ${active ? navColor : "var(--border)"}`,
                  }}
                >
                  {w}w
                </button>
              );
            })}
          </div>
          {selectedTag && (
            <>
              <span className="text-[11px] ml-auto" style={{ color: "var(--ink-4)" }}>
                Focused on <strong style={{ color: "var(--ink-2)" }}>{selectedTag}</strong>
              </span>
              <button
                type="button"
                onClick={() => setTagAndUrl("")}
                className="h-[32px] px-3 rounded-[8px] text-[11px] font-semibold cursor-pointer"
                style={{ background: "var(--surface)", color: "var(--ink-3)", border: "1px solid var(--border)" }}
              >
                Clear
              </button>
            </>
          )}
        </div>

        <div className="px-[18px] py-[14px]">
          {loading ? (
            <div className="text-[12px]" style={{ color: "var(--ink-4)" }}>Loading pattern data…</div>
          ) : error ? (
            <div className="text-[12px]" style={{ color: "#e5484d" }}>Failed to load: {error}</div>
          ) : isEmpty ? (
            <div className="text-[12px]" style={{ color: "var(--ink-4)" }}>
              No tagged behaviors yet. Go to <Link href="/weekly-retro" className="underline">Weekly Retro</Link>,
              open Per-Ticker Details, and check the Behavior chips on your trades. This page will
              start populating as soon as you save.
            </div>
          ) : (
            <>
              {/* Summary tiles */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
                <SummaryTile
                  label="Tags Fired"
                  value={String(totalCount)}
                  sub={`across ${(traps || []).length} distinct tag${(traps || []).length === 1 ? "" : "s"}`}
                  accent={navColor}
                />
                <SummaryTile
                  label="Top Mistake"
                  value={topMistake ? topMistake.tag : "—"}
                  sub={topMistake ? `${topMistake.total_count} time${topMistake.total_count === 1 ? "" : "s"}` : "no mistakes tagged"}
                  accent="#e5484d"
                />
                <SummaryTile
                  label="Positive : Negative"
                  value={`${positiveCount} : ${negativeCount}`}
                  sub={
                    totalCount === 0 ? "—" :
                    positiveCount >= negativeCount
                      ? "clean-trade ratio holding"
                      : `${Math.round((negativeCount / totalCount) * 100)}% of tags are mistakes`
                  }
                  accent={positiveCount >= negativeCount ? "#08a86b" : "#f59f00"}
                />
              </div>

              {/* Heat map */}
              <HeatMap
                traps={traps || []}
                weekStarts={weeksIncluded.map(w => w.week_start)}
                selectedTag={selectedTag}
                onSelect={setTagAndUrl}
                topTrap={topTrap}
              />

              {/* Drill-through */}
              {selectedTag && (
                <DrilldownList
                  tag={selectedTag}
                  trades={drilldownTrades}
                  navColor={navColor}
                />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function SummaryTile({
  label, value, sub, accent,
}: { label: string; value: string; sub: string; accent: string }) {
  return (
    <div
      className="rounded-[12px] p-3"
      style={{ background: "var(--surface)", border: `1px solid color-mix(in oklab, ${accent} 24%, var(--border))` }}
    >
      <div className="text-[9px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>{label}</div>
      <div className="text-[18px] font-semibold mt-0.5" style={{ color: accent, fontFamily: "var(--font-fraunces), Georgia, serif", fontStyle: "italic" }}>
        {value}
      </div>
      <div className="text-[11px] mt-0.5" style={{ color: "var(--ink-3)" }}>{sub}</div>
    </div>
  );
}

function HeatMap({
  traps, weekStarts, selectedTag, onSelect,
}: {
  traps: MindsetTrap[];
  weekStarts: string[];
  selectedTag: string;
  onSelect: (tag: string) => void;
  topTrap: MindsetTrap | null;
}) {
  // Sort by total_count desc but pin Followed Plan first so the
  // positive baseline is anchored at the top of the grid regardless of
  // count magnitude — reads as "here's where clean trades sit relative
  // to your mistakes."
  const sorted = useMemo(() => {
    const arr = [...traps];
    arr.sort((a, b) => {
      if (a.tag === POSITIVE_TAG && b.tag !== POSITIVE_TAG) return -1;
      if (b.tag === POSITIVE_TAG && a.tag !== POSITIVE_TAG) return 1;
      return b.total_count - a.total_count;
    });
    return arr;
  }, [traps]);

  if (sorted.length === 0) return null;

  return (
    <div className="rounded-[12px] overflow-hidden mb-4" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
      <div className="flex items-center gap-2 px-3 py-2.5" style={{ borderBottom: "1px solid var(--border)" }}>
        <span className="text-[12px] font-semibold">Heat Map</span>
        <span className="text-[11px]" style={{ color: "var(--ink-4)" }}>tag × week · click a row to drill through</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-[11px]" style={{ borderCollapse: "separate", borderSpacing: 3 }}>
          <thead>
            <tr>
              <th className="text-left px-2 py-1.5" style={{ color: "var(--ink-4)", fontWeight: 600 }}>Tag</th>
              {weekStarts.map(ws => (
                <th key={ws} className="text-center px-1 py-1.5 whitespace-nowrap" style={{ color: "var(--ink-4)", fontWeight: 500, fontFamily: mono }}>
                  {shortDate(ws)}
                </th>
              ))}
              <th className="text-right px-2 py-1.5" style={{ color: "var(--ink-4)", fontWeight: 600 }}>Total</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(trap => {
              const positive = trap.tag === POSITIVE_TAG;
              const isSelected = selectedTag === trap.tag;
              return (
                <tr key={trap.tag}>
                  <td className="px-2 py-1.5">
                    <button
                      type="button"
                      onClick={() => onSelect(isSelected ? "" : trap.tag)}
                      className="text-left text-[11px] font-semibold cursor-pointer"
                      style={{
                        color: isSelected ? (positive ? "#08a86b" : "#e5484d") : "var(--ink-2)",
                        textDecoration: isSelected ? "underline" : "none",
                      }}
                    >
                      {trap.tag}
                    </button>
                  </td>
                  {trap.series.map(cell => (
                    <td
                      key={cell.week_start}
                      className="text-center px-1 py-1.5 rounded-[6px] cursor-pointer"
                      style={{ ...cellStyle(cell.count, positive), fontFamily: mono }}
                      onClick={() => cell.count > 0 && onSelect(trap.tag)}
                      title={`${trap.tag} · week of ${cell.week_start} · ${cell.count} fire${cell.count === 1 ? "" : "s"}`}
                    >
                      {cell.count > 0 ? cell.count : ""}
                    </td>
                  ))}
                  <td className="text-right px-2 py-1.5" style={{ fontFamily: mono, fontWeight: 700, color: positive ? "#08a86b" : "#e5484d" }}>
                    {trap.total_count}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function DrilldownList({
  tag, trades, navColor,
}: { tag: string; trades: MindsetTrapTrade[]; navColor: string }) {
  return (
    <div className="rounded-[12px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
      <div className="flex items-center gap-2 px-3 py-2.5" style={{ borderBottom: "1px solid var(--border)" }}>
        <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
        <span className="text-[12px] font-semibold">Trades tagged &quot;{tag}&quot;</span>
        <span className="text-[11px]" style={{ color: "var(--ink-4)" }}>
          {trades.length} trade{trades.length === 1 ? "" : "s"} · click week to jump back to that retro
        </span>
      </div>
      {trades.length === 0 ? (
        <div className="px-3 py-3 text-[12px]" style={{ color: "var(--ink-4)" }}>
          No trades in the current window tagged with this behavior.
        </div>
      ) : (
        <table className="w-full text-[12px]">
          <thead>
            <tr>
              <th className="text-left px-3 py-2 text-[10px] uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>Week</th>
              <th className="text-left px-3 py-2 text-[10px] uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>Ticker</th>
              <th className="text-left px-3 py-2 text-[10px] uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>Grade</th>
              <th className="text-left px-3 py-2 text-[10px] uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>Lesson</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((t, i) => (
              <tr key={`${t.week_start}-${t.ticker}-${i}`} style={{ borderTop: "1px solid var(--border)" }}>
                <td className="px-3 py-2" style={{ fontFamily: mono }}>
                  <Link href={`/weekly-retro?week=${t.week_start}`} className="hover:underline" style={{ color: navColor }}>
                    {shortDate(t.week_start)}
                  </Link>
                </td>
                <td className="px-3 py-2 font-semibold" style={{ fontFamily: mono }}>{t.ticker}</td>
                <td className="px-3 py-2">{t.grade || "—"}</td>
                <td className="px-3 py-2" style={{ color: "var(--ink-3)" }}>{t.notes || "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
