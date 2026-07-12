"use client";

// Trader Mindset Phase 3 — compact "Recurring Traps" strip.
//
// Sits above Per-Ticker Details on Weekly Retro. Shows the top-3
// behavior tags over the last 8 weeks so the user sees the pattern
// they're about to reinforce (or break) right before they grade
// this week's trades.
//
// Reads /api/mindset/traps once on mount. Zero-state text nudges the
// user into tagging behaviors so the strip has something to show.
// The chips are display-only in Phase 3; Phase 4 wires them into
// the Trader Mindset page for drill-through.

import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import type { MindsetTrap } from "@/lib/api";
import { log } from "@/lib/log";

// "Followed Plan" is the only positive; it gets the same green tint
// as the chip picker on Per-Ticker Details so the eye reads pattern
// polarity at a glance.
const POSITIVE_TAG = "Followed Plan";

function trapAccent(tag: string): string {
  return tag === POSITIVE_TAG ? "#08a86b" : "#e5484d";
}

export function RecurringTrapsStrip({
  portfolio,
  weeks = 8,
  navColor,
}: {
  portfolio: string;
  weeks?: number;
  navColor: string;
}) {
  const [traps, setTraps] = useState<MindsetTrap[] | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!portfolio) return;
    let cancelled = false;
    setLoading(true);
    api.mindsetTraps(portfolio, weeks)
      .then(res => {
        if (cancelled) return;
        if (res && !res.error) setTraps(res.traps || []);
        else setTraps([]);
      })
      .catch(err => {
        if (cancelled) return;
        log.error("recurring-traps-strip", `mindsetTraps fetch failed for ${portfolio}`, err);
        setTraps([]);
      })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [portfolio, weeks]);

  const topThree = (traps || []).slice(0, 3);
  const totalOverAll = (traps || []).reduce((a, t) => a + t.total_count, 0);
  const hasMore = (traps || []).length > 3;

  return (
    <div
      data-testid="recurring-traps-strip"
      className="mb-4 rounded-[14px] overflow-hidden"
      style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}
    >
      <div
        className="flex items-center gap-2 px-[18px] py-3"
        style={{ borderBottom: "1px solid var(--border)" }}
      >
        <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
        <span className="text-[13px] font-semibold">Recurring Traps</span>
        <span className="text-xs" style={{ color: "var(--ink-4)" }}>
          Last {weeks} weeks · {totalOverAll} tag{totalOverAll === 1 ? "" : "s"} fired
        </span>
        <Link
          href="/trader-mindset"
          className="ml-auto text-[11px] font-semibold hover:underline"
          style={{ color: navColor }}
        >
          Deep view →
        </Link>
      </div>
      <div className="px-[18px] py-3">
        {loading ? (
          <div className="text-[12px]" style={{ color: "var(--ink-4)" }}>
            Loading pattern data…
          </div>
        ) : topThree.length === 0 ? (
          <div className="text-[12px]" style={{ color: "var(--ink-4)" }}>
            No tagged behaviors yet — start checking chips on the Per-Ticker Details below
            and this strip will surface your recurring patterns week by week.
          </div>
        ) : (
          <div className="flex flex-wrap items-center gap-2">
            {topThree.map(t => {
              const accent = trapAccent(t.tag);
              return (
                <Link
                  key={t.tag}
                  href={`/trader-mindset?tag=${encodeURIComponent(t.tag)}`}
                  className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full cursor-pointer hover:brightness-95"
                  title={`Drill into "${t.tag}" on Trader Mindset`}
                  style={{
                    background: `color-mix(in oklab, ${accent} 14%, var(--surface))`,
                    border: `1px solid color-mix(in oklab, ${accent} 40%, var(--border))`,
                    textDecoration: "none",
                  }}
                >
                  <span
                    className="text-[12px] font-semibold"
                    style={{ color: accent, fontFamily: "var(--font-ui, inherit)" }}
                  >
                    {t.tag}
                  </span>
                  <span
                    className="text-[11px] font-semibold px-1.5 py-0.5 rounded-full"
                    style={{
                      background: accent,
                      color: "#fff",
                      fontFamily: "var(--font-jetbrains), monospace",
                    }}
                  >
                    {t.total_count}
                  </span>
                </Link>
              );
            })}
            {hasMore && (
              <span className="text-[11px]" style={{ color: "var(--ink-4)" }}>
                +{(traps || []).length - 3} more
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
