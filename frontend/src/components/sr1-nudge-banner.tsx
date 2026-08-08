// SR1 broker-stop nudge banner (2026-08-07 follow-on to migration 062).
//
// Renders when any open SR1-tier position (peak b1_return < 10%) has a
// missing or too-loose broker stop. Doctrine: park a physical stop at
// B1 fill − 0.75 × ATR21_at_fill. The 0.75× line is the "trades that
// breach it intraday post-fill show ~0% win rate" premise the retired
// SR14 doctrine calibrated.
//
// Auto-clears row-by-row as each stop lands at or above target. The
// "dismiss action" is the user setting the stop in the app to match
// (or tighter than) the calibrated line — same UX as the SR15 nudge.
//
// Rendered on ACS (with clickable ticker chips that open the broker-
// stop editor) and on Risk Manager (non-clickable chips — informational).

"use client";

import { formatCurrency } from "@/lib/format";
import { computeSR1StopTarget, needsSR1StopMove } from "@/lib/sell-rule";
import type { EnrichedPosition } from "@/lib/positions";

interface Props {
  positions: EnrichedPosition[];
  /** Optional click handler — when provided, ticker chips become
   *  buttons that fire this callback (ACS case). When absent, chips
   *  render as inert spans (Risk Manager case). */
  onTickerClick?: (position: EnrichedPosition) => void;
  /** Optional className override, e.g. margin adjustments. */
  className?: string;
}

interface Nudge {
  trade_id: string;
  ticker: string;
  target: number;
  current: number | null;
  pos: EnrichedPosition;
}

export function SR1NudgeBanner({ positions, onTickerClick, className }: Props) {
  const nudges: Nudge[] = positions
    .filter(p =>
      needsSR1StopMove(
        p.b1_max_return_pct ?? p.b1_return_pct,
        p.b1_entry_price,
        p.atr21_entry_pct,
        p.broker_stop_price ?? null,
      )
    )
    .map(p => {
      // computeSR1StopTarget returned non-null (the predicate above
      // wouldn't have passed otherwise). Coalesce to satisfy the
      // narrower type at this call site.
      const target = computeSR1StopTarget(p.b1_entry_price, p.atr21_entry_pct) ?? 0;
      return {
        trade_id: p.trade_id,
        ticker: p.ticker,
        target,
        current: p.broker_stop_price ?? null,
        pos: p,
      };
    });

  if (nudges.length === 0) return null;

  const chipStyle = {
    background: "var(--surface)",
    border: "1px solid var(--border)",
    color: "var(--ink-2)",
    fontFamily: "var(--font-jetbrains), monospace",
  } as const;

  return (
    <div
      className={`px-4 py-3 rounded-[10px] ${className ?? "mb-4"}`}
      style={{
        background: "color-mix(in oklab, #e5484d 10%, var(--surface))",
        border: "1px solid color-mix(in oklab, #e5484d 30%, var(--border))",
        color: "var(--ink-2)",
      }}
      data-testid="sr1-nudge-banner"
    >
      <div className="flex items-start gap-3">
        <span style={{ color: "#dc2626", fontSize: 18, lineHeight: "18px" }}>
          {"\u{1F6E1}️"}
        </span>
        <div className="flex-1">
          <div className="text-[13px] font-semibold" style={{ color: "#dc2626" }}>
            Set 0.75× ATR21 broker stop (SR1) on {nudges.length} position
            {nudges.length === 1 ? "" : "s"}
          </div>
          <div className="text-[12px] mt-1" style={{ color: "var(--ink-3)" }}>
            Doctrine: physical broker stop parks at B1 fill − 0.75 × ATR21.
            Set it at your broker, then update it here to clear the nudge.
          </div>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {nudges.map(n => {
              const tooltip = `Target ${formatCurrency(n.target)}${
                n.current != null ? ` · current ${formatCurrency(n.current)}` : " · unset"
              }`;
              const inner = (
                <>
                  <span style={{ color: "#dc2626" }}>{n.ticker}</span>
                  <span style={{ color: "var(--ink-3)" }}>
                    {" → "}
                    {formatCurrency(n.target)}
                  </span>
                </>
              );
              const commonClass =
                "inline-flex items-center gap-1.5 px-2 py-1 rounded-[8px] text-[11px] font-medium";
              return onTickerClick ? (
                <button
                  key={n.trade_id}
                  onClick={() => onTickerClick(n.pos)}
                  className={commonClass}
                  style={chipStyle}
                  title={tooltip}
                >
                  {inner}
                </button>
              ) : (
                <span
                  key={n.trade_id}
                  className={commonClass}
                  style={chipStyle}
                  title={tooltip}
                >
                  {inner}
                </span>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
