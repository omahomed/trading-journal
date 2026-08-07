// SR15 nudge banner (Migration 062).
//
// Renders when any open position has crossed +20% peak (SR15 territory)
// but its persisted broker_stop_price is still below entry × 1.10.
// Doctrine wants a physical +10% profit-lock stop parked at the broker;
// the app nudges until the user updates broker_stop_price to match.
//
// Auto-clears row-by-row as each stop lands at or above target. The
// "dismiss action" per the design is the user setting the stop — no
// separate acknowledgment button.
//
// Rendered on ACS (with clickable ticker chips that open the broker-
// stop editor) and on Risk Manager (non-clickable chips — informational).

"use client";

import { formatCurrency } from "@/lib/format";
import { needsSR15StopMove } from "@/lib/sell-rule";
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

export function SR15NudgeBanner({ positions, onTickerClick, className }: Props) {
  const nudges: Nudge[] = positions
    .filter(p =>
      needsSR15StopMove(
        p.b1_max_return_pct ?? p.b1_return_pct,
        p.avg_entry,
        p.broker_stop_price ?? null,
      )
    )
    .map(p => ({
      trade_id: p.trade_id,
      ticker: p.ticker,
      target: p.avg_entry * 1.10,
      current: p.broker_stop_price ?? null,
      pos: p,
    }));

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
        background: "color-mix(in oklab, #0891b2 10%, var(--surface))",
        border: "1px solid color-mix(in oklab, #0891b2 30%, var(--border))",
        color: "var(--ink-2)",
      }}
      data-testid="sr15-nudge-banner"
    >
      <div className="flex items-start gap-3">
        <span style={{ color: "#0e7490", fontSize: 18, lineHeight: "18px" }}>
          {"\u{1F6E1}️"}
        </span>
        <div className="flex-1">
          <div className="text-[13px] font-semibold" style={{ color: "#0e7490" }}>
            Move broker stop to +10% profit (SR15) on {nudges.length} position
            {nudges.length === 1 ? "" : "s"}
          </div>
          <div className="text-[12px] mt-1" style={{ color: "var(--ink-3)" }}>
            Peak has crossed +20% — physical broker stop should park at entry × 1.10.
            Set it at your broker, then update it here to clear the nudge.
          </div>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {nudges.map(n => {
              const tooltip = `Target ${formatCurrency(n.target)}${
                n.current != null ? ` · current ${formatCurrency(n.current)}` : " · unset"
              }`;
              const inner = (
                <>
                  <span style={{ color: "#0e7490" }}>{n.ticker}</span>
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
