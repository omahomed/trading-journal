// SR12 Ratcheting Profit Floor nudge banner (Migration 064).
//
// The disaster backstop for gap-down mornings that beat SR7/SR8 to a
// worse price. Once a campaign's peak b1_return crosses +50%, the
// b1_reconcile loop ratchets sr12_floor_pct up to peak / 2 and it never
// moves down. This banner nudges until the physical broker_stop_price
// lands at or above b1_entry × (1 + sr12_floor_pct / 100).
//
// Clean handoff with SR15: SR15's predicate is band-restricted [20, 50);
// this one takes over from 50 up. A single campaign never appears in
// both banners.
//
// Same structural pattern as SR15NudgeBanner — clickable ticker chips
// on ACS (open the broker-stop editor), inert chips on Risk Manager.
// The "dismiss action" is the user updating the stop; no acknowledge
// button.

"use client";

import { formatCurrency } from "@/lib/format";
import { needsSR12FloorMove, computeSR12FloorTarget } from "@/lib/sell-rule";
import type { EnrichedPosition } from "@/lib/positions";

interface Props {
  positions: EnrichedPosition[];
  onTickerClick?: (position: EnrichedPosition) => void;
  className?: string;
}

interface Nudge {
  trade_id: string;
  ticker: string;
  target: number;
  current: number | null;
  pos: EnrichedPosition;
}

export function SR12FloorNudgeBanner({ positions, onTickerClick, className }: Props) {
  const nudges: Nudge[] = positions
    .filter(p =>
      needsSR12FloorMove(
        p.b1_max_return_pct ?? p.b1_return_pct,
        p.b1_entry_price,
        p.broker_stop_price ?? null,
        p.sr12_floor_pct,
      )
    )
    .map(p => ({
      trade_id: p.trade_id,
      ticker: p.ticker,
      target: computeSR12FloorTarget(
        p.b1_max_return_pct ?? p.b1_return_pct,
        p.b1_entry_price,
        p.sr12_floor_pct,
      ) ?? 0,
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

  // Amber floor-family palette. Matches SELL_RULE_FAMILIES.floor color
  // (#d97706) so the banner reads as the same visual family as the
  // "Profit floor" glossary group + badge stripe.
  return (
    <div
      className={`px-4 py-3 rounded-[10px] ${className ?? "mb-4"}`}
      style={{
        background: "color-mix(in oklab, #d97706 10%, var(--surface))",
        border: "1px solid color-mix(in oklab, #d97706 30%, var(--border))",
        color: "var(--ink-2)",
      }}
      data-testid="sr12-floor-nudge-banner"
    >
      <div className="flex items-start gap-3">
        <span style={{ color: "#b45309", fontSize: 18, lineHeight: "18px" }}>
          {"⚓"}
        </span>
        <div className="flex-1">
          <div className="text-[13px] font-semibold" style={{ color: "#b45309" }}>
            Ratchet broker stop to profit floor (SR12) on {nudges.length} position
            {nudges.length === 1 ? "" : "s"}
          </div>
          <div className="text-[12px] mt-1" style={{ color: "var(--ink-3)" }}>
            Peak crossed +50% — MCP doctrine parks a physical stop at half the
            peak gain. Set it at your broker, then update it here to clear.
          </div>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {nudges.map(n => {
              const tooltip = `Target ${formatCurrency(n.target)}${
                n.current != null ? ` · current ${formatCurrency(n.current)}` : " · unset"
              }`;
              const inner = (
                <>
                  <span style={{ color: "#b45309" }}>{n.ticker}</span>
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
