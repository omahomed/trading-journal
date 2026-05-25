import type { TradeDetail } from "./api";
import type { EnrichedPosition } from "./positions";

/**
 * Mobile-only position-state classifier. Maps an open position to one
 * of four card-state buckets the mobile Trade Journal renders distinct
 * pills/footers for:
 *
 *   - 'call'     → equity option (instrument_type=OPTION)
 *   - 'ready'    → multi-BUY stock, last-add cushion >= pyramid trigger
 *   - 'added'    → multi-BUY stock, last-add cushion below trigger
 *   - 'original' → single-BUY stock (no scale-ins yet)
 *
 * No desktop analogue — desktop's "Pyramid Ready" column uses similar
 * logic inline but isn't worth extracting; mobile owns its own
 * classifier so the chip taxonomy can evolve independently.
 *
 * `pyramid_pct` on EnrichedPosition is already the last LIFO lot's
 * return % vs current price (computed by positions.ts using the same
 * LIFO walk this classifier would otherwise re-do). Using it directly
 * keeps the helper to a single source of truth.
 */
export type TradeState = "call" | "ready" | "added" | "original";

export type PyramidRules = {
  trigger_pct: number;
  alloc_pct: number;
};

export function classifyTradeState(
  enriched: EnrichedPosition,
  details: TradeDetail[],
  pyramidRules: PyramidRules,
): TradeState {
  if (enriched.is_option) return "call";

  const tradeBuys = details.filter(
    (d) =>
      d.trade_id === enriched.trade_id &&
      String(d.action || "").toUpperCase() === "BUY",
  );

  if (tradeBuys.length <= 1) return "original";

  // Multi-BUY: the last-add cushion is what positions.ts already
  // computed as pyramid_pct (current_price vs newest open LIFO lot).
  // Boundary is inclusive at trigger_pct — exact threshold reads as
  // ready, matching the "≥" convention used throughout the sizer.
  return enriched.pyramid_pct >= pyramidRules.trigger_pct ? "ready" : "added";
}
