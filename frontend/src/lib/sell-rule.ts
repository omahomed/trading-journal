// Active sell-rule tier classification — single source of truth for the
// peak-return ladder that routes a position into a defensive posture.
//
// Migration 062 (2026-08-07) reshaped the ladder into a peak-based
// one-way ratchet driven by two axes:
//
//   1. Peak B1 return (b1_max_return_pct)
//   2. User-declared SR8 flag (is_declared_sr8, from trades_summary)
//
//   Peak `b1_return`                | Tier  | Stop / defense
//   --------------------------------|-------|--------------------------------
//   < 10%                           | SR1   | Composite stop (Entry − 1×ATR,
//                                   |       | structural low, key MA − max(
//                                   |       | 0.5 ATR, 1%))
//   < 10% AND broker_stop_price set | SR14  | Physical broker stop (two-stop
//                                   |       | model; migration 055)
//   10% – 20%                       | SR11  | BE stop at entry price
//   20% – 50%                       | SR15  | Physical broker stop at
//                                   |       | entry × 1.10 (+10% profit lock)
//   ≥ 50% AND is_declared_sr8=false | SR7   | 21 EMA + cushion cascade
//   ≥ 50% AND is_declared_sr8=true  | SR8   | Weekly MO RS + funnel ladder,
//                                   |       | activation-anchored core
//
// Sticky ratchet: once peak crosses a threshold, the tier is permanent
// (unless the user demotes SR8 → SR7 explicitly). A pullback doesn't
// downgrade the tier — the peak is the anchor.

export type SellRuleTier = "sr1" | "sr14" | "sr11" | "sr15" | "sr7" | "sr8";

export function classifySellRuleTier(
  b1ReturnPct: number | null | undefined,
  brokerStopPrice?: number | null | undefined,
  isDeclaredSr8?: boolean | null | undefined,
): SellRuleTier | null {
  if (b1ReturnPct == null || !Number.isFinite(b1ReturnPct)) return null;

  // Cushion-qualified band splits on user declaration.
  if (b1ReturnPct >= 50) {
    return isDeclaredSr8 ? "sr8" : "sr7";
  }

  // Mid band: 20% – 50% locks the +10% profit floor via broker stop.
  if (b1ReturnPct >= 20) return "sr15";

  // Early band: 10% – 20% is BE-stop territory.
  if (b1ReturnPct >= 10) return "sr11";

  // Below 10% — composite stop only, unless a broker stop is already parked
  // (SR14 two-stop model). Broker-stop presence is the flag; no separate bool.
  const hasBrokerStop =
    brokerStopPrice != null
    && Number.isFinite(brokerStopPrice)
    && brokerStopPrice > 0;
  return hasBrokerStop ? "sr14" : "sr1";
}

// Sort order used by the Sell Rule column header. Lower index sorts first.
// null sorts last regardless of direction (see compareRows).
//
// Order matches the ladder's defensive progression: SR1 (no floor) sorts
// earliest, SR8 (most-defended monster hold) sorts last. SR14 sits
// between SR1 and SR11 — same B1-return bucket as SR1 but with an active
// physical stop, so it's "one step further along in defense" than SR1.
export const SELL_RULE_TIER_ORDER: Record<SellRuleTier, number> = {
  sr1: 0,
  sr14: 1,
  sr11: 2,
  sr15: 3,
  sr7: 4,
  sr8: 5,
};

// Human-friendly display label. Sorts as the raw tier code; use this only
// for tooltips and badge text.
export const SELL_RULE_TIER_LABEL: Record<SellRuleTier, string> = {
  sr1: "SR1",
  sr14: "SR14",
  sr11: "SR11",
  sr15: "SR15",
  sr7: "SR7",
  sr8: "SR8",
};

// Cushion-qualified predicate — same test as the SR8 backend guard.
// A campaign must be cushion-qualified before it can be declared SR8;
// the frontend uses this to enable/disable the right-click Declare
// menu item without a round-trip to the backend.
export function isCushionQualified(b1ReturnPct: number | null | undefined): boolean {
  if (b1ReturnPct == null || !Number.isFinite(b1ReturnPct)) return false;
  return b1ReturnPct >= 50;
}

// SR15 nudge predicate — a position is "waiting for the +10% broker
// stop" when the peak has entered the SR15 band (or above) but the
// user's persisted `broker_stop_price` still sits below entry × 1.10.
// Used by ACS + Risk Manager to render the persistent nudge banner.
export function needsSR15StopMove(
  b1ReturnPct: number | null | undefined,
  entryPrice: number | null | undefined,
  brokerStopPrice: number | null | undefined,
): boolean {
  if (b1ReturnPct == null || !Number.isFinite(b1ReturnPct) || b1ReturnPct < 20) return false;
  if (entryPrice == null || !Number.isFinite(entryPrice) || entryPrice <= 0) return false;
  const target = entryPrice * 1.10;
  const current = brokerStopPrice != null && Number.isFinite(brokerStopPrice) ? brokerStopPrice : 0;
  // Half-a-penny tolerance to swallow floating-point round-off (e.g.
  // 100 × 1.10 = 110.00000000000001 in JS). At real-money precision
  // the nudge is meant to clear the instant the operator types the
  // "+10%" round number into broker_stop_price.
  return current < target - 0.005;
}
