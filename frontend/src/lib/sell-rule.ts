// Active sell-rule tier classification — single source of truth for the
// peak-return ladder that routes a position into a defensive posture.
//
// Post-migration-063 ladder (2026-08-07 cleanup — SR14 retired, folded
// into SR1; broker-stop presence surfaced as a row chip instead of a
// tier promotion):
//
//   Peak `b1_return`                | Tier  | Stop / defense
//   --------------------------------|-------|--------------------------------
//   < 10%                           | SR1   | Composite stop (Entry − 1×ATR,
//                                   |       | structural low, key MA − max(
//                                   |       | 0.5 ATR, 1%)). Broker stop, if
//                                   |       | parked, shows as a 🛡 chip on
//                                   |       | the row — no tier promotion.
//   10% – 20%                       | SR11  | BE stop at entry price
//   20% – 50%                       | SR15  | Physical broker stop at
//                                   |       | entry × 1.10 (+10% profit lock)
//   ≥ 50% AND is_declared_sr8=false | SR7   | 21 EMA + cushion posture
//   ≥ 50% AND is_declared_sr8=true  | SR8   | Weekly MO RS + funnel ladder,
//                                   |       | activation-anchored core
//
// Sticky ratchet: once peak crosses a threshold, the tier is permanent
// (unless the user demotes SR8 → SR7 explicitly). A pullback doesn't
// downgrade the tier — the peak is the anchor.

export type SellRuleTier = "sr1" | "sr11" | "sr15" | "sr7" | "sr8";

export function classifySellRuleTier(
  b1ReturnPct: number | null | undefined,
  // brokerStopPrice retained in the signature for backwards-compat with
  // callers that still pass it — no longer read (SR14 was retired in
  // the 2026-08-07 cleanup; broker-stop presence is a row chip now).
  _brokerStopPrice?: number | null | undefined,
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

  // Below 10% — composite stop only. Broker-stop-parked state shows as
  // a chip on the row (rendered in ACS), not a distinct tier.
  return "sr1";
}

// Sort order used by the Sell Rule column header. Lower index sorts first.
// null sorts last regardless of direction (see compareRows).
//
// Order matches the ladder's defensive progression: SR1 (no floor) sorts
// earliest, SR8 (most-defended monster hold) sorts last.
export const SELL_RULE_TIER_ORDER: Record<SellRuleTier, number> = {
  sr1: 0,
  sr11: 1,
  sr15: 2,
  sr7: 3,
  sr8: 4,
};

// Human-friendly display label. Sorts as the raw tier code; use this only
// for tooltips and badge text.
export const SELL_RULE_TIER_LABEL: Record<SellRuleTier, string> = {
  sr1: "SR1",
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
// stop" when peak sits IN the SR15 band [20%, 50%) and the user's
// persisted `broker_stop_price` still sits below b1_entry × 1.10.
//
// Two guards that came out of the 2026-08-07 review:
//
//   1. Band-restricted, not band-and-above. Once peak crosses 50%,
//      the position is SR7 / SR8 and the +10% floor should ALREADY be
//      parked from the earlier band. Nagging at that point is stale
//      doctrine (DELL @ 166% peak declared SR8 shouldn't nudge).
//
//   2. Anchor is B1 fill price, not the blended avg_entry. The "+10%
//      profit lock" concept is denominated in the first-buy's cost
//      basis — that's the true "cushion crossed +20%" origin. Using
//      blended avg after add-ons would silently move the target price
//      higher every time you scale in (DELL bug: $331 avg × 1.10 =
//      $364 instead of $176 B1 × 1.10 = $193).
export function needsSR15StopMove(
  b1PeakPct: number | null | undefined,
  b1EntryPrice: number | null | undefined,
  brokerStopPrice: number | null | undefined,
): boolean {
  if (b1PeakPct == null || !Number.isFinite(b1PeakPct)) return false;
  // Band-restricted to SR15 [20%, 50%). Sticky ratchet across tiers
  // means once peak crosses 50 we're permanently past SR15's home.
  if (b1PeakPct < 20 || b1PeakPct >= 50) return false;
  if (b1EntryPrice == null || !Number.isFinite(b1EntryPrice) || b1EntryPrice <= 0) return false;
  const target = b1EntryPrice * 1.10;
  const current = brokerStopPrice != null && Number.isFinite(brokerStopPrice) ? brokerStopPrice : 0;
  // Half-a-penny tolerance for JS FP round-off (e.g. 100 × 1.10 =
  // 110.00000000000001). Nudge clears when the operator types the
  // "+10%" round number into broker_stop_price.
  return current < target - 0.005;
}

// SR12 Ratcheting Profit Floor (MCP) — the "give back no more than half
// what I've earned" doctrine. Migration 065 rewrote the anchor from
// sr12_floor_pct (B1-based) to peak_total_pl (the max total P&L the
// campaign ever showed). The 2026-08-07 DELL walkthrough exposed why
// the B1 anchor was wrong on scaled-in positions: a "half of B1 peak
// gain" stop can fire at a NET LOSS on the aggregate.
//
// Target broker stop derives from:
//   target = avg_entry + (peak_total_pl/2 − realized_bank) / shares
//
// The invariant that stops firing at `target` is
//   realized_bank + shares × (target − avg_entry) = peak_total_pl / 2
// i.e. "if this fires, my total P&L across the whole campaign lands at
// exactly half of its historical peak." Method-invariant on total P&L
// (avg-cost vs LIFO don't change the sum).
//
// Auto-clears when current realized_bank already exceeds peak_total_pl/2
// — you've locked in more than half by prior trims and there's nothing
// left for MCP to protect.
export function needsSR12FloorMove(
  peakTotalPl: number | null | undefined,
  realizedBank: number | null | undefined,
  shares: number | null | undefined,
  avgEntry: number | null | undefined,
  brokerStopPrice: number | null | undefined,
): boolean {
  if (peakTotalPl == null || !Number.isFinite(peakTotalPl) || peakTotalPl <= 0) return false;
  if (shares == null || !Number.isFinite(shares) || shares <= 0) return false;
  if (avgEntry == null || !Number.isFinite(avgEntry) || avgEntry <= 0) return false;
  const realized = realizedBank != null && Number.isFinite(realizedBank) ? realizedBank : 0;
  const targetRealized = peakTotalPl / 2;
  // Auto-clear: you've already banked more than half. Nothing to
  // enforce — a further stop would only take back cushion you own.
  if (realized >= targetRealized) return false;
  const target = avgEntry + (targetRealized - realized) / shares;
  const current = brokerStopPrice != null && Number.isFinite(brokerStopPrice) ? brokerStopPrice : 0;
  // Half-a-penny tolerance for JS FP round-off (same as needsSR15StopMove).
  return current < target - 0.005;
}

// Convenience: the exact broker-stop target that clears the nudge.
// Returns null when the campaign isn't armed (no peak_total_pl / no
// shares / already-banked past target).
export function computeSR12FloorTarget(
  peakTotalPl: number | null | undefined,
  realizedBank: number | null | undefined,
  shares: number | null | undefined,
  avgEntry: number | null | undefined,
): number | null {
  if (peakTotalPl == null || !Number.isFinite(peakTotalPl) || peakTotalPl <= 0) return null;
  if (shares == null || !Number.isFinite(shares) || shares <= 0) return null;
  if (avgEntry == null || !Number.isFinite(avgEntry) || avgEntry <= 0) return null;
  const realized = realizedBank != null && Number.isFinite(realizedBank) ? realizedBank : 0;
  const targetRealized = peakTotalPl / 2;
  if (realized >= targetRealized) return null;
  return avgEntry + (targetRealized - realized) / shares;
}
