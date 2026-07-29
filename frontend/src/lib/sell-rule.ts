// Active sell-rule tier classification — single source of truth for the
// 10% / 50% boundaries that route a position to SR1 (Capital Protection),
// SR11 (BE Stop Out), SR8 (Big Cushion), or SR14 (0.75× ATR Stop, the
// two-stop model's first-line broker exit).
//
// Classification is based on B1's (first BUY) return %, not the position's
// average return. SR14 promotion happens in the <10% window when the
// campaign has a `broker_stop_price` set (migration 055) — presence of
// the price IS the flag; no separate boolean needed. Above 10%, the BE
// stop replaces the broker stop and the tier moves to SR11 regardless.

export type SellRuleTier = "sr1" | "sr11" | "sr8" | "sr14";

export function classifySellRuleTier(
  b1ReturnPct: number | null | undefined,
  brokerStopPrice?: number | null | undefined,
): SellRuleTier | null {
  if (b1ReturnPct == null || !Number.isFinite(b1ReturnPct)) return null;
  if (b1ReturnPct < 10) {
    // Two-stop model check — presence of broker_stop_price > 0 means the
    // physical broker stop is parked and will fire first if hit.
    const hasBrokerStop =
      brokerStopPrice != null
      && Number.isFinite(brokerStopPrice)
      && brokerStopPrice > 0;
    return hasBrokerStop ? "sr14" : "sr1";
  }
  if (b1ReturnPct < 50) return "sr11";
  return "sr8";
}

// Sort order used by the Sell Rule column header. Lower index sorts first.
// null sorts last regardless of direction (see compareRows). SR14 sits
// between SR1 and SR11 — same B1-return bucket as SR1 but with an active
// physical stop, so it's "one step further along in defense" than SR1.
export const SELL_RULE_TIER_ORDER: Record<SellRuleTier, number> = {
  sr1: 0,
  sr14: 1,
  sr11: 2,
  sr8: 3,
};
