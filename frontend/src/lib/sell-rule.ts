// Active sell-rule tier classification — single source of truth for the
// 10% / 50% boundaries that route a position to SR1 (Capital Protection),
// SR11 (BE Stop Out), or SR8 (Big Cushion). Classification is based on
// B1's (first BUY) return %, not the position's average return.

export type SellRuleTier = "sr1" | "sr11" | "sr8";

export function classifySellRuleTier(b1ReturnPct: number | null | undefined): SellRuleTier | null {
  if (b1ReturnPct == null || !Number.isFinite(b1ReturnPct)) return null;
  if (b1ReturnPct < 10) return "sr1";
  if (b1ReturnPct < 50) return "sr11";
  return "sr8";
}

// Sort order used by the Sell Rule column header. Lower index sorts first.
// null sorts last regardless of direction (see compareRows).
export const SELL_RULE_TIER_ORDER: Record<SellRuleTier, number> = {
  sr1: 0,
  sr11: 1,
  sr8: 2,
};
