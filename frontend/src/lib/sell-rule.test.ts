import { describe, test, expect } from "vitest";
import {
  classifySellRuleTier,
  SELL_RULE_TIER_ORDER,
  isCushionQualified,
  needsSR15StopMove,
} from "./sell-rule";

describe("classifySellRuleTier — post-migration-062 ladder", () => {
  test("returns null when b1 return is null/undefined/NaN/Infinity", () => {
    expect(classifySellRuleTier(null)).toBeNull();
    expect(classifySellRuleTier(undefined)).toBeNull();
    expect(classifySellRuleTier(NaN)).toBeNull();
    expect(classifySellRuleTier(Infinity)).toBeNull();
    expect(classifySellRuleTier(-Infinity)).toBeNull();
  });

  test("classifies sub-10% return as sr1", () => {
    expect(classifySellRuleTier(-50)).toBe("sr1");
    expect(classifySellRuleTier(-1)).toBe("sr1");
    expect(classifySellRuleTier(0)).toBe("sr1");
    expect(classifySellRuleTier(9.99)).toBe("sr1");
  });

  test("classifies 10%–20% return as sr11 (BE stop band)", () => {
    // Range tightened by migration 062 — was 10–50, now 10–20.
    expect(classifySellRuleTier(10)).toBe("sr11");
    expect(classifySellRuleTier(10.01)).toBe("sr11");
    expect(classifySellRuleTier(15)).toBe("sr11");
    expect(classifySellRuleTier(19.99)).toBe("sr11");
  });

  test("classifies 20%–50% return as sr15 (+10% profit-lock band)", () => {
    // NEW tier from migration 062 — physical broker stop at entry × 1.10.
    expect(classifySellRuleTier(20)).toBe("sr15");
    expect(classifySellRuleTier(20.01)).toBe("sr15");
    expect(classifySellRuleTier(35)).toBe("sr15");
    expect(classifySellRuleTier(49.99)).toBe("sr15");
  });

  test("classifies 50%+ return with declared=false as sr7", () => {
    // Cushion-qualified but undeclared — SR7 (21 EMA cascade).
    expect(classifySellRuleTier(50, null, false)).toBe("sr7");
    expect(classifySellRuleTier(50.01, null, false)).toBe("sr7");
    expect(classifySellRuleTier(120, null, false)).toBe("sr7");
    expect(classifySellRuleTier(300, null, undefined)).toBe("sr7");
    // Omitted flag defaults to undefined → SR7 (safe default).
    expect(classifySellRuleTier(80)).toBe("sr7");
  });

  test("classifies 50%+ return with declared=true as sr8", () => {
    // Declared monster hold — SR8 (weekly MO RS cascade + funnel ladder).
    expect(classifySellRuleTier(50, null, true)).toBe("sr8");
    expect(classifySellRuleTier(120, null, true)).toBe("sr8");
    expect(classifySellRuleTier(300, null, true)).toBe("sr8");
  });

  test("is_declared_sr8 is ignored in sub-50% bands (guard rail)", () => {
    // Declaration is only meaningful when cushion-qualified. Passing
    // is_declared_sr8=true on a sub-qualified campaign must NOT
    // promote it to SR8 — the backend guards the write, but the
    // classifier is defense-in-depth.
    expect(classifySellRuleTier(5, null, true)).toBe("sr1");
    expect(classifySellRuleTier(15, null, true)).toBe("sr11");
    expect(classifySellRuleTier(30, null, true)).toBe("sr15");
  });

  test("tier order ranks sr1 < sr14 < sr11 < sr15 < sr7 < sr8", () => {
    // Defensive-progression ladder — lower rank = closer to open risk,
    // higher rank = more entrenched/defended.
    expect(SELL_RULE_TIER_ORDER.sr1).toBeLessThan(SELL_RULE_TIER_ORDER.sr14);
    expect(SELL_RULE_TIER_ORDER.sr14).toBeLessThan(SELL_RULE_TIER_ORDER.sr11);
    expect(SELL_RULE_TIER_ORDER.sr11).toBeLessThan(SELL_RULE_TIER_ORDER.sr15);
    expect(SELL_RULE_TIER_ORDER.sr15).toBeLessThan(SELL_RULE_TIER_ORDER.sr7);
    expect(SELL_RULE_TIER_ORDER.sr7).toBeLessThan(SELL_RULE_TIER_ORDER.sr8);
  });

  // SR14 (migration 055) — same B1-return band as SR1 but promoted when
  // the campaign has a physical broker_stop_price set. The <10% branch
  // reads the flag; higher bands (SR11/SR15/SR7/SR8) are unaffected
  // because higher-tier stops replace the broker stop bookkeeping.

  test("promotes to sr14 when broker_stop_price is set and return < 10%", () => {
    expect(classifySellRuleTier(-5, 245.0)).toBe("sr14");
    expect(classifySellRuleTier(0, 100.0)).toBe("sr14");
    expect(classifySellRuleTier(9.99, 250.5)).toBe("sr14");
  });

  test("stays sr1 when broker_stop_price is null/undefined/zero/negative", () => {
    expect(classifySellRuleTier(5, null)).toBe("sr1");
    expect(classifySellRuleTier(5, undefined)).toBe("sr1");
    expect(classifySellRuleTier(5, 0)).toBe("sr1");
    expect(classifySellRuleTier(5, -10)).toBe("sr1");
    expect(classifySellRuleTier(5, NaN)).toBe("sr1");
  });

  test("broker_stop_price does NOT override higher tiers", () => {
    // BE stop / SR15 stop / cascade take over as the campaign matures;
    // the broker_stop field is stale bookkeeping at that point and the
    // classifier ignores it.
    expect(classifySellRuleTier(15, 245.0)).toBe("sr11");
    expect(classifySellRuleTier(30, 245.0)).toBe("sr15");
    expect(classifySellRuleTier(55, 245.0, false)).toBe("sr7");
    expect(classifySellRuleTier(120, 245.0, true)).toBe("sr8");
  });

  test("backwards-compat: single-arg call still returns sr1 for <10%", () => {
    // Old callers that haven't been updated to pass broker_stop_price /
    // is_declared_sr8 must keep working.
    expect(classifySellRuleTier(5)).toBe("sr1");
    expect(classifySellRuleTier(9.99)).toBe("sr1");
    expect(classifySellRuleTier(15)).toBe("sr11");
    expect(classifySellRuleTier(30)).toBe("sr15");
    expect(classifySellRuleTier(80)).toBe("sr7");
  });
});

describe("isCushionQualified", () => {
  test("returns false for null/undefined/NaN/Infinity", () => {
    expect(isCushionQualified(null)).toBe(false);
    expect(isCushionQualified(undefined)).toBe(false);
    expect(isCushionQualified(NaN)).toBe(false);
    expect(isCushionQualified(Infinity)).toBe(false);
  });

  test("returns false below 50", () => {
    expect(isCushionQualified(0)).toBe(false);
    expect(isCushionQualified(49.99)).toBe(false);
  });

  test("returns true at or above 50", () => {
    expect(isCushionQualified(50)).toBe(true);
    expect(isCushionQualified(50.01)).toBe(true);
    expect(isCushionQualified(300)).toBe(true);
  });
});

describe("needsSR15StopMove", () => {
  test("false below the 20% threshold", () => {
    expect(needsSR15StopMove(0, 100, null)).toBe(false);
    expect(needsSR15StopMove(19.99, 100, null)).toBe(false);
  });

  test("true when in SR15+ band with no broker stop parked", () => {
    // Entry 100 → target 110. No stop = definitely below target.
    expect(needsSR15StopMove(20, 100, null)).toBe(true);
    expect(needsSR15StopMove(25, 100, 0)).toBe(true);
    expect(needsSR15StopMove(80, 100, undefined)).toBe(true);
  });

  test("true when broker stop parked but still below target", () => {
    // Entry 100 → target 110. Stop at 105 = still needs a nudge.
    expect(needsSR15StopMove(30, 100, 105)).toBe(true);
    expect(needsSR15StopMove(60, 100, 109.99)).toBe(true);
  });

  test("false when broker stop is at or above the target", () => {
    // Entry 100 → target 110. Stop at 110 = target met.
    expect(needsSR15StopMove(30, 100, 110)).toBe(false);
    expect(needsSR15StopMove(60, 100, 115)).toBe(false);
    // Even for a mature SR8-tier campaign, if the +10% floor is
    // already parked, no nudge fires.
    expect(needsSR15StopMove(200, 100, 111)).toBe(false);
  });

  test("false when entry price is missing/invalid", () => {
    // Can't compute the target without a positive entry price.
    expect(needsSR15StopMove(30, 0, null)).toBe(false);
    expect(needsSR15StopMove(30, -1, null)).toBe(false);
    expect(needsSR15StopMove(30, null, null)).toBe(false);
    expect(needsSR15StopMove(30, NaN, null)).toBe(false);
  });
});
