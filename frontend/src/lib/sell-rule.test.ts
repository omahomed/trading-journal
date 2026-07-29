import { describe, test, expect } from "vitest";
import { classifySellRuleTier, SELL_RULE_TIER_ORDER } from "./sell-rule";

describe("classifySellRuleTier", () => {
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

  test("classifies 10–50% return as sr11", () => {
    expect(classifySellRuleTier(10)).toBe("sr11");
    expect(classifySellRuleTier(10.01)).toBe("sr11");
    expect(classifySellRuleTier(25)).toBe("sr11");
    expect(classifySellRuleTier(49.99)).toBe("sr11");
  });

  test("classifies 50%+ return as sr8", () => {
    expect(classifySellRuleTier(50)).toBe("sr8");
    expect(classifySellRuleTier(50.01)).toBe("sr8");
    expect(classifySellRuleTier(120)).toBe("sr8");
  });

  test("tier order ranks sr1 < sr14 < sr11 < sr8", () => {
    // SR14 sits between SR1 and SR11 — same B1-return bucket as SR1 but
    // with a physical broker stop parked, so "one step further along in
    // defense." SR11 (BE stop, +10%+) still ranks above SR14.
    expect(SELL_RULE_TIER_ORDER.sr1).toBeLessThan(SELL_RULE_TIER_ORDER.sr14);
    expect(SELL_RULE_TIER_ORDER.sr14).toBeLessThan(SELL_RULE_TIER_ORDER.sr11);
    expect(SELL_RULE_TIER_ORDER.sr11).toBeLessThan(SELL_RULE_TIER_ORDER.sr8);
  });

  // Migration 055 — SR14 (0.75× ATR Stop) is the same B1-return band as
  // SR1 but promoted when the campaign has a physical broker_stop_price
  // set. The <10% branch reads the flag; higher bands (SR11 / SR8) are
  // unaffected because the broker stop is retired at +10% (BE stop takes
  // over) and irrelevant at +50% (cascade takes over).

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

  test("broker_stop_price does NOT override sr11 tier at +10%", () => {
    // BE stop replaces the broker stop at +10% — presence of broker_stop_price
    // is stale bookkeeping at that point, and the classifier ignores it.
    expect(classifySellRuleTier(15, 245.0)).toBe("sr11");
    expect(classifySellRuleTier(25, 245.0)).toBe("sr11");
  });

  test("broker_stop_price does NOT override sr8 tier at +50%", () => {
    // SR8 cascade takes over at +50%; broker stop is irrelevant.
    expect(classifySellRuleTier(55, 245.0)).toBe("sr8");
    expect(classifySellRuleTier(120, 245.0)).toBe("sr8");
  });

  test("backwards-compat: single-arg call still returns sr1 for <10%", () => {
    // Old callers that haven't been updated to pass broker_stop_price
    // must keep working — no broker stop means classic SR1.
    expect(classifySellRuleTier(5)).toBe("sr1");
    expect(classifySellRuleTier(9.99)).toBe("sr1");
  });
});
