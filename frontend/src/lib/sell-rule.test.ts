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

  test("tier order ranks sr1 < sr11 < sr8", () => {
    expect(SELL_RULE_TIER_ORDER.sr1).toBeLessThan(SELL_RULE_TIER_ORDER.sr11);
    expect(SELL_RULE_TIER_ORDER.sr11).toBeLessThan(SELL_RULE_TIER_ORDER.sr8);
  });
});
