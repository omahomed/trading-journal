import { describe, it, expect } from "vitest";
import { SELL_RULES, SELL_RULE_LABELS } from "./trade-rules";

describe("SELL_RULES canonical taxonomy", () => {
  it("has exactly 13 entries", () => {
    expect(SELL_RULES.length).toBe(13);
  });

  it("uses sequential sr1..sr13 codes in order", () => {
    const expected = Array.from({ length: 13 }, (_, i) => `sr${i + 1}`);
    expect(SELL_RULES.map((r) => r.code)).toEqual(expected);
  });

  it("matches the locked canonical descriptions", () => {
    expect(SELL_RULES.map((r) => r.description)).toEqual([
      "Capital Protection",
      "Selling into Strength",
      "Portfolio Management",
      "Time Stop",
      "Climax Top",
      "8e Momentum Trim",
      "Holding Winners - 21e Violation",
      "Big Cushion Sell Rule",
      "Failed Breakout",
      "Earnings Exit",
      "BE Stop Out (moved at +10%)",
      "TQQQ Strategy Exit",
      "Change of Character",
    ]);
  });

  it("contains no descriptions removed in the cleanup", () => {
    const labels = SELL_RULE_LABELS.join("|");
    for (const removed of [
      "Trailing Stop",
      "Exhaustion Gap",
      "200d Moving Avg Break",
      "Living Below 50d",
      "Scale-Out T1",
      "Scale-Out T2",
      "Scale-Out T3",
      "Market Correction Exit",
      "Profit Taking",
    ]) {
      expect(labels).not.toContain(removed);
    }
  });
});

describe("SELL_RULE_LABELS — DB string format", () => {
  it("formats each label as `${code} ${description}`", () => {
    expect(SELL_RULE_LABELS[0]).toBe("sr1 Capital Protection");
    expect(SELL_RULE_LABELS[12]).toBe("sr13 Change of Character");
  });

  it("has the same length as SELL_RULES", () => {
    expect(SELL_RULE_LABELS.length).toBe(SELL_RULES.length);
  });

  it("matches the DB-stored format used by the migration's canonical allowlist", () => {
    expect(SELL_RULE_LABELS).toContain("sr10 Earnings Exit");
    expect(SELL_RULE_LABELS).toContain("sr11 BE Stop Out (moved at +10%)");
    expect(SELL_RULE_LABELS).toContain("sr12 TQQQ Strategy Exit");
    expect(SELL_RULE_LABELS).toContain("sr9 Failed Breakout");
  });
});
