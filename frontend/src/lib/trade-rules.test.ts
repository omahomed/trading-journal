import { describe, it, expect } from "vitest";
import { SELL_RULES, SELL_RULE_LABELS, RULE_HIERARCHY, BUY_RULE_LABELS } from "./trade-rules";

describe("SELL_RULES canonical taxonomy", () => {
  it("has exactly 16 entries (sr1..sr13 + sr8.1/sr8.2/sr8.3 cascade sub-rules)", () => {
    expect(SELL_RULES.length).toBe(16);
  });

  it("uses codes sr1..sr13 with sr8.1/sr8.2/sr8.3 inserted after sr8", () => {
    expect(SELL_RULES.map((r) => r.code)).toEqual([
      "sr1", "sr2", "sr3", "sr4", "sr5", "sr6", "sr7", "sr8",
      "sr8.1", "sr8.2", "sr8.3",
      "sr9", "sr10", "sr11", "sr12", "sr13",
    ]);
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
      "SR8 Quick Trim",
      "SR8 Quicksand Trim",
      "SR8 Dreadful Dead",
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
    expect(SELL_RULE_LABELS[SELL_RULE_LABELS.length - 1]).toBe("sr13 Change of Character");
    expect(SELL_RULE_LABELS).toContain("sr8.1 SR8 Quick Trim");
    expect(SELL_RULE_LABELS).toContain("sr8.2 SR8 Quicksand Trim");
    expect(SELL_RULE_LABELS).toContain("sr8.3 SR8 Dreadful Dead");
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

describe("SELL_RULES — glossary content fields", () => {
  it("every rule has a non-empty oneLiner", () => {
    for (const r of SELL_RULES) {
      expect(r.oneLiner, `rule ${r.code} missing oneLiner`).toBeTruthy();
      expect(r.oneLiner.length).toBeGreaterThan(20);
    }
  });

  it("sr4 Time Stop has no mechanics (single-rule, doesn't need a body)", () => {
    const sr4 = SELL_RULES.find((r) => r.code === "sr4");
    expect(sr4).toBeDefined();
    expect(sr4!.mechanics).toBeUndefined();
  });

  it("every rule except sr4 has mechanics (15 of 16)", () => {
    const withMechanics = SELL_RULES.filter((r) => r.mechanics);
    expect(withMechanics.length).toBe(15);
  });

  it("sr7 mechanics contains the cushion-tier GFM table", () => {
    const sr7 = SELL_RULES.find((r) => r.code === "sr7");
    expect(sr7!.mechanics).toContain("| Cushion at trigger | Action |");
    expect(sr7!.mechanics).toContain("Up <25% from entry");
    expect(sr7!.mechanics).toContain("Up >50% from entry");
  });

  it("sr8 mechanics contains the weekly MO RS trigger table", () => {
    const sr8 = SELL_RULES.find((r) => r.code === "sr8");
    expect(sr8!.mechanics).toContain("Quick");
    expect(sr8!.mechanics).toContain("Quicksand");
    expect(sr8!.mechanics).toContain("Grateful Dead");
  });
});

describe("BUY_RULE_LABELS canonical taxonomy", () => {
  it("is exported as a readonly array of strings (hoisted from 4 components)", () => {
    expect(Array.isArray(BUY_RULE_LABELS)).toBe(true);
    expect(BUY_RULE_LABELS.length).toBeGreaterThan(0);
    for (const label of BUY_RULE_LABELS) {
      expect(typeof label).toBe("string");
      expect(label.length).toBeGreaterThan(0);
    }
  });

  it("starts with br1.x base breakouts + includes br13.x MO RS Green pair", () => {
    expect(BUY_RULE_LABELS[0]).toBe("br1.1 Consolidation");
    expect(BUY_RULE_LABELS).toContain("br13.1 MO RS Green — Initial Entry");
    expect(BUY_RULE_LABELS).toContain("br13.2 MO RS Green — Reset Entry");
  });

  it("has no duplicate entries", () => {
    expect(new Set(BUY_RULE_LABELS).size).toBe(BUY_RULE_LABELS.length);
  });
});


describe("RULE_HIERARCHY", () => {
  it("has 6 entries", () => {
    expect(RULE_HIERARCHY.length).toBe(6);
  });

  it("every entry has conflict, winner, and reasoning", () => {
    for (const e of RULE_HIERARCHY) {
      expect(e.conflict).toBeTruthy();
      expect(e.winner).toBeTruthy();
      expect(e.reasoning).toBeTruthy();
    }
  });

  it("covers SR8 interaction with each layered rule", () => {
    const conflicts = RULE_HIERARCHY.map((e) => e.conflict).join("|");
    expect(conflicts).toContain("SR2 vs SR8");
    expect(conflicts).toContain("SR3 vs SR8");
    expect(conflicts).toContain("SR11 vs SR8");
    expect(conflicts).toContain("SR13 vs SR8");
    expect(conflicts).toContain("SR7 vs SR8");
  });
});
