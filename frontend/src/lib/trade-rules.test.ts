import { describe, it, expect } from "vitest";
import { SELL_RULES, SELL_RULE_LABELS, RULE_HIERARCHY, BUY_RULE_LABELS } from "./trade-rules";

// Post-migration-063 (2026-08-07 cleanup): SR4, SR6, SR12, SR14 retired
// per the canonical MOTrading Handoff review notes.
//   - SR4 (Time Stop): SR3 covers portfolio-time trims.
//   - SR6 (8e Momentum Trim): 0-for-5 fire quality.
//   - SR12 (TQQQ Strategy Exit): same 21 EMA shape as SR7 on NDX;
//     historical stamps retagged to SR7.
//   - SR14 (0.75× ATR Stop): folded into SR1; broker-stop presence is
//     a chip on the row, not a tier.
//   - SR7 description shortened to "21e Violation".

describe("SELL_RULES canonical taxonomy — post-063", () => {
  it("has exactly 13 entries (17 pre-cleanup minus 4 retired)", () => {
    expect(SELL_RULES.length).toBe(13);
  });

  it("uses the post-cleanup code sequence (SR4/SR6/SR12/SR14 removed)", () => {
    expect(SELL_RULES.map((r) => r.code)).toEqual([
      "sr1", "sr2", "sr3", "sr5", "sr7", "sr8",
      "sr8.1", "sr8.2", "sr8.3",
      "sr9", "sr10", "sr11", "sr13",
    ]);
  });

  it("matches the locked canonical descriptions", () => {
    expect(SELL_RULES.map((r) => r.description)).toEqual([
      "Capital Protection",
      "Selling into Strength",
      "Portfolio Management",
      "Climax Top",
      "21e Violation",
      "Big Cushion Sell Rule",
      "SR8 Quick Trim",
      "SR8 Quicksand Trim",
      "SR8 Grateful Dead",
      "Failed Breakout",
      "Earnings Exit",
      "BE Stop Out (moved at +10%)",
      "Change of Character",
    ]);
  });

  it("no longer contains any retired codes", () => {
    const codes = SELL_RULES.map((r) => r.code);
    for (const retired of ["sr4", "sr6", "sr12", "sr14"]) {
      expect(codes).not.toContain(retired);
    }
  });

  it("SR8.2 mechanics reference the 13w MA (Fibonacci mid line)", () => {
    // 2026-08-07 doc fix: the glossary had said "drifts further below
    // 8w"; the actual MORS engine fires Quicksand on 13w break per
    // DEFAULT_WEEKLY_EMAS = (8, 13, 21). This test locks the fix.
    const sr82 = SELL_RULES.find((r) => r.code === "sr8.2");
    expect(sr82).toBeDefined();
    expect(sr82!.mechanics).toContain("13w");
    expect(sr82!.mechanics).not.toContain("drifts further");
  });
});

describe("SELL_RULE_LABELS — DB string format", () => {
  it("formats each label as `${code} ${description}`", () => {
    expect(SELL_RULE_LABELS[0]).toBe("sr1 Capital Protection");
    expect(SELL_RULE_LABELS[SELL_RULE_LABELS.length - 1]).toBe("sr13 Change of Character");
    expect(SELL_RULE_LABELS).toContain("sr8.1 SR8 Quick Trim");
    expect(SELL_RULE_LABELS).toContain("sr8.2 SR8 Quicksand Trim");
    expect(SELL_RULE_LABELS).toContain("sr8.3 SR8 Grateful Dead");
    expect(SELL_RULE_LABELS).toContain("sr7 21e Violation");
  });

  it("has the same length as SELL_RULES", () => {
    expect(SELL_RULE_LABELS.length).toBe(SELL_RULES.length);
  });

  it("does not include any retired-code labels", () => {
    for (const retired of ["sr4", "sr6", "sr12", "sr14"]) {
      const hit = SELL_RULE_LABELS.find((l) => l.startsWith(retired + " "));
      expect(hit, `retired code ${retired} leaked into SELL_RULE_LABELS`).toBeUndefined();
    }
  });
});

describe("SELL_RULES — glossary content fields", () => {
  it("every rule has a non-empty oneLiner", () => {
    for (const r of SELL_RULES) {
      expect(r.oneLiner, `rule ${r.code} missing oneLiner`).toBeTruthy();
      expect(r.oneLiner.length).toBeGreaterThan(20);
    }
  });

  it("every rule has mechanics after the cleanup", () => {
    // SR4 was the only rule without mechanics pre-cleanup; retiring it
    // leaves every remaining rule with a mechanics body.
    const withMechanics = SELL_RULES.filter((r) => r.mechanics);
    expect(withMechanics.length).toBe(SELL_RULES.length);
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
