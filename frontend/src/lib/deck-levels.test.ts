import { describe, test, expect } from "vitest";
import { classifyDeck, HARD_DECKS, DECK_META } from "./deck-levels";

describe("classifyDeck", () => {
  test("null / undefined / NaN → L0 (defensive)", () => {
    expect(classifyDeck(null)).toBe("L0");
    expect(classifyDeck(undefined)).toBe("L0");
    expect(classifyDeck(NaN)).toBe("L0");
  });

  test("above peak (positive or zero) → L0", () => {
    expect(classifyDeck(0)).toBe("L0");
    expect(classifyDeck(-0.1)).toBe("L0");
    expect(classifyDeck(-5.0)).toBe("L0");
    expect(classifyDeck(-7.49)).toBe("L0");
  });

  test("crosses L1 at 7.5% drawdown magnitude", () => {
    expect(classifyDeck(-7.5)).toBe("L1");
    expect(classifyDeck(-10.0)).toBe("L1");
    expect(classifyDeck(-12.49)).toBe("L1");
  });

  test("crosses L2 at 12.5%", () => {
    expect(classifyDeck(-12.5)).toBe("L2");
    expect(classifyDeck(-14.99)).toBe("L2");
  });

  test("crosses L3 at 15% — worst bucket", () => {
    expect(classifyDeck(-15.0)).toBe("L3");
    expect(classifyDeck(-25.0)).toBe("L3");
    expect(classifyDeck(-100.0)).toBe("L3");
  });

  test("accepts positive magnitude too (Risk Manager convention)", () => {
    // Risk Manager passes `(peak − current) / peak × 100` — always positive
    // when in drawdown. classifyDeck's abs() makes the two callers agree.
    expect(classifyDeck(7.5)).toBe("L1");
    expect(classifyDeck(12.5)).toBe("L2");
    expect(classifyDeck(15.0)).toBe("L3");
  });
});

describe("HARD_DECKS constants", () => {
  test("thresholds match the doctrine (7.5 / 12.5 / 15)", () => {
    // Locked. Changing any of these here without a doctrine change is a bug —
    // Risk Manager reads the same array and its status/gradient logic keys off
    // the exact percentages.
    expect(HARD_DECKS.map(d => d.pct)).toEqual([7.5, 12.5, 15.0]);
    expect(HARD_DECKS.map(d => d.key)).toEqual(["L1", "L2", "L3"]);
  });

  test("DECK_META covers every level with label + sub + color", () => {
    for (const lvl of ["L0", "L1", "L2", "L3"] as const) {
      const m = DECK_META[lvl];
      expect(m.label).toBeTruthy();
      expect(m.sub).toBeTruthy();
      expect(m.color).toMatch(/^#[0-9a-f]{6}$/i);
    }
  });
});
