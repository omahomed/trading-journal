import { describe, test, expect } from "vitest";
import {
  classifyDeck, HARD_DECKS, DECK_META,
  L_SERIES, L_SERIES_META, LEGACY_HARD_DECKS, LEGACY_DECK_META,
  classifyLegacyDeck,
} from "./deck-levels";

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

// ── Migration 068 — L-series (new) ────────────────────────────────
describe("L_SERIES — new cycle-anchored + IXIC-structural levels", () => {
  test("four levels, ordered shallow → deep, caps 80/60/40/20", () => {
    // Locked. Caps come from the doctrine — Risk Manager renders each
    // level's cap % in the pill; changes here without a doctrine change
    // silently shift the exposure governor.
    expect(L_SERIES.map(l => l.key)).toEqual(["L1", "L2", "L3", "L4"]);
    expect(L_SERIES.map(l => l.cap_pct)).toEqual([80, 60, 40, 20]);
  });

  test("every level has an action + trigger string", () => {
    for (const lvl of L_SERIES) {
      expect(lvl.action).toBeTruthy();
      expect(lvl.trigger).toBeTruthy();
      expect(lvl.color).toMatch(/^#[0-9a-f]{6}$/i);
    }
  });

  test("L_SERIES_META covers every level (incl. L0 and new L4)", () => {
    for (const lvl of ["L0", "L1", "L2", "L3", "L4"] as const) {
      const m = L_SERIES_META[lvl];
      expect(m.label).toBeTruthy();
      expect(m.sub).toBeTruthy();
      expect(m.color).toMatch(/^#[0-9a-f]{6}$/i);
    }
  });
});

describe("Legacy exports stay wired for the Analytics scorecard", () => {
  test("HARD_DECKS is the legacy 3-level shape (7.5 / 12.5 / 15)", () => {
    expect(HARD_DECKS).toBe(LEGACY_HARD_DECKS);
    expect(HARD_DECKS.map(d => d.pct)).toEqual([7.5, 12.5, 15.0]);
  });

  test("DECK_META still renders the legacy 'Remove Margin / Go To Cash' copy", () => {
    // Command Center consumes DECK_META keyed off the ATH-drawdown
    // classifier and still expects the old actionable copy while the
    // ATH classifier stands. Migration 068 didn't rework Command Center.
    expect(DECK_META).toBe(LEGACY_DECK_META);
    expect(DECK_META.L1.sub).toBe("Remove Margin");
    expect(DECK_META.L3.sub).toBe("Go To Cash");
  });

  test("classifyDeck is the legacy classifier", () => {
    expect(classifyDeck).toBe(classifyLegacyDeck);
  });
});
