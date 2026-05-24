import { describe, test, expect } from "vitest";
import {
  computeVolatilitySizing,
  VolSizerError,
  type VolSizerInputs,
} from "./vol-sizer";

const GOOGL: VolSizerInputs = {
  equity: 702924,
  entry: 382.97,
  ma: 379.40,
  bufferPct: 1.0,
  atrPct: 2.87,
  tolPct: 1.0,
  targetSizePct: 10,
};

describe("vol-sizer lib", () => {
  describe("GOOGL canonical case", () => {
    const r = computeVolatilitySizing(GOOGL);

    test("positionCapShares = 183 (floor of 70292.4 / 382.97)", () => {
      expect(r.positionCapShares).toBe(183);
    });

    test("all four scenarios bind at 183 shares", () => {
      expect(r.techStop.finalShares).toBe(183);
      expect(r.atrScenarios[0].finalShares).toBe(183);
      expect(r.atrScenarios[1].finalShares).toBe(183);
      expect(r.atrScenarios[2].finalShares).toBe(183);
    });

    test("all four scenarios show capBinds=true (candidate > cap)", () => {
      expect(r.techStop.capBinds).toBe(true);
      expect(r.atrScenarios[0].capBinds).toBe(true);
      expect(r.atrScenarios[1].capBinds).toBe(true);
      expect(r.atrScenarios[2].capBinds).toBe(true);
    });

    test("recommended = 1.5x ATR (referential, reason tech_stop_inside_noise)", () => {
      expect(r.recommended.label).toBe("1.5x ATR");
      expect(r.recommendationReason).toBe("tech_stop_inside_noise");
      expect(r.recommended).toBe(r.atrScenarios[1]);
    });

    test("techStop.atrFraction ≈ 0.67 and warning text includes it", () => {
      expect(r.techStop.atrFraction).toBeCloseTo(0.67, 2);
      expect(r.warning.show).toBe(true);
      expect(r.warning.text).toContain("0.67");
      expect(r.warning.text).toContain("ATR");
    });

    test("ATR scenario fractions equal their multipliers by construction", () => {
      expect(r.atrScenarios[0].atrFraction).toBeCloseTo(1.0, 10);
      expect(r.atrScenarios[1].atrFraction).toBeCloseTo(1.5, 10);
      expect(r.atrScenarios[2].atrFraction).toBeCloseTo(2.0, 10);
    });

    test("derived budget / cap fields", () => {
      expect(r.riskBudget).toBeCloseTo(7029.24, 6);
      expect(r.positionCap).toBeCloseTo(70292.4, 6);
      expect(r.atrPerShare).toBeCloseTo(10.991239, 6);
    });
  });

  describe("tier cap binds ATR sizing", () => {
    // 5% tier with low ATR: candidate shares far exceed positionCapShares.
    const r = computeVolatilitySizing({
      equity: 1_000_000,
      entry: 100,
      ma: 99,
      bufferPct: 1,
      atrPct: 2,
      tolPct: 1,
      targetSizePct: 5,
    });

    test("positionCapShares = 500", () => {
      expect(r.positionCapShares).toBe(500);
    });

    test("all three ATR scenarios cap at 500 with capBinds=true", () => {
      for (const s of r.atrScenarios) {
        expect(s.finalShares).toBe(500);
        expect(s.capBinds).toBe(true);
        expect(s.candidateShares).toBeGreaterThan(500);
      }
    });

    test("tech stop also capped at 500", () => {
      expect(r.techStop.finalShares).toBe(500);
      expect(r.techStop.capBinds).toBe(true);
    });
  });

  describe("ATR sizing binds below tier cap (high ATR)", () => {
    // 8% ATR with 10% tier on $100k equity, $100 entry:
    //   positionCapShares = 100
    //   1.5x: stop=88, rps=12, candidate=floor(1000/12)=83  → final=83, capBinds=false
    //   2x:   stop=84, rps=16, candidate=62                 → final=62, capBinds=false
    const r = computeVolatilitySizing({
      equity: 100_000,
      entry: 100,
      ma: 99,
      bufferPct: 1,
      atrPct: 8,
      tolPct: 1,
      targetSizePct: 10,
    });

    test("positionCapShares = 100", () => {
      expect(r.positionCapShares).toBe(100);
    });

    test("1.5x ATR scenario sizes to 83, cap does not bind", () => {
      expect(r.atrScenarios[1].finalShares).toBe(83);
      expect(r.atrScenarios[1].capBinds).toBe(false);
    });

    test("2x ATR scenario sizes to 62, cap does not bind", () => {
      expect(r.atrScenarios[2].finalShares).toBe(62);
      expect(r.atrScenarios[2].capBinds).toBe(false);
    });
  });

  describe("recommendation boundary around atrFraction = 1.0", () => {
    const base = {
      equity: 100_000,
      entry: 100,
      ma: 100,
      atrPct: 2,
      tolPct: 1,
      targetSizePct: 10,
    } as const;

    test("buffer < 1 ATR → recommended is 1.5x ATR", () => {
      // bufferPct=1.5 → stop=98.5, rps=1.5, atrFraction=0.75
      const r = computeVolatilitySizing({ ...base, bufferPct: 1.5 });
      expect(r.techStop.atrFraction).toBeLessThan(1.0);
      expect(r.recommended).toBe(r.atrScenarios[1]);
      expect(r.recommendationReason).toBe("tech_stop_inside_noise");
    });

    test("buffer > 1 ATR → recommended is tech stop", () => {
      // bufferPct=3 → stop=97, rps=3, atrFraction=1.5
      const r = computeVolatilitySizing({ ...base, bufferPct: 3 });
      expect(r.techStop.atrFraction).toBeGreaterThan(1.0);
      expect(r.recommended).toBe(r.techStop);
      expect(r.recommendationReason).toBe("tech_stop_safe");
    });

    test("recommended is always referentially one of the four scenarios", () => {
      const r1 = computeVolatilitySizing({ ...base, bufferPct: 1.5 });
      const r2 = computeVolatilitySizing({ ...base, bufferPct: 3 });
      expect([r1.techStop, ...r1.atrScenarios]).toContain(r1.recommended);
      expect([r2.techStop, ...r2.atrScenarios]).toContain(r2.recommended);
    });
  });

  describe("floor rounding — all share counts are integers", () => {
    const r = computeVolatilitySizing({
      equity: 123_456,
      entry: 73.42,
      ma: 71.10,
      bufferPct: 1.27,
      atrPct: 3.41,
      tolPct: 0.75,
      targetSizePct: 7.5,
    });

    test("positionCapShares and per-scenario share counts are integers", () => {
      expect(Number.isInteger(r.positionCapShares)).toBe(true);
      expect(Number.isInteger(r.techStop.candidateShares)).toBe(true);
      expect(Number.isInteger(r.techStop.finalShares)).toBe(true);
      for (const s of r.atrScenarios) {
        expect(Number.isInteger(s.candidateShares)).toBe(true);
        expect(Number.isInteger(s.finalShares)).toBe(true);
      }
    });
  });

  describe("no-warning case (tech stop well outside 1 ATR)", () => {
    // bufferPct=5 against atrPct=2 → atrFraction = 2.5
    const r = computeVolatilitySizing({
      equity: 100_000,
      entry: 100,
      ma: 100,
      bufferPct: 5,
      atrPct: 2,
      tolPct: 1,
      targetSizePct: 10,
    });

    test("techStop.atrFraction > 1.0", () => {
      expect(r.techStop.atrFraction).toBeGreaterThan(1.0);
    });

    test("warning hidden with empty text", () => {
      expect(r.warning.show).toBe(false);
      expect(r.warning.text).toBe("");
    });

    test("recommended = tech stop", () => {
      expect(r.recommended).toBe(r.techStop);
      expect(r.recommendationReason).toBe("tech_stop_safe");
    });
  });

  describe("degenerate: tech stop at or above entry (ma > entry)", () => {
    const r = computeVolatilitySizing({
      equity: 100_000,
      entry: 100,
      ma: 105,
      bufferPct: 0,
      atrPct: 2,
      tolPct: 1,
      targetSizePct: 10,
    });

    test("does not throw", () => {
      expect(r).toBeDefined();
    });

    test("techStop scenario returns zero shares with capBinds=false", () => {
      expect(r.techStop.finalShares).toBe(0);
      expect(r.techStop.candidateShares).toBe(0);
      expect(r.techStop.capBinds).toBe(false);
      expect(r.techStop.positionCost).toBe(0);
      expect(r.techStop.riskIfStopped).toBe(0);
    });

    test("ATR scenarios still compute normally (non-zero shares)", () => {
      expect(r.atrScenarios[0].finalShares).toBeGreaterThan(0);
      expect(r.atrScenarios[1].finalShares).toBeGreaterThan(0);
      expect(r.atrScenarios[2].finalShares).toBeGreaterThan(0);
    });

    test("recommendation falls through to 1.5x ATR with degenerate warning text", () => {
      expect(r.recommended).toBe(r.atrScenarios[1]);
      expect(r.warning.show).toBe(true);
      expect(r.warning.text).toContain("at or above entry");
    });
  });

  describe("input validation", () => {
    test("atrPct = 0 throws VolSizerError", () => {
      expect(() => computeVolatilitySizing({ ...GOOGL, atrPct: 0 })).toThrow(
        VolSizerError,
      );
    });

    test("atrPct negative throws", () => {
      expect(() => computeVolatilitySizing({ ...GOOGL, atrPct: -1 })).toThrow(
        VolSizerError,
      );
    });

    test.each([
      ["equity zero", { equity: 0 }],
      ["equity negative", { equity: -1 }],
      ["entry zero", { entry: 0 }],
      ["ma zero", { ma: 0 }],
      ["tolPct zero", { tolPct: 0 }],
      ["targetSizePct zero", { targetSizePct: 0 }],
    ])("rejects %s", (_label, override) => {
      expect(() =>
        computeVolatilitySizing({ ...GOOGL, ...override }),
      ).toThrow(VolSizerError);
    });

    test("bufferPct = 0 is allowed (stop exactly at MA)", () => {
      expect(() =>
        computeVolatilitySizing({ ...GOOGL, bufferPct: 0 }),
      ).not.toThrow();
    });

    test("bufferPct negative is rejected", () => {
      expect(() =>
        computeVolatilitySizing({ ...GOOGL, bufferPct: -0.1 }),
      ).toThrow(VolSizerError);
    });
  });
});
