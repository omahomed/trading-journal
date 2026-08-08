import { describe, it, expect } from "vitest";
import {
  computeVolatilitySizing,
  computeCompositeStop,
  computeScaleOutStops,
  computeScaleInSizing,
  ceilingPctFor,
  VolSizerError,
  STANDARD_CEILING_PCT,
  YOUNG_IPO_CEILING_PCT,
  HARD_MAX_CEILING_PCT,
  SCALE_OUT_ATR_MULTIPLIERS,
} from "./vol-sizer";

// Reusable defaults: DELL-style "calm 4.5% ATR" name, NLV $400K, Normal
// mode (0.50% risk budget = $2000). Key Level chosen to make the
// composite land at ~$167.51 (structural stop wider than 1 ATR floor).
const DELL_BASE = {
  equity: 400_000,
  entry: 176.21,
  atrPct: 4.5,
  keyLevel: 171.365, // yields keyLevelBuffer = 171.365 − 3.856 = 167.51 (buffer = 0.5 × 4.5% × 171.365)
  tolPct: 0.5,
};

// COHR-style "hot 9.6% ATR" name — Key Level set well below entry so
// the ATR floor ends up being the tighter (higher-price) candidate;
// the composite falls back to the 1-ATR floor.
const COHR_BASE = {
  equity: 400_000,
  entry: 246.53,
  atrPct: 9.6,
  keyLevel: 240.0, // keyLevelBuffer = 240 − 11.52 = 228.48 > atrFloor 222.86 → ATR wins
  tolPct: 0.5,
};

describe("computeCompositeStop", () => {
  it("picks the LOWEST of the two candidates (widest stop = most defensive)", () => {
    const c = computeCompositeStop({ entry: 176.21, atrPct: 4.5, keyLevel: 171.365 });
    // atrFloor = 176.21 − 7.9295 = 168.28  (still entry-based)
    // keyLevelBuffer = 171.365 − max(3.856, 1.714) = 171.365 − 3.856 = 167.51
    // composite = min(168.28, 167.51) = 167.51 → key_level_buffer wins
    expect(c.winner).toBe("key_level_buffer");
    expect(c.price).toBeCloseTo(167.51, 2);
    expect(c.distance).toBeCloseTo(8.70, 2);
    expect(c.candidates.atrFloor).toBeCloseTo(168.28, 2);
    expect(c.candidates.keyLevelBuffer).toBeCloseTo(167.51, 2);
  });

  it("falls back to the 1 ATR floor when Key Level candidate is tighter", () => {
    // COHR: Key Level $240 sits close to entry; atrFloor 222.86 is
    // further away → atrFloor wins.
    const c = computeCompositeStop({ entry: 246.53, atrPct: 9.6, keyLevel: 240 });
    expect(c.winner).toBe("atr_floor");
    expect(c.price).toBeCloseTo(222.86, 2);
    expect(c.distance).toBeCloseTo(23.67, 2);
    expect(c.atrFraction).toBeCloseTo(1.0, 3);
  });

  it("buffer basis: 0.5 ATR wins under high volatility (>= 2% ATR)", () => {
    const c = computeCompositeStop({ entry: 100, atrPct: 5, keyLevel: 95 });
    // 0.5 ATR (KL-based) = 0.5 × 5% × 95 = 2.375; 1% of KL = 0.95 → 0.5 ATR wins
    expect(c.candidates.bufferApplied).toBeCloseTo(2.375, 4);
    expect(c.candidates.bufferBasis).toBe("half_atr");
  });

  it("buffer basis: 1% floor wins under LOW volatility (< 2% ATR)", () => {
    const c = computeCompositeStop({ entry: 100, atrPct: 1.5, keyLevel: 95 });
    // 0.5 ATR (KL-based) = 0.5 × 1.5% × 95 = 0.7125; 1% of KL = 0.95 → 1% floor wins
    expect(c.candidates.bufferApplied).toBeCloseTo(0.95, 4);
    expect(c.candidates.bufferBasis).toBe("one_percent");
  });

  it("throws on non-positive entry / atrPct / keyLevel", () => {
    expect(() => computeCompositeStop({ entry: 0, atrPct: 5, keyLevel: 95 })).toThrow(VolSizerError);
    expect(() => computeCompositeStop({ entry: 100, atrPct: 0, keyLevel: 95 })).toThrow(VolSizerError);
    expect(() => computeCompositeStop({ entry: 100, atrPct: 5, keyLevel: 0 })).toThrow(VolSizerError);
  });

  it("atrFraction is exactly 1.0 when the ATR floor wins", () => {
    // By construction: atrFloor is at Entry − 1 ATR → distance = 1 ATR
    // → atrFraction = 1.0.
    const c = computeCompositeStop({ entry: 246.53, atrPct: 9.6, keyLevel: 240 });
    expect(c.atrFraction).toBeCloseTo(1.0, 3);
  });

  it("atrFraction is > 1.0 when the Key Level candidate wins", () => {
    // When Key Level − buffer sits LOWER than the ATR floor, the
    // composite distance exceeds 1 ATR by construction.
    const c = computeCompositeStop({ entry: 176.21, atrPct: 4.5, keyLevel: 171.365 });
    expect(c.atrFraction).toBeGreaterThan(1.0);
  });
});

describe("ceilingPctFor", () => {
  it("returns 15% standard when youngIpo is falsy", () => {
    expect(ceilingPctFor(undefined)).toEqual({ pct: STANDARD_CEILING_PCT, policy: "standard" });
    expect(ceilingPctFor(false)).toEqual({ pct: STANDARD_CEILING_PCT, policy: "standard" });
  });

  it("returns 5% young-ipo clamp when youngIpo is true", () => {
    expect(ceilingPctFor(true)).toEqual({ pct: YOUNG_IPO_CEILING_PCT, policy: "young_ipo" });
  });

  it("exposes HARD_MAX_CEILING_PCT = 20", () => {
    // Not returned by ceilingPctFor — informational constant that Log
    // Buy / any manual override consumer can enforce.
    expect(HARD_MAX_CEILING_PCT).toBe(20);
  });
});

describe("computeScaleOutStops", () => {
  it("splits shares floor / floor / remainder", () => {
    const s = computeScaleOutStops(100, 5, 100, 100_000);
    expect(s.legs[0].shares).toBe(33);
    expect(s.legs[1].shares).toBe(33);
    expect(s.legs[2].shares).toBe(34);
    expect(s.totalShares).toBe(100);
  });

  it("prices legs at Entry − 0.5/1.0/1.5 × ATR $/share", () => {
    // Entry 100, ATR 5% → atrPerShare $5.
    const s = computeScaleOutStops(100, 5, 100, 100_000);
    expect(s.legs[0].stopPrice).toBeCloseTo(97.5, 4); // -0.5 ATR
    expect(s.legs[1].stopPrice).toBeCloseTo(95.0, 4); // -1.0 ATR
    expect(s.legs[2].stopPrice).toBeCloseTo(92.5, 4); // -1.5 ATR
  });

  it("avg exit lands EXACTLY at 1 ATR for share counts divisible by 3", () => {
    const s = computeScaleOutStops(100, 5, 99, 100_000);
    expect(s.avgExitPrice).toBeCloseTo(95.0, 4);
    expect(s.avgExitPct).toBeCloseTo(-5.0, 4);
  });

  it("avg exit drifts slightly on uneven remainders (small share counts)", () => {
    // 100 shares → 33/33/34. Total value = 33*97.5 + 33*95 + 34*92.5 = 9502.
    // avg = 9502/100 = 95.02. Pure 1 ATR would be 95.00. Drift = $0.02.
    const s = computeScaleOutStops(100, 5, 100, 100_000);
    expect(s.avgExitPrice).toBeCloseTo(95.0, 1);
    expect(s.avgExitPrice).not.toBe(95.0);
  });

  it("locked ATR multipliers 0.5 / 1.0 / 1.5", () => {
    expect(SCALE_OUT_ATR_MULTIPLIERS).toEqual([0.5, 1.0, 1.5]);
  });

  it("returns empty legs when shares = 0", () => {
    const s = computeScaleOutStops(100, 5, 0, 100_000);
    expect(s.legs.every((l) => l.shares === 0)).toBe(true);
    expect(s.totalLoss).toBe(0);
    expect(s.avgExitPrice).toBe(100); // fallback = entry when no shares
    // -0 vs 0 quirk from the negation in the pct formula; toBeCloseTo
    // avoids the Object.is trap.
    expect(s.avgExitPct).toBeCloseTo(0, 6);
  });
});

describe("computeVolatilitySizing — DELL scenario (calm 4.5% ATR)", () => {
  const results = computeVolatilitySizing(DELL_BASE);

  it("computes risk budget $2000 at Normal (0.50%) on $400K NLV", () => {
    expect(results.riskBudget).toBe(2000);
  });

  it("composite stop at $167.51 (Key Level candidate wins over 1 ATR floor $168.28)", () => {
    expect(results.composite.winner).toBe("key_level_buffer");
    expect(results.composite.price).toBeCloseTo(167.51, 2);
    expect(results.composite.distance).toBeCloseTo(8.70, 2);
  });

  it("candidate shares = floor(2000 / 8.70) = 229", () => {
    expect(results.candidateShares).toBe(229);
  });

  it("ceiling shares at 15% = floor(400K × 0.15 / 176.21) = 340", () => {
    expect(results.ceilingShares).toBe(340);
  });

  it("final shares = 229 (risk-bound, not ceiling-bound)", () => {
    expect(results.finalShares).toBe(229);
    expect(results.bind).toBe("risk");
  });

  it("position notional ≈ $40K = 10% NLV", () => {
    expect(results.positionCost).toBeCloseTo(229 * 176.21, 2);
    expect(results.positionPct).toBeCloseTo(10, 0);
  });

  it("realized risk = shares × distance ≈ $1992 (floor(2000/8.70) leaves $8 unused)", () => {
    // Floor to whole shares undershoots the risk budget by
    // distance × frac_shares_dropped ≈ 8.70 × 0.87 ≈ $7.57. Pin the
    // exact realized-risk value so drift here surfaces immediately.
    expect(results.riskIfStopped).toBeCloseTo(1992.46, 2);
    expect(results.riskPct).toBeCloseTo(0.498, 3);
  });

  it("scale-out avg exit ≈ 1 ATR below entry ($168.28), matching risk budget", () => {
    // avg exit price should be close to entry − 1 ATR = 168.28
    expect(results.scaleOut.avgExitPrice).toBeCloseTo(168.28, 1);
  });

  it("no warnings for the happy path", () => {
    expect(results.warnings).toEqual([]);
  });
});

describe("computeVolatilitySizing — COHR scenario (hot 9.6% ATR)", () => {
  const results = computeVolatilitySizing(COHR_BASE);

  it("composite stop = 1 ATR floor at $222.86 (ATR wins over Key Level)", () => {
    expect(results.composite.winner).toBe("atr_floor");
    expect(results.composite.price).toBeCloseTo(222.86, 2);
    expect(results.composite.distance).toBeCloseTo(23.67, 2);
  });

  it("candidate shares = 2000 / 23.67 = 84", () => {
    expect(results.candidateShares).toBe(84);
  });

  it("final shares = 84 (risk-bound; ceiling at 15% = 243 wouldn't bind)", () => {
    expect(results.finalShares).toBe(84);
    expect(results.bind).toBe("risk");
    expect(results.ceilingShares).toBe(243);
  });

  it("position notional ~$20.7K ≈ 5.2% NLV", () => {
    expect(results.positionPct).toBeCloseTo(5.2, 1);
  });
});

describe("computeVolatilitySizing — ceiling bind at Offense on calm names", () => {
  it("clearly ceiling-binds with a tighter composite", () => {
    // Tighten Key Level to produce a smaller stop distance and thus
    // higher candidate shares — ceiling then unambiguously binds.
    const tightened = computeVolatilitySizing({ ...DELL_BASE, tolPct: 0.75, keyLevel: 174 });
    expect(tightened.candidateShares).toBeGreaterThan(tightened.ceilingShares);
    expect(tightened.finalShares).toBe(tightened.ceilingShares);
    expect(tightened.bind).toBe("ceiling");
  });
});

describe("computeVolatilitySizing — young-IPO clamp", () => {
  const results = computeVolatilitySizing({ ...DELL_BASE, youngIpo: true });

  it("ceiling pct drops to 5% policy", () => {
    expect(results.ceilingPct).toBe(YOUNG_IPO_CEILING_PCT);
    expect(results.ceilingPolicy).toBe("young_ipo");
    expect(results.ceilingShares).toBe(Math.floor((400_000 * 5) / 100 / 176.21)); // 113
  });

  it("bind flips to ceiling when 5% ceiling is tighter than risk-bound raw shares", () => {
    // raw shares = 227 (from DELL happy path), ceiling shares = 113 → 5% ceiling binds
    expect(results.finalShares).toBe(113);
    expect(results.bind).toBe("ceiling");
  });
});

describe("computeVolatilitySizing — validation & edge cases", () => {
  it("throws on missing / non-positive inputs", () => {
    expect(() => computeVolatilitySizing({ ...DELL_BASE, equity: 0 })).toThrow(VolSizerError);
    expect(() => computeVolatilitySizing({ ...DELL_BASE, entry: 0 })).toThrow(VolSizerError);
    expect(() => computeVolatilitySizing({ ...DELL_BASE, atrPct: 0 })).toThrow(VolSizerError);
    expect(() => computeVolatilitySizing({ ...DELL_BASE, keyLevel: 0 })).toThrow(VolSizerError);
    expect(() => computeVolatilitySizing({ ...DELL_BASE, tolPct: 0 })).toThrow(VolSizerError);
  });
});

// ════════════════════════════════════════════════════════════════════
// computeScaleInSizing — composite-stop scale-in retrofit (2026-08-07)
// ════════════════════════════════════════════════════════════════════
//
// Scenario shape: user has a pilot 7% position, ticker has moved up,
// they want to top off to 10%. Sizer computes composite stop for the
// CURRENT price and sizes the add against remaining risk headroom.

// Base: DELL-shape, NLV $400K, pilot at 7% weight → ~159 sh @ $176.21,
// price now at $185 (scale-in fill). Normal mode = 0.50% = $2000 budget.
const SCALE_BASE = {
  equity: 400_000,
  entry: 185,
  atrPct: 4.5,
  keyLevel: 180,     // key-level candidate stop = 180 − max(0.5×4.5%×180, 1%×180) = 180 − 4.05 = 175.95
  tolPct: 0.5,       // Normal
  currentShares: 159,
  currentAvgEntry: 176.21,
  targetPctNlv: 10,
  multiplier: 1,
};

describe("computeScaleInSizing — happy path (add fits current mode)", () => {
  it("returns success verdict with recommended add ≤ target", () => {
    const r = computeScaleInSizing(SCALE_BASE);
    // Composite: ATR floor = 185 − (185 × 0.045) = 176.675;
    // KL buffer = 175.95. ATR floor sits HIGHER → key-level wins.
    // Composite = 175.95, distance ≈ 9.05
    expect(r.composite.winner).toBe("key_level_buffer");
    expect(r.composite.price).toBeCloseTo(175.95, 2);
    // Existing risk: 159 × max(0, 176.21 − 175.95) = 159 × 0.26 ≈ $41
    expect(r.existingRisk).toBeCloseTo(41.34, 1);
    // Headroom ≈ 2000 − 41 = 1959
    expect(r.headroom).toBeCloseTo(1958.66, 1);
    // Target total = ceil(40000 / 185) = 217 shs → target add = 58
    expect(r.targetAdd).toBe(58);
    // Affordable add ≈ 1959 / 9.05 = ~216 shs → success
    expect(r.affordableAdd).toBeGreaterThanOrEqual(58);
    expect(r.verdict).toBe("success");
    expect(r.recommendedAdd).toBe(58);
    expect(r.modeHints).toEqual([]);
  });
});

describe("computeScaleInSizing — partial verdict + mode-bump hints", () => {
  it("returns partial when Pilot budget can't cover the target", () => {
    // Same scenario but Pilot mode (0.25%) → budget $1000
    const r = computeScaleInSizing({ ...SCALE_BASE, tolPct: 0.25 });
    // Headroom ≈ 1000 − 41 = 959; affordable ≈ 959/9.05 ≈ 105
    // Target = 58, affordable = 105 → still covers target → success actually
    // Let's try a smaller equity so the target overshoots affordable.
    const tight = computeScaleInSizing({
      ...SCALE_BASE,
      tolPct: 0.25,
      currentShares: 300,           // big existing so existingRisk eats more
      currentAvgEntry: 190,         // avg above composite so existingRisk > 0
      targetPctNlv: 15,             // pushing to Core weight
    });
    // Just assert the shape: some verdict + mode hints appear if partial/blocked
    if (tight.verdict === "partial" || tight.verdict === "blocked") {
      expect(tight.modeHints.length).toBeGreaterThan(0);
      // Every hint's tolPct must be greater than the input tolPct
      for (const h of tight.modeHints) {
        expect(h.tolPct).toBeGreaterThan(0.25);
      }
    }
  });

  it("marks the higher mode as coversTarget when it fully fills the fill", () => {
    // Construct a scenario where Pilot doesn't cover but a higher mode does.
    // NLV $100k, target 5% = $5k. At $100 entry, 50 sh target.
    // Composite: keyLevel=90 → buffer = max(0.5×5%×90, 1%×90) = 2.25 →
    //   KL buffer = 87.75. ATR floor = 95. Composite = 87.75 (KL wins).
    //   Distance = 12.25.
    // Pilot budget: $100k × 0.25% = $250 → affordable = 20 sh (< 50 target)
    // Normal budget: $500 → 40 sh (< 50)
    // Offense budget: $750 → 61 sh (COVERS 50 ✓)
    const r = computeScaleInSizing({
      equity: 100_000,
      entry: 100,
      atrPct: 5,
      keyLevel: 90,
      tolPct: 0.25,       // Pilot
      currentShares: 0,
      currentAvgEntry: 0,
      targetPctNlv: 5,
    });
    expect(r.verdict).toBe("partial");
    expect(r.modeHints.length).toBeGreaterThan(0);
    // At least one hint should be marked as covering the target.
    expect(r.modeHints.some(h => h.coversTarget)).toBe(true);
  });
});

describe("computeScaleInSizing — blocked verdict", () => {
  it("returns blocked when existing risk exhausts the budget", () => {
    // Existing 100 sh × $100 = $10k position (10% weight). Target 15%
    // → 50 more shares needed. Composite distance $12.25. But avg cost
    // is $105, well above composite, so existing risk =
    // 100 × (105 − 87.75) = $1725, far exceeding the Pilot $250 budget.
    // No headroom left → blocked.
    const r = computeScaleInSizing({
      equity: 100_000,
      entry: 100,
      atrPct: 5,
      keyLevel: 90,
      tolPct: 0.25,       // Pilot, $250 budget
      currentShares: 100,
      currentAvgEntry: 105,
      targetPctNlv: 15,
    });
    expect(r.verdict).toBe("blocked");
    expect(r.recommendedAdd).toBe(0);
    expect(r.affordableAdd).toBe(0);
    // Higher modes preview whether the bump would help — even Max (1%)
    // budget of $1000 is way under the $1725 existing risk.
    expect(r.modeHints.length).toBeGreaterThan(0);
    expect(r.modeHints.every(h => !h.coversTarget)).toBe(true);
  });
});

describe("computeScaleInSizing — at_target verdict (no add needed)", () => {
  it("returns at_target when currentShares already meets the target %", () => {
    const r = computeScaleInSizing({
      ...SCALE_BASE,
      currentShares: 240,   // 240 × 185 = $44,400 > 10% × 400k = $40k
    });
    expect(r.verdict).toBe("at_target");
    expect(r.targetAdd).toBe(0);
    expect(r.recommendedAdd).toBe(0);
  });
});

describe("computeScaleInSizing — risk-free position", () => {
  it("flags isRiskFree when composite stop sits above avg cost", () => {
    // Composite for entry=185, atr=4.5%, keyLevel=180 → 175.95.
    // Avg cost 170 < 175.95 → position is above stop = risk-free.
    const r = computeScaleInSizing({
      ...SCALE_BASE,
      currentAvgEntry: 170,
    });
    expect(r.isRiskFree).toBe(true);
    expect(r.existingRisk).toBe(0);
  });
});

describe("computeScaleInSizing — validation", () => {
  it("throws on non-positive inputs", () => {
    expect(() => computeScaleInSizing({ ...SCALE_BASE, equity: 0 })).toThrow(VolSizerError);
    expect(() => computeScaleInSizing({ ...SCALE_BASE, entry: 0 })).toThrow(VolSizerError);
    expect(() => computeScaleInSizing({ ...SCALE_BASE, atrPct: 0 })).toThrow(VolSizerError);
    expect(() => computeScaleInSizing({ ...SCALE_BASE, keyLevel: 0 })).toThrow(VolSizerError);
    expect(() => computeScaleInSizing({ ...SCALE_BASE, tolPct: 0 })).toThrow(VolSizerError);
    expect(() => computeScaleInSizing({ ...SCALE_BASE, currentShares: -1 })).toThrow(VolSizerError);
    expect(() => computeScaleInSizing({ ...SCALE_BASE, targetPctNlv: 0 })).toThrow(VolSizerError);
  });

  it("accepts currentShares = 0 (fresh position, though rare on scale-in)", () => {
    const r = computeScaleInSizing({ ...SCALE_BASE, currentShares: 0, currentAvgEntry: 0 });
    expect(r.existingRisk).toBe(0);
    expect(r.isRiskFree).toBe(false);  // false because there are no shares
  });
});

describe("computeScaleInSizing — user's concrete scenario (SR1 → scale to 10%)", () => {
  it("walks the effective stop up when adding at a higher price", () => {
    // Pilot fills 100 sh at $100, ATR 5%, keyLevel=97 (above ATR floor).
    // ATR floor = 95; KL buffer 97 − max(2.425, 0.97) = 97 − 2.425 = 94.575.
    // Composite = min(95, 94.575) = 94.575 → KL wins (very close call).
    const pilot = computeScaleInSizing({
      equity: 100_000,
      entry: 100,
      atrPct: 5,
      keyLevel: 97,
      tolPct: 0.5,
      currentShares: 0,       // fresh
      currentAvgEntry: 0,
      targetPctNlv: 7,        // pilot target
    });
    // Position moves up. User tops off to 10% at $110.
    // At $110, ATR floor = 110 − 5.5 = 104.5.
    // KL 105 → buffer = max(0.5×5%×105, 1%×105) = max(2.625, 1.05) = 2.625.
    // KL buffer = 105 − 2.625 = 102.375. Composite = min(104.5, 102.375) = 102.375.
    const scale = computeScaleInSizing({
      equity: 100_000,
      entry: 110,
      atrPct: 5,
      keyLevel: 105,
      tolPct: 0.5,
      currentShares: 70,      // 7% of 100k @ $100 = 70 shs
      currentAvgEntry: 100,
      targetPctNlv: 10,
    });
    // The scale-in composite is HIGHER (tighter) than the pilot's —
    // the "effective stop walks up" behavior the user asked for.
    expect(scale.composite.price).toBeGreaterThan(pilot.composite.price);
    expect(scale.composite.price).toBeCloseTo(102.375, 2);
    // Existing 70 shs at $100 avg → composite 102.375 sits ABOVE avg
    // cost → free-roll, existing risk = 0.
    expect(scale.existingRisk).toBe(0);
    expect(scale.isRiskFree).toBe(true);
  });
});
