import { describe, test, expect } from "vitest";
import { computeTrim, type TrimInput } from "./sr8-trim-calc";

// Default scenario for an SR8 position with comfortable ADDS under the
// 2026-08-13 doctrine (7.5% core seed, 2.5% cascade steps):
//   NAV 600k, 7.5% core = 45k. Stock @ 100 → core target 450 sh.
//   Position 1500 sh @ 100 = 150k = 25% NAV → ADDS = 1050 shares.
//   B1 return 60% (well past the SR8 50% threshold).
function baseInput(overrides: Partial<TrimInput> = {}): TrimInput {
  return {
    totalShares: 1500,
    currentPrice: 100,
    b1ReturnPct: 60,
    nav: 600_000,
    rule: "sr2",
    ...overrides,
  };
}

describe("computeTrim — position state derivation", () => {
  test("core target = floor(NAV*0.075 / price), adds = total - core", () => {
    const r = computeTrim(baseInput());
    expect(r.coreTargetValue).toBeCloseTo(45_000);
    expect(r.coreTargetShares).toBe(450);
    expect(r.addsShares).toBe(1050);
    expect(r.totalValue).toBeCloseTo(150_000);
    expect(r.totalNavPct).toBeCloseTo(25);
  });

  test("position below core: adds clamps to 0", () => {
    // 200 sh @ 100 = 20k; core target is 450 → position sits below core.
    const r = computeTrim(baseInput({ totalShares: 200 }));
    expect(r.coreTargetShares).toBe(450);
    expect(r.addsShares).toBe(0);
  });

  test("nav=0: coreTargetShares=0, adds=totalShares (treat all as ADDS)", () => {
    const r = computeTrim(baseInput({ nav: 0 }));
    expect(r.coreTargetShares).toBe(0);
    expect(r.addsShares).toBe(1500);
  });

  test("invalid currentPrice (0): returns invalid state with zero trim", () => {
    const r = computeTrim(baseInput({ currentPrice: 0 }));
    expect(r.trimShares).toBe(0);
    expect(r.resultingState).toBe("invalid");
  });

  test("invalid totalShares (0): returns invalid state", () => {
    const r = computeTrim(baseInput({ totalShares: 0 }));
    expect(r.resultingState).toBe("invalid");
  });
});

describe("computeTrim — SR2 (25% trim, ADDS-bound)", () => {
  test("intended within ADDS cap: trims exactly 25%", () => {
    const r = computeTrim(baseInput({ rule: "sr2" }));
    expect(r.intendedTrimShares).toBe(375); // floor(1500 * 0.25)
    expect(r.trimShares).toBe(375);
    expect(r.coreFloorBinds).toBe(false);
    expect(r.resultingShares).toBe(1125);
    expect(r.resultingState).toBe("with-adds");
  });

  test("ADDS smaller than 25% — core floor binds", () => {
    // Force a scenario where core is inflated relative to totalShares so
    // ADDS is small: NAV 12M → core = 7.5% × 12M / 100 = 9000 sh. 1000
    // total sh → ADDS = 0 (position sits below core), no room to trim.
    // 25% of 1000 = 250 intended, 0 available.
    const r = computeTrim(baseInput({
      rule: "sr2", totalShares: 1000, nav: 12_000_000,
    }));
    expect(r.coreTargetShares).toBe(9000);
    expect(r.intendedTrimShares).toBe(250);
    expect(r.trimShares).toBe(0);  // capped at ADDS = 0
    expect(r.coreFloorBinds).toBe(true);
  });

  test("already at core: addsShares=0, trim=0", () => {
    const r = computeTrim(baseInput({ rule: "sr2", totalShares: 450 }));
    expect(r.addsShares).toBe(0);
    expect(r.trimShares).toBe(0);
    expect(r.resultingShares).toBe(450);
  });
});

describe("computeTrim — SR7 (cushion-tiered, ADDS-bound)", () => {
  test("cushion >50%: trim entire ADDS", () => {
    const r = computeTrim(baseInput({ rule: "sr7", b1ReturnPct: 71 }));
    expect(r.sr7CushionTier).toBe("gt50");
    expect(r.trimShares).toBe(1050); // = addsShares
    expect(r.resultingState).toBe("core-only");
  });

  test("cushion 25–50%: trim 50% of total, capped at ADDS", () => {
    // 1500 sh, 50% = 750. ADDS = 1050 (default fixture). Not capped.
    const r = computeTrim(baseInput({ rule: "sr7", b1ReturnPct: 30 }));
    expect(r.sr7CushionTier).toBe("25to50");
    expect(r.intendedTrimShares).toBe(750);
    expect(r.trimShares).toBe(750);
    expect(r.coreFloorBinds).toBe(false);
  });

  test("cushion 25–50% capped at ADDS when intended > ADDS", () => {
    // Push core large so ADDS < intended. NAV 15M → core=11_250 sh. But
    // totalShares=1500 → below core → ADDS=0. Intended = 750; capped 0.
    const r = computeTrim(baseInput({
      rule: "sr7", b1ReturnPct: 30, nav: 15_000_000,
    }));
    expect(r.intendedTrimShares).toBe(750);
    expect(r.trimShares).toBe(0);
    expect(r.coreFloorBinds).toBe(true);
  });

  test("cushion <25% (heavy pullback on SR8 position): full ADDS exit", () => {
    const r = computeTrim(baseInput({ rule: "sr7", b1ReturnPct: 10 }));
    expect(r.sr7CushionTier).toBe("lt25");
    expect(r.trimShares).toBe(1050); // = addsShares
    expect(r.resultingState).toBe("core-only");
  });

  test("null cushion: classifier defaults to <25% (conservative)", () => {
    const r = computeTrim(baseInput({ rule: "sr7", b1ReturnPct: null }));
    expect(r.sr7CushionTier).toBe("lt25");
  });
});

describe("computeTrim — SR8 Quick / Quicksand (target-based)", () => {
  test("Quick reduces position to 5% NAV target (2026-08-13 doctrine)", () => {
    // NAV 600k, 5% = 30k. Px=100 → target 300 sh. Start at 1500 sh.
    const r = computeTrim(baseInput({ rule: "sr8-quick" }));
    expect(r.intendedTrimShares).toBe(1200);
    expect(r.trimShares).toBe(1200);
    expect(r.resultingShares).toBe(300);
    expect(r.resultingNavPct).toBeCloseTo(5.0);
  });

  test("Quicksand reduces position to 2.5% NAV target", () => {
    // NAV 600k, 2.5% = 15k. Px=100 → target 150 sh. Start at 1500 sh.
    const r = computeTrim(baseInput({ rule: "sr8-quicksand" }));
    expect(r.intendedTrimShares).toBe(1350);
    expect(r.trimShares).toBe(1350);
    expect(r.resultingShares).toBe(150);
    expect(r.resultingNavPct).toBeCloseTo(2.5);
  });

  test("Quick + Quicksand from same start produce different trims", () => {
    // COHR-style scenario reworked for the new cascade: NAV $600k, 302 sh
    // @ $358.50 = $108,267 ≈ 18.0% NAV.
    //   Quick target  → floor(0.05  × 600000 / 358.50) = floor( 83.7) =  83
    //   Quicksand     → floor(0.025 × 600000 / 358.50) = floor( 41.8) =  41
    //   Quick trim    = 302 -  83 = 219
    //   Quicksand trim= 302 -  41 = 261
    const cohr = { totalShares: 302, currentPrice: 358.50, nav: 600_000 };
    const quick = computeTrim(baseInput({ rule: "sr8-quick", ...cohr }));
    const sand = computeTrim(baseInput({ rule: "sr8-quicksand", ...cohr }));
    expect(quick.trimShares).toBe(219);
    expect(sand.trimShares).toBe(261);
    expect(quick.trimShares).not.toBe(sand.trimShares);
  });

  test("position already at/below target: trim is 0", () => {
    // 50 sh @ $358.50 = $17,925 ≈ 3.0% NAV. Below Quick's 5% target.
    const r = computeTrim(baseInput({
      rule: "sr8-quick", totalShares: 50, currentPrice: 358.50, nav: 600_000,
    }));
    expect(r.trimShares).toBe(0);
    expect(r.resultingShares).toBe(50);
    // Still below the 7.5% NAV core, so state is below-core.
    expect(r.resultingState).toBe("below-core");
  });

  test("Quicksand sequential after Quick: 2.5% NAV from 5% start", () => {
    // After Quick the position is at 5% NAV. Quicksand drives to 2.5%.
    // 300 sh @ 100 = 30k = 5% NAV. Quicksand target 150 sh → trim 150.
    const r = computeTrim(baseInput({ rule: "sr8-quicksand", totalShares: 300 }));
    expect(r.trimShares).toBe(150);
    expect(r.resultingShares).toBe(150);
    expect(r.resultingNavPct).toBeCloseTo(2.5);
  });

  test("NAV=0: trim=0 (target undefined)", () => {
    const r = computeTrim(baseInput({ rule: "sr8-quick", nav: 0 }));
    expect(r.trimShares).toBe(0);
  });
});

describe("computeTrim — SR8 Grateful Dead / SR13 (full exit)", () => {
  test("Grateful Dead exits everything including core", () => {
    const r = computeTrim(baseInput({ rule: "sr8-grateful-dead" }));
    expect(r.trimShares).toBe(1500);
    expect(r.resultingShares).toBe(0);
    expect(r.resultingState).toBe("closed");
  });

  test("SR13 exits everything", () => {
    const r = computeTrim(baseInput({ rule: "sr13" }));
    expect(r.trimShares).toBe(1500);
    expect(r.resultingState).toBe("closed");
  });
});

// ─────────────────────────────────────────────────────────────────
// Regression tests for the SR8 activation-anchor fix (2026-07-18)
//
// The bug: SR8 Quick/QS targets computed (target_pct) × LIVE NAV / price.
// When NAV grew past activation NAV, target shares > held → no-op
// trims on valid signals → cores undefended.
//
// The fix: anchor targets to sr8_activation_nlv (fixed at first +50%
// crossing). Pass activationNlv + coreShares to computeTrim; result
// exposes anchorSource='activation' vs 'live_fallback' so the UI can
// flag legacy positions.
//
// Post 2026-08-13 doctrine: Quick target = 5% × activation_NLV,
// Quicksand = 2.5%. Regression fixtures below re-anchor to the new
// cascade destinations.
// ─────────────────────────────────────────────────────────────────

describe("computeTrim — SR8 activation anchor (2026-07-18 fix)", () => {
  test("BE regression: Quick target from activation NAV (anchored) vs live-nav bug", () => {
    // NAV grew from activation $430K to live $805K — 87% appreciation.
    // Post-2026-08-13 doctrine (5% Quick):
    //   OLD live-nav formula: 0.05 × 805679 / 288 ≈ 139 shs
    //   NEW anchored:         0.05 × 430249 / 288 ≈  74 shs
    // BE held 224 shs. Anchored trim = 224 − 74 = 150 shs.
    const priceOnSignalDay = 288;
    const activationNlv = 430_249;
    const liveNav = 805_679;
    const coreShares = 224;

    const anchored = computeTrim({
      totalShares: 224,
      currentPrice: priceOnSignalDay,
      b1ReturnPct: 80,
      nav: liveNav,
      activationNlv,
      coreShares,
      rule: "sr8-quick",
    });
    // 0.05 × 430249 / 288 = 74.69 → floor to 74. Trim = 224 − 74 = 150.
    expect(anchored.trimShares).toBeGreaterThanOrEqual(149);
    expect(anchored.trimShares).toBeLessThanOrEqual(151);
    expect(anchored.anchorSource).toBe("activation");
    // Resulting position ≈ 74 shs (= 5% × activation / px).
    expect(anchored.resultingShares).toBeGreaterThanOrEqual(73);
    expect(anchored.resultingShares).toBeLessThanOrEqual(75);

    // Live-nav fallback still fires the trim (5% × 805k = 139 < 224 held),
    // but the target is inflated relative to activation. Sanity-check
    // that the flag surfaces.
    const buggy = computeTrim({
      totalShares: 224,
      currentPrice: priceOnSignalDay,
      b1ReturnPct: 80,
      nav: liveNav,
      // Neither activationNlv nor coreShares supplied — fallback path.
      rule: "sr8-quick",
    });
    expect(buggy.anchorSource).toBe("live_fallback");
    // 5% × 805679 / 288 = 139.87 → 139. Trim = 224 − 139 = 85.
    expect(buggy.trimShares).toBeGreaterThanOrEqual(84);
    expect(buggy.trimShares).toBeLessThanOrEqual(86);
    // The critical property: anchored trims MORE than live-nav (which is
    // the point of the anchor — it corrects the under-defended core).
    expect(anchored.trimShares).toBeGreaterThan(buggy.trimShares);
  });

  test("Quicksand: 2.5% of activation NAV drives the destination", () => {
    // Same BE fixture, QS target = 0.025 × 430249 / 288 ≈ 37.34 → 37.
    // Held 74 (post-Quick) → trim = 74 − 37 = 37.
    const r = computeTrim({
      totalShares: 74,
      currentPrice: 288,
      b1ReturnPct: 75,
      nav: 805_679,
      activationNlv: 430_249,
      coreShares: 224,
      rule: "sr8-quicksand",
    });
    expect(r.trimShares).toBeGreaterThanOrEqual(36);
    expect(r.trimShares).toBeLessThanOrEqual(38);
    expect(r.anchorSource).toBe("activation");
  });

  test("MU adjacent case: small NAV drift, target barely moves", () => {
    // Spec anti-regression: when NAV moves LITTLE from activation, the
    // anchored + live-nav answers should be within ~1-2 shs. Verifies
    // the fix doesn't distort the calm-drift case.
    const activationNlv = 551_423;
    const liveNav = 553_000; // ~0.3% drift
    const priceOnSignal = 900;

    const anchored = computeTrim({
      totalShares: 100,
      currentPrice: priceOnSignal,
      b1ReturnPct: 90,
      nav: liveNav,
      activationNlv,
      coreShares: 116,
      rule: "sr8-quick",
    });
    const liveFallback = computeTrim({
      totalShares: 100,
      currentPrice: priceOnSignal,
      b1ReturnPct: 90,
      nav: liveNav,
      rule: "sr8-quick",
    });
    // Delta of trim should be ≤ 1 share under the small-drift case.
    expect(Math.abs(anchored.trimShares - liveFallback.trimShares)).toBeLessThanOrEqual(1);
  });

  test("coreShares directly wins over derived core (fixed count preserved)", () => {
    // When both activationNlv and coreShares are passed, coreShares is
    // the source of truth for the core count (used in ADDS calcs).
    // Grandfathering: a position declared under the old 15% doctrine
    // keeps its typed coreShares regardless of the new 7.5% seed.
    const r = computeTrim({
      totalShares: 300,
      currentPrice: 288,
      b1ReturnPct: 60,
      nav: 800_000,
      activationNlv: 430_249,
      coreShares: 224,
      rule: "sr2",
    });
    expect(r.coreTargetShares).toBe(224);
    expect(r.addsShares).toBe(76);
    expect(r.anchorSource).toBe("activation");
  });

  test("anchorSource='live_fallback' when neither activationNlv nor coreShares supplied", () => {
    const r = computeTrim(baseInput());
    expect(r.anchorSource).toBe("live_fallback");
  });

  test("anchorSource='activation' when only activationNlv supplied", () => {
    const r = computeTrim(baseInput({ activationNlv: 430_249 }));
    expect(r.anchorSource).toBe("activation");
  });
});

describe("computeTrim — resultingState transitions", () => {
  test("resulting > core: 'with-adds'", () => {
    const r = computeTrim(baseInput({ rule: "sr2" }));
    expect(r.resultingState).toBe("with-adds");
  });

  test("resulting == core: 'core-only'", () => {
    // SR7 gt50 trims ADDS exactly → resulting = core.
    const r = computeTrim(baseInput({ rule: "sr7", b1ReturnPct: 60 }));
    expect(r.resultingState).toBe("core-only");
  });

  test("resulting < core: 'below-core'", () => {
    // SR13 forces full exit → resulting=0 → 'closed' takes priority.
    // To land 'below-core' (resulting > 0 but < core): SR8 Quick on a
    // position above its 5% target. Base scenario: 1500 sh, core 450,
    // Quick targets 300 → trim 1200, resulting 300 < core 450.
    const r = computeTrim(baseInput({ rule: "sr8-quick" }));
    expect(r.trimShares).toBe(1200);
    expect(r.resultingShares).toBe(300);
    expect(r.resultingState).toBe("below-core");
  });
});
