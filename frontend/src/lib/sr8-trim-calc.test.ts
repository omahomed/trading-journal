import { describe, test, expect } from "vitest";
import { computeTrim, type TrimInput } from "./sr8-trim-calc";

// Default scenario for an SR8 position with comfortable ADDS:
//   NAV 600k, 15% core = 90k. Stock @ 100 → core target 900 sh.
//   Position 1500 sh @ 100 = 150k = 25% NAV → ADDS = 600 shares.
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
  test("core target = floor(NAV*0.15 / price), adds = total - core", () => {
    const r = computeTrim(baseInput());
    expect(r.coreTargetValue).toBeCloseTo(90_000);
    expect(r.coreTargetShares).toBe(900);
    expect(r.addsShares).toBe(600);
    expect(r.totalValue).toBeCloseTo(150_000);
    expect(r.totalNavPct).toBeCloseTo(25);
  });

  test("position below core: adds clamps to 0", () => {
    const r = computeTrim(baseInput({ totalShares: 500 }));
    expect(r.coreTargetShares).toBe(900);
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
    // 1000 sh @ 100 = 100k. Core 900 sh. ADDS = 100. 25% of 1000 = 250.
    const r = computeTrim(baseInput({ rule: "sr2", totalShares: 1000 }));
    expect(r.intendedTrimShares).toBe(250);
    expect(r.trimShares).toBe(100); // capped at ADDS
    expect(r.coreFloorBinds).toBe(true);
    expect(r.resultingShares).toBe(900);
    expect(r.resultingState).toBe("core-only");
  });

  test("already at core: addsShares=0, trim=0", () => {
    const r = computeTrim(baseInput({ rule: "sr2", totalShares: 900 }));
    expect(r.addsShares).toBe(0);
    expect(r.trimShares).toBe(0);
    expect(r.resultingShares).toBe(900);
  });
});

describe("computeTrim — SR7 (cushion-tiered, ADDS-bound)", () => {
  test("cushion >50%: trim entire ADDS", () => {
    const r = computeTrim(baseInput({ rule: "sr7", b1ReturnPct: 71 }));
    expect(r.sr7CushionTier).toBe("gt50");
    expect(r.trimShares).toBe(600); // = addsShares
    expect(r.resultingState).toBe("core-only");
  });

  test("cushion 25–50%: trim 50% of total, capped at ADDS", () => {
    // 1500 sh, 50% = 750. ADDS=600. Should cap to 600.
    const r = computeTrim(baseInput({ rule: "sr7", b1ReturnPct: 30 }));
    expect(r.sr7CushionTier).toBe("25to50");
    expect(r.intendedTrimShares).toBe(750);
    expect(r.trimShares).toBe(600);
    expect(r.coreFloorBinds).toBe(true);
  });

  test("cushion 25–50% with intended ≤ ADDS: not capped", () => {
    // 1500 sh, 50% = 750. ADDS huge (push core small via tiny NAV).
    // NAV=10k → core 1500, ADDS=0... that's the wrong way. Instead:
    // NAV=60k → 9k core, 90 sh. ADDS = 1410. Intended 750 < 1410.
    const r = computeTrim(baseInput({ rule: "sr7", b1ReturnPct: 30, nav: 60_000 }));
    expect(r.intendedTrimShares).toBe(750);
    expect(r.trimShares).toBe(750);
    expect(r.coreFloorBinds).toBe(false);
  });

  test("cushion <25% (heavy pullback on SR8 position): full ADDS exit", () => {
    const r = computeTrim(baseInput({ rule: "sr7", b1ReturnPct: 10 }));
    expect(r.sr7CushionTier).toBe("lt25");
    expect(r.trimShares).toBe(600); // = addsShares
    expect(r.resultingState).toBe("core-only");
  });

  test("null cushion: classifier defaults to <25% (conservative)", () => {
    const r = computeTrim(baseInput({ rule: "sr7", b1ReturnPct: null }));
    expect(r.sr7CushionTier).toBe("lt25");
  });
});

describe("computeTrim — SR8 Quick / Quicksand (target-based)", () => {
  test("Quick reduces position to 10% NAV target", () => {
    // NAV 600k, 10% = 60k. Px=100 → target 600 sh. Start at 1500 sh.
    const r = computeTrim(baseInput({ rule: "sr8-quick" }));
    expect(r.intendedTrimShares).toBe(900);
    expect(r.trimShares).toBe(900);
    expect(r.resultingShares).toBe(600);
    expect(r.resultingNavPct).toBeCloseTo(10.0);
  });

  test("Quicksand reduces position to 5% NAV target", () => {
    // NAV 600k, 5% = 30k. Px=100 → target 300 sh. Start at 1500 sh.
    const r = computeTrim(baseInput({ rule: "sr8-quicksand" }));
    expect(r.intendedTrimShares).toBe(1200);
    expect(r.trimShares).toBe(1200);
    expect(r.resultingShares).toBe(300);
    expect(r.resultingNavPct).toBeCloseTo(5.0);
  });

  test("Quick + Quicksand from same start produce different trims", () => {
    // COHR-style scenario from the bug report: NAV $600k, 302 sh @ $358.50.
    // 302 sh × $358.50 = $108,267 ≈ 18.0% NAV.
    //   Quick target  → floor(0.10 × 600000 / 358.50) = floor(167.4) = 167
    //   Quicksand     → floor(0.05 × 600000 / 358.50) = floor( 83.7) =  83
    //   Quick trim    = 302 - 167 = 135
    //   Quicksand trim= 302 -  83 = 219
    const cohr = { totalShares: 302, currentPrice: 358.50, nav: 600_000 };
    const quick = computeTrim(baseInput({ rule: "sr8-quick", ...cohr }));
    const sand = computeTrim(baseInput({ rule: "sr8-quicksand", ...cohr }));
    expect(quick.trimShares).toBe(135);
    expect(sand.trimShares).toBe(219);
    expect(quick.trimShares).not.toBe(sand.trimShares);
  });

  test("position already at/below target: trim is 0", () => {
    // 100 sh @ $358.50 = $35,850 ≈ 6.0% NAV. Below Quick's 10% target.
    const r = computeTrim(baseInput({
      rule: "sr8-quick", totalShares: 100, currentPrice: 358.50, nav: 600_000,
    }));
    expect(r.trimShares).toBe(0);
    expect(r.resultingShares).toBe(100);
    // Still below the 15% NAV core, so state is below-core (not core-only).
    expect(r.resultingState).toBe("below-core");
  });

  test("Quicksand sequential after Quick: targets 5% NAV from 10% start", () => {
    // After Quick the position is at 10% NAV. Quicksand drives to 5%.
    // 600 sh @ 100 = 60k = 10% NAV. Quicksand target 300 sh → trim 300.
    const r = computeTrim(baseInput({ rule: "sr8-quicksand", totalShares: 600 }));
    expect(r.trimShares).toBe(300);
    expect(r.resultingShares).toBe(300);
    expect(r.resultingNavPct).toBeCloseTo(5.0);
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
    // position above its 10% target. Base scenario: 1500 sh, core 900,
    // Quick targets 600 → trim 900, resulting 600 < core 900.
    const r = computeTrim(baseInput({ rule: "sr8-quick" }));
    expect(r.trimShares).toBe(900);
    expect(r.resultingShares).toBe(600);
    expect(r.resultingState).toBe("below-core");
  });
});
