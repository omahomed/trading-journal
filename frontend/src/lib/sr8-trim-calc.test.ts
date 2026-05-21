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

describe("computeTrim — SR8 Quick / Quicksand (5% NAV slice)", () => {
  test("slice within total: sells 5% NAV worth", () => {
    // NAV 600k, 5% = 30k. Px=100 → 300 sh slice.
    const r = computeTrim(baseInput({ rule: "sr8-quick" }));
    expect(r.intendedTrimShares).toBe(300);
    expect(r.trimShares).toBe(300);
    expect(r.resultingShares).toBe(1200);
  });

  test("slice > total: capped at total (position smaller than slice)", () => {
    // 200 sh @ 100 = 20k. Slice = 300. Cap to 200 (full exit).
    const r = computeTrim(baseInput({ rule: "sr8-quick", totalShares: 200 }));
    expect(r.intendedTrimShares).toBe(300);
    expect(r.trimShares).toBe(200);
    expect(r.resultingShares).toBe(0);
    expect(r.resultingState).toBe("closed");
  });

  test("Quicksand on the now-smaller position: same 5% NAV slice", () => {
    // After Quick: 1200 sh. Quicksand trims another 300 (5% NAV).
    const r = computeTrim(baseInput({ rule: "sr8-quicksand", totalShares: 1200 }));
    expect(r.trimShares).toBe(300);
    expect(r.resultingShares).toBe(900);
  });

  test("NAV=0: slice undefined → trim=0", () => {
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

  test("resulting < core (intentional SR13 over-trim): 'below-core'", () => {
    // SR13 forces full exit. resulting = 0 < core → 'closed' takes
    // priority over 'below-core'. To test below-core: contrive a
    // scenario where resulting > 0 but < core.
    // 1000 sh, core 900, SR2 trims 100 → resulting 900 (core-only).
    // To land below-core we'd need a rule that ignores the cap; the
    // only one is SR8-grateful-dead and SR13, but both go to 0.
    // SR8 Quick can do it: 1000 sh, slice 300 → resulting 700 < core 900.
    const r = computeTrim(baseInput({ rule: "sr8-quick", totalShares: 1000 }));
    expect(r.trimShares).toBe(300);
    expect(r.resultingShares).toBe(700);
    expect(r.resultingState).toBe("below-core");
  });
});
