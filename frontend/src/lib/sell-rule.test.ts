import { describe, test, expect } from "vitest";
import {
  classifySellRuleTier,
  SELL_RULE_TIER_ORDER,
  isCushionQualified,
  needsSR15StopMove,
  needsSR12FloorMove,
  computeSR12FloorTarget,
} from "./sell-rule";

describe("classifySellRuleTier — post-migration-062 ladder", () => {
  test("returns null when b1 return is null/undefined/NaN/Infinity", () => {
    expect(classifySellRuleTier(null)).toBeNull();
    expect(classifySellRuleTier(undefined)).toBeNull();
    expect(classifySellRuleTier(NaN)).toBeNull();
    expect(classifySellRuleTier(Infinity)).toBeNull();
    expect(classifySellRuleTier(-Infinity)).toBeNull();
  });

  test("classifies sub-10% return as sr1", () => {
    expect(classifySellRuleTier(-50)).toBe("sr1");
    expect(classifySellRuleTier(-1)).toBe("sr1");
    expect(classifySellRuleTier(0)).toBe("sr1");
    expect(classifySellRuleTier(9.99)).toBe("sr1");
  });

  test("classifies 10%–20% return as sr11 (BE stop band)", () => {
    // Range tightened by migration 062 — was 10–50, now 10–20.
    expect(classifySellRuleTier(10)).toBe("sr11");
    expect(classifySellRuleTier(10.01)).toBe("sr11");
    expect(classifySellRuleTier(15)).toBe("sr11");
    expect(classifySellRuleTier(19.99)).toBe("sr11");
  });

  test("classifies 20%–50% return as sr15 (+10% profit-lock band)", () => {
    // NEW tier from migration 062 — physical broker stop at entry × 1.10.
    expect(classifySellRuleTier(20)).toBe("sr15");
    expect(classifySellRuleTier(20.01)).toBe("sr15");
    expect(classifySellRuleTier(35)).toBe("sr15");
    expect(classifySellRuleTier(49.99)).toBe("sr15");
  });

  test("classifies 50%+ return with declared=false as sr7", () => {
    // Cushion-qualified but undeclared — SR7 (21 EMA cascade).
    expect(classifySellRuleTier(50, null, false)).toBe("sr7");
    expect(classifySellRuleTier(50.01, null, false)).toBe("sr7");
    expect(classifySellRuleTier(120, null, false)).toBe("sr7");
    expect(classifySellRuleTier(300, null, undefined)).toBe("sr7");
    // Omitted flag defaults to undefined → SR7 (safe default).
    expect(classifySellRuleTier(80)).toBe("sr7");
  });

  test("classifies 50%+ return with declared=true as sr8", () => {
    // Declared monster hold — SR8 (weekly MO RS cascade + funnel ladder).
    expect(classifySellRuleTier(50, null, true)).toBe("sr8");
    expect(classifySellRuleTier(120, null, true)).toBe("sr8");
    expect(classifySellRuleTier(300, null, true)).toBe("sr8");
  });

  test("is_declared_sr8 is ignored in sub-50% bands (guard rail)", () => {
    // Declaration is only meaningful when cushion-qualified. Passing
    // is_declared_sr8=true on a sub-qualified campaign must NOT
    // promote it to SR8 — the backend guards the write, but the
    // classifier is defense-in-depth.
    expect(classifySellRuleTier(5, null, true)).toBe("sr1");
    expect(classifySellRuleTier(15, null, true)).toBe("sr11");
    expect(classifySellRuleTier(30, null, true)).toBe("sr15");
  });

  test("tier order ranks sr1 < sr11 < sr15 < sr7 < sr8", () => {
    // Defensive-progression ladder — lower rank = closer to open risk,
    // higher rank = more entrenched/defended. (SR14 retired 2026-08-07.)
    expect(SELL_RULE_TIER_ORDER.sr1).toBeLessThan(SELL_RULE_TIER_ORDER.sr11);
    expect(SELL_RULE_TIER_ORDER.sr11).toBeLessThan(SELL_RULE_TIER_ORDER.sr15);
    expect(SELL_RULE_TIER_ORDER.sr15).toBeLessThan(SELL_RULE_TIER_ORDER.sr7);
    expect(SELL_RULE_TIER_ORDER.sr7).toBeLessThan(SELL_RULE_TIER_ORDER.sr8);
  });

  // Post-migration-063 (2026-08-07 cleanup): SR14 was retired and
  // collapsed into SR1. broker_stop_price no longer promotes a tier —
  // it's surfaced as a 🛡 chip on the ACS row instead. Historical
  // stamps were retagged via migration 063.

  test("broker_stop_price no longer promotes to a distinct tier (post-063)", () => {
    // Under the pre-063 model these all promoted to "sr14"; post-063
    // they stay SR1. The row still shows a "broker stop parked" chip;
    // the tier semantics were the redundant part.
    expect(classifySellRuleTier(-5, 245.0)).toBe("sr1");
    expect(classifySellRuleTier(0, 100.0)).toBe("sr1");
    expect(classifySellRuleTier(9.99, 250.5)).toBe("sr1");
  });

  test("broker_stop_price is ignored across every band (post-063)", () => {
    // Same tier resolution regardless of whether a physical stop is
    // parked — the row-chip visibility is orthogonal to the ladder.
    expect(classifySellRuleTier(15, 245.0)).toBe("sr11");
    expect(classifySellRuleTier(30, 245.0)).toBe("sr15");
    expect(classifySellRuleTier(55, 245.0, false)).toBe("sr7");
    expect(classifySellRuleTier(120, 245.0, true)).toBe("sr8");
  });

  test("backwards-compat: single-arg + null/undefined broker_stop still work", () => {
    // Old callers that haven't been updated to pass is_declared_sr8 must
    // keep working; the broker_stop_price arg is retained for signature
    // stability but ignored.
    expect(classifySellRuleTier(5)).toBe("sr1");
    expect(classifySellRuleTier(9.99)).toBe("sr1");
    expect(classifySellRuleTier(5, null)).toBe("sr1");
    expect(classifySellRuleTier(5, undefined)).toBe("sr1");
    expect(classifySellRuleTier(15)).toBe("sr11");
    expect(classifySellRuleTier(30)).toBe("sr15");
    expect(classifySellRuleTier(80)).toBe("sr7");
  });
});

describe("isCushionQualified", () => {
  test("returns false for null/undefined/NaN/Infinity", () => {
    expect(isCushionQualified(null)).toBe(false);
    expect(isCushionQualified(undefined)).toBe(false);
    expect(isCushionQualified(NaN)).toBe(false);
    expect(isCushionQualified(Infinity)).toBe(false);
  });

  test("returns false below 50", () => {
    expect(isCushionQualified(0)).toBe(false);
    expect(isCushionQualified(49.99)).toBe(false);
  });

  test("returns true at or above 50", () => {
    expect(isCushionQualified(50)).toBe(true);
    expect(isCushionQualified(50.01)).toBe(true);
    expect(isCushionQualified(300)).toBe(true);
  });
});

describe("needsSR15StopMove", () => {
  // Anchor is B1 fill price (2nd arg) — not the blended avg_entry.
  // Restricted to the SR15 band [20%, 50%) — sticky ratchet doctrine
  // means once peak crosses 50 the +10% floor should already be parked
  // and nagging is stale.

  test("false below the 20% threshold", () => {
    expect(needsSR15StopMove(0, 100, null)).toBe(false);
    expect(needsSR15StopMove(19.99, 100, null)).toBe(false);
  });

  test("false at or above the 50% ceiling (SR7 / SR8 territory)", () => {
    // The DELL-at-166% case from 2026-08-07: declared SR8, must NOT
    // appear on the SR15 nudge banner. The +10% floor was meant to be
    // parked back when peak was 20-50; that ship has sailed.
    expect(needsSR15StopMove(50, 100, null)).toBe(false);
    expect(needsSR15StopMove(80, 100, null)).toBe(false);
    expect(needsSR15StopMove(166.47, 176.21, null)).toBe(false);
    expect(needsSR15StopMove(300, 100, null)).toBe(false);
  });

  test("true when in SR15 band with no broker stop parked", () => {
    // B1 entry 100 → target 110. No stop = definitely below target.
    expect(needsSR15StopMove(20, 100, null)).toBe(true);
    expect(needsSR15StopMove(25, 100, 0)).toBe(true);
    expect(needsSR15StopMove(49.99, 100, undefined)).toBe(true);
  });

  test("true when broker stop parked but still below B1-based target", () => {
    // B1 entry 100 → target 110. Stop at 105 = still needs a nudge.
    expect(needsSR15StopMove(30, 100, 105)).toBe(true);
    expect(needsSR15StopMove(45, 100, 109.99)).toBe(true);
  });

  test("false when broker stop is at or above the B1-based target", () => {
    // B1 entry 100 → target 110. Stop at 110 = target met.
    expect(needsSR15StopMove(30, 100, 110)).toBe(false);
    expect(needsSR15StopMove(45, 100, 115)).toBe(false);
  });

  test("target anchors on B1 fill even for a scaled-in campaign", () => {
    // The pre-fix bug: passing blended avg_entry ($331 on DELL after
    // add-ons) would make target = $364 when the correct target from
    // B1 ($176) is $193.83. Nudge must fire ONLY when the stop is
    // below the B1-derived target — so a stop of $200 clears the
    // nudge on a campaign whose blended avg is way higher.
    expect(needsSR15StopMove(30, 176.21, 200)).toBe(false);   // 200 > 193.83 target
    expect(needsSR15StopMove(30, 176.21, 190)).toBe(true);    // 190 < 193.83
  });

  test("false when B1 entry price is missing/invalid", () => {
    // Can't compute the target without a positive B1 fill price.
    expect(needsSR15StopMove(30, 0, null)).toBe(false);
    expect(needsSR15StopMove(30, -1, null)).toBe(false);
    expect(needsSR15StopMove(30, null, null)).toBe(false);
    expect(needsSR15StopMove(30, NaN, null)).toBe(false);
  });
});

describe("needsSR12FloorMove", () => {
  // Ratcheting Profit Floor nudge. Fires when the persisted (or
  // derived-from-peak) sr12_floor_pct puts the target broker stop above
  // the currently parked broker_stop_price. Takes over from SR15 at
  // the 50% band edge — clean handoff, no overlap.

  test("false when B1 entry price is missing/invalid", () => {
    expect(needsSR12FloorMove(80, 0, null, null)).toBe(false);
    expect(needsSR12FloorMove(80, -1, null, null)).toBe(false);
    expect(needsSR12FloorMove(80, null, null, null)).toBe(false);
    expect(needsSR12FloorMove(80, NaN, null, null)).toBe(false);
  });

  test("false when neither persisted floor nor cushion-qualified peak", () => {
    // Peak below 50 and no persisted floor → not armed.
    expect(needsSR12FloorMove(0, 100, null, null)).toBe(false);
    expect(needsSR12FloorMove(49.99, 100, null, null)).toBe(false);
    expect(needsSR12FloorMove(null, 100, null, null)).toBe(false);
  });

  test("armed via persisted sr12_floor_pct with no broker stop", () => {
    // Persisted floor of 50% on a $100 B1 → target $150. No stop = below.
    expect(needsSR12FloorMove(null, 100, null, 50)).toBe(true);
    expect(needsSR12FloorMove(null, 100, 0, 50)).toBe(true);
  });

  test("armed via derived floor when persisted is null and peak >= 50", () => {
    // Not-yet-reconciled row: peak 100 → derived floor 50 → target $150.
    expect(needsSR12FloorMove(100, 100, null, null)).toBe(true);
    expect(needsSR12FloorMove(100, 100, 140, null)).toBe(true);
  });

  test("persisted floor wins over derived even when peak is lower", () => {
    // Sticky ratchet doctrine: persisted floor persists even if peak was
    // recomputed downward (e.g. B1 lot got split-adjusted). Target is
    // driven by persisted, not by the current peak/2.
    // B1 $100, persisted floor 80 → target $180. Peak 60 would yield
    // $130 if we used derived — MUST NOT.
    expect(needsSR12FloorMove(60, 100, 150, 80)).toBe(true);   // 150 < 180
    expect(needsSR12FloorMove(60, 100, 185, 80)).toBe(false);  // 185 > 180
  });

  test("false when broker stop is at or above target", () => {
    // B1 $100, floor 50% → target $150.
    expect(needsSR12FloorMove(null, 100, 150, 50)).toBe(false);
    expect(needsSR12FloorMove(null, 100, 200, 50)).toBe(false);
  });

  test("DELL-style scenario (b1_entry $176.21, peak 166%)", () => {
    // Persisted floor should be 83; target = 176.21 * (1 + 0.83) = 322.4643.
    const target = 176.21 * 1.83;
    expect(needsSR12FloorMove(166, 176.21, target - 1, 83)).toBe(true);
    expect(needsSR12FloorMove(166, 176.21, target, 83)).toBe(false);
    // Broker stop just below target (half-a-penny tolerance edge).
    expect(needsSR12FloorMove(166, 176.21, target - 0.004, 83)).toBe(false);
    expect(needsSR12FloorMove(166, 176.21, target - 0.006, 83)).toBe(true);
  });

  test("handoff from SR15: peak = 50% exact", () => {
    // SR15 goes quiet at peak >= 50; SR12 takes over. Derived floor is
    // 25% at peak 50 → target = B1 * 1.25.
    // B1 $100, target $125.
    expect(needsSR12FloorMove(50, 100, 120, null)).toBe(true);
    expect(needsSR12FloorMove(50, 100, 125, null)).toBe(false);
  });

  test("zero/negative persisted floor is ignored", () => {
    // Backfill misfire safety — a rogue 0 or negative persisted value
    // must not disarm the nudge. Derived path still fires when the peak
    // qualifies. When neither exists, no nudge.
    expect(needsSR12FloorMove(80, 100, 130, 0)).toBe(true);     // falls back to peak/2 → target $140
    expect(needsSR12FloorMove(80, 100, 145, 0)).toBe(false);    // 145 > $140 target
    expect(needsSR12FloorMove(80, 100, 130, -5)).toBe(true);
    expect(needsSR12FloorMove(40, 100, 130, 0)).toBe(false);    // peak sub-50, no persisted → not armed
  });
});

describe("computeSR12FloorTarget", () => {
  test("null when unarmed / missing B1", () => {
    expect(computeSR12FloorTarget(30, 100, null)).toBeNull();
    expect(computeSR12FloorTarget(null, 100, null)).toBeNull();
    expect(computeSR12FloorTarget(80, null, null)).toBeNull();
    expect(computeSR12FloorTarget(80, 0, null)).toBeNull();
    expect(computeSR12FloorTarget(80, -1, null)).toBeNull();
  });

  test("derived from peak when persisted is null", () => {
    // Peak 100 on B1 $100 → target = 100 * 1.5 = 150.
    expect(computeSR12FloorTarget(100, 100, null)).toBeCloseTo(150, 6);
    expect(computeSR12FloorTarget(200, 100, null)).toBeCloseTo(200, 6);
  });

  test("persisted floor wins over derived", () => {
    // Persisted 80 on B1 $100 → target = 100 * 1.8 = 180.
    // Peak (60) would derive 130 — must NOT show up.
    expect(computeSR12FloorTarget(60, 100, 80)).toBeCloseTo(180, 6);
  });

  test("DELL: b1_entry $176.21, floor 83% → $322.4643", () => {
    expect(computeSR12FloorTarget(166, 176.21, 83)).toBeCloseTo(176.21 * 1.83, 6);
  });
});
