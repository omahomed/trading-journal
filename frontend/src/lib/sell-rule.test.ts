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

describe("needsSR12FloorMove — peak_total_pl anchor (post-065)", () => {
  // New signature: (peak_total_pl, realized_bank, shares, avg_entry, broker_stop_price).
  // Target broker stop = avg_entry + (peak_total_pl/2 − realized_bank) / shares.
  // If realized_bank already >= peak_total_pl/2, auto-clears (nothing left to protect).

  test("false when peak_total_pl is missing / non-positive", () => {
    expect(needsSR12FloorMove(null, 0, 100, 50, null)).toBe(false);
    expect(needsSR12FloorMove(0, 0, 100, 50, null)).toBe(false);
    expect(needsSR12FloorMove(-100, 0, 100, 50, null)).toBe(false);
    expect(needsSR12FloorMove(NaN, 0, 100, 50, null)).toBe(false);
  });

  test("false when shares or avg_entry are missing / non-positive", () => {
    expect(needsSR12FloorMove(10000, 0, 0, 50, null)).toBe(false);
    expect(needsSR12FloorMove(10000, 0, -5, 50, null)).toBe(false);
    expect(needsSR12FloorMove(10000, 0, null, 50, null)).toBe(false);
    expect(needsSR12FloorMove(10000, 0, 100, 0, null)).toBe(false);
    expect(needsSR12FloorMove(10000, 0, 100, null, null)).toBe(false);
  });

  test("auto-clears when realized_bank already exceeds peak/2", () => {
    // Peak $20k, already realized $12k > $10k target → nothing left to protect.
    expect(needsSR12FloorMove(20000, 12000, 100, 50, null)).toBe(false);
    expect(needsSR12FloorMove(20000, 10000, 100, 50, null)).toBe(false);
    // Just under target still fires.
    expect(needsSR12FloorMove(20000, 9999, 100, 50, null)).toBe(true);
  });

  test("fires when broker_stop_price is below target", () => {
    // Peak $10k, realized $0, 100 sh @ avg $50 → target realized $5k
    // → target stop = 50 + 5000/100 = $100. No broker stop = below.
    expect(needsSR12FloorMove(10000, 0, 100, 50, null)).toBe(true);
    expect(needsSR12FloorMove(10000, 0, 100, 50, 0)).toBe(true);
    expect(needsSR12FloorMove(10000, 0, 100, 50, 99)).toBe(true);
  });

  test("clears when broker_stop_price meets target", () => {
    expect(needsSR12FloorMove(10000, 0, 100, 50, 100)).toBe(false);
    expect(needsSR12FloorMove(10000, 0, 100, 50, 105)).toBe(false);
  });

  test("half-a-penny tolerance on the target boundary", () => {
    // Target $100 exactly. Nudge clears at 99.995+; fires below that.
    expect(needsSR12FloorMove(10000, 0, 100, 50, 99.996)).toBe(false);
    expect(needsSR12FloorMove(10000, 0, 100, 50, 99.994)).toBe(true);
  });

  test("DELL scenario — post-065 corrected math", () => {
    // avg_entry $331.62, shares 225, realized $36,349.21, peak $85,214.70.
    // target realized = $42,607.35; delta = $6,258.14;
    // target stop = 331.62 + 6258.14/225 = $359.43.
    const dellArgs = [85214.70, 36349.21, 225, 331.62] as const;
    // No stop = below target — fires.
    expect(needsSR12FloorMove(...dellArgs, null)).toBe(true);
    // Old (buggy) B1-anchored target $322.90 — still below correct target, fires.
    expect(needsSR12FloorMove(...dellArgs, 322.90)).toBe(true);
    // Correct target $359.43 — clears.
    expect(needsSR12FloorMove(...dellArgs, 359.43)).toBe(false);
    // Just above the tolerance edge — clears.
    expect(needsSR12FloorMove(...dellArgs, 359.42)).toBe(true);
  });

  test("SNDK-style already-locked-in scenario auto-clears", () => {
    // SNDK peak_total_pl $113,604. If the operator had trimmed enough to
    // realize $60k, the nudge should clear (>$56,802 target).
    expect(needsSR12FloorMove(113604.55, 60000, 50, 275, null)).toBe(false);
    // With less realized, still nudges.
    expect(needsSR12FloorMove(113604.55, 40000, 50, 275, null)).toBe(true);
  });
});

describe("computeSR12FloorTarget — peak_total_pl anchor (post-065)", () => {
  test("null when peak_total_pl / shares / avg_entry missing", () => {
    expect(computeSR12FloorTarget(null, 0, 100, 50)).toBeNull();
    expect(computeSR12FloorTarget(0, 0, 100, 50)).toBeNull();
    expect(computeSR12FloorTarget(10000, 0, 0, 50)).toBeNull();
    expect(computeSR12FloorTarget(10000, 0, 100, 0)).toBeNull();
  });

  test("null when realized_bank already exceeds peak/2", () => {
    // Already locked in more than half — no target to render.
    expect(computeSR12FloorTarget(20000, 15000, 100, 50)).toBeNull();
    expect(computeSR12FloorTarget(20000, 10000, 100, 50)).toBeNull();
  });

  test("returns target = avg_entry + (peak/2 − realized) / shares", () => {
    // Peak $10k, realized $0, 100 sh @ $50 → target $100.
    expect(computeSR12FloorTarget(10000, 0, 100, 50)).toBeCloseTo(100, 6);
    // With realized $2k, target = 50 + 3000/100 = $80.
    expect(computeSR12FloorTarget(10000, 2000, 100, 50)).toBeCloseTo(80, 6);
  });

  test("DELL: peak $85,214.70, realized $36,349.21, 225 sh @ $331.62 → $359.43", () => {
    expect(computeSR12FloorTarget(85214.70, 36349.21, 225, 331.62))
      .toBeCloseTo(359.43, 2);
  });

  test("null realized_bank treated as zero (fresh armed campaign)", () => {
    // Peak $10k, no realized, 100 sh @ $50 → target $100.
    expect(computeSR12FloorTarget(10000, null, 100, 50)).toBeCloseTo(100, 6);
    expect(computeSR12FloorTarget(10000, undefined, 100, 50)).toBeCloseTo(100, 6);
  });
});
