// SR8 Trim Calculator — pure math.
//
// SR8 positions split into:
//   CORE  = 15% of NAV (managed by SR8's weekly MO RS triggers)
//   ADDS  = whatever's above the core (managed by SR7)
//
// When a sell rule fires on an SR8 position, this module answers:
// "given current shares, current price, NAV and the rule, how many
// shares should I sell?" The answer is the recommendation — the user
// executes manually at their broker and logs the sell later.
//
// All inputs are pure numbers; all share counts are integers via
// Math.floor. Never round up: floor avoids the trim dipping into the
// core when math lands on a boundary (e.g. core target 87.6 shares
// would round to 88 and a 25% trim could leave 87, below the core).
//
// CURRENT CUSHION as opposed to b1_max_return_pct: SR7 tiers off the
// LIVE cushion at fire time, not the historical peak. A position can
// be SR8 by historical peak but currently sit below 25% cushion after
// a pullback — that's exactly the case SR7's "<25% full exit of ADDS"
// branch is designed for.

export type TrimRule =
  | "sr2"
  | "sr7"
  | "sr8-quick"
  | "sr8-quicksand"
  | "sr8-grateful-dead"
  | "sr13";

export type SR7CushionTier = "lt25" | "25to50" | "gt50";

export type ResultingState =
  | "core-only"
  | "with-adds"
  | "below-core"
  | "closed"
  | "invalid";

export interface TrimInput {
  totalShares: number;
  currentPrice: number;
  // Cushion above B1 in % (b1_return_pct on EnrichedPosition). Used by
  // SR7 to pick a tier; other rules ignore it.
  b1ReturnPct: number | null;
  nav: number;
  rule: TrimRule;
}

export interface TrimResult {
  rule: TrimRule;
  sr7CushionTier?: SR7CushionTier;

  // Position state (computed)
  coreTargetValue: number;
  coreTargetShares: number;
  addsShares: number;
  currentCushionPct: number | null;
  totalValue: number;
  totalNavPct: number;

  // Trim derivation
  intendedTrimShares: number; // what the rule would trim without core protection
  trimShares: number; // actual amount to sell (capped at ADDS where applicable)
  coreFloorBinds: boolean; // true if cap reduced the trim

  // Resulting state
  resultingShares: number;
  resultingValue: number;
  resultingNavPct: number;
  resultingState: ResultingState;
}

function classifySR7Cushion(cushionPct: number | null): SR7CushionTier {
  // Null cushion → conservative default to <25% (full ADDS exit).
  // Same disposition as a fresh pullback below 25%.
  if (cushionPct == null || !Number.isFinite(cushionPct)) return "lt25";
  if (cushionPct >= 50) return "gt50";
  if (cushionPct >= 25) return "25to50";
  return "lt25";
}

export function computeTrim(input: TrimInput): TrimResult {
  const { totalShares, currentPrice, b1ReturnPct, nav, rule } = input;

  // Defensive guard: invalid price or non-positive shares makes the
  // math undefined (division by zero, negative shares). Bail with a
  // zero-trim result tagged 'invalid' so the caller can show a
  // diagnostic instead of NaN-poisoned numbers.
  const pxValid = Number.isFinite(currentPrice) && currentPrice > 0;
  const sharesValid = Number.isFinite(totalShares) && totalShares > 0;
  const navValid = Number.isFinite(nav) && nav > 0;

  if (!pxValid || !sharesValid) {
    return {
      rule,
      coreTargetValue: 0,
      coreTargetShares: 0,
      addsShares: 0,
      currentCushionPct: b1ReturnPct,
      totalValue: 0,
      totalNavPct: 0,
      intendedTrimShares: 0,
      trimShares: 0,
      coreFloorBinds: false,
      resultingShares: 0,
      resultingValue: 0,
      resultingNavPct: 0,
      resultingState: "invalid",
    };
  }

  // ────────────────────────────── Position state ──────────────────────────────
  const coreTargetValue = navValid ? nav * 0.15 : 0;
  const coreTargetShares = navValid ? Math.floor(coreTargetValue / currentPrice) : 0;
  // If the position is already below the 15% NAV core (heavy pullback
  // or prior trims), addsShares clamps to 0 — there's nothing for
  // ADDS-bound rules to trim.
  const addsShares = Math.max(0, totalShares - coreTargetShares);
  const totalValue = totalShares * currentPrice;
  const totalNavPct = navValid ? (totalValue / nav) * 100 : 0;

  // ────────────────────────────── Per-rule trim derivation ────────────────────
  let intendedTrimShares = 0;
  let trimShares = 0;
  let sr7CushionTier: SR7CushionTier | undefined;

  switch (rule) {
    case "sr2": {
      // Trim 25% of total, capped by ADDS — never sells into core.
      intendedTrimShares = Math.floor(totalShares * 0.25);
      trimShares = Math.min(intendedTrimShares, addsShares);
      break;
    }
    case "sr7": {
      sr7CushionTier = classifySR7Cushion(b1ReturnPct);
      if (sr7CushionTier === "gt50") {
        // Cushion >50%: full exit of ADDS (trim to core).
        intendedTrimShares = addsShares;
        trimShares = addsShares;
      } else if (sr7CushionTier === "25to50") {
        // Cushion 25–50%: trim 50% of total, capped at ADDS.
        intendedTrimShares = Math.floor(totalShares * 0.5);
        trimShares = Math.min(intendedTrimShares, addsShares);
      } else {
        // Cushion <25%: full exit of ADDS (same as >50% — but rule
        // documentation calls this "full exit" of the position; on an
        // SR8 position we still preserve the core).
        intendedTrimShares = addsShares;
        trimShares = addsShares;
      }
      break;
    }
    case "sr8-quick":
    case "sr8-quicksand": {
      // TARGET-based, not slice-based. The rule text "Trim 5% NAV
      // (15% → 10%)" describes the slice magnitude in the canonical
      // case where the position starts exactly at the 15% NAV core;
      // the operative semantic is the DESTINATION NAV %, not a fixed
      // slice. A position currently at 18% NAV trims more than 5% on
      // Quick (down to 10%); a position already at 10% trims zero.
      //
      //   Quick      → reduce to 10% NAV
      //   Quicksand  → reduce to  5% NAV
      //
      // If totalShares <= targetShares, trim is 0 (nothing to do —
      // position already at or below the destination). No core-floor
      // cap here: these rules are explicitly reducing the core itself.
      if (!navValid) {
        intendedTrimShares = 0;
        trimShares = 0;
      } else {
        const targetPct = rule === "sr8-quick" ? 0.10 : 0.05;
        const targetValue = nav * targetPct;
        const targetShares = Math.floor(targetValue / currentPrice);
        const trim = Math.max(0, totalShares - targetShares);
        intendedTrimShares = trim;
        trimShares = trim;
      }
      break;
    }
    case "sr8-grateful-dead":
    case "sr13": {
      // Full exit, including the SR8 core. SR13 character-change voids
      // the SR8 premise; SR8 Grateful Dead is the rule's one-way exit.
      intendedTrimShares = totalShares;
      trimShares = totalShares;
      break;
    }
  }

  const coreFloorBinds = intendedTrimShares > trimShares;
  const resultingShares = Math.max(0, totalShares - trimShares);
  const resultingValue = resultingShares * currentPrice;
  const resultingNavPct = navValid ? (resultingValue / nav) * 100 : 0;

  let resultingState: ResultingState;
  if (resultingShares === 0) {
    resultingState = "closed";
  } else if (resultingShares < coreTargetShares) {
    resultingState = "below-core";
  } else if (resultingShares === coreTargetShares) {
    resultingState = "core-only";
  } else {
    resultingState = "with-adds";
  }

  return {
    rule,
    sr7CushionTier,
    coreTargetValue,
    coreTargetShares,
    addsShares,
    currentCushionPct: b1ReturnPct,
    totalValue,
    totalNavPct,
    intendedTrimShares,
    trimShares,
    coreFloorBinds,
    resultingShares,
    resultingValue,
    resultingNavPct,
    resultingState,
  };
}

// Human-readable labels for the rule dropdown. Keep in sync with TrimRule.
export const RULE_OPTIONS: readonly { value: TrimRule; label: string; hint: string }[] = [
  { value: "sr2", label: "SR2 — Selling into Strength", hint: "Trim 25%, capped at ADDS" },
  { value: "sr7", label: "SR7 — 21e Violation", hint: "Cushion-tiered (auto)" },
  { value: "sr8-quick", label: "SR8 Quick — RS breaks 8w MA", hint: "Reduce to 10% NAV" },
  { value: "sr8-quicksand", label: "SR8 Quicksand — RS drifts further", hint: "Reduce to 5% NAV" },
  { value: "sr8-grateful-dead", label: "SR8 Grateful Dead — RS breaks 21w MA", hint: "Full exit" },
  { value: "sr13", label: "SR13 — Change of Character", hint: "Full exit including core" },
] as const;
