// SR8 Trim Calculator — pure math.
//
// SR8 positions split into:
//   CORE  = fixed share count locked at SR8 activation (2026-07-18)
//   ADDS  = whatever's above the core (managed by SR7)
//
// ANCHORING INVARIANT (2026-07-18 fix). Prior to this rewrite, "core"
// was defined as 15% × LIVE NAV / current_price — a share count that
// grew with portfolio appreciation. When live NAV grew past activation
// NAV, SR8 Quick/Quicksand targets computed as (10% × live NAV / px)
// exceeded the fixed core share count → trim signals no-op'd → cores
// went undefended on valid signals. See BE regression (6/26): core
// 224 shs, old formula gave Quick target 319 shs (target > held →
// zero trim). New formula uses activation-day NLV → target 149 shs
// (valid 75-shs trim).
//
// The new contract:
//   activationNlv → the campaign's SR8_activation_nlv (fixed at the
//     moment cushion first crossed +50% from B1). Anchors Quick/QS
//     targets: quick_target = 0.10 × activationNlv / price.
//   coreShares → the fixed share count from activation. Directly
//     drives the ADDS math (adds = held − coreShares) so SR7 tiers
//     honor the anchor too.
//   nav (live) — kept for display metrics only (`totalNavPct`,
//     `resultingNavPct`). Never enters a trim-target computation.
//
// Legacy fallback: when activationNlv or coreShares is null (position
// pre-dates backfill or hasn't hit +50% cushion yet), the calc falls
// back to computing core as 15% × live nav — the OLD (buggy) formula.
// The result carries `anchorSource: 'live_fallback'` so the UI can
// flag it. Positions with a legit anchor are `'activation'`.
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
  /** Live NAV — display metrics only; never enters trim-target math. */
  nav: number;
  /** trades_summary.sr8_activation_nlv — fixed at SR8 activation, drives
   *  Quick/QS trim destinations. Null = position pre-dates backfill or
   *  hasn't crossed +50% cushion yet; falls back to live nav. */
  activationNlv?: number | null;
  /** trades_summary.sr8_core_shares — fixed at SR8 activation, drives
   *  the core-vs-ADDS split. Null → derive from activationNlv (or, if
   *  that's also null, from live nav — the legacy path). */
  coreShares?: number | null;
  rule: TrimRule;
}

export type AnchorSource = "activation" | "live_fallback";

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

  /** Which NLV drove the trim destination computation. 'activation'
   *  means the anchored formula fired; 'live_fallback' means the input
   *  lacked activationNlv/coreShares and we used live nav — the old
   *  buggy formula. UIs should badge live_fallback prominently. */
  anchorSource: AnchorSource;
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
  // Anchoring: prefer the fixed activation-day NLV + core_shares when
  // both are supplied; fall back to live nav (legacy path) otherwise.
  const anchorNlvRaw = input.activationNlv;
  const anchorSharesRaw = input.coreShares;
  const anchorNlvValid = anchorNlvRaw != null && Number.isFinite(anchorNlvRaw) && anchorNlvRaw > 0;
  const anchorSharesValid = anchorSharesRaw != null && Number.isFinite(anchorSharesRaw) && anchorSharesRaw > 0;
  const anchorSource: AnchorSource = (anchorNlvValid || anchorSharesValid) ? "activation" : "live_fallback";
  // Actual anchor $ used by trim-destination formulas.
  const anchorNlv = anchorNlvValid ? (anchorNlvRaw as number) : nav;

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
      anchorSource,
    };
  }

  // ────────────────────────────── Position state ──────────────────────────────
  // Core preference order:
  //   1. `coreShares` passed explicitly (canonical — fixed at activation)
  //   2. Derived from `activationNlv × 15% / price` (activation-anchored)
  //   3. Legacy `live nav × 15% / price` (bug-prone; only when neither
  //      anchor field is present).
  let coreTargetShares: number;
  let coreTargetValue: number;
  if (anchorSharesValid) {
    coreTargetShares = Math.floor(anchorSharesRaw as number);
    coreTargetValue = coreTargetShares * currentPrice;
  } else if (anchorNlvValid) {
    coreTargetValue = anchorNlv * 0.15;
    coreTargetShares = Math.floor(coreTargetValue / currentPrice);
  } else if (navValid) {
    coreTargetValue = nav * 0.15;
    coreTargetShares = Math.floor(coreTargetValue / currentPrice);
  } else {
    coreTargetValue = 0;
    coreTargetShares = 0;
  }
  // If the position is already below the core (heavy pullback or prior
  // trims), addsShares clamps to 0 — there's nothing for ADDS-bound
  // rules to trim.
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
      // TARGET-based (fixed anchor share count). Prior to 2026-07-18
      // this computed against live NAV, which inflated the target
      // count when NAV grew and let the trim no-op silently. The new
      // formula uses activation-day NLV (via anchorNlv):
      //
      //   Quick      → reduce to 10% × activation_NLV / current_price
      //   Quicksand  → reduce to  5% × activation_NLV / current_price
      //   Grateful   → 0 (unchanged; handled in the sr13/GD branch)
      //
      // If totalShares <= targetShares, trim is 0 (nothing to do —
      // position already at or below the destination). No core-floor
      // cap here: these rules are explicitly reducing the core itself.
      if (!(anchorNlv > 0)) {
        intendedTrimShares = 0;
        trimShares = 0;
      } else {
        const targetPct = rule === "sr8-quick" ? 0.10 : 0.05;
        const targetValue = anchorNlv * targetPct;
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
    anchorSource,
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
