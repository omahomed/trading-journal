/**
 * Pyramid Sizer — per-lot risk-accounted sizing for pyramid adds.
 *
 * Six-rule model (2026-07-18 redesign):
 *   1. LOCATION  price ≤ 21EMA + 1×ATR$/share   else block ("extended")
 *   2. PROGRESS  last held buy up ≥5% for full size; 0–5% prorated;
 *                below last buy = block
 *   3. BUDGET    campaign_budget = mode% × NAV
 *                campaign_risk   = Σ lot.shares × max(0, lot.entry − lot.stop)
 *                headroom        = budget − campaign_risk   ≤ 0 = block
 *   4. SIZE      composite       = MIN(Entry − 1 ATR, Key Level − max(0.5 ATR, 1%))
 *                risk_bound      = floor(headroom ÷ stop_dist)
 *                notional_cap    = floor(5% × NAV ÷ Entry)   ← per-add cap
 *                final_shares    = floor(min(risk_bound, notional_cap) × progress_mult)
 *   5. STOP      composite as above; trails 21 EMA − 0.5 ATR after entry
 *                (rising only). Handled at broker per pinned callout —
 *                this module only emits the initial composite.
 *   6. CEILING   (existing + final) × current_price ≤ 25% × NAV
 *                else clip final_shares down; if ≤ 0, block
 *
 * Composite stop uses the same MIN-of-two candidates the Volatility
 * Sizer uses — imported from ./vol-sizer to keep behavior in lockstep.
 * Any tweak to the composite formula must land there, not here.
 */

import { computeCompositeStop, type CompositeStop } from "./vol-sizer";

export interface PyramidLotInput {
  /** Shares currently held from this BUY row (already lot-closure netted). */
  shares: number;
  /** Cost basis for this lot (the BUY row's `amount` field). */
  entry: number;
  /** Current stop_loss for this lot (from the BUY row's stored value).
   *  When ≥ entry (or ≥ current price), the lot is treated as
   *  risk-free — max(0, entry − stop) collapses to 0 and it contributes
   *  nothing to campaign risk. */
  stopLoss: number;
  /** Optional identifier for the held-lots view row (trx_id like B1/A1). */
  label?: string;
}

export interface PyramidSizerInputs {
  equity: number;            // NAV
  entry: number;             // Entry price for the new add
  atrPct: number;            // ATR21 %
  ema21: number;             // 21 EMA from priceLookup
  keyLevel: number;          // User-typed anchor for the new add's composite
  tolPct: number;            // Sizing mode risk % (Pilot 0.25 / Normal 0.50 / …)
  heldLots: PyramidLotInput[];
  /** Current price on the ticker (rule 1 gate + rule 6 appreciation). */
  currentPrice: number;
  /** Price of the last held BUY row (rule 2). NaN / 0 means "no prior
   *  buy" — a fresh campaign; rule 2 is treated as PASS in that case. */
  lastHeldBuyPrice: number;
}

export type PyramidBind =
  | "risk"          // headroom is the binding constraint
  | "notional_cap"  // per-add 5% cap binds
  | "progress"      // prorated multiplier clips final shares
  | "ceiling"       // 25% NAV cap clips
  | "blocked";      // one or more gates blocked; final_shares = 0

export interface LotRiskView {
  /** Same fields as the input lot, plus computed risk. Enables the
   *  held-lots table to render whose stop is where. */
  shares: number;
  entry: number;
  stopLoss: number;
  risk: number;
  label?: string;
}

export interface GateResult {
  passed: boolean;
  reason?: string;
}

export interface LocationResult extends GateResult {
  /** Entry − (21EMA + 1 ATR) — negative when the gate passes, positive
   *  when we're extended above the line. Surfaces "by how much" to
   *  the user without a second calc. */
  ceilingPrice: number;
}

export interface ProgressResult extends GateResult {
  /** (current_price − last_held_buy_price) / last_held_buy_price × 100.
   *  NaN when there's no prior held buy — treated as PASS with
   *  multiplier=1 (fresh campaign start). */
  profitPct: number;
  /** 0.0 – 1.0. 0 when blocked, 1 when profit% ≥ 5, linear between. */
  multiplier: number;
  lastHeldBuyPrice: number;
}

export interface BudgetResult extends GateResult {
  budgetDollars: number;
  campaignRisk: number;
  headroom: number;
  lotRisks: LotRiskView[];
}

export interface CeilingResult extends GateResult {
  maxTotalNotional: number;
  projectedNotional: number;
}

export interface PyramidSizerResults {
  location: LocationResult;
  progress: ProgressResult;
  budget: BudgetResult;
  ceiling: CeilingResult;

  /** Composite stop for the NEW add (rule 5 initial value). */
  composite: CompositeStop;
  atrPerShare: number;
  stopDistance: number;

  /** Intermediate share counts, exposed so the UI can show which
   *  constraint bound the size. */
  riskBoundShares: number;
  notionalCapShares: number;
  progressCappedShares: number;

  finalShares: number;
  bind: PyramidBind;

  /** Position stats before the add. */
  existingShares: number;
  existingNotional: number;
  existingNotionalPct: number;

  /** Add stats. */
  addNotional: number;
  addNotionalPct: number;
  addRiskDollars: number;
  addRiskPct: number;

  /** Position stats after the add. */
  projectedShares: number;
  projectedNotional: number;
  projectedNotionalPct: number;

  /** True when any gate blocked (regardless of final_shares > 0). */
  blocked: boolean;
  blockReasons: string[];
}

export class PyramidSizerError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PyramidSizerError";
  }
}

/** Per-add cap (5% NAV) and campaign ceiling (25% NAV) constants.
 *  These are policy — not user-tunable per attempt. Change here means
 *  a global policy shift, worth its own commit. */
export const PYRAMID_ADD_CAP_PCT = 5;
export const PYRAMID_CAMPAIGN_CEILING_PCT = 25;
/** Rule 2 full-size threshold. Below this the multiplier prorates
 *  linearly toward zero; below zero, the add is blocked. */
export const PYRAMID_FULL_SIZE_TRIGGER_PCT = 5;
/** Rule 1 extension bound: how far above the 21 EMA (in ATR units)
 *  we tolerate before blocking. */
export const PYRAMID_LOCATION_ATR_MULTIPLE = 1;

function evaluateLocation(price: number, ema21: number, atrPerShare: number): LocationResult {
  const ceilingPrice = ema21 + PYRAMID_LOCATION_ATR_MULTIPLE * atrPerShare;
  if (!(ema21 > 0)) {
    return {
      passed: false,
      reason: "21 EMA unavailable — cannot evaluate location gate.",
      ceilingPrice: 0,
    };
  }
  if (price > ceilingPrice) {
    const extensionAtr = (price - ema21) / atrPerShare;
    return {
      passed: false,
      reason: `Extended ${extensionAtr.toFixed(2)}× ATR above 21 EMA. Rule 1 requires ≤ 1× ATR.`,
      ceilingPrice,
    };
  }
  return { passed: true, ceilingPrice };
}

function evaluateProgress(price: number, lastHeldBuyPrice: number): ProgressResult {
  if (!(lastHeldBuyPrice > 0)) {
    // No prior held buy → fresh campaign; treat as PASS at full size.
    // Belt-and-suspenders: pyramid is a pyramid, so this branch is
    // rarely reached in practice — but not throwing keeps the sizer
    // useful for a scenario tool.
    return {
      passed: true,
      profitPct: Number.NaN,
      multiplier: 1,
      lastHeldBuyPrice,
    };
  }
  const profitPct = ((price - lastHeldBuyPrice) / lastHeldBuyPrice) * 100;
  if (profitPct < 0) {
    return {
      passed: false,
      reason: `Below last held buy (${profitPct.toFixed(2)}%). Rule 2 requires ≥ 0%.`,
      profitPct,
      multiplier: 0,
      lastHeldBuyPrice,
    };
  }
  const multiplier = Math.min(1, profitPct / PYRAMID_FULL_SIZE_TRIGGER_PCT);
  return { passed: true, profitPct, multiplier, lastHeldBuyPrice };
}

function evaluateBudget(
  equity: number,
  tolPct: number,
  heldLots: PyramidLotInput[],
): BudgetResult {
  const budgetDollars = (equity * tolPct) / 100;
  const lotRisks: LotRiskView[] = heldLots.map((l) => ({
    shares: l.shares,
    entry: l.entry,
    stopLoss: l.stopLoss,
    risk: l.shares * Math.max(0, l.entry - l.stopLoss),
    label: l.label,
  }));
  const campaignRisk = lotRisks.reduce((s, r) => s + r.risk, 0);
  const headroom = budgetDollars - campaignRisk;
  if (headroom <= 0) {
    return {
      passed: false,
      reason: `No headroom: campaign risk $${campaignRisk.toFixed(0)} ≥ budget $${budgetDollars.toFixed(0)}.`,
      budgetDollars,
      campaignRisk,
      headroom,
      lotRisks,
    };
  }
  return { passed: true, budgetDollars, campaignRisk, headroom, lotRisks };
}

function evaluateCeiling(
  existingShares: number,
  proposedAdd: number,
  currentPrice: number,
  equity: number,
): CeilingResult {
  const maxTotalNotional = (equity * PYRAMID_CAMPAIGN_CEILING_PCT) / 100;
  const projectedNotional = (existingShares + proposedAdd) * currentPrice;
  if (projectedNotional > maxTotalNotional) {
    return {
      passed: false,
      reason: `Position would breach ${PYRAMID_CAMPAIGN_CEILING_PCT}% NAV ceiling.`,
      maxTotalNotional,
      projectedNotional,
    };
  }
  return { passed: true, maxTotalNotional, projectedNotional };
}

export function computePyramidSizing(input: PyramidSizerInputs): PyramidSizerResults {
  const { equity, entry, atrPct, ema21, keyLevel, tolPct, heldLots, currentPrice, lastHeldBuyPrice } = input;

  if (!(equity > 0)) throw new PyramidSizerError("equity must be > 0");
  if (!(entry > 0)) throw new PyramidSizerError("entry must be > 0");
  if (!(atrPct > 0)) throw new PyramidSizerError("atrPct must be > 0");
  if (!(keyLevel > 0)) throw new PyramidSizerError("keyLevel must be > 0");
  if (!(tolPct > 0)) throw new PyramidSizerError("tolPct must be > 0");
  if (!(currentPrice > 0)) throw new PyramidSizerError("currentPrice must be > 0");

  const atrPerShare = (entry * atrPct) / 100;
  const composite = computeCompositeStop({ entry, atrPct, keyLevel });
  const stopDistance = composite.distance;

  // Position stats before the add.
  const existingShares = heldLots.reduce((s, l) => s + l.shares, 0);
  const existingNotional = existingShares * currentPrice;
  const existingNotionalPct = equity > 0 ? (existingNotional / equity) * 100 : 0;

  // Gates.
  const location = evaluateLocation(currentPrice, ema21, atrPerShare);
  const progress = evaluateProgress(currentPrice, lastHeldBuyPrice);
  const budget = evaluateBudget(equity, tolPct, heldLots);

  const blockReasons: string[] = [];
  if (!location.passed && location.reason) blockReasons.push(location.reason);
  if (!progress.passed && progress.reason) blockReasons.push(progress.reason);
  if (!budget.passed && budget.reason) blockReasons.push(budget.reason);

  // Sizing math. When a gate has blocked, we still compute the raw
  // intermediates so the UI can show "would have been" numbers, but
  // final_shares clamps to 0. Ceiling is evaluated against the
  // pre-clip candidate to catch "already-at-ceiling" cleanly.
  const riskBoundShares = stopDistance > 0
    ? Math.floor(Math.max(0, budget.headroom) / stopDistance)
    : 0;
  const notionalCapShares = Math.floor((equity * PYRAMID_ADD_CAP_PCT) / 100 / entry);
  const preProgressShares = Math.min(riskBoundShares, notionalCapShares);
  const progressCappedShares = Math.floor(preProgressShares * progress.multiplier);

  const ceiling = evaluateCeiling(existingShares, progressCappedShares, currentPrice, equity);

  // Ceiling clip: shrink until within the 25% cap. If the existing
  // position ALONE is already at/over the ceiling, ceilingClipShares
  // goes to 0 (or negative — clamped).
  const ceilingRoomShares = Math.floor(
    Math.max(0, (equity * PYRAMID_CAMPAIGN_CEILING_PCT) / 100 - existingShares * currentPrice) / currentPrice,
  );
  const ceilingClippedShares = Math.min(progressCappedShares, ceilingRoomShares);

  let finalShares = 0;
  let blocked = false;
  if (blockReasons.length > 0) {
    blocked = true;
    finalShares = 0;
  } else if (!ceiling.passed && ceilingClippedShares <= 0) {
    // Ceiling gate hard-blocks only when there's zero room left.
    blocked = true;
    blockReasons.push(ceiling.reason || "");
    finalShares = 0;
  } else {
    finalShares = Math.max(0, ceilingClippedShares);
  }

  // Bind indicator (only meaningful when not blocked).
  let bind: PyramidBind;
  if (blocked || finalShares === 0) {
    bind = "blocked";
  } else if (finalShares < progressCappedShares) {
    bind = "ceiling";
  } else if (progress.multiplier < 1 && progressCappedShares < preProgressShares) {
    bind = "progress";
  } else if (notionalCapShares < riskBoundShares) {
    bind = "notional_cap";
  } else {
    bind = "risk";
  }

  // Post-add stats.
  const addNotional = finalShares * entry;
  const addNotionalPct = equity > 0 ? (addNotional / equity) * 100 : 0;
  const addRiskDollars = finalShares * stopDistance;
  const addRiskPct = equity > 0 ? (addRiskDollars / equity) * 100 : 0;
  const projectedShares = existingShares + finalShares;
  const projectedNotional = projectedShares * currentPrice;
  const projectedNotionalPct = equity > 0 ? (projectedNotional / equity) * 100 : 0;

  return {
    location,
    progress,
    budget,
    ceiling,
    composite,
    atrPerShare,
    stopDistance,
    riskBoundShares,
    notionalCapShares,
    progressCappedShares,
    finalShares,
    bind,
    existingShares,
    existingNotional,
    existingNotionalPct,
    addNotional,
    addNotionalPct,
    addRiskDollars,
    addRiskPct,
    projectedShares,
    projectedNotional,
    projectedNotionalPct,
    blocked,
    blockReasons,
  };
}
