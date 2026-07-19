/**
 * Composite-stop volatility sizing for the Position Sizer's Volatility tab.
 *
 * The math:
 *   Risk Budget ($)   = NLV × Mode%                       (Pilot 0.25 / Normal 0.50 / Offense 0.75)
 *   Composite Stop    = MIN of:
 *                        • Entry − 1 ATR21                (ATR floor — never sizes tighter than 1 ATR)
 *                        • Key Level − max(0.5 ATR, 1%)   (structural low or key MA the user types from the chart)
 *   Stop Distance ($) = Entry − Composite Stop
 *   Raw shares        = Risk Budget ÷ Stop Distance
 *   Final shares      = min(Raw, Ceiling × NLV ÷ Entry)
 *                        Ceiling policy: 15% standard, 5% young-IPO clamp
 *                        (20% is the documented hard max but not auto-selectable — use Log Buy manually for conviction plays)
 *
 * The composite is intentionally the WIDEST of the two candidates (LOWEST
 * price = furthest from entry). That's the "give the trade room" rule:
 * whichever candidate sits further from entry wins, because taking the
 * tighter of the two would let the trade fail on noise the other candidate
 * says is still inside the setup. The trade-off is smaller position, which
 * is the intentionally-conservative tilt.
 *
 * Scale-out ladder anchored to ENTRY:
 *   Tier 1  at Entry − 0.5 ATR   (warning-shot partial)
 *   Tier 2  at Entry − 1.0 ATR   (composite anchor for most trades)
 *   Tier 3  at Entry − 1.5 ATR   (deep flush)
 *   Shares split floor / floor / remainder.
 *   Avg exit = 1 ATR below entry — matches the risk budget when the
 *   composite lands at 1 ATR. When composite is TIGHTER than 1 ATR (Key
 *   Level candidate won), tier 2 and 3 sit BELOW the composite; the ladder
 *   is intentionally saying "if the trade breaches your structural stop,
 *   give it more room, not less."
 *
 * Degenerate / edge behavior:
 *   - Non-positive equity / entry / atrPct / tolPct / keyLevel → VolSizerError.
 *   - Composite Stop ≥ Entry (bad key level, or entry inside a support
 *     that's actually resistance) → warning banner, finalShares = 0.
 *   - Key Level > Entry → still valid input; the key-level candidate just
 *     resolves to (KeyLevel − buffer) which may be above entry, in which
 *     case the ATR floor typically wins.
 */

export interface VolSizerInputs {
  /** Portfolio equity / NLV. */
  equity: number;
  /** Entry price per share. */
  entry: number;
  /** ATR21 as percentage (e.g. 4.5 for 4.5%). */
  atrPct: number;
  /** Structural low OR key MA — user types this from the chart. Used
   *  as the anchor for the key-level candidate stop. */
  keyLevel: number;
  /** Sizing-mode risk tolerance percentage (Pilot 0.25 / Normal 0.50 /
   *  Offense 0.75). */
  tolPct: number;
  /** True when the ticker is a young IPO (≤12 months since IPO by user
   *  judgment) — triggers the 5% ceiling clamp. Default false. */
  youngIpo?: boolean;
}

/** Which candidate stop won the composite MIN. */
export type CompositeWinner = "atr_floor" | "key_level_buffer";

export interface CompositeStop {
  /** Winning stop price (the LOWER of the two candidates). */
  price: number;
  /** Entry − price. */
  distance: number;
  /** distance / entry × 100. */
  distancePct: number;
  /** distance / atrPerShare — how many ATRs of headroom. Always ≥ 1
   *  by construction (the ATR floor guarantees it). */
  atrFraction: number;
  winner: CompositeWinner;
  /** Human-readable label for the winning candidate. */
  winnerLabel: string;
  candidates: {
    /** Entry − 1 ATR. Always available. */
    atrFloor: number;
    /** Key Level − max(0.5 ATR, 1% of entry). */
    keyLevelBuffer: number;
    /** The buffer applied under the key level, in $/share, for
     *  display: "Key Level − $X.XX buffer". */
    bufferApplied: number;
    /** Which of {0.5 × ATR, 1% of entry} the buffer resolved to. */
    bufferBasis: "half_atr" | "one_percent";
  };
}

/** Which constraint governed the final share count. */
export type Bind = "risk" | "ceiling";

/** Which policy tier the ceiling came from. */
export type CeilingPolicy = "standard" | "young_ipo";

export interface ScaleOutLeg {
  /** ATR multiplier below entry (0.5, 1.0, or 1.5). */
  atrMultiple: number;
  stopPrice: number;
  shares: number;
  loss: number;
  lossPctNlv: number;
}

export interface ScaleOutStops {
  entry: number;
  totalShares: number;
  legs: [ScaleOutLeg, ScaleOutLeg, ScaleOutLeg];
  totalLoss: number;
  totalLossPctNlv: number;
  /** Weighted average exit price. Equal thirds at 0.5/1.0/1.5 ATR
   *  give avg = entry − 1 ATR ≈ the risk budget anchor. */
  avgExitPrice: number;
  avgExitPct: number;
}

export interface VolSizerResults {
  riskBudget: number;
  atrPerShare: number;
  ceilingPct: number;
  ceilingShares: number;
  ceilingPolicy: CeilingPolicy;

  composite: CompositeStop;

  /** riskBudget ÷ composite.distance, floored. */
  candidateShares: number;
  /** min(candidateShares, ceilingShares). */
  finalShares: number;
  bind: Bind;
  positionCost: number;
  positionPct: number;
  riskIfStopped: number;
  riskPct: number;

  scaleOut: ScaleOutStops;

  /** Zero or more non-blocking advisories. Blocking errors throw. */
  warnings: string[];
}

export class VolSizerError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "VolSizerError";
  }
}

/** Position-size ceiling policy. `standard` = the everyday cap;
 *  `young_ipo` = the tighter clamp for IPOs ≤ 12 months old;
 *  `hard_max` = the absolute ceiling that even manual overrides can't
 *  breach (currently informational only — the sizer doesn't return it
 *  as a policy, but Log Buy consumers can enforce it). */
export const STANDARD_CEILING_PCT = 15;
export const YOUNG_IPO_CEILING_PCT = 5;
export const HARD_MAX_CEILING_PCT = 20;

/** Locked scale-out multipliers. Equal-thirds shares split at
 *  0.5 / 1.0 / 1.5 ATR below entry → avg exit at 1 ATR below entry,
 *  which is the risk-budget anchor when the composite lands at 1 ATR.
 *  The 1.25 ATR-avg proposal was rejected as a 25% risk-budget overrun
 *  — see the design conversation and comment on
 *  ScaleOutStops.avgExitPrice. */
export const SCALE_OUT_ATR_MULTIPLIERS: readonly [0.5, 1.0, 1.5] = [0.5, 1.0, 1.5] as const;

function winnerLabelOf(winner: CompositeWinner): string {
  return winner === "atr_floor" ? "1 ATR floor" : "Key Level − buffer";
}

/** Compute the composite stop from entry, ATR%, and the user's Key Level. */
export function computeCompositeStop(args: {
  entry: number;
  atrPct: number;
  keyLevel: number;
}): CompositeStop {
  const { entry, atrPct, keyLevel } = args;
  if (!(entry > 0)) throw new VolSizerError("entry must be > 0");
  if (!(atrPct > 0)) throw new VolSizerError("atrPct must be > 0");
  if (!(keyLevel > 0)) throw new VolSizerError("keyLevel must be > 0");

  const atrPerShare = (entry * atrPct) / 100;
  const atrFloor = entry - atrPerShare;

  // Buffer under Key Level scales with the name's volatility: max of
  // half an ATR or 1% of entry. Under a hot 9.6% name, 0.5 ATR wins;
  // under a calm 2% name, the 1% floor kicks in so we don't stop
  // uselessly tight when the name barely moves.
  const halfAtrBuffer = atrPerShare / 2;
  const onePctBuffer = entry * 0.01;
  const bufferApplied = Math.max(halfAtrBuffer, onePctBuffer);
  const bufferBasis: CompositeStop["candidates"]["bufferBasis"] =
    halfAtrBuffer >= onePctBuffer ? "half_atr" : "one_percent";
  const keyLevelBuffer = keyLevel - bufferApplied;

  // Composite = LOWEST price = WIDEST stop = MOST DEFENSIVE placement.
  // The narrower (higher price) candidate loses; the trade gets the
  // room the wider candidate says it needs.
  const winner: CompositeWinner = atrFloor <= keyLevelBuffer ? "atr_floor" : "key_level_buffer";
  const price = Math.min(atrFloor, keyLevelBuffer);
  const distance = entry - price;
  const distancePct = (distance / entry) * 100;
  const atrFraction = distance / atrPerShare;

  return {
    price,
    distance,
    distancePct,
    atrFraction,
    winner,
    winnerLabel: winnerLabelOf(winner),
    candidates: { atrFloor, keyLevelBuffer, bufferApplied, bufferBasis },
  };
}

/** Ceiling in percent-of-NLV for a given policy flag. */
export function ceilingPctFor(youngIpo: boolean | undefined): { pct: number; policy: CeilingPolicy } {
  return youngIpo
    ? { pct: YOUNG_IPO_CEILING_PCT, policy: "young_ipo" }
    : { pct: STANDARD_CEILING_PCT, policy: "standard" };
}

/** Split totalShares into three equal-ish legs (floor, floor, remainder)
 *  and price each leg at Entry − multiplier × ATR $/share. Locked ATR
 *  multipliers 0.5 / 1.0 / 1.5. */
export function computeScaleOutStops(entry: number, atrPct: number, totalShares: number, equity: number): ScaleOutStops {
  const shares = Math.max(0, Math.floor(totalShares));
  const atrPerShare = (entry * atrPct) / 100;
  const base = Math.floor(shares / 3);
  const legShares: [number, number, number] = [base, base, shares - 2 * base];

  const legs = SCALE_OUT_ATR_MULTIPLIERS.map((mult, i) => {
    const stopPrice = entry - mult * atrPerShare;
    const shs = legShares[i];
    const lossPerShare = entry - stopPrice;
    const loss = shs * lossPerShare;
    const lossPctNlv = equity > 0 ? (loss / equity) * 100 : 0;
    return { atrMultiple: mult, stopPrice, shares: shs, loss, lossPctNlv };
  }) as [ScaleOutLeg, ScaleOutLeg, ScaleOutLeg];

  const totalLoss = legs[0].loss + legs[1].loss + legs[2].loss;
  const totalLossPctNlv = equity > 0 ? (totalLoss / equity) * 100 : 0;
  // Share-weighted average exit. For equal thirds this is exactly
  // Entry − 1 ATR (the middle leg). Compute explicitly so uneven
  // remainders (small share counts) still land correctly.
  const totalStopValue = legs[0].shares * legs[0].stopPrice + legs[1].shares * legs[1].stopPrice + legs[2].shares * legs[2].stopPrice;
  const avgExitPrice = shares > 0 ? totalStopValue / shares : entry;
  const avgExitPct = entry > 0 ? -((entry - avgExitPrice) / entry) * 100 : 0;

  return { entry, totalShares: shares, legs, totalLoss, totalLossPctNlv, avgExitPrice, avgExitPct };
}

/** Compute the whole volatility sizer result — risk budget, composite
 *  stop, final shares, scale-out ladder — from user inputs. Pure. */
export function computeVolatilitySizing(input: VolSizerInputs): VolSizerResults {
  const { equity, entry, atrPct, keyLevel, tolPct, youngIpo } = input;

  if (!(equity > 0)) throw new VolSizerError("equity must be > 0");
  if (!(entry > 0)) throw new VolSizerError("entry must be > 0");
  if (!(atrPct > 0)) throw new VolSizerError("atrPct must be > 0");
  if (!(keyLevel > 0)) throw new VolSizerError("keyLevel must be > 0");
  if (!(tolPct > 0)) throw new VolSizerError("tolPct must be > 0");

  const riskBudget = (equity * tolPct) / 100;
  const atrPerShare = (entry * atrPct) / 100;

  const { pct: ceilingPct, policy: ceilingPolicy } = ceilingPctFor(youngIpo);
  const ceilingShares = Math.floor((equity * ceilingPct) / 100 / entry);

  const composite = computeCompositeStop({ entry, atrPct, keyLevel });

  const warnings: string[] = [];

  let candidateShares = 0;
  let finalShares = 0;
  let bind: Bind = "risk";
  if (composite.distance <= 0) {
    warnings.push(
      `Composite stop (${composite.price.toFixed(2)}) is at or above entry (${entry.toFixed(2)}) — Key Level too high, or the ATR is degenerate.`,
    );
  } else {
    candidateShares = Math.floor(riskBudget / composite.distance);
    finalShares = Math.min(candidateShares, ceilingShares);
    bind = ceilingShares < candidateShares ? "ceiling" : "risk";
  }

  const positionCost = finalShares * entry;
  const positionPct = equity > 0 ? (positionCost / equity) * 100 : 0;
  const riskIfStopped = finalShares * composite.distance;
  const riskPct = equity > 0 ? (riskIfStopped / equity) * 100 : 0;

  const scaleOut = computeScaleOutStops(entry, atrPct, finalShares, equity);

  return {
    riskBudget,
    atrPerShare,
    ceilingPct,
    ceilingShares,
    ceilingPolicy,
    composite,
    candidateShares,
    finalShares,
    bind,
    positionCost,
    positionPct,
    riskIfStopped,
    riskPct,
    scaleOut,
    warnings,
  };
}
