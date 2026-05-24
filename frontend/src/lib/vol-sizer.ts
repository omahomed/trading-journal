/**
 * Pure volatility-sizing math for the Volatility Sizer page.
 *
 * Given equity, entry, MA-based tech stop, and ATR%, returns:
 *   - The tech-stop sizing scenario (bound by the risk budget OR the
 *     position tier cap, whichever is smaller).
 *   - Three ATR-cushion scenarios at fixed multipliers 1x / 1.5x / 2x.
 *   - A recommended scenario (tech stop when it sits >= 1 ATR away from
 *     entry; otherwise the 1.5x ATR scenario — sub-1-ATR stops get
 *     chopped by daily noise).
 *   - A warning when the tech stop sits inside 1 ATR.
 *
 * Degenerate / edge behavior:
 *   - `atrPct <= 0` throws `VolSizerError` (every ATR scenario would
 *     divide by zero risk-per-share).
 *   - `equity`, `entry`, `ma`, `tolPct`, `targetSizePct` must be > 0;
 *     otherwise throws.
 *   - `bufferPct` must be >= 0 (negative makes no sense; 0 means
 *     "stop exactly at MA" which is valid).
 *   - Tech stop at or above entry (e.g. `ma > entry` with small buffer)
 *     is NOT treated as an error — the tech-stop scenario returns
 *     `finalShares: 0` and the recommendation correctly falls through
 *     to the 1.5x ATR cushion.
 */

export interface VolSizerInputs {
  equity: number;
  entry: number;
  ma: number;
  bufferPct: number;
  atrPct: number;
  tolPct: number;
  targetSizePct: number;
}

export type ScenarioLabel = "Tech Stop" | "1x ATR" | "1.5x ATR" | "2x ATR";

export interface SizingScenario {
  label: ScenarioLabel;
  effectiveStop: number;
  stopDistancePct: number;
  atrFraction: number;
  candidateShares: number;
  finalShares: number;
  capBinds: boolean;
  positionCost: number;
  positionPct: number;
  riskIfStopped: number;
  riskPct: number;
}

export type RecommendationReason =
  | "tech_stop_safe"
  | "tech_stop_inside_noise";

export interface VolSizerResults {
  riskBudget: number;
  atrPerShare: number;
  positionCap: number;
  positionCapShares: number;

  techStop: SizingScenario;
  atrScenarios: [SizingScenario, SizingScenario, SizingScenario];

  recommended: SizingScenario;
  recommendationReason: RecommendationReason;

  warning: { show: boolean; text: string };
}

export class VolSizerError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "VolSizerError";
  }
}

const ATR_MULTIPLIERS: readonly [1, 1.5, 2] = [1, 1.5, 2] as const;

function labelFor(multiplier: 1 | 1.5 | 2): ScenarioLabel {
  if (multiplier === 1) return "1x ATR";
  if (multiplier === 1.5) return "1.5x ATR";
  return "2x ATR";
}

function buildScenario(args: {
  label: ScenarioLabel;
  entry: number;
  equity: number;
  atrPct: number;
  effectiveStop: number;
  riskBudget: number;
  positionCapShares: number;
}): SizingScenario {
  const { label, entry, equity, atrPct, effectiveStop, riskBudget, positionCapShares } = args;

  const riskPerShare = entry - effectiveStop;
  const stopDistancePct = (riskPerShare / entry) * 100;
  const atrFraction = stopDistancePct / atrPct;

  if (riskPerShare <= 0) {
    return {
      label,
      effectiveStop,
      stopDistancePct,
      atrFraction,
      candidateShares: 0,
      finalShares: 0,
      capBinds: false,
      positionCost: 0,
      positionPct: 0,
      riskIfStopped: 0,
      riskPct: 0,
    };
  }

  const candidateShares = Math.floor(riskBudget / riskPerShare);
  const finalShares = Math.min(candidateShares, positionCapShares);
  const capBinds = positionCapShares < candidateShares;
  const positionCost = finalShares * entry;
  const positionPct = (positionCost / equity) * 100;
  const riskIfStopped = finalShares * riskPerShare;
  const riskPct = (riskIfStopped / equity) * 100;

  return {
    label,
    effectiveStop,
    stopDistancePct,
    atrFraction,
    candidateShares,
    finalShares,
    capBinds,
    positionCost,
    positionPct,
    riskIfStopped,
    riskPct,
  };
}

export function computeVolatilitySizing(input: VolSizerInputs): VolSizerResults {
  const { equity, entry, ma, bufferPct, atrPct, tolPct, targetSizePct } = input;

  if (!(equity > 0)) throw new VolSizerError("equity must be > 0");
  if (!(entry > 0)) throw new VolSizerError("entry must be > 0");
  if (!(ma > 0)) throw new VolSizerError("ma must be > 0");
  if (!(bufferPct >= 0)) throw new VolSizerError("bufferPct must be >= 0");
  if (!(atrPct > 0)) throw new VolSizerError("atrPct must be > 0");
  if (!(tolPct > 0)) throw new VolSizerError("tolPct must be > 0");
  if (!(targetSizePct > 0)) throw new VolSizerError("targetSizePct must be > 0");

  const riskBudget = (equity * tolPct) / 100;
  const atrPerShare = (entry * atrPct) / 100;
  const positionCap = (equity * targetSizePct) / 100;
  const positionCapShares = Math.floor(positionCap / entry);

  const techEffectiveStop = ma * (1 - bufferPct / 100);
  const techStop = buildScenario({
    label: "Tech Stop",
    entry,
    equity,
    atrPct,
    effectiveStop: techEffectiveStop,
    riskBudget,
    positionCapShares,
  });

  const atrScenarios = ATR_MULTIPLIERS.map((m) => {
    const effStop = entry * (1 - (m * atrPct) / 100);
    return buildScenario({
      label: labelFor(m),
      entry,
      equity,
      atrPct,
      effectiveStop: effStop,
      riskBudget,
      positionCapShares,
    });
  }) as [SizingScenario, SizingScenario, SizingScenario];

  let recommended: SizingScenario;
  let recommendationReason: RecommendationReason;
  if (techStop.atrFraction >= 1.0) {
    recommended = techStop;
    recommendationReason = "tech_stop_safe";
  } else {
    recommended = atrScenarios[1];
    recommendationReason = "tech_stop_inside_noise";
  }

  const warningShow = techStop.atrFraction < 1.0;
  let warningText = "";
  if (warningShow) {
    if (techStop.stopDistancePct <= 0) {
      warningText =
        "Tech stop is at or above entry — invalid stop. Consider 1.5x ATR or skip.";
    } else {
      warningText =
        `Tech stop is ${techStop.atrFraction.toFixed(2)} ATR — daily noise will likely stop you out. Consider 1.5x ATR or skip.`;
    }
  }

  return {
    riskBudget,
    atrPerShare,
    positionCap,
    positionCapShares,
    techStop,
    atrScenarios,
    recommended,
    recommendationReason,
    warning: { show: warningShow, text: warningText },
  };
}
