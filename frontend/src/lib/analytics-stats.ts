"use client";

import type { TradePosition, JournalHistoryPoint } from "./api";

const realized = (t: TradePosition) => parseFloat(String(t.realized_pl || 0));

export function computeWinRate(closed: TradePosition[]): number {
  if (closed.length === 0) return 0;
  const wins = closed.filter(t => realized(t) > 0).length;
  return (wins / closed.length) * 100;
}

export function computeProfitFactor(closed: TradePosition[]): number {
  const grossProfit = closed.filter(t => realized(t) > 0).reduce((a, t) => a + realized(t), 0);
  const grossLoss = Math.abs(closed.filter(t => realized(t) < 0).reduce((a, t) => a + realized(t), 0));
  return grossLoss > 0 ? grossProfit / grossLoss : 0;
}

export interface HoldRatioResult {
  winnersHold: number;
  losersHold: number;
  ratio: number;
}

export function computeHoldRatio(closed: TradePosition[]): HoldRatioResult {
  const holdDays = (t: TradePosition): number | null => {
    const oStr = String(t.open_date || "").trim();
    const cStr = String(t.closed_date || "").trim();
    if (!oStr || !cStr) return null;
    const open = new Date(oStr);
    const close = new Date(cStr);
    if (isNaN(open.getTime()) || isNaN(close.getTime())) return null;
    return Math.max(0, Math.floor((close.getTime() - open.getTime()) / 86400000));
  };
  const avgHold = (arr: TradePosition[]): number => {
    const valid = arr.map(holdDays).filter((d): d is number => d !== null);
    return valid.length > 0 ? valid.reduce((a, b) => a + b, 0) / valid.length : 0;
  };
  const wins = closed.filter(t => realized(t) > 0);
  const losses = closed.filter(t => realized(t) < 0);
  const winnersHold = avgHold(wins);
  const losersHold = avgHold(losses);
  const ratio = losersHold > 0 ? winnersHold / losersHold : 0;
  return { winnersHold, losersHold, ratio };
}

export interface OnePctCompliance {
  passRate: number;
  withinRule: number;
  breaches: number;
  totalLosses: number;
}

// 1% Rule: a closed loss is "within rule" if its account-impact %
// (realized_pl / NLV at open) is ≥ −1.0%. Caller passes the already-
// filtered set of closed losing trades for the year of interest, plus
// the journal history needed to look up NLV at each trade's open date.
//
// Mirrors the inline computation in analytics.tsx (Loss Discipline
// section). When totalLosses === 0, passRate is 100 by convention —
// callers that want a separate "no losses yet" state should branch on
// totalLosses themselves.
export function computeOnePctCompliance(
  closedLosses: TradePosition[],
  journalHistory: JournalHistoryPoint[],
): OnePctCompliance {
  const getNlvAtOpen = (openDate: string): number | null => {
    const d = String(openDate).slice(0, 10);
    const sorted = journalHistory
      .filter(h => String(h.day).slice(0, 10) <= d)
      .sort((a, b) => String(b.day).localeCompare(String(a.day)));
    return sorted.length > 0 ? sorted[0].end_nlv : null;
  };
  const impacts = closedLosses
    .map(t => {
      const nlv = getNlvAtOpen(t.open_date);
      return nlv && nlv > 0 ? (realized(t) / nlv) * 100 : null;
    })
    .filter((i): i is number => i !== null);
  const totalLosses = impacts.length;
  const withinRule = impacts.filter(i => i >= -1.0).length;
  const breaches = totalLosses - withinRule;
  const passRate = totalLosses > 0 ? (withinRule / totalLosses) * 100 : 100;
  return { passRate, withinRule, breaches, totalLosses };
}

// Window the most recent N closed trades by closed_date. Trades without
// a closed_date are excluded — they're either open (no closure) or have
// missing data. Caller passes the full closed-trade array; this helper
// owns the sort + slice. Used by Dashboard's Discipline Pulse panel for
// trailing-window metrics.
export function trailingClosedTrades(closed: TradePosition[], n: number): TradePosition[] {
  return [...closed]
    .filter(t => Boolean(t.closed_date))
    .sort((a, b) => String(b.closed_date || "").localeCompare(String(a.closed_date || "")))
    .slice(0, n);
}

// Same windowing applied to the losing subset. Useful for 1% Rule
// compliance tile, which is naturally scoped to losses.
export function trailingClosedLosses(closed: TradePosition[], n: number): TradePosition[] {
  return trailingClosedTrades(
    closed.filter(t => parseFloat(String(t.realized_pl || 0)) < 0),
    n,
  );
}

export type Last10Outcome = "win" | "loss" | "be";

export interface Last10Trade {
  trade_id: string;
  ticker: string;
  status: string;
  open_date: string;
  pl: number;
  outcome: Last10Outcome;
  rule?: string;
}

export interface Last10Stats {
  trades: Last10Trade[]; // oldest → newest (left → right of strip)
  count: number;
  winRate: number;
  netPl: number;
  avgWin: number;
  avgLoss: number;
  profitFactor: number;
  ltdWinRate: number;
  beDeadzone: number;
}

// Window the most recent N trades by open_date and compute summary
// stats. P&L per trade is whatever the caller passes — the contract
// is: open trades use overall_pl, closed trades use realized_pl.
// Outcomes are classified with a small breakeven dead-zone so trades
// that closed essentially flat don't pollute win/loss counts.
//
// ltdWinRate flows through unchanged for the "vs LTD X%" subtitle —
// caller-provided so this module doesn't need to know which window
// the comparison is against.
export function computeLast10Stats(
  trades: { trade_id: string; ticker: string; status: string; open_date: string; pl: number; rule?: string }[],
  ltdWinRate: number,
  windowSize: number = 10,
  beDeadzone: number = 50,
): Last10Stats {
  const sorted = [...trades].sort((a, b) =>
    String(b.open_date || "").localeCompare(String(a.open_date || ""))
  );
  const topN = sorted.slice(0, windowSize);
  const ordered = topN.reverse();
  const withOutcome: Last10Trade[] = ordered.map(t => ({
    ...t,
    outcome: Math.abs(t.pl) < beDeadzone ? "be" : t.pl > 0 ? "win" : "loss",
  }));
  const count = withOutcome.length;
  const wins = withOutcome.filter(t => t.outcome === "win");
  const losses = withOutcome.filter(t => t.outcome === "loss");
  const grossProfit = wins.reduce((a, t) => a + t.pl, 0);
  const grossLoss = Math.abs(losses.reduce((a, t) => a + t.pl, 0));
  const winRate = count > 0 ? (wins.length / count) * 100 : 0;
  const netPl = withOutcome.reduce((a, t) => a + t.pl, 0);
  const avgWin = wins.length > 0 ? grossProfit / wins.length : 0;
  const avgLoss = losses.length > 0 ? -grossLoss / losses.length : 0;
  const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : 0;
  return { trades: withOutcome, count, winRate, netPl, avgWin, avgLoss, profitFactor, ltdWinRate, beDeadzone };
}


// ═══════════════════════════════════════════════════════════════════════
// Edge Report helpers (Overview + Scenarios tabs)
//
// All pure functions. Take already-filtered trade arrays; caller owns the
// cohort / year-window slicing.
// ═══════════════════════════════════════════════════════════════════════


/** Prior-day NLV lookup — the END NLV on the last journal day STRICTLY
 *  before `openDate`. Used as the denominator for account-impact %,
 *  Brandt NAV normalization, and fixed-size scaling.
 *
 *  Strict `<` (not `<=`) is intentional: end_nlv on the open_date row
 *  reflects that day's activity INCLUDING the trade being evaluated, so
 *  using `<=` puts the loss in its own denominator and understates
 *  impact %. See the analytics.tsx Loss Discipline commit for the
 *  historical context. */
export function getPriorDayNlv(
  journal: JournalHistoryPoint[],
  openDate: string,
): number | null {
  const d = String(openDate).slice(0, 10);
  if (!d) return null;
  const sorted = journal
    .filter(h => String(h.day).slice(0, 10) < d)
    .sort((a, b) => String(b.day).localeCompare(String(a.day)));
  return sorted.length > 0 ? Number(sorted[0].end_nlv ?? 0) || null : null;
}


/** "Trade was open at any point during year Y" — the year filter used
 *  by the Edge Report toggle. Trades that closed BEFORE year start are
 *  dropped; trades that opened AFTER year end are dropped; anything
 *  else stays. Undefined / missing dates fall through — better to
 *  include an under-annotated trade and let the aggregate cards show
 *  a "—" than to silently drop it. */
export function tradeWasOpenInYear(t: TradePosition, year: number): boolean {
  const cd = String(t.closed_date || "").slice(0, 10);
  const od = String(t.open_date || "").slice(0, 10);
  const yearStart = `${year}-01-01`;
  const yearEnd = `${year}-12-31`;
  if (od && od > yearEnd) return false;
  if (cd && cd < yearStart) return false;
  return true;
}


/** Years derived from the loaded data — smallest open_year present up
 *  to max(current_year, largest closed_year). The picker defaults to
 *  the largest year that actually has ≥1 trade; that stays stable at
 *  year rollover instead of jumping to an empty new year on Jan 1. */
export function availableTradeYears(
  closed: TradePosition[],
  open: TradePosition[],
): { years: number[]; defaultYear: number } {
  const raw: number[] = [];
  const push = (d?: string | null) => {
    const s = String(d || "").slice(0, 4);
    const y = Number(s);
    if (Number.isFinite(y) && y >= 2000 && y <= 2100) raw.push(y);
  };
  closed.forEach(t => { push(t.open_date); push(t.closed_date); });
  open.forEach(t => { push(t.open_date); });
  const now = new Date().getFullYear();
  if (raw.length === 0) return { years: [now], defaultYear: now };
  const min = Math.min(...raw);
  const max = Math.max(...raw, now);
  const years: number[] = [];
  for (let y = min; y <= max; y++) years.push(y);
  // Default = most recent year that actually contains a trade — avoids
  // the empty-current-year problem on Jan 1 before any 2027 trade lands.
  const maxWithData = Math.max(...raw);
  return { years, defaultYear: maxWithData };
}


// ────────────────────────────────────────────────────────────────────
// Pareto distribution
// ────────────────────────────────────────────────────────────────────

export interface ParetoRank {
  trade_id: string;
  ticker: string;
  pnl: number;
  cumulative: number;
  cumPctOfNet: number;
}
export interface ParetoResult {
  ranks: ParetoRank[];
  netPl: number;
  /** Rank at which cumulative P&L first meets or exceeds netPl. Null
   *  when netPl ≤ 0 (no "break-even rank" concept applies). */
  breakevenRank: number | null;
  /** Top-N summary. Returns 0/0 for n ≤ 0 or empty ranks. */
  topN: (n: number) => { net: number; pctOfNet: number; count: number };
}

/** Cumulative-P&L Pareto distribution: ranks all trades descending by
 *  P&L, tracks a running total, and expresses each cumulative point as
 *  a % of net. Motivating output shape: "Top 5 trades (2%) = 79% of net
 *  · top 10 = 103% · trades 11+ net ≈ $0." */
export function paretoDistribution(
  trades: TradePosition[],
  pnlOf: (t: TradePosition) => number = t => realized(t),
): ParetoResult {
  const ranks: ParetoRank[] = [];
  const sorted = [...trades].sort((a, b) => pnlOf(b) - pnlOf(a));
  const netPl = sorted.reduce((a, t) => a + pnlOf(t), 0);
  let running = 0;
  for (const t of sorted) {
    const p = pnlOf(t);
    running += p;
    ranks.push({
      trade_id: String(t.trade_id || ""),
      ticker: String(t.ticker || ""),
      pnl: p,
      cumulative: running,
      cumPctOfNet: netPl !== 0 ? (running / netPl) * 100 : 0,
    });
  }
  let breakevenRank: number | null = null;
  if (netPl > 0) {
    for (let i = 0; i < ranks.length; i++) {
      if (ranks[i].cumulative >= netPl) { breakevenRank = i + 1; break; }
    }
  }
  return {
    ranks,
    netPl,
    breakevenRank,
    topN(n: number) {
      if (n <= 0 || ranks.length === 0) return { net: 0, pctOfNet: 0, count: 0 };
      const slice = ranks.slice(0, Math.min(n, ranks.length));
      const net = slice.reduce((a, r) => a + r.pnl, 0);
      const pctOfNet = netPl !== 0 ? (net / netPl) * 100 : 0;
      return { net, pctOfNet, count: slice.length };
    },
  };
}


// ────────────────────────────────────────────────────────────────────
// Hold-time buckets
// ────────────────────────────────────────────────────────────────────

export interface HoldTimeBucket {
  label: string;
  loInclusive: number;
  hiInclusive: number | null; // null = "and up"
  n: number;
  winRate: number;
  netPl: number;
}

/** Bucket lineup matches the Edge Report spec: 0–1d / 2–5d / 6–15d /
 *  16–40d / 41d+. Hold days = floor((close - open) / 1 day). Open
 *  trades (no closed_date) are included only when caller passes an
 *  "as-if-closed" cohort — see enrichOpenAsClosed. */
export function holdTimeBuckets(
  trades: TradePosition[],
  pnlOf: (t: TradePosition) => number = t => realized(t),
): HoldTimeBucket[] {
  const bucketDefs: { label: string; lo: number; hi: number | null }[] = [
    { label: "0–1 days",   lo:  0, hi:  1 },
    { label: "2–5 days",   lo:  2, hi:  5 },
    { label: "6–15 days",  lo:  6, hi: 15 },
    { label: "16–40 days", lo: 16, hi: 40 },
    { label: "41+ days",   lo: 41, hi: null },
  ];
  const holdDays = (t: TradePosition): number | null => {
    const o = String(t.open_date || "").trim();
    const c = String(t.closed_date || "").trim();
    if (!o || !c) return null;
    const od = new Date(o).getTime();
    const cd = new Date(c).getTime();
    if (isNaN(od) || isNaN(cd)) return null;
    return Math.max(0, Math.floor((cd - od) / 86_400_000));
  };
  return bucketDefs.map(b => {
    const inBucket = trades.filter(t => {
      const d = holdDays(t);
      if (d == null) return false;
      if (d < b.lo) return false;
      if (b.hi != null && d > b.hi) return false;
      return true;
    });
    const wins = inBucket.filter(t => pnlOf(t) > 0);
    const netPl = inBucket.reduce((a, t) => a + pnlOf(t), 0);
    return {
      label: b.label,
      loInclusive: b.lo,
      hiInclusive: b.hi,
      n: inBucket.length,
      winRate: inBucket.length > 0 ? (wins.length / inBucket.length) * 100 : 0,
      netPl,
    };
  });
}


// ────────────────────────────────────────────────────────────────────
// Brandt NAV normalization
// ────────────────────────────────────────────────────────────────────

export interface BrandtNormalized {
  avgTradePctNav: number | null;   // expectancy as % of NAV
  avgWinPctNav: number | null;
  avgLossPctNav: number | null;    // signed ≤ 0
  n: number;
  nWithNlv: number;                // trades that had a prior-day NLV
}

/** Expectancy / avg winner / avg loser as % of the prior-day NAV at
 *  each trade's open. Trades without a prior-day journal row are
 *  excluded from the percentage math but counted toward n; the caller
 *  can compare n vs nWithNlv to flag coverage gaps. */
export function brandtNormalized(
  trades: TradePosition[],
  journal: JournalHistoryPoint[],
  pnlOf: (t: TradePosition) => number = t => realized(t),
): BrandtNormalized {
  const pctPerTrade: number[] = [];
  const pctWins: number[] = [];
  const pctLosses: number[] = [];
  for (const t of trades) {
    const nlv = getPriorDayNlv(journal, String(t.open_date || ""));
    if (nlv == null || nlv <= 0) continue;
    const pct = (pnlOf(t) / nlv) * 100;
    pctPerTrade.push(pct);
    if (pnlOf(t) > 0) pctWins.push(pct);
    else if (pnlOf(t) < 0) pctLosses.push(pct);
  }
  const mean = (arr: number[]): number | null =>
    arr.length > 0 ? arr.reduce((a, v) => a + v, 0) / arr.length : null;
  return {
    n: trades.length,
    nWithNlv: pctPerTrade.length,
    avgTradePctNav: mean(pctPerTrade),
    avgWinPctNav: mean(pctWins),
    avgLossPctNav: mean(pctLosses),
  };
}


// ────────────────────────────────────────────────────────────────────
// Stop-cap scenario (upper bound — no MAE yet)
// ────────────────────────────────────────────────────────────────────

export interface StopCapRow {
  capPct: number;
  breachCount: number;
  dollarsSaved: number;
  /** Placeholder for future MAE-aware backtest — number of winners
   *  that would have been prematurely stopped. Zero today because the
   *  helper doesn't yet see MAE data; the frontend renders the "upper
   *  bound" caveat until this drops below Infinity. */
  clippedWinnerCount: number;
  clippedWinnerForegonePl: number;
}

/** Stop-cap upper bound: for each cap X, count losers whose |return_pct|
 *  exceeded X and sum the "dollars past the cap." Assumes no winner
 *  would have been prematurely stopped — a real backtest requires MAE
 *  data (which is coming from the separate MAE/MFE branch).
 *
 *  When MAE data lands, callers can pass a `maePctOf` to enable the
 *  clipped-winner accounting (item 8 in the Edge Report spec). Until
 *  then the helper returns 0 for `clippedWinnerCount` and callers show
 *  the "upper bound" caveat banner on the tab. */
export function stopCapScenario(
  trades: TradePosition[],
  caps: readonly number[] = [3, 4, 5, 6, 7, 8, 10],
  opts: {
    returnPctOf?: (t: TradePosition) => number;
    totalCostOf?: (t: TradePosition) => number;
    /** Optional MAE-aware winner clipping. When provided, winners whose
     *  MAE went beyond -X% are treated as "clipped" (stopped early)
     *  and their realized P&L is counted as foregone. */
    maePctOf?: (t: TradePosition) => number | null;
  } = {},
): StopCapRow[] {
  const returnPctOf = opts.returnPctOf ?? ((t: TradePosition) => Number((t as any).return_pct ?? 0));
  const totalCostOf = opts.totalCostOf ?? ((t: TradePosition) => Number(t.total_cost ?? 0));
  const maePctOf = opts.maePctOf;
  return caps.map(X => {
    let dollarsSaved = 0;
    let breachCount = 0;
    let clippedWinnerCount = 0;
    let clippedWinnerForegonePl = 0;
    for (const t of trades) {
      const rp = returnPctOf(t);
      const cost = totalCostOf(t);
      if (rp < -X && cost > 0) {
        breachCount += 1;
        dollarsSaved += ((Math.abs(rp) - X) / 100) * cost;
      }
      if (maePctOf) {
        const mae = maePctOf(t);
        const realizedPl = realized(t);
        if (mae != null && mae < -X && realizedPl > 0) {
          clippedWinnerCount += 1;
          clippedWinnerForegonePl += realizedPl;
        }
      }
    }
    return { capPct: X, breachCount, dollarsSaved, clippedWinnerCount, clippedWinnerForegonePl };
  });
}


// ────────────────────────────────────────────────────────────────────
// Fixed-size scenario
// ────────────────────────────────────────────────────────────────────

export interface FixedSizeRow {
  targetPct: number;
  scaledPnl: number;
  nWithSize: number;   // trades that had a resolvable b_size_pct
  nDropped: number;    // trades missing either b_cost or prior_nlv
}

/** Fixed-size upper bound: what would the total P&L be if every trade
 *  had been sized to T% of prior-day NAV, scaling linearly from its
 *  actual b1 position size? Caveat: ignores margin/heat — spec's
 *  "linear scaling" note. */
export function fixedSizeScenario(
  trades: TradePosition[],
  journal: JournalHistoryPoint[],
  targets: readonly number[] = [5, 7.5, 10, 12.5, 15],
  opts: {
    /** Total cost of the B1 (initial) lots. Defaults to total_cost, but
     *  callers with a proper B1-slice can pass a tighter helper. */
    b1CostOf?: (t: TradePosition) => number;
    pnlOf?: (t: TradePosition) => number;
  } = {},
): FixedSizeRow[] {
  const b1CostOf = opts.b1CostOf ?? ((t: TradePosition) => Number(t.total_cost ?? 0));
  const pnlOf = opts.pnlOf ?? realized;
  return targets.map(T => {
    let scaledPnl = 0;
    let nWithSize = 0;
    let nDropped = 0;
    for (const t of trades) {
      const cost = b1CostOf(t);
      const priorNlv = getPriorDayNlv(journal, String(t.open_date || ""));
      if (!(cost > 0) || !priorNlv || priorNlv <= 0) { nDropped += 1; continue; }
      const bSizePct = (cost / priorNlv) * 100;
      if (bSizePct <= 0) { nDropped += 1; continue; }
      scaledPnl += pnlOf(t) * (T / bSizePct);
      nWithSize += 1;
    }
    return { targetPct: T, scaledPnl, nWithSize, nDropped };
  });
}


// ────────────────────────────────────────────────────────────────────
// Risk metrics — aggregate stop distance, loss size, position size
// ────────────────────────────────────────────────────────────────────

export interface RiskMetrics {
  n: number;
  /** avg (entry - stop) / entry * 100, over trades with stop_loss > 0 */
  avgStopDistancePct: number | null;
  nWithStop: number;
  /** avg return_pct across losing trades (signed, ≤ 0) */
  avgRealizedLossPct: number | null;
  nLosers: number;
  /** median (total_cost / prior_day_nlv * 100) — actual position size */
  medianPositionSizePct: number | null;
  /** mean of the same */
  avgPositionSizePct: number | null;
  nWithPositionSize: number;
  /** theoretical risk budget = avg position % × avg stop distance % / 100.
   *  Uses the two averages above; null when either is null. */
  avgRiskPerTradePct: number | null;
}

/** Aggregate risk-shape metrics for the cohort. Mirrors the New Entry
 *  page's math (stop distance from entry, position size as % of prior-day
 *  NAV) so the operator sees the actualized-vs-planned side of the risk
 *  formula. Coverage counts included because stop_loss population is
 *  ~55% in production data. */
export function riskMetrics(
  trades: TradePosition[],
  journal: JournalHistoryPoint[],
): RiskMetrics {
  const stopDistances: number[] = [];
  const positionSizes: number[] = [];
  const losses: number[] = [];
  for (const t of trades) {
    const entry = Number((t as any).avg_entry ?? 0);
    const stop = Number((t as any).stop_loss ?? 0);
    if (entry > 0 && stop > 0 && stop < entry) {
      stopDistances.push(((entry - stop) / entry) * 100);
    }
    const cost = Number(t.total_cost ?? 0);
    const priorNlv = getPriorDayNlv(journal, String(t.open_date || ""));
    if (cost > 0 && priorNlv && priorNlv > 0) {
      positionSizes.push((cost / priorNlv) * 100);
    }
    const rp = Number((t as any).return_pct ?? NaN);
    const pl = realized(t);
    if (pl < 0 && Number.isFinite(rp)) losses.push(rp);
  }
  const mean = (arr: number[]): number | null =>
    arr.length > 0 ? arr.reduce((a, v) => a + v, 0) / arr.length : null;
  const median = (arr: number[]): number | null => {
    if (arr.length === 0) return null;
    const s = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(s.length / 2);
    return s.length % 2 === 0 ? (s[mid - 1] + s[mid]) / 2 : s[mid];
  };
  const avgStop = mean(stopDistances);
  const avgPos = mean(positionSizes);
  const avgRisk = (avgStop != null && avgPos != null)
    ? (avgPos * avgStop) / 100
    : null;
  return {
    n: trades.length,
    avgStopDistancePct: avgStop,
    nWithStop: stopDistances.length,
    avgRealizedLossPct: mean(losses),
    nLosers: losses.length,
    medianPositionSizePct: median(positionSizes),
    avgPositionSizePct: avgPos,
    nWithPositionSize: positionSizes.length,
    avgRiskPerTradePct: avgRisk,
  };
}


// ────────────────────────────────────────────────────────────────────
// Repeat-offender tickers
// ────────────────────────────────────────────────────────────────────

export interface RepeatOffender {
  ticker: string;
  attempts: number;
  wins: number;
  losses: number;
  netPl: number;
  avgPl: number;
  bestPl: number;
  worstPl: number;
  trades: TradePosition[];
}

/** Group trades by ticker; return only those with ≥ minAttempts entries
 *  in the cohort. Sorted by ascending netPl (biggest bleeders surface
 *  first — matches the PDF Section 5 "GEV / LEU / LITE / SOXL" framing).
 *  Options are joined into their underlying via the leading symbol. */
export function repeatOffenders(
  trades: TradePosition[],
  opts: { minAttempts?: number; pnlOf?: (t: TradePosition) => number } = {},
): RepeatOffender[] {
  const minAttempts = opts.minAttempts ?? 3;
  const pnlOf = opts.pnlOf ?? realized;
  const underlyingOf = (t: TradePosition): string => {
    const raw = String(t.ticker || "").trim();
    if ((t as any).instrument_type === "OPTION") {
      const first = raw.split(/\s+/)[0];
      return first || raw;
    }
    return raw;
  };
  const byTicker = new Map<string, TradePosition[]>();
  for (const t of trades) {
    const key = underlyingOf(t);
    if (!key) continue;
    const list = byTicker.get(key);
    if (list) list.push(t);
    else byTicker.set(key, [t]);
  }
  const out: RepeatOffender[] = [];
  for (const [ticker, arr] of byTicker.entries()) {
    if (arr.length < minAttempts) continue;
    const pnls = arr.map(pnlOf);
    const netPl = pnls.reduce((a, v) => a + v, 0);
    const wins = pnls.filter(p => p > 0).length;
    const losses = pnls.filter(p => p < 0).length;
    out.push({
      ticker,
      attempts: arr.length,
      wins,
      losses,
      netPl,
      avgPl: netPl / arr.length,
      bestPl: Math.max(...pnls),
      worstPl: Math.min(...pnls),
      trades: arr,
    });
  }
  out.sort((a, b) => a.netPl - b.netPl);
  return out;
}


// ────────────────────────────────────────────────────────────────────
// Setup scorecard (per-rule profit factor + auto-verdict)
// ────────────────────────────────────────────────────────────────────

export type SetupVerdict = "core" | "keep" | "watch" | "kill" | "small-n";

export interface SetupScorecardRow {
  setup: string;
  n: number;
  wins: number;
  losses: number;
  winPct: number;
  grossWin: number;
  grossLoss: number;      // absolute (positive)
  netPl: number;
  profitFactor: number;    // Infinity when grossLoss === 0 and grossWin > 0
  expectancy: number;
  avgHoldDays: number | null;
  verdict: SetupVerdict;
  /** Constituent trades — enables the drill-down modal. */
  trades: TradePosition[];
}

/** Per-setup rollup mirroring the PDF's Section 3 "Setup Scorecard".
 *
 *  Verdict thresholds (n ≥ minN required to qualify):
 *    PF ≥ 5  → Core
 *    PF ≥ 2  → Keep
 *    PF ≥ 1  → Watch
 *    PF < 1  → Kill
 *    n < minN → Small n (parked at end of table)
 *
 *  Sort order: qualifying rows first by PF desc, then small-n by n desc.
 *  Rule field is trimmed; empty rules bucketed as "(unlabeled)". */
export function setupScorecard(
  trades: TradePosition[],
  opts: { minN?: number; pnlOf?: (t: TradePosition) => number } = {},
): SetupScorecardRow[] {
  const minN = opts.minN ?? 5;
  const pnlOf = opts.pnlOf ?? realized;
  const holdDays = (t: TradePosition): number | null => {
    const o = String(t.open_date || "").trim();
    const c = String(t.closed_date || "").trim();
    if (!o || !c) return null;
    const od = new Date(o).getTime();
    const cd = new Date(c).getTime();
    if (isNaN(od) || isNaN(cd)) return null;
    return Math.max(0, Math.floor((cd - od) / 86_400_000));
  };
  const byRule = new Map<string, TradePosition[]>();
  for (const t of trades) {
    const r = String(t.rule || "").trim() || "(unlabeled)";
    const list = byRule.get(r);
    if (list) list.push(t);
    else byRule.set(r, [t]);
  }
  const rows: SetupScorecardRow[] = [];
  for (const [setup, arr] of byRule.entries()) {
    const wins = arr.filter(t => pnlOf(t) > 0);
    const losses = arr.filter(t => pnlOf(t) < 0);
    const grossWin = wins.reduce((a, t) => a + pnlOf(t), 0);
    const grossLoss = Math.abs(losses.reduce((a, t) => a + pnlOf(t), 0));
    const netPl = arr.reduce((a, t) => a + pnlOf(t), 0);
    const pf = grossLoss > 0 ? grossWin / grossLoss : (grossWin > 0 ? Infinity : 0);
    const expectancy = arr.length > 0 ? netPl / arr.length : 0;
    const holds = arr.map(holdDays).filter((d): d is number => d !== null);
    const avgHoldDays = holds.length > 0 ? holds.reduce((a, b) => a + b, 0) / holds.length : null;
    const winPct = arr.length > 0 ? (wins.length / arr.length) * 100 : 0;
    let verdict: SetupVerdict;
    if (arr.length < minN) verdict = "small-n";
    else if (pf >= 5) verdict = "core";
    else if (pf >= 2) verdict = "keep";
    else if (pf >= 1) verdict = "watch";
    else verdict = "kill";
    rows.push({
      setup, n: arr.length, wins: wins.length, losses: losses.length,
      winPct, grossWin, grossLoss, netPl, profitFactor: pf,
      expectancy, avgHoldDays, verdict, trades: arr,
    });
  }
  rows.sort((a, b) => {
    const aQual = a.verdict !== "small-n";
    const bQual = b.verdict !== "small-n";
    if (aQual !== bQual) return aQual ? -1 : 1;
    if (aQual && bQual) {
      const aPf = a.profitFactor === Infinity ? 999 : a.profitFactor;
      const bPf = b.profitFactor === Infinity ? 999 : b.profitFactor;
      return bPf - aPf;
    }
    return b.n - a.n;
  });
  return rows;
}


// ────────────────────────────────────────────────────────────────────
// Regime cross-tab (open month × market window)
// ────────────────────────────────────────────────────────────────────

export interface RegimeCell {
  month: string;         // "2026-04"
  window: string;        // journal.market_window on/before open_date
  n: number;
  winRate: number;
  netPl: number;
}

/** Groups closed (or at-mark) trades by (open_month × regime on
 *  open_date). Regime is resolved via the `regimeOf` opt — pass an MCT
 *  state resolver (POWERTREND / UPTREND / CORRECTION / RALLY MODE) for
 *  the M Factor bucket, or omit for the legacy journal.market_window
 *  fallback.
 *
 *  Both resolvers use "state as of open_date" semantics (last-known
 *  value at or before that day) — this is regime lookup, not risk
 *  sizing. */
export function regimeCrossTab(
  trades: TradePosition[],
  journal: JournalHistoryPoint[],
  pnlOf: (t: TradePosition) => number = t => realized(t),
  opts: { regimeOf?: (openDate: string) => string } = {},
): RegimeCell[] {
  const windowOnOrBefore = (day: string): string => {
    const d = String(day).slice(0, 10);
    if (!d) return "";
    const sorted = journal
      .filter(h => String(h.day).slice(0, 10) <= d)
      .sort((a, b) => String(b.day).localeCompare(String(a.day)));
    return sorted.length > 0 ? String((sorted[0] as any).market_window || "") : "";
  };
  const resolve = opts.regimeOf ?? windowOnOrBefore;
  const byKey = new Map<string, { month: string; window: string; pnls: number[] }>();
  for (const t of trades) {
    const od = String(t.open_date || "").slice(0, 10);
    if (!od) continue;
    const month = od.slice(0, 7);
    const win = resolve(od) || "Unknown";
    const key = `${month}|${win}`;
    const cell = byKey.get(key) ?? { month, window: win, pnls: [] };
    cell.pnls.push(pnlOf(t));
    byKey.set(key, cell);
  }
  const cells: RegimeCell[] = [];
  for (const { month, window, pnls } of byKey.values()) {
    const wins = pnls.filter(p => p > 0).length;
    const netPl = pnls.reduce((a, v) => a + v, 0);
    cells.push({
      month,
      window,
      n: pnls.length,
      winRate: pnls.length > 0 ? (wins / pnls.length) * 100 : 0,
      netPl,
    });
  }
  cells.sort((a, b) => a.month.localeCompare(b.month) || a.window.localeCompare(b.window));
  return cells;
}


// ────────────────────────────────────────────────────────────────────
// Insights — auto-generated Action List (PDF Section 10)
// ────────────────────────────────────────────────────────────────────

export type InsightSeverity = "critical" | "warning" | "info";

export interface InsightItem {
  label: string;
  detail?: string;
  netPl?: number;      // rendered as currency when present
  pct?: number;        // rendered as % when present
  count?: number;      // rendered as bare int when present
}

export interface Insight {
  id: string;
  severity: InsightSeverity;
  icon: string;
  title: string;
  /** Prose explanation of the finding + recommendation. */
  detail: string;
  /** Optional per-item rows (setups, tickers, weeks). Limited by caller. */
  items?: InsightItem[];
  /** Optional dollar impact rollup shown next to the title. */
  impactDollars?: number;
}

/** Generate the PDF Section 10 "Action List" automatically from the
 *  cohort. Insights are ordered severity-first (critical → warning →
 *  info), then by impactDollars desc within a severity. Empty when the
 *  cohort has nothing worth flagging.
 *
 *  Rules encoded (all data-derived, no hard-coded picks):
 *   - Kill setups: n ≥ 5, PF < 1
 *   - Gate setups: n ≥ 5, 1 ≤ PF < 2
 *   - Penalty-box tickers: ≥ minAttempts attempts with negative net P&L
 *   - Correction-window bleed: net loss on trades opened during a
 *     CORRECTION MCT state (down-cycle protocol violation)
 *   - Catastrophe stop: losers breaching the configured stop cap
 *   - Overtrading weeks: weeks with > entryBudgetPerWeek fresh entries
 */
export function generateInsights(
  trades: TradePosition[],
  opts: {
    mctStateResolver?: (openDate: string) => string;
    catastropheStopPct?: number;      // default 8
    entryBudgetPerWeek?: number;      // default 5
    correctionStates?: string[];       // default ['CORRECTION']
    penaltyBoxMinAttempts?: number;    // default 3
  } = {},
): Insight[] {
  const catStop = opts.catastropheStopPct ?? 8;
  const entryBudget = opts.entryBudgetPerWeek ?? 5;
  const correctionStates = new Set(opts.correctionStates ?? ["CORRECTION"]);
  const penaltyMin = opts.penaltyBoxMinAttempts ?? 3;
  const mctStateResolver = opts.mctStateResolver;

  const insights: Insight[] = [];
  const scorecard = setupScorecard(trades);

  // 1) Kill setups
  const killSetups = scorecard.filter(r => r.verdict === "kill");
  if (killSetups.length > 0) {
    const totalDrag = killSetups.reduce((a, r) => a + r.netPl, 0);
    insights.push({
      id: "kill-setups",
      severity: "critical",
      icon: "🚫",
      title: `${killSetups.length} setup${killSetups.length === 1 ? "" : "s"} to consider retiring`,
      detail: "Profit factor < 1 across ≥ 5 trades. These setups are net losers on statistically meaningful samples. Retire from the playbook or add a regime gate.",
      impactDollars: totalDrag,
      items: killSetups.map(r => ({
        label: r.setup,
        detail: `n=${r.n} · PF=${r.profitFactor.toFixed(2)} · Win% ${r.winPct.toFixed(0)}`,
        netPl: r.netPl,
      })),
    });
  }

  // 2) Gate setups (PF 1..2)
  const gateSetups = scorecard.filter(r => r.verdict === "watch");
  if (gateSetups.length > 0) {
    insights.push({
      id: "gate-setups",
      severity: "warning",
      icon: "🚧",
      title: `${gateSetups.length} setup${gateSetups.length === 1 ? "" : "s"} on the fence`,
      detail: "PF between 1 and 2 — marginal. Consider gating to POWERTREND/UPTREND only, or tightening entry criteria before running them again.",
      items: gateSetups.map(r => ({
        label: r.setup,
        detail: `n=${r.n} · PF=${r.profitFactor.toFixed(2)} · Win% ${r.winPct.toFixed(0)}`,
        netPl: r.netPl,
      })),
    });
  }

  // 3) Penalty-box tickers
  const offenders = repeatOffenders(trades, { minAttempts: penaltyMin });
  const penaltyTickers = offenders.filter(o => o.netPl < 0);
  if (penaltyTickers.length > 0) {
    const totalDrag = penaltyTickers.reduce((a, o) => a + o.netPl, 0);
    insights.push({
      id: "penalty-box",
      severity: "warning",
      icon: "📛",
      title: `${penaltyTickers.length} ticker${penaltyTickers.length === 1 ? "" : "s"} in the penalty box`,
      detail: `≥ ${penaltyMin} attempts with negative net P&L. A name that stops you out repeatedly is telling you something about the name — or the read of it. Consider a "no re-entry this quarter" rule.`,
      impactDollars: totalDrag,
      items: penaltyTickers.slice(0, 10).map(o => ({
        label: o.ticker,
        detail: `${o.attempts} attempts · ${o.wins}W / ${o.losses}L`,
        netPl: o.netPl,
      })),
    });
  }

  // 4) Correction-window bleed
  if (mctStateResolver) {
    const correctionTrades = trades.filter(t => {
      const state = mctStateResolver(String(t.open_date || ""));
      return correctionStates.has(state);
    });
    if (correctionTrades.length > 0) {
      const correctionNet = correctionTrades.reduce((a, t) => a + realized(t), 0);
      if (correctionNet < 0) {
        const wins = correctionTrades.filter(t => realized(t) > 0).length;
        const winPct = (wins / correctionTrades.length) * 100;
        insights.push({
          id: "correction-bleed",
          severity: "critical",
          icon: "🌩️",
          title: `${correctionTrades.length} entries opened during CORRECTION`,
          detail: `Down-Cycle Protocol says no fresh entries on a negative Trend Count. Win rate on these attempts: ${winPct.toFixed(0)}%. The rule isn't missing — the compliance is.`,
          impactDollars: correctionNet,
        });
      }
    }
  }

  // 5) Catastrophe stop
  const bigLosers: Array<{ t: TradePosition; savings: number; rp: number }> = [];
  for (const t of trades) {
    const rp = Number((t as any).return_pct ?? NaN);
    const cost = Number(t.total_cost ?? 0);
    if (!Number.isFinite(rp) || cost <= 0) continue;
    if (realized(t) < 0 && rp < -catStop) {
      const savings = ((Math.abs(rp) - catStop) / 100) * cost;
      bigLosers.push({ t, savings, rp });
    }
  }
  if (bigLosers.length > 0) {
    const totalSavings = bigLosers.reduce((a, x) => a + x.savings, 0);
    insights.push({
      id: "catastrophe-stop",
      severity: "info",
      icon: "🛡️",
      title: `${bigLosers.length} loss${bigLosers.length === 1 ? "" : "es"} breached −${catStop}%`,
      detail: `A hard −${catStop}% catastrophe stop would have saved ~${totalSavings >= 0 ? "" : "-"}$${Math.abs(totalSavings).toFixed(0)} (upper bound — assumes no winner clipped). Cheap insurance with near-zero risk of clipping a future monster.`,
      impactDollars: totalSavings,
      items: bigLosers.slice(0, 5).map(({ t, rp }) => ({
        label: `${t.ticker} (${t.trade_id})`,
        detail: `Return ${rp.toFixed(1)}%`,
        netPl: realized(t),
      })),
    });
  }

  // 6) Overtrading weeks
  const byWeek = new Map<string, number>();
  for (const t of trades) {
    const od = String(t.open_date || "").slice(0, 10);
    if (!od) continue;
    const d = new Date(od);
    if (isNaN(d.getTime())) continue;
    const year = d.getUTCFullYear();
    const jan1 = new Date(Date.UTC(year, 0, 1));
    const dayOfYear = Math.floor((d.getTime() - jan1.getTime()) / 86_400_000);
    const week = Math.floor(dayOfYear / 7) + 1;
    const key = `${year}-W${String(week).padStart(2, "0")}`;
    byWeek.set(key, (byWeek.get(key) ?? 0) + 1);
  }
  const overWeeks = Array.from(byWeek.entries())
    .filter(([, n]) => n > entryBudget)
    .sort((a, b) => b[1] - a[1]);
  if (overWeeks.length > 0) {
    insights.push({
      id: "overtrading",
      severity: "info",
      icon: "🔥",
      title: `${overWeeks.length} week${overWeeks.length === 1 ? "" : "s"} exceeded the ${entryBudget}-entry budget`,
      detail: "The year is made in a handful of concentrated trades. Weeks with > entry-budget fresh entries are the churn tax — they cluster in bad tape and drag the year.",
      items: overWeeks.slice(0, 10).map(([k, n]) => ({
        label: k,
        count: n,
      })),
    });
  }

  return insights;
}


/** Build a "state as of date" resolver from the sparse mctStateByDateRange
 *  response. Returns a function that, given an ISO date, returns the state
 *  in effect on or before that date. Case-normalized to Title Case for
 *  display in the Regime × Month table. */
export function makeMctStateResolver(
  states: Array<{ trade_date: string; state: string }>,
): (openDate: string) => string {
  // Sort DESC once so lookups can early-return on the first hit.
  const sorted = [...states].sort((a, b) =>
    String(b.trade_date).localeCompare(String(a.trade_date))
  );
  return (openDate: string): string => {
    const d = String(openDate).slice(0, 10);
    if (!d) return "";
    for (const s of sorted) {
      if (String(s.trade_date).slice(0, 10) <= d) return String(s.state || "");
    }
    return "";
  };
}
