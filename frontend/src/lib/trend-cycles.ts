// Trend Cycle math — pure computation over a journal-history array.
// Groups consecutive rows by sign(trend_count) into "legs" (one per
// contiguous positive-sign or negative-sign block) and computes the
// per-leg metrics + cross-leg aggregates the Trend Cycle Review page
// renders. No API calls, no side effects — all derivation.
//
// Semantic anchors (mirror the engine's Phase 11 rule):
//   sign > 0  → in a positive 21e leg (Step-4 up-day armed)
//   sign < 0  → in a negative 21e leg (VIOLATION_21EMA fired)
//   sign = 0 or trend_count null  → gap (pre-first-arm, or holiday NULL)
//     These rows are NOT included in any leg. If they fall BETWEEN two
//     same-sign legs, the legs stay separate (a null row breaks a leg).

import type { JournalHistoryPoint } from "@/lib/api";

export interface TrendCycleLeg {
  /** Chronological number of this leg, starting at 1 for the OLDEST
   *  leg in the journal history. Stable across filters and sorts. */
  cycle_number: number;
  sign: 1 | -1;
  /** ISO date of the first row in this leg (inclusive). */
  start_date: string;
  /** ISO date of the last row in this leg (inclusive). */
  end_date: string;
  /** Signed count on the first day of this leg (e.g. +1 on an arm bar). */
  count_at_start: number;
  /** Signed count on the last day of this leg. */
  count_at_end: number;
  /** Trading days spanned. Always ≥ 1. */
  duration_days: number;

  /** End NLV of the day BEFORE the leg's first day. Null for the first
   *  leg in history if no prior row exists (return_pct then falls back
   *  to using end_nlv of the first day as the anchor). */
  start_nlv: number | null;
  end_nlv: number;
  return_dollars: number;
  return_pct: number | null;

  /** NASDAQ (^IXIC) close on the day before the leg's first day, and
   *  the last day of the leg. Kept for reference / potential future use. */
  ndx_start: number | null;
  ndx_end: number;
  ndx_return_pct: number | null;

  /** SPY close on the day before the leg's first day, and the last day
   *  of the leg. SPY is the alpha benchmark (broader-market comparison
   *  than NDX, which is tech-heavy and would flatter a CANSLIM
   *  portfolio's alpha in tech-up tapes). */
  spy_start: number | null;
  spy_end: number;
  spy_return_pct: number | null;
  /** portfolio return − SPY return, in percentage-point delta. */
  alpha_pct: number | null;

  /** Intra-leg max drawdown: biggest peak-to-trough dip of end_nlv
   *  WITHIN the leg, expressed as negative percent. 0 if the leg was
   *  monotonically rising. */
  max_drawdown_pct: number;
  /** Same as above but in dollars. */
  max_drawdown_dollars: number;

  /** Behavior averages over the leg's days. pct_invested values in the
   *  DB are stored as percentages already (e.g. 106.5 = 106.5% invested
   *  when using margin), so avg_pct_invested is the raw average without
   *  a fraction→percentage conversion. */
  avg_pct_invested: number;
  avg_portfolio_heat: number;
}

export interface TrendCycleAggregates {
  total_legs: number;
  positive_legs: number;
  negative_legs: number;

  /** Win rate ACROSS ALL LEGS: legs with return > 0 / total legs.
   *  Legs with null return are excluded from both numerator + denom. */
  win_rate: number | null;

  /** Average return % on positive-sign legs. Null if no positive legs. */
  avg_positive_return_pct: number | null;
  /** Average return % on negative-sign legs (typically negative or ~0
   *  if the user cuts exposure well). Null if no negative legs. */
  avg_negative_return_pct: number | null;

  /** avg_win × win_rate − avg_loss × loss_rate over ALL legs (winners +
   *  losers, sign-agnostic). $/leg. Null if no legs closed with a
   *  determinable return. */
  expectancy_pct: number | null;
  expectancy_dollars: number | null;

  /** Cumulative alpha vs SPY: sum of alpha_pct across all legs where
   *  alpha could be computed. Rough proxy for the total edge from
   *  leg-based sizing behavior vs. broad-market buy-and-hold.
   *  Null if no legs. */
  cumulative_alpha_pct: number | null;

  /** Avg % invested — split by leg sign so the "did I actually cut
   *  exposure in negative legs" question is visible at a glance.
   *  Values are raw percentages (e.g. 85 = 85% invested), not fractions. */
  avg_pct_invested_positive: number | null;
  avg_pct_invested_negative: number | null;

  /** Longest positive / negative leg in trading days. */
  longest_positive_days: number;
  longest_negative_days: number;

  /** Cycle Anatomy — historical baselines for "is this leg abnormal?"
   *  Split by sign, all durations in trading days, all returns/DDs
   *  in percent. Nulls for a sign when no legs of that sign exist. */
  avg_positive_duration: number | null;
  median_positive_duration: number | null;
  shortest_positive_duration: number | null;
  avg_positive_dd_pct: number | null;

  avg_negative_duration: number | null;
  median_negative_duration: number | null;
  shortest_negative_duration: number | null;
  avg_negative_dd_pct: number | null;
}

/** Group + compute. Rows should be journal-history-shape — the JournalHistoryPoint
 *  interface is a superset; we only touch fields we need. */
export function computeTrendCycles(rows: JournalHistoryPoint[]): {
  legs: TrendCycleLeg[];
  aggregates: TrendCycleAggregates;
} {
  // Sort ascending by day so leg detection walks in chronological order.
  const sorted = [...rows].sort((a, b) => String(a.day).localeCompare(String(b.day)));

  const legs: TrendCycleLeg[] = [];
  let cursor = 0;

  // Coerce trend_count defensively — the API can hand back numbers,
  // stringified numbers, or NaN-ish placeholders depending on how the
  // Pandas DataFrame serialized. Normalize once so the loop below sees
  // real numbers.
  const asTc = (row: JournalHistoryPoint): number | null => {
    const raw = (row as any).trend_count;
    if (raw == null) return null;
    const n = Number(raw);
    if (!isFinite(n)) return null;
    return n;
  };

  while (cursor < sorted.length) {
    const tc = asTc(sorted[cursor]);
    if (tc == null || tc === 0) {
      // Not part of any leg; advance.
      cursor += 1;
      continue;
    }
    const sign: 1 | -1 = tc > 0 ? 1 : -1;

    // Consume as many consecutive same-sign rows as possible.
    const legRows: JournalHistoryPoint[] = [];
    let j = cursor;
    while (j < sorted.length) {
      const next_tc = asTc(sorted[j]);
      if (next_tc == null || next_tc === 0) break;
      const next_sign = next_tc > 0 ? 1 : -1;
      if (next_sign !== sign) break;
      legRows.push(sorted[j]);
      j += 1;
    }

    // Anchor row = day BEFORE the leg starts. Used for return anchors.
    // Falls back to null if this is the very first row in history.
    const anchor = cursor > 0 ? sorted[cursor - 1] : null;
    // Chronological numbering — first leg detected is #1 (oldest,
    // since we walked sorted-ascending). Stable across filters/sorts.
    legs.push(buildLeg(sign, legRows, anchor, legs.length + 1));
    cursor = j;
  }

  return { legs, aggregates: buildAggregates(legs) };
}

function buildLeg(
  sign: 1 | -1,
  legRows: JournalHistoryPoint[],
  anchor: JournalHistoryPoint | null,
  cycle_number: number,
): TrendCycleLeg {
  const first = legRows[0];
  const last = legRows[legRows.length - 1];

  // Anchor NLV = end_nlv of the day BEFORE the leg started. Falls
  // back to beg_nlv of the first leg day if no anchor (first leg in
  // history). Used only for the P&L $ dollar-difference formula —
  // return_pct is TWR-chained separately below.
  const start_nlv = anchor
    ? Number((anchor as any).end_nlv) || null
    : Number((first as any).beg_nlv) || null;
  const end_nlv = Number(last.end_nlv) || 0;

  // Sum cash flows across the leg so the P&L $ and P&L % ignore
  // contributions/withdrawals (a $10k deposit shouldn't count as leg
  // "profit"). daily_return values in the API payload are already
  // computed by the backend using the TWR formula
  //   daily_return = (end_nlv - beg_nlv - cash_change) / (beg_nlv + cash_change)
  // so chaining ∏(1 + daily_return) - 1 gives the clean TWR for the leg.
  let twr_multiplier = 1;
  let sum_cash = 0;
  let has_return_data = false;
  for (const r of legRows) {
    const dr = Number((r as any).daily_return);
    if (isFinite(dr)) {
      twr_multiplier *= (1 + dr);
      has_return_data = true;
    }
    const cf = Number((r as any).cash_change);
    if (isFinite(cf)) sum_cash += cf;
  }
  const return_pct = has_return_data ? (twr_multiplier - 1) * 100 : null;
  const return_dollars = start_nlv != null && start_nlv > 0
    ? end_nlv - start_nlv - sum_cash
    : 0;

  const ndx_start = anchor && (anchor as any).nasdaq
    ? Number((anchor as any).nasdaq) || null
    : null;
  const ndx_end = Number((last as any).nasdaq) || 0;
  const ndx_return_pct = ndx_start != null && ndx_start !== 0
    ? (ndx_end - ndx_start) / ndx_start * 100
    : null;

  const spy_start = anchor && (anchor as any).spy
    ? Number((anchor as any).spy) || null
    : null;
  const spy_end = Number((last as any).spy) || 0;
  const spy_return_pct = spy_start != null && spy_start !== 0
    ? (spy_end - spy_start) / spy_start * 100
    : null;
  // Alpha is measured against SPY (broad-market benchmark), not NDX
  // (tech-heavy — would flatter tech-tilted portfolios).
  const alpha_pct = return_pct != null && spy_return_pct != null
    ? return_pct - spy_return_pct
    : null;

  // Intra-leg drawdown, computed on a CASH-FLOW-ADJUSTED synthetic
  // equity curve. Same TWR trick as return_pct above: each bar's
  // synth_nlv = prior_synth × (1 + daily_return), which isolates
  // trading gains/losses from contributions and withdrawals. A
  // deposit mid-leg no longer resets the peak upward, a withdrawal
  // no longer registers as a fake drawdown. Starts from the leg's
  // anchor NLV (day before it began) — falls back to beg_nlv of the
  // first day if no anchor exists (first leg in history).
  let synth_nlv = start_nlv ?? Number((first as any).beg_nlv) ?? 0;
  let peak_synth = synth_nlv;
  let maxDdPct = 0;
  let maxDdDollars = 0;
  for (const row of legRows) {
    const dr = Number((row as any).daily_return);
    if (isFinite(dr)) {
      synth_nlv = synth_nlv * (1 + dr);
    }
    if (synth_nlv > peak_synth) peak_synth = synth_nlv;
    if (peak_synth > 0) {
      const ddPct = (synth_nlv - peak_synth) / peak_synth * 100;
      if (ddPct < maxDdPct) maxDdPct = ddPct;
      const ddDollars = synth_nlv - peak_synth;
      if (ddDollars < maxDdDollars) maxDdDollars = ddDollars;
    }
  }

  // Behavior averages. pct_invested values from the DB are already
  // percentages (e.g. 106.5 = 106.5% invested using margin), so no
  // fraction→percentage scaling here.
  const invSum = legRows.reduce((s, r) => s + (Number(r.pct_invested) || 0), 0);
  const heatSum = legRows.reduce((s, r) => s + (Number(r.portfolio_heat) || 0), 0);

  return {
    cycle_number,
    sign,
    start_date: String(first.day).slice(0, 10),
    end_date: String(last.day).slice(0, 10),
    count_at_start: Number((first as any).trend_count) || 0,
    count_at_end: Number((last as any).trend_count) || 0,
    duration_days: legRows.length,
    start_nlv,
    end_nlv,
    return_dollars,
    return_pct,
    ndx_start,
    ndx_end,
    ndx_return_pct,
    spy_start,
    spy_end,
    spy_return_pct,
    alpha_pct,
    max_drawdown_pct: maxDdPct,
    max_drawdown_dollars: maxDdDollars,
    avg_pct_invested: legRows.length > 0 ? invSum / legRows.length : 0,
    avg_portfolio_heat: legRows.length > 0 ? heatSum / legRows.length : 0,
  };
}

function buildAggregates(legs: TrendCycleLeg[]): TrendCycleAggregates {
  const positive = legs.filter(l => l.sign === 1);
  const negative = legs.filter(l => l.sign === -1);

  const legsWithReturn = legs.filter(l => l.return_pct != null);
  const winners = legsWithReturn.filter(l => (l.return_pct as number) > 0);
  const losers = legsWithReturn.filter(l => (l.return_pct as number) < 0);
  // Scratches (return_pct == 0) are counted in total but excluded from
  // avg_win / avg_loss like Campaign Review does.

  const avg = (arr: number[]) => arr.length > 0
    ? arr.reduce((s, v) => s + v, 0) / arr.length
    : null;

  const avgPosReturn = positive.length > 0
    ? avg(positive.map(l => l.return_pct ?? 0))
    : null;
  const avgNegReturn = negative.length > 0
    ? avg(negative.map(l => l.return_pct ?? 0))
    : null;

  // Expectancy across ALL legs (positive + negative) — the system's
  // per-leg edge, sign-agnostic. Same formula as Campaign Review.
  const decidedTrades = winners.length + losers.length;
  const winRate = decidedTrades > 0 ? winners.length / decidedTrades : null;
  const avgWinPct = winners.length > 0
    ? winners.reduce((s, l) => s + (l.return_pct as number), 0) / winners.length
    : 0;
  const avgLossPct = losers.length > 0
    ? Math.abs(losers.reduce((s, l) => s + (l.return_pct as number), 0)) / losers.length
    : 0;
  const expectancy_pct = winRate != null
    ? avgWinPct * winRate - avgLossPct * (1 - winRate)
    : null;

  const avgWinDollars = winners.length > 0
    ? winners.reduce((s, l) => s + l.return_dollars, 0) / winners.length
    : 0;
  const avgLossDollars = losers.length > 0
    ? Math.abs(losers.reduce((s, l) => s + l.return_dollars, 0)) / losers.length
    : 0;
  const expectancy_dollars = winRate != null
    ? avgWinDollars * winRate - avgLossDollars * (1 - winRate)
    : null;

  const legsWithAlpha = legs.filter(l => l.alpha_pct != null);
  const cumulative_alpha_pct = legsWithAlpha.length > 0
    ? legsWithAlpha.reduce((s, l) => s + (l.alpha_pct as number), 0)
    : null;

  const median = (arr: number[]): number | null => {
    if (arr.length === 0) return null;
    const s = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(s.length / 2);
    return s.length % 2 === 0 ? (s[mid - 1] + s[mid]) / 2 : s[mid];
  };
  const min = (arr: number[]): number | null =>
    arr.length === 0 ? null : Math.min(...arr);

  const posDurations = positive.map(l => l.duration_days);
  const negDurations = negative.map(l => l.duration_days);

  return {
    total_legs: legs.length,
    positive_legs: positive.length,
    negative_legs: negative.length,
    win_rate: winRate,
    avg_positive_return_pct: avgPosReturn,
    avg_negative_return_pct: avgNegReturn,
    expectancy_pct,
    expectancy_dollars,
    cumulative_alpha_pct,
    avg_pct_invested_positive: avg(positive.map(l => l.avg_pct_invested)),
    avg_pct_invested_negative: avg(negative.map(l => l.avg_pct_invested)),
    longest_positive_days: positive.reduce((m, l) => Math.max(m, l.duration_days), 0),
    longest_negative_days: negative.reduce((m, l) => Math.max(m, l.duration_days), 0),

    avg_positive_duration: avg(posDurations),
    median_positive_duration: median(posDurations),
    shortest_positive_duration: min(posDurations),
    avg_positive_dd_pct: avg(positive.map(l => l.max_drawdown_pct)),

    avg_negative_duration: avg(negDurations),
    median_negative_duration: median(negDurations),
    shortest_negative_duration: min(negDurations),
    avg_negative_dd_pct: avg(negative.map(l => l.max_drawdown_pct)),
  };
}
