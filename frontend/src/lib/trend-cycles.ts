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
   *  the last day of the leg. Used to compute the benchmark comparison. */
  ndx_start: number | null;
  ndx_end: number;
  ndx_return_pct: number | null;
  /** portfolio return − NDX return, in percentage-point delta. */
  alpha_pct: number | null;

  /** Intra-leg max drawdown: biggest peak-to-trough dip of end_nlv
   *  WITHIN the leg, expressed as negative percent. 0 if the leg was
   *  monotonically rising. */
  max_drawdown_pct: number;
  /** Same as above but in dollars. */
  max_drawdown_dollars: number;

  /** Behavior averages over the leg's days. */
  avg_pct_invested: number;
  avg_portfolio_heat: number;
  avg_score: number | null;
  a_grade_days: number;   // score >= 4
  d_grade_days: number;   // score <= 2 (D or F)
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

  /** Cumulative alpha: sum of alpha_pct across all legs where alpha
   *  could be computed. Rough proxy for the total edge from leg-based
   *  sizing behavior. Null if no legs. */
  cumulative_alpha_pct: number | null;

  /** Avg % invested — split by leg sign so the "did I actually cut
   *  exposure in negative legs" question is visible at a glance. */
  avg_pct_invested_positive: number | null;
  avg_pct_invested_negative: number | null;

  /** Longest positive / negative leg in trading days. */
  longest_positive_days: number;
  longest_negative_days: number;
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

  while (cursor < sorted.length) {
    const r = sorted[cursor];
    const tc = (r as any).trend_count;
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
      const next_tc = (sorted[j] as any).trend_count;
      if (next_tc == null || next_tc === 0) break;
      const next_sign = next_tc > 0 ? 1 : -1;
      if (next_sign !== sign) break;
      legRows.push(sorted[j]);
      j += 1;
    }

    // Anchor row = day BEFORE the leg starts. Used for return anchors.
    // Falls back to null if this is the very first row in history.
    const anchor = cursor > 0 ? sorted[cursor - 1] : null;
    legs.push(buildLeg(sign, legRows, anchor));
    cursor = j;
  }

  return { legs, aggregates: buildAggregates(legs) };
}

function buildLeg(
  sign: 1 | -1,
  legRows: JournalHistoryPoint[],
  anchor: JournalHistoryPoint | null,
): TrendCycleLeg {
  const first = legRows[0];
  const last = legRows[legRows.length - 1];

  const start_nlv = anchor ? Number(anchor.end_nlv) || null : null;
  const end_nlv = Number(last.end_nlv) || 0;
  const return_dollars = start_nlv != null ? end_nlv - start_nlv : 0;
  const return_pct = start_nlv != null && start_nlv !== 0
    ? (end_nlv - start_nlv) / start_nlv * 100
    : null;

  const ndx_start = anchor && (anchor as any).nasdaq
    ? Number((anchor as any).nasdaq) || null
    : null;
  const ndx_end = Number((last as any).nasdaq) || 0;
  const ndx_return_pct = ndx_start != null && ndx_start !== 0
    ? (ndx_end - ndx_start) / ndx_start * 100
    : null;
  const alpha_pct = return_pct != null && ndx_return_pct != null
    ? return_pct - ndx_return_pct
    : null;

  // Intra-leg drawdown: rolling peak, biggest dip from any peak.
  let peak = start_nlv ?? Number(first.end_nlv) ?? 0;
  let maxDdPct = 0;
  let maxDdDollars = 0;
  for (const row of legRows) {
    const v = Number(row.end_nlv) || 0;
    if (v > peak) peak = v;
    if (peak > 0) {
      const dd = (v - peak) / peak * 100;
      if (dd < maxDdPct) maxDdPct = dd;
      const ddDollars = v - peak;
      if (ddDollars < maxDdDollars) maxDdDollars = ddDollars;
    }
  }

  // Behavior averages
  const invSum = legRows.reduce((s, r) => s + (Number(r.pct_invested) || 0), 0);
  const heatSum = legRows.reduce((s, r) => s + (Number(r.portfolio_heat) || 0), 0);
  const scoreRows = legRows.filter(r => (r as any).score != null && (r as any).score > 0);
  const scoreSum = scoreRows.reduce((s, r) => s + (Number((r as any).score) || 0), 0);
  const a_grade = legRows.filter(r => Number((r as any).score || 0) >= 4).length;
  const d_grade = legRows.filter(r => {
    const s = Number((r as any).score || 0);
    return s > 0 && s <= 2;
  }).length;

  return {
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
    alpha_pct,
    max_drawdown_pct: maxDdPct,
    max_drawdown_dollars: maxDdDollars,
    avg_pct_invested: legRows.length > 0 ? invSum / legRows.length : 0,
    avg_portfolio_heat: legRows.length > 0 ? heatSum / legRows.length : 0,
    avg_score: scoreRows.length > 0 ? scoreSum / scoreRows.length : null,
    a_grade_days: a_grade,
    d_grade_days: d_grade,
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
  };
}
