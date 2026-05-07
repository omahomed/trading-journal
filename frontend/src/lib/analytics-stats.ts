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

export type Last10Outcome = "win" | "loss" | "be";

export interface Last10Trade {
  trade_id: string;
  ticker: string;
  status: string;
  open_date: string;
  pl: number;
  outcome: Last10Outcome;
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
  trades: { trade_id: string; ticker: string; status: string; open_date: string; pl: number }[],
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
