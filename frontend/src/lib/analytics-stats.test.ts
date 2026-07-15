import { describe, it, expect, test } from "vitest";
import {
  computeWinRate,
  computeProfitFactor,
  computeHoldRatio,
  computeOnePctCompliance,
  computeLast10Stats,
  trailingClosedTrades,
  trailingClosedLosses,
  getPriorDayNlv,
  tradeWasOpenInYear,
  availableTradeYears,
  paretoDistribution,
  holdTimeBuckets,
  brandtNormalized,
  stopCapScenario,
  fixedSizeScenario,
  regimeCrossTab,
  setupScorecard,
  riskMetrics,
  repeatOffenders,
  makeMctStateResolver,
  generateInsights,
  winnerMaeDistribution,
  loserMfeDistribution,
  entryQualityBySetup,
} from "./analytics-stats";
import type { TradePosition, JournalHistoryPoint } from "./api";

const closed = (overrides: Partial<any>) => ({
  trade_id: "T",
  ticker: "X",
  status: "CLOSED",
  open_date: "2026-01-01",
  closed_date: "2026-01-15",
  shares: 0,
  avg_entry: 100,
  total_cost: 0,
  realized_pl: 0,
  rule: "",
  ...overrides,
}) as TradePosition;

describe("computeWinRate", () => {
  it("returns 0 for empty input", () => {
    expect(computeWinRate([])).toBe(0);
  });
  it("calculates percentage of trades with positive realized_pl", () => {
    const trades = [
      closed({ realized_pl: 1000 }),
      closed({ realized_pl: 500 }),
      closed({ realized_pl: -300 }),
      closed({ realized_pl: 0 }), // breakeven excluded from wins
    ];
    expect(computeWinRate(trades)).toBe(50);
  });
});

describe("computeProfitFactor", () => {
  it("returns 0 when there are no losing trades and no winners", () => {
    expect(computeProfitFactor([])).toBe(0);
  });
  it("returns 0 when grossLoss is 0 (matches analytics.tsx convention)", () => {
    const trades = [closed({ realized_pl: 1000 }), closed({ realized_pl: 500 })];
    expect(computeProfitFactor(trades)).toBe(0);
  });
  it("calculates grossProfit / grossLoss correctly", () => {
    const trades = [
      closed({ realized_pl: 3000 }),
      closed({ realized_pl: 1000 }),
      closed({ realized_pl: -2000 }),
    ];
    expect(computeProfitFactor(trades)).toBe(2);
  });
});

describe("computeHoldRatio", () => {
  it("returns ratio=0 when there are no losers", () => {
    const r = computeHoldRatio([closed({ realized_pl: 1000, open_date: "2026-01-01", closed_date: "2026-01-11" })]);
    expect(r.ratio).toBe(0);
    expect(r.winnersHold).toBe(10);
    expect(r.losersHold).toBe(0);
  });
  it("computes winnersHold / losersHold correctly", () => {
    const r = computeHoldRatio([
      // Winner: 20 days
      closed({ realized_pl: 1000, open_date: "2026-01-01", closed_date: "2026-01-21" }),
      // Loser: 5 days
      closed({ realized_pl: -500, open_date: "2026-02-01", closed_date: "2026-02-06" }),
    ]);
    expect(r.winnersHold).toBe(20);
    expect(r.losersHold).toBe(5);
    expect(r.ratio).toBe(4);
  });
  it("ignores trades with missing or invalid dates", () => {
    const r = computeHoldRatio([
      closed({ realized_pl: 1000, open_date: "", closed_date: "2026-01-15" }),
      closed({ realized_pl: -500, open_date: "2026-02-01", closed_date: "" }),
    ]);
    expect(r.winnersHold).toBe(0);
    expect(r.losersHold).toBe(0);
    expect(r.ratio).toBe(0);
  });
});

describe("computeOnePctCompliance", () => {
  const journal: JournalHistoryPoint[] = [
    { day: "2026-01-01", end_nlv: 500_000 } as any,
    { day: "2026-02-01", end_nlv: 500_000 } as any,
  ];

  it("returns passRate=100 with 0 losses (caller branches on totalLosses for empty state)", () => {
    const r = computeOnePctCompliance([], journal);
    expect(r.totalLosses).toBe(0);
    expect(r.passRate).toBe(100);
    expect(r.breaches).toBe(0);
    expect(r.withinRule).toBe(0);
  });

  it("classifies a breach correctly (impact more negative than -1%)", () => {
    // -1.2% impact: -6000 / 500000 * 100 = -1.2 → BREACH
    const losses = [closed({ realized_pl: -6000, open_date: "2026-01-15" })];
    const r = computeOnePctCompliance(losses, journal);
    expect(r.totalLosses).toBe(1);
    expect(r.withinRule).toBe(0);
    expect(r.breaches).toBe(1);
    expect(r.passRate).toBe(0);
  });

  it("classifies a within-rule loss correctly (impact ≥ -1%)", () => {
    // -0.5% impact: -2500 / 500000 * 100 = -0.5 → WITHIN
    const losses = [closed({ realized_pl: -2500, open_date: "2026-01-15" })];
    const r = computeOnePctCompliance(losses, journal);
    expect(r.totalLosses).toBe(1);
    expect(r.withinRule).toBe(1);
    expect(r.breaches).toBe(0);
    expect(r.passRate).toBe(100);
  });

  it("excludes trades with no NLV available at open date", () => {
    // Open before earliest journal entry → no NLV → excluded
    const losses = [closed({ realized_pl: -2500, open_date: "2025-12-15" })];
    const r = computeOnePctCompliance(losses, journal);
    expect(r.totalLosses).toBe(0);
  });
});

describe("computeLast10Stats", () => {
  const trade = (i: number, pl: number, status = "CLOSED") => ({
    trade_id: `T${i}`,
    ticker: `TKR${i}`,
    status,
    open_date: `2026-04-${String(i).padStart(2, "0")}`,
    pl,
  });

  it("returns empty stats for 0 trades", () => {
    const r = computeLast10Stats([], 0);
    expect(r.count).toBe(0);
    expect(r.winRate).toBe(0);
    expect(r.netPl).toBe(0);
    expect(r.profitFactor).toBe(0);
    expect(r.trades).toEqual([]);
  });

  it("windows to most recent 10 by open_date", () => {
    // 15 trades, all open in April 2026
    const trades = Array.from({ length: 15 }, (_, i) => trade(i + 1, 1000));
    const r = computeLast10Stats(trades, 60);
    expect(r.count).toBe(10);
    // Most recent 10 are days 06..15; ordered oldest → newest in result
    expect(r.trades[0].open_date).toBe("2026-04-06");
    expect(r.trades[9].open_date).toBe("2026-04-15");
  });

  it("does not pad when fewer than 10 trades exist", () => {
    const trades = [trade(1, 1000), trade(2, -500), trade(3, 250)];
    const r = computeLast10Stats(trades, 60);
    expect(r.count).toBe(3);
    expect(r.trades.length).toBe(3);
  });

  it("classifies outcomes win/loss/be with the dead-zone", () => {
    const trades = [
      trade(1, 1000),   // win
      trade(2, -500),   // loss
      trade(3, 30),     // be (|pl| < 50 default deadzone)
      trade(4, -10),    // be
    ];
    const r = computeLast10Stats(trades, 60);
    const outcomes = r.trades.map(t => t.outcome);
    expect(outcomes).toContain("win");
    expect(outcomes).toContain("loss");
    expect(outcomes.filter(o => o === "be").length).toBe(2);
  });

  it("respects custom dead-zone", () => {
    const trades = [trade(1, 200)];
    const r = computeLast10Stats(trades, 60, 10, 500); // beDeadzone=500
    // |200| < 500 → breakeven
    expect(r.trades[0].outcome).toBe("be");
  });

  it("computes net P&L, avgWin, avgLoss, profitFactor over the windowed set", () => {
    const trades = [
      trade(1, 4000),
      trade(2, -1000),
      trade(3, 2000),
      trade(4, -500),
    ];
    const r = computeLast10Stats(trades, 50);
    // wins: 4000 + 2000 = 6000; losses: -1500
    expect(r.netPl).toBe(4500);
    expect(r.avgWin).toBe(3000);
    expect(r.avgLoss).toBe(-750);
    expect(r.profitFactor).toBe(4); // 6000 / 1500
    expect(r.winRate).toBe(50);
  });

  it("passes ltdWinRate through unchanged for the comparison subtitle", () => {
    const r = computeLast10Stats([trade(1, 1000)], 73.4);
    expect(r.ltdWinRate).toBe(73.4);
  });

  it("supports mixed open + closed trades — caller pre-classifies pl per status", () => {
    // T1 OPEN @ 2026-04-01 with overall_pl=+500; T2 CLOSED @ 2026-04-02 with realized=-200
    // Sort desc → [T2, T1]; take top 2; reverse → ordered oldest→newest = [T1, T2]
    const trades = [trade(1, 500, "OPEN"), trade(2, -200, "CLOSED")];
    const r = computeLast10Stats(trades, 50);
    expect(r.count).toBe(2);
    expect(r.netPl).toBe(300);
    // Index 0 is the older trade (T1) → OPEN, win
    expect(r.trades[0].status).toBe("OPEN");
    expect(r.trades[0].outcome).toBe("win");
    expect(r.trades[1].status).toBe("CLOSED");
    expect(r.trades[1].outcome).toBe("loss");
  });

  it("threads the rule field through to the windowed output", () => {
    const trades = [
      { trade_id: "T1", ticker: "AAA", status: "CLOSED", open_date: "2026-04-01", pl: 1000, rule: "br1.1 Consolidation" },
      { trade_id: "T2", ticker: "BBB", status: "OPEN",   open_date: "2026-04-02", pl: 500,  rule: "RS leader" },
    ];
    const r = computeLast10Stats(trades, 50);
    // Oldest → newest: [T1, T2]
    expect(r.trades[0].rule).toBe("br1.1 Consolidation");
    expect(r.trades[1].rule).toBe("RS leader");
  });
});

describe("trailingClosedTrades", () => {
  it("returns [] for empty input", () => {
    expect(trailingClosedTrades([], 30)).toEqual([]);
  });

  it("excludes trades without closed_date (open trades)", () => {
    const trades = [
      closed({ trade_id: "T1", closed_date: "2026-04-01", realized_pl: 100 }),
      closed({ trade_id: "T2", closed_date: "", realized_pl: 100 }),
      closed({ trade_id: "T3", closed_date: null, realized_pl: 100 }),
    ];
    const r = trailingClosedTrades(trades, 10);
    expect(r.length).toBe(1);
    expect(r[0].trade_id).toBe("T1");
  });

  it("sorts by closed_date descending and slices to N", () => {
    const trades = [
      closed({ trade_id: "T1", closed_date: "2026-01-15" }),
      closed({ trade_id: "T2", closed_date: "2026-04-01" }),
      closed({ trade_id: "T3", closed_date: "2026-02-20" }),
      closed({ trade_id: "T4", closed_date: "2026-03-10" }),
    ];
    const r = trailingClosedTrades(trades, 2);
    expect(r.length).toBe(2);
    expect(r[0].trade_id).toBe("T2"); // most recent
    expect(r[1].trade_id).toBe("T4"); // second most recent
  });

  it("returns all available when N exceeds count", () => {
    const trades = [
      closed({ trade_id: "T1", closed_date: "2026-01-15" }),
      closed({ trade_id: "T2", closed_date: "2026-02-15" }),
    ];
    expect(trailingClosedTrades(trades, 30).length).toBe(2);
  });
});

describe("trailingClosedLosses", () => {
  it("filters to losses then windows", () => {
    const trades = [
      closed({ trade_id: "W1", closed_date: "2026-04-01", realized_pl: 1000 }),
      closed({ trade_id: "L1", closed_date: "2026-03-01", realized_pl: -500 }),
      closed({ trade_id: "L2", closed_date: "2026-02-01", realized_pl: -300 }),
      closed({ trade_id: "BE", closed_date: "2026-01-01", realized_pl: 0 }),
    ];
    const r = trailingClosedLosses(trades, 5);
    expect(r.length).toBe(2);
    expect(r.map(t => t.trade_id)).toEqual(["L1", "L2"]); // sorted desc by closed_date
  });

  it("returns [] when no losses exist", () => {
    const trades = [closed({ realized_pl: 1000 }), closed({ realized_pl: 500 })];
    expect(trailingClosedLosses(trades, 30)).toEqual([]);
  });
});


// ═══════════════════════════════════════════════════════════════════════
// Edge Report helpers
// ═══════════════════════════════════════════════════════════════════════

function jrow(day: string, endNlv: number, marketWindow: string = ""): JournalHistoryPoint {
  return {
    day, end_nlv: endNlv, market_window: marketWindow,
    daily_pct_change: 0, portfolio_ltd: 0, pct_invested: 0, portfolio_heat: 0, score: 0,
  } as unknown as JournalHistoryPoint;
}


describe("getPriorDayNlv — strict '<'", () => {
  test("returns the last end_nlv strictly BEFORE the open date", () => {
    const j = [
      jrow("2026-04-01", 100_000),
      jrow("2026-04-02", 105_000),
      jrow("2026-04-03", 108_000),
    ];
    expect(getPriorDayNlv(j, "2026-04-03")).toBe(105_000);
    // Same-day open uses 04-02, NOT 04-03 (the strict-inequality fix).
    expect(getPriorDayNlv(j, "2026-04-02")).toBe(100_000);
  });

  test("returns null when no journal row precedes the open date", () => {
    expect(getPriorDayNlv([jrow("2026-04-05", 100_000)], "2026-04-03")).toBeNull();
  });

  test("empty / bad input → null", () => {
    expect(getPriorDayNlv([], "2026-04-01")).toBeNull();
    expect(getPriorDayNlv([jrow("2026-04-01", 100_000)], "")).toBeNull();
  });

  test("end_nlv = 0 treated as no data (defensive)", () => {
    expect(getPriorDayNlv([jrow("2026-04-01", 0)], "2026-04-05")).toBeNull();
  });
});


describe("tradeWasOpenInYear", () => {
  test("closed IN Y is in scope", () => {
    expect(tradeWasOpenInYear(
      closed({ open_date: "2026-03-01", closed_date: "2026-04-01" }), 2026,
    )).toBe(true);
  });

  test("opened Dec 2025 + closed Jan 2026 → in scope for 2026 (the Q2-answer bug fix)", () => {
    expect(tradeWasOpenInYear(
      closed({ open_date: "2025-12-15", closed_date: "2026-01-08" }), 2026,
    )).toBe(true);
  });

  test("opened Dec 2025 + still open → in scope for 2026", () => {
    expect(tradeWasOpenInYear(
      closed({ open_date: "2025-12-15", closed_date: null, status: "OPEN" }), 2026,
    )).toBe(true);
  });

  test("fully closed BEFORE year start → out of scope", () => {
    expect(tradeWasOpenInYear(
      closed({ open_date: "2025-06-01", closed_date: "2025-12-15" }), 2026,
    )).toBe(false);
  });

  test("opened AFTER year end → out of scope", () => {
    expect(tradeWasOpenInYear(
      closed({ open_date: "2027-01-05", closed_date: "2027-01-08" }), 2026,
    )).toBe(false);
  });
});


describe("availableTradeYears", () => {
  test("default = most recent year WITH DATA (not current calendar year)", () => {
    const closedTrades = [
      closed({ open_date: "2024-08-01", closed_date: "2025-02-01" }),
      closed({ open_date: "2026-03-01", closed_date: "2026-04-01" }),
    ];
    const openTrades = [closed({ open_date: "2026-05-01", closed_date: null, status: "OPEN" })];
    const { years, defaultYear } = availableTradeYears(closedTrades, openTrades);
    expect(years[0]).toBe(2024);
    expect(years).toContain(2026);
    // Avoids the empty-year problem on Jan 1 before 2027 has any trade.
    expect(defaultYear).toBe(2026);
  });

  test("empty data → current calendar year fallback", () => {
    const { years, defaultYear } = availableTradeYears([], []);
    const cy = new Date().getFullYear();
    expect(years).toEqual([cy]);
    expect(defaultYear).toBe(cy);
  });
});


describe("paretoDistribution", () => {
  test("ranks descending by P&L; cumulative + net computed correctly", () => {
    const trades = [
      closed({ trade_id: "A", realized_pl: 100 }),
      closed({ trade_id: "B", realized_pl: 500 }),
      closed({ trade_id: "C", realized_pl: -50 }),
      closed({ trade_id: "D", realized_pl: 200 }),
    ];
    const p = paretoDistribution(trades);
    expect(p.ranks.map(r => r.trade_id)).toEqual(["B", "D", "A", "C"]);
    expect(p.ranks.map(r => r.cumulative)).toEqual([500, 700, 800, 750]);
    expect(p.netPl).toBe(750);
  });

  test("break-even rank is the smallest N where cumulative ≥ net", () => {
    const trades = [
      closed({ trade_id: "A", realized_pl: 500 }),
      closed({ trade_id: "B", realized_pl: 300 }),
      closed({ trade_id: "C", realized_pl: -200 }),
      closed({ trade_id: "D", realized_pl: -100 }),
    ];
    // Sorted: 500, 300, -100, -200. Cum: 500, 800, 700, 500. Net = 500.
    expect(paretoDistribution(trades).breakevenRank).toBe(1);
  });

  test("topN summary", () => {
    const trades = [
      closed({ trade_id: "A", realized_pl: 100 }),
      closed({ trade_id: "B", realized_pl: 900 }),
    ];
    const p = paretoDistribution(trades);
    expect(p.topN(1)).toEqual({ net: 900, pctOfNet: 90, count: 1 });
    expect(p.topN(2)).toEqual({ net: 1000, pctOfNet: 100, count: 2 });
    expect(p.topN(10).count).toBe(2);
  });

  test("empty input → empty ranks + null breakeven", () => {
    const p = paretoDistribution([]);
    expect(p.ranks).toEqual([]);
    expect(p.netPl).toBe(0);
    expect(p.breakevenRank).toBeNull();
  });
});


describe("holdTimeBuckets", () => {
  test("5 buckets, classified by close_date − open_date", () => {
    const trades = [
      closed({ trade_id: "A", open_date: "2026-04-01", closed_date: "2026-04-01", realized_pl: -50 }), // 0d
      closed({ trade_id: "B", open_date: "2026-04-01", closed_date: "2026-04-04", realized_pl: 100 }), // 3d
      closed({ trade_id: "C", open_date: "2026-04-01", closed_date: "2026-04-11", realized_pl: 250 }), // 10d
      closed({ trade_id: "D", open_date: "2026-04-01", closed_date: "2026-05-01", realized_pl: -100 }), // 30d
      closed({ trade_id: "E", open_date: "2026-04-01", closed_date: "2026-05-31", realized_pl: 800 }),  // 60d
    ];
    const buckets = holdTimeBuckets(trades);
    expect(buckets.map(b => b.n)).toEqual([1, 1, 1, 1, 1]);
    expect(buckets[0].label).toBe("0–1 days");
    expect(buckets[4].label).toBe("41+ days");
    expect(buckets[4].hiInclusive).toBeNull();
  });

  test("win rate per bucket", () => {
    const trades = [
      closed({ open_date: "2026-04-01", closed_date: "2026-04-04", realized_pl: 100 }),
      closed({ open_date: "2026-04-02", closed_date: "2026-04-05", realized_pl: -50 }),
      closed({ open_date: "2026-04-03", closed_date: "2026-04-06", realized_pl: 200 }),
    ];
    const buckets = holdTimeBuckets(trades);
    expect(buckets[1].n).toBe(3);
    expect(buckets[1].winRate).toBeCloseTo(66.66, 1);
  });

  test("open trades (no closed_date) are skipped", () => {
    const trades = [closed({ open_date: "2026-04-01", closed_date: null, realized_pl: 100 })];
    const buckets = holdTimeBuckets(trades);
    expect(buckets.reduce((a, b) => a + b.n, 0)).toBe(0);
  });
});


describe("brandtNormalized", () => {
  test("avg trade / winner / loser as % of prior-day NAV", () => {
    const j = [jrow("2026-03-31", 100_000), jrow("2026-04-30", 100_000)];
    const trades = [
      closed({ open_date: "2026-04-01", realized_pl: 500 }),
      closed({ open_date: "2026-04-02", realized_pl: 1500 }),
      closed({ open_date: "2026-05-01", realized_pl: -300 }),
    ];
    const b = brandtNormalized(trades, j);
    expect(b.avgTradePctNav).toBeCloseTo((0.5 + 1.5 + -0.3) / 3, 4);
    expect(b.avgWinPctNav).toBeCloseTo((0.5 + 1.5) / 2, 4);
    expect(b.avgLossPctNav).toBeCloseTo(-0.3, 4);
    expect(b.nWithNlv).toBe(3);
  });

  test("trades without prior-day NLV excluded from pct math but counted in n", () => {
    const j = [jrow("2026-04-05", 100_000)];
    const trades = [
      closed({ open_date: "2026-04-01", realized_pl: 500 }),  // no prior-day journal row
      closed({ open_date: "2026-04-10", realized_pl: 100 }),
    ];
    const b = brandtNormalized(trades, j);
    expect(b.n).toBe(2);
    expect(b.nWithNlv).toBe(1);
    expect(b.avgTradePctNav).toBeCloseTo(0.1, 4);
  });
});


describe("stopCapScenario", () => {
  test("counts breaches and sums dollars past the cap", () => {
    const trades = [
      // -8% return, $1000 cost. Cap 3%: (8-3)/100 * 1000 = $50 saved.
      closed({ realized_pl: -80, total_cost: 1000, ...({ return_pct: -8 } as any) }),
      closed({ realized_pl: -20, total_cost: 1000, ...({ return_pct: -2 } as any) }),
      closed({ realized_pl: 500, total_cost: 1000, ...({ return_pct: 50 } as any) }),
    ];
    const rows = stopCapScenario(trades, [3, 5]);
    expect(rows[0].breachCount).toBe(1);
    expect(rows[0].dollarsSaved).toBeCloseTo(50, 3);
    expect(rows[1].dollarsSaved).toBeCloseTo(30, 3);
  });

  test("clippedWinnerCount = 0 without maePctOf (upper-bound mode)", () => {
    const trades = [closed({ realized_pl: 200, total_cost: 1000, ...({ return_pct: 20 } as any) })];
    const rows = stopCapScenario(trades, [5]);
    expect(rows[0].clippedWinnerCount).toBe(0);
    expect(rows[0].clippedWinnerForegonePl).toBe(0);
  });

  test("clippedWinner accounting fires when maePctOf is provided", () => {
    const trades = [
      closed({ trade_id: "W", realized_pl: 200, total_cost: 1000,
               ...({ return_pct: 20, mae_pct: -8 } as any) }),
      closed({ trade_id: "L", realized_pl: -80, total_cost: 1000,
               ...({ return_pct: -8, mae_pct: -8 } as any) }),
    ];
    const rows = stopCapScenario(trades, [5], {
      maePctOf: t => Number((t as any).mae_pct ?? null),
    });
    expect(rows[0].clippedWinnerCount).toBe(1);
    expect(rows[0].clippedWinnerForegonePl).toBeCloseTo(200, 3);
  });
});


describe("fixedSizeScenario", () => {
  test("scales P&L linearly by (target / actual b_size_pct)", () => {
    const j = [jrow("2026-03-31", 100_000)];
    const trades = [closed({ open_date: "2026-04-01", total_cost: 10_000, realized_pl: 1_000 })];
    const rows = fixedSizeScenario(trades, j, [10, 15]);
    expect(rows[0].scaledPnl).toBe(1_000);   // T === actual → unchanged
    expect(rows[1].scaledPnl).toBe(1_500);   // 15/10 × 1000
  });

  test("trades missing NLV are dropped, counted in nDropped", () => {
    const trades = [closed({ open_date: "2026-04-01", total_cost: 10_000, realized_pl: 100 })];
    const rows = fixedSizeScenario(trades, [], [10]);
    expect(rows[0].nWithSize).toBe(0);
    expect(rows[0].nDropped).toBe(1);
    expect(rows[0].scaledPnl).toBe(0);
  });
});


describe("regimeCrossTab", () => {
  test("groups by open_month × market_window on open_date", () => {
    const j = [
      jrow("2026-04-01", 100_000, "UPTREND"),
      jrow("2026-05-01", 100_000, "POWERTREND"),
    ];
    const trades = [
      closed({ open_date: "2026-04-05", realized_pl: 100 }),
      closed({ open_date: "2026-04-10", realized_pl: 200 }),
      closed({ open_date: "2026-05-05", realized_pl: -50 }),
    ];
    const cells = regimeCrossTab(trades, j);
    expect(cells).toHaveLength(2);
    expect(cells[0]).toMatchObject({ month: "2026-04", window: "UPTREND", n: 2, netPl: 300 });
    expect(cells[0].winRate).toBe(100);
    expect(cells[1]).toMatchObject({ month: "2026-05", window: "POWERTREND", n: 1, netPl: -50 });
  });

  test("trades without a journal row on/before open_date → window = 'Unknown'", () => {
    const trades = [closed({ open_date: "2026-04-05", realized_pl: 100 })];
    expect(regimeCrossTab(trades, [])[0].window).toBe("Unknown");
  });

  test("regimeOf opt overrides the journal.market_window lookup", () => {
    const trades = [
      closed({ open_date: "2026-04-05", realized_pl: 100 }),
      closed({ open_date: "2026-04-10", realized_pl: 200 }),
    ];
    const cells = regimeCrossTab(trades, [], undefined, {
      regimeOf: () => "POWERTREND",
    });
    expect(cells).toHaveLength(1);
    expect(cells[0]).toMatchObject({ window: "POWERTREND", n: 2, netPl: 300 });
  });
});


describe("setupScorecard", () => {
  test("groups by rule, computes PF + verdict, sorts by PF desc", () => {
    // 5 trades on rule A: 4 winners (+100 each = +400), 1 loser (-50) → PF = 8
    // 5 trades on rule B: 2 winners (+50 each = +100), 3 losers (-100 each = -300) → PF = 0.33
    const trades = [
      ...Array.from({ length: 4 }, (_, i) => closed({ trade_id: `A${i}`, rule: "A", realized_pl: 100 })),
      closed({ trade_id: "A4", rule: "A", realized_pl: -50 }),
      ...Array.from({ length: 2 }, (_, i) => closed({ trade_id: `B${i}`, rule: "B", realized_pl: 50 })),
      ...Array.from({ length: 3 }, (_, i) => closed({ trade_id: `B${i + 2}`, rule: "B", realized_pl: -100 })),
    ];
    const rows = setupScorecard(trades);
    expect(rows).toHaveLength(2);
    expect(rows[0].setup).toBe("A");
    expect(rows[0].profitFactor).toBeCloseTo(8, 5);
    expect(rows[0].verdict).toBe("core");
    expect(rows[1].setup).toBe("B");
    expect(rows[1].profitFactor).toBeCloseTo(1/3, 5);
    expect(rows[1].verdict).toBe("kill");
  });

  test("small n (< minN) parked at end without verdict", () => {
    const trades = [
      ...Array.from({ length: 5 }, (_, i) => closed({ trade_id: `A${i}`, rule: "A", realized_pl: -100 })),
      closed({ trade_id: "B0", rule: "B", realized_pl: 500 }), // n=1 winner
      closed({ trade_id: "B1", rule: "B", realized_pl: 500 }),
    ];
    const rows = setupScorecard(trades);
    expect(rows).toHaveLength(2);
    expect(rows[0].setup).toBe("A"); // qualifying first even though it's a kill
    expect(rows[1].setup).toBe("B");
    expect(rows[1].verdict).toBe("small-n");
  });

  test("PF = Infinity when only winners", () => {
    const trades = Array.from({ length: 5 }, (_, i) => closed({ trade_id: `T${i}`, rule: "X", realized_pl: 100 }));
    const rows = setupScorecard(trades);
    expect(rows[0].profitFactor).toBe(Infinity);
    expect(rows[0].verdict).toBe("core");
  });

  test("empty rule → '(unlabeled)' bucket", () => {
    const trades = Array.from({ length: 5 }, (_, i) => closed({ trade_id: `T${i}`, rule: "", realized_pl: 10 }));
    expect(setupScorecard(trades)[0].setup).toBe("(unlabeled)");
  });

  test("verdict thresholds — PF 2 boundary is inclusive → keep", () => {
    // 5 trades: 2 winners at +200 (+400 total), 2 losers at -100 (-200 total)
    // Add one more +100 winner → wins=+500, losses=-200 → PF=2.5. Aim for exactly 2:
    // 4 winners at +100 (+400), 2 losers at -100 (-200) → PF=2 (need n≥5, so 6 total is fine)
    const trades = [
      ...Array.from({ length: 4 }, (_, i) => closed({ trade_id: `W${i}`, rule: "K", realized_pl: 100 })),
      ...Array.from({ length: 2 }, (_, i) => closed({ trade_id: `L${i}`, rule: "K", realized_pl: -100 })),
    ];
    const row = setupScorecard(trades)[0];
    expect(row.profitFactor).toBeCloseTo(2, 5);
    expect(row.verdict).toBe("keep");
  });
});


describe("riskMetrics", () => {
  test("computes avg stop distance from entry, average loss, median position size", () => {
    const j: JournalHistoryPoint[] = [
      { day: "2026-03-31", end_nlv: 100_000 } as any,
    ];
    const trades = [
      closed({ trade_id: "T1", ticker: "A", open_date: "2026-04-05",
               avg_entry: 100, stop_loss: 92, total_cost: 10_000, realized_pl: -500, return_pct: -5 }),
      closed({ trade_id: "T2", ticker: "B", open_date: "2026-04-05",
               avg_entry: 100, stop_loss: 95, total_cost: 5_000, realized_pl: 1_000, return_pct: 20 }),
    ];
    const m = riskMetrics(trades, j);
    // stop distances: (100-92)/100=8%, (100-95)/100=5% → avg 6.5%
    expect(m.avgStopDistancePct).toBeCloseTo(6.5, 5);
    expect(m.nWithStop).toBe(2);
    // Only 1 loser at -5%
    expect(m.avgRealizedLossPct).toBe(-5);
    expect(m.nLosers).toBe(1);
    // Positions: 10_000 / 100_000 = 10%, 5_000 / 100_000 = 5% → median = 7.5%
    expect(m.medianPositionSizePct).toBe(7.5);
    // Risk: avg pos 7.5% × avg stop 6.5% / 100 = 0.4875%
    expect(m.avgRiskPerTradePct).toBeCloseTo(0.4875, 5);
  });

  test("stop_loss >= entry is discarded (invalid)", () => {
    const trades = [
      closed({ avg_entry: 100, stop_loss: 100, total_cost: 1_000, realized_pl: 0 }),
      closed({ avg_entry: 100, stop_loss: 110, total_cost: 1_000, realized_pl: 0 }),
      closed({ avg_entry: 100, stop_loss: 90,  total_cost: 1_000, realized_pl: 0 }),
    ];
    const m = riskMetrics(trades, []);
    expect(m.nWithStop).toBe(1);
    expect(m.avgStopDistancePct).toBe(10);
  });

  test("empty cohort → all nulls, zeros", () => {
    const m = riskMetrics([], []);
    expect(m.n).toBe(0);
    expect(m.avgStopDistancePct).toBeNull();
    expect(m.avgRealizedLossPct).toBeNull();
    expect(m.medianPositionSizePct).toBeNull();
    expect(m.avgRiskPerTradePct).toBeNull();
  });
});


describe("repeatOffenders", () => {
  test("groups by ticker; skips those below minAttempts", () => {
    const trades = [
      closed({ trade_id: "T1", ticker: "GEV", realized_pl: -1000 }),
      closed({ trade_id: "T2", ticker: "GEV", realized_pl: -2000 }),
      closed({ trade_id: "T3", ticker: "GEV", realized_pl: 500 }),
      closed({ trade_id: "T4", ticker: "ALAB", realized_pl: 5000 }),
      closed({ trade_id: "T5", ticker: "ALAB", realized_pl: -100 }),
    ];
    const out = repeatOffenders(trades);
    expect(out).toHaveLength(1); // Only GEV has ≥ 3
    expect(out[0].ticker).toBe("GEV");
    expect(out[0].attempts).toBe(3);
    expect(out[0].netPl).toBe(-2500);
    expect(out[0].wins).toBe(1);
    expect(out[0].losses).toBe(2);
    expect(out[0].bestPl).toBe(500);
    expect(out[0].worstPl).toBe(-2000);
  });

  test("sorted by netPl asc (biggest bleeders first)", () => {
    const mk = (t: string, pl: number) => closed({ ticker: t, realized_pl: pl });
    const trades = [
      mk("A", 100), mk("A", 100), mk("A", 100),   // net +300
      mk("B", -500), mk("B", -500), mk("B", -500), // net -1500
      mk("C", -50), mk("C", -50), mk("C", -50),    // net -150
    ];
    const out = repeatOffenders(trades);
    expect(out.map(o => o.ticker)).toEqual(["B", "C", "A"]);
  });

  test("options join to underlying via leading symbol", () => {
    const trades = [
      closed({ ticker: "HOOD 260918 $110C", realized_pl: -200, instrument_type: "OPTION" as any }),
      closed({ ticker: "HOOD 260918 $110C", realized_pl: -100, instrument_type: "OPTION" as any }),
      closed({ ticker: "HOOD", realized_pl: 1000 }),
    ];
    const out = repeatOffenders(trades, { minAttempts: 3 });
    expect(out).toHaveLength(1);
    expect(out[0].ticker).toBe("HOOD");
    expect(out[0].attempts).toBe(3);
    expect(out[0].netPl).toBe(700);
  });
});


describe("makeMctStateResolver", () => {
  test("returns state as of on-or-before date", () => {
    const resolver = makeMctStateResolver([
      { trade_date: "2026-04-01", state: "UPTREND" },
      { trade_date: "2026-04-22", state: "POWERTREND" },
      { trade_date: "2026-06-05", state: "CORRECTION" },
    ]);
    expect(resolver("2026-04-10")).toBe("UPTREND");
    expect(resolver("2026-04-22")).toBe("POWERTREND");
    expect(resolver("2026-05-15")).toBe("POWERTREND");
    expect(resolver("2026-07-01")).toBe("CORRECTION");
    expect(resolver("2026-03-01")).toBe("");
  });

  test("empty input → resolver returns ''", () => {
    const resolver = makeMctStateResolver([]);
    expect(resolver("2026-04-10")).toBe("");
  });
});


describe("generateInsights", () => {
  test("flags kill setups (n≥5, PF<1)", () => {
    const trades = [
      ...Array.from({ length: 5 }, (_, i) =>
        closed({ trade_id: `T${i}`, rule: "PB 21e", realized_pl: -1000 })),
    ];
    const ins = generateInsights(trades);
    const kill = ins.find(i => i.id === "kill-setups");
    expect(kill).toBeDefined();
    expect(kill!.severity).toBe("critical");
    expect(kill!.items).toHaveLength(1);
    expect(kill!.impactDollars).toBe(-5000);
  });

  test("flags penalty-box tickers (≥ minAttempts with net loss)", () => {
    const trades = [
      closed({ ticker: "GEV", realized_pl: -1000 }),
      closed({ ticker: "GEV", realized_pl: -1000 }),
      closed({ ticker: "GEV", realized_pl: -1000 }),
    ];
    const ins = generateInsights(trades);
    const penalty = ins.find(i => i.id === "penalty-box");
    expect(penalty).toBeDefined();
    expect(penalty!.items![0].label).toBe("GEV");
  });

  test("catastrophe stop counts losers past cap", () => {
    const trades = [
      closed({ ticker: "A", realized_pl: -1200, total_cost: 10_000, return_pct: -12 }),
      closed({ ticker: "B", realized_pl: -500, total_cost: 10_000, return_pct: -5 }),
    ];
    const ins = generateInsights(trades, { catastropheStopPct: 8 });
    const stop = ins.find(i => i.id === "catastrophe-stop");
    expect(stop).toBeDefined();
    expect(stop!.items).toHaveLength(1);
    // Savings = (12 - 8)% × 10_000 = $400
    expect(stop!.impactDollars).toBeCloseTo(400, 2);
  });

  test("correction-window bleed flags net loss during CORRECTION", () => {
    const trades = [
      closed({ open_date: "2026-06-15", realized_pl: -2000 }),
      closed({ open_date: "2026-06-20", realized_pl: -1000 }),
    ];
    const ins = generateInsights(trades, {
      mctStateResolver: () => "CORRECTION",
    });
    const bleed = ins.find(i => i.id === "correction-bleed");
    expect(bleed).toBeDefined();
    expect(bleed!.severity).toBe("critical");
    expect(bleed!.impactDollars).toBe(-3000);
  });

  test("no insights when the cohort is empty", () => {
    expect(generateInsights([])).toEqual([]);
  });

  test("overtrading flags weeks over the budget", () => {
    // 10 trades in the same week (2026-01-05..2026-01-11)
    const trades = Array.from({ length: 10 }, (_, i) =>
      closed({ trade_id: `T${i}`, open_date: `2026-01-0${(i % 5) + 5}` }));
    const ins = generateInsights(trades, { entryBudgetPerWeek: 5 });
    const over = ins.find(i => i.id === "overtrading");
    expect(over).toBeDefined();
    expect(over!.items!.length).toBeGreaterThan(0);
  });
});


describe("winnerMaeDistribution", () => {
  test("buckets winners by |MAE| depth", () => {
    const trades = [
      closed({ realized_pl: 100, mae_pct: -1.0 }),   // bucket 0-2
      closed({ realized_pl: 100, mae_pct: -3.5 }),   // 2-5
      closed({ realized_pl: 100, mae_pct: -6.0 }),   // 5-8
      closed({ realized_pl: 100, mae_pct: -9.0 }),   // 8-12
      closed({ realized_pl: 100, mae_pct: -15.0 }),  // ≥12
      closed({ realized_pl: -100, mae_pct: -5.0 }),  // ← loser, excluded
    ];
    const r = winnerMaeDistribution(trades as any);
    expect(r.n).toBe(5);
    expect(r.buckets.map(b => b.n)).toEqual([1, 1, 1, 1, 1]);
    expect(r.buckets.map(b => Math.round(b.pct))).toEqual([20, 20, 20, 20, 20]);
  });

  test("skips trades without mae_pct", () => {
    const trades = [
      closed({ realized_pl: 100 }),                 // no mae — excluded
      closed({ realized_pl: 100, mae_pct: -3.0 }),  // included
    ];
    expect(winnerMaeDistribution(trades as any).n).toBe(1);
  });

  test("empty cohort → empty buckets, zero pct", () => {
    const r = winnerMaeDistribution([]);
    expect(r.n).toBe(0);
    expect(r.buckets.every(b => b.n === 0 && b.pct === 0)).toBe(true);
  });
});


describe("loserMfeDistribution", () => {
  test("buckets losers by MFE magnitude, excludes winners", () => {
    const trades = [
      closed({ realized_pl: -100, mfe_pct: 1.5 }),   // 0-2
      closed({ realized_pl: -100, mfe_pct: 4.0 }),   // 2-5
      closed({ realized_pl: -100, mfe_pct: 8.0 }),   // 5-10
      closed({ realized_pl: -100, mfe_pct: 15.0 }),  // 10-20
      closed({ realized_pl: -100, mfe_pct: 30.0 }),  // ≥20
      closed({ realized_pl: 100, mfe_pct: 50.0 }),   // winner — excluded
    ];
    const r = loserMfeDistribution(trades as any);
    expect(r.n).toBe(5);
    expect(r.buckets.map(b => b.n)).toEqual([1, 1, 1, 1, 1]);
  });

  test("bucket boundaries are [lo, hi) — hi is EXCLUSIVE", () => {
    // 5.0% falls into the [5, 10) bucket, not [2, 5).
    const trades = [closed({ realized_pl: -100, mfe_pct: 5.0 })];
    const r = loserMfeDistribution(trades as any);
    expect(r.buckets[1].n).toBe(0);  // 2-5
    expect(r.buckets[2].n).toBe(1);  // 5-10
  });
});


describe("entryQualityBySetup", () => {
  test("computes per-setup avg MAE / winner-MAE / worst MAE", () => {
    const trades = [
      // A: 5 trades, 3 winners
      closed({ rule: "A", realized_pl: 100, mae_pct: -2 }),
      closed({ rule: "A", realized_pl: 100, mae_pct: -4 }),
      closed({ rule: "A", realized_pl: 100, mae_pct: -6 }),
      closed({ rule: "A", realized_pl: -50, mae_pct: -10 }),
      closed({ rule: "A", realized_pl: -50, mae_pct: -12 }),
    ];
    const rows = entryQualityBySetup(trades as any);
    expect(rows).toHaveLength(1);
    const a = rows[0];
    expect(a.n).toBe(5);
    // avgMae = (-2-4-6-10-12)/5 = -6.8
    expect(a.avgMae).toBeCloseTo(-6.8, 5);
    // avgMaeOnWinners = (-2-4-6)/3 = -4
    expect(a.avgMaeOnWinners).toBeCloseTo(-4, 5);
    expect(a.worstMae).toBe(-12);
  });

  test("filters out setups below minN", () => {
    const trades = [
      ...Array.from({ length: 5 }, (_, i) => closed({ rule: "A", realized_pl: 10, mae_pct: -1 })),
      closed({ rule: "B", realized_pl: 10, mae_pct: -1 }),   // n=1 only
    ];
    const rows = entryQualityBySetup(trades as any);
    expect(rows.map(r => r.setup)).toEqual(["A"]);
  });

  test("skips trades without mae_pct", () => {
    const trades = [
      closed({ rule: "A", realized_pl: 10, mae_pct: -1 }),
      closed({ rule: "A", realized_pl: 10 }),  // no mae — excluded
      closed({ rule: "A", realized_pl: 10, mae_pct: -1 }),
      closed({ rule: "A", realized_pl: 10, mae_pct: -1 }),
      closed({ rule: "A", realized_pl: 10, mae_pct: -1 }),
      closed({ rule: "A", realized_pl: 10, mae_pct: -1 }),
    ];
    const rows = entryQualityBySetup(trades as any);
    expect(rows[0].n).toBe(5);   // 5 with mae, 1 excluded
  });

  test("sorted by avgMaeOnWinners ascending (most punishing first)", () => {
    // A: winner MAE avg -2
    // B: winner MAE avg -8
    // C: no winners → sorted last
    const mk = (rule: string, pl: number, mae: number) =>
      closed({ rule, realized_pl: pl, mae_pct: mae });
    const trades = [
      ...Array.from({ length: 5 }, () => mk("A", 100, -2)),
      ...Array.from({ length: 5 }, () => mk("B", 100, -8)),
      ...Array.from({ length: 5 }, () => mk("C", -100, -5)),   // all losers
    ];
    const rows = entryQualityBySetup(trades as any);
    // Most punishing WINNER MAE first (B: -8), then A (-2), then C (no winners → end)
    expect(rows.map(r => r.setup)).toEqual(["B", "A", "C"]);
  });
});
