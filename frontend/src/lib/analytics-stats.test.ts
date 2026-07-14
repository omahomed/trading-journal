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
});
