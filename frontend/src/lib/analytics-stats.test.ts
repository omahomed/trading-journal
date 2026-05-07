import { describe, it, expect } from "vitest";
import {
  computeWinRate,
  computeProfitFactor,
  computeHoldRatio,
  computeOnePctCompliance,
  computeLast10Stats,
  trailingClosedTrades,
  trailingClosedLosses,
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
