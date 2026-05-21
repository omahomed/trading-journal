import { describe, it, expect } from "vitest";
import { computeEnrichedPositions } from "./positions";
import type { TradePosition, TradeDetail } from "./api";

describe("computeEnrichedPositions", () => {
  it("overall_pl = unrealized + realized_bank for partial-closed open trade", () => {
    const trade: TradePosition = {
      trade_id: "T1",
      ticker: "AAPL",
      status: "OPEN",
      shares: 50,
      avg_entry: 100,
      total_cost: 5000,
      realized_pl: 1000,
      rule: "",
      instrument_type: "STOCK",
      multiplier: 1,
      open_date: "2026-01-01",
    } as any;

    const details: TradeDetail[] = [
      { trade_id: "T1", action: "BUY",  date: "2026-01-01", shares: 100, amount: 100 } as any,
      { trade_id: "T1", action: "SELL", date: "2026-02-01", shares: 50,  amount: 120 } as any,
    ];

    const [enriched] = computeEnrichedPositions([trade], details, 100_000, { AAPL: 130 });

    // Remaining 50 sh @ avg 100, current price 130 → unrealized = (130 - 100) * 50 * 1 = 1500
    expect(enriched.unrealized_pl).toBeCloseTo(1500);
    // LIFO realized bank: 50 sh sold @ 120 from a basis of 100 → (120 - 100) * 50 = 1000
    expect(enriched.realized_bank).toBeCloseTo(1000);
    // Overall P&L = unrealized + realized_bank
    expect(enriched.overall_pl).toBeCloseTo(2500);
  });

  it("OPTION trade applies multiplier=100 for unrealized P&L and current value", () => {
    const trade: TradePosition = {
      trade_id: "O1",
      ticker: "AAPL  260117C00150000",
      status: "OPEN",
      shares: 5,
      avg_entry: 2.0,
      total_cost: 1000,
      realized_pl: 0,
      rule: "",
      instrument_type: "OPTION",
      multiplier: 100,
      open_date: "2026-01-01",
    } as any;

    const details: TradeDetail[] = [
      { trade_id: "O1", action: "BUY", date: "2026-01-01", shares: 5, amount: 2.0 } as any,
    ];

    const [enriched] = computeEnrichedPositions([trade], details, 100_000, { "AAPL  260117C00150000": 3.0 });

    // (3 - 2) * 5 * 100 = 500
    expect(enriched.unrealized_pl).toBeCloseTo(500);
    // 5 contracts × $3.00 × 100 = $1,500
    expect(enriched.current_value).toBeCloseTo(1500);
    expect(enriched.is_option).toBe(true);
    expect(enriched.multiplier).toBe(100);
  });

  it("falls back to summary avg_entry as currentPrice when livePrices missing", () => {
    const trade: TradePosition = {
      trade_id: "T2",
      ticker: "MSFT",
      status: "OPEN",
      shares: 100,
      avg_entry: 400,
      total_cost: 40000,
      realized_pl: 0,
      rule: "",
      instrument_type: "STOCK",
      multiplier: 1,
      open_date: "2026-01-01",
    } as any;

    const details: TradeDetail[] = [
      { trade_id: "T2", action: "BUY", date: "2026-01-01", shares: 100, amount: 400 } as any,
    ];

    const [enriched] = computeEnrichedPositions([trade], details, 100_000, {});
    // No live price → currentPrice = summaryEntry = 400 → unrealized = 0
    expect(enriched.unrealized_pl).toBeCloseTo(0);
    expect(enriched.current_price).toBeCloseTo(400);
  });
});

describe("computeEnrichedPositions — Sell Rule tier (persistent b1_max_return_pct)", () => {
  // Single-position helper. Each test sets b1_entry_price / b1_max_return_pct
  // on the trade row and a live price; we assert which tier the classifier
  // resolves to. The point of the suite: pullbacks must not auto-demote.
  function singleStock(opts: {
    b1Entry?: number | null;
    b1Max?: number | null;
    livePrice?: number;
  }) {
    const trade = {
      trade_id: "T1",
      ticker: "AAPL",
      status: "OPEN",
      shares: 100,
      avg_entry: 100,
      total_cost: 10_000,
      realized_pl: 0,
      rule: "",
      instrument_type: "STOCK",
      multiplier: 1,
      open_date: "2026-01-01",
      b1_entry_price: opts.b1Entry === undefined ? 100 : opts.b1Entry,
      b1_max_return_pct: opts.b1Max === undefined ? null : opts.b1Max,
    } as any;
    const details: TradeDetail[] = [
      { trade_id: "T1", action: "BUY", date: "2026-01-01", shares: 100, amount: 100 } as any,
    ];
    const livePrices: Record<string, number> =
      opts.livePrice !== undefined ? { AAPL: opts.livePrice } : {};
    return computeEnrichedPositions([trade], details, 100_000, livePrices)[0];
  }

  it("COHR pullback case: stored max 70%, current 30% → SR8 (no demote)", () => {
    const p = singleStock({ b1Entry: 100, b1Max: 70, livePrice: 130 });
    expect(p.b1_return_pct).toBeCloseTo(30);
    expect(p.b1_max_return_pct).toBeCloseTo(70);
    expect(p.sell_rule_tier).toBe("sr8");
  });

  it("new peak: stored 30%, current 55% → SR8 (effective max = 55)", () => {
    const p = singleStock({ b1Entry: 100, b1Max: 30, livePrice: 155 });
    expect(p.sell_rule_tier).toBe("sr8");
  });

  it("brand-new position post-deploy: stored null, current 5% → SR1", () => {
    const p = singleStock({ b1Entry: 100, b1Max: null, livePrice: 105 });
    expect(p.b1_max_return_pct).toBeNull();
    expect(p.sell_rule_tier).toBe("sr1");
  });

  it("only stored set (no current price data) → tier from stored", () => {
    // currentPrice falls back to summaryEntry (100) when livePrice missing;
    // that produces b1_return_pct=0 against b1_entry=100. The stored max
    // of 60 wins via Math.max → effective 60 → SR8.
    const p = singleStock({ b1Entry: 100, b1Max: 60 });
    expect(p.b1_return_pct).toBeCloseTo(0);
    expect(p.sell_rule_tier).toBe("sr8");
  });

  it("both null → tier null (column renders dash)", () => {
    const p = singleStock({ b1Entry: null, b1Max: null });
    expect(p.b1_return_pct).toBeNull();
    expect(p.b1_max_return_pct).toBeNull();
    expect(p.sell_rule_tier).toBeNull();
  });

  it("stored = -10 (peaked at loss), current = -20 → SR1, no demote needed", () => {
    // Both negative; max(-20, -10) = -10 → still SR1. The point is the
    // classifier doesn't crash on negatives and Math.max is taken correctly.
    const p = singleStock({ b1Entry: 100, b1Max: -10, livePrice: 80 });
    expect(p.sell_rule_tier).toBe("sr1");
  });

  it("boundary: stored exactly at 50% → SR8 (≥ 50 ladder)", () => {
    const p = singleStock({ b1Entry: 100, b1Max: 50, livePrice: 100 });
    expect(p.sell_rule_tier).toBe("sr8");
  });

  it("boundary: stored 49.99%, current 49.99% → SR11 (still below 50)", () => {
    const p = singleStock({ b1Entry: 100, b1Max: 49.99, livePrice: 149.99 });
    expect(p.sell_rule_tier).toBe("sr11");
  });
});
