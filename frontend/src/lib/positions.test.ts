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
