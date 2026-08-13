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

  it("open_risk = (current − stop) × shares × mult (DELL winner scenario)", () => {
    // DELL: 225 sh @ avg $331.62, stop $359.44, current $453.77.
    // open_risk = ($453.77 − $359.44) × 225 = $21,224.25.
    const trade: TradePosition = {
      trade_id: "T-DELL", ticker: "DELL", status: "OPEN",
      shares: 225, avg_entry: 331.62, total_cost: 74615.19,
      realized_pl: 36349.21, stop_loss: 359.44, rule: "",
      instrument_type: "STOCK", multiplier: 1, open_date: "2026-04-06",
    } as any;
    const details: TradeDetail[] = [
      { trade_id: "T-DELL", action: "BUY", date: "2026-04-06",
        shares: 225, amount: 331.62, stop_loss: 359.44 } as any,
    ];
    const [enriched] = computeEnrichedPositions([trade], details, 589_400, { DELL: 453.77 });
    expect(enriched.avg_stop).toBeCloseTo(359.44, 2);
    expect(enriched.open_risk).toBeCloseTo(21224.25, 2);
    expect(enriched.open_risk_pct).toBeCloseTo((21224.25 / 589_400) * 100, 4);
  });

  it("open_risk for a LOSER uses (current − stop), NOT (entry − stop) — MU scenario", () => {
    // Regression: prior anchor-to-MAX(current, entry) implementation
    // double-counted MU's paper loss (once in Overall P&L as unrealized,
    // once in Open Risk as loss-still-to-come). Coach Claude audit
    // 2026-08-08. Correct value: ($877.57 − $847.84) × 80 = $2,378.40.
    // Buggy value under old formula: $5,038.40. Preserves the invariant
    // Overall P&L − Open Risk = Projected P&L.
    const trade: TradePosition = {
      trade_id: "T-MU", ticker: "MU", status: "OPEN",
      shares: 80, avg_entry: 910.82, total_cost: 72865.60,
      realized_pl: 0, stop_loss: 847.84, rule: "",
      instrument_type: "STOCK", multiplier: 1, open_date: "2026-08-05",
    } as any;
    const details: TradeDetail[] = [
      { trade_id: "T-MU", action: "BUY", date: "2026-08-05",
        shares: 80, amount: 910.82, stop_loss: 847.84 } as any,
    ];
    const [enriched] = computeEnrichedPositions([trade], details, 589_400, { MU: 877.57 });
    expect(enriched.open_risk).toBeCloseTo(2378.40, 1);
  });

  it("open_risk for a FRESH position ≈ Trade Risk $ (current = entry)", () => {
    // Current equals entry → open_risk = (entry − stop) × shares.
    const trade: TradePosition = {
      trade_id: "T-FRESH", ticker: "X", status: "OPEN",
      shares: 100, avg_entry: 100, total_cost: 10000,
      realized_pl: 0, stop_loss: 95, rule: "",
      instrument_type: "STOCK", multiplier: 1, open_date: "2026-01-01",
    } as any;
    const details: TradeDetail[] = [
      { trade_id: "T-FRESH", action: "BUY", date: "2026-01-01",
        shares: 100, amount: 100, stop_loss: 95 } as any,
    ];
    const [enriched] = computeEnrichedPositions([trade], details, 100_000, { X: 100 });
    expect(enriched.open_risk).toBeCloseTo(500, 2);
  });

  it("open_risk floors at 0 when stop is at or above current price", () => {
    // Anomaly: stop above current. In practice the stop should have
    // fired. Floor at 0 to keep the display non-negative.
    const trade: TradePosition = {
      trade_id: "T-ANOMALY", ticker: "X", status: "OPEN",
      shares: 100, avg_entry: 100, total_cost: 10000,
      realized_pl: 0, stop_loss: 110, rule: "",
      instrument_type: "STOCK", multiplier: 1, open_date: "2026-01-01",
    } as any;
    const details: TradeDetail[] = [
      { trade_id: "T-ANOMALY", action: "BUY", date: "2026-01-01",
        shares: 100, amount: 100, stop_loss: 110 } as any,
    ];
    const [enriched] = computeEnrichedPositions([trade], details, 100_000, { X: 105 });
    expect(enriched.open_risk).toBe(0);
    expect(enriched.open_risk_pct).toBe(0);
  });

  it("open_risk for a long option = current × shares × 100 (worst case = premium loss)", () => {
    // Long option: effective stop = 0. open_risk = current × shares × mult.
    const trade: TradePosition = {
      trade_id: "O-OPT", ticker: "AAPL  260117C00150000", status: "OPEN",
      shares: 5, avg_entry: 2.0, total_cost: 1000,
      realized_pl: 0, stop_loss: 1.0, rule: "",
      instrument_type: "OPTION", multiplier: 100, open_date: "2026-01-01",
    } as any;
    const details: TradeDetail[] = [
      { trade_id: "O-OPT", action: "BUY", date: "2026-01-01",
        shares: 5, amount: 2.0, stop_loss: 1.0 } as any,
    ];
    const [enriched] = computeEnrichedPositions([trade], details, 100_000, { "AAPL  260117C00150000": 3.0 });
    // current $3 × 5 contracts × 100 = $1,500 (worst case = full premium loss).
    expect(enriched.open_risk).toBeCloseTo(1500);
    expect(enriched.current_value).toBeCloseTo(1500);
  });

  it("invariant: Overall P&L − Open Risk = Projected P&L (winner and loser)", () => {
    // Coach Claude's cross-check. The three fleet numbers must
    // reconcile per position AND in the header. Broke under the
    // 2026-08-08 MAX(current, entry) anchor on losers; this test
    // locks the correct formula in place going forward.
    const winner: TradePosition = {
      trade_id: "T-W", ticker: "W", status: "OPEN",
      shares: 100, avg_entry: 100, total_cost: 10000, realized_pl: 0,
      stop_loss: 105, rule: "", instrument_type: "STOCK", multiplier: 1,
      open_date: "2026-01-01",
    } as any;
    const wDetails: TradeDetail[] = [
      { trade_id: "T-W", action: "BUY", date: "2026-01-01",
        shares: 100, amount: 100, stop_loss: 105 } as any,
    ];
    const [w] = computeEnrichedPositions([winner], wDetails, 100_000, { W: 120 });
    expect(w.overall_pl - w.open_risk).toBeCloseTo(w.projected_pl, 4);

    const loser: TradePosition = {
      trade_id: "T-L", ticker: "L", status: "OPEN",
      shares: 100, avg_entry: 100, total_cost: 10000, realized_pl: 0,
      stop_loss: 85, rule: "", instrument_type: "STOCK", multiplier: 1,
      open_date: "2026-01-01",
    } as any;
    const lDetails: TradeDetail[] = [
      { trade_id: "T-L", action: "BUY", date: "2026-01-01",
        shares: 100, amount: 100, stop_loss: 85 } as any,
    ];
    const [l] = computeEnrichedPositions([loser], lDetails, 100_000, { L: 90 });
    expect(l.overall_pl - l.open_risk).toBeCloseTo(l.projected_pl, 4);
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

  // Migration 062 (2026-08-07) reshaped the ladder: peak ≥ 50 now maps
  // to SR7 by default (cushion-qualified but undeclared); SR8 requires
  // an explicit is_declared_sr8 flag. The tests below use the derived
  // TradePosition shape which passes is_declared_sr8 through, defaulting
  // to false → SR7 for the peak-crossed cases.

  it("COHR pullback case: stored max 70%, current 30% → SR7 (no demote, undeclared)", () => {
    const p = singleStock({ b1Entry: 100, b1Max: 70, livePrice: 130 });
    expect(p.b1_return_pct).toBeCloseTo(30);
    expect(p.b1_max_return_pct).toBeCloseTo(70);
    expect(p.sell_rule_tier).toBe("sr7");
  });

  it("new peak: stored 30%, current 55% → SR7 (effective max = 55, undeclared)", () => {
    const p = singleStock({ b1Entry: 100, b1Max: 30, livePrice: 155 });
    expect(p.sell_rule_tier).toBe("sr7");
  });

  it("brand-new position post-deploy: stored null, current 5% → SR1", () => {
    const p = singleStock({ b1Entry: 100, b1Max: null, livePrice: 105 });
    expect(p.b1_max_return_pct).toBeNull();
    expect(p.sell_rule_tier).toBe("sr1");
  });

  it("only stored set (no current price data) → tier from stored", () => {
    // currentPrice falls back to summaryEntry (100) when livePrice missing;
    // that produces b1_return_pct=0 against b1_entry=100. The stored max
    // of 60 wins via Math.max → effective 60 → SR7 (cushion-qualified
    // but undeclared).
    const p = singleStock({ b1Entry: 100, b1Max: 60 });
    expect(p.b1_return_pct).toBeCloseTo(0);
    expect(p.sell_rule_tier).toBe("sr7");
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

  it("boundary: stored exactly at 50% → SR7 (cushion-qualified, undeclared)", () => {
    const p = singleStock({ b1Entry: 100, b1Max: 50, livePrice: 100 });
    expect(p.sell_rule_tier).toBe("sr7");
  });

  it("boundary: stored 49.99%, current 49.99% → SR15 (20-50 band)", () => {
    // Range shifted by migration 062: 10-20 = SR11, 20-50 = SR15
    // (new tier, +10% profit-lock band).
    const p = singleStock({ b1Entry: 100, b1Max: 49.99, livePrice: 149.99 });
    expect(p.sell_rule_tier).toBe("sr15");
  });
});

describe("computeEnrichedPositions — strategy-based tier override (br7.1 → SR7)", () => {
  // 2026-08-12: TQQQ Strategy positions (br7.1) are managed via 21EMA
  // violation cascade regardless of b1_return band. Encoded as a buy-
  // rule-prefix override on the classifier. Scope: PRIMARY rule of B1
  // only; confluence rules don't trigger the override.

  function tqqqPosition(opts: {
    buyRule?: string | null;
    b1Max?: number | null;
    livePrice?: number;
    isDeclaredSr8?: boolean;
  }) {
    const trade = {
      trade_id: "T-TQQQ", ticker: "TQQQ", status: "OPEN",
      shares: 100, avg_entry: 70, total_cost: 7000, realized_pl: 0,
      rule: opts.buyRule === undefined ? "br7.1 TQQQ Strategy" : opts.buyRule,
      buy_rule: opts.buyRule === undefined ? "br7.1 TQQQ Strategy" : opts.buyRule,
      instrument_type: "STOCK", multiplier: 1, open_date: "2026-07-29",
      b1_entry_price: 70,
      b1_max_return_pct: opts.b1Max === undefined ? null : opts.b1Max,
      is_declared_sr8: opts.isDeclaredSr8 ?? false,
    } as any;
    const details: TradeDetail[] = [
      { trade_id: "T-TQQQ", action: "BUY", date: "2026-07-29",
        shares: 100, amount: 70, rule: trade.rule } as any,
    ];
    const livePrices: Record<string, number> =
      opts.livePrice !== undefined ? { TQQQ: opts.livePrice } : {};
    return computeEnrichedPositions([trade], details, 100_000, livePrices)[0];
  }

  it("br7.1 forces SR7 even at 1% peak (would default to SR1 without override)", () => {
    // The user's live CanSlim TQQQ scenario: peak ~1%, would land in SR1
    // band by the ladder; TQQQ Strategy doctrine overrides to SR7.
    const p = tqqqPosition({ b1Max: 1, livePrice: 70.7 });
    expect(p.sell_rule_tier).toBe("sr7");
  });

  it("br7.1 forces SR7 in the SR11 band (10-20%)", () => {
    const p = tqqqPosition({ b1Max: 15, livePrice: 80.5 });
    expect(p.sell_rule_tier).toBe("sr7");
  });

  it("br7.1 forces SR7 in the SR15 band (20-50%)", () => {
    const p = tqqqPosition({ b1Max: 35, livePrice: 94.5 });
    expect(p.sell_rule_tier).toBe("sr7");
  });

  it("br7.1 at 60% → SR7 (already what the ladder would say; override is a no-op)", () => {
    const p = tqqqPosition({ b1Max: 60, livePrice: 112 });
    expect(p.sell_rule_tier).toBe("sr7");
  });

  it("br7.1 + declared SR8 → SR8 wins (explicit doctrine beats implicit override)", () => {
    // Rare (leveraged ETF isn't typical monster) but the invariant matters:
    // explicit user declaration always wins over strategy defaults.
    const p = tqqqPosition({ b1Max: 60, livePrice: 112, isDeclaredSr8: true });
    expect(p.sell_rule_tier).toBe("sr8");
  });

  it("no buy rule (rule='') → normal ladder applies, no override", () => {
    // LTG-style TQQQ without a strategy tag: peak 25% → SR15 (unchanged).
    const p = tqqqPosition({ buyRule: "", b1Max: 25, livePrice: 87.5 });
    expect(p.sell_rule_tier).toBe("sr15");
  });

  it("different buy rule (br1.2 Cup w Handle) → normal ladder applies", () => {
    // Only br7.1 is in STRATEGY_TIER_OVERRIDES; other rules don't force
    // a tier. Same peak that produced SR15 above stays SR15.
    const p = tqqqPosition({ buyRule: "br1.2 Cup w Handle", b1Max: 25, livePrice: 87.5 });
    expect(p.sell_rule_tier).toBe("sr15");
  });

  it("case + whitespace tolerance on the prefix match", () => {
    // "  BR7.1   TQQQ Strategy" — leading/trailing whitespace + uppercase.
    const p = tqqqPosition({ buyRule: "  BR7.1   TQQQ Strategy", b1Max: 5, livePrice: 73.5 });
    expect(p.sell_rule_tier).toBe("sr7");
  });
});
