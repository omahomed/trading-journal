import { describe, test, expect } from "vitest";
import { classifyTradeState, type PyramidRules } from "./trade-state";
import type { EnrichedPosition } from "./positions";
import type { TradeDetail } from "./api";

const PYRAMID_RULES: PyramidRules = { trigger_pct: 5, alloc_pct: 20 };

function enrichedFixture(opts: {
  trade_id?: string;
  is_option?: boolean;
  pyramid_pct?: number;
} = {}): EnrichedPosition {
  return {
    trade_id: opts.trade_id ?? "T1",
    ticker: "AAA",
    shares: 100,
    avg_entry: 50,
    total_cost: 5000,
    realized_pl: 0,
    rule: "",
    buy_notes: "",
    risk_budget: 0,
    open_date: "2026-05-01",
    days_held: 5,
    avg_stop: 48,
    risk_dollars: 200,
    signed_risk: -200,
    risk_pct: -0.2,
    current_price: 55,
    current_value: 5500,
    unrealized_pl: 500,
    overall_pl: 500,
    return_pct: 10,
    pos_size_pct: 5.5,
    is_option: opts.is_option ?? false,
    multiplier: 1,
    pyramid_pct: opts.pyramid_pct ?? 0,
    risk_status: "At Risk",
    projected_pl: -200,
    projected_pct: -0.2,
    realized_bank: 0,
    expiration: null,
    manual_price: null,
    grade: null,
    strategy: null,
    b1_return_pct: null,
    b1_max_return_pct: null,
    sell_rule_tier: null,
  };
}

function detailFixture(opts: {
  trade_id: string;
  action?: "BUY" | "SELL";
  date?: string;
}): TradeDetail {
  return {
    trade_id: opts.trade_id,
    ticker: "AAA",
    action: opts.action ?? "BUY",
    date: opts.date ?? "2026-05-01",
    shares: 50,
    amount: 50,
    value: 2500,
    rule: "",
  };
}

describe("classifyTradeState", () => {
  test("equity option → 'call' regardless of lot count", () => {
    const enriched = enrichedFixture({ is_option: true, pyramid_pct: 12 });
    const details = [
      detailFixture({ trade_id: "T1" }),
      detailFixture({ trade_id: "T1", date: "2026-05-05" }),
    ];
    expect(classifyTradeState(enriched, details, PYRAMID_RULES)).toBe("call");
  });

  test("single BUY → 'original' (no scale-ins)", () => {
    const enriched = enrichedFixture({ pyramid_pct: 0 });
    const details = [detailFixture({ trade_id: "T1" })];
    expect(classifyTradeState(enriched, details, PYRAMID_RULES)).toBe("original");
  });

  test("zero BUY rows (defensive: malformed data) → 'original'", () => {
    const enriched = enrichedFixture();
    expect(classifyTradeState(enriched, [], PYRAMID_RULES)).toBe("original");
  });

  test("multi-BUY with last-add cushion past trigger → 'ready'", () => {
    const enriched = enrichedFixture({ pyramid_pct: 6 });
    const details = [
      detailFixture({ trade_id: "T1", date: "2026-05-01" }),
      detailFixture({ trade_id: "T1", date: "2026-05-05" }),
    ];
    expect(classifyTradeState(enriched, details, PYRAMID_RULES)).toBe("ready");
  });

  test("multi-BUY with last-add cushion below trigger → 'added'", () => {
    const enriched = enrichedFixture({ pyramid_pct: 3 });
    const details = [
      detailFixture({ trade_id: "T1", date: "2026-05-01" }),
      detailFixture({ trade_id: "T1", date: "2026-05-05" }),
    ];
    expect(classifyTradeState(enriched, details, PYRAMID_RULES)).toBe("added");
  });

  test("boundary: pyramid_pct === trigger_pct → 'ready' (inclusive)", () => {
    const enriched = enrichedFixture({ pyramid_pct: 5 });
    const details = [
      detailFixture({ trade_id: "T1", date: "2026-05-01" }),
      detailFixture({ trade_id: "T1", date: "2026-05-05" }),
    ];
    expect(classifyTradeState(enriched, details, PYRAMID_RULES)).toBe("ready");
  });

  test("multi-BUY at-or-below-zero cushion → 'added' (not ready)", () => {
    const enriched = enrichedFixture({ pyramid_pct: -2 });
    const details = [
      detailFixture({ trade_id: "T1", date: "2026-05-01" }),
      detailFixture({ trade_id: "T1", date: "2026-05-05" }),
    ];
    expect(classifyTradeState(enriched, details, PYRAMID_RULES)).toBe("added");
  });

  test("BUY count filtered by trade_id (SELLs ignored)", () => {
    const enriched = enrichedFixture({ trade_id: "T2", pyramid_pct: 6 });
    const details = [
      // Different trade_id — ignored.
      detailFixture({ trade_id: "T1" }),
      detailFixture({ trade_id: "T1", date: "2026-05-05" }),
      // Single BUY for T2; SELL doesn't count toward "multi-BUY".
      detailFixture({ trade_id: "T2", action: "BUY" }),
      detailFixture({ trade_id: "T2", action: "SELL", date: "2026-05-05" }),
    ];
    expect(classifyTradeState(enriched, details, PYRAMID_RULES)).toBe("original");
  });
});
