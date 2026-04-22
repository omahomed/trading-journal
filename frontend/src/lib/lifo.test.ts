import { describe, it, expect } from "vitest";
import { runLifoEngine } from "./lifo";
import type { TradeDetail } from "./api";

// Minimal fixture builder — TradeDetail has [key: string]: any so we can
// include stop_loss alongside the required fields.
function tx(partial: Partial<TradeDetail> & { stop_loss?: number }): TradeDetail {
  return {
    trade_id: "T1",
    ticker: "AAPL",
    action: "BUY",
    date: "2026-01-01",
    shares: 0,
    amount: 0,
    value: 0,
    rule: "",
    ...partial,
  };
}

describe("runLifoEngine", () => {
  describe("empty input", () => {
    it("returns zeros with summary fallbacks when no transactions", () => {
      const result = runLifoEngine([], 50, 100);
      expect(result).toEqual({
        risk: 0,
        avgStop: 50,
        avgCost: 50,
        projectedPl: 0,
        realizedBank: 0,
      });
    });
  });

  describe("open positions", () => {
    it("single buy with stop reports weighted avgCost, avgStop, risk", () => {
      const result = runLifoEngine(
        [tx({ action: "BUY", date: "2026-01-10", shares: 100, amount: 50, stop_loss: 48 })],
        50,
        100,
      );
      expect(result.avgCost).toBe(50);
      expect(result.avgStop).toBe(48);
      expect(result.risk).toBe(200); // 100 * (50 - 48)
      expect(result.realizedBank).toBe(0);
    });

    it("scale-in weights avgCost and avgStop correctly", () => {
      const result = runLifoEngine(
        [
          tx({ action: "BUY", date: "2026-01-10", shares: 100, amount: 50, stop_loss: 48 }),
          tx({ action: "BUY", date: "2026-01-15", shares: 100, amount: 60, stop_loss: 58 }),
        ],
        50,
        200,
      );
      expect(result.avgCost).toBe(55); // (100*50 + 100*60) / 200
      expect(result.avgStop).toBe(53); // (100*48 + 100*58) / 200
      expect(result.risk).toBe(400); // 200 * (55 - 53)
    });

    it("falls back to price when stop_loss is 0", () => {
      const result = runLifoEngine(
        [tx({ action: "BUY", date: "2026-01-10", shares: 100, amount: 50, stop_loss: 0 })],
        50,
        100,
      );
      // stop defaults to price → avgStop == avgCost → risk == 0
      expect(result.avgStop).toBe(50);
      expect(result.avgCost).toBe(50);
      expect(result.risk).toBe(0);
    });

    it("falls back to summaryEntry when amount is 0", () => {
      const result = runLifoEngine(
        [tx({ action: "BUY", date: "2026-01-10", shares: 100, amount: 0, stop_loss: 48 })],
        50,
        100,
      );
      expect(result.avgCost).toBe(50); // Fallback to summaryEntry
      expect(result.avgStop).toBe(48);
    });
  });

  describe("LIFO matching on sells", () => {
    it("sell consumes newest lot first", () => {
      const result = runLifoEngine(
        [
          tx({ action: "BUY", date: "2026-01-10", shares: 100, amount: 50, stop_loss: 48 }),
          tx({ action: "BUY", date: "2026-01-15", shares: 50, amount: 55, stop_loss: 53 }),
          tx({ action: "SELL", date: "2026-01-20", shares: 50, amount: 60 }),
        ],
        50,
        100,
      );
      // Sell 50@60 eats the newer 50@55 lot → realized = 50*(60-55) = 250
      expect(result.realizedBank).toBe(250);
      // Remaining: just the 100@50 lot
      expect(result.avgCost).toBe(50);
      expect(result.avgStop).toBe(48);
    });

    it("sell spans two lots, popping the newest entirely", () => {
      const result = runLifoEngine(
        [
          tx({ action: "BUY", date: "2026-01-10", shares: 100, amount: 50, stop_loss: 48 }),
          tx({ action: "BUY", date: "2026-01-15", shares: 50, amount: 55, stop_loss: 53 }),
          tx({ action: "SELL", date: "2026-01-20", shares: 80, amount: 60 }),
        ],
        50,
        100,
      );
      // 50@55 fully consumed (50 * 5 = 250) + 30 from 100@50 (30 * 10 = 300) → 550
      expect(result.realizedBank).toBe(550);
      // Remaining: 70 shares @ 50 stop 48
      expect(result.avgCost).toBe(50);
      expect(result.avgStop).toBe(48);
      expect(result.risk).toBe(140); // 70 * (50 - 48)
    });

    it("losing trade produces negative realizedBank", () => {
      const result = runLifoEngine(
        [
          tx({ action: "BUY", date: "2026-01-10", shares: 100, amount: 50, stop_loss: 48 }),
          tx({ action: "SELL", date: "2026-01-20", shares: 100, amount: 45 }),
        ],
        50,
        100,
      );
      expect(result.realizedBank).toBe(-500);
      // Fully closed, no open inventory
      expect(result.risk).toBe(0);
    });

    it("full exit empties inventory and risk goes to zero", () => {
      const result = runLifoEngine(
        [
          tx({ action: "BUY", date: "2026-01-10", shares: 100, amount: 50, stop_loss: 48 }),
          tx({ action: "SELL", date: "2026-01-20", shares: 100, amount: 60 }),
        ],
        50,
        100,
      );
      expect(result.realizedBank).toBe(1000);
      expect(result.risk).toBe(0);
      // No open inventory → falls back to summaryEntry for avgCost
      expect(result.avgCost).toBe(50);
    });
  });

  describe("ordering", () => {
    it("sorts by date ascending before processing", () => {
      const result = runLifoEngine(
        [
          tx({ action: "SELL", date: "2026-01-20", shares: 50, amount: 60 }),
          tx({ action: "BUY", date: "2026-01-10", shares: 100, amount: 50, stop_loss: 48 }),
          tx({ action: "BUY", date: "2026-01-15", shares: 50, amount: 55, stop_loss: 53 }),
        ],
        50,
        100,
      );
      // Same result as the LIFO ordering test above
      expect(result.realizedBank).toBe(250);
      expect(result.avgCost).toBe(50);
    });

    it("on same date, buys process before sells", () => {
      const result = runLifoEngine(
        [
          tx({ action: "SELL", date: "2026-01-10", shares: 50, amount: 60 }),
          tx({ action: "BUY", date: "2026-01-10", shares: 100, amount: 50, stop_loss: 48 }),
        ],
        50,
        100,
      );
      // Buy must land in inventory first so the sell has something to match
      expect(result.realizedBank).toBe(500); // 50 * (60 - 50)
      expect(result.avgCost).toBe(50);
    });
  });

  describe("projectedPl (floor)", () => {
    it("is the sum of (stop - entry) across open lots plus realizedBank", () => {
      const result = runLifoEngine(
        [tx({ action: "BUY", date: "2026-01-10", shares: 100, amount: 50, stop_loss: 48 })],
        50,
        100,
      );
      // (48 - 50) * 100 + 0 = -200 (worst-case floor if stop is hit today)
      expect(result.projectedPl).toBe(-200);
    });

    it("includes realizedBank from prior sells", () => {
      const result = runLifoEngine(
        [
          tx({ action: "BUY", date: "2026-01-10", shares: 100, amount: 50, stop_loss: 48 }),
          tx({ action: "BUY", date: "2026-01-15", shares: 50, amount: 55, stop_loss: 53 }),
          tx({ action: "SELL", date: "2026-01-20", shares: 50, amount: 60 }),
        ],
        50,
        100,
      );
      // After sell: remaining 100 @ 50, stop 48 → inv floor = (48-50)*100 = -200
      // Realized = 250 → projectedPl = -200 + 250 = 50
      expect(result.projectedPl).toBe(50);
    });
  });
});
