import { describe, test, expect } from "vitest";
import { walkLedger } from "./campaign-detail-walk";
import type { TradeDetail } from "./api";

const det = (overrides: Partial<TradeDetail> & { detail_id: number; trade_id: string; action: string; date: string; shares: number }) =>
  ({
    ticker: "AAA",
    amount: 100,
    rule: "",
    notes: "",
    trx_id: "",
    instrument_type: "STOCK",
    multiplier: 1,
    ...overrides,
  }) as unknown as TradeDetail;

describe("walkLedger", () => {
  test("untouched Buy lot: remaining == shares, status='Open', openLotCount=1", () => {
    const details = [
      det({ detail_id: 1, trade_id: "T1", action: "BUY", date: "2026-01-05", shares: 100, trx_id: "B1" }),
    ];
    const r = walkLedger(details);
    expect(r.perDetail.get(1)).toEqual({ remaining: 100, status: "Open" });
    expect(r.openLotCount).toBe(1);
  });

  test("partial sell: Buy status='Partial', remaining<shares; Sell status='Closed'", () => {
    // B1 100 @ $100, S1 30 → Buy remaining 70, Partial.
    const details = [
      det({ detail_id: 1, trade_id: "T1", action: "BUY", date: "2026-01-05", shares: 100, trx_id: "B1" }),
      det({ detail_id: 2, trade_id: "T1", action: "SELL", date: "2026-01-10", shares: 30, trx_id: "S1" }),
    ];
    const r = walkLedger(details);
    expect(r.perDetail.get(1)).toEqual({ remaining: 70, status: "Partial" });
    expect(r.perDetail.get(2)).toEqual({ remaining: null, status: "Closed" });
    expect(r.openLotCount).toBe(1);
  });

  test("fully closed Buy: remaining=0, status='Closed', openLotCount=0", () => {
    const details = [
      det({ detail_id: 1, trade_id: "T1", action: "BUY", date: "2026-01-05", shares: 100, trx_id: "B1" }),
      det({ detail_id: 2, trade_id: "T1", action: "SELL", date: "2026-01-10", shares: 100, trx_id: "S1" }),
    ];
    const r = walkLedger(details);
    expect(r.perDetail.get(1)).toEqual({ remaining: 0, status: "Closed" });
    expect(r.perDetail.get(2)).toEqual({ remaining: null, status: "Closed" });
    expect(r.openLotCount).toBe(0);
  });

  test("LIFO order: latest BUY consumed first", () => {
    // B1 100 @ $100 (Jan 5), A1 50 @ $110 (Feb 5), S1 50 → A1 fully closed,
    // B1 untouched (still 100).
    const details = [
      det({ detail_id: 1, trade_id: "T1", action: "BUY", date: "2026-01-05", shares: 100, trx_id: "B1" }),
      det({ detail_id: 2, trade_id: "T1", action: "BUY", date: "2026-02-05", shares: 50, trx_id: "A1" }),
      det({ detail_id: 3, trade_id: "T1", action: "SELL", date: "2026-03-05", shares: 50, trx_id: "S1" }),
    ];
    const r = walkLedger(details);
    expect(r.perDetail.get(1)).toEqual({ remaining: 100, status: "Open" });
    expect(r.perDetail.get(2)).toEqual({ remaining: 0, status: "Closed" });
    expect(r.openLotCount).toBe(1);
  });

  test("LIFO sell spans two lots: newest fully + older partially", () => {
    // B1 100 (oldest), A1 50 (newest), S1 80 → A1 consumes 50, B1 consumes 30.
    const details = [
      det({ detail_id: 1, trade_id: "T1", action: "BUY", date: "2026-01-05", shares: 100, trx_id: "B1" }),
      det({ detail_id: 2, trade_id: "T1", action: "BUY", date: "2026-02-05", shares: 50, trx_id: "A1" }),
      det({ detail_id: 3, trade_id: "T1", action: "SELL", date: "2026-03-05", shares: 80, trx_id: "S1" }),
    ];
    const r = walkLedger(details);
    expect(r.perDetail.get(1)).toEqual({ remaining: 70, status: "Partial" });
    expect(r.perDetail.get(2)).toEqual({ remaining: 0, status: "Closed" });
    expect(r.openLotCount).toBe(1);
  });

  test("LIFO matching never crosses campaigns", () => {
    // T1 has B1 100. T2 has B1 50 + S1 50. T2's S1 must NOT touch T1's B1.
    const details = [
      det({ detail_id: 1, trade_id: "T1", action: "BUY", date: "2026-01-05", shares: 100, trx_id: "B1" }),
      det({ detail_id: 2, trade_id: "T2", action: "BUY", date: "2026-01-05", shares: 50, trx_id: "B1" }),
      det({ detail_id: 3, trade_id: "T2", action: "SELL", date: "2026-01-10", shares: 50, trx_id: "S1" }),
    ];
    const r = walkLedger(details);
    expect(r.perDetail.get(1)).toEqual({ remaining: 100, status: "Open" });
    expect(r.perDetail.get(2)).toEqual({ remaining: 0, status: "Closed" });
    expect(r.openLotCount).toBe(1);
  });

  test("BUY-before-SELL tiebreak on same date", () => {
    // Same date: BUY must be processed first so the SELL can consume it.
    const details = [
      det({ detail_id: 1, trade_id: "T1", action: "SELL", date: "2026-01-05", shares: 50, trx_id: "S1" }),
      det({ detail_id: 2, trade_id: "T1", action: "BUY", date: "2026-01-05", shares: 50, trx_id: "B1" }),
    ];
    const r = walkLedger(details);
    expect(r.perDetail.get(2)).toEqual({ remaining: 0, status: "Closed" });
    expect(r.openLotCount).toBe(0);
  });

  test("empty input returns empty result", () => {
    const r = walkLedger([]);
    expect(r.perDetail.size).toBe(0);
    expect(r.openLotCount).toBe(0);
  });
});
