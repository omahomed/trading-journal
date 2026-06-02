"use client";

// Per-detail-row LIFO walker for the Campaign Detail page.
//
// Both the KPI strip (Open Lots, Unrealized P&L, Market Value) and the
// ledger table's per-row Remaining + Status cells need the same data:
// for each Buy detail row, how many shares are still open after the
// campaign's full BUY/SELL tape has been replayed LIFO.
//
// The codebase already has `runLifoEngine` in lib/lifo.ts, but it
// returns CAMPAIGN-LEVEL aggregates (avg cost, avg stop, realized
// bank) — not per-trx_id remaining. This walker complements it.
//
// The walk mirrors trade_calc._walk_inventory (backend) and
// runLifoEngine's sort/match rules (frontend) so the displayed
// remaining matches everywhere else in the app.

import type { TradeDetail } from "./api";

export interface LedgerRowInfo {
  // Buy lot: shares still open post-LIFO (0..shares).
  //   - == shares  → "Open"
  //   - 0 < x < shares → "Partial"
  //   - ≈ 0  → "Closed"
  // Sell row: not meaningful; null. The Status pill on a Sell row
  // always reads "Closed" — the sale itself is a completed event.
  remaining: number | null;
  status: "Open" | "Partial" | "Closed";
}

export interface LedgerWalkResult {
  // Keyed by trades_details.id (the PK). Stable across re-fetches.
  perDetail: Map<number, LedgerRowInfo>;
  // Count of Buy lots with remaining > 0 — fuels the "Open Lots" KPI.
  openLotCount: number;
}

const EPSILON = 0.00001;

export function walkLedger(details: TradeDetail[]): LedgerWalkResult {
  // Group by trade_id so each campaign's BUY/SELL tape is walked
  // independently. LIFO matching never crosses campaigns.
  const byTrade = new Map<string, TradeDetail[]>();
  for (const d of details) {
    const tid = String(d.trade_id || "");
    if (!tid) continue;
    if (!byTrade.has(tid)) byTrade.set(tid, []);
    byTrade.get(tid)!.push(d);
  }

  const perDetail = new Map<number, LedgerRowInfo>();
  let openLotCount = 0;

  for (const [, txns] of byTrade) {
    // Sort: date asc, BUY before SELL on same date. Identical to
    // runLifoEngine's convention so the avg-cost/avg-stop values
    // surfaced elsewhere stay consistent with the remaining we
    // compute here.
    const sorted = [...txns].sort((a, b) => {
      const da = String(a.date || "");
      const db = String(b.date || "");
      if (da !== db) return da.localeCompare(db);
      const aIsBuy = String(a.action).toUpperCase() === "BUY" ? 0 : 1;
      const bIsBuy = String(b.action).toUpperCase() === "BUY" ? 0 : 1;
      return aIsBuy - bIsBuy;
    });

    // Inventory: parallel array of buy lots, never popped — fully-
    // consumed lots stay at remaining=0 so we can still emit a
    // "Closed" status for their detail rows after the walk finishes.
    const inv: { detail_id: number; remaining: number; initialShares: number }[] = [];

    for (const tx of sorted) {
      const action = String(tx.action || "").toUpperCase();
      const shares = Math.abs(parseFloat(String(tx.shares || 0)));
      const detail_id = (tx as { detail_id?: number }).detail_id;

      if (action === "BUY") {
        inv.push({
          detail_id: detail_id ?? -1,
          remaining: shares,
          initialShares: shares,
        });
      } else if (action === "SELL") {
        let toSell = shares;
        // LIFO: walk from the newest non-empty lot backwards.
        for (let i = inv.length - 1; i >= 0 && toSell > EPSILON; i--) {
          const lot = inv[i];
          if (lot.remaining <= EPSILON) continue;
          const take = Math.min(toSell, lot.remaining);
          lot.remaining -= take;
          toSell -= take;
        }
        if (detail_id != null) {
          perDetail.set(detail_id, { remaining: null, status: "Closed" });
        }
      }
    }

    // Populate per-Buy info from final inventory state.
    for (const lot of inv) {
      if (lot.detail_id < 0) continue;
      let status: "Open" | "Partial" | "Closed";
      if (lot.remaining >= lot.initialShares - EPSILON) {
        status = "Open";
      } else if (lot.remaining < EPSILON) {
        status = "Closed";
      } else {
        status = "Partial";
      }
      perDetail.set(lot.detail_id, {
        remaining: Math.max(0, lot.remaining),
        status,
      });
      if (lot.remaining > EPSILON) openLotCount += 1;
    }
  }

  return { perDetail, openLotCount };
}
