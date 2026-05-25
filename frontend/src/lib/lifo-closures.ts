"use client";

import { log } from "./log";
import type { LotClosure, TradeDetail } from "./api";

// Per-row LIFO P&L shape consumed by the trade-journal Transaction
// History table (desktop) and the mobile trade-journal detail sheet.
// BUY rows accumulate realized P&L from attributed closures + exit
// price of the latest closure; SELL rows render fully-closed with no
// P&L attribution (closures attribute to BUYs, never to SELLs — same
// LIFO semantics as backend trade_calc.compute_lifo_summary).
export type LifoRow = {
  tx: TradeDetail;
  displayShares: number;
  remaining: number;
  exitPrice: number;
  realizedPl: number;
  returnPct: number;
  unrealizedPl: number;
  status: string;
  value: number;
  isSell: boolean;
};

// Walk the persisted lot_closures (migration 017) into the per-row P&L
// shape the trade-journal view consumes. Closures attribute to BUY rows
// (matching the LIFO semantics in trade_calc.compute_lifo_summary):
// each closure's realized P&L adds to its parent BUY's realizedPl, the
// BUY's remaining shares shrink by the closure's shares, and exitPrice
// tracks the most recent SELL price.
//
// Open trades with no SELLs hit this with `closures = []` and produce
// only BUY rows in their initial state; the closure loop is a no-op.
//
// Extracted from desktop trade-journal.tsx so the mobile detail sheet
// can consume the same per-row P&L attribution without duplicating the
// walker. Pure function — no React, no DOM.
export function lotClosuresToLifoRows(
  txns: TradeDetail[],
  closures: LotClosure[],
  enrichedEntry: number,
  multiplier: number,
): { rowData: LifoRow[]; realizedBank: number } {
  // Chronological sort (BUY before SELL within the same date) so rowData
  // iteration order — and the table render order — matches the backend
  // LIFO walk that produced the closures.
  const sorted = [...txns].sort((a, b) => {
    const da = String(a.date || "");
    const db = String(b.date || "");
    if (da !== db) return da.localeCompare(db);
    return String(a.action).toUpperCase() === "BUY" ? -1 : 1;
  });

  // Initial rows. BUY rows start fully-open with zero closures attributed;
  // SELL rows are fully-closed with no P&L attribution (closures attribute
  // to BUY rows in the original walk; preserve that exactly).
  const rowData: LifoRow[] = sorted.map((tx) => {
    const action = String(tx.action || "").toUpperCase();
    const txShares = Math.abs(parseFloat(String(tx.shares || 0)));
    const txAmount = parseFloat(String(tx.amount || 0));
    const txValue = txShares * txAmount * multiplier;
    if (action === "SELL") {
      return {
        tx,
        displayShares: -txShares,
        remaining: 0,
        exitPrice: 0,
        realizedPl: 0,
        returnPct: 0,
        unrealizedPl: 0,
        status: "Closed",
        value: -txValue,
        isSell: true,
      };
    }
    return {
      tx,
      displayShares: txShares,
      remaining: txShares,
      exitPrice: 0,
      realizedPl: 0,
      returnPct: 0,
      unrealizedPl: 0,
      status: "Open",
      value: txValue,
      isSell: false,
    };
  });

  // trx_id → BUY row lookup for O(1) closure attribution. Only BUY rows
  // are addressable; closures attribute to them, never to SELL rows.
  const buyRowByTrxId = new Map<string, LifoRow>();
  for (const row of rowData) {
    if (!row.isSell) {
      const trx = String((row.tx as unknown as { trx_id?: string }).trx_id || "");
      if (trx) buyRowByTrxId.set(trx, row);
    }
  }

  // Apply closures. The API returns them sorted by closed_at ASC within a
  // trade, matching the original walk's iteration order (chronological
  // SELLs). exitPrice on the BUY row is overwritten on each closure —
  // last write wins, so the chronologically-latest SELL's price ends up
  // displayed (same semantics as the original walk).
  let realizedBank = 0;
  for (const closure of closures) {
    const buyRow = buyRowByTrxId.get(closure.buy_trx_id);
    if (!buyRow) {
      // Stale or orphaned closure — shouldn't happen in well-formed data
      // (lot_closures rows are DELETE-then-INSERTed by the recompute path).
      // Logged so we notice if it ever does during the rewire's rollout.
      log.warn(
        "lifo-closures",
        "closure references unknown buy_trx_id",
        {
          buy_trx_id: closure.buy_trx_id,
          sell_trx_id: closure.sell_trx_id,
          trade_id: closure.trade_id,
        },
      );
      continue;
    }
    const closureShares = Number(closure.shares || 0);
    const sellPrice = Number(closure.sell_price || 0);
    const closureRpl = Number(closure.realized_pl || 0);

    buyRow.realizedPl += closureRpl;
    buyRow.remaining -= closureShares;
    buyRow.exitPrice = sellPrice;

    const buyRowPrice = parseFloat(String(buyRow.tx.amount || 0)) || enrichedEntry;
    buyRow.returnPct =
      buyRowPrice > 0 ? ((sellPrice - buyRowPrice) / buyRowPrice) * 100 : 0;

    if (buyRow.remaining < 0.00001) {
      buyRow.status = "Closed";
      buyRow.remaining = 0; // clamp tiny float residuals so "−0" doesn't render
    }

    realizedBank += closureRpl;
  }

  return { rowData, realizedBank };
}
