import type { TradeDetail } from "@/lib/api";

export interface LifoResult {
  risk: number;
  avgStop: number;
  avgCost: number;
  projectedPl: number;
  realizedBank: number;
}

/**
 * LIFO engine — mirrors the backend trade_calc.compute_lifo_summary with the
 * extra stop-loss weighting the active-campaign view needs. Processes
 * buy/sell transactions in date order, LIFO matching for sells. Returns
 * initial risk (shares * (avg_entry - avg_stop)), weighted avg cost, weighted
 * avg stop, projected floor (open unrealized @ stop + realized), and realized
 * bank. Pure function — no React or DOM dependencies.
 */
export function runLifoEngine(
  tradeDetails: TradeDetail[],
  summaryEntry: number,
  summaryShares: number,
): LifoResult {
  if (tradeDetails.length === 0) {
    return { risk: 0, avgStop: summaryEntry, avgCost: summaryEntry, projectedPl: 0, realizedBank: 0 };
  }

  // Sort by date, buys before sells on same date
  const sorted = [...tradeDetails].sort((a, b) => {
    const da = String(a.date || "");
    const db = String(b.date || "");
    if (da !== db) return da.localeCompare(db);
    const aIsBuy = String(a.action).toUpperCase() === "BUY" ? 0 : 1;
    const bIsBuy = String(b.action).toUpperCase() === "BUY" ? 0 : 1;
    return aIsBuy - bIsBuy;
  });

  const inventory: { qty: number; price: number; stop: number }[] = [];
  let realizedBank = 0;

  for (const tx of sorted) {
    const action = String(tx.action || "").toUpperCase();
    const txShares = Math.abs(parseFloat(String(tx.shares || 0)));

    if (action === "BUY") {
      let price = parseFloat(String(tx.amount || 0));
      if (price === 0) price = summaryEntry;
      let stop = parseFloat(String(tx.stop_loss || 0));
      if (stop === 0) stop = price; // fallback: stop = entry
      inventory.push({ qty: txShares, price, stop });
    } else if (action === "SELL") {
      let toSell = txShares;
      const sellPrice = parseFloat(String(tx.amount || 0));
      let costBasis = 0;
      let soldQty = 0;

      while (toSell > 0 && inventory.length > 0) {
        const last = inventory[inventory.length - 1]; // LIFO
        const take = Math.min(toSell, last.qty);
        costBasis += take * last.price;
        soldQty += take;
        last.qty -= take;
        toSell -= take;
        if (last.qty < 0.00001) inventory.pop();
      }
      const revenue = soldQty * sellPrice;
      realizedBank += revenue - costBasis;
    }
  }

  // Calculate from remaining inventory
  let totalOpenShares = 0;
  let weightedCost = 0;
  let weightedStop = 0;
  let inventoryProjPl = 0;

  for (const item of inventory) {
    if (item.qty > 0) {
      totalOpenShares += item.qty;
      weightedCost += item.qty * item.price;
      weightedStop += item.qty * item.stop;
      inventoryProjPl += (item.stop - item.price) * item.qty;
    }
  }

  const avgCost = totalOpenShares > 0 ? weightedCost / totalOpenShares : summaryEntry;
  const avgLogStop = totalOpenShares > 0 ? weightedStop / totalOpenShares : 0;
  const masterStop = avgLogStop > 0 ? avgLogStop : avgCost;
  const initialRisk = Math.max(0, (avgCost - masterStop) * totalOpenShares);
  const projectedFloor = inventoryProjPl + realizedBank;

  return {
    risk: initialRisk,
    avgStop: masterStop,
    avgCost,
    projectedPl: projectedFloor,
    realizedBank,
  };
}
