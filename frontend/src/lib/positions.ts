"use client";

import type { TradePosition, TradeDetail } from "./api";
import { runLifoEngine } from "./lifo";
import { parseOptionTicker } from "./options";
import { classifySellRuleTier, type SellRuleTier } from "./sell-rule";

export interface EnrichedPosition {
  trade_id: string;
  ticker: string;
  shares: number;
  avg_entry: number;
  total_cost: number;
  realized_pl: number;
  rule: string;
  buy_notes: string;
  risk_budget: number;
  open_date: string;
  days_held: number;
  avg_stop: number;
  // Legacy non-negative LIFO risk (used by Risk Monitor's budget alert).
  risk_dollars: number;
  // (avg_stop − avg_entry) × shares × multiplier — multiplier-correct for
  // options. Signed: negative = at risk, zero = free roll, positive = stop
  // locks in profit.
  signed_risk: number;
  // signed_risk / equity × 100 — same sign convention.
  risk_pct: number;
  current_price: number;
  current_value: number;
  unrealized_pl: number;
  overall_pl: number;
  return_pct: number;
  pos_size_pct: number;
  is_option: boolean;
  multiplier: number;
  pyramid_pct: number;
  risk_status: "Free Roll" | "At Risk";
  projected_pl: number;
  // projected_pl / equity × 100. Same total-exposure shape as projected_pl,
  // bound to the Risk % column so it tracks realized losses on closed lots.
  projected_pct: number;
  realized_bank: number;
  expiration: Date | null;
  manual_price: number | null;
  grade: number | null;
  strategy: string | null;
  // B1 (first BUY) return % from its entry to the current price. Drives
  // sell_rule_tier classification; surfaced here so tooltips/diagnostics
  // can show the raw % alongside the tier badge if needed.
  b1_return_pct: number | null;
  // Persistent peak B1 return ever observed for this campaign (migration
  // 036). Auto-promoted on observation, never auto-demoted. Sell Rule
  // tier derives from max(b1_return_pct, b1_max_return_pct) — see
  // computeEnrichedPositions. NULL pre-backfill; falls back to current.
  b1_max_return_pct: number | null;
  sell_rule_tier: SellRuleTier | null;
}

export function computeEnrichedPositions(
  openTrades: TradePosition[],
  allDetails: TradeDetail[],
  equity: number,
  livePrices: Record<string, number> = {},
): EnrichedPosition[] {
  const now = new Date();

  return openTrades.map(trade => {
    const tradeDetails = allDetails.filter(d => d.trade_id === trade.trade_id);
    const ticker = trade.ticker || "";

    // Migration 016: instrument_type + multiplier are the source of truth.
    // Fallback to (isOption ? 100 : 1) only if the row pre-dates the backfill.
    const isOption = String((trade as any).instrument_type || "").toUpperCase() === "OPTION";
    const multRaw = parseFloat(String((trade as any).multiplier || 0));
    const multiplier = multRaw > 0 ? multRaw : (isOption ? 100 : 1);

    const shares = trade.shares || 0;
    const summaryEntry = trade.avg_entry || 0;
    const lifo = runLifoEngine(tradeDetails, summaryEntry, shares, multiplier);

    const firstDate = tradeDetails.length > 0
      ? new Date(tradeDetails[0].date)
      : new Date(trade.open_date);
    const daysHeld = Math.max(1, Math.floor((now.getTime() - firstDate.getTime()) / 86_400_000));

    const currentPrice = livePrices[ticker] || summaryEntry;
    const avgEntry = lifo.avgCost;
    const avgStop = lifo.avgStop;

    const currentValue = shares * currentPrice * multiplier;
    const unrealizedPl = (currentPrice - avgEntry) * shares * multiplier;
    const overallPl = unrealizedPl + lifo.realizedBank;
    const returnPct = avgEntry > 0 ? ((currentPrice - avgEntry) / avgEntry) * 100 : 0;
    const posSizePct = equity > 0 ? (currentValue / equity) * 100 : 0;

    // Signed risk — multiplier-correct. The legacy LIFO `risk` field omits
    // the contract multiplier, so option Risk $ values were understated by
    // 100×. We compute the new column directly here. avgStop=0 means no
    // stop has been entered; treat that as zero risk to match the historic
    // Free Roll behavior of the engine.
    const stopForRisk = avgStop > 0 ? avgStop : avgEntry;
    const signedRisk = (stopForRisk - avgEntry) * shares * multiplier;
    const riskPct = equity > 0 ? (signedRisk / equity) * 100 : 0;

    const riskBudget = parseFloat(String(trade.risk_budget || 0));

    // Pyramid: last LIFO lot's return %. Walk the buy/sell tape, LIFO-match
    // sells, and look at what the most recent open lot is up.
    let pyramidPct = 0;
    if (tradeDetails.length > 0 && currentPrice > 0) {
      const sortedTx = [...tradeDetails].sort((a, b) => {
        const da = String(a.date || "");
        const db = String(b.date || "");
        if (da !== db) return da.localeCompare(db);
        const aR = String(a.action).toUpperCase() === "BUY" ? 0 : 1;
        const bR = String(b.action).toUpperCase() === "BUY" ? 0 : 1;
        return aR - bR;
      });
      const inv: { qty: number; price: number }[] = [];
      for (const tx of sortedTx) {
        const action = String(tx.action || "").toUpperCase();
        const txShares = Math.abs(parseFloat(String(tx.shares || 0)));
        if (action === "BUY") {
          let price = parseFloat(String(tx.amount || 0));
          if (price === 0) price = summaryEntry;
          inv.push({ qty: txShares, price });
        } else if (action === "SELL") {
          let toSell = txShares;
          while (toSell > 0 && inv.length > 0) {
            const last = inv[inv.length - 1];
            const take = Math.min(toSell, last.qty);
            last.qty -= take;
            toSell -= take;
            if (last.qty < 0.00001) inv.pop();
          }
        }
      }
      if (inv.length > 0) {
        const lastLotPrice = inv[inv.length - 1].price;
        if (lastLotPrice > 0) {
          pyramidPct = ((currentPrice - lastLotPrice) / lastLotPrice) * 100;
        }
      }
    }

    const riskStatus: "Free Roll" | "At Risk" = signedRisk >= 0 ? "Free Roll" : "At Risk";
    const expiration = isOption ? (parseOptionTicker(ticker)?.exp ?? null) : null;

    // B1 (first BUY) return % — backend supplies b1_entry_price via a
    // correlated subquery on trades_details. Null when the campaign has
    // no BUY rows or the price is missing/zero (data corruption / pre-app
    // History rows). sell_rule_tier is then null and the column renders "—".
    const b1EntryRaw = parseFloat(String((trade as any).b1_entry_price ?? ""));
    const b1EntryPrice = Number.isFinite(b1EntryRaw) && b1EntryRaw > 0 ? b1EntryRaw : null;
    const b1ReturnPct = b1EntryPrice !== null && currentPrice > 0
      ? ((currentPrice - b1EntryPrice) / b1EntryPrice) * 100
      : null;

    // Persistent peak (migration 036). Sell Rule tier is fundamentally
    // state: SR8 cores don't auto-demote on a pullback, so classifying
    // from current B1 return alone mis-tiers leaders that have pulled
    // back below 50%. Use max(current, stored). Auto-promote fires
    // fire-and-forget from active-campaign.tsx when current > stored.
    const b1MaxRaw = parseFloat(String((trade as any).b1_max_return_pct ?? ""));
    const b1MaxStored = Number.isFinite(b1MaxRaw) ? b1MaxRaw : null;
    const effectiveMax = b1ReturnPct !== null && b1MaxStored !== null
      ? Math.max(b1ReturnPct, b1MaxStored)
      : (b1ReturnPct ?? b1MaxStored);
    const sellRuleTier = classifySellRuleTier(effectiveMax);

    return {
      trade_id: trade.trade_id,
      ticker,
      shares,
      avg_entry: avgEntry,
      total_cost: parseFloat(String(trade.total_cost || 0)),
      realized_pl: parseFloat(String(trade.realized_pl || 0)),
      rule: trade.rule || "",
      buy_notes: trade.buy_notes || "",
      risk_budget: riskBudget,
      open_date: trade.open_date || "",
      days_held: daysHeld,
      avg_stop: avgStop,
      // Non-negative magnitude of at-risk dollars, multiplier-correct.
      // Mirrors what legacy callers expect from risk_dollars (≥ 0, equals
      // |signed_risk| when at risk, 0 when free roll) but no longer
      // understates option exposure by 100× — the LIFO engine's lifo.risk
      // value is multiplier-blind and must not leak into v2 fields.
      risk_dollars: Math.max(0, -signedRisk),
      signed_risk: signedRisk,
      risk_pct: riskPct,
      current_price: currentPrice,
      current_value: currentValue,
      unrealized_pl: unrealizedPl,
      overall_pl: overallPl,
      return_pct: returnPct,
      pos_size_pct: posSizePct,
      is_option: isOption,
      multiplier,
      pyramid_pct: pyramidPct,
      risk_status: riskStatus,
      projected_pl: lifo.projectedPl,
      projected_pct: equity > 0 ? (lifo.projectedPl / equity) * 100 : 0,
      realized_bank: lifo.realizedBank,
      expiration,
      manual_price: (() => {
        const raw = (trade as any).manual_price;
        if (raw === null || raw === undefined || raw === "") return null;
        const n = parseFloat(String(raw));
        return isFinite(n) && n > 0 ? n : null;
      })(),
      grade: typeof (trade as any).grade === "number" ? (trade as any).grade : null,
      strategy: trade.strategy ?? null,
      b1_return_pct: b1ReturnPct,
      b1_max_return_pct: b1MaxStored,
      sell_rule_tier: sellRuleTier,
    };
  });
}
