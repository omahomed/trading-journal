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
  // B1 (first BUY) fill price — the cost-basis anchor for the campaign.
  // Distinct from avg_entry which blends across every fill after scale-
  // ins. SR15 nudge uses b1_entry_price × 1.10 as the +10% profit-lock
  // target so that add-ons don't silently walk the target higher.
  // Optional so pre-062 test fixtures don't have to spell it out (same
  // discipline as broker_stop_price + is_declared_sr8).
  b1_entry_price?: number | null;
  // Persistent peak B1 return ever observed for this campaign (migration
  // 036). Auto-promoted on observation, never auto-demoted. Sell Rule
  // tier derives from max(b1_return_pct, b1_max_return_pct) — see
  // computeEnrichedPositions. NULL pre-backfill; falls back to current.
  b1_max_return_pct: number | null;
  // Migration 055 — physical broker stop price parked at −0.75× ATR21 from
  // B1 fill. Presence promotes tier from SR1 → SR14 in the <10% B1-return
  // window; null means single-stop campaign (classic SR1). Surfaced so
  // tooltips + edit modals + right-click quick-tag can render/mutate it.
  // Optional so pre-migration test fixtures don't have to spell it out;
  // matches the same discipline as mae_pct / sr8_activation_* fields.
  broker_stop_price?: number | null;
  sell_rule_tier: SellRuleTier | null;
  // Migration 046 — excursion metrics passed through from the summary
  // row. Optional + nullable so pre-migration fixtures / legacy DBs
  // don't have to spell them out.
  mae_pct?: number | null;
  mfe_pct?: number | null;
  atr21_entry_pct?: number | null;
  max_retrace_pct?: number | null;
  // Migration 048 — SR8 activation anchor. Trim targets in
  // sr8-trim-calc.ts prefer these over live NAV when present. Null
  // for campaigns pre-dating backfill or below +50% cushion.
  sr8_activation_date?: string | null;
  sr8_activation_nlv?: number | null;
  sr8_core_shares?: number | null;
  // Migration 062 — user-declared SR8 flag. FALSE by default; TRUE only
  // when the user explicitly promotes a cushion-qualified campaign via
  // the ACS right-click menu. Splits SR7 (qualified but undeclared)
  // from SR8 (declared). Optional so pre-062 test fixtures don't have
  // to spell it out.
  is_declared_sr8?: boolean;
  // Migration 064 — SR12 Ratcheting Profit Floor (MCP). Persisted
  // floor as a % of B1 entry. Ratcheted up on every new peak by the
  // b1_reconcile loop; never moves down. Present when the campaign
  // has ever crossed +50% peak (persistence beats current-peak's
  // "am I still up 50" question). NULL = never armed.
  //
  // DEPRECATED post-migration-065 — superseded by peak_total_pl as the
  // SR12 anchor. Left on the row for a graceful cutover; no consumer
  // reads it after this migration.
  sr12_floor_pct?: number | null;
  // Migration 065 — the max total P&L this campaign ever showed
  // (realized_bank + shares × (day_high − avg_cost) using end-of-day
  // state per bar). Backfilled by scripts/backfill_peak_total_pl.py;
  // ratcheted forward daily by b1_reconcile. Frontend derives the
  // SR12 target broker stop as avg_entry + (peak_total_pl/2 −
  // realized_pl) / shares — the price at which firing locks in
  // exactly half of the peak observed. NULL = not yet backfilled
  // or never cushion-qualified.
  peak_total_pl?: number | null;
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
    // Migration 055 — broker_stop_price is the SR14 two-stop flag.
    // Presence of a positive value promotes tier from SR1 → SR14 while
    // B1 return < 10%. Backend returns the column as-is; parseFloat is
    // enough since NULL becomes NaN which classifier reads as "no
    // broker stop set."
    const brokerStopRaw = (trade as any).broker_stop_price;
    const brokerStopPrice = brokerStopRaw != null
      ? parseFloat(String(brokerStopRaw))
      : null;
    // Migration 062 — is_declared_sr8 splits SR8 (declared) from SR7
    // (qualified but undeclared). Passing the flag to the classifier
    // keeps that split as the single source of truth; ACS + Risk
    // Manager both read the same tier code downstream.
    const isDeclaredSr8Raw = (trade as any).is_declared_sr8;
    const isDeclaredSr8 = isDeclaredSr8Raw === true || isDeclaredSr8Raw === "true";
    // Post-migration-063: brokerStopPrice is no longer read by the
    // classifier (SR14 retired). Retained on the row as a chip signal.
    // The 2nd arg is passed as null so future readers of the call site
    // aren't misled about whether it affects the tier.
    const sellRuleTier = classifySellRuleTier(
      effectiveMax, null, isDeclaredSr8,
    );

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
      b1_entry_price: b1EntryPrice,
      b1_max_return_pct: b1MaxStored,
      // Migration 055 — surfaced for tooltips + edit UI. NaN normalized
      // to null so the type stays "number | null" not "number | NaN".
      broker_stop_price: (brokerStopPrice != null && Number.isFinite(brokerStopPrice))
        ? brokerStopPrice
        : null,
      sell_rule_tier: sellRuleTier,
      mae_pct:          _passThroughNum((trade as any).mae_pct),
      mfe_pct:          _passThroughNum((trade as any).mfe_pct),
      atr21_entry_pct:  _passThroughNum((trade as any).atr21_entry_pct),
      max_retrace_pct:  _passThroughNum((trade as any).max_retrace_pct),
      // Migration 048 anchor. sr8_activation_date is a string (YYYY-MM-DD)
      // from the DB; pass through untouched. Values only populated for
      // SR8-tier campaigns after backfill / live activation.
      sr8_activation_date: (trade as any).sr8_activation_date ?? null,
      sr8_activation_nlv:  _passThroughNum((trade as any).sr8_activation_nlv),
      sr8_core_shares:     _passThroughNum((trade as any).sr8_core_shares),
      is_declared_sr8: isDeclaredSr8,
      // Migration 064 — persisted SR12 floor pct (DEPRECATED post-065).
      sr12_floor_pct: _passThroughNum((trade as any).sr12_floor_pct),
      // Migration 065 — peak_total_pl (new SR12 MCP anchor). Read the
      // snake_case name emitted by _normalize_trades — see the COL_MAP
      // entry `Peak_Total_Pl` → `peak_total_pl` in api/main.py.
      peak_total_pl: _passThroughNum((trade as any).peak_total_pl),
    };
  });
}

// psycopg2 sometimes deserializes NUMERIC as strings; normalize once
// here so the EnrichedPosition type stays strictly numeric | null.
function _passThroughNum(v: unknown): number | null {
  if (v == null || v === "") return null;
  const n = typeof v === "number" ? v : parseFloat(String(v));
  return Number.isFinite(n) ? n : null;
}
