"use client";

import { useEffect, useMemo, useState } from "react";
import { Check, Lock } from "lucide-react";
import { useSearchParams } from "next/navigation";
import { api, getActivePortfolio, type TradeDetail, type TradePosition } from "@/lib/api";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";
import { usePortfolio } from "@/lib/portfolio-context";
import { SIZING_MODES, mctStateToSizingMode } from "@/lib/sizing-mode";
import { computeVolatilitySizing, type VolSizerResults, type SizingScenario } from "@/lib/vol-sizer";
import { MobileSelectSheet } from "./mobile-select-sheet";
import { MobileHoldingPicker } from "./mobile-holding-picker";

/**
 * Mobile Position Sizer — tab-switching shell + per-tab implementations.
 *
 * Tabs (mirrors desktop order):
 *   - volatility  → New Position Sizer (vol-sizer lib)
 *   - scalein     → Scale In Sizer (inline math)
 *   - pyramid     → Pyramid Sizer (inline math)
 *   - trim        → Trim Sizer (inline math + LIFO walk)
 *   - options     → Options Sizer (inline math, this PR — closes arc)
 *
 * State persists across tab switches (matches desktop pattern: a user
 * iterating on one trade across multiple lenses shouldn't lose their
 * inputs). The needsX flags below drive which input sections render
 * for the active tab — values stay in state when their input hides.
 *
 * Mount fetch: journalLatest (equity) + rallyPrefix (MCT state →
 * sizing mode default) + tradesOpen (holding picker for Scale-In /
 * Pyramid / Trim). Per-ticker debounced priceLookup auto-fills
 * entry + ATR when the user types in the Volatility ticker input;
 * picking a holding in Scale-In fires the same priceLookup inline.
 *
 * Active-portfolio switch triggers window.location.reload() via
 * usePortfolio().setActive(), so this component doesn't need its
 * own reactivity wiring.
 */

type TabKey = "volatility" | "scalein" | "pyramid" | "trim" | "options";

// Shape of a successful scale-in computation. The {error} branch is
// handled inline; this captures the resolved-add path only.
type ScaleSuccess = {
  holding: TradePosition;
  recommendedAdd: number;
  newTotal: number;
  newAvgCost: number;
  costOfAdd: number;
  totalRiskAtNew: number;
  newWeight: number;
  stop: number;
  riskPerShare: number;
  maxRiskDol: number;
  maxRisk: number;
  targetAdd: number;
  avgEntry: number;
  currShares: number;
  verdict: "success" | "partial";
  isRiskFree: boolean;
  existingRisk: number;
  newAddRisk: number;
};

type TabInputs = {
  needsHolding?: boolean;
  needsTickerInput?: boolean;
  needsMaBuffer?: boolean;
  needsAtr?: boolean;
  needsSizingMode?: boolean;
  needsTargetSize?: boolean;
};

// ── Options result shapes (consumed by OptionsResultBlock) ────────
type OptionsRiskRow = {
  label: string;
  pct: number;
  budget: number;
  contracts: number;
  totalCost: number;
  pctNlv: number;
};

type OptionsRiskResult = {
  mode: "risk";
  cpc: number;
  hardCapBudget: number;
  rows: OptionsRiskRow[];
  recContracts: number;
  recTotal: number;
  recPct: number;
  recLimiting: string;
  recBudget: number;
};

type OptionsEquivTier = {
  label: string;
  pct: number;
  positionValue: number;
  sharesEquiv: number;
  contracts: number;
  totalCost: number;
  pctNlv: number;
};

type OptionsEquivResult = {
  mode: "equivalent";
  cpc: number;
  hardCapBudget: number;
  positionTiers: OptionsEquivTier[];
  price: number;
};

type OptionsResult = OptionsRiskResult | OptionsEquivResult;

const TABS: ReadonlyArray<{ key: TabKey; label: string; icon: string }> = [
  { key: "volatility", label: "Sizer", icon: "⚖️" },
  { key: "scalein", label: "Scale In", icon: "📐" },
  { key: "pyramid", label: "Pyramid", icon: "🔺" },
  { key: "trim", label: "Trim", icon: "✂️" },
  { key: "options", label: "Options", icon: "🎰" },
];

const TAB_INPUTS: Record<TabKey, TabInputs> = {
  volatility: {
    needsTickerInput: true,
    needsMaBuffer: true,
    needsAtr: true,
    needsSizingMode: true,
    needsTargetSize: true,
  },
  scalein: {
    needsHolding: true,
    needsMaBuffer: true,
    needsSizingMode: true,
    needsTargetSize: true,
  },
  pyramid: { needsHolding: true, needsAtr: true },
  trim: { needsHolding: true, needsTargetSize: true },
  options: { needsTickerInput: true, needsSizingMode: true, needsTargetSize: true },
};

const TAB_KEYS: ReadonlySet<TabKey> = new Set(TABS.map((t) => t.key));
const isTabKey = (s: string | null | undefined): s is TabKey =>
  typeof s === "string" && TAB_KEYS.has(s as TabKey);

// Locked to match desktop's SIZE_OPTIONS (position-sizer.tsx).
const SIZE_OPTIONS = [
  { label: "Starter", pct: 2.5 },
  { label: "Half", pct: 5 },
  { label: "Standard", pct: 7.5 },
  { label: "Full", pct: 10 },
  { label: "Overweight", pct: 12.5 },
  { label: "Core", pct: 15 },
  { label: "Core+", pct: 17.5 },
  { label: "Max", pct: 20 },
] as const;

const DEFAULT_SIZE_INDEX = 3; // Full (10%) — matches Phase 1 anchor

// Pyramid cushion-tier mapping — distinct from MCT-driven SIZING_MODES.
// Keyed off the position's unrealized gain (cushionPct), not market state.
// Locked to desktop's inline values in pyramidResults useMemo (L437-484).
type PyramidTier = { name: string; tolPct: number; atrMult: number };
const PYRAMID_TIER_3: PyramidTier = { name: "Tier 3 (Defense)", tolPct: 0.5, atrMult: 1.0 };
const PYRAMID_TIER_2: PyramidTier = { name: "Tier 2 (Moderate)", tolPct: 0.65, atrMult: 1.5 };
const PYRAMID_TIER_1: PyramidTier = { name: "Tier 1 (High Cushion)", tolPct: 1.0, atrMult: 2.0 };

function pyramidTierForCushion(cushionPct: number): PyramidTier {
  if (cushionPct >= 20) return PYRAMID_TIER_1;
  if (cushionPct >= 5) return PYRAMID_TIER_2;
  return PYRAMID_TIER_3;
}

const PYRAMID_HARD_CAP_PCT = 20;
const DEFAULT_PYRAMID_RULES = { trigger_pct: 5, alloc_pct: 20 };

// LIFO inventory walker — verbatim from desktop's buildLIFOInventory
// (position-sizer.tsx:107-137). Filter to one trade, chronological sort
// with BUY-before-SELL tiebreak, SELLs pop from end. Result: array of
// surviving BUY lots in chronological order; inventory[last] is the
// most recent BUY not fully consumed.
type InventoryLot = { qty: number; price: number };

function buildLIFOInventory(
  details: TradeDetail[],
  tradeId: string,
  fallbackAvg: number,
): InventoryLot[] {
  const trxs = details
    .filter((d) => d.trade_id === tradeId)
    .sort((a, b) => {
      const dateCompare = (a.date || "").localeCompare(b.date || "");
      if (dateCompare !== 0) return dateCompare;
      return (
        (a.action?.toUpperCase() === "BUY" ? 0 : 1) -
        (b.action?.toUpperCase() === "BUY" ? 0 : 1)
      );
    });

  const inventory: InventoryLot[] = [];
  for (const tx of trxs) {
    const action = (tx.action || "").toUpperCase();
    const shares = Math.abs(tx.shares || 0);
    let price = tx.amount || 0;

    if (action === "BUY") {
      if (price === 0) price = fallbackAvg;
      inventory.push({ qty: shares, price });
    } else if (action === "SELL") {
      let sellQty = shares;
      while (sellQty > 0 && inventory.length > 0) {
        const last = inventory[inventory.length - 1];
        const take = Math.min(sellQty, last.qty);
        last.qty -= take;
        sellQty -= take;
        if (last.qty < 0.00001) inventory.pop();
      }
    }
  }
  return inventory;
}

function lifoAvgCost(inventory: InventoryLot[]): number {
  const totalShares = inventory.reduce((s, l) => s + l.qty, 0);
  const totalCost = inventory.reduce((s, l) => s + l.qty * l.price, 0);
  return totalShares > 0 ? totalCost / totalShares : 0;
}

export function MobilePositionSizer() {
  const { activePortfolio } = usePortfolio();
  const searchParams = useSearchParams();

  // Initial tab from ?tab= URL param, consumed once on mount.
  const initialTabRaw = searchParams?.get("tab") ?? null;
  const initialTab: TabKey = isTabKey(initialTabRaw) ? initialTabRaw : "volatility";

  // Inputs (user-editable, shared across tabs that consume them)
  const [activeTab, setActiveTab] = useState<TabKey>(initialTab);
  const [ticker, setTicker] = useState("");
  const [entryPrice, setEntryPrice] = useState("");
  const [maLevel, setMaLevel] = useState("");
  const [buffer, setBuffer] = useState("1.00");
  const [sizingMode, setSizingMode] = useState<0 | 1 | 2>(1); // overwritten on mount
  const [sizingModeManual, setSizingModeManual] = useState(false);
  const [sizeIdx, setSizeIdx] = useState<number>(DEFAULT_SIZE_INDEX);
  const [selectedTradeId, setSelectedTradeId] = useState<string | null>(null);

  // Options-only state. optMode toggles between "risk" (recommend N
  // contracts from sizing mode + tier table) and "equivalent" (translate
  // stock-position size into contract count). costPerContract is the
  // per-share option premium ($1.00 default → $100/contract).
  const [optMode, setOptMode] = useState<"risk" | "equivalent">("risk");
  const [costPerContract, setCostPerContract] = useState("1.00");

  // Fetched / lifecycle
  const [equity, setEquity] = useState<number | null>(null);
  const [atrPct, setAtrPct] = useState<number | null>(null);
  const [holdings, setHoldings] = useState<TradePosition[]>([]);
  const [allDetails, setAllDetails] = useState<TradeDetail[]>([]);
  const [pyramidRules, setPyramidRules] = useState<{ trigger_pct: number; alloc_pct: number }>(
    DEFAULT_PYRAMID_RULES,
  );
  const [loading, setLoading] = useState(true);
  const [priceError, setPriceError] = useState<string | null>(null);
  const [priceLoading, setPriceLoading] = useState(false);

  const flags = TAB_INPUTS[activeTab];

  // Mount fetch — equity + MCT state + open holdings + LIFO details +
  // pyramid config. tradesOpen drives the Scale-In / Pyramid / Trim
  // holding pickers. tradesOpenDetails feeds Pyramid's lastBuy peek +
  // Trim's LIFO P&L walk (PR3). pyramid_rules is Pyramid-only — single
  // global config, fetched once on mount; defaults kick in on catch.
  useEffect(() => {
    let cancelled = false;
    Promise.all([
      api.journalLatest(getActivePortfolio()).catch((err) => {
        log.error("mobile-position-sizer", "journalLatest fetch failed", err);
        return null;
      }),
      api.rallyPrefix().catch((err) => {
        log.error("mobile-position-sizer", "rallyPrefix fetch failed", err);
        return null;
      }),
      api.tradesOpen(getActivePortfolio()).catch((err) => {
        log.error("mobile-position-sizer", "tradesOpen fetch failed", err);
        return [];
      }),
      api.tradesOpenDetails(getActivePortfolio()).catch((err) => {
        log.error("mobile-position-sizer", "tradesOpenDetails fetch failed", err);
        return { details: [], lot_closures: [] };
      }),
      api.config("pyramid_rules").catch((err) => {
        log.error("mobile-position-sizer", "config pyramid_rules fetch failed", err);
        return { key: "pyramid_rules", value: DEFAULT_PYRAMID_RULES };
      }),
    ]).then(([j, rally, open, details, pyrCfg]) => {
      if (cancelled) return;
      const endNlv = j ? parseFloat(String((j as { end_nlv?: number | string }).end_nlv ?? 0)) : 0;
      setEquity(Number.isFinite(endNlv) && endNlv > 0 ? endNlv : null);
      const stateStr = (rally as { state?: string } | null)?.state ?? null;
      setSizingMode((prev) => (sizingModeManual ? prev : mctStateToSizingMode(stateStr)));
      setHoldings(Array.isArray(open) ? (open as TradePosition[]) : []);
      const detailsArr = (details as { details?: TradeDetail[] } | null)?.details;
      setAllDetails(Array.isArray(detailsArr) ? detailsArr : []);
      const cfgVal = (pyrCfg as { value?: { trigger_pct?: number; alloc_pct?: number } } | null)?.value;
      if (cfgVal && typeof cfgVal.trigger_pct === "number" && typeof cfgVal.alloc_pct === "number") {
        setPyramidRules({ trigger_pct: cfgVal.trigger_pct, alloc_pct: cfgVal.alloc_pct });
      }
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Volatility ticker → priceLookup, debounced 600ms. Scale-In picks
  // its price up via the holding-picker onSelect handler below
  // (matches desktop's L692-701 inline lookup on holding pick).
  useEffect(() => {
    if (activeTab !== "volatility") return;
    const t = ticker.trim();
    if (!t) {
      setPriceError(null);
      setPriceLoading(false);
      return;
    }
    setPriceLoading(true);
    const timeout = setTimeout(() => {
      api
        .priceLookup(t)
        .then((data) => {
          if (data && typeof data.price === "number") {
            setEntryPrice(String(data.price));
            if (typeof data.atr_pct === "number") setAtrPct(data.atr_pct);
            setPriceError(null);
          }
        })
        .catch((err) => {
          log.debug("mobile-position-sizer", "priceLookup failed", err);
          setPriceError("Couldn't fetch price");
        })
        .finally(() => setPriceLoading(false));
    }, 600);
    return () => clearTimeout(timeout);
  }, [ticker, activeTab]);

  // Holding selection → priceLookup auto-fill (mirrors desktop L692-701).
  const handleHoldingSelect = (h: TradePosition) => {
    setSelectedTradeId(h.trade_id);
    setPriceLoading(true);
    api
      .priceLookup(h.ticker)
      .then((data) => {
        if (data && typeof data.price === "number") {
          setEntryPrice(String(data.price));
          if (typeof data.atr_pct === "number") setAtrPct(data.atr_pct);
          setPriceError(null);
        }
      })
      .catch((err) => {
        log.debug("mobile-position-sizer", "holding priceLookup failed", err);
        setPriceError("Couldn't fetch price");
      })
      .finally(() => setPriceLoading(false));
  };

  // Derived inputs
  const entry = parseFloat(entryPrice) || 0;
  const ma = parseFloat(maLevel) || 0;
  const buf = parseFloat(buffer) || 1;
  const atr = atrPct ?? 0;
  const eq = equity ?? 0;
  const tolPct = SIZING_MODES[sizingMode].pct;
  const targetSize = SIZE_OPTIONS[sizeIdx].pct;

  // Calculated stop banner derives from raw inputs (so it surfaces
  // before the audit lib has all its inputs).
  const calcStop = ma > 0 ? ma * (1 - buf / 100) : 0;
  const stopDistPct = entry > 0 && calcStop > 0 && calcStop < entry ? ((entry - calcStop) / entry) * 100 : 0;
  const calcAtrFraction = atr > 0 && stopDistPct > 0 ? stopDistPct / atr : null;

  // Volatility audit — delegates to the shared vol-sizer lib.
  const audit: VolSizerResults | null = useMemo(() => {
    if (activeTab !== "volatility") return null;
    if (entry <= 0 || eq <= 0 || atr <= 0 || ma <= 0) return null;
    if (calcStop >= entry) return null;
    try {
      return computeVolatilitySizing({
        equity: eq,
        entry,
        ma,
        bufferPct: buf,
        atrPct: atr,
        tolPct,
        targetSizePct: targetSize,
      });
    } catch (err) {
      log.error("mobile-position-sizer", "vol-sizer compute failed", err);
      return null;
    }
  }, [activeTab, entry, ma, buf, eq, atr, tolPct, targetSize, calcStop]);

  const rec = audit?.recommended ?? null;
  const recIsTechStop = audit?.recommendationReason === "tech_stop_safe";

  // Scale-In audit — inline math mirroring desktop scaleResults
  // (position-sizer.tsx:377-435) field-for-field. INLINE per directive;
  // shared-lib extraction is a separate concern.
  const scale = useMemo(() => {
    if (activeTab !== "scalein") return null;
    const holding = holdings.find((h) => h.trade_id === selectedTradeId);
    if (!holding) return null;
    if (eq <= 0 || entry <= 0 || ma <= 0) return null;

    const stop = ma * (1 - buf / 100);
    const newAddRiskPerShare = entry - stop;
    if (newAddRiskPerShare <= 0) {
      return {
        error: `Stop (${formatCurrency(stop)}) is at or above current price (${formatCurrency(entry)}).`,
      };
    }

    const multiplier = Number(holding.multiplier ?? 1) || 1;
    const currShares = Number(holding.shares ?? 0) || 0;
    const avgEntry = Number(holding.avg_entry ?? 0) || 0;
    const currValue = currShares * entry;

    const existingRiskPerShare = Math.max(0, avgEntry - stop);
    const existingRisk = currShares * existingRiskPerShare * multiplier;
    const isRiskFree = existingRiskPerShare === 0 && currShares > 0;

    const targetValue = eq * (targetSize / 100);
    const targetTotalShares = Math.ceil(targetValue / entry);
    const targetAdd = targetTotalShares - currShares;

    const maxRisk = SIZING_MODES[sizingMode].pct;
    const maxRiskDol = eq * (maxRisk / 100);
    const remainingBudget = maxRiskDol - existingRisk;

    if (targetAdd <= 0) {
      return {
        error: `You are already at or above the target weight! (Current: ${formatCurrency(currValue, { decimals: 0 })} vs Target: ${formatCurrency(targetValue, { decimals: 0 })})`,
      };
    }
    if (remainingBudget <= 0) {
      return {
        error: `NO ADD — Existing ${currShares} shares risk ${formatCurrency(existingRisk, { decimals: 0 })}, exceeding the ${maxRisk}% risk budget of ${formatCurrency(maxRiskDol, { decimals: 0 })}. Tighten your stop above ${formatCurrency(avgEntry)} avg cost.`,
      };
    }

    const affordableAdd = Math.floor(remainingBudget / (newAddRiskPerShare * multiplier));
    if (affordableAdd <= 0) {
      return {
        error: `NO ADD — Risk budget exhausted by existing position.`,
      };
    }

    const recommendedAdd = Math.min(targetAdd, affordableAdd);
    const newTotal = currShares + recommendedAdd;
    const newAvgCost =
      newTotal > 0 ? (currShares * avgEntry + recommendedAdd * entry) / newTotal : 0;
    const costOfAdd = recommendedAdd * entry;
    const newAddRisk = recommendedAdd * newAddRiskPerShare * multiplier;
    const totalRiskAtNew = existingRisk + newAddRisk;
    const newWeight = eq > 0 ? (newTotal * entry / eq) * 100 : 0;
    const verdict: "success" | "partial" = affordableAdd >= targetAdd ? "success" : "partial";

    return {
      holding,
      recommendedAdd,
      newTotal,
      newAvgCost,
      costOfAdd,
      totalRiskAtNew,
      newWeight,
      stop,
      riskPerShare: newAddRiskPerShare,
      maxRiskDol,
      maxRisk,
      targetAdd,
      avgEntry,
      currShares,
      verdict,
      isRiskFree,
      existingRisk,
      newAddRisk,
    };
  }, [activeTab, holdings, selectedTradeId, eq, entry, ma, buf, sizingMode, targetSize]);

  // ── Pyramid derivations (per-trade LIFO inventory + avg cost) ──
  // holdingInventory: array of surviving BUY lots for the selected
  // trade, after applying chronological SELL pops (LIFO).
  // holdingAvgCost: weighted avg of remaining lots; fallback to
  // summary's avg_entry when the walk yields zero (matches desktop
  // L295-299).
  const selectedHolding = useMemo(
    () => holdings.find((h) => h.trade_id === selectedTradeId) ?? null,
    [holdings, selectedTradeId],
  );

  const holdingInventory = useMemo(() => {
    if (!selectedHolding) return [];
    return buildLIFOInventory(
      allDetails,
      selectedHolding.trade_id,
      Number(selectedHolding.avg_entry ?? 0) || 0,
    );
  }, [allDetails, selectedHolding]);

  const holdingAvgCost = useMemo(() => {
    if (!selectedHolding) return 0;
    const avg = lifoAvgCost(holdingInventory);
    return avg > 0 ? avg : Number(selectedHolding.avg_entry ?? 0) || 0;
  }, [selectedHolding, holdingInventory]);

  // Pyramid audit — inline math mirroring desktop pyramidResults
  // (position-sizer.tsx:437-484) field-for-field. Cushion-driven
  // tiers (NOT MCT-driven SIZING_MODES) defined inline via
  // pyramidTierForCushion above.
  const pyramid = useMemo(() => {
    if (activeTab !== "pyramid") return null;
    if (!selectedHolding) return null;
    if (entry <= 0 || eq <= 0 || atr <= 0) return null;
    if (holdingInventory.length === 0) return null;

    const shares = Number(selectedHolding.shares ?? 0) || 0;
    const avgCost = holdingAvgCost;
    const lastBuy = holdingInventory[holdingInventory.length - 1];
    const lastBuyPrice = lastBuy.price;
    const lastBuyProfitPct = lastBuyPrice > 0 ? ((entry - lastBuyPrice) / lastBuyPrice) * 100 : 0;
    const cushionPct = avgCost > 0 ? ((entry - avgCost) / avgCost) * 100 : 0;

    const baseAddPct = pyramidRules.alloc_pct / 100;
    const thresholdPct = pyramidRules.trigger_pct;

    let scaleFactor: number;
    if (lastBuyProfitPct >= thresholdPct) scaleFactor = 1.0;
    else if (lastBuyProfitPct > 0) scaleFactor = lastBuyProfitPct / thresholdPct;
    else scaleFactor = 0;

    const pyramidMaxShares = Math.ceil(shares * baseAddPct * scaleFactor);

    const tier = pyramidTierForCushion(cushionPct);
    const dailyRiskBudget = eq * (tier.tolPct / 100);
    const atrRiskBudget = dailyRiskBudget * tier.atrMult;
    const atrDecimal = atr / 100;
    const maxSharesAtr = Math.floor(atrRiskBudget / (entry * atrDecimal));
    const maxSharesCap = Math.floor((eq * (PYRAMID_HARD_CAP_PCT / 100)) / entry);
    const positionCeiling = Math.min(maxSharesAtr, maxSharesCap);
    const roomToAdd = Math.max(0, positionCeiling - Math.floor(shares));

    const pyramidAllowed = Math.min(pyramidMaxShares, roomToAdd);
    const pyramidValue = pyramidAllowed * entry;
    const baseAdd = Math.floor(shares * baseAddPct);

    const newTotalAfter = Math.floor(shares) + pyramidAllowed;
    const newAvgCostAfter =
      newTotalAfter > 0
        ? (Math.floor(shares) * avgCost + pyramidAllowed * entry) / newTotalAfter
        : 0;
    const newWeightAfter = eq > 0 ? (newTotalAfter * entry) / eq * 100 : 0;
    const currentWeight = eq > 0 ? (shares * entry) / eq * 100 : 0;

    return {
      ticker: selectedHolding.ticker,
      shares,
      avgCost,
      lastBuyPrice,
      lastBuyProfitPct,
      cushionPct,
      scaleFactor,
      pyramidMaxShares,
      baseAdd,
      tier,
      positionCeiling,
      roomToAdd,
      pyramidAllowed,
      pyramidValue,
      newTotalAfter,
      newAvgCostAfter,
      newWeightAfter,
      currentWeight,
    };
  }, [activeTab, selectedHolding, entry, eq, atr, pyramidRules, holdingInventory, holdingAvgCost]);

  // Trim audit — inline math mirroring desktop trimResults
  // (position-sizer.tsx:486-531) field-for-field. Sell-side flow:
  // computes shares-to-sell to hit target weight, then walks the
  // LIFO inventory to attribute cost basis of the trimmed lots.
  // Shortfall fallback uses holdingData.avg_entry per L519 (NOT
  // the holdingAvgCost weighted-avg useMemo used by Pyramid).
  const trim = useMemo(() => {
    if (activeTab !== "trim") return null;
    if (!selectedHolding) return null;
    if (entry <= 0 || eq <= 0) return null;

    const currShares = Number(selectedHolding.shares ?? 0) || 0;
    const currVal = currShares * entry;
    const currWeight = eq > 0 ? (currVal / eq) * 100 : 0;
    const targetWeight = targetSize;

    if (targetWeight >= currWeight) {
      return {
        error: `Target (${targetWeight}%) is higher than Current (${currWeight.toFixed(1)}%). No trim needed.`,
      };
    }

    const targetVal = eq * (targetWeight / 100);
    const valueToSell = currVal - targetVal;
    const sharesToSell = Math.ceil(valueToSell / entry);
    const remaining = Math.max(0, currShares - sharesToSell);
    const actualNewWeight = eq > 0 ? (remaining * entry / eq) * 100 : 0;

    // LIFO P&L — deep-copy lots, pop newest until sharesNeeded reaches 0
    const inventory = holdingInventory.map((l) => ({ ...l }));
    let sharesNeeded = sharesToSell;
    let accumulatedCost = 0;

    while (sharesNeeded > 0 && inventory.length > 0) {
      const lastLot = inventory[inventory.length - 1];
      const take = Math.min(sharesNeeded, lastLot.qty);
      accumulatedCost += take * lastLot.price;
      lastLot.qty -= take;
      sharesNeeded -= take;
      if (lastLot.qty < 0.00001) inventory.pop();
    }
    // Shortfall fallback (matches desktop L519): use summary avg_entry,
    // NOT the weighted-avg holdingAvgCost.
    if (sharesNeeded > 0) {
      accumulatedCost += sharesNeeded * (Number(selectedHolding.avg_entry ?? 0) || 0);
    }

    const costBasisTrimmed = accumulatedCost;
    const cashGenerated = sharesToSell * entry;
    const lifoPnl = cashGenerated - costBasisTrimmed;
    const avgCostSold = sharesToSell > 0 ? costBasisTrimmed / sharesToSell : 0;

    return {
      ticker: selectedHolding.ticker,
      sharesToSell,
      remaining,
      actualNewWeight,
      targetWeight,
      currWeight,
      cashGenerated,
      costBasisTrimmed,
      lifoPnl,
      avgCostSold,
    };
  }, [activeTab, selectedHolding, entry, eq, targetSize, holdingInventory]);

  // Options audit — inline math mirroring desktop optResults
  // (position-sizer.tsx:534-576) field-for-field. Two render branches:
  //   risk      → 3 tier rows + standalone recommendation card driven by
  //               sizingMode (MCT-derived)
  //   equivalent → 8 position-size tier rows (filtered to <=20% via
  //               SIZE_OPTIONS) + success banner indexed by targetSize
  // Hard cap is a literal 5% of NLV per options trade.
  const options = useMemo((): OptionsResult | null => {
    if (activeTab !== "options") return null;
    const cpc = (parseFloat(costPerContract) || 0) * 100;
    if (cpc <= 0 || eq <= 0) return null;
    const hardCapBudget = eq * 0.05;

    if (optMode === "risk") {
      const tiers = [
        { label: "Conservative (1%)", pct: 1.0 },
        { label: "Normal (2%)", pct: 2.0 },
        { label: "Aggressive (3%)", pct: 3.0 },
      ];
      const rows = tiers.map((t) => {
        const budget = eq * (t.pct / 100);
        const contracts = Math.min(
          Math.floor(budget / cpc),
          Math.floor(hardCapBudget / cpc),
        );
        const totalCost = contracts * cpc;
        const pctNlv = eq > 0 ? (totalCost / eq) * 100 : 0;
        return { ...t, budget, contracts, totalCost, pctNlv };
      });
      const recBudget = eq * (SIZING_MODES[sizingMode].pct / 100);
      const recContracts = Math.min(
        Math.floor(recBudget / cpc),
        Math.floor(hardCapBudget / cpc),
      );
      const recTotal = recContracts * cpc;
      const recPct = eq > 0 ? (recTotal / eq) * 100 : 0;
      const recLimiting =
        recContracts === Math.floor(recBudget / cpc) ? "Risk Budget" : "Hard Cap (5%)";
      return {
        mode: "risk",
        cpc,
        hardCapBudget,
        rows,
        recContracts,
        recTotal,
        recPct,
        recLimiting,
        recBudget,
      };
    }

    // optMode === "equivalent"
    if (entry <= 0) return null;
    const positionTiers = SIZE_OPTIONS.filter((s) => s.pct <= 20).map((s) => {
      const positionValue = eq * (s.pct / 100);
      const sharesEquiv = Math.floor(positionValue / entry);
      const contracts = Math.ceil(sharesEquiv / 100);
      const totalCost = contracts * cpc;
      const pctNlv = eq > 0 ? (totalCost / eq) * 100 : 0;
      return { label: s.label, pct: s.pct, positionValue, sharesEquiv, contracts, totalCost, pctNlv };
    });
    return { mode: "equivalent", cpc, hardCapBudget, positionTiers, price: entry };
  }, [activeTab, costPerContract, eq, sizingMode, optMode, entry]);

  const equityDisplay = equity != null
    ? formatCurrency(equity, { decimals: 0 })
    : loading
      ? "…"
      : "—";

  const atrDisplay = atrPct != null ? `${atrPct.toFixed(1)}%` : "—";

  return (
    <div className="flex flex-col gap-2.5 pt-2">
      {activePortfolio && (
        <div className="text-[11px] text-m-text-dim">
          Sizing for <span className="text-m-text-muted">{activePortfolio.name}</span>
        </div>
      )}

      {/* Tab bar — horizontal scrollable, edge-to-edge bleed via -mx-5 */}
      <div
        role="tablist"
        aria-label="Sizer tabs"
        className="-mx-5 flex gap-1 overflow-x-auto border-b-[0.5px] border-m-border px-5"
      >
        {TABS.map((t) => {
          const isActive = activeTab === t.key;
          return (
            <button
              key={t.key}
              type="button"
              role="tab"
              aria-selected={isActive}
              aria-controls={`sizer-panel-${t.key}`}
              onClick={() => setActiveTab(t.key)}
              className={
                "-mb-px shrink-0 whitespace-nowrap border-b-2 px-3 py-2 text-[12px] font-medium " +
                (isActive
                  ? "border-m-accent text-m-accent"
                  : "border-transparent text-m-text-faint")
              }
            >
              <span aria-hidden="true">{t.icon}</span> {t.label}
            </button>
          );
        })}
      </div>

      {/* ── Tab: Volatility ── */}
      {activeTab === "volatility" && (
        <div id="sizer-panel-volatility" role="tabpanel" className="flex flex-col gap-2.5">
          {/* Ticker card */}
          {flags.needsTickerInput && (
            <div className="rounded-m-lg border-[0.5px] border-m-border bg-m-surface px-[18px] py-[14px]">
              <input
                type="text"
                value={ticker}
                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                placeholder="TICKER"
                inputMode="text"
                autoCapitalize="characters"
                autoCorrect="off"
                spellCheck={false}
                aria-label="Ticker symbol"
                className="w-full bg-transparent font-m-num text-[28px] font-medium tracking-[-0.02em] text-m-text placeholder:text-m-text-faint focus:outline-none"
              />
              <div className="mt-1.5 flex items-center gap-3 text-xs text-m-text-dim">
                {priceLoading ? (
                  <span className="font-m-num">Fetching price…</span>
                ) : priceError ? (
                  <span className="font-m-num text-m-warn">{priceError}</span>
                ) : (
                  <span className="font-m-num">{ticker ? "Live price + ATR" : "Type a ticker"}</span>
                )}
              </div>
            </div>
          )}

          {/* 2×2 input grid (Entry / NLV / MA / Buffer) */}
          <div className="grid grid-cols-2 gap-2">
            <NumberFieldCell
              label="Entry"
              value={entryPrice}
              onChange={setEntryPrice}
              ariaLabel="Entry price"
              placeholder="0.00"
            />
            <ReadOnlyFieldCell
              label="NLV"
              labelIcon={<Lock size={9} strokeWidth={1} className="text-m-text-dim" aria-hidden="true" />}
              value={equityDisplay}
            />
            <NumberFieldCell
              label="Key MA"
              value={maLevel}
              onChange={setMaLevel}
              ariaLabel="Key MA level"
              placeholder="0.00"
            />
            <NumberFieldCell
              label="Buffer"
              value={buffer}
              onChange={setBuffer}
              ariaLabel="Buffer percent"
              suffix="%"
              placeholder="1.00"
            />
          </div>

          {/* ATR row */}
          {flags.needsAtr && (
            <div className="flex items-baseline justify-between rounded-m-md border-[0.5px] border-m-border bg-m-surface px-[14px] py-[10px]">
              <span className="text-[11px] font-medium text-m-text-dim">ATR % (21D)</span>
              <span className="font-m-num text-xl font-medium tabular-nums tracking-[-0.01em] text-m-text">
                {atrDisplay}
              </span>
            </div>
          )}

          {/* Mode + Size pickers (shared with Scale-In, but Volatility
              renders them here to keep the Volatility layout self-
              contained). Scale-In's render block below mirrors. */}
          <div className="grid grid-cols-2 gap-2">
            <ModePickerTile
              sizingMode={sizingMode}
              onChange={(i) => {
                setSizingMode(i);
                setSizingModeManual(true);
              }}
            />
            <SizePickerTile sizeIdx={sizeIdx} onChange={setSizeIdx} />
          </div>

          {/* Calculated Stop banner */}
          {calcStop > 0 && entry > 0 && (
            <div
              data-testid="calc-stop-banner"
              className="rounded-m-md border-[0.5px] px-[14px] py-[10px] text-[12px]"
              style={{
                background: "var(--m-accent-tint)",
                borderColor: "var(--m-accent-border)",
                color: "var(--m-text)",
              }}
            >
              Calculated Stop: <strong>{formatCurrency(calcStop)}</strong> (MA {formatCurrency(ma)} − {buf.toFixed(1)}% buffer) — {stopDistPct.toFixed(1)}% below entry
              {calcAtrFraction !== null && ` · ${calcAtrFraction.toFixed(2)}× ATR`}
            </div>
          )}

          {audit?.warning?.show && (
            <div
              data-testid="vol-warning"
              className="rounded-m-md border-[0.5px] px-[14px] py-[10px] text-[12px]"
              style={{
                background: "color-mix(in oklab, var(--m-warn) 12%, transparent)",
                borderColor: "var(--m-warn-border)",
                color: "var(--m-warn)",
              }}
            >
              {audit.warning.text}
            </div>
          )}

          {/* Context grid */}
          <div className="grid grid-cols-3 gap-2">
            <MiniMetric
              label="Risk Budget"
              value={audit ? formatCurrency(audit.riskBudget, { decimals: 0 }) : "—"}
              sub={`${tolPct.toFixed(2)}%`}
            />
            <MiniMetric
              label="ATR Noise"
              value={atrPct != null ? `${atrPct.toFixed(2)}%` : "—"}
              sub={audit ? `${formatCurrency(audit.atrPerShare)}/sh` : undefined}
            />
            <MiniMetric
              label="Position Cap"
              value={audit ? `${audit.positionCapShares} shs` : "—"}
              sub={`${targetSize}% NLV`}
            />
          </div>

          {audit && (
            <ScenarioRow
              scenario={audit.techStop}
              targetSize={targetSize}
              isRecommended={!!recIsTechStop}
            />
          )}

          {audit && (
            <div className="flex flex-col gap-2">
              {audit.atrScenarios.map((s, i) => (
                <ScenarioRow
                  key={s.label}
                  scenario={s}
                  targetSize={targetSize}
                  isRecommended={!recIsTechStop && i === 1}
                />
              ))}
            </div>
          )}

          {rec && audit && (
            <div
              data-testid="verdict-card"
              className="rounded-m-xl border-[0.5px] px-5 pt-4 pb-4"
              style={{
                background: "color-mix(in oklab, var(--m-accent) 10%, var(--m-surface))",
                borderColor: "var(--m-accent-border)",
              }}
            >
              <div className="text-[11px] font-medium text-m-text-dim uppercase tracking-[0.08em]">
                Recommended
              </div>
              <div className="mt-1.5 flex items-baseline gap-2">
                <span
                  data-testid="audit-shares"
                  className="font-m-num text-[38px] font-medium tabular-nums tracking-[-0.03em] text-m-text"
                >
                  {rec.finalShares.toLocaleString()}
                </span>
                <span className="text-[15px] text-m-text-muted">
                  shares · {formatCurrency(rec.positionCost, { decimals: 0 })}
                </span>
              </div>
              <div className="mt-1 text-[11px] text-m-text-dim">
                Sized by {recIsTechStop ? "tech stop" : "1.5× ATR cushion"} · bound by{" "}
                {rec.capBinds ? `position-size tier (${targetSize}% NLV)` : `risk budget (${tolPct.toFixed(2)}%)`}
              </div>
            </div>
          )}

          {!audit && (
            <div className="rounded-m-xl border-[0.5px] border-m-border bg-m-surface px-5 pt-4 pb-3.5">
              <div className="text-[11px] font-medium text-m-text-dim uppercase tracking-[0.08em]">
                Recommended
              </div>
              <div className="mt-1.5 flex items-baseline gap-2">
                <span
                  data-testid="audit-shares"
                  className="font-m-num text-[38px] font-medium tabular-nums tracking-[-0.03em] text-m-text"
                >
                  —
                </span>
                <span className="text-[15px] text-m-text-muted">shares</span>
              </div>
              <div className="mt-1 text-[11px] text-m-text-dim">
                Enter ticker, MA, and ATR to size.
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Tab: Scale-In ── */}
      {activeTab === "scalein" && (
        <div id="sizer-panel-scalein" role="tabpanel" className="flex flex-col gap-2.5">
          <MobileHoldingPicker
            holdings={holdings}
            selectedTradeId={selectedTradeId}
            onSelect={handleHoldingSelect}
            portfolioName={activePortfolio?.name}
          />

          {/* Fetch status — surfaces the same indicator the Volatility
              ticker card uses, since holding-select fires priceLookup. */}
          {(priceLoading || priceError) && (
            <div
              className="text-[11px]"
              style={{ color: priceError ? "var(--m-warn)" : "var(--m-text-dim)" }}
            >
              {priceLoading ? "Fetching price…" : priceError}
            </div>
          )}

          {/* 2×2 input grid: Current Price / Equity / MA / Buffer */}
          <div className="grid grid-cols-2 gap-2">
            <NumberFieldCell
              label="Current Price"
              value={entryPrice}
              onChange={setEntryPrice}
              ariaLabel="Current price"
              placeholder="0.00"
            />
            <NumberFieldCell
              label="Account Equity"
              value={equity != null ? String(equity) : ""}
              onChange={(v) => {
                const n = parseFloat(v);
                setEquity(Number.isFinite(n) && n > 0 ? n : null);
              }}
              ariaLabel="Account equity"
              placeholder="0"
            />
            <NumberFieldCell
              label="Key MA"
              value={maLevel}
              onChange={setMaLevel}
              ariaLabel="Key MA level"
              placeholder="0.00"
            />
            <NumberFieldCell
              label="Buffer"
              value={buffer}
              onChange={setBuffer}
              ariaLabel="Buffer percent"
              suffix="%"
              placeholder="1.00"
            />
          </div>

          {/* Mode + Size pickers */}
          <div className="grid grid-cols-2 gap-2">
            <ModePickerTile
              sizingMode={sizingMode}
              onChange={(i) => {
                setSizingMode(i);
                setSizingModeManual(true);
              }}
            />
            <SizePickerTile sizeIdx={sizeIdx} onChange={setSizeIdx} />
          </div>

          {/* Calculated Stop banner (info) */}
          {calcStop > 0 && entry > 0 && (
            <div
              data-testid="scale-calc-stop-banner"
              className="rounded-m-md border-[0.5px] px-[14px] py-[10px] text-[12px]"
              style={{
                background: "var(--m-accent-tint)",
                borderColor: "var(--m-accent-border)",
                color: "var(--m-text)",
              }}
            >
              Calculated Stop: <strong>{formatCurrency(calcStop)}</strong> (MA {formatCurrency(ma)} − {buf.toFixed(1)}% buffer) — {stopDistPct.toFixed(1)}% below price
            </div>
          )}

          <ScaleInResultBlock result={scale} equity={eq} />
        </div>
      )}

      {/* ── Tab: Pyramid ── */}
      {activeTab === "pyramid" && (
        <div id="sizer-panel-pyramid" role="tabpanel" className="flex flex-col gap-2.5">
          <MobileHoldingPicker
            holdings={holdings}
            selectedTradeId={selectedTradeId}
            onSelect={handleHoldingSelect}
            portfolioName={activePortfolio?.name}
          />

          {(priceLoading || priceError) && (
            <div
              className="text-[11px]"
              style={{ color: priceError ? "var(--m-warn)" : "var(--m-text-dim)" }}
            >
              {priceLoading ? "Fetching price…" : priceError}
            </div>
          )}

          {/* Inputs: Current Price + Equity on row 1, ATR % spans row 2 */}
          <div className="grid grid-cols-2 gap-2">
            <NumberFieldCell
              label="Current Price"
              value={entryPrice}
              onChange={setEntryPrice}
              ariaLabel="Current price"
              placeholder="0.00"
            />
            <NumberFieldCell
              label="Account Equity"
              value={equity != null ? String(equity) : ""}
              onChange={(v) => {
                const n = parseFloat(v);
                setEquity(Number.isFinite(n) && n > 0 ? n : null);
              }}
              ariaLabel="Account equity"
              placeholder="0"
            />
          </div>
          <NumberFieldCell
            label="ATR %"
            value={atrPct != null ? String(atrPct) : ""}
            onChange={(v) => {
              const n = parseFloat(v);
              setAtrPct(Number.isFinite(n) ? n : null);
            }}
            ariaLabel="ATR percent"
            suffix="%"
            placeholder="5.0"
          />

          {/* Pyramid Rules expander (collapsible, default closed) */}
          <PyramidRulesExpander rules={pyramidRules} />

          <PyramidResultBlock result={pyramid} rules={pyramidRules} />
        </div>
      )}

      {/* ── Tab: Trim ── */}
      {activeTab === "trim" && (
        <div id="sizer-panel-trim" role="tabpanel" className="flex flex-col gap-2.5">
          <MobileHoldingPicker
            holdings={holdings}
            selectedTradeId={selectedTradeId}
            onSelect={handleHoldingSelect}
            portfolioName={activePortfolio?.name}
          />

          {(priceLoading || priceError) && (
            <div
              className="text-[11px]"
              style={{ color: priceError ? "var(--m-warn)" : "var(--m-text-dim)" }}
            >
              {priceLoading ? "Fetching price…" : priceError}
            </div>
          )}

          {/* Inputs: Current Price + Equity (auto+editable) */}
          <div className="grid grid-cols-2 gap-2">
            <NumberFieldCell
              label="Current Price"
              value={entryPrice}
              onChange={setEntryPrice}
              ariaLabel="Current price"
              placeholder="0.00"
            />
            <NumberFieldCell
              label="Account Equity"
              value={equity != null ? String(equity) : ""}
              onChange={(v) => {
                const n = parseFloat(v);
                setEquity(Number.isFinite(n) && n > 0 ? n : null);
              }}
              ariaLabel="Account equity"
              placeholder="0"
            />
          </div>

          {/* Target size picker (single tile, takes full width since
              there's no Mode picker to pair with on Trim) */}
          <SizePickerTile sizeIdx={sizeIdx} onChange={setSizeIdx} />

          <TrimResultBlock result={trim} />
        </div>
      )}

      {/* ── Tab: Options ── */}
      {activeTab === "options" && (
        <div id="sizer-panel-options" role="tabpanel" className="flex flex-col gap-2.5">
          {/* Method picker (Risk / Equivalent). Picker-tile + bottom
              sheet pattern reused from Mode/Size pickers — no segmented
              control precedent in the mobile codebase. */}
          <OptionsMethodPickerTile optMode={optMode} onChange={setOptMode} />

          {/* Cost per Contract + Account Equity (always visible) */}
          <div className="grid grid-cols-2 gap-2">
            <NumberFieldCell
              label="Cost / Contract ($)"
              value={costPerContract}
              onChange={setCostPerContract}
              ariaLabel="Cost per contract"
              placeholder="1.00"
            />
            <NumberFieldCell
              label="Account Equity"
              value={equity != null ? String(equity) : ""}
              onChange={(v) => {
                const n = parseFloat(v);
                setEquity(Number.isFinite(n) && n > 0 ? n : null);
              }}
              ariaLabel="Account equity"
              placeholder="0"
            />
          </div>

          {/* Risk mode → Mode picker drives the highlighted recommendation */}
          {optMode === "risk" && (
            <ModePickerTile
              sizingMode={sizingMode}
              onChange={(i) => {
                setSizingMode(i);
                setSizingModeManual(true);
              }}
            />
          )}

          {/* Equivalent mode → Ticker + Stock Price + Size picker */}
          {optMode === "equivalent" && (
            <>
              <div className="grid grid-cols-2 gap-2">
                <label className="block rounded-m-md border-[0.5px] border-m-border bg-m-surface px-[14px] py-[10px]">
                  <span className="mb-0.5 block text-[10px] font-medium text-m-text-dim">
                    Ticker
                  </span>
                  <input
                    type="text"
                    value={ticker}
                    onChange={(e) => setTicker(e.target.value.toUpperCase())}
                    placeholder="XYZ"
                    autoCapitalize="characters"
                    autoCorrect="off"
                    spellCheck={false}
                    aria-label="Ticker symbol"
                    className="w-full bg-transparent font-m-num text-lg font-medium tabular-nums text-m-text placeholder:text-m-text-faint focus:outline-none"
                  />
                </label>
                <NumberFieldCell
                  label="Stock Price"
                  value={entryPrice}
                  onChange={setEntryPrice}
                  ariaLabel="Stock price"
                  placeholder="0.00"
                />
              </div>
              <SizePickerTile sizeIdx={sizeIdx} onChange={setSizeIdx} />
            </>
          )}

          <OptionsResultBlock
            result={options}
            sizingMode={sizingMode}
            costPerContract={costPerContract}
            equity={eq}
            ticker={ticker}
            targetSize={targetSize}
          />
        </div>
      )}
    </div>
  );
}

// ── Scale-In result block ─────────────────────────────────────────

function ScaleInResultBlock({
  result,
  equity,
}: {
  result: { error: string } | ScaleSuccess | null;
  equity: number;
}) {
  if (result == null) {
    return (
      <div className="rounded-m-xl border-[0.5px] border-m-border bg-m-surface px-5 py-5 text-center">
        <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-m-text-dim">
          Scale Ticket
        </div>
        <div className="mt-2 text-[12px] text-m-text-dim">
          Select a holding and enter Current Price + MA to size the add.
        </div>
      </div>
    );
  }

  if ("error" in result) {
    return (
      <div
        data-testid="scale-error-banner"
        className="rounded-m-md border-[0.5px] px-[14px] py-3 text-[12px]"
        style={{
          background: "color-mix(in oklab, var(--m-down) 12%, transparent)",
          borderColor: "var(--m-down)",
          color: "var(--m-down)",
        }}
      >
        {result.error}
      </div>
    );
  }

  const r = result;
  return (
    <div className="flex flex-col gap-3">
      {/* Risk-free banner */}
      {r.isRiskFree && (
        <div
          data-testid="scale-risk-free-banner"
          className="rounded-m-md border-[0.5px] px-[14px] py-2.5 text-[12px]"
          style={{
            background: "color-mix(in oklab, var(--m-accent) 12%, var(--m-surface))",
            borderColor: "var(--m-accent-border)",
            color: "var(--m-accent)",
          }}
        >
          ✓ Position is risk-free — stop {formatCurrency(r.stop)} sits above your {formatCurrency(r.avgEntry)} avg cost. Existing shares contribute $0 to the risk budget.
        </div>
      )}

      {/* Scale Ticket — 4 stacked cards */}
      <div>
        <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.10em] text-m-text-dim">
          Scale Ticket
        </div>
        <div className="grid grid-cols-2 gap-2">
          <ResultCard
            label="ADD SHARES"
            value={`+${r.recommendedAdd}`}
            tone="up"
          />
          <ResultCard
            label="EST. COST"
            value={formatCurrency(r.costOfAdd, { decimals: 0 })}
          />
          <ResultCard
            label="NEW TOTAL"
            value={`${r.newTotal} shs`}
            sub={`${r.newWeight.toFixed(1)}% weight`}
          />
          <ResultCard
            label="NEW AVG COST"
            value={formatCurrency(r.newAvgCost)}
            sub={`from ${formatCurrency(r.avgEntry)}`}
          />
        </div>
      </div>

      {/* Risk Management — 3 stacked cards */}
      <div>
        <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.10em] text-m-text-dim">
          Risk Management
        </div>
        <div className="grid grid-cols-1 gap-2">
          <ResultCard
            label="Global Stop"
            value={formatCurrency(r.stop)}
            sub={`−${((r.riskPerShare / (r.riskPerShare + r.stop)) * 100).toFixed(1)}% from price`}
            tone="down"
            inline
          />
          <ResultCard
            label="Total Risk at New Size"
            value={formatCurrency(r.totalRiskAtNew, { decimals: 0 })}
            sub={`${equity > 0 ? ((r.totalRiskAtNew / equity) * 100).toFixed(2) : "0.00"}% of NLV`}
            tone="warn"
            inline
          />
          <ResultCard
            label="Risk Budget"
            value={formatCurrency(r.maxRiskDol, { decimals: 0 })}
            sub={`${r.maxRisk}% of equity`}
            inline
          />
        </div>
      </div>

      {/* Verdict */}
      {r.verdict === "success" ? (
        <div
          data-testid="scale-verdict-success"
          className="rounded-m-md border-[0.5px] px-[14px] py-3 text-[12px]"
          style={{
            background: "color-mix(in oklab, var(--m-accent) 10%, var(--m-surface))",
            borderColor: "var(--m-accent-border)",
            color: "var(--m-text)",
          }}
        >
          <strong>ADD {r.recommendedAdd} shares</strong> to reach {r.newWeight.toFixed(1)}% — total risk {formatCurrency(r.totalRiskAtNew, { decimals: 0 })} within {formatCurrency(r.maxRiskDol, { decimals: 0 })} budget.
        </div>
      ) : (
        <div
          data-testid="scale-verdict-partial"
          className="rounded-m-md border-[0.5px] px-[14px] py-3 text-[12px]"
          style={{
            background: "color-mix(in oklab, var(--m-warn) 12%, var(--m-surface))",
            borderColor: "var(--m-warn-border)",
            color: "var(--m-text)",
          }}
        >
          <strong>RISK LIMIT:</strong> Full target ({r.targetAdd} shares) would exceed budget. Safe add: {r.recommendedAdd} shares ({r.newWeight.toFixed(1)}% weight).
        </div>
      )}
    </div>
  );
}


// ── Pyramid rules expander ────────────────────────────────────────

type PyramidRulesShape = { trigger_pct: number; alloc_pct: number };

function PyramidRulesExpander({ rules }: { rules: PyramidRulesShape }) {
  return (
    <details className="rounded-m-md border-[0.5px] border-m-border bg-m-surface px-4 py-2.5">
      <summary
        className="cursor-pointer text-[12px] font-semibold text-m-text-muted"
        data-testid="pyramid-rules-summary"
      >
        View Pyramid Rules
      </summary>
      <div className="mt-2 pb-1 text-[12px] leading-relaxed text-m-text-dim">
        <p className="mb-1">
          <strong className="text-m-text">How it works:</strong>
        </p>
        <ol className="ml-4 flex list-decimal flex-col gap-0.5">
          <li>
            Each add is capped at <strong className="text-m-text">{rules.alloc_pct}%</strong> of your current shares
          </li>
          <li>
            Your last buy must be up <strong className="text-m-text">at least {rules.trigger_pct}%</strong> for a full-size add
          </li>
          <li>
            If last buy is up less than {rules.trigger_pct}%, the add scales proportionally:{" "}
            <code className="rounded-[4px] bg-m-surface-2 px-1 py-px font-m-num text-[11px] tabular-nums text-m-text-muted">
              (profit% / {rules.trigger_pct}%) × {rules.alloc_pct}%
            </code>
          </li>
          <li>
            If last buy is <strong className="text-m-text">flat or down</strong>, no add is allowed
          </li>
          <li>
            The add is also capped by your ATR limit and {rules.alloc_pct}% hard cap
          </li>
        </ol>
      </div>
    </details>
  );
}

// ── Pyramid result block ──────────────────────────────────────────

type PyramidSuccess = {
  ticker: string;
  shares: number;
  avgCost: number;
  lastBuyPrice: number;
  lastBuyProfitPct: number;
  cushionPct: number;
  scaleFactor: number;
  pyramidMaxShares: number;
  baseAdd: number;
  tier: { name: string; tolPct: number; atrMult: number };
  positionCeiling: number;
  roomToAdd: number;
  pyramidAllowed: number;
  pyramidValue: number;
  newTotalAfter: number;
  newAvgCostAfter: number;
  newWeightAfter: number;
  currentWeight: number;
};

function PyramidResultBlock({
  result,
  rules,
}: {
  result: PyramidSuccess | null;
  rules: PyramidRulesShape;
}) {
  if (result == null) {
    return (
      <div className="rounded-m-xl border-[0.5px] border-m-border bg-m-surface px-5 py-5 text-center">
        <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-m-text-dim">
          Pyramid Analysis
        </div>
        <div className="mt-2 text-[12px] text-m-text-dim">
          Select a holding and enter price, equity, and ATR to compute the pyramid.
        </div>
      </div>
    );
  }

  const r = result;
  const lastBuyToneIsUp = r.lastBuyProfitPct >= 0;

  return (
    <div className="flex flex-col gap-3">
      <div className="text-[13px] font-semibold text-m-text">
        Pyramid Analysis: <span className="font-m-num tabular-nums text-m-accent">{r.ticker}</span>
      </div>

      {/* Last Buy Info — 3 stacked inline cards */}
      <div>
        <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.10em] text-m-text-dim">
          Last Buy Info
        </div>
        <div className="grid grid-cols-1 gap-2">
          <ResultCard
            label="Last Buy Price"
            value={formatCurrency(r.lastBuyPrice)}
            inline
          />
          <ResultCard
            label="Last Buy P&L"
            value={`${r.lastBuyProfitPct >= 0 ? "+" : ""}${r.lastBuyProfitPct.toFixed(2)}%`}
            sub={`${formatCurrency(r.lastBuyPrice * (r.lastBuyProfitPct / 100))}/share`}
            tone={lastBuyToneIsUp ? "up" : "down"}
            inline
          />
          <ResultCard
            label="Total Cushion"
            value={`${r.cushionPct >= 0 ? "+" : ""}${r.cushionPct.toFixed(2)}%`}
            sub={`avg cost ${formatCurrency(r.avgCost)}`}
            tone="warn"
            inline
          />
        </div>
      </div>

      {/* Pyramid Calculation — 3 stacked inline cards */}
      <div>
        <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.10em] text-m-text-dim">
          Pyramid Calculation
        </div>
        <div className="grid grid-cols-1 gap-2">
          <ResultCard
            label={`Base Add (${rules.alloc_pct}%)`}
            value={`${r.baseAdd} shs`}
            sub={`${rules.alloc_pct}% of ${Math.floor(r.shares)} shares`}
            inline
          />
          <ResultCard
            label="Scale Factor"
            value={`${(r.scaleFactor * 100).toFixed(0)}%`}
            sub={`last buy up ${r.lastBuyProfitPct.toFixed(1)}% (need ${rules.trigger_pct}%)`}
            inline
          />
          <ResultCard
            label="Pyramid Max"
            value={`${r.pyramidMaxShares} shs`}
            sub="after scaling"
            inline
          />
        </div>
      </div>

      {/* Ceiling Check — 3 stacked inline cards */}
      <div>
        <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.10em] text-m-text-dim">
          Ceiling Check
        </div>
        <div className="grid grid-cols-1 gap-2">
          <ResultCard
            label="Position Ceiling"
            value={`${r.positionCeiling} shs`}
            sub={`${r.tier.name} · ${r.tier.atrMult.toFixed(1)}× ATR`}
            inline
          />
          <ResultCard
            label="Current Position"
            value={`${Math.floor(r.shares)} shs`}
            sub={`${r.currentWeight.toFixed(1)}% weight`}
            inline
          />
          <ResultCard
            label="Room to Add"
            value={`${r.roomToAdd} shs`}
            sub="before hitting ceiling"
            inline
          />
        </div>
      </div>

      {/* Verdict — 4 branches mirroring desktop L1015-1047 */}
      <PyramidVerdict result={r} />
    </div>
  );
}

function PyramidVerdict({ result }: { result: PyramidSuccess }) {
  const r = result;

  if (r.scaleFactor === 0) {
    const direction = r.lastBuyProfitPct < 0 ? "down" : "flat";
    return (
      <div
        data-testid="pyramid-verdict-error"
        className="rounded-m-md border-[0.5px] px-[14px] py-3 text-[12px]"
        style={{
          background: "color-mix(in oklab, var(--m-down) 12%, transparent)",
          borderColor: "var(--m-down)",
          color: "var(--m-down)",
        }}
      >
        <strong>NO ADD</strong> — Last buy is {direction} ({r.lastBuyProfitPct.toFixed(2)}%). Wait for it to work.
      </div>
    );
  }

  if (r.pyramidAllowed === 0 && r.pyramidMaxShares > 0) {
    return (
      <div
        data-testid="pyramid-verdict-warning"
        className="rounded-m-md border-[0.5px] px-[14px] py-3 text-[12px]"
        style={{
          background: "color-mix(in oklab, var(--m-warn) 12%, var(--m-surface))",
          borderColor: "var(--m-warn-border)",
          color: "var(--m-text)",
        }}
      >
        <strong>NO ROOM</strong> — Pyramid says {r.pyramidMaxShares} shares, but position is at ATR/cap ceiling ({r.positionCeiling} shs).
      </div>
    );
  }

  if (r.pyramidAllowed > 0) {
    const limitedBy = r.pyramidAllowed === r.pyramidMaxShares ? "Pyramid pace" : "ATR/Cap ceiling";
    return (
      <div className="flex flex-col gap-3">
        <div
          data-testid="pyramid-verdict-success"
          className="rounded-m-md border-[0.5px] px-[14px] py-3 text-[12px]"
          style={{
            background: "color-mix(in oklab, var(--m-accent) 10%, var(--m-surface))",
            borderColor: "var(--m-accent-border)",
            color: "var(--m-text)",
          }}
        >
          <strong>ADD {r.pyramidAllowed} shares</strong> ({formatCurrency(r.pyramidValue, { decimals: 0 })}) — limited by: {limitedBy}.
        </div>

        {/* 3 follow-up cards on success */}
        <div className="grid grid-cols-1 gap-2">
          <ResultCard
            label="Add Shares"
            value={`${r.pyramidAllowed} shs`}
            sub={formatCurrency(r.pyramidValue, { decimals: 0 })}
            tone="up"
            inline
          />
          <ResultCard
            label="New Total"
            value={`${r.newTotalAfter} shs`}
            sub={`${r.newWeightAfter.toFixed(1)}% weight`}
            inline
          />
          <ResultCard
            label="New Avg Cost"
            value={formatCurrency(r.newAvgCostAfter)}
            sub={`from ${formatCurrency(r.avgCost)}`}
            inline
          />
        </div>
      </div>
    );
  }

  return (
    <div
      data-testid="pyramid-verdict-info"
      className="rounded-m-md border-[0.5px] border-m-border bg-m-surface-2 px-[14px] py-3 text-[12px] text-m-text-muted"
    >
      Scale factor resulted in 0 shares. Last buy needs more profit before adding.
    </div>
  );
}

// ── Trim result block ────────────────────────────────────────────

type TrimSuccess = {
  ticker: string;
  sharesToSell: number;
  remaining: number;
  actualNewWeight: number;
  targetWeight: number;
  currWeight: number;
  cashGenerated: number;
  costBasisTrimmed: number;
  lifoPnl: number;
  avgCostSold: number;
};

function TrimResultBlock({ result }: { result: { error: string } | TrimSuccess | null }) {
  if (result == null) {
    return (
      <div className="rounded-m-xl border-[0.5px] border-m-border bg-m-surface px-5 py-5 text-center">
        <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-m-text-dim">
          Sell Ticket
        </div>
        <div className="mt-2 text-[12px] text-m-text-dim">
          Select a holding, enter price + equity, and pick a target weight.
        </div>
      </div>
    );
  }

  if ("error" in result) {
    return (
      <div
        data-testid="trim-no-trim-needed"
        className="rounded-m-md border-[0.5px] px-[14px] py-3 text-[12px]"
        style={{
          background: "color-mix(in oklab, var(--m-warn) 12%, var(--m-surface))",
          borderColor: "var(--m-warn-border)",
          color: "var(--m-text)",
        }}
      >
        {result.error}
      </div>
    );
  }

  const r = result;
  return (
    <div className="flex flex-col gap-3">
      <div className="text-[13px] font-semibold text-m-text">
        Sell Ticket: <span className="font-m-num tabular-nums text-m-accent">{r.ticker}</span>
      </div>

      {/* Sell Ticket — 3 stacked inline cards */}
      <div>
        <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.10em] text-m-text-dim">
          Sell Ticket
        </div>
        <div className="grid grid-cols-1 gap-2">
          <ResultCard
            label="Shares to Sell"
            value={`−${r.sharesToSell}`}
            tone="down"
            inline
          />
          <ResultCard
            label="Remaining"
            value={`${r.remaining} shs`}
            inline
          />
          <ResultCard
            label="New Weight"
            value={`${r.actualNewWeight.toFixed(1)}%`}
            sub={`target: ${r.targetWeight}%`}
            tone="up"
            inline
          />
        </div>
      </div>

      {/* Financial Impact (LIFO) — 3 stacked inline cards */}
      <div>
        <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.10em] text-m-text-dim">
          Financial Impact (LIFO)
        </div>
        <div className="grid grid-cols-1 gap-2">
          <ResultCard
            label="Cash Generated"
            value={formatCurrency(r.cashGenerated, { decimals: 0 })}
            tone="up"
            inline
          />
          <ResultCard
            label="Cost Basis (Sold)"
            value={formatCurrency(r.costBasisTrimmed, { decimals: 0 })}
            sub={`avg ${formatCurrency(r.avgCostSold)}/sh`}
            inline
          />
          <ResultCard
            label="Realized P&L"
            value={formatCurrency(r.lifoPnl, { showSign: true, signGlyph: "unicode", decimals: 0 })}
            sub={
              r.costBasisTrimmed > 0
                ? `${(r.lifoPnl / r.costBasisTrimmed * 100).toFixed(2)}% return`
                : undefined
            }
            tone={r.lifoPnl >= 0 ? "up" : "down"}
            inline
          />
        </div>
      </div>

      {/* Verdict — 2 branches mirroring desktop L1082-1091.
          Success boundary is >= 0 (break-even = profit lock). */}
      <TrimVerdict result={r} />
    </div>
  );
}

function TrimVerdict({ result }: { result: TrimSuccess }) {
  if (result.lifoPnl >= 0) {
    return (
      <div
        data-testid="trim-verdict-success"
        className="rounded-m-md border-[0.5px] px-[14px] py-3 text-[12px]"
        style={{
          background: "color-mix(in oklab, var(--m-accent) 10%, var(--m-surface))",
          borderColor: "var(--m-accent-border)",
          color: "var(--m-text)",
        }}
      >
        <strong>Profit Lock:</strong> This trim locks in {formatCurrency(result.lifoPnl, { decimals: 0 })} profit.
      </div>
    );
  }
  return (
    <div
      data-testid="trim-verdict-loss"
      className="rounded-m-md border-[0.5px] px-[14px] py-3 text-[12px]"
      style={{
        background: "color-mix(in oklab, var(--m-warn) 12%, var(--m-surface))",
        borderColor: "var(--m-warn-border)",
        color: "var(--m-text)",
      }}
    >
      <strong>Note:</strong> This trim realizes a loss of {formatCurrency(Math.abs(result.lifoPnl), { decimals: 0 })} based on your most recent purchases (LIFO).
    </div>
  );
}

// ── Picker tiles ──────────────────────────────────────────────────

function ModePickerTile({
  sizingMode,
  onChange,
}: {
  sizingMode: 0 | 1 | 2;
  onChange: (i: 0 | 1 | 2) => void;
}) {
  const m = SIZING_MODES[sizingMode];
  const displayLabel =
    m.key === "defense" ? "Defense" : m.key === "normal" ? "Normal" : "Offense";
  return (
    <MobileSelectSheet
      triggerLabel="Mode"
      triggerValue={displayLabel}
      triggerSubValue={`${m.pct.toFixed(2)}%`}
      triggerAccent
      triggerSelected
      sheetTitle="Sizing mode"
    >
      {(close) => (
        <div className="flex flex-col">
          {SIZING_MODES.map((opt) => {
            const isActive = opt.index === sizingMode;
            const label =
              opt.key === "defense" ? "Defense" : opt.key === "normal" ? "Normal" : "Offense";
            return (
              <button
                key={opt.key}
                type="button"
                role="option"
                aria-selected={isActive}
                onClick={() => {
                  onChange(opt.index);
                  close();
                }}
                className="flex min-h-[52px] items-center justify-between border-b-[0.5px] border-m-border px-5 py-3.5 text-left last:border-b-0"
              >
                <span className="flex flex-col">
                  <span className={`text-base ${isActive ? "font-medium" : ""} text-m-text`}>
                    {label}
                  </span>
                  <span className="font-m-num text-[12px] tabular-nums text-m-text-dim">
                    {opt.pct.toFixed(2)}%
                  </span>
                </span>
                {isActive && (
                  <Check size={20} strokeWidth={2} className="text-m-accent" aria-hidden="true" />
                )}
              </button>
            );
          })}
        </div>
      )}
    </MobileSelectSheet>
  );
}

function SizePickerTile({
  sizeIdx,
  onChange,
}: {
  sizeIdx: number;
  onChange: (i: number) => void;
}) {
  const s = SIZE_OPTIONS[sizeIdx];
  return (
    <MobileSelectSheet
      triggerLabel="Size"
      triggerValue={s.label}
      triggerSubValue={`${s.pct}%`}
      sheetTitle="Target size"
    >
      {(close) => (
        <div className="flex flex-col">
          {SIZE_OPTIONS.map((opt, i) => {
            const isActive = i === sizeIdx;
            return (
              <button
                key={opt.label}
                type="button"
                role="option"
                aria-selected={isActive}
                onClick={() => {
                  onChange(i);
                  close();
                }}
                className="flex min-h-[52px] items-center justify-between border-b-[0.5px] border-m-border px-5 py-3.5 text-left last:border-b-0"
              >
                <span className="flex flex-col">
                  <span className={`text-base ${isActive ? "font-medium" : ""} text-m-text`}>
                    {opt.label}
                  </span>
                  <span className="font-m-num text-[12px] tabular-nums text-m-text-dim">
                    {opt.pct}% of NLV
                  </span>
                </span>
                {isActive && (
                  <Check size={20} strokeWidth={2} className="text-m-accent" aria-hidden="true" />
                )}
              </button>
            );
          })}
        </div>
      )}
    </MobileSelectSheet>
  );
}

// ── Result-card primitives ────────────────────────────────────────

function ResultCard({
  label,
  value,
  sub,
  tone,
  inline,
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: "up" | "down" | "warn";
  inline?: boolean;
}) {
  const toneClass =
    tone === "up"
      ? "text-m-accent"
      : tone === "down"
        ? "text-m-down"
        : tone === "warn"
          ? "text-m-warn"
          : "text-m-text";
  if (inline) {
    return (
      <div className="rounded-m-md border-[0.5px] border-m-border bg-m-surface px-[14px] py-[10px]">
        <div className="flex items-baseline justify-between gap-2">
          <span className="text-[10px] font-medium uppercase tracking-[0.08em] text-m-text-dim">
            {label}
          </span>
          <span className={`font-m-num text-base font-medium tabular-nums ${toneClass}`}>
            {value}
          </span>
        </div>
        {sub && (
          <div className="mt-0.5 text-right text-[10px] text-m-text-dim font-m-num tabular-nums">
            {sub}
          </div>
        )}
      </div>
    );
  }
  return (
    <div className="rounded-m-md border-[0.5px] border-m-border bg-m-surface px-[14px] py-3">
      <div className="text-[10px] font-medium uppercase tracking-[0.08em] text-m-text-dim">
        {label}
      </div>
      <div className={`mt-1 font-m-num text-[18px] font-medium tabular-nums ${toneClass}`}>
        {value}
      </div>
      {sub && (
        <div className="mt-0.5 font-m-num text-[10px] tabular-nums text-m-text-dim">{sub}</div>
      )}
    </div>
  );
}

// ── Volatility sub-components (preserved from prior arc) ──────────

function MiniMetric({
  label,
  value,
  sub,
}: {
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <div className="rounded-m-md border-[0.5px] border-m-border bg-m-surface px-[14px] py-[10px]">
      <div className="text-[10px] font-medium uppercase tracking-[0.08em] text-m-text-dim">
        {label}
      </div>
      <div className="mt-1 font-m-num text-base font-medium tabular-nums text-m-text">
        {value}
      </div>
      {sub && (
        <div className="text-[10px] text-m-text-dim mt-0.5 font-m-num tabular-nums">{sub}</div>
      )}
    </div>
  );
}

function ScenarioRow({
  scenario,
  targetSize,
  isRecommended,
}: {
  scenario: SizingScenario;
  targetSize: number;
  isRecommended: boolean;
}) {
  const accentVar = scenario.label === "Tech Stop" ? "var(--m-accent)" : "var(--m-warn)";
  return (
    <div
      data-testid={`scenario-${scenario.label.replace(/\s+/g, "-").replace("×", "x").toLowerCase()}`}
      className="rounded-m-md border-[0.5px] px-[14px] py-[10px]"
      style={{
        background: isRecommended
          ? `color-mix(in oklab, ${accentVar} 10%, var(--m-surface))`
          : "var(--m-surface)",
        borderColor: isRecommended ? "var(--m-accent-border)" : "var(--m-border)",
        borderLeftWidth: isRecommended ? 3 : 1,
        borderLeftColor: accentVar,
      }}
    >
      <div className="flex items-center justify-between">
        <div className="text-[10px] font-medium uppercase tracking-[0.08em] text-m-text-dim">
          {scenario.label}
        </div>
        {isRecommended && (
          <span
            data-testid="recommended-pill"
            className="text-[9px] uppercase tracking-[0.08em] font-semibold px-1.5 py-0.5 rounded-[6px]"
            style={{ background: "var(--m-accent)", color: "var(--m-bg)" }}
          >
            Recommended
          </span>
        )}
      </div>
      <div className="mt-1 flex items-baseline justify-between">
        <span className="font-m-num text-[20px] font-medium tabular-nums text-m-text">
          {scenario.finalShares.toLocaleString()} <span className="text-[12px] text-m-text-muted">shs</span>
        </span>
        <span className="font-m-num text-[12px] tabular-nums text-m-text-muted">
          {scenario.positionPct.toFixed(1)}% NLV
        </span>
      </div>
      <div className="mt-0.5 flex items-baseline justify-between text-[11px] text-m-text-dim">
        <span>Stop {formatCurrency(scenario.effectiveStop)} ({scenario.atrFraction.toFixed(2)}× ATR)</span>
        {scenario.capBinds && (
          <span data-testid="cap-binds" className="text-m-warn">capped @ {targetSize}%</span>
        )}
      </div>
    </div>
  );
}

// ── Field primitives ──────────────────────────────────────────────

function NumberFieldCell({
  label,
  value,
  onChange,
  ariaLabel,
  suffix,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  ariaLabel: string;
  suffix?: string;
  placeholder?: string;
}) {
  return (
    <label className="block rounded-m-md border-[0.5px] border-m-border bg-m-surface px-[14px] py-[10px]">
      <span className="mb-0.5 block text-[10px] font-medium text-m-text-dim">{label}</span>
      <span className="flex items-baseline gap-1">
        <input
          type="text"
          inputMode="decimal"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          aria-label={ariaLabel}
          placeholder={placeholder}
          className="min-w-0 flex-1 bg-transparent font-m-num text-lg font-medium tabular-nums text-m-text placeholder:text-m-text-faint focus:outline-none"
        />
        {suffix && (
          <span className="font-m-num text-lg font-medium tabular-nums text-m-text-dim">
            {suffix}
          </span>
        )}
      </span>
    </label>
  );
}

function ReadOnlyFieldCell({
  label,
  labelIcon,
  value,
}: {
  label: string;
  labelIcon?: React.ReactNode;
  value: string;
}) {
  return (
    <div className="rounded-m-md border-[0.5px] border-m-border bg-m-surface px-[14px] py-[10px]">
      <div className="mb-0.5 flex items-center gap-1 text-[10px] font-medium text-m-text-dim">
        {label}
        {labelIcon}
      </div>
      <div className="font-m-num text-lg font-medium tabular-nums text-m-text">{value}</div>
    </div>
  );
}

// ── Options picker tile (Method: Risk / Equivalent) ───────────────

const OPTIONS_METHODS = [
  { key: "risk", label: "Risk", description: "Premium = max risk" },
  { key: "equivalent", label: "Equivalent", description: "Match stock exposure" },
] as const;

function OptionsMethodPickerTile({
  optMode,
  onChange,
}: {
  optMode: "risk" | "equivalent";
  onChange: (m: "risk" | "equivalent") => void;
}) {
  const current = OPTIONS_METHODS.find((m) => m.key === optMode) ?? OPTIONS_METHODS[0];
  return (
    <MobileSelectSheet
      triggerLabel="Method"
      triggerValue={current.label}
      triggerSubValue={current.description}
      triggerSelected
      sheetTitle="Sizing method"
    >
      {(close) => (
        <div className="flex flex-col">
          {OPTIONS_METHODS.map((opt) => {
            const isActive = opt.key === optMode;
            return (
              <button
                key={opt.key}
                type="button"
                role="option"
                aria-selected={isActive}
                onClick={() => {
                  onChange(opt.key);
                  close();
                }}
                className="flex min-h-[52px] items-center justify-between border-b-[0.5px] border-m-border px-5 py-3.5 text-left last:border-b-0"
              >
                <span className="flex flex-col">
                  <span className={`text-base ${isActive ? "font-medium" : ""} text-m-text`}>
                    {opt.label}
                  </span>
                  <span className="text-[12px] text-m-text-dim">{opt.description}</span>
                </span>
                {isActive && (
                  <Check size={20} strokeWidth={2} className="text-m-accent" aria-hidden="true" />
                )}
              </button>
            );
          })}
        </div>
      )}
    </MobileSelectSheet>
  );
}

// ── Options result block ──────────────────────────────────────────

function OptionsResultBlock({
  result,
  sizingMode,
  costPerContract,
  equity,
  ticker,
  targetSize,
}: {
  result: OptionsResult | null;
  sizingMode: 0 | 1 | 2;
  costPerContract: string;
  equity: number;
  ticker: string;
  targetSize: number;
}) {
  if (result == null) {
    return (
      <div className="rounded-m-xl border-[0.5px] border-m-border bg-m-surface px-5 py-5 text-center">
        <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-m-text-dim">
          Options Sizer
        </div>
        <div className="mt-2 text-[12px] text-m-text-dim">
          Enter cost per contract + equity to size your options trade.
        </div>
      </div>
    );
  }

  if (result.mode === "risk") {
    const r = result;
    const modeLabel = SIZING_MODES[sizingMode].label.split(" ")[0];
    return (
      <div className="flex flex-col gap-3">
        <div className="text-[13px] font-semibold text-m-text">Risk-Based Options Sizing</div>

        {/* Summary — 3 stacked inline cards */}
        <div className="grid grid-cols-1 gap-2">
          <ResultCard
            label="Selected Risk Budget"
            value={formatCurrency(r.recBudget, { decimals: 0 })}
            sub={`${SIZING_MODES[sizingMode].pct}% of equity (${modeLabel})`}
            inline
          />
          <ResultCard
            label="Cost / Contract"
            value={formatCurrency(r.cpc, { decimals: 0 })}
            sub={`$${costPerContract} × 100 shares`}
            inline
          />
          <ResultCard
            label="Recommended"
            value={`${r.recContracts} contract${r.recContracts !== 1 ? "s" : ""}`}
            sub={`${formatCurrency(r.recTotal, { decimals: 0 })} (${r.recPct.toFixed(1)}% NLV) · ${r.recLimiting}`}
            tone="up"
            inline
          />
        </div>

        {/* Warning when single contract exceeds budget */}
        {r.recContracts === 0 && (
          <div
            data-testid="options-risk-warning"
            className="rounded-m-md border-[0.5px] px-[14px] py-3 text-[12px]"
            style={{
              background: "color-mix(in oklab, var(--m-warn) 12%, var(--m-surface))",
              borderColor: "var(--m-warn-border)",
              color: "var(--m-text)",
            }}
          >
            A single contract ({formatCurrency(r.cpc, { decimals: 0 })}) exceeds your risk budget ({formatCurrency(r.recBudget, { decimals: 0 })}). Consider a cheaper strike or spread.
          </div>
        )}

        {/* All Risk Tiers — card stack (no row highlight to mirror desktop) */}
        <div>
          <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.10em] text-m-text-dim">
            All Risk Tiers
          </div>
          <div className="grid grid-cols-1 gap-2">
            {r.rows.map((row) => (
              <div
                key={row.label}
                data-testid={`options-risk-tier-${row.pct}`}
                className="rounded-m-md border-[0.5px] border-m-border bg-m-surface px-[14px] py-[10px]"
              >
                <div className="flex items-baseline justify-between">
                  <span className="text-[12px] font-medium text-m-text">{row.label}</span>
                  <span className="font-m-num text-base font-medium tabular-nums text-m-text">
                    {row.contracts} {row.contracts === 1 ? "contract" : "contracts"}
                  </span>
                </div>
                <div className="mt-1 grid grid-cols-3 gap-2 font-m-num text-[11px] tabular-nums text-m-text-dim">
                  <span>Budget {formatCurrency(row.budget, { decimals: 0 })}</span>
                  <span>Cost {formatCurrency(row.totalCost, { decimals: 0 })}</span>
                  <span className="text-right">{row.pctNlv.toFixed(2)}% NLV</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="text-[11px] text-m-text-faint">
          Hard cap: 5% of NLV ({formatCurrency(r.hardCapBudget, { decimals: 0 })}) — no tier can exceed this.
        </div>
      </div>
    );
  }

  // Equivalent mode
  const r = result;
  const selected = r.positionTiers.find((t) => t.pct === targetSize);
  return (
    <div className="flex flex-col gap-3">
      <div className="text-[13px] font-semibold text-m-text">
        Position Equivalent: <span className="font-m-num tabular-nums text-m-accent">{ticker || "—"}</span>
      </div>
      <div className="text-[12px] text-m-text-dim">
        How many option contracts replicate stock exposure at each position size tier.
      </div>

      {/* Summary — 3 stacked inline cards */}
      <div className="grid grid-cols-1 gap-2">
        <ResultCard label="Stock Price" value={formatCurrency(r.price)} sub={ticker || "—"} inline />
        <ResultCard
          label="Cost / Contract"
          value={formatCurrency(r.cpc, { decimals: 0 })}
          sub={`$${costPerContract} × 100 shares`}
          inline
        />
        <ResultCard
          label="Account Equity"
          value={formatCurrency(equity, { decimals: 0 })}
          inline
        />
      </div>

      {/* Position Tiers — card stack with active-targetSize highlight */}
      <div>
        <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.10em] text-m-text-dim">
          Position Tiers
        </div>
        <div className="grid grid-cols-1 gap-2">
          {r.positionTiers.map((t) => {
            const isActive = t.pct === targetSize;
            return (
              <div
                key={t.label}
                data-testid={`options-equiv-tier-${t.pct}`}
                aria-current={isActive ? "true" : undefined}
                className="rounded-m-md border-[0.5px] px-[14px] py-[10px]"
                style={{
                  background: isActive
                    ? "color-mix(in oklab, var(--m-accent) 10%, var(--m-surface))"
                    : "var(--m-surface)",
                  borderColor: isActive ? "var(--m-accent-border)" : "var(--m-border)",
                  borderLeftWidth: isActive ? 3 : 1,
                  borderLeftColor: isActive ? "var(--m-accent)" : "var(--m-border)",
                }}
              >
                <div className="flex items-baseline justify-between">
                  <span className="text-[12px] font-medium text-m-text">
                    {t.label} <span className="text-m-text-dim">({t.pct}%)</span>
                    {isActive && <span className="ml-1 text-m-accent">←</span>}
                  </span>
                  <span className="font-m-num text-base font-medium tabular-nums text-m-text">
                    {t.contracts} {t.contracts === 1 ? "contract" : "contracts"}
                  </span>
                </div>
                <div className="mt-1 grid grid-cols-3 gap-2 font-m-num text-[11px] tabular-nums text-m-text-dim">
                  <span>Stock {formatCurrency(t.positionValue, { decimals: 0 })}</span>
                  <span>{t.sharesEquiv} shs equiv</span>
                  <span className="text-right">{t.pctNlv.toFixed(2)}% NLV</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Success banner mirroring desktop verbiage */}
      {selected && (
        <div
          data-testid="options-equiv-banner"
          className="rounded-m-md border-[0.5px] px-[14px] py-3 text-[12px]"
          style={{
            background: "color-mix(in oklab, var(--m-accent) 10%, var(--m-surface))",
            borderColor: "var(--m-accent-border)",
            color: "var(--m-text)",
          }}
        >
          At <strong>{targetSize}%</strong> target: Buy <strong>{selected.contracts} contract{selected.contracts !== 1 ? "s" : ""}</strong> ({selected.sharesEquiv} share equivalent) for {formatCurrency(selected.totalCost, { decimals: 0 })} ({selected.pctNlv.toFixed(1)}% of NLV).
        </div>
      )}
    </div>
  );
}
