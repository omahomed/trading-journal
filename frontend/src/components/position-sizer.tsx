"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import { api, getActivePortfolio, type TradePosition, type TradeDetail } from "@/lib/api";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";
import { SearchSelect } from "./search-select";

type SizerTab = "volatility" | "scalein" | "pyramid" | "trim" | "options";

const TABS: { key: SizerTab; label: string; icon: string }[] = [
  { key: "volatility", label: "New Position Sizer", icon: "⚖️" },
  { key: "scalein", label: "Scale In Sizer", icon: "📐" },
  { key: "pyramid", label: "Pyramid Sizer", icon: "🔺" },
  { key: "trim", label: "Trim (Sell Down)", icon: "✂️" },
  { key: "options", label: "Options Sizer", icon: "🎰" },
];

// SIZING_MODES + the MCT-state → mode mapping live in @/lib/sizing-mode
// so Position Sizer and Log Buy stay in lockstep — no chance of one
// drifting to a different risk percentage than the other.
import {
  SIZING_MODES as SIZING_MODES_BASE,
  mctStateToSizingMode,
  deriveAutoSizingMode,
  exitLadderFloor,
  describeMctSource,
  type ExitAlert,
} from "@/lib/sizing-mode";
import { computeVolatilitySizing, type SizingScenario, type VolSizerResults } from "@/lib/vol-sizer";

// Local view shape — keeps the component-internal usage of
// `SIZING_MODES[i].label` untouched. The labels here include the leading
// emoji (Position Sizer's old style); the shared lib stores icon
// separately so Log Buy can render its own layout.
const SIZING_MODES = SIZING_MODES_BASE.map(m => ({
  key: m.key,
  label: `${m.icon} ${m.label}`,
  pct: m.pct,
}));

const SIZE_OPTIONS = [
  { label: "Starter (2.5%)", pct: 2.5 }, { label: "Half (5%)", pct: 5 },
  { label: "Standard (7.5%)", pct: 7.5 }, { label: "Full (10%)", pct: 10 },
  { label: "Overweight (12.5%)", pct: 12.5 }, { label: "Core (15%)", pct: 15 },
  { label: "Core+ (17.5%)", pct: 17.5 }, { label: "Max (20%)", pct: 20 },
];

// --- Shared UI Primitives ---

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-[10px] uppercase tracking-[0.10em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>{label}</label>
      {children}
    </div>
  );
}

function Radio({ checked, onClick, label }: { checked: boolean; onClick: () => void; label: string }) {
  return (
    <label className="flex items-center gap-2 cursor-pointer text-[13px]" onClick={onClick}>
      <span className="w-[16px] h-[16px] rounded-full flex items-center justify-center shrink-0"
            style={{ border: `2px solid ${checked ? "#08a86b" : "var(--border)"}` }}>
        {checked && <span className="w-[8px] h-[8px] rounded-full" style={{ background: "#08a86b" }} />}
      </span>
      <span style={{ color: checked ? "var(--ink)" : "var(--ink-3)" }}>{label}</span>
    </label>
  );
}

function MetricCard({ label, value, sub, color, accent }: { label: string; value: string; sub?: string; color?: string; accent?: string }) {
  return (
    <div className="p-4 rounded-[12px] relative overflow-hidden" style={{
      border: "1px solid var(--border)",
      borderLeft: accent ? `4px solid ${accent}` : "1px solid var(--border)",
      background: accent ? `color-mix(in oklab, ${accent} 4%, var(--surface))` : undefined,
    }}>
      <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>{label}</div>
      <div className="text-[24px] font-semibold mt-1 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace", color: color || "var(--ink)" }}>{value}</div>
      {sub && <div className="text-[11px] mt-0.5" style={{ color: "var(--ink-4)" }}>{sub}</div>}
    </div>
  );
}

function Banner({ type, children }: { type: "info" | "success" | "warning" | "error"; children: React.ReactNode }) {
  const colors = {
    info: { bg: "#1e40af", text: "#3b82f6" },
    success: { bg: "#08a86b", text: "#16a34a" },
    warning: { bg: "#f59f00", text: "#d97706" },
    error: { bg: "#e5484d", text: "#dc2626" },
  };
  const c = colors[type];
  return (
    <div className="px-4 py-3 rounded-[10px] text-[13px] font-medium"
         style={{ background: `color-mix(in oklab, ${c.bg} 10%, var(--surface))`, color: c.text, border: `1px solid color-mix(in oklab, ${c.bg} 30%, var(--border))` }}>
      {children}
    </div>
  );
}

const inputCls = "w-full h-[42px] px-3.5 rounded-[10px] text-[13px] outline-none";
const inputStyle: React.CSSProperties = {
  background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)",
  fontFamily: "var(--font-jetbrains), monospace",
};

// --- LIFO Engine ---
interface InventoryLot { qty: number; price: number }

function buildLIFOInventory(details: TradeDetail[], tradeId: string, fallbackAvg: number): InventoryLot[] {
  const trxs = details
    .filter(d => d.trade_id === tradeId)
    .sort((a, b) => {
      const dateCompare = (a.date || "").localeCompare(b.date || "");
      if (dateCompare !== 0) return dateCompare;
      return (a.action?.toUpperCase() === "BUY" ? 0 : 1) - (b.action?.toUpperCase() === "BUY" ? 0 : 1);
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

// --- Main Component ---

export function PositionSizer({ navColor, onNavigate, initialTab, onTabConsumed, initialHoldingTradeId, onHoldingConsumed }: { navColor: string; onNavigate?: (page: string) => void; initialTab?: string; onTabConsumed?: () => void; initialHoldingTradeId?: string; onHoldingConsumed?: () => void }) {
  const [tab, setTab] = useState<SizerTab>((initialTab as SizerTab) || "volatility");

  useEffect(() => {
    if (initialTab && ["volatility", "scalein", "pyramid", "trim", "options"].includes(initialTab)) {
      setTab(initialTab as SizerTab);
      onTabConsumed?.();
    }
  }, [initialTab, onTabConsumed]);
  const [equity, setEquity] = useState(0);
  const [openTrades, setOpenTrades] = useState<TradePosition[]>([]);
  const [allDetails, setAllDetails] = useState<TradeDetail[]>([]);
  // mfSuggestion (V10 MA-stack heuristic) is gone — sizing mode now reads
  // from MCT state directly. See mctState + sizingModeManual below.
  const [calculated, setCalculated] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  // stopMode + atrMultiplier let ATR scenarios round-trip from the Sizer
  // through Log Buy without resolving to a dollar price here (Log Buy
  // fetches its own atrPct from /api/prices/lookup and recomputes the
  // effective stop). Tech-stop scenarios continue to emit a resolved
  // `stop` value and tag stopMode='price' so the receiver flips Log Buy
  // out of its default pct mode (pre-existing bug ride-along).
  const sendToLogBuy = (data: {
    ticker: string;
    shares: number;
    price: number;
    stop?: number;
    stopMode?: "price" | "atr";
    atrMultiplier?: 1 | 1.5 | 2;
    trade_id?: string;
    action?: string;
  }) => {
    localStorage.setItem("ps_prefill", JSON.stringify(data));
    if (onNavigate) onNavigate("logbuy");
  };

  const sendToLogSell = (data: { ticker: string; shares: number; price: number; trade_id?: string }) => {
    localStorage.setItem("ps_prefill_sell", JSON.stringify(data));
    if (onNavigate) onNavigate("logsell");
  };

  // Shared inputs
  // sizingMode default is Normal (1) — overwritten on mount by the MCT
  // state read in the load-effect below. Stays Normal if the read fails
  // (mctStateToSizingMode falls back to safe middle ground).
  const [sizingMode, setSizingMode] = useState<0 | 1 | 2>(1);
  // mctState + sizingModeManual track WHY the current mode is what it is.
  // - sizingModeManual=false → set by MCT state read (auto)
  // - sizingModeManual=true  → user clicked a Radio (override). Reset by
  //   the "Reset to auto" button, which re-applies the MCT mapping.
  const [mctState, setMctState] = useState<string | null>(null);
  // Active exit-ladder alerts. Drive the sizing-mode floor — a fired
  // 21 EMA Violation / Confirmed Break downshifts to Normal; a fired
  // 50 SMA Violation downshifts to Defense, regardless of what the
  // M Factor state alone would have picked. See lib/sizing-mode#
  // exitLadderFloor for the full rule.
  const [activeExits, setActiveExits] = useState<readonly { signal: string; severity?: string }[]>([]);
  const [sizingModeManual, setSizingModeManual] = useState(false);
  const [entryPrice, setEntryPrice] = useState("");
  const [maLevel, setMaLevel] = useState("");
  const [buffer, setBuffer] = useState("1.00");
  const [atrPct, setAtrPct] = useState("5.0");
  const [ticker, setTicker] = useState("");
  // selectedHolding is shared by Scale-In / Pyramid / Trim tabs. The
  // legacy Volatility "Audit Active Position" mode also used it but
  // that mode is gone — the state stays for the surviving consumers.
  const [selectedHolding, setSelectedHolding] = useState("");

  // Active Campaign Summary v2 deep-links into this view via
  // ?tab=pyramid&trade_id=… — apply the trade_id once on mount, then call
  // onHoldingConsumed so the parent can clear the search param. Mirrors the
  // initialTab / onTabConsumed pattern above.
  useEffect(() => {
    if (initialHoldingTradeId) {
      setSelectedHolding(initialHoldingTradeId);
      onHoldingConsumed?.();
    }
  }, [initialHoldingTradeId, onHoldingConsumed]);
  const [targetSize, setTargetSize] = useState(10);
  const [maxRiskPct, setMaxRiskPct] = useState("0.75");
  const [costPerContract, setCostPerContract] = useState("1.00");
  const [optMode, setOptMode] = useState<"risk" | "equivalent">("risk");
  const [fetching, setFetching] = useState(false);

  // Pyramid config
  const [pyramidRules, setPyramidRules] = useState({ trigger_pct: 5, alloc_pct: 20 });

  // Auto-fetch price + ATR when ticker changes (debounced)
  useEffect(() => {
    if (!ticker || ticker.length < 1) return;
    const timeout = setTimeout(() => {
      setFetching(true);
      api.priceLookup(ticker).then(data => {
        if (data && !("error" in data)) {
          setEntryPrice(String(data.price));
          setAtrPct(String(data.atr_pct));
        }
      }).catch((err) => {
        log.debug.devOnly("position-sizer", "priceLookup missing (expected)", err);
      }).finally(() => setFetching(false));
    }, 600);
    return () => clearTimeout(timeout);
  }, [ticker]);

  useEffect(() => {
    Promise.all([
      api.journalLatest(getActivePortfolio()).catch((err) => {
        log.error("position-sizer", "journalLatest fetch failed", err);
        return { end_nlv: 100000 };
      }),
      api.tradesOpen(getActivePortfolio()).catch((err) => {
        log.error("position-sizer", "tradesOpen fetch failed", err);
        return [];
      }),
      api.tradesOpenDetails(getActivePortfolio()).catch((err) => {
        log.error("position-sizer", "tradesOpenDetails fetch failed", err);
        return { details: [], lot_closures: [] };
      }),
      // V11 MCT state drives default sizing mode. Replaces the legacy
      // /api/market/mfactor MA-stack heuristic. rallyPrefix returns
      // {state: POWERTREND|UPTREND|RALLY MODE|CORRECTION, ...}; we
      // discard everything except `state` here.
      api.rallyPrefix().catch((err) => {
        log.error("position-sizer", "rallyPrefix fetch failed", err);
        return { prefix: "" };
      }),
      api.config("pyramid_rules").catch((err) => {
        log.error("position-sizer", "config pyramid_rules fetch failed", err);
        return { value: { trigger_pct: 5, alloc_pct: 20 } };
      }),
    ]).then(([j, open, details, rally, pyrCfg]) => {
      setEquity(parseFloat(String((j as any).end_nlv || 100000)));
      setOpenTrades(open as TradePosition[]);
      setAllDetails(details.details);
      const stateStr = (rally as { state?: string } | null)?.state ?? null;
      const exits = ((rally as { active_exits?: ExitAlert[] } | null)?.active_exits ?? []) as ExitAlert[];
      setMctState(stateStr);
      setActiveExits(exits);
      // Only auto-apply when the user hasn't manually overridden — but on
      // mount the user can't have, so this is effectively unconditional.
      // The guard becomes meaningful when this effect re-runs (it doesn't
      // today, but defending against a future deps change).
      // Auto-mode now takes the more conservative of (M Factor state,
      // exit-ladder floor). E.g. POWERTREND + 50 SMA Violation → Defense,
      // not Offense — even if the regime hasn't flipped to CORRECTION yet.
      setSizingMode(deriveAutoSizingMode(stateStr, exits).idx);
      setSizingModeManual(false);
      if (pyrCfg && (pyrCfg as any).value) {
        setPyramidRules((pyrCfg as any).value);
      }
    });
  }, []);

  const resetCalc = useCallback(() => { setCalculated(false); setErrorMsg(""); }, []);

  // --- Derived Values ---
  const entry = parseFloat(entryPrice) || 0;
  const ma = parseFloat(maLevel) || 0;
  const buf = parseFloat(buffer) || 1;
  const atr = parseFloat(atrPct) || 5;
  const riskPct = SIZING_MODES[sizingMode].pct;
  const riskBudget = equity * (riskPct / 100);
  const calcStop = ma > 0 ? ma * (1 - buf / 100) : 0;
  const stopDist = entry > 0 && calcStop > 0 ? entry - calcStop : 0;
  const stopDistPct = entry > 0 && stopDist > 0 ? (stopDist / entry) * 100 : 0;

  // Holding data
  const holdingData = openTrades.find(t => t.trade_id === selectedHolding);
  const holdingInventory = useMemo(() => {
    if (!holdingData) return [];
    return buildLIFOInventory(allDetails, holdingData.trade_id, holdingData.avg_entry || 0);
  }, [holdingData, allDetails]);

  const holdingAvgCost = useMemo(() => {
    if (!holdingData) return 0;
    const avg = lifoAvgCost(holdingInventory);
    return avg > 0 ? avg : (holdingData.avg_entry || 0);
  }, [holdingData, holdingInventory]);

  // --- Calculation Handlers ---
  const handleCalculate = () => {
    setErrorMsg("");
    setCalculated(false);

    if (tab === "volatility") {
      if (!ticker || entry <= 0 || atr <= 0) {
        setErrorMsg("Please ensure Ticker, Price, and ATR are entered correctly.");
        return;
      }
      if (ma <= 0) {
        setErrorMsg("Please enter a Key MA Level — the new sizer always grounds risk against a technical stop.");
        return;
      }
      const stop = ma * (1 - buf / 100);
      if (stop >= entry) {
        setErrorMsg(`Stop (${formatCurrency(stop)}) is at or above entry price (${formatCurrency(entry)}).`);
        return;
      }
    } else if (tab === "scalein") {
      if (ma <= 0) {
        setErrorMsg("Enter a Key MA Level to calculate your global stop.");
        return;
      }
      const stop = ma * (1 - buf / 100);
      if (stop >= entry) {
        setErrorMsg(`Stop (${formatCurrency(stop)}) is at or above current price (${formatCurrency(entry)}).`);
        return;
      }
      if (!holdingData) {
        setErrorMsg("Select a holding first.");
        return;
      }
    } else if (tab === "pyramid") {
      if (!holdingData || entry <= 0 || atr <= 0) {
        setErrorMsg("Please ensure Position, Price, and ATR are entered correctly.");
        return;
      }
      if (holdingInventory.length === 0) {
        setErrorMsg("No buy transactions found for this position.");
        return;
      }
    } else if (tab === "trim") {
      if (!holdingData || entry <= 0) {
        setErrorMsg("Select a holding and enter a price.");
        return;
      }
    }

    setCalculated(true);
  };

  // ━━━ Volatility Sizer Results ━━━
  // Delegates to the shared `computeVolatilitySizing` lib (see
  // frontend/src/lib/vol-sizer.ts). The `calculated` upstream gate +
  // handleCalculate's input validation should make the lib's input
  // assertions unreachable in normal flow; the try/catch is a belt to
  // keep a malformed input from crashing the whole page.
  const volResults: VolSizerResults | null = useMemo(() => {
    if (!calculated || tab !== "volatility") return null;
    try {
      return computeVolatilitySizing({
        equity,
        entry,
        ma,
        bufferPct: buf,
        atrPct: atr,
        tolPct: SIZING_MODES[sizingMode].pct,
        targetSizePct: targetSize,
      });
    } catch (err) {
      log.error("position-sizer", "vol-sizer compute failed", err);
      return null;
    }
  }, [calculated, tab, entry, atr, ma, buf, sizingMode, equity, targetSize]);

  // ━━━ Scale-In Results ━━━
  const scaleResults = useMemo(() => {
    if (!calculated || tab !== "scalein" || !holdingData) return null;
    const stop = ma * (1 - buf / 100);
    const newAddRiskPerShare = entry - stop;
    if (newAddRiskPerShare <= 0) return null;

    const currShares = holdingData.shares || 0;
    const avgEntry = holdingData.avg_entry || 0;
    // Contract multiplier — 100 for options, 1 for equities. Required to
    // dollarize per-share risk into notional risk; previously omitted, which
    // understated option risk by 100× and let the budget guard mis-fire.
    const multiplier = holdingData.multiplier || 1;
    const currValue = currShares * entry;

    // Real-money risk on existing shares is only the portion of cost basis
    // that sits above the stop. When stop ≥ avg_entry the position is
    // "risk-free" — worst case is locking in profit, not losing capital —
    // so existing shares contribute zero to the risk budget. This lets the
    // user pyramid into a winner that has moved above its stop instead of
    // being blocked by an open-risk calculation that double-counts gains
    // already protected by the trailing stop.
    const existingRiskPerShare = Math.max(0, avgEntry - stop);
    const existingRisk = currShares * existingRiskPerShare * multiplier;
    const isRiskFree = existingRiskPerShare === 0 && currShares > 0;

    const targetValue = equity * (targetSize / 100);
    const targetTotalShares = Math.ceil(targetValue / entry);
    const targetAdd = targetTotalShares - currShares;

    const maxRisk = SIZING_MODES[sizingMode].pct;
    const maxRiskDol = equity * (maxRisk / 100);
    const remainingBudget = maxRiskDol - existingRisk;
    const affordableAdd = remainingBudget > 0
      ? Math.floor(remainingBudget / (newAddRiskPerShare * multiplier))
      : 0;

    if (targetAdd <= 0) return { error: `You are already at or above the target weight! (Current: ${formatCurrency(currValue, { decimals: 0 })} vs Target: ${formatCurrency(targetValue, { decimals: 0 })})` };
    if (affordableAdd <= 0) {
      return { error: `NO ADD - Existing ${currShares} shares risk ${formatCurrency(existingRisk, { decimals: 0 })} of capital below stop, exhausting the ${formatCurrency(maxRiskDol, { decimals: 0 })} budget. Tighten your stop above your ${formatCurrency(avgEntry)} avg cost or reduce position.` };
    }

    const recommendedAdd = Math.min(targetAdd, affordableAdd);
    const newTotal = currShares + recommendedAdd;
    const newAvgCost = newTotal > 0 ? (currShares * avgEntry + recommendedAdd * entry) / newTotal : 0;
    const costOfAdd = recommendedAdd * entry;
    // Total real-money risk after the add: locked-in risk on existing
    // shares (zero when risk-free) plus the new shares' risk-to-stop.
    const newAddRisk = recommendedAdd * newAddRiskPerShare * multiplier;
    const totalRiskAtNew = existingRisk + newAddRisk;
    const newWeight = equity > 0 ? (newTotal * entry / equity) * 100 : 0;
    const verdict = affordableAdd >= targetAdd ? "success" : "partial";

    return {
      recommendedAdd, newTotal, newAvgCost, costOfAdd, totalRiskAtNew, newWeight,
      stop, riskPerShare: newAddRiskPerShare, maxRiskDol, maxRisk, targetAdd,
      avgEntry, currShares, verdict, isRiskFree, existingRisk, newAddRisk,
    };
  }, [calculated, tab, holdingData, ma, buf, entry, equity, targetSize, sizingMode]);

  // ━━━ Pyramid Results ━━━
  const pyramidResults = useMemo(() => {
    if (!calculated || tab !== "pyramid" || !holdingData || holdingInventory.length === 0) return null;

    const shares = holdingData.shares || 0;
    const avgCost = holdingAvgCost;
    const lastBuy = holdingInventory[holdingInventory.length - 1];
    const lastBuyPrice = lastBuy.price;
    const lastBuyProfitPct = ((entry - lastBuyPrice) / lastBuyPrice) * 100;
    const cushionPct = avgCost > 0 ? ((entry - avgCost) / avgCost) * 100 : 0;

    const baseAddPct = pyramidRules.alloc_pct / 100;
    const thresholdPct = pyramidRules.trigger_pct;

    let scaleFactor: number;
    if (lastBuyProfitPct >= thresholdPct) scaleFactor = 1.0;
    else if (lastBuyProfitPct > 0) scaleFactor = lastBuyProfitPct / thresholdPct;
    else scaleFactor = 0;

    const pyramidMaxShares = Math.ceil(shares * baseAddPct * scaleFactor);

    // ATR / Hard Cap Ceiling (same tolerance tiers as vol sizer)
    let tierName = "Tier 3 (Defense)";
    let tolPct = 0.5;
    let atrMultiplier = 1.0;
    if (cushionPct >= 20) { tierName = "Tier 1 (High Cushion)"; tolPct = 1.0; atrMultiplier = 2.0; }
    else if (cushionPct >= 5) { tierName = "Tier 2 (Moderate)"; tolPct = 0.65; atrMultiplier = 1.5; }

    const dailyRiskBudget = equity * (tolPct / 100);
    const atrRiskBudget = dailyRiskBudget * atrMultiplier;
    const atrDecimal = atr / 100;
    const maxSharesAtr = Math.floor(atrRiskBudget / (entry * atrDecimal));
    const maxSharesCap = Math.floor((equity * 0.20) / entry);
    const positionCeiling = Math.min(maxSharesAtr, maxSharesCap);
    const roomToAdd = Math.max(0, positionCeiling - Math.floor(shares));

    const pyramidAllowed = Math.min(pyramidMaxShares, roomToAdd);
    const pyramidValue = pyramidAllowed * entry;

    const baseAdd = Math.floor(shares * baseAddPct);

    return {
      lastBuyPrice, lastBuyProfitPct, cushionPct, avgCost,
      scaleFactor, pyramidMaxShares, baseAdd,
      positionCeiling, roomToAdd, pyramidAllowed, pyramidValue,
      tierName, atrMultiplier, shares,
    };
  }, [calculated, tab, holdingData, holdingInventory, holdingAvgCost, entry, atr, equity, pyramidRules]);

  // ━━━ Trim Results ━━━
  const trimResults = useMemo(() => {
    if (!calculated || tab !== "trim" || !holdingData) return null;
    const currShares = holdingData.shares || 0;
    const currVal = currShares * entry;
    const currWeight = equity > 0 ? (currVal / equity) * 100 : 0;
    const targetWeight = targetSize;

    if (targetWeight >= currWeight) {
      return { error: `Target (${targetWeight}%) is higher than Current (${currWeight.toFixed(1)}%). No trim needed.` };
    }

    const targetVal = equity * (targetWeight / 100);
    const valueToSell = currVal - targetVal;
    const sharesToSell = Math.ceil(valueToSell / entry);
    const remaining = Math.max(0, currShares - sharesToSell);
    const actualNewWeight = equity > 0 ? (remaining * entry / equity) * 100 : 0;

    // LIFO P&L
    const inventory = [...holdingInventory.map(l => ({ ...l }))]; // deep copy
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
    // Fallback for remainder
    if (sharesNeeded > 0) {
      accumulatedCost += sharesNeeded * (holdingData.avg_entry || 0);
    }

    const costBasisTrimmed = accumulatedCost;
    const cashGenerated = sharesToSell * entry;
    const lifoPnl = cashGenerated - costBasisTrimmed;
    const avgCostSold = sharesToSell > 0 ? costBasisTrimmed / sharesToSell : 0;

    return {
      sharesToSell, remaining, actualNewWeight, targetWeight,
      cashGenerated, costBasisTrimmed, lifoPnl, avgCostSold, currWeight,
    };
  }, [calculated, tab, holdingData, holdingInventory, entry, equity, targetSize]);

  // ━━━ Options Results ━━━
  const optResults = useMemo(() => {
    if (!calculated || tab !== "options") return null;
    const cpc = (parseFloat(costPerContract) || 0) * 100;
    if (cpc <= 0 || equity <= 0) return null;
    const hardCapBudget = equity * 0.05;

    if (optMode === "risk") {
      // Risk-based: show contracts at 1%/2%/3%/4%/5% risk tiers.
      // Max (5%) tier matches the literal hard cap — at default sizing
      // modes it renders the same numbers as the footer's hard-cap note,
      // intentional for tier-row transparency.
      const tiers = [
        { label: "Conservative (1%)", pct: 1.0 },
        { label: "Normal (2%)", pct: 2.0 },
        { label: "Aggressive (3%)", pct: 3.0 },
        { label: "Heavy (4%)", pct: 4.0 },
        { label: "Max (5%)", pct: 5.0 },
      ];
      const rows = tiers.map(t => {
        const budget = equity * (t.pct / 100);
        const contracts = Math.min(Math.floor(budget / cpc), Math.floor(hardCapBudget / cpc));
        const totalCost = contracts * cpc;
        const pctNlv = equity > 0 ? (totalCost / equity) * 100 : 0;
        return { ...t, budget, contracts, totalCost, pctNlv };
      });
      // Recommended = selected sizing mode
      const recIdx = sizingMode; // 0=defense, 1=normal, 2=offense
      const recBudget = riskBudget;
      const recContracts = Math.min(Math.floor(recBudget / cpc), Math.floor(hardCapBudget / cpc));
      const recTotal = recContracts * cpc;
      const recPct = equity > 0 ? (recTotal / equity) * 100 : 0;
      const recLimiting = recContracts === Math.floor(recBudget / cpc) ? "Risk Budget" : "Hard Cap (5%)";
      return { mode: "risk" as const, cpc, hardCapBudget, rows, recContracts, recTotal, recPct, recLimiting, recBudget };
    } else {
      // Position equivalent: how many contracts to match stock exposure
      const price = parseFloat(entryPrice) || 0;
      if (price <= 0) return null;
      const positionTiers = SIZE_OPTIONS.filter(s => s.pct <= 20).map(s => {
        const positionValue = equity * (s.pct / 100);
        const sharesEquiv = Math.floor(positionValue / price);
        const contracts = Math.ceil(sharesEquiv / 100); // 1 contract = 100 shares
        const totalCost = contracts * cpc;
        const pctNlv = equity > 0 ? (totalCost / equity) * 100 : 0;
        return { label: s.label, pct: s.pct, positionValue, sharesEquiv, contracts, totalCost, pctNlv };
      });
      return { mode: "equivalent" as const, cpc, hardCapBudget, positionTiers, price: price };
    }
  }, [calculated, tab, costPerContract, equity, riskBudget, optMode, entryPrice, sizingMode]);

  const needsHolding = ["scalein", "pyramid", "trim"].includes(tab);
  const needsMaBuffer = ["scalein", "volatility"].includes(tab);
  const needsTarget = tab === "trim" || tab === "volatility" || tab === "scalein" || (tab === "options" && optMode === "equivalent");

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          Position <em className="italic" style={{ color: navColor }}>Sizer</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>{getActivePortfolio()}</div>
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 mb-6 overflow-x-auto pb-0.5" style={{ borderBottom: "2px solid var(--border)" }}>
        {TABS.map(t => (
          <button key={t.key} onClick={() => { setTab(t.key); resetCalc(); }}
                  className="px-4 py-2 text-[12px] font-medium whitespace-nowrap transition-all"
                  style={{
                    color: tab === t.key ? navColor : "var(--ink-4)",
                    borderBottom: tab === t.key ? `2px solid ${navColor}` : "2px solid transparent",
                    marginBottom: -2,
                  }}>
            {t.icon} {t.label}
          </button>
        ))}
      </div>

      {/* Tab header */}
      <div className="mb-5">
        <h2 className="text-[18px] font-semibold flex items-center gap-2">
          {TABS.find(t => t.key === tab)?.icon} {TABS.find(t => t.key === tab)?.label}
        </h2>
        <div className="text-[13px] mt-1" style={{ color: "var(--ink-4)" }}>
          {tab === "volatility" && "Normalize risk by sizing based on ATR volatility AND technical stop."}
          {tab === "scalein" && "Scale up to target weight while respecting global stop and risk budget."}
          {tab === "pyramid" && `Size add-on purchases to winning positions. Max ${pyramidRules.alloc_pct}% of shares per add, gated by last buy's profit.`}
          {tab === "trim" && "Calculate shares to sell to reach a desired weight, with LIFO P&L estimation."}
          {tab === "options" && "Size option positions using risk budget. Premium = max risk."}
        </div>
      </div>

      {/* Pyramid Rules Expander */}
      {tab === "pyramid" && (
        <details className="mb-4 rounded-[10px] overflow-hidden" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
          <summary className="px-4 py-2.5 text-[12px] font-semibold cursor-pointer" style={{ color: "var(--ink-3)" }}>
            View Pyramid Rules
          </summary>
          <div className="px-4 pb-3 text-[12px] leading-relaxed" style={{ color: "var(--ink-3)" }}>
            <p className="mb-1"><strong>How it works:</strong></p>
            <ol className="list-decimal ml-4 flex flex-col gap-0.5">
              <li>Each add is capped at <strong>{pyramidRules.alloc_pct}%</strong> of your current shares</li>
              <li>Your last buy must be up <strong>at least {pyramidRules.trigger_pct}%</strong> for a full-size add</li>
              <li>If last buy is up less than {pyramidRules.trigger_pct}%, the add scales proportionally: <code style={{ background: "var(--surface)", padding: "1px 4px", borderRadius: 4 }}>(profit% / {pyramidRules.trigger_pct}%) x {pyramidRules.alloc_pct}%</code></li>
              <li>If last buy is <strong>flat or down</strong>, no add is allowed</li>
              <li>The add is also capped by your ATR limit and {pyramidRules.alloc_pct}% hard cap</li>
            </ol>
          </div>
        </details>
      )}

      {/* Volatility Sizer Rules Expander */}
      {tab === "volatility" && (
        <details className="mb-4 rounded-[10px] overflow-hidden" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
          <summary className="px-4 py-2.5 text-[12px] font-semibold cursor-pointer" style={{ color: "var(--ink-3)" }}>
            View Sizer Rules
          </summary>
          <div className="px-4 pb-3 text-[12px] leading-relaxed" style={{ color: "var(--ink-3)" }}>
            <p className="mb-1"><strong>Sizing Mode (driven by M Factor state):</strong></p>
            <ul className="list-disc ml-4 mb-2">
              <li><strong>Defense:</strong> 0.50% risk — equity curve flat/down</li>
              <li><strong>Normal:</strong> 0.75% risk — equity curve recovering</li>
              <li><strong>Offense:</strong> 1.00% risk — equity curve strong, confirmed uptrend</li>
            </ul>
            <p className="mb-1"><strong>What this sizer shows:</strong></p>
            <ul className="list-disc ml-4 mb-2">
              <li><strong>Tech Stop:</strong> shares the risk budget can absorb to your MA-based stop</li>
              <li><strong>1× / 1.5× / 2× ATR cushion:</strong> shares the risk budget can absorb if your stop is one of those ATR distances below entry</li>
              <li>Every scenario is capped at your target tier ({"<= "}20% of NLV)</li>
            </ul>
            <p className="mb-1"><strong>Recommendation logic:</strong></p>
            <ul className="list-disc ml-4">
              <li>Tech stop ≥ 1 ATR away → size to the tech stop</li>
              <li>Tech stop {"<"} 1 ATR away → daily noise will chop you out; size to 1.5× ATR instead</li>
            </ul>
          </div>
        </details>
      )}

      {/* ═══════════ INPUTS ═══════════ */}
      <div className="flex flex-col gap-5 mb-6">

        {/* Ticker (new trade tabs) */}
        {tab === "volatility" && (
          <Field label="Ticker Symbol">
            <div className="relative">
              <input type="text" value={ticker} onChange={e => { setTicker(e.target.value.toUpperCase()); resetCalc(); }}
                     placeholder="XYZ" className={inputCls} style={inputStyle} />
              {fetching && (
                <div className="absolute right-3 top-1/2 -translate-y-1/2">
                  <span className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin inline-block" style={{ color: "var(--ink-4)" }} />
                </div>
              )}
            </div>
          </Field>
        )}

        {/* Holding picker (scalein, pyramid, trim).
            Searchable so the user can type the ticker (e.g. "GEV") to
            jump straight to the campaign — meaningful once openTrades
            grows past ~10 positions. The selection side-effects
            (price + ATR auto-fill) live in a named handler so the
            SearchSelect onChange stays trivially typed. Empty value
            ("") is the "no selection" sentinel and is prepended as
            an explicit option so the label renders cleanly. */}
        {needsHolding && (
          <Field label="Select Position">
            <SearchSelect
              value={selectedHolding}
              onChange={(v) => {
                setSelectedHolding(v);
                resetCalc();
                const h = openTrades.find(t => t.trade_id === v);
                if (h) {
                  setFetching(true);
                  api.priceLookup(h.ticker).then(data => {
                    if (data && !("error" in data)) {
                      setEntryPrice(String(data.price));
                      setAtrPct(String(data.atr_pct));
                    }
                  }).catch(() => {}).finally(() => setFetching(false));
                }
              }}
              options={[
                { value: "", label: "Select..." },
                ...openTrades.map(t => ({ value: t.trade_id, label: `${t.ticker} (${t.shares} shs) | ${t.trade_id}` })),
              ]}
              placeholder="Select..."
            />
            {holdingData && (
              <div className="mt-2 text-[12px] px-3 py-2 rounded-[8px]" style={{ background: "var(--bg)", color: "var(--ink-3)" }}>
                {holdingData.shares} shs @ {formatCurrency(parseFloat(String(holdingData.avg_entry || 0)))}
                {holdingData.rule ? ` · ${holdingData.rule}` : ""}
              </div>
            )}
          </Field>
        )}

        {/* Entry Price + Equity */}
        {tab !== "options" && (
          <div className="grid grid-cols-2 gap-4">
            <Field label={needsHolding ? "Current Price ($)" : "Entry Price ($)"}>
              <input type="number" value={entryPrice} onChange={e => { setEntryPrice(e.target.value); resetCalc(); }}
                     step="0.01" placeholder="0.00" className={inputCls} style={inputStyle} />
            </Field>
            <Field label="Account Equity (NLV)">
              <input type="number" value={equity > 0 ? String(equity) : ""} onChange={e => { setEquity(parseFloat(e.target.value) || 0); resetCalc(); }}
                     step="1000" className={inputCls} style={inputStyle} />
            </Field>
          </div>
        )}

        {/* MA Level + Buffer */}
        {needsMaBuffer && (
          <div className="grid grid-cols-2 gap-4">
            <Field label="Key MA Level ($)">
              <input type="number" value={maLevel} onChange={e => { setMaLevel(e.target.value); resetCalc(); }}
                     step="0.01" placeholder="0.00" className={inputCls} style={inputStyle} />
            </Field>
            <Field label="Buffer (%)">
              <input type="number" value={buffer} onChange={e => { setBuffer(e.target.value); resetCalc(); }}
                     step="0.1" placeholder="1.00" className={inputCls} style={inputStyle} />
            </Field>
          </div>
        )}

        {/* Calculated stop info banner.
            On the volatility tab we annotate the banner with the tech
            stop's ATR fraction (read from volResults once available)
            and show a warning sub-banner when the stop sits inside 1
            ATR. Both come from the shared vol-sizer lib — the banner
            reflects the lib's view of the inputs the user can see. */}
        {needsMaBuffer && calcStop > 0 && entry > 0 && (() => {
          const volAtr = atr > 0 && entry > 0 && calcStop > 0 ? stopDistPct / atr : null;
          return (
            <>
              <Banner type="info">
                Calculated Stop: <strong>{formatCurrency(calcStop)}</strong> (MA ${ma.toFixed(2)} - {buf.toFixed(1)}% buffer) — {stopDistPct.toFixed(1)}% below entry
                {tab === "volatility" && volAtr !== null && ` · ${volAtr.toFixed(2)}× ATR`}
              </Banner>
              {tab === "volatility" && volResults?.warning?.show && (
                <Banner type="warning">{volResults.warning.text}</Banner>
              )}
            </>
          );
        })()}

        {/* ATR (volatility, pyramid) */}
        {(tab === "volatility" || tab === "pyramid") && (
          <Field label="ATR % (21-Day)">
            <input type="number" value={atrPct} onChange={e => { setAtrPct(e.target.value); resetCalc(); }}
                   step="0.1" placeholder="5.0" className={inputCls} style={inputStyle} />
          </Field>
        )}

        {/* Sizing Mode (new trade tabs) */}
        {(tab === "volatility" || tab === "scalein" || tab === "options") && (
          <>
            <div className="px-4 py-2.5 rounded-[10px] text-[12px] flex items-center justify-between gap-3 flex-wrap"
                 data-testid="sizer-mode-indicator"
                 style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
              <div>
                <span style={{ color: "var(--ink-4)" }}>
                  {sizingModeManual ? "Manual:" : "Auto:"}
                </span>{" "}
                <strong>{SIZING_MODES[sizingMode].label}</strong>
                {!sizingModeManual && (
                  <span style={{ color: "var(--ink-4)" }}>
                    {" "}({describeMctSource(mctState, exitLadderFloor(activeExits))})
                  </span>
                )}
              </div>
              {sizingModeManual && (
                <button type="button"
                        data-testid="sizer-reset-to-auto"
                        onClick={() => {
                          // "Reset to auto" replays the same derivation
                          // the load effect used so exit-ladder floor
                          // applies here too — clicking Reset right
                          // after a 50 SMA Violation lands you on
                          // Defense, not raw-state Offense.
                          setSizingMode(deriveAutoSizingMode(mctState, activeExits).idx);
                          setSizingModeManual(false);
                          resetCalc();
                        }}
                        className="text-[11px] px-2 py-0.5 rounded-[6px] underline"
                        style={{ color: "var(--ink-3)" }}>
                  Reset to auto
                </button>
              )}
            </div>
            <Field label="Sizing Mode">
              <div className="flex gap-4 mt-1">
                {SIZING_MODES.map((m, i) => (
                  <Radio key={m.key} checked={sizingMode === i}
                         onClick={() => {
                           setSizingMode(i as 0 | 1 | 2);
                           setSizingModeManual(true);
                           resetCalc();
                         }}
                         label={m.label} />
                ))}
              </div>
            </Field>
          </>
        )}

        {/* Scale-in: max risk derived from sizing mode (shown as read-only) */}

        {/* Options: cost per contract + equity */}
        {tab === "options" && (
          <>
            <Field label="Sizing Method">
              <div className="flex gap-4 mt-1">
                <Radio checked={optMode === "risk"} onClick={() => { setOptMode("risk"); resetCalc(); }} label="Risk-Based (premium = max risk)" />
                <Radio checked={optMode === "equivalent"} onClick={() => { setOptMode("equivalent"); resetCalc(); }} label="Position Equivalent (match stock exposure)" />
              </div>
            </Field>
            <div className="grid grid-cols-2 gap-4">
              <Field label="Cost per Contract ($)">
                <input type="number" value={costPerContract} onChange={e => { setCostPerContract(e.target.value); resetCalc(); }}
                       step="0.05" min="0.01" placeholder="1.00" className={inputCls} style={inputStyle} />
              </Field>
              <Field label="Account Equity (NLV)">
                <input type="number" value={equity > 0 ? String(equity) : ""} onChange={e => { setEquity(parseFloat(e.target.value) || 0); resetCalc(); }}
                       step="1000" className={inputCls} style={inputStyle} />
              </Field>
            </div>
            {optMode === "equivalent" && (
              <div className="grid grid-cols-2 gap-4">
                <Field label="Ticker">
                  <input type="text" value={ticker} onChange={e => { setTicker(e.target.value.toUpperCase()); resetCalc(); }}
                         placeholder="XYZ" className={inputCls} style={inputStyle} />
                </Field>
                <Field label="Stock Price ($)">
                  <input type="number" value={entryPrice} onChange={e => { setEntryPrice(e.target.value); resetCalc(); }}
                         step="0.01" placeholder="0.00" className={inputCls} style={inputStyle} />
                </Field>
              </div>
            )}
          </>
        )}

        {/* Target position size */}
        {needsTarget && (
          <Field label="Target Position Size">
            <div className="flex flex-wrap gap-1.5 mt-1">
              {SIZE_OPTIONS.map(s => (
                <button key={s.label} onClick={() => { setTargetSize(s.pct); resetCalc(); }}
                        className="px-3 py-1.5 rounded-[8px] text-[11px] font-medium transition-all"
                        style={{
                          background: targetSize === s.pct ? navColor : "var(--bg)",
                          color: targetSize === s.pct ? "#fff" : "var(--ink-4)",
                          border: `1px solid ${targetSize === s.pct ? navColor : "var(--border)"}`,
                        }}>
                  {s.label}
                </button>
              ))}
            </div>
          </Field>
        )}
      </div>

      {/* Error Banner */}
      {errorMsg && (
        <div className="mb-4">
          <Banner type="error">{errorMsg}</Banner>
        </div>
      )}

      {/* Calculate button */}
      <button onClick={handleCalculate}
              className="h-[44px] px-8 rounded-[10px] text-[13px] font-semibold text-white mb-8 transition-all hover:brightness-110"
              style={{ background: "#6366f1" }}>
        {tab === "trim" ? "Calculate Trim Impact" : tab === "scalein" ? "Calculate Add-On" : tab === "pyramid" ? "Run Pyramid Analysis" : "Calculate Size"}
      </button>

      {/* ═══════════ RESULTS ═══════════ */}
      {calculated && !errorMsg && (
        <div style={{ animation: "slide-up 0.15s ease-out" }}>

          {/* ── VOLATILITY SIZER ── */}
          {tab === "volatility" && volResults && (
            <VolatilityResults
              ticker={ticker}
              entry={entry}
              equity={equity}
              targetSize={targetSize}
              tolPct={SIZING_MODES[sizingMode].pct}
              modeName={SIZING_MODES_BASE[sizingMode].label}
              results={volResults}
              onSendToLogBuy={(args) => sendToLogBuy(args)}
            />
          )}

          {/* ── SCALE IN ── */}
          {tab === "scalein" && scaleResults && (
            <>
              {"error" in scaleResults ? (
                <Banner type="error">{scaleResults.error}</Banner>
              ) : (
                <>
                  {scaleResults.isRiskFree && (
                    <div className="mb-4 px-3 py-2 rounded-[8px] text-[12px]"
                         style={{ background: "color-mix(in oklab, #08a86b 10%, var(--surface))",
                                  color: "#08a86b",
                                  border: "1px solid color-mix(in oklab, #08a86b 30%, var(--border))" }}>
                      ✓ Position is risk-free — stop {formatCurrency(scaleResults.stop)} sits above your {formatCurrency(scaleResults.avgEntry)} avg cost. Existing shares contribute $0 to the risk budget; only new-add risk counts.
                    </div>
                  )}
                  <h3 className="text-[15px] font-semibold mb-4">SCALE TICKET</h3>
                  <div className="grid grid-cols-4 gap-3 mb-4">
                    <MetricCard label="ADD SHARES" value={`+${scaleResults.recommendedAdd}`}
                                accent="#08a86b" color="#08a86b" />
                    <MetricCard label="EST. COST" value={formatCurrency(scaleResults.costOfAdd)}
                                accent="#6366f1" />
                    <MetricCard label="NEW TOTAL" value={`${scaleResults.newTotal} shs`}
                                sub={`${scaleResults.newWeight.toFixed(1)}% Weight`}
                                accent="#3b82f6" />
                    <MetricCard label="NEW AVG COST" value={formatCurrency(scaleResults.newAvgCost)}
                                sub={`From ${formatCurrency(scaleResults.avgEntry)}`}
                                accent="#f59f00" />
                  </div>

                  <h3 className="text-[15px] font-semibold mb-4">RISK MANAGEMENT</h3>
                  <div className="grid grid-cols-3 gap-3 mb-6">
                    <MetricCard label="Global Stop" value={formatCurrency(scaleResults.stop)}
                                sub={`-${(scaleResults.riskPerShare / entry * 100).toFixed(1)}% from price`}
                                accent="#e5484d" />
                    <MetricCard label="Total Risk at New Size" value={formatCurrency(scaleResults.totalRiskAtNew, { decimals: 0 })}
                                sub={`${(scaleResults.totalRiskAtNew / equity * 100).toFixed(2)}% of NLV`}
                                accent="#f59f00" />
                    <MetricCard label="Risk Budget" value={formatCurrency(scaleResults.maxRiskDol, { decimals: 0 })}
                                sub={`${scaleResults.maxRisk}% of Equity`}
                                accent="#6366f1" />
                  </div>

                  <h3 className="text-[14px] font-semibold mb-2">The Verdict</h3>
                  {scaleResults.verdict === "success" ? (
                    <Banner type="success">
                      ADD {scaleResults.recommendedAdd} shares to reach {scaleResults.newWeight.toFixed(1)}% — Total risk {formatCurrency(scaleResults.totalRiskAtNew, { decimals: 0 })} within {formatCurrency(scaleResults.maxRiskDol, { decimals: 0 })} budget.
                    </Banner>
                  ) : (
                    <Banner type="warning">
                      RISK LIMIT: Full target ({scaleResults.targetAdd} shares) would exceed budget. Safe add: {scaleResults.recommendedAdd} shares ({scaleResults.newWeight.toFixed(1)}% weight). Scale up on next pullback to MA.
                    </Banner>
                  )}

                  <div className="mt-4">
                    <button onClick={() => sendToLogBuy({ ticker: holdingData?.ticker || "", shares: scaleResults.recommendedAdd, price: entry, stop: scaleResults.stop, stopMode: "price", trade_id: holdingData?.trade_id, action: "scale_in" })}
                            className="w-full h-[48px] rounded-[12px] text-[13px] font-semibold transition-all hover:brightness-95 cursor-pointer"
                            style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }}>
                      📝 Send to Log Buy — {holdingData?.ticker} (+{scaleResults.recommendedAdd} shs @ {formatCurrency(entry)})
                    </button>
                  </div>
                </>
              )}
            </>
          )}

          {/* ── PYRAMID SIZER ── */}
          {tab === "pyramid" && pyramidResults && (
            <>
              <h3 className="text-[15px] font-semibold mb-4">Pyramid Analysis: {holdingData?.ticker}</h3>

              {/* Last Buy Info */}
              <div className="grid grid-cols-3 gap-3 mb-4">
                <MetricCard label="Last Buy Price" value={formatCurrency(pyramidResults.lastBuyPrice)}
                            accent="#3b82f6" />
                <MetricCard label="Last Buy P&L" value={`${pyramidResults.lastBuyProfitPct.toFixed(2)}%`}
                            sub={`${formatCurrency(entry - pyramidResults.lastBuyPrice)}/share`}
                            color={pyramidResults.lastBuyProfitPct >= 0 ? "#08a86b" : "#e5484d"}
                            accent={pyramidResults.lastBuyProfitPct >= 0 ? "#08a86b" : "#e5484d"} />
                <MetricCard label="Total Cushion" value={`${pyramidResults.cushionPct.toFixed(2)}%`}
                            sub={`Avg Cost: ${formatCurrency(pyramidResults.avgCost)}`}
                            accent="#f59f00" />
              </div>

              {/* Pyramid Calculation */}
              <h3 className="text-[14px] font-semibold mb-3">Pyramid Calculation</h3>
              <div className="grid grid-cols-3 gap-3 mb-4">
                <MetricCard label={`Base Add (${pyramidRules.alloc_pct}%)`} value={`${pyramidResults.baseAdd} shs`}
                            sub={`${pyramidRules.alloc_pct}% of ${Math.floor(pyramidResults.shares)} shares`} />
                <MetricCard label="Scale Factor" value={`${(pyramidResults.scaleFactor * 100).toFixed(0)}%`}
                            sub={`Last buy up ${pyramidResults.lastBuyProfitPct.toFixed(1)}% (need ${pyramidRules.trigger_pct}%)`} />
                <MetricCard label="Pyramid Max" value={`${pyramidResults.pyramidMaxShares} shs`}
                            sub="After scaling" />
              </div>

              {/* Ceiling Check */}
              <div className="grid grid-cols-3 gap-3 mb-6">
                <MetricCard label="Position Ceiling" value={`${pyramidResults.positionCeiling} shs`}
                            sub={`${pyramidResults.tierName} | ${pyramidResults.atrMultiplier.toFixed(1)}x ATR`} />
                <MetricCard label="Current Position" value={`${Math.floor(pyramidResults.shares)} shs`}
                            sub={`${(pyramidResults.shares * entry / equity * 100).toFixed(1)}% Weight`} />
                <MetricCard label="Room to Add" value={`${pyramidResults.roomToAdd} shs`}
                            sub="Before hitting ceiling" />
              </div>

              {/* Verdict */}
              <h3 className="text-[14px] font-semibold mb-2">The Verdict</h3>
              {pyramidResults.scaleFactor === 0 ? (
                <Banner type="error">
                  NO ADD — Last buy is {pyramidResults.lastBuyProfitPct < 0 ? "down" : "flat"} ({pyramidResults.lastBuyProfitPct.toFixed(2)}%). Wait for it to work.
                </Banner>
              ) : pyramidResults.pyramidAllowed === 0 && pyramidResults.pyramidMaxShares > 0 ? (
                <Banner type="warning">
                  NO ROOM — Pyramid says {pyramidResults.pyramidMaxShares} shares, but position is at ATR/cap ceiling ({pyramidResults.positionCeiling} shs).
                </Banner>
              ) : pyramidResults.pyramidAllowed > 0 ? (
                <>
                  <Banner type="success">
                    ADD {pyramidResults.pyramidAllowed} shares ({formatCurrency(pyramidResults.pyramidValue, { decimals: 0 })}) — Limited by: {pyramidResults.pyramidAllowed === pyramidResults.pyramidMaxShares ? "Pyramid pace" : "ATR/Cap ceiling"}
                  </Banner>
                  <div className="grid grid-cols-3 gap-3 mt-4">
                    <MetricCard label="Add Shares" value={`${pyramidResults.pyramidAllowed} shs`}
                                sub={formatCurrency(pyramidResults.pyramidValue, { decimals: 0 })} accent="#08a86b" />
                    <MetricCard label="New Total" value={`${Math.floor(pyramidResults.shares) + pyramidResults.pyramidAllowed} shs`}
                                sub={`${((Math.floor(pyramidResults.shares) + pyramidResults.pyramidAllowed) * entry / equity * 100).toFixed(1)}% Weight`} />
                    <MetricCard label="New Avg Cost" value={formatCurrency((pyramidResults.avgCost * pyramidResults.shares + entry * pyramidResults.pyramidAllowed) / (pyramidResults.shares + pyramidResults.pyramidAllowed))}
                                sub={`From ${formatCurrency(pyramidResults.avgCost)}`} />
                  </div>
                  <div className="mt-4">
                    <button onClick={() => sendToLogBuy({ ticker: holdingData?.ticker || "", shares: pyramidResults.pyramidAllowed, price: entry, trade_id: holdingData?.trade_id, action: "scale_in" })}
                            className="w-full h-[48px] rounded-[12px] text-[13px] font-semibold transition-all hover:brightness-95 cursor-pointer"
                            style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }}>
                      📝 Send to Log Buy — {holdingData?.ticker} (+{pyramidResults.pyramidAllowed} shs @ {formatCurrency(entry)})
                    </button>
                  </div>
                </>
              ) : (
                <Banner type="info">Scale factor resulted in 0 shares. Last buy needs more profit before adding.</Banner>
              )}
            </>
          )}

          {/* ── TRIM ── */}
          {tab === "trim" && trimResults && (
            <>
              {"error" in trimResults ? (
                <Banner type="warning">{trimResults.error}</Banner>
              ) : (
                <>
                  <h3 className="text-[15px] font-semibold mb-4">Sell Ticket</h3>
                  <div className="grid grid-cols-3 gap-3 mb-4">
                    <MetricCard label="SHARES TO SELL" value={`-${trimResults.sharesToSell}`}
                                color="#e5484d" accent="#e5484d" />
                    <MetricCard label="REMAINING" value={`${trimResults.remaining} shs`}
                                accent="#3b82f6" />
                    <MetricCard label="NEW WEIGHT" value={`${trimResults.actualNewWeight.toFixed(1)}%`}
                                sub={`Target: ${trimResults.targetWeight}%`}
                                accent="#08a86b" />
                  </div>

                  <h3 className="text-[15px] font-semibold mb-4">Financial Impact (LIFO)</h3>
                  <div className="grid grid-cols-3 gap-3 mb-6">
                    <MetricCard label="Cash Generated" value={formatCurrency(trimResults.cashGenerated)}
                                accent="#08a86b" />
                    <MetricCard label="Cost Basis (Sold)" value={formatCurrency(trimResults.costBasisTrimmed)}
                                sub={`Avg: ${formatCurrency(trimResults.avgCostSold)}/sh`}
                                accent="#6366f1" />
                    <MetricCard label="Realized P&L" value={formatCurrency(trimResults.lifoPnl)}
                                sub={trimResults.costBasisTrimmed > 0 ? `${(trimResults.lifoPnl / trimResults.costBasisTrimmed * 100).toFixed(2)}% Return` : undefined}
                                color={trimResults.lifoPnl >= 0 ? "#08a86b" : "#e5484d"}
                                accent={trimResults.lifoPnl >= 0 ? "#08a86b" : "#e5484d"} />
                  </div>

                  <h3 className="text-[14px] font-semibold mb-2">The Verdict</h3>
                  {trimResults.lifoPnl >= 0 ? (
                    <Banner type="success">
                      Profit Lock: This trim locks in {formatCurrency(trimResults.lifoPnl)} profit.
                    </Banner>
                  ) : (
                    <Banner type="warning">
                      Note: This trim realizes a loss of {formatCurrency(Math.abs(trimResults.lifoPnl))} based on your most recent purchases (LIFO).
                    </Banner>
                  )}

                  {trimResults.sharesToSell > 0 && (
                    <div className="mt-4">
                      <button onClick={() => sendToLogSell({ ticker: holdingData?.ticker || "", shares: trimResults.sharesToSell, price: entry, trade_id: holdingData?.trade_id })}
                              className="w-full h-[48px] rounded-[12px] text-[13px] font-semibold transition-all hover:brightness-95 cursor-pointer"
                              style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }}>
                        📝 Send to Log Sell — {holdingData?.ticker} ({trimResults.sharesToSell} shs @ {formatCurrency(entry)})
                      </button>
                    </div>
                  )}
                </>
              )}
            </>
          )}

          {/* ── OPTIONS ── */}
          {tab === "options" && optResults && (
            <>
              {optResults.mode === "risk" ? (
                <>
                  <h3 className="text-[15px] font-semibold mb-4">Risk-Based Options Sizing</h3>
                  <div className="grid grid-cols-3 gap-3 mb-4">
                    <MetricCard label="Selected Risk Budget" value={formatCurrency(optResults.recBudget, { decimals: 0 })}
                                sub={`${SIZING_MODES[sizingMode].pct}% of equity (${SIZING_MODES[sizingMode].label.split(" ")[0]})`}
                                accent="#6366f1" />
                    <MetricCard label="Cost per Contract" value={formatCurrency(optResults.cpc, { decimals: 0 })}
                                sub={`$${costPerContract} x 100 shares`}
                                accent="#f59f00" />
                    <MetricCard label="Recommended" value={`${optResults.recContracts} contract${optResults.recContracts !== 1 ? "s" : ""}`}
                                sub={`${formatCurrency(optResults.recTotal, { decimals: 0 })} (${optResults.recPct.toFixed(1)}% NLV) · ${optResults.recLimiting}`}
                                color={navColor} accent="#08a86b" />
                  </div>

                  {optResults.recContracts === 0 && (
                    <div className="mb-4">
                      <Banner type="warning">
                        A single contract ({formatCurrency(optResults.cpc, { decimals: 0 })}) exceeds your risk budget ({formatCurrency(optResults.recBudget, { decimals: 0 })}). Consider a cheaper strike or spread.
                      </Banner>
                    </div>
                  )}

                  {/* All risk tiers table */}
                  <h4 className="text-[13px] font-semibold mb-3" style={{ color: "var(--ink-3)" }}>All Risk Tiers</h4>
                  <div className="rounded-[10px] overflow-hidden mb-4" style={{ border: "1px solid var(--border)" }}>
                    <table className="w-full text-[12px]" style={{ borderCollapse: "collapse" }}>
                      <thead>
                        <tr>
                          {["Risk Tier", "Risk %", "Budget", "Contracts", "Total Cost", "% NLV"].map(h => (
                            <th key={h} className="text-left px-3 py-2.5 text-[10px] uppercase tracking-[0.08em] font-semibold"
                                style={{ color: "var(--ink-4)", background: "var(--surface-2)", borderBottom: "1px solid var(--border)" }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {optResults.rows.map((r, i) => (
                          <tr key={r.label} style={{ borderBottom: i < optResults.rows.length - 1 ? "1px solid var(--border)" : "none" }}>
                            <td className="px-3 py-2.5 font-medium">{r.label}</td>
                            <td className="px-3 py-2.5" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{r.pct}%</td>
                            <td className="px-3 py-2.5 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{formatCurrency(r.budget, { decimals: 0 })}</td>
                            <td className="px-3 py-2.5 font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{r.contracts}</td>
                            <td className="px-3 py-2.5 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{formatCurrency(r.totalCost, { decimals: 0 })}</td>
                            <td className="px-3 py-2.5" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{r.pctNlv.toFixed(2)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>
                    Hard cap: 5% of NLV ({formatCurrency(optResults.hardCapBudget, { decimals: 0 })}) — no tier can exceed this.
                  </div>
                </>
              ) : (
                <>
                  <h3 className="text-[15px] font-semibold mb-2">Position Equivalent — {ticker || "—"}</h3>
                  <div className="text-[12px] mb-4" style={{ color: "var(--ink-4)" }}>
                    How many option contracts replicate stock exposure at each position size tier.
                  </div>

                  <div className="grid grid-cols-3 gap-3 mb-4">
                    <MetricCard label="Stock Price" value={formatCurrency(optResults.price || 0)} sub={ticker || "—"} accent="#6366f1" />
                    <MetricCard label="Cost per Contract" value={formatCurrency(optResults.cpc, { decimals: 0 })} sub={`$${costPerContract} x 100 shares`} accent="#f59f00" />
                    <MetricCard label="Account Equity" value={formatCurrency(equity, { decimals: 0 })} accent="#08a86b" />
                  </div>

                  <div className="rounded-[10px] overflow-hidden mb-4" style={{ border: "1px solid var(--border)" }}>
                    <table className="w-full text-[12px]" style={{ borderCollapse: "collapse" }}>
                      <thead>
                        <tr>
                          {["Position Size", "Stock Value", "Shares Equiv", "Contracts", "Options Cost", "% NLV"].map(h => (
                            <th key={h} className="text-left px-3 py-2.5 text-[10px] uppercase tracking-[0.08em] font-semibold"
                                style={{ color: "var(--ink-4)", background: "var(--surface-2)", borderBottom: "1px solid var(--border)" }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {(optResults.positionTiers || []).map((t, i) => (
                          <tr key={t.label} style={{ borderBottom: i < (optResults.positionTiers || []).length - 1 ? "1px solid var(--border)" : "none",
                                                     background: t.pct === targetSize ? "var(--surface-2)" : "transparent" }}>
                            <td className="px-3 py-2.5 font-medium">{t.label}{t.pct === targetSize ? " ←" : ""}</td>
                            <td className="px-3 py-2.5 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{formatCurrency(t.positionValue, { decimals: 0 })}</td>
                            <td className="px-3 py-2.5" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{t.sharesEquiv}</td>
                            <td className="px-3 py-2.5 font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{t.contracts}</td>
                            <td className="px-3 py-2.5 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{formatCurrency(t.totalCost, { decimals: 0 })}</td>
                            <td className="px-3 py-2.5" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{t.pctNlv.toFixed(2)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {(() => {
                    const sel = (optResults.positionTiers || []).find(t => t.pct === targetSize);
                    if (!sel) return null;
                    return (
                      <Banner type="success">
                        At <strong>{targetSize}%</strong> target: Buy <strong>{sel.contracts} contract{sel.contracts !== 1 ? "s" : ""}</strong> ({sel.sharesEquiv} share equivalent) for {formatCurrency(sel.totalCost, { decimals: 0 })} ({sel.pctNlv.toFixed(1)}% of NLV).
                      </Banner>
                    );
                  })()}
                </>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ── Volatility Sizer output (extracted: keeps the main render readable
// and makes per-card pill/binding-constraint logic colocated with the
// presentation). Consumes the shared `vol-sizer` lib output verbatim;
// any future shape change should land in the lib, not here.
function VolatilityResults({
  ticker, entry, equity, targetSize, tolPct, modeName, results, onSendToLogBuy,
}: {
  ticker: string;
  entry: number;
  equity: number;
  targetSize: number;
  tolPct: number;
  modeName: string;
  results: VolSizerResults;
  onSendToLogBuy: (args: {
    ticker: string;
    shares: number;
    price: number;
    stop?: number;
    stopMode?: "price" | "atr";
    atrMultiplier?: 1 | 1.5 | 2;
    action: string;
  }) => void;
}) {
  const rec = results.recommended;
  const recIsTechStop = results.recommendationReason === "tech_stop_safe";
  const method = recIsTechStop ? "tech stop" : "1.5× ATR cushion";
  // Map the recommended scenario's label back to the multiplier the
  // ATR pills in Log Buy expect. Returns null for the tech-stop case,
  // which routes through the price-mode branch instead.
  const atrMultFromLabel = (label: typeof rec.label): 1 | 1.5 | 2 | null => {
    if (label === "1x ATR") return 1;
    if (label === "1.5x ATR") return 1.5;
    if (label === "2x ATR") return 2;
    return null;
  };
  const recAtrMult = atrMultFromLabel(rec.label);
  const constraint = rec.capBinds
    ? `position-size tier (${targetSize}% NLV)`
    : `risk budget (${tolPct}%)`;

  return (
    <div data-testid="vol-results">
      <h3 className="text-[15px] font-semibold mb-4">Sizing Profile: {ticker || "—"}</h3>

      {/* Context grid */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <MetricCard label="Risk Budget" value={formatCurrency(results.riskBudget, { decimals: 0 })}
                    sub={`${tolPct}% Risk (${modeName} Mode)`}
                    accent="#6366f1" />
        <MetricCard label="ATR Noise" value={`${(results.atrPerShare / entry * 100).toFixed(2)}%`}
                    sub={`${formatCurrency(results.atrPerShare)}/share`}
                    accent="#f59f00" />
        <MetricCard label="Position Cap" value={`${results.positionCapShares} shs`}
                    sub={`${formatCurrency(results.positionCap, { decimals: 0 })} (${targetSize}% NLV)`}
                    accent="#3b82f6" />
      </div>

      {/* Tech Stop row */}
      <div className="mb-3">
        <ScenarioCard scenario={results.techStop} entry={entry} equity={equity} targetSize={targetSize}
                      accent="#3b82f6" tone="tech" isRecommended={recIsTechStop} />
      </div>

      {/* ATR Cushion grid */}
      <div className="grid grid-cols-3 gap-3 mb-6">
        {results.atrScenarios.map((s, i) => (
          <ScenarioCard key={s.label} scenario={s} entry={entry} equity={equity} targetSize={targetSize}
                        accent="#f59f00" tone="atr"
                        isRecommended={!recIsTechStop && i === 1} />
        ))}
      </div>

      {/* Verdict */}
      <h3 className="text-[14px] font-semibold mb-2">The Verdict</h3>
      <Banner type="success">
        <div>
          RECOMMENDED: Buy <strong>{rec.finalShares}</strong> shares · <strong>{rec.positionPct.toFixed(1)}%</strong> of NLV
        </div>
        <div className="mt-1 text-[12px] font-normal" style={{ opacity: 0.85 }}>
          Sized by {method} · bound by {constraint}
        </div>
        {results.warning.show && (
          <div className="mt-2 px-3 py-2 rounded-[8px] text-[12px] font-medium"
               style={{
                 background: "color-mix(in oklab, #f59f00 12%, transparent)",
                 color: "#d97706",
                 border: "1px solid color-mix(in oklab, #f59f00 30%, transparent)",
               }}>
            {results.warning.text}
          </div>
        )}
      </Banner>

      <div className="mt-4">
        <button onClick={() => {
                  // ATR scenario → emit multiplier; Log Buy fetches atrPct
                  // itself and recomputes the stop. Tech stop → emit
                  // resolved dollar price; stopMode='price' flips the
                  // receiver out of its default pct mode.
                  if (recAtrMult !== null) {
                    onSendToLogBuy({ ticker, shares: rec.finalShares, price: entry, stopMode: "atr", atrMultiplier: recAtrMult, action: "new" });
                  } else {
                    onSendToLogBuy({ ticker, shares: rec.finalShares, price: entry, stop: rec.effectiveStop, stopMode: "price", action: "new" });
                  }
                }}
                className="w-full h-[48px] rounded-[12px] text-[13px] font-semibold transition-all hover:brightness-95 cursor-pointer"
                style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }}>
          📝 Send to Log Buy — {ticker || "—"} ({rec.finalShares} shs @ {formatCurrency(entry)})
        </button>
      </div>
    </div>
  );
}

function ScenarioCard({
  scenario, entry, equity, targetSize, accent, tone, isRecommended,
}: {
  scenario: SizingScenario;
  entry: number;
  equity: number;
  targetSize: number;
  accent: string;
  tone: "tech" | "atr";
  isRecommended: boolean;
}) {
  const borderWidth = isRecommended ? 6 : 4;
  return (
    <div className="p-4 rounded-[12px] relative overflow-hidden"
         data-testid={`scenario-${scenario.label.replace(/\s+/g, "-").replace("×", "x").toLowerCase()}`}
         style={{
           border: "1px solid var(--border)",
           borderLeft: `${borderWidth}px solid ${accent}`,
           background: `color-mix(in oklab, ${accent} ${isRecommended ? 7 : 4}%, var(--surface))`,
         }}>
      <div className="flex items-center justify-between mb-1.5">
        <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>
          {scenario.label}
        </div>
        {isRecommended && (
          <span className="text-[9px] uppercase tracking-[0.08em] font-semibold px-2 py-0.5 rounded-[6px]"
                data-testid="recommended-pill"
                style={{ background: "#08a86b", color: "#fff" }}>
            Recommended
          </span>
        )}
      </div>
      <div className="flex items-baseline justify-between">
        <div>
          <div className="text-[22px] font-semibold privacy-mask"
               style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
            {scenario.finalShares} <span className="text-[13px] font-normal" style={{ color: "var(--ink-4)" }}>shs</span>
          </div>
          <div className="text-[11px] mt-0.5" style={{ color: "var(--ink-4)" }}>
            Stop {formatCurrency(scenario.effectiveStop)} · {scenario.stopDistancePct.toFixed(2)}% ({tone === "atr" ? `${scenario.atrFraction.toFixed(1)}× ATR` : `${scenario.atrFraction.toFixed(2)}× ATR`})
          </div>
        </div>
        <div className="text-right">
          <div className="text-[13px] font-medium privacy-mask"
               style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
            {formatCurrency(scenario.positionCost, { decimals: 0 })}
          </div>
          <div className="text-[10px]" style={{ color: "var(--ink-4)" }}>
            {scenario.positionPct.toFixed(1)}% NLV
          </div>
        </div>
      </div>
      <div className="mt-2 flex items-baseline justify-between text-[11px]" style={{ color: "var(--ink-4)" }}>
        <span>Risk if stopped: <strong style={{ color: "var(--ink)" }}>{formatCurrency(scenario.riskIfStopped, { decimals: 0 })}</strong> ({scenario.riskPct.toFixed(2)}% NLV)</span>
        {scenario.capBinds && (
          <span data-testid="cap-binds" style={{ color: "#d97706" }}>capped @ {targetSize}% NLV</span>
        )}
      </div>
    </div>
  );
}
