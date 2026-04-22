"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import { api, getActivePortfolio, type TradePosition, type TradeDetail } from "@/lib/api";

type SizerTab = "normal" | "volatility" | "scalein" | "pyramid" | "trim" | "options";

const TABS: { key: SizerTab; label: string; icon: string }[] = [
  { key: "normal", label: "Normal Sizer", icon: "📏" },
  { key: "volatility", label: "Volatility Sizer", icon: "⚖️" },
  { key: "scalein", label: "Scale In Sizer", icon: "📐" },
  { key: "pyramid", label: "Pyramid Sizer", icon: "🔺" },
  { key: "trim", label: "Trim (Sell Down)", icon: "✂️" },
  { key: "options", label: "Options Sizer", icon: "🎰" },
];

const SIZING_MODES = [
  { key: "defense", label: "🛡️ Defense (0.50%)", pct: 0.5 },
  { key: "normal", label: "⚖️ Normal (0.75%)", pct: 0.75 },
  { key: "offense", label: "⚔️ Offense (1.00%)", pct: 1.0 },
];

const VOL_PROFILES = [
  { key: "tight", label: "Tight (1.0x)", mult: 1.0 },
  { key: "normal", label: "Normal (1.25x)", mult: 1.25 },
  { key: "highvol", label: "High-Vol (1.5x)", mult: 1.5 },
];

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

export function PositionSizer({ navColor, onNavigate, initialTab, onTabConsumed }: { navColor: string; onNavigate?: (page: string) => void; initialTab?: string; onTabConsumed?: () => void }) {
  const [tab, setTab] = useState<SizerTab>((initialTab as SizerTab) || "normal");

  useEffect(() => {
    if (initialTab && ["normal", "volatility", "scalein", "pyramid", "trim", "options"].includes(initialTab)) {
      setTab(initialTab as SizerTab);
      onTabConsumed?.();
    }
  }, [initialTab, onTabConsumed]);
  const [equity, setEquity] = useState(0);
  const [openTrades, setOpenTrades] = useState<TradePosition[]>([]);
  const [allDetails, setAllDetails] = useState<TradeDetail[]>([]);
  const [mfSuggestion, setMfSuggestion] = useState("Unknown");
  const [calculated, setCalculated] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  const sendToLogBuy = (data: { ticker: string; shares: number; price: number; stop?: number; trade_id?: string; action?: string }) => {
    localStorage.setItem("ps_prefill", JSON.stringify(data));
    if (onNavigate) onNavigate("logbuy");
  };

  const sendToLogSell = (data: { ticker: string; shares: number; price: number; trade_id?: string }) => {
    localStorage.setItem("ps_prefill_sell", JSON.stringify(data));
    if (onNavigate) onNavigate("logsell");
  };

  // Shared inputs
  const [sizingMode, setSizingMode] = useState(1);
  const [entryPrice, setEntryPrice] = useState("");
  const [maLevel, setMaLevel] = useState("");
  const [buffer, setBuffer] = useState("1.00");
  const [atrPct, setAtrPct] = useState("5.0");
  const [volProfile, setVolProfile] = useState(0);
  const [ticker, setTicker] = useState("");
  const [selectedHolding, setSelectedHolding] = useState("");
  const [targetSize, setTargetSize] = useState(10);
  const [maxRiskPct, setMaxRiskPct] = useState("0.75");
  const [costPerContract, setCostPerContract] = useState("1.00");
  const [optMode, setOptMode] = useState<"risk" | "equivalent">("risk");
  const [fetching, setFetching] = useState(false);

  // Volatility sizer mode
  const [volSizerMode, setVolSizerMode] = useState<"new" | "audit">("new");

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
      }).catch(() => {}).finally(() => setFetching(false));
    }, 600);
    return () => clearTimeout(timeout);
  }, [ticker]);

  useEffect(() => {
    Promise.all([
      api.journalLatest(getActivePortfolio()).catch(() => ({ end_nlv: 100000 })),
      api.tradesOpen(getActivePortfolio()).catch(() => []),
      api.tradesOpenDetails(getActivePortfolio()).catch(() => []),
      api.mfactor().catch(() => ({})),
      api.config("pyramid_rules").catch(() => ({ value: { trigger_pct: 5, alloc_pct: 20 } })),
    ]).then(([j, open, details, mf, pyrCfg]) => {
      setEquity(parseFloat(String((j as any).end_nlv || 100000)));
      setOpenTrades(open as TradePosition[]);
      setAllDetails(details as TradeDetail[]);
      const nasdaq = (mf as any)?.nasdaq;
      if (nasdaq) {
        if (nasdaq.above_21ema && nasdaq.above_50sma) { setMfSuggestion("Powertrend"); setSizingMode(2); }
        else if (nasdaq.above_21ema) { setMfSuggestion("Open"); setSizingMode(1); }
        else { setMfSuggestion("Closed"); setSizingMode(0); }
      }
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

    if (tab === "normal") {
      if (!ticker || entry <= 0 || ma <= 0) {
        setErrorMsg("Please enter Ticker, Entry Price, and Key MA Level.");
        return;
      }
      const stop = ma * (1 - buf / 100);
      if (stop >= entry) {
        setErrorMsg(`Stop ($${stop.toFixed(2)}) is at or above entry price ($${entry.toFixed(2)}).`);
        return;
      }
    } else if (tab === "volatility") {
      if (volSizerMode === "new") {
        if (!ticker || entry <= 0 || atr <= 0) {
          setErrorMsg("Please ensure Ticker, Price, and ATR are entered correctly.");
          return;
        }
      } else {
        if (!holdingData || entry <= 0 || atr <= 0) {
          setErrorMsg("Please ensure Ticker, Price, and ATR are entered correctly.");
          return;
        }
      }
    } else if (tab === "scalein") {
      if (ma <= 0) {
        setErrorMsg("Enter a Key MA Level to calculate your global stop.");
        return;
      }
      const stop = ma * (1 - buf / 100);
      if (stop >= entry) {
        setErrorMsg(`Stop ($${stop.toFixed(2)}) is at or above current price ($${entry.toFixed(2)}).`);
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

  // ━━━ Normal Sizer Results ━━━
  const normalResults = useMemo(() => {
    if (!calculated || tab !== "normal") return null;
    const stop = ma * (1 - buf / 100);
    const riskPerShare = entry - stop;
    if (riskPerShare <= 0) return null;

    const riskShares = Math.ceil(riskBudget / riskPerShare);
    const targetPct = targetSize;
    const targetShares = Math.ceil((equity * targetPct / 100) / entry);
    const finalShares = Math.min(riskShares, targetShares);
    const finalVal = finalShares * entry;
    const finalPctNlv = equity > 0 ? (finalVal / equity) * 100 : 0;
    const limitFactor = finalShares === targetShares && targetShares < riskShares
      ? `Target Size (${targetPct}%)`
      : `MA Support ($${ma})`;

    return { riskShares, targetShares, finalShares, finalVal, finalPctNlv, limitFactor, stop, riskPerShare };
  }, [calculated, tab, ma, buf, entry, riskBudget, targetSize, equity]);

  // ━━━ Volatility Sizer Results ━━━
  const volResults = useMemo(() => {
    if (!calculated || tab !== "volatility") return null;

    let tierName: string;
    let tolPct: number;
    let atrMultiplier: number;
    let cushionPct = 0;
    const avgCost = volSizerMode === "audit" ? holdingAvgCost : entry;
    const shares = volSizerMode === "audit" ? (holdingData?.shares || 0) : 0;

    if (volSizerMode === "new") {
      if (sizingMode === 2) { tierName = "Offense Mode"; tolPct = 1.0; }
      else if (sizingMode === 1) { tierName = "Normal Mode"; tolPct = 0.75; }
      else { tierName = "Defense Mode"; tolPct = 0.5; }
      atrMultiplier = VOL_PROFILES[volProfile].mult;
    } else {
      cushionPct = avgCost > 0 ? ((entry - avgCost) / avgCost) * 100 : 0;
      if (cushionPct >= 20) { tierName = "Tier 1 (High Cushion)"; tolPct = 1.0; atrMultiplier = 2.0; }
      else if (cushionPct >= 5) { tierName = "Tier 2 (Moderate)"; tolPct = 0.65; atrMultiplier = 1.5; }
      else { tierName = "Tier 3 (Defense)"; tolPct = 0.5; atrMultiplier = 1.0; }
    }

    const dailyRiskBudget = equity * (tolPct / 100);
    const atrRiskBudget = dailyRiskBudget * atrMultiplier;
    const atrDecimal = atr / 100;
    const maxSharesVol = Math.ceil(atrRiskBudget / (entry * atrDecimal));

    // Tech Stop Limit
    let maxSharesTech = 999999;
    let effectiveStop = 0;
    let techDistPct = 0;
    if (volSizerMode === "new" && ma > 0) {
      effectiveStop = ma * (1 - buf / 100);
      if (effectiveStop < entry) {
        const rps = entry - effectiveStop;
        techDistPct = (rps / entry) * 100;
        maxSharesTech = Math.ceil(dailyRiskBudget / rps);
        if (targetSize > 0) {
          const targetCap = Math.ceil((equity * targetSize / 100) / entry);
          maxSharesTech = Math.min(maxSharesTech, targetCap);
        }
      }
    }

    // Hard Cap (20%)
    const maxSharesCap = Math.floor((equity * 0.20) / entry);

    // Target cap
    let maxSharesTarget = 999999;
    if (volSizerMode === "new" && targetSize > 0) {
      maxSharesTarget = Math.ceil((equity * targetSize / 100) / entry);
    }

    const finalMaxShares = Math.min(maxSharesVol, maxSharesTech, maxSharesCap, maxSharesTarget);
    const finalMaxVal = finalMaxShares * entry;

    // Limiting factor
    let limitReason = "Volatility (ATR)";
    if (finalMaxShares === maxSharesTarget && maxSharesTarget < Math.min(maxSharesVol, maxSharesTech, maxSharesCap)) {
      limitReason = `Target Size (${targetSize}%)`;
    } else if (finalMaxShares === maxSharesCap) {
      limitReason = "Hard Cap (20%)";
    } else if (finalMaxShares === maxSharesTech) {
      limitReason = `MA Support ($${ma})`;
    }

    // Trade Risk $
    let riskPerShare: number;
    let riskLabel: string;
    if (effectiveStop > 0 && effectiveStop < entry) {
      riskPerShare = entry - effectiveStop;
      riskLabel = `Stop $${effectiveStop.toFixed(2)} (${(riskPerShare / entry * 100).toFixed(1)}%)`;
    } else {
      riskPerShare = entry * atrDecimal;
      riskLabel = `1 ATR (${atr.toFixed(1)}%)`;
    }
    const finalRiskDol = finalMaxShares * riskPerShare;

    // ATR info
    const atrRiskAtVol = maxSharesVol * entry * atrDecimal;
    const atrCostPct = equity > 0 ? (maxSharesVol * entry / equity) * 100 : 0;

    // Tech Stop info
    const techRiskAtMax = effectiveStop > 0 ? maxSharesTech * (entry - effectiveStop) : 0;
    const techCostPct = equity > 0 ? (maxSharesTech * entry / equity) * 100 : 0;

    return {
      tierName, tolPct, atrMultiplier, dailyRiskBudget, atrRiskBudget,
      maxSharesVol, maxSharesTech, maxSharesCap, maxSharesTarget,
      finalMaxShares, finalMaxVal, limitReason,
      effectiveStop, techDistPct, riskPerShare, riskLabel, finalRiskDol,
      atrRiskAtVol, atrCostPct, techRiskAtMax, techCostPct,
      cushionPct, shares,
    };
  }, [calculated, tab, volSizerMode, entry, atr, ma, buf, sizingMode, volProfile, equity, targetSize, holdingData, holdingAvgCost]);

  // ━━━ Scale-In Results ━━━
  const scaleResults = useMemo(() => {
    if (!calculated || tab !== "scalein" || !holdingData) return null;
    const stop = ma * (1 - buf / 100);
    const riskPerShare = entry - stop;
    if (riskPerShare <= 0) return null;

    const currShares = holdingData.shares || 0;
    const avgEntry = holdingData.avg_entry || 0;
    const currValue = currShares * entry;

    const targetValue = equity * (targetSize / 100);
    const targetTotalShares = Math.ceil(targetValue / entry);
    const targetAdd = targetTotalShares - currShares;

    const maxRisk = SIZING_MODES[sizingMode].pct;
    const maxRiskDol = equity * (maxRisk / 100);
    const maxTotalShares = Math.ceil(maxRiskDol / riskPerShare);
    const affordableAdd = maxTotalShares - currShares;

    if (targetAdd <= 0) return { error: `You are already at or above the target weight! (Current: $${currValue.toLocaleString(undefined, { maximumFractionDigits: 0 })} vs Target: $${targetValue.toLocaleString(undefined, { maximumFractionDigits: 0 })})` };
    if (affordableAdd <= 0) {
      const riskAtCurr = currShares * riskPerShare;
      return { error: `NO ADD - Your current ${currShares} shares already risk $${riskAtCurr.toLocaleString(undefined, { maximumFractionDigits: 0 })} (budget: $${maxRiskDol.toLocaleString(undefined, { maximumFractionDigits: 0 })}). Tighten your stop or reduce position.` };
    }

    const recommendedAdd = Math.min(targetAdd, affordableAdd);
    const newTotal = currShares + recommendedAdd;
    const newAvgCost = newTotal > 0 ? (currShares * avgEntry + recommendedAdd * entry) / newTotal : 0;
    const costOfAdd = recommendedAdd * entry;
    const totalRiskAtNew = newTotal * riskPerShare;
    const newWeight = equity > 0 ? (newTotal * entry / equity) * 100 : 0;
    const verdict = affordableAdd >= targetAdd ? "success" : "partial";

    return {
      recommendedAdd, newTotal, newAvgCost, costOfAdd, totalRiskAtNew, newWeight,
      stop, riskPerShare, maxRiskDol, maxRisk, targetAdd, avgEntry, currShares, verdict,
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
      // Risk-based: show contracts at 1%, 2%, 3% risk tiers
      const tiers = [
        { label: "Conservative (1%)", pct: 1.0 },
        { label: "Normal (2%)", pct: 2.0 },
        { label: "Aggressive (3%)", pct: 3.0 },
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

  // --- Helpers ---
  const fmtDol = (v: number, digits = 0) => `$${v.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits })}`;

  const needsHolding = ["scalein", "pyramid", "trim"].includes(tab);
  const needsMaBuffer = ["normal", "scalein"].includes(tab) || (tab === "volatility" && volSizerMode === "new");
  const needsTarget = tab === "normal" || tab === "trim" || (tab === "volatility" && volSizerMode === "new") || tab === "scalein" || (tab === "options" && optMode === "equivalent");

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          Position <em className="italic" style={{ color: navColor }}>Sizer</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>CanSlim</div>
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
          {tab === "normal" && "Size positions based on a key support level with buffer. No ATR involved."}
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

      {/* Volatility Tier Rules Expander */}
      {tab === "volatility" && (
        <details className="mb-4 rounded-[10px] overflow-hidden" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
          <summary className="px-4 py-2.5 text-[12px] font-semibold cursor-pointer" style={{ color: "var(--ink-3)" }}>
            View Tier System Rules
          </summary>
          <div className="px-4 pb-3 text-[12px] leading-relaxed" style={{ color: "var(--ink-3)" }}>
            <p className="mb-1"><strong>Sizing Mode (New Trades):</strong></p>
            <ul className="list-disc ml-4 mb-2">
              <li><strong>Defense:</strong> 0.50% Risk — equity curve flat/down</li>
              <li><strong>Normal:</strong> 0.75% Risk — equity curve recovering</li>
              <li><strong>Offense:</strong> 1.00% Risk — equity curve strong, confirmed uptrend</li>
            </ul>
            <p className="mb-1"><strong>Stock Volatility Profile (New Trades):</strong></p>
            <ul className="list-disc ml-4 mb-2">
              <li><strong>Tight:</strong> 1.0x ATR — low-volatility, tight setups</li>
              <li><strong>Normal:</strong> 1.25x ATR — standard growth stocks</li>
              <li><strong>High-Vol:</strong> 1.5x ATR — high-volatility names</li>
            </ul>
            <p className="mb-1"><strong>Tolerance Tiers (Active Positions):</strong></p>
            <ul className="list-disc ml-4">
              <li><strong>Tier 1 (High Cushion):</strong> Profit &gt; 20% → 1.00% Risk, 2.0x ATR</li>
              <li><strong>Tier 2 (Moderate):</strong> Profit 5%-20% → 0.65% Risk, 1.5x ATR</li>
              <li><strong>Tier 3 (Defense):</strong> Profit &lt; 5% → 0.50% Risk, 1.0x ATR</li>
            </ul>
          </div>
        </details>
      )}

      {/* ═══════════ INPUTS ═══════════ */}
      <div className="flex flex-col gap-5 mb-6">

        {/* Volatility Sizer: Mode Toggle */}
        {tab === "volatility" && (
          <Field label="Sizing Context">
            <div className="flex gap-4 mt-1">
              <Radio checked={volSizerMode === "new"} onClick={() => { setVolSizerMode("new"); resetCalc(); }} label="🆕 New Trade" />
              <Radio checked={volSizerMode === "audit"} onClick={() => { setVolSizerMode("audit"); resetCalc(); }} label="🔍 Audit Active Position" />
            </div>
          </Field>
        )}

        {/* Ticker (new trade tabs) */}
        {(tab === "normal" || tab === "options" || (tab === "volatility" && volSizerMode === "new")) && tab !== "options" && (
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

        {/* Holding picker (scalein, pyramid, trim, or vol audit mode) */}
        {(needsHolding || (tab === "volatility" && volSizerMode === "audit")) && (
          <Field label="Select Position">
            <select value={selectedHolding} onChange={e => {
              setSelectedHolding(e.target.value);
              resetCalc();
              // Auto-fill price for holding
              const h = openTrades.find(t => t.trade_id === e.target.value);
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
                    className={inputCls} style={{ ...inputStyle, appearance: "none" as const }}>
              <option value="">Select...</option>
              {openTrades.map(t => (
                <option key={t.trade_id} value={t.trade_id}>{t.ticker} ({t.shares} shs) | {t.trade_id}</option>
              ))}
            </select>
            {holdingData && (
              <div className="mt-2 text-[12px] px-3 py-2 rounded-[8px]" style={{ background: "var(--bg)", color: "var(--ink-3)" }}>
                {holdingData.shares} shs @ ${parseFloat(String(holdingData.avg_entry || 0)).toFixed(2)}
                {holdingData.rule ? ` · ${holdingData.rule}` : ""}
                {tab === "volatility" && volSizerMode === "audit" && holdingAvgCost > 0 && (
                  <span className="ml-2">(LIFO Avg: ${holdingAvgCost.toFixed(2)})</span>
                )}
              </div>
            )}
          </Field>
        )}

        {/* Entry Price + Equity */}
        {tab !== "options" && (
          <div className="grid grid-cols-2 gap-4">
            <Field label={needsHolding || (tab === "volatility" && volSizerMode === "audit") ? "Current Price ($)" : "Entry Price ($)"}>
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

        {/* Calculated stop info banner */}
        {needsMaBuffer && calcStop > 0 && entry > 0 && (
          <Banner type="info">
            Calculated Stop: <strong>${calcStop.toFixed(2)}</strong> (MA ${ma.toFixed(2)} - {buf.toFixed(1)}% buffer) — {stopDistPct.toFixed(1)}% below entry
          </Banner>
        )}

        {/* ATR (volatility, pyramid) */}
        {(tab === "volatility" || tab === "pyramid") && (
          <Field label="ATR % (21-Day)">
            <input type="number" value={atrPct} onChange={e => { setAtrPct(e.target.value); resetCalc(); }}
                   step="0.1" placeholder="5.0" className={inputCls} style={inputStyle} />
          </Field>
        )}

        {/* Sizing Mode + Vol Profile (new trade tabs) */}
        {(tab === "normal" || (tab === "volatility" && volSizerMode === "new") || tab === "scalein" || tab === "options") && (
          <>
            <div className="px-4 py-2.5 rounded-[10px] text-[12px]" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
              <span style={{ color: "var(--ink-4)" }}>M Factor:</span>{" "}
              <strong>{mfSuggestion}</strong>
              <span style={{ color: "var(--ink-4)" }}> → suggesting </span>
              <strong>{SIZING_MODES[sizingMode].label}</strong>
            </div>
            <Field label="Sizing Mode">
              <div className="flex gap-4 mt-1">
                {SIZING_MODES.map((m, i) => (
                  <Radio key={m.key} checked={sizingMode === i} onClick={() => { setSizingMode(i); resetCalc(); }} label={m.label} />
                ))}
              </div>
            </Field>
          </>
        )}

        {/* Vol Profile (volatility new trade only) */}
        {tab === "volatility" && volSizerMode === "new" && (
          <Field label="Stock Volatility Profile">
            <div className="flex gap-4 mt-1">
              {VOL_PROFILES.map((p, i) => (
                <Radio key={p.key} checked={volProfile === i} onClick={() => { setVolProfile(i); resetCalc(); }} label={p.label} />
              ))}
            </div>
          </Field>
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
        {tab === "trim" ? "Calculate Trim Impact" : tab === "scalein" ? "Calculate Add-On" : tab === "pyramid" ? "Run Pyramid Analysis" : tab === "volatility" ? "Run Sizing Audit" : "Calculate Size"}
      </button>

      {/* ═══════════ RESULTS ═══════════ */}
      {calculated && !errorMsg && (
        <div style={{ animation: "slide-up 0.15s ease-out" }}>

          {/* ── NORMAL SIZER ── */}
          {tab === "normal" && normalResults && (
            <>
              <h3 className="text-[15px] font-semibold mb-4">Sizing Profile: {ticker || "—"}</h3>
              <div className="grid grid-cols-3 gap-3 mb-4">
                <MetricCard label="Risk Budget" value={fmtDol(riskBudget)}
                            sub={`${riskPct}% Risk (${SIZING_MODES[sizingMode].key.charAt(0).toUpperCase() + SIZING_MODES[sizingMode].key.slice(1)} Mode)`}
                            accent="#6366f1" />
                <MetricCard label="Stop Distance" value={`${stopDistPct.toFixed(1)}%`}
                            sub={`$${normalResults.riskPerShare.toFixed(2)}/share`}
                            accent="#f59f00" />
                <MetricCard label="Target Size" value={`${targetSize}%`}
                            sub={fmtDol(equity * targetSize / 100)}
                            accent="#3b82f6" />
              </div>
              <div className="grid grid-cols-3 gap-3 mb-6">
                <MetricCard label="Risk-Based Limit" value={`${normalResults.riskShares} shs`}
                            sub={`${fmtDol(normalResults.riskShares * entry)} (${(normalResults.riskShares * entry / equity * 100).toFixed(1)}% NLV)`} />
                <MetricCard label="Target Limit" value={`${normalResults.targetShares} shs`}
                            sub={`${fmtDol(normalResults.targetShares * entry)} (${targetSize}% NLV)`} />
                <MetricCard label="Limiting Factor" value={normalResults.limitFactor}
                            sub="Determines Final Size" />
              </div>

              <h3 className="text-[14px] font-semibold mb-2">The Verdict</h3>
              <Banner type="success">
                RECOMMENDED SIZE: Buy <strong>{normalResults.finalShares}</strong> shares ({normalResults.finalPctNlv.toFixed(1)}% of NLV).
              </Banner>

              <div className="mt-4">
                <button onClick={() => sendToLogBuy({ ticker, shares: normalResults.finalShares, price: entry, stop: normalResults.stop, action: "new" })}
                        className="w-full h-[48px] rounded-[12px] text-[13px] font-semibold transition-all hover:brightness-95 cursor-pointer"
                        style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }}>
                  📝 Send to Log Buy — {ticker || "—"} ({normalResults.finalShares} shs @ ${entry.toFixed(2)})
                </button>
              </div>
            </>
          )}

          {/* ── VOLATILITY SIZER ── */}
          {tab === "volatility" && volResults && (
            <>
              <h3 className="text-[15px] font-semibold mb-4">Sizing Profile: {volSizerMode === "audit" ? holdingData?.ticker : ticker || "—"}</h3>
              <div className="grid grid-cols-3 gap-3 mb-4">
                <MetricCard label="Risk Budget" value={fmtDol(volResults.dailyRiskBudget)}
                            sub={`${volResults.tolPct}% Risk (${volResults.tierName})`}
                            accent="#6366f1" />
                <MetricCard label="Volatility Risk" value={`${atr.toFixed(2)}%`}
                            sub="ATR (Noise)"
                            accent="#f59f00" />
                {volSizerMode === "new" ? (
                  <div className="p-4 rounded-[12px] relative overflow-hidden" style={{
                    border: "1px solid var(--border)",
                    borderLeft: "4px solid #3b82f6",
                    background: "color-mix(in oklab, #3b82f6 4%, var(--surface))",
                  }}>
                    <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>Buy Cost</div>
                    <div className="flex items-baseline justify-between mt-1.5">
                      <span className="text-[11px]" style={{ color: "var(--ink-4)" }}>ATR Limit</span>
                      <span className="text-[17px] font-semibold privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{fmtDol(volResults.maxSharesVol * entry)}</span>
                    </div>
                    <div className="flex items-baseline justify-between mt-1">
                      <span className="text-[11px]" style={{ color: "var(--ink-4)" }}>{volResults.effectiveStop > 0 ? "Tech Stop" : "Hard Cap"}</span>
                      <span className="text-[17px] font-semibold privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{fmtDol((volResults.effectiveStop > 0 ? volResults.maxSharesTech : volResults.maxSharesCap) * entry)}</span>
                    </div>
                  </div>
                ) : (
                  <MetricCard label="Profit Cushion" value={`${volResults.cushionPct.toFixed(2)}%`}
                              sub={volResults.tierName}
                              accent={volResults.cushionPct >= 20 ? "#08a86b" : volResults.cushionPct >= 5 ? "#f59f00" : "#e5484d"} />
                )}
              </div>

              {/* ATR Boost banner */}
              {volResults.atrMultiplier > 1.0 && (
                <div className="mb-4">
                  <Banner type="info">
                    {volSizerMode === "new"
                      ? `ATR Boost: ATR budget scaled ${volResults.atrMultiplier.toFixed(1)}x (${fmtDol(volResults.atrRiskBudget)}) — stock volatility profile`
                      : `Confidence Boost: ATR budget scaled ${volResults.atrMultiplier.toFixed(1)}x (${fmtDol(volResults.atrRiskBudget)}) — earned by ${volResults.cushionPct.toFixed(1)}% profit cushion`
                    }
                  </Banner>
                </div>
              )}

              <div className="grid grid-cols-3 gap-3 mb-6">
                <MetricCard label="ATR Limit" value={`${volResults.maxSharesVol} shs`}
                            sub={`Risk ${fmtDol(volResults.atrRiskAtVol)} · ${volResults.atrCostPct.toFixed(1)}% NLV`} />
                {volSizerMode === "new" && volResults.effectiveStop > 0 ? (
                  <MetricCard label="Tech Stop Limit" value={`${volResults.maxSharesTech} shs`}
                              sub={`Risk ${fmtDol(volResults.techRiskAtMax)} · ${volResults.techCostPct.toFixed(1)}% NLV`}
                              accent={volResults.maxSharesTech < volResults.maxSharesVol ? "#f59f00" : undefined} />
                ) : (
                  <MetricCard label="Hard Cap Limit" value={`${volResults.maxSharesCap} shs`}
                              sub="20% Max Alloc" />
                )}
                <MetricCard label="Trade Risk $" value={fmtDol(volResults.finalRiskDol)}
                            sub={volResults.riskLabel} />
              </div>

              <h3 className="text-[14px] font-semibold mb-2">The Verdict</h3>

              {volSizerMode === "new" ? (
                <>
                  <Banner type="success">
                    RECOMMENDED SIZE: Buy <strong>{volResults.finalMaxShares}</strong> shares ({(volResults.finalMaxVal / equity * 100).toFixed(1)}% of NLV).
                  </Banner>
                  {volResults.limitReason.startsWith("MA") && (
                    <div className="mt-2">
                      <Banner type="info">
                        Note: Sized for technicals. Your stop (${volResults.effectiveStop.toFixed(2)}) is {volResults.techDistPct.toFixed(1)}% away (including buffer).
                      </Banner>
                    </div>
                  )}
                  <div className="mt-4">
                    <button onClick={() => sendToLogBuy({ ticker, shares: volResults.finalMaxShares, price: entry, stop: volResults.effectiveStop || entry * 0.92, action: "new" })}
                            className="w-full h-[48px] rounded-[12px] text-[13px] font-semibold transition-all hover:brightness-95 cursor-pointer"
                            style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }}>
                      📝 Send to Log Buy — {ticker || "—"} ({volResults.finalMaxShares} shs @ ${entry.toFixed(2)})
                    </button>
                  </div>
                </>
              ) : (
                <>
                  {/* Audit mode: Start/Target/Action */}
                  <div className="grid grid-cols-3 gap-3 mb-4">
                    <MetricCard label="Start Position" value={`${Math.floor(volResults.shares)} shs`}
                                sub={`${(volResults.shares * entry / equity * 100).toFixed(1)}% Weight`} />
                    <MetricCard label="Target Position" value={`${volResults.finalMaxShares} shs`}
                                sub={`${(volResults.finalMaxVal / equity * 100).toFixed(1)}% Weight`} />
                    {(() => {
                      const diff = volResults.shares - volResults.finalMaxShares;
                      if (diff > 0) {
                        return <MetricCard label="Action Required" value={`TRIM ${Math.floor(diff)}`}
                                           sub={`Sell ${fmtDol(diff * entry)}`} color="#e5484d" accent="#e5484d" />;
                      } else if (diff < 0) {
                        return <MetricCard label="Action Required" value={`CAN ADD ${Math.abs(Math.floor(diff))}`}
                                           sub={`Buy up to ${fmtDol(Math.abs(diff) * entry)}`} color="#08a86b" accent="#08a86b" />;
                      } else {
                        return <MetricCard label="Action Required" value="AT LIMIT" sub="No room to add" />;
                      }
                    })()}
                  </div>
                  {(() => {
                    const diff = volResults.shares - volResults.finalMaxShares;
                    if (diff > 0) {
                      return <Banner type="warning">OVERWEIGHT: You are holding {Math.floor(diff)} shares too many for this volatility/technical profile.</Banner>;
                    } else if (diff < 0) {
                      const add = Math.abs(Math.floor(diff));
                      return <Banner type="success">Room to add up to {add} shares ({(add * entry / equity * 100).toFixed(1)}% of NLV) within limits.</Banner>;
                    } else {
                      return <Banner type="info">Position is exactly at the {volResults.finalMaxShares} share limit.</Banner>;
                    }
                  })()}
                </>
              )}
            </>
          )}

          {/* ── SCALE IN ── */}
          {tab === "scalein" && scaleResults && (
            <>
              {"error" in scaleResults ? (
                <Banner type="error">{scaleResults.error}</Banner>
              ) : (
                <>
                  <h3 className="text-[15px] font-semibold mb-4">PYRAMID TICKET</h3>
                  <div className="grid grid-cols-4 gap-3 mb-4">
                    <MetricCard label="ADD SHARES" value={`+${scaleResults.recommendedAdd}`}
                                accent="#08a86b" color="#08a86b" />
                    <MetricCard label="EST. COST" value={fmtDol(scaleResults.costOfAdd, 2)}
                                accent="#6366f1" />
                    <MetricCard label="NEW TOTAL" value={`${scaleResults.newTotal} shs`}
                                sub={`${scaleResults.newWeight.toFixed(1)}% Weight`}
                                accent="#3b82f6" />
                    <MetricCard label="NEW AVG COST" value={`$${scaleResults.newAvgCost.toFixed(2)}`}
                                sub={`From $${scaleResults.avgEntry.toFixed(2)}`}
                                accent="#f59f00" />
                  </div>

                  <h3 className="text-[15px] font-semibold mb-4">RISK MANAGEMENT</h3>
                  <div className="grid grid-cols-3 gap-3 mb-6">
                    <MetricCard label="Global Stop" value={`$${scaleResults.stop.toFixed(2)}`}
                                sub={`-${(scaleResults.riskPerShare / entry * 100).toFixed(1)}% from price`}
                                accent="#e5484d" />
                    <MetricCard label="Total Risk at New Size" value={fmtDol(scaleResults.totalRiskAtNew)}
                                sub={`${(scaleResults.totalRiskAtNew / equity * 100).toFixed(2)}% of NLV`}
                                accent="#f59f00" />
                    <MetricCard label="Risk Budget" value={fmtDol(scaleResults.maxRiskDol)}
                                sub={`${scaleResults.maxRisk}% of Equity`}
                                accent="#6366f1" />
                  </div>

                  <h3 className="text-[14px] font-semibold mb-2">The Verdict</h3>
                  {scaleResults.verdict === "success" ? (
                    <Banner type="success">
                      ADD {scaleResults.recommendedAdd} shares to reach {scaleResults.newWeight.toFixed(1)}% — Total risk {fmtDol(scaleResults.totalRiskAtNew)} within {fmtDol(scaleResults.maxRiskDol)} budget.
                    </Banner>
                  ) : (
                    <Banner type="warning">
                      RISK LIMIT: Full target ({scaleResults.targetAdd} shares) would exceed budget. Safe add: {scaleResults.recommendedAdd} shares ({scaleResults.newWeight.toFixed(1)}% weight). Scale up on next pullback to MA.
                    </Banner>
                  )}

                  <div className="mt-4">
                    <button onClick={() => sendToLogBuy({ ticker: holdingData?.ticker || "", shares: scaleResults.recommendedAdd, price: entry, stop: scaleResults.stop, trade_id: holdingData?.trade_id, action: "scale_in" })}
                            className="w-full h-[48px] rounded-[12px] text-[13px] font-semibold transition-all hover:brightness-95 cursor-pointer"
                            style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }}>
                      📝 Send to Log Buy — {holdingData?.ticker} (+{scaleResults.recommendedAdd} shs @ ${entry.toFixed(2)})
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
                <MetricCard label="Last Buy Price" value={`$${pyramidResults.lastBuyPrice.toFixed(2)}`}
                            accent="#3b82f6" />
                <MetricCard label="Last Buy P&L" value={`${pyramidResults.lastBuyProfitPct.toFixed(2)}%`}
                            sub={`$${(entry - pyramidResults.lastBuyPrice).toFixed(2)}/share`}
                            color={pyramidResults.lastBuyProfitPct >= 0 ? "#08a86b" : "#e5484d"}
                            accent={pyramidResults.lastBuyProfitPct >= 0 ? "#08a86b" : "#e5484d"} />
                <MetricCard label="Total Cushion" value={`${pyramidResults.cushionPct.toFixed(2)}%`}
                            sub={`Avg Cost: $${pyramidResults.avgCost.toFixed(2)}`}
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
                    ADD {pyramidResults.pyramidAllowed} shares ({fmtDol(pyramidResults.pyramidValue)}) — Limited by: {pyramidResults.pyramidAllowed === pyramidResults.pyramidMaxShares ? "Pyramid pace" : "ATR/Cap ceiling"}
                  </Banner>
                  <div className="grid grid-cols-3 gap-3 mt-4">
                    <MetricCard label="Add Shares" value={`${pyramidResults.pyramidAllowed} shs`}
                                sub={fmtDol(pyramidResults.pyramidValue)} accent="#08a86b" />
                    <MetricCard label="New Total" value={`${Math.floor(pyramidResults.shares) + pyramidResults.pyramidAllowed} shs`}
                                sub={`${((Math.floor(pyramidResults.shares) + pyramidResults.pyramidAllowed) * entry / equity * 100).toFixed(1)}% Weight`} />
                    <MetricCard label="New Avg Cost" value={`$${((pyramidResults.avgCost * pyramidResults.shares + entry * pyramidResults.pyramidAllowed) / (pyramidResults.shares + pyramidResults.pyramidAllowed)).toFixed(2)}`}
                                sub={`From $${pyramidResults.avgCost.toFixed(2)}`} />
                  </div>
                  <div className="mt-4">
                    <button onClick={() => sendToLogBuy({ ticker: holdingData?.ticker || "", shares: pyramidResults.pyramidAllowed, price: entry, trade_id: holdingData?.trade_id, action: "scale_in" })}
                            className="w-full h-[48px] rounded-[12px] text-[13px] font-semibold transition-all hover:brightness-95 cursor-pointer"
                            style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }}>
                      📝 Send to Log Buy — {holdingData?.ticker} (+{pyramidResults.pyramidAllowed} shs @ ${entry.toFixed(2)})
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
                    <MetricCard label="Cash Generated" value={fmtDol(trimResults.cashGenerated, 2)}
                                accent="#08a86b" />
                    <MetricCard label="Cost Basis (Sold)" value={fmtDol(trimResults.costBasisTrimmed, 2)}
                                sub={`Avg: $${trimResults.avgCostSold.toFixed(2)}/sh`}
                                accent="#6366f1" />
                    <MetricCard label="Realized P&L" value={fmtDol(trimResults.lifoPnl, 2)}
                                sub={trimResults.costBasisTrimmed > 0 ? `${(trimResults.lifoPnl / trimResults.costBasisTrimmed * 100).toFixed(2)}% Return` : undefined}
                                color={trimResults.lifoPnl >= 0 ? "#08a86b" : "#e5484d"}
                                accent={trimResults.lifoPnl >= 0 ? "#08a86b" : "#e5484d"} />
                  </div>

                  <h3 className="text-[14px] font-semibold mb-2">The Verdict</h3>
                  {trimResults.lifoPnl >= 0 ? (
                    <Banner type="success">
                      Profit Lock: This trim locks in {fmtDol(trimResults.lifoPnl, 2)} profit.
                    </Banner>
                  ) : (
                    <Banner type="warning">
                      Note: This trim realizes a loss of {fmtDol(Math.abs(trimResults.lifoPnl), 2)} based on your most recent purchases (LIFO).
                    </Banner>
                  )}

                  {trimResults.sharesToSell > 0 && (
                    <div className="mt-4">
                      <button onClick={() => sendToLogSell({ ticker: holdingData?.ticker || "", shares: trimResults.sharesToSell, price: entry, trade_id: holdingData?.trade_id })}
                              className="w-full h-[48px] rounded-[12px] text-[13px] font-semibold transition-all hover:brightness-95 cursor-pointer"
                              style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }}>
                        📝 Send to Log Sell — {holdingData?.ticker} ({trimResults.sharesToSell} shs @ ${entry.toFixed(2)})
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
                    <MetricCard label="Selected Risk Budget" value={fmtDol(optResults.recBudget)}
                                sub={`${SIZING_MODES[sizingMode].pct}% of equity (${SIZING_MODES[sizingMode].label.split(" ")[0]})`}
                                accent="#6366f1" />
                    <MetricCard label="Cost per Contract" value={fmtDol(optResults.cpc)}
                                sub={`$${costPerContract} x 100 shares`}
                                accent="#f59f00" />
                    <MetricCard label="Recommended" value={`${optResults.recContracts} contract${optResults.recContracts !== 1 ? "s" : ""}`}
                                sub={`${fmtDol(optResults.recTotal)} (${optResults.recPct.toFixed(1)}% NLV) · ${optResults.recLimiting}`}
                                color={navColor} accent="#08a86b" />
                  </div>

                  {optResults.recContracts === 0 && (
                    <div className="mb-4">
                      <Banner type="warning">
                        A single contract ({fmtDol(optResults.cpc)}) exceeds your risk budget ({fmtDol(optResults.recBudget)}). Consider a cheaper strike or spread.
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
                            <td className="px-3 py-2.5 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{fmtDol(r.budget)}</td>
                            <td className="px-3 py-2.5 font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{r.contracts}</td>
                            <td className="px-3 py-2.5 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{fmtDol(r.totalCost)}</td>
                            <td className="px-3 py-2.5" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{r.pctNlv.toFixed(2)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>
                    Hard cap: 5% of NLV ({fmtDol(optResults.hardCapBudget)}) — no tier can exceed this.
                  </div>
                </>
              ) : (
                <>
                  <h3 className="text-[15px] font-semibold mb-2">Position Equivalent — {ticker || "—"}</h3>
                  <div className="text-[12px] mb-4" style={{ color: "var(--ink-4)" }}>
                    How many option contracts replicate stock exposure at each position size tier.
                  </div>

                  <div className="grid grid-cols-3 gap-3 mb-4">
                    <MetricCard label="Stock Price" value={`$${(optResults.price || 0).toFixed(2)}`} sub={ticker || "—"} accent="#6366f1" />
                    <MetricCard label="Cost per Contract" value={fmtDol(optResults.cpc)} sub={`$${costPerContract} x 100 shares`} accent="#f59f00" />
                    <MetricCard label="Account Equity" value={fmtDol(equity)} accent="#08a86b" />
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
                            <td className="px-3 py-2.5 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{fmtDol(t.positionValue)}</td>
                            <td className="px-3 py-2.5" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{t.sharesEquiv}</td>
                            <td className="px-3 py-2.5 font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{t.contracts}</td>
                            <td className="px-3 py-2.5 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{fmtDol(t.totalCost)}</td>
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
                        At <strong>{targetSize}%</strong> target: Buy <strong>{sel.contracts} contract{sel.contracts !== 1 ? "s" : ""}</strong> ({sel.sharesEquiv} share equivalent) for {fmtDol(sel.totalCost)} ({sel.pctNlv.toFixed(1)}% of NLV).
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
