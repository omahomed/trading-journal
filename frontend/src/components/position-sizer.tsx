"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import { api, getActivePortfolio, type TradePosition, type TradeDetail } from "@/lib/api";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";
import { SearchSelect } from "./search-select";

type SizerTab = "volatility" | "scalein" | "pyramid" | "trim" | "options";

const TABS: { key: SizerTab; label: string; icon: string }[] = [
  { key: "volatility", label: "New Entry", icon: "⚖️" },
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
  SIZING_MODES_DISPLAY,
  mctStateToSizingMode,
  deriveAutoSizingMode,
  exitLadderFloor,
  describeMctSource,
  type ExitAlert,
  type SizingModeIndex,
} from "@/lib/sizing-mode";
import { computeVolatilitySizing, type VolSizerResults, type ScaleOutStops, SCALE_OUT_ATR_MULTIPLIERS } from "@/lib/vol-sizer";
import { computePyramidSizing, type PyramidSizerResults, PYRAMID_ADD_CAP_PCT, PYRAMID_CAMPAIGN_CEILING_PCT, PYRAMID_FULL_SIZE_TRIGGER_PCT } from "@/lib/pyramid-sizer";

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
  { label: "Shotgun (2.5%)", pct: 2.5 }, { label: "Half (5%)", pct: 5 },
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
// InventoryLot carries the source BUY row's stop_loss so the Pyramid
// tab can compute per-lot risk-to-stop without a second lookup. Field
// stays undefined for legacy callers that don't need it — the
// pyramid path fills a zero fallback (treated as "no stop set").
interface InventoryLot { qty: number; price: number; stopLoss: number; label?: string }

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
      const stopLoss = Number((tx as any).stop_loss ?? 0) || 0;
      const label = String((tx as any).trx_id ?? "") || undefined;
      inventory.push({ qty: shares, price, stopLoss, label });
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
    stopMode?: "price" | "atr" | "ladder";
    atrMultiplier?: 1 | 1.5 | 2;
    ladderShares?: [number, number, number];
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
  // (mctStateToSizingMode falls back to safe middle ground). Index 3
  // (Pilot, 0.25%) is reachable ONLY via manual radio click; the auto
  // path returns 0|1|2 only.
  // Includes 3 (Max, 1.00%) — a manual-only conviction upshift. Auto-
  // pick from MCT state still returns 0/1/2 only.
  const [sizingMode, setSizingMode] = useState<SizingModeIndex>(1);
  // mctState + sizingModeManual track WHY the current mode is what it is.
  // - sizingModeManual=false → set by MCT state read (auto)
  // - sizingModeManual=true  → user clicked a Radio (override). Reset by
  //   the "Reset to auto" button, which re-applies the MCT mapping.
  const [mctState, setMctState] = useState<string | null>(null);
  // Active exit-ladder alerts. Drive the sizing-mode floor — a fired
  // 21 EMA Violation / Confirmed Break downshifts to Normal; a fired
  // 50 SMA Violation downshifts to Pilot, regardless of what the
  // M Factor state alone would have picked. See lib/sizing-mode#
  // exitLadderFloor for the full rule.
  const [activeExits, setActiveExits] = useState<readonly { signal: string; severity?: string }[]>([]);
  const [sizingModeManual, setSizingModeManual] = useState(false);
  const [entryPrice, setEntryPrice] = useState("");
  // maLevel + buffer are used by the LEGACY MA-tech-stop path
  // (Scale-In tab). The Volatility tab moved to the composite-stop
  // model — user types a single Key Level and the sizer applies its
  // own buffer of max(0.5 ATR, 1%). See @/lib/vol-sizer.
  const [maLevel, setMaLevel] = useState("");
  const [buffer, setBuffer] = useState("1.00");
  // Volatility tab inputs (composite-stop model).
  const [keyLevelStr, setKeyLevelStr] = useState("");
  const [youngIpo, setYoungIpo] = useState(false);
  const [atrPct, setAtrPct] = useState("5.0");
  // Live MA levels from /api/prices/lookup — read-only display cells
  // on the Volatility tab, each with a "Use as Key Level" one-click
  // paste into keyLevelStr. Null when the ticker has < 21 / < 50 bars.
  const [ema21, setEma21] = useState<number | null>(null);
  const [sma50, setSma50] = useState<number | null>(null);
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

  // Auto-fetch price + ATR + 21 EMA + 50 SMA when ticker changes (debounced)
  useEffect(() => {
    if (!ticker || ticker.length < 1) return;
    const timeout = setTimeout(() => {
      setFetching(true);
      api.priceLookup(ticker).then(data => {
        if (data && !("error" in data)) {
          setEntryPrice(String(data.price));
          setAtrPct(String(data.atr_pct));
          setEma21(data.ema_21);
          setSma50(data.sma_50);
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
      // exit-ladder floor). E.g. POWERTREND + 50 SMA Violation → Pilot,
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
  // Volatility tab uses Key Level (composite-stop model).
  const keyLevel = parseFloat(keyLevelStr) || 0;

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
      if (keyLevel <= 0) {
        setErrorMsg("Please enter a Key Level — structural low or key MA from the chart. The composite stop needs it.");
        return;
      }
      // Composite validation is done inside the vol-sizer lib; a
      // Key Level that's degenerately high just makes the ATR floor
      // win the MIN — no error here.
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
      if (keyLevel <= 0) {
        setErrorMsg("Please enter a Key Level — anchor for the new add's composite stop.");
        return;
      }
      if (!(ema21 && ema21 > 0)) {
        setErrorMsg("21 EMA unavailable — cannot evaluate location gate (rule 1).");
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
        atrPct: atr,
        keyLevel,
        tolPct: SIZING_MODES[sizingMode].pct,
        youngIpo,
      });
    } catch (err) {
      log.error("position-sizer", "vol-sizer compute failed", err);
      return null;
    }
  }, [calculated, tab, entry, atr, keyLevel, sizingMode, equity, youngIpo]);

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
  // Delegates to computePyramidSizing (six-rule model): rules 1-6 with
  // per-lot risk accounting sourced from InventoryLot.stopLoss. The
  // legacy percent-of-shares model was retired 2026-07-18. See
  // @/lib/pyramid-sizer for the full formula.
  const pyramidResults: PyramidSizerResults | null = useMemo(() => {
    if (!calculated || tab !== "pyramid" || !holdingData || holdingInventory.length === 0) return null;
    if (!(ema21 && ema21 > 0)) return null;
    if (keyLevel <= 0) return null;

    const heldLots = holdingInventory.map((l) => ({
      shares: l.qty,
      entry: l.price,
      stopLoss: l.stopLoss,
      label: l.label,
    }));
    // Last held BUY (chronologically) — the reference for rule 2. LIFO
    // inventory is date-ascending, so inventory[end] is the newest.
    const lastHeldBuyPrice = holdingInventory[holdingInventory.length - 1]?.price ?? 0;

    try {
      return computePyramidSizing({
        equity,
        entry,
        atrPct: atr,
        ema21,
        keyLevel,
        tolPct: SIZING_MODES[sizingMode].pct,
        heldLots,
        currentPrice: entry, // Position Sizer uses `entry` as current
        lastHeldBuyPrice,
      });
    } catch (err) {
      log.error("position-sizer", "pyramid-sizer compute failed", err);
      return null;
    }
  }, [calculated, tab, holdingData, holdingInventory, entry, atr, equity, ema21, keyLevel, sizingMode]);

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
      const recIdx = sizingMode; // 0=pilot, 1=normal, 2=offense
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
  // Volatility tab moved to composite-stop model (Key Level input); the
  // MA + Buffer pair is now only relevant for Scale-In.
  const needsMaBuffer = tab === "scalein";
  // Pyramid tab also uses Key Level + the 21EMA/50SMA Use → shortcuts;
  // its composite-stop calc is imported from the same vol-sizer helper.
  const needsKeyLevel = tab === "volatility" || tab === "pyramid";
  // Volatility tab derives its ceiling from policy (15% / 5% young-IPO),
  // no user-picked target from the ladder — so it's excluded here.
  const needsTarget = tab === "trim" || tab === "scalein" || (tab === "options" && optMode === "equivalent");

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
        {tab === "volatility" ? (
          // Principle statement for the risk-first sizing model — echoes
          // the "Position Sizer" header's serif-italic + accent treatment.
          // Formula documentation lives in "View Sizer Rules" below.
          <div
            className="mt-3 mb-1 text-center italic text-[16px] leading-relaxed"
            style={{
              fontFamily: "var(--font-fraunces), Georgia, serif",
              color: "var(--ink-2)",
              letterSpacing: "0.01em",
            }}
          >
            Position weight is the output of the{" "}
            <span style={{ color: navColor, fontWeight: 500 }}>stop</span>,
            never the input of conviction.
          </div>
        ) : (
          <div className="text-[13px] mt-1" style={{ color: "var(--ink-4)" }}>
            {tab === "scalein" && "Scale up to target weight while respecting global stop and risk budget."}
            {tab === "pyramid" && "Per-lot risk-accounted add sizing. Gated by 4 rules: location (≤ 21EMA + 1 ATR), progress (last buy up ≥ 5%), budget (mode% × NAV − Σ lot risks), and 25% NAV campaign ceiling."}
            {tab === "trim" && "Calculate shares to sell to reach a desired weight, with LIFO P&L estimation."}
            {tab === "options" && "Size option positions using risk budget. Premium = max risk."}
          </div>
        )}
      </div>

      {/* Pyramid Rules Expander — six-rule composite model. */}
      {tab === "pyramid" && (
        <details className="mb-4 rounded-[10px] overflow-hidden" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
          <summary className="px-4 py-2.5 text-[12px] font-semibold cursor-pointer" style={{ color: "var(--ink-3)" }}>
            View Pyramid Rules
          </summary>
          <div className="px-4 pb-3 text-[12px] leading-relaxed" style={{ color: "var(--ink-3)" }}>
            <p className="mb-1"><strong>Four gates + sizing math (evaluated in order):</strong></p>
            <ol className="list-decimal ml-4 mb-2 flex flex-col gap-0.5">
              <li><strong>Location:</strong> current price ≤ 21 EMA + 1 × ATR$/share. Extended above the line → no add.</li>
              <li><strong>Progress:</strong> last held BUY up ≥ <strong>{PYRAMID_FULL_SIZE_TRIGGER_PCT}%</strong> for full-size multiplier; 0–{PYRAMID_FULL_SIZE_TRIGGER_PCT}% prorated; below last buy → no add.</li>
              <li><strong>Budget:</strong> campaign budget = Mode% × NAV (Pilot 0.25 / Normal 0.50 / Offense 0.75 / Max 1.00). Campaign risk = Σ (held shares × max(0, entry − stop)). Headroom = budget − risk. Zero headroom → no add.</li>
              <li><strong>Ceiling:</strong> (existing + new) × current price ≤ <strong>{PYRAMID_CAMPAIGN_CEILING_PCT}%</strong> NAV. Beyond that, appreciation is telling you to trim, not add.</li>
            </ol>
            <p className="mb-1"><strong>Sizing:</strong></p>
            <ul className="list-disc ml-4 mb-2">
              <li>Composite stop = MIN(Entry − 1 ATR, Key Level − max(0.5 ATR, 1%) of Key Level) — same shape as the Volatility Sizer.</li>
              <li>risk_bound_shares = headroom ÷ stop_distance</li>
              <li>notional_cap_shares = <strong>{PYRAMID_ADD_CAP_PCT}%</strong> NAV ÷ Entry (per-add cap)</li>
              <li>final_shares = min(risk_bound, notional_cap) × progress_multiplier, then clipped by the 25% ceiling.</li>
            </ul>
            <p className="mb-1"><strong>Per-lot risk accounting:</strong> each held BUY row's stored stop_loss drives its risk contribution. A lot with stop ≥ its cost basis reads as risk-free and releases full headroom for new adds. Trailing stops on winners auto-compound the budget.</p>
            <p className="mb-1"><strong>Broker setup:</strong> every filled add carries its own trailing stop (21 EMA − 0.5 ATR, rising only). The output card includes a pinned callout with the exact stop price to set at your broker — the sizer's per-lot accounting only stays honest if the trailing stops are actually placed.</p>
          </div>
        </details>
      )}

      {/* Volatility Sizer Rules Expander — composite-stop model. */}
      {tab === "volatility" && (
        <details className="mb-4 rounded-[10px] overflow-hidden" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
          <summary className="px-4 py-2.5 text-[12px] font-semibold cursor-pointer" style={{ color: "var(--ink-3)" }}>
            View Sizer Rules
          </summary>
          <div className="px-4 pb-3 text-[12px] leading-relaxed" style={{ color: "var(--ink-3)" }}>
            <p className="mb-1"><strong>The formula, four steps:</strong></p>
            <ol className="list-decimal ml-4 mb-2">
              <li><strong>Risk Budget ($)</strong> = NLV × Mode% (Pilot 0.25 / Normal 0.50 / Offense 0.75, from M Factor)</li>
              <li><strong>Composite Stop</strong> = LOWEST of:
                <ul className="list-disc ml-4">
                  <li>Entry − 1 ATR21 (the ATR floor — never sizes tighter than 1 ATR)</li>
                  <li>Key Level − max(0.5 ATR, 1%) of Key Level (structural low or key MA you typed from the chart)</li>
                </ul>
              </li>
              <li><strong>Raw shares</strong> = Risk Budget ÷ (Entry − Composite Stop)</li>
              <li><strong>Final shares</strong> = min(Raw, Ceiling × NLV ÷ Entry)</li>
            </ol>
            <p className="mb-1"><strong>Sizing Mode (auto from M Factor):</strong></p>
            <ul className="list-disc ml-4 mb-2">
              <li><strong>Pilot 0.25%</strong> — CORRECTION / RALLY MODE / UPTREND UNDER PRESSURE (probe)</li>
              <li><strong>Normal 0.50%</strong> — UPTREND</li>
              <li><strong>Offense 0.75%</strong> — POWERTREND</li>
              <li>Manual override is downward-only. Unknown state → Pilot.</li>
            </ul>
            <p className="mb-1"><strong>Ceiling policy (no ladder to pick):</strong></p>
            <ul className="list-disc ml-4 mb-2">
              <li><strong>15%</strong> standard — the everyday cap for full-conviction entries</li>
              <li><strong>5%</strong> young-IPO clamp — check the box when the name is ≤12 months since IPO</li>
              <li><strong>20%</strong> is the documented hard max — reserved for manual Log Buy overrides on high-conviction plays, not auto-selectable here</li>
            </ul>
            <p className="mb-1"><strong>Which candidate wins the composite?</strong> The LOWEST price = the WIDEST stop = the most defensive placement. Whichever candidate sits further from entry wins; the trade gets the room the wider candidate says it needs. Trade-off is a smaller position — the intentionally-conservative tilt.</p>
            <p className="mb-1 mt-2"><strong>Which constraint binds?</strong></p>
            <ul className="list-disc ml-4 mb-2">
              <li><strong>Risk-bound:</strong> the composite stop drives share count. Most trades in Normal / Pilot mode.</li>
              <li><strong>Ceiling-bound:</strong> the 15% (or 5%) cap drives share count. Reachable only in Offense mode on calm names — a market-timing discipline hiding inside a sizing formula.</li>
            </ul>
            <p className="mb-1"><strong>Scale-Out Stops (3-leg ladder, B1 lots):</strong></p>
            <ul className="list-disc ml-4">
              <li>Legs at Entry − 0.5 / 1.0 / 1.5 ATR (shares split floor / floor / remainder)</li>
              <li>Average exit = 1 ATR below entry — matches the risk budget when composite = 1 ATR</li>
              <li>Under a Key-Level-wins composite, tier 2 and 3 sit BELOW the composite. Intentional: "if it breaches your structural stop, give it more room, not less"</li>
              <li>Sends into Log Buy as a laddered stop</li>
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
                      // Pyramid tab needs 21 EMA + 50 SMA on holding
                      // pick — otherwise rule 1 (location) can't fire
                      // and the Use → cells never render.
                      setEma21(data.ema_21);
                      setSma50(data.sma_50);
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

        {/* MA Level + Buffer — Scale-In tab only. Volatility tab moved
            to the composite-stop model (Key Level input below). */}
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

        {/* Scale-In calculated-stop info banner. */}
        {needsMaBuffer && calcStop > 0 && entry > 0 && (
          <Banner type="info">
            Calculated Stop: <strong>{formatCurrency(calcStop)}</strong> (MA ${ma.toFixed(2)} - {buf.toFixed(1)}% buffer) — {stopDistPct.toFixed(1)}% below entry
          </Banner>
        )}

        {/* Key Level + Young-IPO clamp — Volatility tab only. Key Level
            is the user-typed structural low OR key MA from the chart;
            the sizer applies its own buffer of max(0.5 ATR, 1%) and
            composites against the 1-ATR floor. Young-IPO checkbox
            clamps the ceiling to 5% (default 15%). */}
        {needsKeyLevel && (
          <>
            {/* Live 21 EMA / 50 SMA display cells (populated by
                priceLookup). Each has a "Use as Key Level" button that
                pastes the value into keyLevelStr in one click — removes
                the tab-to-chart step. Cells hide entirely when the
                ticker has insufficient history (null from backend). */}
            {(ema21 !== null || sma50 !== null) && (
              <div className="grid grid-cols-2 gap-4">
                {ema21 !== null && (
                  <div className="p-3 rounded-[10px] flex items-center justify-between"
                       data-testid="ema21-cell"
                       style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
                    <div>
                      <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>21 EMA</div>
                      <div className="text-[16px] font-semibold privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                        {formatCurrency(ema21)}
                      </div>
                    </div>
                    <button type="button" onClick={() => { setKeyLevelStr(String(ema21)); resetCalc(); }}
                            data-testid="use-ema21-btn"
                            className="text-[11px] px-2.5 py-1 rounded-[6px] transition-all hover:brightness-95 cursor-pointer"
                            style={{ background: navColor, color: "#fff", border: "none" }}>
                      Use →
                    </button>
                  </div>
                )}
                {sma50 !== null && (
                  <div className="p-3 rounded-[10px] flex items-center justify-between"
                       data-testid="sma50-cell"
                       style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
                    <div>
                      <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>50 SMA</div>
                      <div className="text-[16px] font-semibold privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                        {formatCurrency(sma50)}
                      </div>
                    </div>
                    <button type="button" onClick={() => { setKeyLevelStr(String(sma50)); resetCalc(); }}
                            data-testid="use-sma50-btn"
                            className="text-[11px] px-2.5 py-1 rounded-[6px] transition-all hover:brightness-95 cursor-pointer"
                            style={{ background: navColor, color: "#fff", border: "none" }}>
                      Use →
                    </button>
                  </div>
                )}
              </div>
            )}
            <Field label="Key Level ($)">
              <input type="number" value={keyLevelStr} onChange={e => { setKeyLevelStr(e.target.value); resetCalc(); }}
                     step="0.01" placeholder="e.g. structural low or 21 EMA" className={inputCls} style={inputStyle}
                     data-testid="key-level-input" />
            </Field>
            {/* Young IPO clamp — Volatility only. Pyramid tab has its own
                fixed 5% add cap + 25% campaign ceiling from policy. */}
            {tab === "volatility" && (
              <div className="flex items-center gap-2.5 text-[12px]" style={{ color: "var(--ink-3)" }}>
                <input type="checkbox" id="young-ipo-checkbox" checked={youngIpo}
                       onChange={e => { setYoungIpo(e.target.checked); resetCalc(); }}
                       data-testid="young-ipo-checkbox"
                       style={{ width: 15, height: 15, accentColor: navColor }} />
                <label htmlFor="young-ipo-checkbox" className="cursor-pointer select-none">
                  Young IPO (≤12 mo since IPO) — clamp ceiling to 5% NLV
                </label>
              </div>
            )}
          </>
        )}

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
                          // Pilot, not raw-state Offense.
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
              {/* Iterates SIZING_MODES_DISPLAY (Pilot · Normal ·
                  Offense) so the radio row reads left-to-right as a
                  conservatism spectrum. The canonical SIZING_MODES
                  lookup (index 0/1/2) stays intact for indexing. */}
              <div className="flex gap-4 mt-1">
                {SIZING_MODES_DISPLAY.map(m => (
                  <Radio key={m.key} checked={sizingMode === m.index}
                         onClick={() => {
                           setSizingMode(m.index);
                           setSizingModeManual(true);
                           resetCalc();
                         }}
                         label={`${m.icon} ${m.label}`} />
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

          {/* ── PYRAMID SIZER (six-rule composite model) ── */}
          {tab === "pyramid" && pyramidResults && (
            <PyramidResults
              ticker={holdingData?.ticker || ""}
              tradeId={holdingData?.trade_id}
              entry={entry}
              tolPct={SIZING_MODES[sizingMode].pct}
              modeName={SIZING_MODES_BASE[sizingMode].label}
              ema21={ema21}
              results={pyramidResults}
              onSendToLogBuy={(args) => sendToLogBuy(args)}
            />
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

// ── Volatility Sizer output (composite-stop model). One answer card,
// composite winner subtitle, bind indicator, and a locked 0.5/1.0/1.5
// ATR scale-out ladder. All math lives in @/lib/vol-sizer.
function VolatilityResults({
  ticker, entry, tolPct, modeName, results, onSendToLogBuy,
}: {
  ticker: string;
  entry: number;
  tolPct: number;
  modeName: string;
  results: VolSizerResults;
  onSendToLogBuy: (args: {
    ticker: string;
    shares: number;
    price: number;
    stop?: number;
    stopMode?: "price" | "atr" | "ladder";
    atrMultiplier?: 1 | 1.5 | 2;
    ladderShares?: [number, number, number];
    action: string;
  }) => void;
}) {
  const { composite, scaleOut } = results;
  const bindLabel = results.bind === "risk"
    ? `risk budget (${tolPct}%)`
    : `${results.ceilingPct}% ${results.ceilingPolicy === "young_ipo" ? "young-IPO" : "position"} ceiling`;

  // Composite winner subtitle — tells the user which of the two
  // candidate stops governed. When Key Level wins, mention the buffer
  // basis so the reader can trace the number back.
  const compositeSubtitle = composite.winner === "atr_floor"
    ? `1 ATR floor: Entry − ${formatCurrency(results.atrPerShare)} ATR`
    : `Key Level − ${formatCurrency(composite.candidates.bufferApplied)} buffer (${composite.candidates.bufferBasis === "half_atr" ? "0.5 ATR" : "1% floor"})`;

  return (
    <div data-testid="vol-results">
      <h3 className="text-[15px] font-semibold mb-4">Sizing Profile: {ticker || "—"}</h3>

      {/* Context row */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <MetricCard label="Risk Budget" value={formatCurrency(results.riskBudget, { decimals: 0 })}
                    sub={`${tolPct}% Risk (${modeName} Mode)`}
                    accent="#6366f1" />
        <MetricCard label="ATR" value={`${(results.atrPerShare / entry * 100).toFixed(2)}%`}
                    sub={`${formatCurrency(results.atrPerShare)}/share`}
                    accent="#f59f00" />
        <MetricCard label="Ceiling" value={`${results.ceilingShares} shs`}
                    sub={`${results.ceilingPct}% NLV${results.ceilingPolicy === "young_ipo" ? " · young IPO" : ""}`}
                    accent="#3b82f6" />
      </div>

      {/* THE ANSWER — two tiles side by side. Left = shares + composite
          detail; Right = notional + NLV% + bind. Eyes travel a much
          shorter horizontal distance than the original single wide card. */}
      <div className="grid grid-cols-2 gap-3 mb-3" data-testid="composite-answer">
        {/* Left tile — Recommended shares + composite stop. */}
        <div className="p-4 rounded-[14px] relative overflow-hidden"
             data-testid="composite-answer-shares"
             style={{
               border: "1px solid var(--border)",
               borderLeft: "6px solid #08a86b",
               background: "color-mix(in oklab, #08a86b 7%, var(--surface))",
             }}>
          <div className="text-[10px] uppercase tracking-[0.10em] font-semibold mb-2" style={{ color: "var(--ink-4)" }}>
            Recommended
          </div>
          <div className="text-[28px] font-semibold privacy-mask"
               style={{ fontFamily: "var(--font-jetbrains), monospace", color: "#08a86b" }}
               data-testid="final-shares">
            {results.finalShares} <span className="text-[14px] font-normal" style={{ color: "var(--ink-4)" }}>shs</span>
          </div>
          <div className="text-[12px] mt-2" style={{ color: "var(--ink-4)" }}>
            Composite stop: <strong style={{ color: "#3b82f6" }}>{formatCurrency(composite.price)}</strong>{" "}
            · <strong style={{ color: "var(--ink-3)" }}>{composite.distancePct.toFixed(2)}%</strong> below entry ·{" "}
            <strong style={{ color: "var(--ink-3)" }}>{composite.atrFraction.toFixed(2)}× ATR</strong>
          </div>
          <div className="text-[11px] mt-0.5" style={{ color: "var(--ink-4)" }}>
            Winner: <strong style={{ color: "var(--ink-3)" }}>{compositeSubtitle}</strong>
          </div>
          <div className="mt-3 pt-2 text-[11px]"
               style={{ borderTop: "1px dashed var(--border)", color: "var(--ink-4)" }}>
            Risk if stopped: <strong style={{ color: "#d97706" }}>{formatCurrency(results.riskIfStopped, { decimals: 0 })}</strong>{" "}
            (<strong style={{ color: "#d97706" }}>{results.riskPct.toFixed(2)}%</strong> NLV)
          </div>
        </div>

        {/* Right tile — Notional + NLV% + bind. Same accent so they
            read as a pair, mirror layout: right-aligned totals, bind
            footer where the risk footer sits on the left. */}
        <div className="p-4 rounded-[14px] relative overflow-hidden"
             data-testid="composite-answer-notional"
             style={{
               border: "1px solid var(--border)",
               borderLeft: "6px solid #08a86b",
               background: "color-mix(in oklab, #08a86b 7%, var(--surface))",
             }}>
          <div className="flex items-center justify-between mb-2">
            <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>
              Notional
            </div>
            <span className="text-[9px] uppercase tracking-[0.08em] font-semibold px-2 py-0.5 rounded-[6px]"
                  data-testid="bind-badge"
                  style={{ background: "#08a86b", color: "#fff" }}>
              {results.bind === "risk" ? "Risk-bound" : "Ceiling-bound"}
            </span>
          </div>
          <div className="text-[28px] font-semibold privacy-mask"
               style={{ fontFamily: "var(--font-jetbrains), monospace", color: "#08a86b" }}>
            {formatCurrency(results.positionCost, { decimals: 0 })}
          </div>
          <div className="text-[12px] mt-2" style={{ color: "var(--ink-4)" }}>
            <strong style={{ color: "#08a86b" }}>{results.positionPct.toFixed(1)}%</strong> of NLV
          </div>
          <div className="mt-3 pt-2 text-[11px]"
               style={{ borderTop: "1px dashed var(--border)", color: "var(--ink-4)" }}>
            Bound by <strong style={{ color: results.bind === "ceiling" ? "#d97706" : "var(--ink-3)" }}>{bindLabel}</strong>
          </div>
        </div>
      </div>

      {/* Scale-out ladder — locked 0.5/1.0/1.5 ATR. */}
      <ScaleOutStopsCard
        ladder={scaleOut}
        atrPerShare={results.atrPerShare}
        accent="#0ea5a4"
        onSendLadder={(ladder) => {
          onSendToLogBuy({
            ticker,
            shares: ladder.totalShares,
            price: ladder.entry,
            stopMode: "ladder",
            ladderShares: [ladder.legs[0].shares, ladder.legs[1].shares, ladder.legs[2].shares],
            action: "new",
          });
        }}
      />

      {results.warnings.length > 0 && (
        <div className="mt-3">
          {results.warnings.map((w) => (
            <Banner key={w} type="warning">{w}</Banner>
          ))}
        </div>
      )}

      <div className="mt-4">
        <button onClick={() => {
                  // Send with the resolved composite stop as a dollar
                  // price. Log Buy consumes stopMode='price' to skip
                  // its default pct-mode ATR recomputation.
                  onSendToLogBuy({
                    ticker,
                    shares: results.finalShares,
                    price: entry,
                    stop: composite.price,
                    stopMode: "price",
                    action: "new",
                  });
                }}
                className="w-full h-[48px] rounded-[12px] text-[13px] font-semibold transition-all hover:brightness-95 cursor-pointer"
                style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }}>
          📝 Send to Log Buy — {ticker || "—"} ({results.finalShares} shs @ {formatCurrency(entry)})
        </button>
      </div>
    </div>
  );
}

// ── Pyramid Sizer output (six-rule composite model, 2026-07-18) ──
// Two-tile answer card (matching the Volatility redesign) + a held-
// lots table showing the per-lot risk contributions that drive the
// budget gate, + a prominent broker-setup callout so the trader
// knows to place the trailing stop at the broker immediately after
// the add fills. All math lives in @/lib/pyramid-sizer.
function PyramidResults({
  ticker, tradeId, entry, tolPct, modeName, ema21, results, onSendToLogBuy,
}: {
  ticker: string;
  tradeId?: string;
  entry: number;
  tolPct: number;
  modeName: string;
  ema21: number | null;
  results: PyramidSizerResults;
  onSendToLogBuy: (args: { ticker: string; shares: number; price: number; trade_id?: string; action: string }) => void;
}) {
  const { composite, budget, progress, location, ceiling } = results;
  const bindLabel = (() => {
    switch (results.bind) {
      case "risk":         return `risk budget (${tolPct.toFixed(2)}%)`;
      case "notional_cap": return `per-add cap (${PYRAMID_ADD_CAP_PCT}% NAV)`;
      case "progress":     return `progress multiplier (${(progress.multiplier * 100).toFixed(0)}%)`;
      case "ceiling":      return `campaign ceiling (${PYRAMID_CAMPAIGN_CEILING_PCT}% NAV)`;
      case "blocked":      return "blocked";
    }
  })();
  const compositeSubtitle = composite.winner === "atr_floor"
    ? `1 ATR floor: Entry − ${formatCurrency(results.atrPerShare)}`
    : `Key Level − ${formatCurrency(composite.candidates.bufferApplied)} buffer (${composite.candidates.bufferBasis === "half_atr" ? "0.5 ATR" : "1% floor"})`;
  const brokerAnchorLine = ema21 !== null && ema21 > 0
    ? `Trails: 21 EMA − 0.5 ATR (recompute daily: stop = MAX(${formatCurrency(composite.price)}, 21EMA_today − 0.5 ATR))`
    : "Trails: 21 EMA − 0.5 ATR (rising only)";

  return (
    <div data-testid="pyramid-results">
      <h3 className="text-[15px] font-semibold mb-4">Pyramid Analysis: {ticker || "—"}</h3>

      {/* Context row — budget, ATR, campaign risk. */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <MetricCard label="Budget" value={formatCurrency(budget.budgetDollars, { decimals: 0 })}
                    sub={`${tolPct.toFixed(2)}% × NAV · ${modeName}`}
                    accent="#6366f1" />
        <MetricCard label="Campaign Risk" value={formatCurrency(budget.campaignRisk, { decimals: 0 })}
                    sub={`Σ held-lot risk · ${results.existingShares} shs held`}
                    accent="#f59f00" />
        <MetricCard label="Headroom" value={formatCurrency(Math.max(0, budget.headroom), { decimals: 0 })}
                    sub={budget.headroom > 0 ? "Available for this add" : "None — no add"}
                    accent={budget.headroom > 0 ? "#08a86b" : "#e5484d"} />
      </div>

      {/* Block notices — surfaced above the tiles when any gate fails. */}
      {results.blocked && (
        <div className="mb-3">
          <Banner type="error">
            <div className="font-semibold mb-1" data-testid="pyramid-blocked">
              {results.finalShares === 0 ? "NO ADD" : `ADD CLIPPED to ${results.finalShares} shs`}
            </div>
            {results.blockReasons.map((r, i) => (
              <div key={i} className="text-[12px] font-normal" style={{ opacity: 0.9 }}>
                · {r}
              </div>
            ))}
          </Banner>
        </div>
      )}

      {/* THE ANSWER — two side-by-side tiles. */}
      <div className="grid grid-cols-2 gap-3 mb-3" data-testid="pyramid-answer">
        {/* Left tile — new add sizing. */}
        <div className="p-4 rounded-[14px] relative overflow-hidden"
             data-testid="pyramid-answer-shares"
             style={{
               border: "1px solid var(--border)",
               borderLeft: `6px solid ${results.finalShares > 0 ? "#08a86b" : "#94a3b8"}`,
               background: `color-mix(in oklab, ${results.finalShares > 0 ? "#08a86b" : "#94a3b8"} 7%, var(--surface))`,
             }}>
          <div className="text-[10px] uppercase tracking-[0.10em] font-semibold mb-2" style={{ color: "var(--ink-4)" }}>
            New Add
          </div>
          <div className="text-[28px] font-semibold privacy-mask"
               style={{ fontFamily: "var(--font-jetbrains), monospace", color: results.finalShares > 0 ? "#08a86b" : "var(--ink-4)" }}
               data-testid="pyramid-final-shares">
            {results.finalShares} <span className="text-[14px] font-normal" style={{ color: "var(--ink-4)" }}>shs</span>
          </div>
          <div className="text-[12px] mt-2" style={{ color: "var(--ink-4)" }}>
            Composite stop: <strong style={{ color: "#3b82f6" }}>{formatCurrency(composite.price)}</strong>{" "}
            · <strong style={{ color: "var(--ink-3)" }}>{composite.distancePct.toFixed(2)}%</strong> below entry ·{" "}
            <strong style={{ color: "var(--ink-3)" }}>{composite.atrFraction.toFixed(2)}× ATR</strong>
          </div>
          <div className="text-[11px] mt-0.5" style={{ color: "var(--ink-4)" }}>
            Winner: <strong style={{ color: "var(--ink-3)" }}>{compositeSubtitle}</strong>
          </div>
          <div className="mt-3 pt-2 text-[11px]"
               style={{ borderTop: "1px dashed var(--border)", color: "var(--ink-4)" }}>
            Risk if stopped: <strong style={{ color: "#d97706" }}>{formatCurrency(results.addRiskDollars, { decimals: 0 })}</strong>{" "}
            (<strong style={{ color: "#d97706" }}>{results.addRiskPct.toFixed(2)}%</strong> NAV)
          </div>
        </div>

        {/* Right tile — campaign after add. */}
        <div className="p-4 rounded-[14px] relative overflow-hidden"
             data-testid="pyramid-answer-campaign"
             style={{
               border: "1px solid var(--border)",
               borderLeft: `6px solid ${results.finalShares > 0 ? "#08a86b" : "#94a3b8"}`,
               background: `color-mix(in oklab, ${results.finalShares > 0 ? "#08a86b" : "#94a3b8"} 7%, var(--surface))`,
             }}>
          <div className="flex items-center justify-between mb-2">
            <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>
              Campaign After Add
            </div>
            {results.finalShares > 0 && (
              <span className="text-[9px] uppercase tracking-[0.08em] font-semibold px-2 py-0.5 rounded-[6px]"
                    data-testid="pyramid-bind-badge"
                    style={{ background: "#08a86b", color: "#fff" }}>
                {results.bind === "risk" ? "Risk-bound" :
                 results.bind === "notional_cap" ? "Cap-bound" :
                 results.bind === "ceiling" ? "Ceiling-bound" :
                 results.bind === "progress" ? "Progress-bound" : "Bound"}
              </span>
            )}
          </div>
          <div className="text-[28px] font-semibold privacy-mask"
               style={{ fontFamily: "var(--font-jetbrains), monospace", color: results.finalShares > 0 ? "#08a86b" : "var(--ink-4)" }}>
            {formatCurrency(results.projectedNotional, { decimals: 0 })}
          </div>
          <div className="text-[12px] mt-2" style={{ color: "var(--ink-4)" }}>
            <strong style={{ color: "#08a86b" }}>{results.projectedNotionalPct.toFixed(1)}%</strong> of NAV ·{" "}
            <strong style={{ color: "var(--ink-3)" }}>{results.projectedShares} shs</strong> total
          </div>
          <div className="text-[11px] mt-0.5" style={{ color: "var(--ink-4)" }}>
            Was: {formatCurrency(results.existingNotional, { decimals: 0 })} · {results.existingNotionalPct.toFixed(1)}% NAV · {results.existingShares} shs
          </div>
          <div className="mt-3 pt-2 text-[11px]"
               style={{ borderTop: "1px dashed var(--border)", color: "var(--ink-4)" }}>
            Bound by <strong style={{ color: results.bind === "ceiling" ? "#d97706" : "var(--ink-3)" }}>{bindLabel}</strong>
          </div>
        </div>
      </div>

      {/* Broker-setup callout — pinned so the user has zero excuse to
          skip setting the trailing stop when the add fills. */}
      {results.finalShares > 0 && (
        <div className="p-4 rounded-[12px] mb-3"
             data-testid="pyramid-broker-callout"
             style={{
               border: "1px solid color-mix(in oklab, #f59f00 30%, var(--border))",
               background: "color-mix(in oklab, #f59f00 8%, var(--surface))",
             }}>
          <div className="text-[11px] uppercase tracking-[0.10em] font-semibold mb-1" style={{ color: "#d97706" }}>
            📌 SET UP AT BROKER
          </div>
          <div className="text-[13px] mb-1" style={{ color: "var(--ink)" }}>
            Trailing stop on this add: <strong>{results.finalShares} shs @ {formatCurrency(composite.price)}</strong>
          </div>
          <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>
            {brokerAnchorLine}
          </div>
        </div>
      )}

      {/* Held-lots table — per-row risk decomposition. */}
      {budget.lotRisks.length > 0 && (
        <div className="p-4 rounded-[12px] mb-3"
             data-testid="pyramid-held-lots"
             style={{ border: "1px solid var(--border)", background: "var(--surface)" }}>
          <div className="text-[11px] uppercase tracking-[0.10em] font-semibold mb-2" style={{ color: "var(--ink-4)" }}>
            Held Lots (Campaign Risk Contribution)
          </div>
          <div className="flex flex-col gap-1 text-[12px]" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
            {budget.lotRisks.map((l, i) => (
              <div key={i} className="grid grid-cols-[52px_1fr_1fr_1fr] items-baseline gap-2" data-testid={`pyramid-lot-row-${i}`}>
                <span className="font-semibold" style={{ color: "var(--ink-3)" }}>{l.label || `L${i + 1}`}</span>
                <span style={{ color: "var(--ink)" }}>{l.shares} sh @ {formatCurrency(l.entry)}</span>
                <span style={{ color: "var(--ink-3)" }}>stop {formatCurrency(l.stopLoss)}</span>
                <span className="text-right">
                  {l.risk === 0
                    ? <span style={{ color: "#08a86b" }}>risk-free</span>
                    : <span style={{ color: "#d97706" }}>risk {formatCurrency(l.risk, { decimals: 0 })}</span>}
                </span>
              </div>
            ))}
          </div>
          <div className="mt-2 pt-2 text-[11px] flex items-baseline justify-between"
               style={{ borderTop: "1px dashed var(--border)", color: "var(--ink-4)" }}>
            <span>Total: <strong style={{ color: "var(--ink)" }}>{results.existingShares} shs</strong></span>
            <span>Campaign risk: <strong style={{ color: "#d97706" }}>{formatCurrency(budget.campaignRisk, { decimals: 0 })}</strong></span>
          </div>
        </div>
      )}

      {/* Send to Log Buy — only when we have a positive add. */}
      {results.finalShares > 0 && (
        <div>
          <button onClick={() => onSendToLogBuy({
                    ticker,
                    shares: results.finalShares,
                    price: entry,
                    trade_id: tradeId,
                    action: "scale_in",
                  })}
                  className="w-full h-[48px] rounded-[12px] text-[13px] font-semibold transition-all hover:brightness-95 cursor-pointer"
                  style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }}>
            📝 Send to Log Buy — {ticker || "—"} (+{results.finalShares} shs @ {formatCurrency(entry)})
          </button>
        </div>
      )}

      {/* Silence unused-warn for reference locals we intentionally exposed
          in the API for future UI expansions (e.g. tooltip callouts). */}
      <div className="sr-only" aria-hidden>
        {location.ceilingPrice} {ceiling.maxTotalNotional}
      </div>
    </div>
  );
}

// Advisory 3-leg scale-out at Entry − 0.5 / 1.0 / 1.5 ATR. Equal-thirds
// share split → average exit lands at 1 ATR below entry, matching the
// risk budget when the composite lands at 1 ATR. See @/lib/vol-sizer
// for the ratio-choice rationale.
function ScaleOutStopsCard({
  ladder, atrPerShare, accent, onSendLadder,
}: {
  ladder: ScaleOutStops;
  atrPerShare: number;
  accent: string;
  onSendLadder: (ladder: ScaleOutStops) => void;
}) {
  return (
    <div className="p-4 rounded-[12px] relative overflow-hidden"
         data-testid="scale-out-stops"
         style={{
           border: "1px solid var(--border)",
           borderLeft: `4px solid ${accent}`,
           background: `color-mix(in oklab, ${accent} 4%, var(--surface))`,
         }}>
      <div className="flex items-center justify-between mb-1.5">
        <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>
          Scale-Out Stops
        </div>
        <span className="text-[9px] uppercase tracking-[0.08em] font-semibold px-2 py-0.5 rounded-[6px]"
              style={{ background: `color-mix(in oklab, ${accent} 15%, transparent)`, color: accent }}>
          3-Leg ATR Ladder
        </span>
      </div>

      <div className="text-[11px] mb-2" style={{ color: "var(--ink-4)" }}>
        {ladder.totalShares} shs @ {formatCurrency(ladder.entry)} · ATR = {formatCurrency(atrPerShare)}/share
      </div>

      <div className="flex flex-col gap-1">
        {ladder.legs.map((leg) => (
          <div key={leg.atrMultiple} className="grid grid-cols-[52px_1fr_60px_1fr] items-baseline gap-2 text-[11px]"
               data-testid={`scale-out-leg-${SCALE_OUT_ATR_MULTIPLIERS.indexOf(leg.atrMultiple as 0.5 | 1.0 | 1.5)}`}
               style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
            <span className="font-semibold" style={{ color: accent }}>−{leg.atrMultiple.toFixed(2)} ATR</span>
            <span style={{ color: "var(--ink)" }}>{formatCurrency(leg.stopPrice)}</span>
            <span style={{ color: "var(--ink-3)" }}>{leg.shares} shs</span>
            <span className="text-right" style={{ color: "var(--ink-3)" }}>
              −{formatCurrency(leg.loss, { decimals: 0 })} <span style={{ color: "var(--ink-4)" }}>({leg.lossPctNlv.toFixed(2)}%)</span>
            </span>
          </div>
        ))}
      </div>

      <div className="mt-2 pt-2 flex items-baseline justify-between text-[11px]"
           style={{ borderTop: "1px dashed var(--border)", color: "var(--ink-4)" }}>
        <span>Risk if fully stopped: <strong style={{ color: "var(--ink)" }}>{formatCurrency(ladder.totalLoss, { decimals: 0 })}</strong> ({ladder.totalLossPctNlv.toFixed(2)}% NLV)</span>
        <span style={{ color: "var(--ink-4)" }}>avg exit {ladder.avgExitPct.toFixed(2)}%</span>
      </div>

      <button type="button"
              onClick={() => onSendLadder(ladder)}
              disabled={ladder.totalShares <= 0}
              className="mt-3 w-full h-[34px] rounded-[10px] text-[12px] font-semibold transition-all hover:brightness-95 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
              data-testid="send-ladder-to-logbuy"
              style={{
                background: `color-mix(in oklab, ${accent} 12%, var(--surface))`,
                color: accent,
                border: `1px solid color-mix(in oklab, ${accent} 40%, var(--border))`,
              }}>
        📝 Send to Log Buy with ladder ({ladder.totalShares} shs)
      </button>
    </div>
  );
}
