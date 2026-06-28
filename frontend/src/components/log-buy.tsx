"use client";

import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { api, getActivePortfolio, type TradePosition, type TradeDetail, type Strategy } from "@/lib/api";
import { uploadWithTimeout } from "@/lib/upload-with-timeout";
import { UploadTracker, type UploadEntry, type UploadKind } from "./upload-tracker";
import { StrategyChip } from "./strategy-chip";
import { SearchSelect } from "./search-select";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";

const BUY_RULES = [
  "br1.1 Consolidation", "br1.2 Cup w Handle", "br1.3 Cup w/o Handle", "br1.4 Double Bottom",
  "br1.5 IPO Base", "br1.6 Flat Base", "br1.7 Consolidation Pivot", "br1.8 High Tight Flag",
  "br2.1 HVE", "br2.2 HVSI", "br2.3 HV1",
  "br3.1 Reclaim 21e", "br3.2 Reclaim 50s", "br3.3 Reclaim 200s", "br3.4 Reclaim 10W", "br3.5 Reclaim 8e", "br3.6 Green Line Break",
  "br4.1 PB 21e", "br4.2 PB 50s", "br4.3 PB 10w", "br4.4 PB 200s", "br4.5 PB 8e", "br4.6 VWAP",
  "br5.1 Undercut & Rally", "br5.2 Upside Reversal",
  "br6.1 Gapper", "br6.2 Continuation Gap Up",
  "br7.1 TQQQ Strategy", "br7.2 New High after Gentle PB", "br7.3 JL Century Mark",
  "br8.1 Daily STL Break", "br8.2 Weekly STL Break", "br8.3 Monthly STL Break",
  "br9.1 21e Strategy",
  "br10.1 Hedging with leverage product",
  "br11.1 Shorting",
  "br12.1 Option Play",
];

// SIZING_MODES + the MCT-state → mode mapping live in @/lib/sizing-mode
// so Log Buy stays in lockstep with Position Sizer. Local re-export
// keeps existing call sites (SIZING_MODES[i].label / .pct / .icon)
// untouched.
import {
  SIZING_MODES,
  mctStateToSizingMode,
  deriveAutoSizingMode,
  exitLadderFloor,
  describeMctSource,
  type ExitAlert,
} from "@/lib/sizing-mode";

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-[10px] uppercase tracking-[0.10em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>
        {label}
      </label>
      {children}
    </div>
  );
}

const inputCls = "w-full h-[42px] px-3.5 rounded-[10px] text-[13px] outline-none transition-colors";
const inputStyle: React.CSSProperties = {
  background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)",
  fontFamily: "var(--font-jetbrains), monospace",
};

function DropZone({ label, icon, files, onFiles, multiple, accept }: {
  label: string; icon: string; files: File[]; onFiles: (f: File[]) => void; multiple?: boolean; accept?: string;
}) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDragging(false);
    const dropped = Array.from(e.dataTransfer.files).filter(f => !accept || accept.split(",").some(a => f.type === a.trim()));
    if (dropped.length > 0) onFiles(multiple ? [...files, ...dropped] : [dropped[0]]);
  };

  const hasFiles = files.length > 0;
  const borderColor = dragging ? "#6366f1" : hasFiles ? "#08a86b" : "var(--border)";
  const bgColor = dragging ? "color-mix(in oklab, #6366f1 5%, var(--bg))" : hasFiles ? "color-mix(in oklab, #08a86b 5%, var(--bg))" : "var(--bg)";
  const textColor = hasFiles ? "#08a86b" : "var(--ink-4)";

  return (
    <div>
      {label && (
        <div className="text-[11px] font-medium mb-1.5 flex items-center gap-1" style={{ color: "var(--ink-3)" }}>
          {icon && <span>{icon}</span>} {label}
        </div>
      )}
      <div
        className="flex flex-col items-center justify-center rounded-[10px] cursor-pointer transition-all"
        style={{ border: `1.5px dashed ${borderColor}`, background: bgColor, padding: hasFiles ? "10px 16px" : "16px", minHeight: 70 }}
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        {hasFiles ? (
          <div className="flex flex-wrap gap-1.5 w-full">
            {files.map((f, i) => (
              <div key={i} className="flex items-center gap-1.5 px-2 py-1 rounded-[6px] text-[10px]"
                   style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-3)" }}>
                <span className="truncate" style={{ maxWidth: 120 }}>{f.name}</span>
                <button onClick={e => { e.stopPropagation(); onFiles(files.filter((_, j) => j !== i)); }}
                        className="text-[10px] font-bold cursor-pointer" style={{ color: "#e5484d" }}>x</button>
              </div>
            ))}
          </div>
        ) : (
          <>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={dragging ? "#6366f1" : "var(--ink-4)"} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="mb-1.5 opacity-60">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <span className="text-[11px]" style={{ color: textColor }}>{dragging ? "Drop files here" : "Drag & drop or click to upload"}</span>
            <span className="text-[9px] mt-0.5" style={{ color: "var(--ink-5)" }}>PNG, JPG, PDF</span>
          </>
        )}
        <input ref={inputRef} type="file" accept={accept} multiple={multiple} className="hidden"
               onChange={e => { if (e.target.files) onFiles(multiple ? [...files, ...Array.from(e.target.files)] : Array.from(e.target.files)); e.target.value = ""; }} />
      </div>
    </div>
  );
}

// SearchSelect lives in ./search-select. Originally inline here; extracted
// 2026-05-29 so Log Sell could reuse the same combobox. Two consumers
// (log-buy + log-sell) and growing — trade-manager.tsx still has its own
// simpler inline copy, deferred for a future consolidation pass.

function Radio({ checked, onClick, label }: { checked: boolean; onClick: () => void; label: string }) {
  return (
    <label className="flex items-center gap-2 cursor-pointer text-[13px]" onClick={onClick}>
      <span className="w-[18px] h-[18px] rounded-full flex items-center justify-center shrink-0"
            style={{ border: `2px solid ${checked ? "#08a86b" : "var(--border)"}` }}>
        {checked && <span className="w-[10px] h-[10px] rounded-full" style={{ background: "#08a86b" }} />}
      </span>
      <span style={{ color: checked ? "var(--ink)" : "var(--ink-3)" }}>{label}</span>
    </label>
  );
}

export function LogBuy({ navColor }: { navColor: string }) {
  const [equity, setEquity] = useState(0);
  const [openTrades, setOpenTrades] = useState<TradePosition[]>([]);
  // mctState drives the read-only sizing-mode display below. Replaces
  // the V10 mFactorSuggestion / radio buttons. No manual override on
  // Log Buy by design — Position Sizer is the override surface.
  const [mctState, setMctState] = useState<string | null>(null);
  // Active exit-ladder alerts. Feed the sizing-mode floor (see lib/
  // sizing-mode#exitLadderFloor) so Log Buy's auto-mode matches what
  // Position Sizer would derive for the same market state.
  const [activeExits, setActiveExits] = useState<readonly { signal: string; severity?: string }[]>([]);

  const [allDetails, setAllDetails] = useState<TradeDetail[]>([]);
  const [campPrice, setCampPrice] = useState(0);
  const [actionType, setActionType] = useState<"new" | "scalein">("new");
  const [date, setDate] = useState(() => {
    const n = new Date();
    return `${n.getFullYear()}-${String(n.getMonth() + 1).padStart(2, "0")}-${String(n.getDate()).padStart(2, "0")}`;
  });
  const [time, setTime] = useState(() => {
    const n = new Date();
    return `${String(n.getHours()).padStart(2, "0")}:${String(n.getMinutes()).padStart(2, "0")}`;
  });
  const [ticker, setTicker] = useState("");
  const [fetchingPrice, setFetchingPrice] = useState(false);
  const [tradeId, setTradeId] = useState("");
  const [rule, setRule] = useState("");
  const [selectedCampaign, setSelectedCampaign] = useState("");
  // Strategy tagging (Migration 019). Defaults to CanSlim — matches the DB
  // column DEFAULT and the user's primary strategy. On scale-in we render
  // the field read-only and prefill from the parent campaign so the strategy
  // can never drift mid-campaign.
  const [strategy, setStrategy] = useState("CanSlim");
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  // sizingMode default Normal (1) — overwritten on mount by the MCT
  // state read. Stays Normal if the read fails (mctStateToSizingMode
  // falls back to safe middle ground). sizingModeManual flips true the
  // moment the user picks a different mode for this Log Buy submission;
  // the risk-per-trade math (riskPctInput below) reads from sizingMode
  // either way, so the override flows naturally into the saved buy.
  // Override is form-local: refresh / navigate-away / submit all reset
  // it (the component remounts and the auto pick takes over again).
  const [sizingMode, setSizingMode] = useState<0 | 1 | 2>(1);
  const [sizingModeManual, setSizingModeManual] = useState(false);
  const [shares, setShares] = useState("");
  const [price, setPrice] = useState("");
  const [stopMode, setStopMode] = useState<"price" | "pct" | "atr">("pct");
  const [stopValue, setStopValue] = useState("");
  const [slPct, setSlPct] = useState("5.0");
  // ATR stop loss mode: multiplier × atrPct% below entry. atrPct is captured
  // from /api/prices/lookup at the same time we fetch price; backend returns
  // 0.0 for tickers with <21 bars (the "ATR unavailable" sentinel). The pills
  // disable in that case and the user falls back to Price or Percentage.
  const [atrPct, setAtrPct] = useState(0);
  const [atrMultiplier, setAtrMultiplier] = useState<1 | 1.5 | 2>(1.5);
  // For options the stop-loss field is hidden by default — premium-based
  // stops don't follow the < 8% stock convention, and 50% is a placeholder
  // not a meaningful default. The user can reveal it via "Show stop loss".
  // Reset to true for stocks via the ticker-keyed effect below.
  const [showStopLoss, setShowStopLoss] = useState(true);
  const [notes, setNotes] = useState("");
  const [errors, setErrors] = useState<string[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [entryCharts, setEntryCharts] = useState<File[]>([]);
  const [positionCharts, setPositionCharts] = useState<File[]>([]);
  const [msScreenshot, setMsScreenshot] = useState<File | null>(null);

  useEffect(() => {
    Promise.all([
      api.journalLatest(getActivePortfolio()).catch((err) => {
        log.error("log-buy", "journalLatest fetch failed", err);
        return { end_nlv: 100000 };
      }),
      api.tradesOpen(getActivePortfolio()).catch((err) => {
        log.error("log-buy", "tradesOpen fetch failed", err);
        return [];
      }),
      // V11 MCT state drives sizing mode. Replaces the legacy
      // /api/market/mfactor MA-stack heuristic. We only read `state`.
      api.rallyPrefix().catch((err) => {
        log.error("log-buy", "rallyPrefix fetch failed", err);
        return { prefix: "" };
      }),
      api.tradesOpenDetails(getActivePortfolio()).catch((err) => {
        log.error("log-buy", "tradesOpenDetails fetch failed", err);
        return { details: [], lot_closures: [] };
      }),
      api.listStrategies({ active: true, portfolio: getActivePortfolio() }).catch((err) => {
        log.error("log-buy", "listStrategies fetch failed", err);
        return [] as Strategy[];
      }),
    ]).then(([j, open, rally, det, strats]) => {
      setEquity(parseFloat(String(j.end_nlv || 100000)));
      setOpenTrades(open as TradePosition[]);
      setAllDetails(det.details);
      const stateStr = (rally as { state?: string } | null)?.state ?? null;
      const exits = ((rally as { active_exits?: ExitAlert[] } | null)?.active_exits ?? []) as ExitAlert[];
      setMctState(stateStr);
      setActiveExits(exits);
      // Match the Position Sizer's derivation so the two pages agree:
      // exit-ladder floor downshifts auto-mode (e.g. 50 SMA Violation
      // forces Defense even on POWERTREND).
      setSizingMode(deriveAutoSizingMode(stateStr, exits).idx);
      setStrategies(strats);
    });
  }, []);

  useEffect(() => {
    if (actionType === "new" && (!tradeId || tradeId.endsWith("-0XX"))) {
      api.nextTradeId(getActivePortfolio(), date).then(r => {
        if (r.trade_id) setTradeId(r.trade_id);
      }).catch(() => {
        const n = new Date();
        setTradeId(`${n.getFullYear()}${String(n.getMonth() + 1).padStart(2, "0")}-0XX`);
      });
    }
  }, [actionType]);

  // pendingAtrDefault: set true when an importer prefill arrives with no
  // stop fields. The actual ATR-vs-pct fallback decision waits for the
  // priceLookup effect to resolve atrPct (a separate effect below picks
  // the mode once we know whether ATR is available).
  // atrResolved: gates the fallback effect on a COMPLETED price lookup.
  // fetchingPrice alone is insufficient — it starts false (before the
  // 600ms debounce) and goes true only when the fetch starts, so the
  // fallback effect would otherwise fire on mount with atrPct=0 and
  // prematurely lock in the pct/5.0 default before the fetch even ran.
  const [pendingAtrDefault, setPendingAtrDefault] = useState(false);
  const [atrResolved, setAtrResolved] = useState(false);

  // When the importer (or Sizer) prefills a price, the ticker-watching
  // auto-fetch effect below would otherwise overwrite it with the live
  // price 600ms later — losing the IBKR execution price the user just
  // clicked through to log. This ref tells that effect to skip the
  // setPrice on the next priceLookup cycle (ATR still updates).
  // Cleared on consumption, so a subsequent ticker edit re-enables the
  // normal auto-price behavior.
  const skipNextPriceFetch = useRef(false);

  // Prefill from Position Sizer (via localStorage). Payload may carry an
  // explicit stopMode (Sizer's ATR scenarios) or just a resolved `stop`
  // dollar price (Sizer's tech-stop scenario + importer). Importer prefill
  // without any stop hint sets pendingAtrDefault so a separate effect can
  // pick ATR vs. percentage once atrPct lands.
  useEffect(() => {
    try {
      const raw = localStorage.getItem("ps_prefill");
      if (!raw) return;
      const data = JSON.parse(raw);
      localStorage.removeItem("ps_prefill");
      if (data.ticker) setTicker(data.ticker);
      if (data.shares) setShares(String(data.shares));
      if (data.price) { setPrice(String(data.price)); skipNextPriceFetch.current = true; }
      if (data.date) setDate(String(data.date));
      if (data.time) setTime(String(data.time));
      // Mode + value reception. Symmetric branches:
      //   stopMode='atr' → ATR mode with explicit multiplier
      //   stopMode='price' → Price mode with explicit resolved stop
      //   data.stop alone (legacy / tech-stop scenario) → Price mode +
      //     resolved stop. Pre-B-3 the receiver only set stopValue and
      //     left stopMode in its "pct" default, so the user had to flip
      //     manually. Fixed here as the ride-along.
      //   no stop hints at all → mark pendingAtrDefault for the
      //     atrPct-watcher effect below.
      if (data.stopMode === "atr" && (data.atrMultiplier === 1 || data.atrMultiplier === 1.5 || data.atrMultiplier === 2)) {
        setStopMode("atr");
        setAtrMultiplier(data.atrMultiplier);
      } else if (data.stopMode === "price" && typeof data.stop === "number") {
        setStopMode("price");
        setStopValue(String(data.stop.toFixed(2)));
      } else if (typeof data.stop === "number") {
        setStopMode("price");
        setStopValue(String(data.stop.toFixed(2)));
      } else {
        setPendingAtrDefault(true);
      }
      if (data.action === "scale_in" && data.trade_id) {
        setActionType("scalein");
        setSelectedCampaign(data.trade_id);
      } else {
        setActionType("new");
      }
    } catch { /* ignore */ }
  }, []);

  // Pick the default stop mode once atrPct resolves. Only fires when an
  // importer prefill arrived without explicit stop signals (the
  // pendingAtrDefault flag). Gated on atrResolved so we wait for the
  // first priceLookup to complete — without this, the effect would fire
  // on mount with atrPct=0 (initial state) and prematurely lock in the
  // pct/5.0 default. If ATR is available (atrPct > 0), default to ATR
  // mode at 1.5×. If sparse-history / option / fetch failure leaves
  // atrPct at 0 after the lookup, fall through to existing pct/5.0
  // default (no regression).
  useEffect(() => {
    if (!pendingAtrDefault || !atrResolved) return;
    if (atrPct > 0) {
      setStopMode("atr");
      setAtrMultiplier(1.5);
    }
    // else: leave stopMode at "pct" with slPct="5.0" — current behavior.
    setPendingAtrDefault(false);
  }, [pendingAtrDefault, atrResolved, atrPct]);

  // When the ticker resolves to an equity option, bump the default stop %
  // from the stock convention (5%) to the user's option playbook (50% of
  // premium). Only fires while the field still holds the stock default
  // "5.0", so a manual override survives a ticker re-edit.
  // ATR ride-along: ATR mode is meaningless for options (the underlying-
  // share atrPct doesn't size the premium stop), so when switching to an
  // option ticker, reset stopMode from "atr" back to "pct". The ATR
  // radio itself is hidden on option tickers (see render); this resets
  // any prior selection that would otherwise survive the ticker switch.
  useEffect(() => {
    const isOptionTicker = /^\S+\s+\d{6}\s+\$[0-9.]+(C|P)$/.test(ticker.trim());
    if (isOptionTicker && slPct === "5.0") setSlPct("50");
    if (!isOptionTicker && slPct === "50") setSlPct("5.0");
    if (isOptionTicker && stopMode === "atr") setStopMode("pct");
  }, [ticker]);

  // Stop-loss visibility tracks ticker shape: hide for options, show for
  // stocks. Kept separate from the slPct flip above so each effect stays
  // single-purpose. The "Show stop loss" link sets this true; switching
  // back to a stock ticker re-defaults to true (which is a no-op if it
  // was already true from a prior reveal).
  useEffect(() => {
    const isOptionTicker = /^\S+\s+\d{6}\s+\$[0-9.]+(C|P)$/.test(ticker.trim());
    setShowStopLoss(!isOptionTicker);
  }, [ticker]);

  // Auto-fetch price + ATR when ticker changes (debounced). Same
  // /api/prices/lookup call the Position Sizer uses. atr_pct === 0 is
  // the backend's "insufficient history" sentinel — ATR pills disable
  // in that case.
  useEffect(() => {
    if (!ticker || ticker.length < 1 || actionType !== "new") return;
    const timeout = setTimeout(() => {
      setFetchingPrice(true);
      api.priceLookup(ticker).then(data => {
        if (data && !("error" in data)) {
          // Prefill (importer / Sizer) wins for one cycle — keep the
          // IBKR execution price the user just clicked through. ATR
          // always refreshes since the right-rail sizer needs it.
          if (skipNextPriceFetch.current) {
            skipNextPriceFetch.current = false;
          } else {
            setPrice(String(data.price));
          }
          setAtrPct(data.atr_pct);
        }
      }).catch((err) => {
        log.debug.devOnly("log-buy", "priceLookup missing (expected)", err);
        // Fetch failed (503 from yfinance) — clear ATR so pills disable.
        setAtrPct(0);
      }).finally(() => {
        setFetchingPrice(false);
        setAtrResolved(true);
      });
    }, 600);
    return () => clearTimeout(timeout);
  }, [ticker, actionType]);

  // ── Computed values ──
  const sharesNum = parseFloat(shares) || 0;
  const priceNum = parseFloat(price) || 0;
  // Equity options carry a 100× contract multiplier; otherwise stocks are 1×.
  // Detected from ticker shape so the user doesn't have to flag it manually —
  // matches the same regex the backend uses to route option price lookups.
  const isOption = /^\S+\s+\d{6}\s+\$[0-9.]+(C|P)$/.test(ticker.trim());
  const multiplier = isOption ? 100 : 1;
  const unitLabel = isOption ? "Contracts" : "Shares";
  const totalCost = sharesNum * priceNum * multiplier;
  const riskPctInput = SIZING_MODES[sizingMode].pct;
  const riskBudget = equity * (riskPctInput / 100);

  let stopPrice = 0;
  if (stopMode === "price") {
    stopPrice = parseFloat(stopValue) || 0;
  } else if (stopMode === "atr") {
    // ATR mode: stop = price × (1 − multiplier × atrPct/100). Inline
    // duplication with the submit-body branch is intentional (the spec
    // explicitly rules out extracting a shared helper from vol-sizer).
    stopPrice = priceNum > 0 && atrPct > 0 ? priceNum * (1 - (atrMultiplier * atrPct) / 100) : 0;
  } else {
    // Default stop = 50% of premium for options (per the user's playbook),
    // 5% for stocks. The user can still override slPct manually.
    const defaultPct = isOption ? 50 : 5;
    const pct = parseFloat(slPct) || defaultPct;
    stopPrice = priceNum > 0 ? priceNum * (1 - pct / 100) : 0;
  }
  const stopDist = priceNum > 0 && stopPrice > 0 ? priceNum - stopPrice : 0;
  const stopPct = priceNum > 0 && stopPrice > 0 ? ((priceNum - stopPrice) / priceNum) * 100 : 0;
  const riskDollars = stopDist * sharesNum * multiplier;
  const posSizePct = equity > 0 ? (totalCost / equity) * 100 : 0;
  const recommendedShares = stopDist > 0 ? Math.floor(riskBudget / (stopDist * multiplier)) : 0;
  const recommendedCost = recommendedShares * priceNum * multiplier;

  const rbmStop = sharesNum > 0 && riskBudget > 0 ? priceNum - (riskBudget / (sharesNum * multiplier)) : 0;
  const riskViolation = riskDollars > riskBudget && riskBudget > 0 && stopPrice > 0;
  const withinBudget = riskDollars > 0 && riskDollars <= riskBudget;

  const selectedCamp = openTrades.find(t => t.trade_id === selectedCampaign);

  // Fetch live price + ATR when campaign is selected for scale-in. Mirrors
  // the new-campaign priceLookup effect (~lines 345-364): the same response
  // already carries atr_pct, so propagate it into atrPct (and flip
  // atrResolved in .finally) — otherwise atrPct stays 0 and the UI falsely
  // renders "ATR unavailable (insufficient history)" even on tickers the
  // backend has ample history for.
  useEffect(() => {
    if (actionType !== "scalein" || !selectedCamp?.ticker) { setCampPrice(0); return; }
    api.priceLookup(selectedCamp.ticker).then(data => {
      if (data && !("error" in data)) {
        setCampPrice(data.price);
        setAtrPct(data.atr_pct);
      }
    }).catch(() => {
      setAtrPct(0);
    }).finally(() => {
      setAtrResolved(true);
    });
  }, [actionType, selectedCamp?.ticker]);

  // Scale-in: inherit the parent campaign's strategy and lock the dropdown.
  // A campaign's strategy is fixed at creation; scaling in must never
  // reclassify it. Falls back to CanSlim for legacy rows that pre-date
  // Migration 019 (defensive — post-migration every row has a value).
  useEffect(() => {
    if (actionType === "scalein") {
      setStrategy(selectedCamp?.strategy || "CanSlim");
    }
  }, [actionType, selectedCamp?.strategy]);

  // ── Scale-In Cockpit computations ──
  const scaleIn = useMemo(() => {
    if (actionType !== "scalein" || !selectedCamp) return null;

    const campDetails = allDetails.filter(d => d.trade_id === selectedCamp.trade_id);
    const curShares = selectedCamp.shares || 0;
    const avgEntry = selectedCamp.avg_entry || 0;
    const livePrice = campPrice || avgEntry;
    const currentReturn = avgEntry > 0 ? ((livePrice - avgEntry) / avgEntry) * 100 : 0;
    const currentValue = curShares * livePrice;
    const currentPosPct = equity > 0 ? (currentValue / equity) * 100 : 0;

    // Rebuild LIFO inventory to get last lot price
    const sorted = [...campDetails].sort((a, b) => {
      const da = String(a.date || ""); const db = String(b.date || "");
      if (da !== db) return da.localeCompare(db);
      return (String(a.action).toUpperCase() === "BUY" ? 0 : 1) - (String(b.action).toUpperCase() === "BUY" ? 0 : 1);
    });
    const inv: { qty: number; price: number; stop: number }[] = [];
    for (const tx of sorted) {
      const action = String(tx.action || "").toUpperCase();
      const txShares = Math.abs(parseFloat(String(tx.shares || 0)));
      if (action === "BUY") {
        let p = parseFloat(String(tx.amount || 0)); if (p === 0) p = avgEntry;
        let s = parseFloat(String(tx.stop_loss || 0)); if (s === 0) s = p;
        inv.push({ qty: txShares, price: p, stop: s });
      } else if (action === "SELL") {
        let toSell = txShares;
        while (toSell > 0 && inv.length > 0) {
          const last = inv[inv.length - 1];
          const take = Math.min(toSell, last.qty);
          last.qty -= take; toSell -= take;
          if (last.qty < 0.00001) inv.pop();
        }
      }
    }

    // Last lot return
    const lastLot = inv.length > 0 ? inv[inv.length - 1] : null;
    const lastLotPrice = lastLot?.price || avgEntry;
    const lastLotReturn = lastLotPrice > 0 ? ((livePrice - lastLotPrice) / lastLotPrice) * 100 : 0;

    // Pyramid check: last lot must be up >= 5%, max add = 20% of current shares
    const pyramidReady = lastLotReturn >= 5;
    const maxPyramidShares = Math.floor(curShares * 0.20);

    // Post-add projection
    const addShares = sharesNum || 0;
    const addPrice = priceNum || livePrice;
    const newTotalShares = curShares + addShares;
    const newAvgCost = newTotalShares > 0 ? ((curShares * avgEntry) + (addShares * addPrice)) / newTotalShares : 0;
    const newValue = newTotalShares * livePrice;
    const newPosPct = equity > 0 ? (newValue / equity) * 100 : 0;

    // Stop rules based on current return
    // >= 20% → stop at least entry + 5% (lock gains)
    // >= 10% → stop at break-even
    // otherwise → original risk budget stop
    let stopRule = "";
    let minStop = 0;
    let riskFreeAdd = false;
    if (currentReturn >= 20) {
      minStop = avgEntry * 1.05;
      stopRule = "Up 20%+ → stop at +5% gain minimum";
      riskFreeAdd = true;
    } else if (currentReturn >= 10) {
      minStop = avgEntry;
      stopRule = "Up 10%+ → stop at break-even";
      riskFreeAdd = true;
    }

    // Combined stop: weighted stop across all lots that keeps risk within budget
    const addStop = stopPrice > 0 ? stopPrice : (addPrice > 0 ? addPrice * 0.92 : 0);
    let combinedStop = 0;
    if (newTotalShares > 0) {
      // Weighted stop from existing lots + new add
      let totalWeightedStop = 0;
      let totalQty = 0;
      for (const lot of inv) {
        if (lot.qty > 0) {
          totalWeightedStop += lot.qty * lot.stop;
          totalQty += lot.qty;
        }
      }
      if (addShares > 0 && addStop > 0) {
        totalWeightedStop += addShares * addStop;
        totalQty += addShares;
      }
      combinedStop = totalQty > 0 ? totalWeightedStop / totalQty : 0;
    }

    // If stop rules require a higher stop, enforce that
    if (minStop > 0 && combinedStop < minStop) {
      combinedStop = minStop;
    }

    // Risk with combined stop
    const combinedRisk = combinedStop > 0 && newTotalShares > 0
      ? Math.max(0, (newAvgCost - combinedStop) * newTotalShares) : 0;
    const combinedRiskPct = equity > 0 ? (combinedRisk / equity) * 100 : 0;

    return {
      curShares, avgEntry, livePrice, currentReturn, currentValue, currentPosPct,
      lastLotPrice, lastLotReturn, pyramidReady, maxPyramidShares,
      newTotalShares, newAvgCost, newValue, newPosPct,
      stopRule, minStop, riskFreeAdd, combinedStop, combinedRisk, combinedRiskPct,
      addShares, addPrice,
    };
  }, [actionType, selectedCamp, allDetails, campPrice, equity, sharesNum, priceNum, stopPrice]);

  const validate = () => {
    const e: string[] = [];
    const w: string[] = [];
    const t = actionType === "scalein" ? selectedCamp?.ticker || "" : ticker;
    if (!t.trim()) e.push("Ticker is required");
    if (!rule.trim()) e.push("Buy Rule is required");
    if (!strategy.trim()) e.push("Strategy is required");
    if (sharesNum <= 0) e.push("Shares must be > 0");
    if (priceNum <= 0) e.push("Price must be > 0");
    // Stop-related checks skip when the user has opted into no-stop
    // (showStopLoss=false, only possible for options). The < 8%
    // recommendation is additionally suppressed for any option ticker
    // even when revealed — it's a stock-price-distance heuristic that
    // doesn't translate to premium-based stops.
    if (showStopLoss && stopPrice > 0 && stopPrice >= priceNum) e.push("Stop must be below entry price");
    if (!isOption && stopPct > 10) w.push(`Stop is ${stopPct.toFixed(1)}% wide — recommend < 8%`);
    if (posSizePct > 25) e.push(`Position size ${posSizePct.toFixed(1)}% exceeds 25% max`);
    if (riskViolation) w.push(`Trade Risk ${formatCurrency(riskDollars, { decimals: 0 })} exceeds Risk Budget ${formatCurrency(riskBudget, { decimals: 0 })}. Move stop to ${formatCurrency(rbmStop)} to stay within Risk Budget.`);
    setErrors(e); setWarnings(w);
    return e.length === 0;
  };

  const [submitting, setSubmitting] = useState(false);
  const [submitResult, setSubmitResult] = useState<{ ok: boolean; msg: string } | null>(null);

  // Background upload tracker — populated when a submit succeeds, then
  // updated as each upload resolves or fails. Separate from `submitting`
  // so the submit button re-enables once the DB write is done; uploads
  // run independently and surface their status here.
  const [uploadEntries, setUploadEntries] = useState<UploadEntry[]>([]);
  const uploadEntriesRef = useRef<UploadEntry[]>([]);
  uploadEntriesRef.current = uploadEntries;

  const fireUpload = useCallback((entry: UploadEntry) => {
    uploadWithTimeout(entry.file, entry.portfolio, entry.tradeId, entry.ticker, entry.kind)
      .then(result => {
        setUploadEntries(prev => prev.map(e =>
          e.id === entry.id
            ? { ...e, status: result.ok ? "done" : "failed", error: result.error }
            : e,
        ));
      });
  }, []);

  const onRetryUpload = useCallback((id: string) => {
    const entry = uploadEntriesRef.current.find(e => e.id === id);
    if (!entry) return;
    setUploadEntries(prev => prev.map(e =>
      e.id === id ? { ...e, status: "uploading", error: undefined } : e,
    ));
    fireUpload({ ...entry, status: "uploading", error: undefined });
  }, [fireUpload]);

  const onDismissTracker = useCallback(() => setUploadEntries([]), []);

  const handleSubmit = async () => {
    if (!validate()) return;
    setSubmitting(true);
    setSubmitResult(null);

    try {
      const body = {
        portfolio: getActivePortfolio(),
        action_type: actionType,
        ticker: actionType === "scalein" ? (selectedCamp?.ticker || "") : ticker,
        trade_id: actionType === "scalein" ? selectedCampaign : tradeId,
        shares: parseFloat(shares),
        price: parseFloat(price),
        stop_loss: showStopLoss
          ? (stopMode === "price"
              ? parseFloat(stopValue)
              : stopMode === "atr"
                ? parseFloat(price) * (1 - (atrMultiplier * atrPct) / 100)
                : parseFloat(price) * (1 - parseFloat(slPct) / 100))
          : null,
        rule,
        strategy,
        notes,
        date: date,
        time: time,
      };

      const result = await api.logBuy(body);

      if (result.error) {
        setSubmitResult({ ok: false, msg: result.error });
      } else {
        // Snapshot attached files into the upload tracker, then fire each
        // upload in the background. The submit chain no longer blocks on
        // them — a stalled R2 / Vision call can't hang the "Saving…"
        // button anymore. Per-file status (uploading / done / failed) is
        // rendered by <UploadTracker> with a Retry button on failures.
        const tid = actionType === "scalein" ? selectedCampaign : tradeId;
        const portfolio = getActivePortfolio();
        const entriesToFire: UploadEntry[] = [];
        const mk = (file: File, kind: UploadKind): UploadEntry => ({
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}-${entriesToFire.length}`,
          file, fileName: file.name, kind, portfolio, tradeId: tid, ticker, status: "uploading",
        });
        for (const f of entryCharts) entriesToFire.push(mk(f, "entry"));
        for (const f of positionCharts) entriesToFire.push(mk(f, "position_change"));
        if (msScreenshot) entriesToFire.push(mk(msScreenshot, "marketsurge"));

        if (entriesToFire.length > 0) {
          setUploadEntries(prev => [...prev, ...entriesToFire]);
          for (const entry of entriesToFire) fireUpload(entry);
        }

        setSubmitResult({ ok: true, msg: `Logged ${result.trx_id || "B1"}: ${shares} shs of ${ticker} @ $${price}` });
        // Reset form — file lists are cleared here, but the File refs are
        // already captured in uploadEntries so the background uploads are
        // unaffected.
        setTicker(""); setShares(""); setPrice(""); setStopValue(""); setNotes(""); setRule("");
        setStrategy("CanSlim");
        setEntryCharts([]); setPositionCharts([]); setMsScreenshot(null);

        // Refresh server-derived state so a same-page second submit reads
        // fresh data. Without this, openTrades + allDetails stay at the
        // pre-submit snapshot — scale-ins see old share counts, the next
        // "new" buy reuses the just-consumed trade_id, etc.
        const [refreshedOpen, refreshedDetails] = await Promise.all([
          api.tradesOpen(getActivePortfolio()).catch(() => [] as TradePosition[]),
          api.tradesOpenDetails(getActivePortfolio()).catch(() => ({ details: [] as TradeDetail[], lot_closures: [] })),
        ]);
        setOpenTrades(refreshedOpen as TradePosition[]);
        setAllDetails(refreshedDetails.details);
        if (actionType === "new") {
          setTradeId("");
          const next = await api.nextTradeId(getActivePortfolio(), date).catch(() => ({ trade_id: "" }));
          if ((next as { trade_id?: string }).trade_id) setTradeId(String((next as { trade_id?: string }).trade_id));
        }
      }
    } catch (err: any) {
      setSubmitResult({ ok: false, msg: err.message || "Failed to log buy" });
    }

    setSubmitting(false);
  };

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          Log a <em className="italic" style={{ color: navColor }}>Buy</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>Entry, add-on, or pyramid</div>
      </div>

      <div className="grid gap-6" style={{ gridTemplateColumns: "3fr 2fr", alignItems: "start" }}>
        {/* ════════════════════════════════════════════════════════
            LEFT — Trade Details (scrollable form)
            ════════════════════════════════════════════════════════ */}
        <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
          <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
            <span className="text-[13px] font-semibold">Trade Details</span>
          </div>
          <div className="p-5 flex flex-col gap-5">

            {/* Action Type + Date + Time */}
            <div className="grid grid-cols-[2fr_1fr] gap-4">
              <Field label="Action Type">
                <div className="flex gap-4 mt-1">
                  <Radio checked={actionType === "new"} onClick={() => { setActionType("new"); setSelectedCampaign(""); }} label="Start New Campaign" />
                  <Radio checked={actionType === "scalein"} onClick={() => setActionType("scalein")} label="Scale In (Add to Existing)" />
                </div>
              </Field>
              <div className="flex flex-col gap-3">
                <Field label="Date">
                  <input type="date" value={date} onChange={e => setDate(e.target.value)} className={inputCls} style={inputStyle} />
                </Field>
                <Field label="Time">
                  <input type="time" value={time} onChange={e => setTime(e.target.value)} className={inputCls} style={inputStyle} />
                </Field>
              </div>
            </div>

            {/* Ticker + Trade ID (new) or Campaign Picker (scale-in) */}
            {actionType === "new" ? (
              <>
                <div className="grid grid-cols-2 gap-4">
                  <Field label="Ticker Symbol">
                    <input type="text" value={ticker} onChange={e => setTicker(e.target.value.toUpperCase())}
                           className={inputCls} style={inputStyle} />
                  </Field>
                  <Field label="Trade ID">
                    <input type="text" value={tradeId} onChange={e => setTradeId(e.target.value)}
                           className={inputCls} style={inputStyle} />
                  </Field>
                </div>
                {isOption && (
                  <div className="text-[12px] px-3 py-2 rounded-[8px] flex items-center gap-2"
                       style={{ background: "color-mix(in oklab, #f59f00 10%, var(--surface))", color: "#92400e" }}>
                    <span className="font-semibold">OPTION ×{multiplier}</span>
                    <span style={{ color: "var(--ink-3)" }}>·</span>
                    <span>Stop default 50% of premium · all dollar fields below shown as notional</span>
                  </div>
                )}
              </>
            ) : (
              <Field label="Select Existing Campaign">
                <SearchSelect
                  value={selectedCampaign ? `${openTrades.find(t => t.trade_id === selectedCampaign)?.ticker || ""} | ${selectedCampaign}` : ""}
                  onChange={(v) => {
                    const id = v.split(" | ")[1]?.trim() || "";
                    setSelectedCampaign(id);
                  }}
                  options={openTrades.map(t => `${t.ticker} | ${t.trade_id}`)}
                  placeholder="Search campaigns..."
                />
                {selectedCamp && (
                  <div className="mt-2 text-[12px] px-3 py-2 rounded-[8px]" style={{ background: "var(--bg)", color: "var(--ink-3)" }}>
                    Holding: {selectedCamp.shares} shs @ {formatCurrency(parseFloat(String(selectedCamp.avg_entry || 0)))}
                  </div>
                )}
              </Field>
            )}

            {/* Buy Rule (searchable) */}
            <Field label="Buy Rule *">
              <SearchSelect value={rule} onChange={setRule} options={BUY_RULES} placeholder="Type to search rules..." />
            </Field>

            {/* Strategy (Migration 019). Read-only on scale-in (inherited
                from parent campaign — strategy is fixed at creation and
                must never drift mid-campaign). Phase 2: visual swatch
                rendering moved to <StrategyChip>. The structured option
                still has an inline-only label so the search filter still
                matches against the strategy name. */}
            <Field label="Strategy *">
              <SearchSelect
                value={strategy}
                onChange={setStrategy}
                disabled={actionType === "scalein"}
                options={strategies.map(s => ({
                  value: s.name,
                  label: s.name,
                  renderPrefix: () => <StrategyChip name={s.name} color={s.color} size="sm" showName={false} />,
                }))}
                placeholder="Select strategy..."
              />
              {actionType === "scalein" && (
                <div className="mt-1.5 text-[11px]" style={{ color: "var(--ink-4)" }}>
                  Inherited from parent campaign
                </div>
              )}
            </Field>

            {/* Shares + Price */}
            <div className="grid grid-cols-2 gap-4">
              <Field label={`${unitLabel} to Add`}>
                <input type="number" value={shares} onChange={e => setShares(e.target.value)}
                       min="0" step="1" placeholder="0" className={inputCls} style={inputStyle} />
              </Field>
              <Field label={isOption ? "Premium per Contract ($)" : "Price ($)"}>
                <input type="number" value={price} onChange={e => setPrice(e.target.value)}
                       min="0" step="0.01" placeholder="0.00" className={inputCls} style={inputStyle} />
              </Field>
            </div>
            {isOption && sharesNum > 0 && priceNum > 0 && (
              <div className="text-[11px] -mt-2" style={{ color: "var(--ink-4)", fontFamily: "var(--font-jetbrains), monospace" }}>
                {sharesNum} × {formatCurrency(priceNum)} × {multiplier} = {formatCurrency(totalCost)} premium
              </div>
            )}

            {/* Stop Loss — hidden by default for options; revealed via the
                small link below. Stock trades always render the full block. */}
            {showStopLoss ? (
              <div className="flex flex-col gap-2">
                <div className="grid grid-cols-2 gap-4">
                  <Field label="Stop Loss Mode">
                    <div className="flex gap-4 mt-1 flex-wrap">
                      <Radio checked={stopMode === "price"} onClick={() => setStopMode("price")} label="Price Level ($)" />
                      <Radio checked={stopMode === "pct"} onClick={() => setStopMode("pct")} label="Percentage (%)" />
                      {/* ATR radio hidden on option tickers — options don't
                          have meaningful underlying-share ATR for sizing. */}
                      {!isOption && (
                        <Radio checked={stopMode === "atr"} onClick={() => setStopMode("atr")} label="ATR (×)" />
                      )}
                    </div>
                  </Field>
                  <Field label={stopMode === "price" ? "Stop Price ($)" : stopMode === "atr" ? "ATR Multiplier" : "Stop Loss %"}>
                    {stopMode === "price" ? (
                      <input type="number" value={stopValue} onChange={e => setStopValue(e.target.value)}
                             step="0.01" placeholder="0.00" className={inputCls} style={inputStyle} />
                    ) : stopMode === "atr" ? (
                      <div className="flex gap-2 mt-1" data-testid="logbuy-atr-pills">
                        {([1, 1.5, 2] as const).map(m => {
                          const selected = atrMultiplier === m;
                          const disabled = atrPct === 0;
                          return (
                            <button key={m} type="button"
                                    onClick={() => { if (!disabled) setAtrMultiplier(m); }}
                                    disabled={disabled}
                                    aria-pressed={selected}
                                    aria-label={`${m}× ATR`}
                                    className="px-3 py-1.5 rounded-[8px] text-[12px] font-semibold transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                                    style={{
                                      background: selected ? "var(--ink-1)" : "var(--bg)",
                                      color: selected ? "var(--surface)" : "var(--ink-2)",
                                      border: `1px solid ${selected ? "var(--ink-1)" : "var(--border)"}`,
                                      cursor: disabled ? "not-allowed" : "pointer",
                                    }}>
                              {m}×
                            </button>
                          );
                        })}
                      </div>
                    ) : (
                      <input type="number" value={slPct} onChange={e => setSlPct(e.target.value)}
                             step="0.5" placeholder="5.0" className={inputCls} style={inputStyle} />
                    )}
                  </Field>
                </div>
                {/* ATR mode confirmation line — the resolved stop preview.
                    Unavailability notice when atrPct = 0 (sparse history or
                    fetch failure). The "Default mode for buys with no stop…"
                    explanatory caption from the original mockup was removed
                    post-deployment review — users selecting ATR mode know
                    what ATR mode is. */}
                {stopMode === "atr" && (
                  atrPct > 0 ? (
                    <div className="text-[12px] font-medium mt-0.5" style={{ color: "#3b82f6", fontFamily: "var(--font-jetbrains), monospace" }}>
                      → Stop {formatCurrency(stopPrice)} · {(atrMultiplier * atrPct).toFixed(1)}% below entry
                    </div>
                  ) : (
                    <div className="text-[12px] mt-0.5" style={{ color: "var(--ink-4)" }}>
                      ATR unavailable for this ticker (insufficient history). Use Price or Percentage mode.
                    </div>
                  )
                )}
              </div>
            ) : (
              <button type="button" onClick={() => setShowStopLoss(true)}
                      className="self-start text-[12px] hover:underline cursor-pointer"
                      style={{ color: "var(--ink-4)", background: "transparent", border: "none", padding: 0 }}>
                Show stop loss
              </button>
            )}

            {/* Notes */}
            <Field label="Buy Rationale (Notes)">
              <input type="text" value={notes} onChange={e => setNotes(e.target.value)}
                     className={inputCls} style={{ ...inputStyle, fontFamily: "inherit" }} />
            </Field>

            {/* Chart Documentation */}
            <div>
              <div className="flex items-center gap-2 mb-3 mt-1">
                <span className="text-[14px]">📸</span>
                <span className="text-[13px] font-semibold">Chart Documentation (Optional)</span>
              </div>
              <div className="grid grid-cols-2 gap-3">
                {[
                  { label: "Entry Charts (Weekly / Daily)", icon: "📈", files: entryCharts, setFiles: setEntryCharts },
                  { label: "Position Changes (Add-ons / Trims / Exits)", icon: "🔄", files: positionCharts, setFiles: setPositionCharts },
                ].map(slot => (
                  <DropZone key={slot.label} label={slot.label} icon={slot.icon} files={slot.files}
                            onFiles={slot.setFiles} multiple accept="image/png,image/jpeg,application/pdf" />
                ))}
              </div>
            </div>

            {/* MarketSurge */}
            <div>
              <div className="flex items-center gap-2 mb-2 mt-1">
                <span className="text-[14px]">🔬</span>
                <span className="text-[13px] font-semibold">MarketSurge Fundamentals (Optional)</span>
              </div>
              <div className="text-[11px] mb-2" style={{ color: "var(--ink-4)" }}>
                Upload a MarketSurge screenshot to auto-extract ratings and fundamentals via AI.
              </div>
              <DropZone label="" icon="" files={msScreenshot ? [msScreenshot] : []}
                        onFiles={fs => setMsScreenshot(fs[0] || null)} accept="image/png,image/jpeg,application/pdf" />
            </div>

            {/* Errors + Warnings */}
            {errors.length > 0 && (
              <div className="flex flex-col gap-1.5">
                {errors.map((e, i) => (
                  <div key={i} className="text-[12px] font-medium px-3 py-2 rounded-[8px]" style={{ background: "color-mix(in oklab, #e5484d 10%, var(--surface))", color: "#dc2626", border: "1px solid color-mix(in oklab, #e5484d 30%, var(--border))" }}>{e}</div>
                ))}
              </div>
            )}
            {warnings.length > 0 && (
              <div className="flex flex-col gap-1.5">
                {warnings.map((w, i) => (
                  <div key={i} className="text-[12px] font-medium px-3 py-2 rounded-[8px]" style={{ background: "color-mix(in oklab, #f59f00 10%, var(--surface))", color: "#d97706", border: "1px solid color-mix(in oklab, #f59f00 30%, var(--border))" }}>{w}</div>
                ))}
              </div>
            )}

            {/* Submit */}
            {submitResult && (
              <div className="mb-3 px-4 py-2.5 rounded-[10px] text-[12px] font-medium"
                   style={{
                     background: submitResult.ok ? "color-mix(in oklab, #08a86b 10%, var(--surface))" : "color-mix(in oklab, #e5484d 10%, var(--surface))",
                     color: submitResult.ok ? "#08a86b" : "#e5484d",
                     border: `1px solid ${submitResult.ok ? "color-mix(in oklab, #08a86b 30%, var(--border))" : "color-mix(in oklab, #e5484d 30%, var(--border))"}`,
                   }}>
                {submitResult.ok ? "✅" : "❌"} {submitResult.msg}
              </div>
            )}
            <UploadTracker entries={uploadEntries} onRetry={onRetryUpload} onDismiss={onDismissTracker} />
            <button onClick={handleSubmit} disabled={submitting}
                    className="w-full h-[48px] rounded-[12px] text-[14px] font-semibold text-white transition-all hover:brightness-110 cursor-pointer disabled:opacity-50"
                    style={{ background: "#08a86b" }}>
              {submitting ? "Saving..." : "LOG BUY ORDER"}
            </button>
          </div>
        </div>

        {/* ════════════════════════════════════════════════════════
            RIGHT — Live Sizer / Scale-In Cockpit (sticky)
            ════════════════════════════════════════════════════════ */}
        <div className="sticky top-[72px] flex flex-col gap-4">

          {/* ── SCALE-IN COCKPIT ── */}
          {actionType === "scalein" && scaleIn ? (
            <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
              <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
                <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
                <span className="text-[13px] font-semibold">Scale-In Cockpit</span>
                <span className="text-xs" style={{ color: "var(--ink-4)" }}>{selectedCamp?.ticker || ""}</span>
              </div>
              <div className="p-5 flex flex-col gap-4">

                {/* Current Position */}
                <div>
                  <div className="text-[10px] uppercase tracking-[0.08em] font-semibold mb-2" style={{ color: "var(--ink-4)" }}>Current Position</div>
                  <div className="grid grid-cols-2 gap-2.5">
                    <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                      <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Shares</div>
                      <div className="text-[18px] font-semibold mt-0.5" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                        {scaleIn.curShares.toLocaleString()}
                      </div>
                    </div>
                    <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                      <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Avg Entry</div>
                      <div className="text-[18px] font-semibold mt-0.5 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                        {formatCurrency(scaleIn.avgEntry)}
                      </div>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-2.5 mt-2.5">
                    <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                      <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Live Price</div>
                      <div className="text-[18px] font-semibold mt-0.5 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                        {formatCurrency(scaleIn.livePrice)}
                      </div>
                    </div>
                    <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                      <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Return</div>
                      <div className="text-[18px] font-semibold mt-0.5" style={{ fontFamily: "var(--font-jetbrains), monospace", color: scaleIn.currentReturn >= 0 ? "#16a34a" : "#e5484d" }}>
                        {scaleIn.currentReturn >= 0 ? "+" : ""}{scaleIn.currentReturn.toFixed(2)}%
                      </div>
                    </div>
                  </div>
                  <div className="mt-2.5 text-[12px] font-medium px-3 py-1.5 rounded-[8px] privacy-mask" style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink-3)" }}>
                    Position: {scaleIn.currentPosPct.toFixed(1)}% of NLV · {formatCurrency(scaleIn.currentValue, { decimals: 0 })}
                  </div>
                </div>

                {/* Divider */}
                <div style={{ borderTop: "1px solid var(--border)" }} />

                {/* Pyramid Check */}
                <div>
                  <div className="text-[10px] uppercase tracking-[0.08em] font-semibold mb-2" style={{ color: "var(--ink-4)" }}>Pyramid Check</div>
                  <div className="px-3 py-2.5 rounded-[10px] text-[12px]" style={{
                    background: scaleIn.pyramidReady ? "color-mix(in oklab, #08a86b 10%, var(--surface))" : "color-mix(in oklab, #f59f00 10%, var(--surface))",
                    border: `1px solid ${scaleIn.pyramidReady ? "color-mix(in oklab, #08a86b 30%, var(--border))" : "color-mix(in oklab, #f59f00 30%, var(--border))"}`,
                    color: scaleIn.pyramidReady ? "#16a34a" : "#d97706",
                  }}>
                    <div className="font-semibold mb-1">
                      {scaleIn.pyramidReady ? "Pyramid Ready" : "Not Ready"}
                    </div>
                    <div style={{ color: "var(--ink-3)" }}>
                      Last lot @ {formatCurrency(scaleIn.lastLotPrice)} → {scaleIn.lastLotReturn >= 0 ? "+" : ""}{scaleIn.lastLotReturn.toFixed(2)}%
                      {!scaleIn.pyramidReady && " (need +5%)"}
                    </div>
                  </div>
                  <div className="grid grid-cols-1 gap-2.5 mt-2.5">
                    <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                      <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Max Pyramid Add (20%)</div>
                      <div className="text-[18px] font-semibold mt-0.5" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                        {scaleIn.maxPyramidShares} shs
                        <span className="text-[12px] font-normal ml-2 privacy-mask" style={{ color: "var(--ink-4)" }}>
                          ~{formatCurrency(scaleIn.maxPyramidShares * scaleIn.livePrice, { decimals: 0 })}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Divider */}
                <div style={{ borderTop: "1px solid var(--border)" }} />

                {/* Post-Add Projection */}
                {scaleIn.addShares > 0 && (
                  <>
                    <div>
                      <div className="text-[10px] uppercase tracking-[0.08em] font-semibold mb-2" style={{ color: "var(--ink-4)" }}>Post-Add Projection</div>
                      <div className="grid grid-cols-2 gap-2.5">
                        <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                          <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>New Shares</div>
                          <div className="text-[18px] font-semibold mt-0.5" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                            {scaleIn.newTotalShares.toLocaleString()}
                          </div>
                        </div>
                        <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                          <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>New Avg Cost</div>
                          <div className="text-[18px] font-semibold mt-0.5 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                            {formatCurrency(scaleIn.newAvgCost)}
                          </div>
                        </div>
                      </div>
                      <div className="mt-2.5 text-[12px] font-medium px-3 py-1.5 rounded-[8px] privacy-mask"
                           style={{
                             background: scaleIn.newPosPct > 25 ? "color-mix(in oklab, #e5484d 10%, var(--surface))" : scaleIn.newPosPct > 15 ? "color-mix(in oklab, #f59f00 10%, var(--surface))" : "color-mix(in oklab, #08a86b 10%, var(--surface))",
                             color: scaleIn.newPosPct > 25 ? "#dc2626" : scaleIn.newPosPct > 15 ? "#d97706" : "#16a34a",
                             border: `1px solid ${scaleIn.newPosPct > 25 ? "color-mix(in oklab, #e5484d 30%, var(--border))" : scaleIn.newPosPct > 15 ? "color-mix(in oklab, #f59f00 30%, var(--border))" : "color-mix(in oklab, #08a86b 30%, var(--border))"}`,
                           }}>
                        New position: {scaleIn.newPosPct.toFixed(1)}% of NLV · {formatCurrency(scaleIn.newValue, { decimals: 0 })}
                      </div>
                    </div>

                    {/* Divider */}
                    <div style={{ borderTop: "1px solid var(--border)" }} />
                  </>
                )}

                {/* Stop & Risk */}
                <div>
                  <div className="text-[10px] uppercase tracking-[0.08em] font-semibold mb-2" style={{ color: "var(--ink-4)" }}>Combined Stop & Risk</div>

                  {/* Stop rule status */}
                  {scaleIn.stopRule && (
                    <div className="px-3 py-2.5 rounded-[10px] text-[12px] mb-2.5" style={{
                      background: "color-mix(in oklab, #08a86b 10%, var(--surface))",
                      border: "1px solid color-mix(in oklab, #08a86b 30%, var(--border))",
                      color: "#16a34a",
                    }}>
                      <div className="font-semibold">Risk-Free Add</div>
                      <div style={{ color: "var(--ink-3)" }}>{scaleIn.stopRule}</div>
                      <div className="font-semibold mt-1">Min stop: {formatCurrency(scaleIn.minStop)}</div>
                    </div>
                  )}

                  <div className="grid grid-cols-2 gap-2.5">
                    <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                      <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Combined Stop</div>
                      <div className="text-[18px] font-semibold mt-0.5" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                        {scaleIn.combinedStop > 0 ? formatCurrency(scaleIn.combinedStop) : "—"}
                      </div>
                    </div>
                    <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                      <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Total Risk</div>
                      <div className="text-[18px] font-semibold mt-0.5 privacy-mask" style={{
                        fontFamily: "var(--font-jetbrains), monospace",
                        color: scaleIn.riskFreeAdd ? "#16a34a" : scaleIn.combinedRiskPct > 1 ? "#e5484d" : "var(--ink)",
                      }}>
                        {scaleIn.riskFreeAdd ? "$0" : scaleIn.combinedRisk > 0 ? formatCurrency(scaleIn.combinedRisk, { decimals: 0 }) : "—"}
                      </div>
                      {!scaleIn.riskFreeAdd && scaleIn.combinedRisk > 0 && (
                        <div className="text-[10px] mt-0.5" style={{ color: "var(--ink-4)" }}>
                          {scaleIn.combinedRiskPct.toFixed(2)}% of NLV
                        </div>
                      )}
                    </div>
                  </div>
                </div>

              </div>
            </div>

          ) : (
            /* ── NEW TRADE SIZER (original) ── */
            <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
              <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
                <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
                <span className="text-[13px] font-semibold">Live Sizer</span>
                <span className="text-xs" style={{ color: "var(--ink-4)" }}>Risk-based position sizing</span>
              </div>
              <div className="p-5 flex flex-col gap-4">

                {/* Sizing mode — MCT-driven by default; user can override
                    for this Log Buy submission. Override is form-local
                    (component remount on submit / refresh re-applies the
                    auto pick). riskPctInput below reads from sizingMode
                    either way, so the override flows into the saved buy. */}
                <div className="px-3 py-2 rounded-[8px] text-[12px] flex items-center justify-between gap-3 flex-wrap"
                     data-testid="logbuy-sizing-mode-indicator"
                     style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
                  <div>
                    <span style={{ color: "var(--ink-4)" }}>Sizing:</span>{" "}
                    <span className="font-semibold">
                      {SIZING_MODES[sizingMode].icon}{" "}
                      {SIZING_MODES[sizingMode].key.charAt(0).toUpperCase() + SIZING_MODES[sizingMode].key.slice(1)}{" "}
                      ({SIZING_MODES[sizingMode].pct.toFixed(2)}% risk)
                    </span>
                    <span style={{ color: "var(--ink-4)" }}>
                      {" "}{sizingModeManual ? "— manual override" : describeMctSource(mctState, exitLadderFloor(activeExits))}
                    </span>
                  </div>
                  {sizingModeManual && (
                    <button type="button"
                            data-testid="logbuy-reset-to-mct"
                            onClick={() => {
                              // Re-apply the combined auto rule so the
                              // exit-ladder floor still binds after a
                              // reset (matches Position Sizer behavior).
                              setSizingMode(deriveAutoSizingMode(mctState, activeExits).idx);
                              setSizingModeManual(false);
                            }}
                            className="text-[11px] px-2 py-0.5 rounded-[6px] underline"
                            style={{ color: "var(--ink-3)" }}>
                      Reset to MCT
                    </button>
                  )}
                </div>

                {/* Sizing mode override radios. Same SIZING_MODES + index
                    contract as Position Sizer (defense=0, normal=1,
                    offense=2). Clicking any radio flips sizingModeManual
                    so the indicator copy + Reset button surface above. */}
                <Field label="Override Sizing Mode">
                  <div className="flex flex-col gap-1.5 mt-1">
                    {SIZING_MODES.map((m, i) => (
                      <Radio key={m.key}
                             checked={sizingMode === i}
                             onClick={() => {
                               setSizingMode(i as 0 | 1 | 2);
                               setSizingModeManual(true);
                             }}
                             label={`${m.icon} ${m.label}`} />
                    ))}
                  </div>
                </Field>

                {/* Account Equity */}
                <Field label="Account Equity">
                  <div className="h-[42px] px-3.5 rounded-[10px] flex items-center text-[15px] font-semibold privacy-mask"
                       style={{ background: "var(--bg)", border: "1px solid var(--border)", fontFamily: "var(--font-jetbrains), monospace" }}>
                    {formatCurrency(equity, { decimals: 0 })}
                  </div>
                </Field>

                {/* ATR informational row. Renders only when the user has
                    entered a ticker (priceLookup has fired). Subtle styling
                    mirrors the sizing-mode indicator copy above — this is
                    metadata for the trader, not a primary metric. atrPct=0
                    is the backend's "insufficient history" sentinel; we
                    surface that here so the unavailability matches the
                    pills-disabled state on the left form. */}
                {ticker.trim().length > 0 && (
                  <div className="px-3 py-2 rounded-[8px] text-[12px]"
                       data-testid="logbuy-atr-info"
                       style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
                    <span style={{ color: "var(--ink-4)" }}>ATR (21d):</span>{" "}
                    {atrPct > 0 ? (
                      <>
                        <span className="font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                          {atrPct.toFixed(2)}%
                        </span>
                        <span style={{ color: "var(--ink-4)" }}> · </span>
                        <span className="font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                          {formatCurrency(priceNum * atrPct / 100)}/sh
                        </span>
                      </>
                    ) : (
                      <span style={{ color: "var(--ink-4)" }}>unavailable for this ticker</span>
                    )}
                  </div>
                )}

                {/* Risk Budget + Stop Dist */}
                <div className="grid grid-cols-2 gap-2.5">
                  <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                    <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Risk $</div>
                    <div className="text-[20px] font-semibold mt-0.5 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                      {formatCurrency(riskBudget, { decimals: 0 })}
                    </div>
                  </div>
                  <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                    <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Stop Dist</div>
                    <div className="text-[20px] font-semibold mt-0.5" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                      {stopDist > 0 ? formatCurrency(stopDist) : "—"}
                    </div>
                  </div>
                </div>

                {/* Shares + Cost */}
                <div className="grid grid-cols-2 gap-2.5">
                  <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                    <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Shares</div>
                    <div className="text-[20px] font-semibold mt-0.5" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                      {recommendedShares > 0 ? recommendedShares.toLocaleString() : sharesNum > 0 ? sharesNum.toLocaleString() : "—"}
                    </div>
                  </div>
                  <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                    <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Cost</div>
                    <div className="text-[20px] font-semibold mt-0.5 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                      {recommendedCost > 0 ? formatCurrency(recommendedCost, { decimals: 0 }) : totalCost > 0 ? formatCurrency(totalCost, { decimals: 0 }) : "—"}
                    </div>
                  </div>
                </div>

                {/* Position weight + rule check */}
                {priceNum > 0 && (sharesNum > 0 || recommendedShares > 0) && (
                  <>
                    <div className="text-[12px] font-medium px-3 py-2 rounded-[8px]"
                         style={{
                           background: posSizePct > 25 ? "color-mix(in oklab, #e5484d 10%, var(--surface))" : posSizePct > 15 ? "color-mix(in oklab, #f59f00 10%, var(--surface))" : "color-mix(in oklab, #08a86b 10%, var(--surface))",
                           color: posSizePct > 25 ? "#dc2626" : posSizePct > 15 ? "#d97706" : "#16a34a",
                           border: `1px solid ${posSizePct > 25 ? "color-mix(in oklab, #e5484d 30%, var(--border))" : posSizePct > 15 ? "color-mix(in oklab, #f59f00 30%, var(--border))" : "color-mix(in oklab, #08a86b 30%, var(--border))"}`,
                         }}>
                      {posSizePct.toFixed(2)}% of NLV
                      {stopPct > 0 && ` · within ${stopPct.toFixed(1)}% rule`}
                      {stopPct > 0 && stopPct <= 8 ? " ✓" : ""}
                    </div>

                    {stopPrice > 0 && (
                      <div className="text-[12px] font-medium px-3 py-2 rounded-[8px]"
                           style={{
                             background: riskViolation ? "color-mix(in oklab, #e5484d 10%, var(--surface))" : withinBudget ? "color-mix(in oklab, #08a86b 10%, var(--surface))" : "var(--bg)",
                             color: riskViolation ? "#dc2626" : withinBudget ? "#16a34a" : "var(--ink-3)",
                             border: `1px solid ${riskViolation ? "color-mix(in oklab, #e5484d 30%, var(--border))" : withinBudget ? "color-mix(in oklab, #08a86b 30%, var(--border))" : "var(--border)"}`,
                           }}>
                        <span className="font-semibold">Rule check</span>
                        <br />
                        {riskViolation
                          ? `Trade Risk ${formatCurrency(riskDollars, { decimals: 0 })} exceeds Risk Budget ${formatCurrency(riskBudget, { decimals: 0 })}. Move stop to ${formatCurrency(rbmStop)} to stay within Risk Budget.`
                          : withinBudget
                            ? `Trade Risk ${formatCurrency(riskDollars, { decimals: 0 })} within Risk Budget ${formatCurrency(riskBudget, { decimals: 0 })} ✓`
                            : "Enter stop loss to validate"
                        }
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
