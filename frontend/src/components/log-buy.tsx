"use client";

import { useState, useEffect, useRef, useMemo } from "react";
import { api, type TradePosition, type TradeDetail } from "@/lib/api";

const BUY_RULES = [
  "br1.1 Consolidation", "br1.2 Cup w Handle", "br1.3 Cup w/o Handle", "br1.4 Double Bottom",
  "br1.5 IPO Base", "br1.6 Flat Base", "br1.7 Consolidation Pivot", "br1.8 High Tight Flag",
  "br2.1 HVE", "br2.2 HVSI", "br2.3 HV1",
  "br3.1 Reclaim 21e", "br3.2 Reclaim 50s", "br3.3 Reclaim 200s", "br3.4 Reclaim 10W", "br3.5 Reclaim 8e",
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

const SIZING_MODES = [
  { key: "defense", label: "Defense (0.50%)", pct: 0.5, icon: "🛡️" },
  { key: "normal", label: "Normal (0.75%)", pct: 0.75, icon: "⚖️" },
  { key: "offense", label: "Offense (1.00%)", pct: 1.0, icon: "⚔️" },
];

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
            <span className="text-[9px] mt-0.5" style={{ color: "var(--ink-5)" }}>PNG, JPG, JPEG</span>
          </>
        )}
        <input ref={inputRef} type="file" accept={accept} multiple={multiple} className="hidden"
               onChange={e => { if (e.target.files) onFiles(multiple ? [...files, ...Array.from(e.target.files)] : Array.from(e.target.files)); e.target.value = ""; }} />
      </div>
    </div>
  );
}

function SearchSelect({ value, onChange, options, placeholder }: {
  value: string; onChange: (v: string) => void; options: string[]; placeholder?: string;
}) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const filtered = search
    ? options.filter(o => o.toLowerCase().includes(search.toLowerCase()))
    : options;

  return (
    <div ref={ref} className="relative">
      <button type="button" onClick={() => setOpen(!open)}
              className={inputCls + " flex items-center justify-between text-left"}
              style={{ ...inputStyle, fontFamily: "inherit", cursor: "pointer" }}>
        <span style={{ opacity: value ? 1 : 0.5 }}>{value || placeholder || "Select..."}</span>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--ink-4)" strokeWidth="2"><path d="M6 9l6 6 6-6"/></svg>
      </button>
      {open && (
        <div className="absolute z-50 mt-1 w-full rounded-[10px] overflow-hidden shadow-lg"
             style={{ background: "var(--surface)", border: "1px solid var(--border)", maxHeight: 280 }}>
          <div className="p-2" style={{ borderBottom: "1px solid var(--border)" }}>
            <input type="text" value={search} onChange={e => setSearch(e.target.value)}
                   placeholder="Type to search rules..." autoFocus
                   className="w-full h-[34px] px-3 rounded-[8px] text-[12px] outline-none"
                   style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }} />
          </div>
          <div className="overflow-y-auto" style={{ maxHeight: 220 }}>
            {filtered.map(o => (
              <button key={o} type="button"
                      onClick={() => { onChange(o); setOpen(false); setSearch(""); }}
                      className="w-full text-left px-3 py-2 text-[12px] transition-colors hover:brightness-95"
                      style={{ background: o === value ? "var(--surface-2)" : "transparent", color: "var(--ink)" }}>
                {o}
              </button>
            ))}
            {filtered.length === 0 && (
              <div className="px-3 py-3 text-[12px] text-center" style={{ color: "var(--ink-4)" }}>No matches</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

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
  const [mFactorSuggestion, setMFactorSuggestion] = useState("Unknown");

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
  const [sizingMode, setSizingMode] = useState(1);
  const [shares, setShares] = useState("");
  const [price, setPrice] = useState("");
  const [stopMode, setStopMode] = useState<"price" | "pct">("price");
  const [stopValue, setStopValue] = useState("");
  const [slPct, setSlPct] = useState("8.0");
  const [notes, setNotes] = useState("");
  const [errors, setErrors] = useState<string[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [entryCharts, setEntryCharts] = useState<File[]>([]);
  const [positionCharts, setPositionCharts] = useState<File[]>([]);
  const [msScreenshot, setMsScreenshot] = useState<File | null>(null);

  useEffect(() => {
    Promise.all([
      api.journalLatest("CanSlim").catch(() => ({ end_nlv: 100000 })),
      api.tradesOpen("CanSlim").catch(() => []),
      api.mfactor().catch(() => ({})),
      api.tradesOpenDetails("CanSlim").catch(() => []),
    ]).then(([j, open, mf, det]) => {
      setEquity(parseFloat(String(j.end_nlv || 100000)));
      setOpenTrades(open as TradePosition[]);
      setAllDetails(det as TradeDetail[]);
      const nasdaq = (mf as any)?.nasdaq;
      if (nasdaq) {
        if (nasdaq.above_21ema && nasdaq.above_50sma) {
          setMFactorSuggestion("Powertrend"); setSizingMode(2);
        } else if (nasdaq.above_21ema) {
          setMFactorSuggestion("Open"); setSizingMode(1);
        } else {
          setMFactorSuggestion("Closed"); setSizingMode(0);
        }
      }
    });
  }, []);

  useEffect(() => {
    if (actionType === "new" && (!tradeId || tradeId.endsWith("-0XX"))) {
      api.nextTradeId("CanSlim", date).then(r => {
        if (r.trade_id) setTradeId(r.trade_id);
      }).catch(() => {
        const n = new Date();
        setTradeId(`${n.getFullYear()}${String(n.getMonth() + 1).padStart(2, "0")}-0XX`);
      });
    }
  }, [actionType]);

  // Prefill from Position Sizer (via localStorage)
  useEffect(() => {
    try {
      const raw = localStorage.getItem("ps_prefill");
      if (!raw) return;
      const data = JSON.parse(raw);
      localStorage.removeItem("ps_prefill");
      if (data.ticker) setTicker(data.ticker);
      if (data.shares) setShares(String(data.shares));
      if (data.price) setPrice(String(data.price));
      if (data.stop) setStopValue(String(data.stop.toFixed(2)));
      if (data.date) setDate(String(data.date));
      if (data.time) setTime(String(data.time));
      if (data.action === "scale_in" && data.trade_id) {
        setActionType("scalein");
        setSelectedCampaign(data.trade_id);
      } else {
        setActionType("new");
      }
    } catch { /* ignore */ }
  }, []);

  // Auto-fetch price when ticker changes (debounced)
  useEffect(() => {
    if (!ticker || ticker.length < 1 || actionType !== "new") return;
    const timeout = setTimeout(() => {
      setFetchingPrice(true);
      api.priceLookup(ticker).then(data => {
        if (data && !("error" in data)) {
          setPrice(String(data.price));
        }
      }).catch(() => {}).finally(() => setFetchingPrice(false));
    }, 600);
    return () => clearTimeout(timeout);
  }, [ticker, actionType]);

  // ── Computed values ──
  const sharesNum = parseFloat(shares) || 0;
  const priceNum = parseFloat(price) || 0;
  const totalCost = sharesNum * priceNum;
  const riskPctInput = SIZING_MODES[sizingMode].pct;
  const riskBudget = equity * (riskPctInput / 100);

  let stopPrice = 0;
  if (stopMode === "price") {
    stopPrice = parseFloat(stopValue) || 0;
  } else {
    const pct = parseFloat(slPct) || 8;
    stopPrice = priceNum > 0 ? priceNum * (1 - pct / 100) : 0;
  }
  const stopDist = priceNum > 0 && stopPrice > 0 ? priceNum - stopPrice : 0;
  const stopPct = priceNum > 0 && stopPrice > 0 ? ((priceNum - stopPrice) / priceNum) * 100 : 0;
  const riskDollars = stopDist * sharesNum;
  const posSizePct = equity > 0 ? (totalCost / equity) * 100 : 0;
  const recommendedShares = stopDist > 0 ? Math.floor(riskBudget / stopDist) : 0;
  const recommendedCost = recommendedShares * priceNum;

  const rbmStop = sharesNum > 0 && riskBudget > 0 ? priceNum - (riskBudget / sharesNum) : 0;
  const riskViolation = riskDollars > riskBudget && riskBudget > 0 && stopPrice > 0;
  const withinBudget = riskDollars > 0 && riskDollars <= riskBudget;

  const selectedCamp = openTrades.find(t => t.trade_id === selectedCampaign);

  // Fetch live price when campaign is selected for scale-in
  useEffect(() => {
    if (actionType !== "scalein" || !selectedCamp?.ticker) { setCampPrice(0); return; }
    api.priceLookup(selectedCamp.ticker).then(data => {
      if (data && !("error" in data)) setCampPrice(data.price);
    }).catch(() => {});
  }, [actionType, selectedCamp?.ticker]);

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
    if (sharesNum <= 0) e.push("Shares must be > 0");
    if (priceNum <= 0) e.push("Price must be > 0");
    if (stopPrice > 0 && stopPrice >= priceNum) e.push("Stop must be below entry price");
    if (stopPct > 10) w.push(`Stop is ${stopPct.toFixed(1)}% wide — recommend < 8%`);
    if (posSizePct > 25) e.push(`Position size ${posSizePct.toFixed(1)}% exceeds 25% max`);
    if (riskViolation) w.push(`Risk $${riskDollars.toFixed(0)} > Budget $${riskBudget.toFixed(0)}. Move stop to $${rbmStop.toFixed(2)}`);
    setErrors(e); setWarnings(w);
    return e.length === 0;
  };

  const [submitting, setSubmitting] = useState(false);
  const [submitResult, setSubmitResult] = useState<{ ok: boolean; msg: string } | null>(null);

  const handleSubmit = async () => {
    if (!validate()) return;
    setSubmitting(true);
    setSubmitResult(null);

    try {
      const body = {
        portfolio: "CanSlim",
        action_type: actionType,
        ticker,
        trade_id: actionType === "scalein" ? selectedCampaign : tradeId,
        shares: parseFloat(shares),
        price: parseFloat(price),
        stop_loss: stopMode === "price" ? parseFloat(stopValue) : parseFloat(price) * (1 - parseFloat(slPct) / 100),
        rule,
        notes,
        date: date,
        time: time,
      };

      const result = await api.logBuy(body);

      if (result.error) {
        setSubmitResult({ ok: false, msg: result.error });
      } else {
        // Upload images if any
        const tid = actionType === "scalein" ? selectedCampaign : tradeId;
        const uploadPromises: Promise<any>[] = [];
        for (const file of entryCharts) {
          uploadPromises.push(api.uploadImage(file, "CanSlim", tid, ticker, "entry"));
        }
        for (const file of positionCharts) {
          uploadPromises.push(api.uploadImage(file, "CanSlim", tid, ticker, "position_change"));
        }
        if (msScreenshot) {
          uploadPromises.push(api.uploadImage(msScreenshot, "CanSlim", tid, ticker, "marketsurge"));
        }
        if (uploadPromises.length > 0) {
          await Promise.all(uploadPromises);
        }

        setSubmitResult({ ok: true, msg: `Logged ${result.trx_id || "B1"}: ${shares} shs of ${ticker} @ $${price}` });
        // Reset form
        setTicker(""); setShares(""); setPrice(""); setStopValue(""); setNotes(""); setRule("");
        setEntryCharts([]); setPositionCharts([]); setMsScreenshot(null);
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
                    Holding: {selectedCamp.shares} shs @ ${parseFloat(String(selectedCamp.avg_entry || 0)).toFixed(2)}
                  </div>
                )}
              </Field>
            )}

            {/* Buy Rule (searchable) */}
            <Field label="Buy Rule">
              <SearchSelect value={rule} onChange={setRule} options={BUY_RULES} placeholder="Type to search rules..." />
            </Field>

            {/* Shares + Price */}
            <div className="grid grid-cols-2 gap-4">
              <Field label="Shares to Add">
                <input type="number" value={shares} onChange={e => setShares(e.target.value)}
                       min="0" step="1" placeholder="0" className={inputCls} style={inputStyle} />
              </Field>
              <Field label="Price ($)">
                <input type="number" value={price} onChange={e => setPrice(e.target.value)}
                       min="0" step="0.01" placeholder="0.00" className={inputCls} style={inputStyle} />
              </Field>
            </div>

            {/* Stop Loss */}
            <div className="grid grid-cols-2 gap-4">
              <Field label="Stop Loss Mode">
                <div className="flex gap-4 mt-1">
                  <Radio checked={stopMode === "price"} onClick={() => setStopMode("price")} label="Price Level ($)" />
                  <Radio checked={stopMode === "pct"} onClick={() => setStopMode("pct")} label="Percentage (%)" />
                </div>
              </Field>
              <Field label={stopMode === "price" ? "Stop Price ($)" : "Stop Loss %"}>
                {stopMode === "price" ? (
                  <input type="number" value={stopValue} onChange={e => setStopValue(e.target.value)}
                         step="0.01" placeholder="0.00" className={inputCls} style={inputStyle} />
                ) : (
                  <input type="number" value={slPct} onChange={e => setSlPct(e.target.value)}
                         step="0.5" placeholder="8.0" className={inputCls} style={inputStyle} />
                )}
              </Field>
            </div>

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
                            onFiles={slot.setFiles} multiple accept="image/png,image/jpeg" />
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
                        onFiles={fs => setMsScreenshot(fs[0] || null)} accept="image/png,image/jpeg" />
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
                        ${scaleIn.avgEntry.toFixed(2)}
                      </div>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-2.5 mt-2.5">
                    <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                      <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Live Price</div>
                      <div className="text-[18px] font-semibold mt-0.5 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                        ${scaleIn.livePrice.toFixed(2)}
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
                    Position: {scaleIn.currentPosPct.toFixed(1)}% of NLV · ${scaleIn.currentValue.toLocaleString(undefined, { maximumFractionDigits: 0 })}
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
                      Last lot @ ${scaleIn.lastLotPrice.toFixed(2)} → {scaleIn.lastLotReturn >= 0 ? "+" : ""}{scaleIn.lastLotReturn.toFixed(2)}%
                      {!scaleIn.pyramidReady && " (need +5%)"}
                    </div>
                  </div>
                  <div className="grid grid-cols-1 gap-2.5 mt-2.5">
                    <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                      <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Max Pyramid Add (20%)</div>
                      <div className="text-[18px] font-semibold mt-0.5" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                        {scaleIn.maxPyramidShares} shs
                        <span className="text-[12px] font-normal ml-2 privacy-mask" style={{ color: "var(--ink-4)" }}>
                          ~${(scaleIn.maxPyramidShares * scaleIn.livePrice).toLocaleString(undefined, { maximumFractionDigits: 0 })}
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
                            ${scaleIn.newAvgCost.toFixed(2)}
                          </div>
                        </div>
                      </div>
                      <div className="mt-2.5 text-[12px] font-medium px-3 py-1.5 rounded-[8px] privacy-mask"
                           style={{
                             background: scaleIn.newPosPct > 25 ? "color-mix(in oklab, #e5484d 10%, var(--surface))" : scaleIn.newPosPct > 15 ? "color-mix(in oklab, #f59f00 10%, var(--surface))" : "color-mix(in oklab, #08a86b 10%, var(--surface))",
                             color: scaleIn.newPosPct > 25 ? "#dc2626" : scaleIn.newPosPct > 15 ? "#d97706" : "#16a34a",
                             border: `1px solid ${scaleIn.newPosPct > 25 ? "color-mix(in oklab, #e5484d 30%, var(--border))" : scaleIn.newPosPct > 15 ? "color-mix(in oklab, #f59f00 30%, var(--border))" : "color-mix(in oklab, #08a86b 30%, var(--border))"}`,
                           }}>
                        New position: {scaleIn.newPosPct.toFixed(1)}% of NLV · ${scaleIn.newValue.toLocaleString(undefined, { maximumFractionDigits: 0 })}
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
                      <div className="font-semibold mt-1">Min stop: ${scaleIn.minStop.toFixed(2)}</div>
                    </div>
                  )}

                  <div className="grid grid-cols-2 gap-2.5">
                    <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                      <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Combined Stop</div>
                      <div className="text-[18px] font-semibold mt-0.5" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                        {scaleIn.combinedStop > 0 ? `$${scaleIn.combinedStop.toFixed(2)}` : "—"}
                      </div>
                    </div>
                    <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                      <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Total Risk</div>
                      <div className="text-[18px] font-semibold mt-0.5 privacy-mask" style={{
                        fontFamily: "var(--font-jetbrains), monospace",
                        color: scaleIn.riskFreeAdd ? "#16a34a" : scaleIn.combinedRiskPct > 1 ? "#e5484d" : "var(--ink)",
                      }}>
                        {scaleIn.riskFreeAdd ? "$0" : scaleIn.combinedRisk > 0 ? `$${scaleIn.combinedRisk.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : "—"}
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

                {/* M Factor suggestion */}
                <div className="px-3 py-2 rounded-[8px] text-[12px]" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
                  <span style={{ color: "var(--ink-4)" }}>M Factor:</span>{" "}
                  <span className="font-semibold">{mFactorSuggestion}</span>
                  <span style={{ color: "var(--ink-4)" }}> → </span>
                  <span className="font-semibold">{SIZING_MODES[sizingMode].icon} {SIZING_MODES[sizingMode].key.charAt(0).toUpperCase() + SIZING_MODES[sizingMode].key.slice(1)}</span>
                </div>

                {/* Sizing mode radios */}
                <Field label="Sizing Mode">
                  <div className="flex flex-col gap-1.5 mt-1">
                    {SIZING_MODES.map((m, i) => (
                      <Radio key={m.key} checked={sizingMode === i} onClick={() => setSizingMode(i)} label={`${m.icon} ${m.label}`} />
                    ))}
                  </div>
                </Field>

                {/* Account Equity */}
                <Field label="Account Equity">
                  <div className="h-[42px] px-3.5 rounded-[10px] flex items-center text-[15px] font-semibold privacy-mask"
                       style={{ background: "var(--bg)", border: "1px solid var(--border)", fontFamily: "var(--font-jetbrains), monospace" }}>
                    ${equity.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                  </div>
                </Field>

                {/* Risk Budget + Stop Dist */}
                <div className="grid grid-cols-2 gap-2.5">
                  <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                    <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Risk $</div>
                    <div className="text-[20px] font-semibold mt-0.5 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                      ${riskBudget.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </div>
                  </div>
                  <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                    <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Stop Dist</div>
                    <div className="text-[20px] font-semibold mt-0.5" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                      {stopDist > 0 ? `$${stopDist.toFixed(2)}` : "—"}
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
                      {recommendedCost > 0 ? `$${recommendedCost.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : totalCost > 0 ? `$${totalCost.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : "—"}
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
                          ? `Risk $${riskDollars.toFixed(0)} > Budget $${riskBudget.toFixed(0)}. Move stop to $${rbmStop.toFixed(2)}`
                          : withinBudget
                            ? `Risk $${riskDollars.toFixed(0)} within budget $${riskBudget.toFixed(0)} ✓`
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
