"use client";

import { useEffect, useMemo, useState } from "react";
import { Check, Lock } from "lucide-react";
import { useSearchParams } from "next/navigation";
import { api, getActivePortfolio, type TradePosition } from "@/lib/api";
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
 *   - volatility  → New Position Sizer (vol-sizer lib, Step 3)
 *   - scalein     → Scale In Sizer (inline math, this PR)
 *   - pyramid     → Coming soon (PR2)
 *   - trim        → Coming soon (PR3)
 *   - options     → Coming soon (PR4)
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

  // Fetched / lifecycle
  const [equity, setEquity] = useState<number | null>(null);
  const [atrPct, setAtrPct] = useState<number | null>(null);
  const [holdings, setHoldings] = useState<TradePosition[]>([]);
  const [loading, setLoading] = useState(true);
  const [priceError, setPriceError] = useState<string | null>(null);
  const [priceLoading, setPriceLoading] = useState(false);

  const flags = TAB_INPUTS[activeTab];

  // Mount fetch — equity + MCT state + open holdings. tradesOpen drives
  // the Scale-In / Pyramid / Trim holding pickers; Volatility ignores
  // it. Pyramid's pyramid_rules config + Scale-In/Trim's
  // tradesOpenDetails LIFO inventory are deferred to PR2/PR3 (Scale-In
  // here reads avg_entry directly off the picker selection).
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
    ]).then(([j, rally, open]) => {
      if (cancelled) return;
      const endNlv = j ? parseFloat(String((j as { end_nlv?: number | string }).end_nlv ?? 0)) : 0;
      setEquity(Number.isFinite(endNlv) && endNlv > 0 ? endNlv : null);
      const stateStr = (rally as { state?: string } | null)?.state ?? null;
      setSizingMode((prev) => (sizingModeManual ? prev : mctStateToSizingMode(stateStr)));
      setHoldings(Array.isArray(open) ? (open as TradePosition[]) : []);
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

      {/* ── Tab: Coming Soon (pyramid / trim / options) ── */}
      {(activeTab === "pyramid" || activeTab === "trim" || activeTab === "options") && (
        <div
          id={`sizer-panel-${activeTab}`}
          role="tabpanel"
          className="rounded-m-xl border-[0.5px] border-m-border bg-m-surface px-5 py-8 text-center"
        >
          <div className="text-[14px] font-medium text-m-text">Coming soon</div>
          <div className="mt-1.5 text-[12px] text-m-text-dim">
            {activeTab === "pyramid"
              ? "Pyramid sizer mobile build lands in a follow-up PR."
              : activeTab === "trim"
                ? "Trim (sell-down) sizer mobile build lands in a follow-up PR."
                : "Options sizer mobile build lands in a follow-up PR."}
          </div>
          <div className="mt-1 text-[11px] text-m-text-faint">Use desktop until then.</div>
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
