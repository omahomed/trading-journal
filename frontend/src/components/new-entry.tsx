"use client";

// New Entry — Parallel Position Sizer (ATR Risk-Unit Model).
//
// This page is a PARALLEL implementation of a redesigned sizing model to
// be evaluated side-by-side with the existing Position Sizer before that
// one is retired. The math here is not simplified or "improved" — every
// number came out of an analysis of 204 realized 2026 trades and was
// signed off before this file existed.
//
// Model (post-Refinement 1 — speculative-tier denominator + stop
// validation, superseding the original 8-point spec):
//
//   1. RISK UNIT — from the M-Factor state → SIZING_MODES mapping
//      (Pilot 0.25% / Normal 0.50% / Offense 0.75%).
//
//   2. DENOMINATOR SELECTION
//      * DEFAULT (non-speculative names) — trailing 12-mo realized
//        avg loss %, floored at 4.0%.
//      * SPECULATIVE TIER (1.5× ATR21 > SPEC_TIER_STOP_PCT):
//          - MA level entered → denominator = tech_stop_dist_%
//            subject to the stop-validation banner (below).
//          - No MA level → denominator = 1.5× ATR21 (fallback), stated
//            on the verdict card.
//
//   3. STOP VALIDATION (speculative tier only) — asymmetric by design.
//         stop_atr_mult < STOP_ATR_MULT_MIN     → RED, HARD BLOCK.
//         MIN ≤ stop_atr_mult ≤ TARGET          → GREEN, sizing renders.
//         stop_atr_mult > STOP_ATR_MULT_TARGET  → AMBER, INFORMATIONAL —
//                                                 sizing STILL renders.
//                                                 Suggests a lower entry
//                                                 (max_valid_entry) to
//                                                 bring the stop inside
//                                                 the target multiplier.
//      Rationale for the asymmetric BLOCK on the tight side: sizing is
//      risk ÷ stop, so a tight stop on a volatile name yields a LARGE
//      position stopped out by daily noise — the measured 2026 churn
//      leak: 79 trades ≤1-day hold, −$88K, 11% win rate.
//
//   4. FORMULA — position_size_% = risk_unit_% / denominator_% × 100.
//
//   5. TARGET POSITION SIZE — Shotgun 2.5% … Max 20%. Default Overweight
//      (12.5%). All buttons freely selectable; caps formula output.
//
//   6. CATASTROPHE BACKSTOP — displays BOTH stops (tech stop + 1.5×
//      ATR21). Gap-tail % NLV = position_size_% × active_denominator_%
//      / 100.
//
//   7. TREND COUNT — negative → passive amber banner (Down-Cycle
//      Protocol: SR8 cascade only). Gates nothing.
//
//   8. VERDICT — "Buy N shares · X.X% NLV · risk 0.XX% NLV · gap tail
//      0.XX% NLV · [M-state] risk unit" with an explicit denominator
//      source line ("Sized off …") beneath.
//
// STOP_ATR_MULT_MIN and STOP_ATR_MULT_TARGET are CONFIG VALUES, kept
// together at the top of the file so recalibration from MAE/MFE data
// (separate branch) is a one-line change. They are provisional.

import { useCallback, useEffect, useMemo, useState } from "react";
import { api, getActivePortfolio } from "@/lib/api";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";
import {
  SIZING_MODES,
  SIZING_MODES_DISPLAY,
  deriveAutoSizingMode,
  describeMctSource,
  clampManualToDownwardOnly,
  type ExitAlert,
} from "@/lib/sizing-mode";

// ─────────────────────────────────────────────────────────────────────
// Constants (config values — grouped so recalibration is a one-liner)
// ─────────────────────────────────────────────────────────────────────

// Trailing-avg-loss client-side floor. Reflects the trader's actual
// exit behavior (median 1-day hold; sizing against a tighter
// denominator concentrates risk on any single position).
const AVG_LOSS_FLOOR_PCT = 4.0;

// Catastrophe-backstop multiplier: 1.5× ATR21 below entry.
const HARD_STOP_ATR_MULT = 1.5;

// Speculative-tier tripwire — a 1.5× ATR21 stop wider than 8% flags
// the name as too volatile for the default denominator model.
const SPEC_TIER_STOP_PCT = 8.0;

// Speculative-tier stop-validation window. tech_stop_dist / ATR21_%
// below MIN → HARD BLOCK. Between MIN and TARGET (inclusive) → GREEN.
// Above TARGET → AMBER informational. Provisional; will be recalibrated
// from MAE/MFE excursion data once that feature accumulates history.
const STOP_ATR_MULT_MIN = 1.0;
const STOP_ATR_MULT_TARGET = 1.5;

// Buffer default (%). Matches Position Sizer's default so the two
// pages compute the same tech stop when a user enters the same MA.
const DEFAULT_BUFFER_PCT = 1.0;

// Trailing-avg-loss window in months. Matches the backend default.
const TRAILING_WINDOW_MONTHS = 12;

// Target Position Size cap buttons — same lineup as Position Sizer so
// the two pages stay in visual lockstep. Default is Overweight (12.5%);
// every button stays selectable.
const SIZE_OPTIONS = [
  { label: "Shotgun (2.5%)",   pct:  2.5 },
  { label: "Half (5%)",        pct:  5   },
  { label: "Standard (7.5%)",  pct:  7.5 },
  { label: "Full (10%)",       pct: 10   },
  { label: "Overweight (12.5%)", pct: 12.5 },
  { label: "Core (15%)",       pct: 15   },
  { label: "Core+ (17.5%)",    pct: 17.5 },
  { label: "Max (20%)",        pct: 20   },
];
const DEFAULT_SIZE_CAP_PCT = 12.5;

// Fallback NLV if journalLatest returns nothing.
const FALLBACK_NLV = 100_000;

// ─────────────────────────────────────────────────────────────────────
// Pure math (unit-tested in new-entry.test.tsx)
// ─────────────────────────────────────────────────────────────────────

export type DenominatorSource = "avg-loss" | "tech-stop" | "atr-fallback";
export type StopValidation = "green" | "amber" | "red" | "none";

export interface NewEntryInputs {
  entry: number;             // Entry price, dollars
  atrPct: number;            // Daily ATR% of price (from priceLookup)
  nlv: number;               // Prior-day End NLV, dollars
  riskUnitPct: number;       // From SIZING_MODES[sizingMode].pct
  avgLossPct: number | null; // Trailing-avg-loss magnitude (positive).
                             //   Null when the endpoint returned no
                             //   sample; the 4% floor still applies.
  targetCapPct: number;      // Selected target-size button (2.5 … 20)
  maLevel?: number;          // Key MA level ($) — required only for
                             //   speculative-tier tech-stop sizing.
  bufferPct?: number;        // Buffer below MA (%). Defaults to 1.0.
}

export interface NewEntryResults {
  // Denominator selection
  isSpeculative: boolean;
  denominatorPct: number;
  denominatorSource: DenominatorSource;
  denominatorFloored: boolean;    // avg-loss case only

  // Tech stop math (populated whenever MA level provided; used to size
  // on the speculative tier, informational otherwise)
  techStopPrice: number | null;
  techStopDistPct: number | null;
  stopAtrMult: number | null;
  stopValidation: StopValidation;
  maxValidEntry: number | null;

  // 1.5× ATR21 catastrophe backstop (always populated)
  atrStopPrice: number;
  atrStopPct: number;

  // Sizing outputs — null when blocked (red).
  blocked: boolean;
  formulaPct: number | null;
  capBound: boolean;
  posSizePct: number | null;
  posDollars: number | null;
  shares: number | null;

  // Gap tail computed off the ACTIVE denominator.
  gapTailPctNlv: number | null;
  gapTailDollars: number | null;
}

/**
 * Compute the full New Entry recommendation. Callers guard the "any
 * input missing / bad" case by NOT calling this — entry/atrPct/nlv/
 * riskUnitPct/targetCapPct must be finite positive numbers. avgLossPct
 * may be null (endpoint returned no sample) and maLevel is optional.
 */
export function computeNewEntry(i: NewEntryInputs): NewEntryResults {
  const isSpeculative = HARD_STOP_ATR_MULT * i.atrPct > SPEC_TIER_STOP_PCT;

  // ATR catastrophe backstop — always populated so the verdict card
  // can show it alongside whichever stop the sizing math actually uses.
  const atrStopPct = HARD_STOP_ATR_MULT * i.atrPct;
  const atrStopPrice = i.entry * (1 - atrStopPct / 100);

  // Tech-stop math (populated whenever MA provided, used per denominator
  // selection rules below). Buffer defaults to DEFAULT_BUFFER_PCT to
  // match Position Sizer.
  const buffer = i.bufferPct ?? DEFAULT_BUFFER_PCT;
  const hasMa = typeof i.maLevel === "number" && Number.isFinite(i.maLevel) && i.maLevel > 0;
  const techStopPrice = hasMa ? i.maLevel! * (1 - buffer / 100) : null;
  const techStopDistPct = techStopPrice != null
    ? ((i.entry - techStopPrice) / i.entry) * 100
    : null;
  const stopAtrMult = techStopDistPct != null && i.atrPct > 0
    ? techStopDistPct / i.atrPct
    : null;

  // Denominator selection.
  let denominatorPct: number;
  let denominatorSource: DenominatorSource;
  let denominatorFloored = false;
  let stopValidation: StopValidation = "none";
  let blocked = false;
  let maxValidEntry: number | null = null;

  if (!isSpeculative) {
    // Default path — trailing avg loss with 4% floor.
    const magnitude = i.avgLossPct != null ? i.avgLossPct : 0;
    denominatorPct = Math.max(AVG_LOSS_FLOOR_PCT, magnitude);
    denominatorFloored = denominatorPct === AVG_LOSS_FLOOR_PCT
      && (i.avgLossPct == null || i.avgLossPct < AVG_LOSS_FLOOR_PCT);
    denominatorSource = "avg-loss";
  } else if (!hasMa || techStopDistPct == null || techStopDistPct <= 0) {
    // Speculative, no MA → fall back to 1.5× ATR21 as denominator so
    // the page still produces a recommendation. Verdict card notes
    // this is the fallback path.
    denominatorPct = atrStopPct;
    denominatorSource = "atr-fallback";
  } else {
    // Speculative with MA → tech-stop-distance, validated against ATR.
    denominatorPct = techStopDistPct;
    denominatorSource = "tech-stop";

    if (stopAtrMult! < STOP_ATR_MULT_MIN) {
      stopValidation = "red";
      blocked = true;    // sizing NOT rendered
    } else if (stopAtrMult! <= STOP_ATR_MULT_TARGET) {
      stopValidation = "green";
    } else {
      stopValidation = "amber";
      // Entry price that would bring stop_atr_mult down to exactly
      // TARGET (holding tech stop fixed). Solve for E in:
      //   (E - tech_stop) / E = TARGET × atrPct / 100
      // → E = tech_stop / (1 - TARGET × atrPct / 100)
      const denom = 1 - (STOP_ATR_MULT_TARGET * i.atrPct) / 100;
      maxValidEntry = denom > 0 ? techStopPrice! / denom : null;
    }
  }

  // Blocked case — return everything needed for the RED banner but no
  // sizing math. Verdict card will render only the banner + stop rows.
  if (blocked) {
    return {
      isSpeculative,
      denominatorPct,
      denominatorSource,
      denominatorFloored,
      techStopPrice,
      techStopDistPct,
      stopAtrMult,
      stopValidation,
      maxValidEntry,
      atrStopPrice,
      atrStopPct,
      blocked: true,
      formulaPct: null,
      capBound: false,
      posSizePct: null,
      posDollars: null,
      shares: null,
      gapTailPctNlv: null,
      gapTailDollars: null,
    };
  }

  // Formula: risk_unit_% / denominator_% × 100 (see file header math).
  const formulaPct = (i.riskUnitPct / denominatorPct) * 100;
  const capBound = i.targetCapPct <= formulaPct;
  const posSizePct = Math.min(formulaPct, i.targetCapPct);
  const posDollars = (posSizePct / 100) * i.nlv;
  const shares = Math.floor(posDollars / i.entry);

  // Gap tail off the ACTIVE denominator — the loss you eat at exit if
  // the position moves to the denominator level from entry.
  const gapTailPctNlv = (posSizePct * denominatorPct) / 100;
  const gapTailDollars = (gapTailPctNlv / 100) * i.nlv;

  return {
    isSpeculative,
    denominatorPct,
    denominatorSource,
    denominatorFloored,
    techStopPrice,
    techStopDistPct,
    stopAtrMult,
    stopValidation,
    maxValidEntry,
    atrStopPrice,
    atrStopPct,
    blocked: false,
    formulaPct,
    capBound,
    posSizePct,
    posDollars,
    shares,
    gapTailPctNlv,
    gapTailDollars,
  };
}

// ─────────────────────────────────────────────────────────────────────
// Small primitives
// ─────────────────────────────────────────────────────────────────────

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  // Nested label pattern: the child input is implicitly associated with
  // the label so screen readers + getByLabelText both work without
  // threading htmlFor/id through every callsite.
  return (
    <label className="block">
      <span className="block text-[10px] uppercase tracking-[0.08em] font-semibold mb-1"
            style={{ color: "var(--ink-4)" }}>{label}</span>
      {children}
      {hint && (
        <div className="text-[10px] mt-1" style={{ color: "var(--ink-4)" }}>{hint}</div>
      )}
    </label>
  );
}

function Radio({ checked, onClick, label, disabled = false, title }: {
  checked: boolean;
  onClick: () => void;
  label: string;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <button type="button" onClick={onClick} disabled={disabled} title={title}
            className="flex items-center gap-2 px-3 py-2 rounded-[8px] text-[12px] text-left w-full transition-all disabled:opacity-40"
            style={{
              background: checked ? "var(--surface-2)" : "transparent",
              border: `1px solid ${checked ? "var(--nav)" : "var(--border)"}`,
              color: "var(--ink)",
              cursor: disabled ? "not-allowed" : "pointer",
            }}>
      <span className="inline-block w-3 h-3 rounded-full"
            style={{
              border: `2px solid ${checked ? "var(--nav)" : "var(--ink-4)"}`,
              background: checked ? "var(--nav)" : "transparent",
            }} />
      <span>{label}</span>
    </button>
  );
}

// ─────────────────────────────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────────────────────────────

export function NewEntry({ navColor }: { navColor: string }) {
  // Auto-fetched context
  const [nlv, setNlv] = useState<number>(FALLBACK_NLV);
  const [mctState, setMctState] = useState<string | null>(null);
  const [trendCount, setTrendCount] = useState<number | null>(null);
  const [activeExits, setActiveExits] = useState<readonly ExitAlert[]>([]);
  const [avgLossPct, setAvgLossPct] = useState<number | null>(null);
  const [avgLossSample, setAvgLossSample] = useState<number>(0);

  // Sizing mode state
  const [sizingMode, setSizingMode] = useState<0 | 1 | 2>(0);
  const [sizingModeManual, setSizingModeManual] = useState(false);

  // User inputs
  const [ticker, setTicker] = useState("");
  const [entry, setEntry] = useState("");
  const [atrPct, setAtrPct] = useState<number | null>(null);
  const [maLevel, setMaLevel] = useState("");
  const [bufferPct, setBufferPct] = useState(String(DEFAULT_BUFFER_PCT));
  const [targetCapPct, setTargetCapPct] = useState<number>(DEFAULT_SIZE_CAP_PCT);
  const [fetchingPrice, setFetchingPrice] = useState(false);
  const [priceError, setPriceError] = useState("");

  // Mount effect — pull all baseline context in parallel.
  useEffect(() => {
    const portfolio = getActivePortfolio();
    Promise.all([
      api.journalLatest(portfolio).catch(() => null),
      api.rallyPrefix().catch(() => null),
      api.trailingAvgLoss(portfolio, TRAILING_WINDOW_MONTHS).catch(() => null),
    ]).then(([journal, rally, avgLoss]) => {
      const endNlv = Number((journal as any)?.end_nlv ?? FALLBACK_NLV);
      setNlv(Number.isFinite(endNlv) && endNlv > 0 ? endNlv : FALLBACK_NLV);

      const state = (rally as { state?: string } | null)?.state ?? null;
      setMctState(state);
      const exits = ((rally as { active_exits?: ExitAlert[] } | null)?.active_exits ?? []) as ExitAlert[];
      setActiveExits(exits);
      const tc = (rally as { trend_count?: number | null } | null)?.trend_count ?? null;
      setTrendCount(typeof tc === "number" ? tc : null);

      setSizingMode(deriveAutoSizingMode(state, exits).idx);
      setSizingModeManual(false);

      const loss = (avgLoss as { avg_loss_pct?: number | null } | null)?.avg_loss_pct ?? null;
      const sample = (avgLoss as { sample_size?: number } | null)?.sample_size ?? 0;
      setAvgLossPct(typeof loss === "number" ? Math.abs(loss) : null);
      setAvgLossSample(sample);
    }).catch(err => log.error("new-entry", "mount fetch failed", err));
  }, []);

  // Debounced price + ATR lookup on ticker changes.
  useEffect(() => {
    const t = ticker.trim().toUpperCase();
    if (!t) { setEntry(""); setAtrPct(null); setPriceError(""); return; }
    const timer = window.setTimeout(() => {
      setFetchingPrice(true);
      setPriceError("");
      api.priceLookup(t).then(r => {
        setEntry(String(r.price ?? ""));
        setAtrPct(typeof r.atr_pct === "number" ? r.atr_pct : null);
      }).catch(err => {
        log.error("new-entry", "priceLookup failed", err);
        setPriceError("Couldn't fetch price");
      }).finally(() => setFetchingPrice(false));
    }, 600);
    return () => window.clearTimeout(timer);
  }, [ticker]);

  const autoIdx = useMemo(
    () => deriveAutoSizingMode(mctState, activeExits).idx,
    [mctState, activeExits],
  );
  const floor = useMemo(
    () => deriveAutoSizingMode(mctState, activeExits).source.floor,
    [mctState, activeExits],
  );

  const handleModePick = useCallback((idx: 0 | 1 | 2) => {
    const clamped = clampManualToDownwardOnly(autoIdx, idx);
    setSizingMode(clamped);
    setSizingModeManual(clamped !== autoIdx);
  }, [autoIdx]);

  const handleResetToAuto = useCallback(() => {
    setSizingMode(autoIdx);
    setSizingModeManual(false);
  }, [autoIdx]);

  const entryNum = Number(entry);
  const maNum = Number(maLevel);
  const bufferNum = Number(bufferPct);
  const canCompute =
    Number.isFinite(entryNum) && entryNum > 0
    && atrPct != null && Number.isFinite(atrPct) && atrPct > 0
    && nlv > 0;

  const results = useMemo(() => {
    if (!canCompute) return null;
    return computeNewEntry({
      entry: entryNum,
      atrPct: atrPct!,
      nlv,
      riskUnitPct: SIZING_MODES[sizingMode].pct,
      avgLossPct,
      targetCapPct,
      maLevel: Number.isFinite(maNum) && maNum > 0 ? maNum : undefined,
      bufferPct: Number.isFinite(bufferNum) && bufferNum >= 0 ? bufferNum : undefined,
    });
  }, [canCompute, entryNum, atrPct, nlv, sizingMode, avgLossPct, targetCapPct, maNum, bufferNum]);

  const modeSource = describeMctSource(mctState, floor);
  const currentMode = SIZING_MODES[sizingMode];

  return (
    <div style={{ ["--nav" as string]: navColor, animation: "slide-up 0.18s ease-out" }}>
      {/* Header */}
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0"
            style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          New <em className="italic" style={{ color: navColor }}>Entry</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
          ATR risk-unit model · Parallel to Position Sizer
        </div>
      </div>

      {/* Rules disclosure */}
      <details className="mb-4 rounded-[10px] overflow-hidden"
               style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
        <summary className="px-4 py-2.5 text-[12px] font-semibold cursor-pointer"
                 style={{ color: "var(--ink-3)" }}>
          View New Entry Rules
        </summary>
        <div className="px-4 pb-3 text-[12px] leading-relaxed" style={{ color: "var(--ink-3)" }}>
          <p className="mb-1"><strong>Risk Unit (M Factor):</strong></p>
          <ul className="list-disc ml-4 mb-2">
            <li>POWERTREND → <strong>Offense (0.75%)</strong></li>
            <li>UPTREND → <strong>Normal (0.50%)</strong></li>
            <li>RALLY MODE / UUP / CORRECTION → <strong>Pilot (0.25%)</strong></li>
          </ul>
          <p className="mb-1"><strong>Denominator (default):</strong></p>
          <ul className="list-disc ml-4 mb-2">
            <li><code>max(4.0%, |trailing 12-mo avg loss|)</code> — realized average loss, not the hard stop</li>
          </ul>
          <p className="mb-1"><strong>Speculative tier (1.5× ATR21 &gt; 8%):</strong></p>
          <ul className="list-disc ml-4 mb-2">
            <li>Sizes off the <strong>tech stop</strong> (MA × (1 − buffer)) when an MA is entered</li>
            <li>No MA → falls back to <strong>1.5× ATR21</strong> as the denominator</li>
            <li><strong>Stop validation:</strong> tech-stop-distance ÷ ATR21 mult decides:
              <ul className="list-[circle] ml-4 mt-1">
                <li><strong>&lt; {STOP_ATR_MULT_MIN.toFixed(1)}× ATR</strong> — RED, sizing off. Tight stops on volatile names → oversized positions stopped out by noise.</li>
                <li><strong>{STOP_ATR_MULT_MIN.toFixed(1)}–{STOP_ATR_MULT_TARGET.toFixed(1)}× ATR</strong> — GREEN, validated.</li>
                <li><strong>&gt; {STOP_ATR_MULT_TARGET.toFixed(1)}× ATR</strong> — AMBER, sizes anyway; page suggests a lower entry to bring the stop inside the guideline.</li>
              </ul>
            </li>
          </ul>
          <p className="mb-1"><strong>Formula:</strong></p>
          <ul className="list-disc ml-4 mb-2">
            <li>Position size % = <code>risk unit % / denominator % × 100</code></li>
            <li>Cap = the selected Target Position Size button (default Overweight 12.5%)</li>
          </ul>
          <p className="mb-1"><strong>Catastrophe backstop:</strong></p>
          <ul className="list-disc ml-4 mb-2">
            <li>Always shows BOTH the tech stop (when MA entered) and the 1.5× ATR21 stop</li>
            <li>Gap-tail % NLV = position size % × active denominator % / 100</li>
          </ul>
          <p className="mb-2 text-[11px]" style={{ color: "var(--ink-4)" }}>
            Stop-validation multiples ({STOP_ATR_MULT_MIN.toFixed(1)}× MIN /{" "}
            {STOP_ATR_MULT_TARGET.toFixed(1)}× TARGET) are provisional and will
            be recalibrated from MAE/MFE excursion data once that dataset lands.
          </p>
        </div>
      </details>

      {/* Trend Count banner (passive; does not gate) */}
      {trendCount != null && trendCount < 0 && (
        <div className="mb-4 rounded-[10px] px-4 py-3 flex items-center gap-3"
             data-testid="new-entry-trend-count-banner"
             style={{
               background: "color-mix(in oklab, #f59f00 12%, var(--surface))",
               border: "1px solid color-mix(in oklab, #f59f00 40%, var(--border))",
               color: "#a16207",
             }}>
          <span className="text-[16px]">⚠️</span>
          <div>
            <div className="text-[12px] font-semibold">
              Trend Count negative ({trendCount})
            </div>
            <div className="text-[11px]" style={{ color: "var(--ink-3)" }}>
              Down-Cycle Protocol: SR8 cascade only. This does not block New Entry sizing — informational.
            </div>
          </div>
        </div>
      )}

      {/* Sizing-mode indicator */}
      <div className="mb-4 rounded-[10px] px-4 py-3 flex items-center flex-wrap gap-3"
           data-testid="new-entry-mode-indicator"
           style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <div className="flex items-center gap-2">
          <span className="text-[16px]">{currentMode.icon}</span>
          <div>
            <div className="text-[11px] uppercase tracking-[0.08em] font-semibold"
                 style={{ color: "var(--ink-4)" }}>
              {sizingModeManual ? "Manual" : "Auto"}
            </div>
            <div className="text-[13px] font-semibold">
              {currentMode.label}
              {sizingModeManual ? (
                <span className="text-[11px] font-normal" style={{ color: "var(--ink-4)" }}>
                  {" "}— downward override
                </span>
              ) : (
                <span className="text-[11px] font-normal" style={{ color: "var(--ink-4)" }}>
                  {" "}({modeSource})
                </span>
              )}
            </div>
          </div>
        </div>
        {sizingModeManual && (
          <button type="button" onClick={handleResetToAuto}
                  data-testid="new-entry-reset-to-auto"
                  className="ml-auto h-[30px] px-3 rounded-[8px] text-[11px] font-medium transition-colors"
                  style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink-2)" }}>
            Reset to auto
          </button>
        )}
      </div>

      {/* Inputs — 4 columns: Ticker | Entry | MA Level | Buffer */}
      <div className="grid gap-4 mb-4"
           style={{ gridTemplateColumns: "1fr 1fr 1fr 0.7fr" }}>
        <Field label="Ticker" hint={priceError || (fetchingPrice ? "Fetching…" : ticker ? "" : "Auto-fills entry price + ATR")}>
          <input type="text" value={ticker}
                 onChange={e => setTicker(e.target.value.toUpperCase())}
                 placeholder="XYZ"
                 className="w-full h-[38px] px-3 rounded-[10px] text-[13px] outline-none"
                 style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)" }} />
        </Field>
        <Field label="Entry price" hint={atrPct != null ? `ATR21 ${atrPct.toFixed(2)}%` : "—"}>
          <input type="number" value={entry}
                 onChange={e => setEntry(e.target.value)}
                 step="0.01" placeholder="0.00"
                 className="w-full h-[38px] px-3 rounded-[10px] text-[13px] outline-none"
                 style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: "var(--font-jetbrains), monospace" }} />
        </Field>
        <Field label="Key MA level ($)" hint="Optional. Required to size off tech stop on speculative names.">
          <input type="number" value={maLevel}
                 onChange={e => setMaLevel(e.target.value)}
                 step="0.01" placeholder="0.00"
                 className="w-full h-[38px] px-3 rounded-[10px] text-[13px] outline-none"
                 style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: "var(--font-jetbrains), monospace" }} />
        </Field>
        <Field label="Buffer (%)" hint={`Default ${DEFAULT_BUFFER_PCT}`}>
          <input type="number" value={bufferPct}
                 onChange={e => setBufferPct(e.target.value)}
                 step="0.1" placeholder={String(DEFAULT_BUFFER_PCT)}
                 className="w-full h-[38px] px-3 rounded-[10px] text-[13px] outline-none"
                 style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: "var(--font-jetbrains), monospace" }} />
        </Field>
      </div>

      {/* Sizing mode override row (downward-only) */}
      <div className="mb-4">
        <div className="text-[10px] uppercase tracking-[0.08em] font-semibold mb-2"
             style={{ color: "var(--ink-4)" }}>
          Sizing Mode override (downward-only)
        </div>
        <div className="grid gap-2" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
          {SIZING_MODES_DISPLAY.map(m => {
            const isBlocked = m.index > autoIdx;
            return (
              <Radio key={m.key}
                     checked={sizingMode === m.index}
                     onClick={() => handleModePick(m.index)}
                     disabled={isBlocked}
                     title={isBlocked
                       ? `Regime permits ${SIZING_MODES[autoIdx].label} or smaller; larger tiers disabled on New Entry.`
                       : undefined}
                     label={`${m.icon} ${m.label}`} />
            );
          })}
        </div>
      </div>

      {/* Target Position Size buttons */}
      <div className="mb-5">
        <div className="text-[10px] uppercase tracking-[0.08em] font-semibold mb-2"
             style={{ color: "var(--ink-4)" }}>
          Target Position Size (cap on formula output)
        </div>
        <div className="grid gap-2" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
          {SIZE_OPTIONS.map(s => (
            <button key={s.pct} type="button"
                    onClick={() => setTargetCapPct(s.pct)}
                    data-testid={`new-entry-size-${s.pct}`}
                    className="h-[38px] rounded-[8px] text-[11px] font-medium transition-all"
                    style={{
                      background: targetCapPct === s.pct ? navColor : "var(--bg)",
                      color: targetCapPct === s.pct ? "#fff" : "var(--ink-4)",
                      border: `1px solid ${targetCapPct === s.pct ? navColor : "var(--border)"}`,
                    }}>
              {s.label}
            </button>
          ))}
        </div>
      </div>

      {/* Stop-validation banner (speculative tier only) */}
      {results && results.stopValidation !== "none" && (
        <StopValidationBanner results={results} entryPrice={entryNum} />
      )}

      {/* Verdict */}
      {results ? (
        <VerdictCard
          results={results}
          entryPrice={entryNum}
          nlv={nlv}
          sizingMode={sizingMode}
          currentMode={currentMode}
          avgLossSample={avgLossSample}
          navColor={navColor}
        />
      ) : (
        <div className="rounded-[14px] px-5 py-6 text-center text-[13px] mb-4"
             style={{ background: "var(--bg)", border: "1px dashed var(--border)", color: "var(--ink-4)" }}>
          Enter a ticker + entry price to see the recommendation.
          {avgLossPct == null && avgLossSample === 0 && (
            <div className="mt-2 text-[11px]">
              (No closed equity losses yet for {getActivePortfolio()} — the 4% denominator floor will apply.)
            </div>
          )}
        </div>
      )}

      {/* Trailing-avg-loss context tile */}
      <div className="rounded-[10px] px-4 py-3 mb-4"
           data-testid="new-entry-avgloss-tile"
           style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <div className="text-[10px] uppercase tracking-[0.08em] font-semibold mb-1"
             style={{ color: "var(--ink-4)" }}>
          Trailing {TRAILING_WINDOW_MONTHS}-mo realized avg loss ({getActivePortfolio()})
        </div>
        <div className="text-[13px]" style={{ color: "var(--ink-2)" }}>
          {avgLossPct != null ? (
            <>
              <strong>−{avgLossPct.toFixed(2)}%</strong>{" "}
              <span style={{ color: "var(--ink-4)" }}>
                ({avgLossSample} closed equity losses).{" "}
                Denominator (default path): {Math.max(AVG_LOSS_FLOOR_PCT, avgLossPct).toFixed(2)}%
                {avgLossPct < AVG_LOSS_FLOOR_PCT && ` (floored at ${AVG_LOSS_FLOOR_PCT}%)`}
              </span>
            </>
          ) : (
            <>
              <em>No sample yet.</em>{" "}
              <span style={{ color: "var(--ink-4)" }}>
                Denominator uses the {AVG_LOSS_FLOOR_PCT}% floor.
              </span>
            </>
          )}
        </div>
      </div>

      {/* NLV + context footer */}
      <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>
        Prior-day NLV: <strong style={{ color: "var(--ink-3)" }}>{formatCurrency(nlv, { decimals: 0 })}</strong>
        {" · "}M Factor: <strong style={{ color: "var(--ink-3)" }}>{mctState || "unknown"}</strong>
        {trendCount != null && <>{" · "}Trend Count: <strong style={{ color: "var(--ink-3)" }}>{trendCount > 0 ? `+${trendCount}` : trendCount}</strong></>}
      </div>
    </div>
  );
}


// ─────────────────────────────────────────────────────────────────────
// Sub-components (kept in-file — they're tightly coupled to the model
// and don't earn independent existence)
// ─────────────────────────────────────────────────────────────────────

function StopValidationBanner({
  results, entryPrice,
}: { results: NewEntryResults; entryPrice: number }) {
  if (results.stopValidation === "green") {
    return (
      <div data-testid="new-entry-stop-validation-banner"
           data-variant="green"
           className="mb-4 rounded-[10px] px-4 py-3 flex items-center gap-3"
           style={{
             background: "color-mix(in oklab, #08a86b 12%, var(--surface))",
             border: "1px solid color-mix(in oklab, #08a86b 40%, var(--border))",
             color: "#16a34a",
           }}>
        <span className="text-[16px]">✅</span>
        <div className="text-[12px] font-semibold">
          Stop validated ({results.stopAtrMult!.toFixed(2)}× ATR).
        </div>
      </div>
    );
  }
  if (results.stopValidation === "amber") {
    return (
      <div data-testid="new-entry-stop-validation-banner"
           data-variant="amber"
           className="mb-4 rounded-[10px] px-4 py-3 flex items-start gap-3"
           style={{
             background: "color-mix(in oklab, #f59f00 12%, var(--surface))",
             border: "1px solid color-mix(in oklab, #f59f00 40%, var(--border))",
             color: "#a16207",
           }}>
        <span className="text-[16px]">⚠️</span>
        <div>
          <div className="text-[12px] font-semibold">
            Stop is {results.stopAtrMult!.toFixed(2)}× ATR — wider than the current {STOP_ATR_MULT_TARGET.toFixed(1)}× guideline.
          </div>
          <div className="text-[11px] mt-0.5" style={{ color: "var(--ink-3)" }}>
            {results.maxValidEntry != null ? (
              <>Entry valid at <strong>${results.maxValidEntry.toFixed(2)}</strong> or below to bring your stop inside {STOP_ATR_MULT_TARGET.toFixed(1)}× ATR. Sizing still renders below.</>
            ) : (
              <>Sizing still renders below.</>
            )}
          </div>
        </div>
      </div>
    );
  }
  if (results.stopValidation === "red") {
    return (
      <div data-testid="new-entry-stop-validation-banner"
           data-variant="red"
           className="mb-4 rounded-[10px] px-4 py-3 flex items-start gap-3"
           style={{
             background: "color-mix(in oklab, #e5484d 12%, var(--surface))",
             border: "2px solid color-mix(in oklab, #e5484d 55%, var(--border))",
             color: "#b91c1c",
           }}>
        <span className="text-[16px]">🛑</span>
        <div>
          <div className="text-[12px] font-bold">
            Stop inside daily noise ({results.stopAtrMult!.toFixed(2)}× ATR) — sizing off.
          </div>
          <div className="text-[11px] mt-0.5" style={{ color: "var(--ink-3)" }}>
            Sizing off a tight stop on a volatile name produces an oversized position with a near-certain random stop-out. Widen the stop or skip this entry.
          </div>
        </div>
      </div>
    );
  }
  return null;
}


function VerdictCard({
  results, entryPrice, nlv, sizingMode, currentMode, avgLossSample, navColor,
}: {
  results: NewEntryResults;
  entryPrice: number;
  nlv: number;
  sizingMode: 0 | 1 | 2;
  currentMode: (typeof SIZING_MODES)[number];
  avgLossSample: number;
  navColor: string;
}) {
  const modeLabel = currentMode.label.split(" (")[0];
  const denominatorLine = renderDenominatorLine(results, avgLossSample);

  // Blocked (RED) — no sizing math, just the stops for reference.
  if (results.blocked) {
    return (
      <div className="rounded-[14px] overflow-hidden mb-4"
           data-testid="new-entry-verdict"
           data-blocked="true"
           style={{ background: "var(--surface)", border: `1px solid var(--border)` }}>
        <div className="px-5 py-4">
          <div className="text-[10px] uppercase tracking-[0.1em] font-bold mb-2"
               style={{ color: "#b91c1c" }}>
            Verdict — blocked
          </div>
          <div className="text-[13px]" style={{ color: "var(--ink-2)" }}>
            Sizing suppressed by stop validation. Reference stops below.
          </div>
        </div>
        <StopReferenceRows results={results} entryPrice={entryPrice} />
      </div>
    );
  }

  return (
    <div className="rounded-[14px] overflow-hidden mb-4"
         data-testid="new-entry-verdict"
         style={{ background: "var(--surface)", border: `2px solid ${navColor}`, boxShadow: "var(--card-shadow)" }}>
      <div className="px-5 py-4">
        <div className="text-[10px] uppercase tracking-[0.1em] font-bold mb-2"
             style={{ color: navColor }}>
          Verdict
        </div>
        <div className="text-[22px] font-bold mb-1" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
          Buy {results.shares!.toLocaleString()} shares
        </div>
        <div className="text-[12px] mb-3" style={{ color: "var(--ink-3)" }}>
          {results.posSizePct!.toFixed(1)}% NLV
          {" · "}risk {SIZING_MODES[sizingMode].pct.toFixed(2)}% NLV
          {" · "}gap tail {results.gapTailPctNlv!.toFixed(2)}% NLV
          {" · "}{modeLabel} risk unit
        </div>
        <div className="text-[11px] grid gap-1" style={{ color: "var(--ink-4)" }}>
          <div>
            <strong>Position cost:</strong> {formatCurrency(results.posDollars!, { decimals: 0 })}
            {" @ "}${entryPrice.toFixed(2)}/share
          </div>
          <div data-testid="new-entry-denominator-source">
            <strong>Sized off:</strong> {denominatorLine}
          </div>
          <div>
            <strong>Formula:</strong> {SIZING_MODES[sizingMode].pct.toFixed(2)}% ÷ {results.denominatorPct.toFixed(2)}% × 100 = {results.formulaPct!.toFixed(2)}%
          </div>
          <div>
            <strong>Cap:</strong> Target Position Size {(() => {
              // Look up the cap value the user clicked. It's the smaller
              // of (formula, cap). posSizePct === formula ⇒ cap didn't
              // bind. We render the cap value regardless.
              return results.capBound ? `${results.posSizePct!.toFixed(2)}%` : `set above formula`;
            })()}
            {results.capBound
              ? <span style={{ color: navColor }}> — BOUND (cap tighter than formula)</span>
              : <span> — formula {results.formulaPct!.toFixed(2)}% is under the cap</span>}
          </div>
        </div>
      </div>

      <StopReferenceRows results={results} entryPrice={entryPrice} />
    </div>
  );
}


function StopReferenceRows({
  results, entryPrice,
}: { results: NewEntryResults; entryPrice: number }) {
  return (
    <div className="px-5 py-3"
         style={{ background: "var(--bg)", borderTop: "1px solid var(--border)" }}>
      <div className="text-[10px] uppercase tracking-[0.08em] font-semibold mb-2"
           style={{ color: "var(--ink-4)" }}>
        Stop-loss reference (enter one at broker)
      </div>
      <div className="grid gap-1 text-[12px]" style={{ color: "var(--ink-2)" }}>
        {results.techStopPrice != null && results.techStopDistPct != null && (
          <div data-testid="new-entry-tech-stop-row">
            <strong>Tech stop</strong> (MA − buffer):{" "}
            <span style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
              ${results.techStopPrice.toFixed(2)}
            </span>
            {" "}(−{results.techStopDistPct.toFixed(2)}% below entry
            {results.stopAtrMult != null && `, ${results.stopAtrMult.toFixed(2)}× ATR`})
            {results.denominatorSource === "tech-stop" && (
              <span style={{ color: "var(--nav)" }}> — ACTIVE denominator</span>
            )}
          </div>
        )}
        <div data-testid="new-entry-atr-stop-row">
          <strong>1.5× ATR21 stop</strong> (catastrophe backstop):{" "}
          <span style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
            ${results.atrStopPrice.toFixed(2)}
          </span>
          {" "}(−{results.atrStopPct.toFixed(2)}% below entry)
          {results.denominatorSource === "atr-fallback" && (
            <span style={{ color: "#a16207" }}> — ACTIVE denominator (no MA entered)</span>
          )}
        </div>
        {results.gapTailPctNlv != null && results.gapTailDollars != null && (
          <div className="text-[11px] mt-1" style={{ color: "var(--ink-4)" }}>
            If active stop trips:{" "}
            <strong style={{ color: "#e5484d" }}>
              −{results.gapTailPctNlv.toFixed(2)}% NLV
              {" "}({formatCurrency(results.gapTailDollars, { decimals: 0 })})
            </strong>
          </div>
        )}
      </div>
    </div>
  );
}


function renderDenominatorLine(results: NewEntryResults, avgLossSample: number): React.ReactNode {
  if (results.denominatorSource === "avg-loss") {
    return (
      <>
        trailing avg loss{" "}
        <strong>{results.denominatorPct.toFixed(2)}%</strong>
        {results.denominatorFloored && (
          <span style={{ color: "#a16207" }}> · floored at {AVG_LOSS_FLOOR_PCT.toFixed(1)}%</span>
        )}
        {avgLossSample > 0 && (
          <span style={{ color: "var(--ink-4)" }}> · {avgLossSample} closed losses</span>
        )}
      </>
    );
  }
  if (results.denominatorSource === "tech-stop") {
    return (
      <>
        tech stop <strong>{results.denominatorPct.toFixed(2)}%</strong>
        {" · "}
        {results.stopAtrMult != null && (
          <span>{results.stopAtrMult.toFixed(2)}× ATR · </span>
        )}
        <span>speculative tier</span>
      </>
    );
  }
  // atr-fallback
  return (
    <>
      1.5× ATR21 <strong>{results.denominatorPct.toFixed(2)}%</strong>
      {" · "}no MA entered · speculative tier
    </>
  );
}
