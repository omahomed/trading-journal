"use client";

// New Entry — Parallel Position Sizer (ATR Risk-Unit Model).
//
// This page is a PARALLEL implementation of a redesigned sizing model to
// be evaluated side-by-side with the existing Position Sizer before that
// one is retired. The math here is not simplified or "improved" — every
// number came out of an analysis of 204 realized 2026 trades and was
// signed off before this file existed.
//
// Model (verbatim from the build spec):
//
//   1. RISK UNIT — from the M-Factor state → SIZING_MODES mapping
//      (Pilot 0.25% / Normal 0.50% / Offense 0.75%).
//
//   2. DENOMINATOR — TRAILING REALIZED AVERAGE LOSS, not the hard stop.
//      * Backend: /api/analytics/trailing-avg-loss returns the trailing
//        12-month AVG(return_pct) over CLOSED equity campaigns with
//        return_pct < 0. Options excluded (a bought long call going to
//        zero is a -100% return that doesn't reflect the trader's
//        equity-exit behavior).
//      * Client applies a 4.0% floor:
//            denominator_% = max(4.0%, |avg_loss_%|)
//      * The floor exists because the trader exits whole positions fast
//        (median losing hold = 1 day, median loss = ~3.7%); sizing
//        against a tighter denominator than that puts too much on any
//        single position.
//
//   3. FORMULA — position_size_% = risk_unit_% / denominator_% × 100.
//      Because both inputs are expressed as percentages of the trade
//      /NLV, the ×100 turns their ratio back into a percentage of NLV.
//      Verified against the ALAB 4/14 entry: risk_unit 0.75, denom
//      4.58 → 16.4% raw → 12.5% cap → 300+ shares at $1M NAV (actual
//      buy was 300).
//
//   4. TARGET POSITION SIZE — Shotgun 2.5% … Max 20% button row. Default
//      Overweight (12.5%), acting as the CAP on the formula output. All
//      buttons stay selectable so the user can override the cap in
//      either direction; only the DEFAULT is 12.5%. Cap measured at
//      cost against prior-day NLV.
//
//   5. HARD STOP — 1.5× ATR21 below entry, displayed as the CATASTROPHE
//      BACKSTOP (not the sizing input). "If gapped through stop:
//      -X.XX% NLV" where X = position_size_% × stop_%. If 1.5× ATR21 >
//      8%, flag SPECULATIVE TIER (visual warning; the formula already
//      shrinks size via denominator).
//
//   6. TREND COUNT BANNER — Trend Count is NOT an M-Factor state; it's
//      a separate environment indicator. Negative → passive amber
//      banner ("Down-Cycle Protocol: SR8 cascade only"). Gates
//      nothing; sizing still renders.
//
//   7. SCALE-OUT STOPS — NOT carried over. The Position Sizer's Scale-
//      Out Stops feed the recommendation via a locked -3/-5/-7 ladder;
//      here they do not exist. (An optional collapsed display is fine
//      for reference, but nothing feeds shares.)
//
//   8. VERDICT — "Buy N shares · X.X% NLV · risk 0.XX% NLV · gap tail
//      0.XX% NLV · [M-state] risk unit". Every number is traceable to
//      the spec above.
//
// Manual mode override is DOWNWARD ONLY (via clampManualToDownwardOnly).
// "Reset to auto" discards the user pick and re-applies the regime
// mapping. Position Sizer + Log Buy retain their bidirectional manual
// override; this stricter guard is New Entry-only.

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
// Constants
// ─────────────────────────────────────────────────────────────────────

// Trailing-average-loss client-side floor. The endpoint returns the raw
// aggregate; the 4% minimum here reflects the trader's actual exit
// behavior (median 1-day hold; sizing against a tighter denominator
// concentrates risk on any single position).
const AVG_LOSS_FLOOR_PCT = 4.0;

// Hard-stop multiplier and speculative-tier tripwire. 8% is the point
// beyond which even a 1.5× ATR cushion prices in enough noise that the
// symbol reads more like a lottery ticket than a swing entry.
const HARD_STOP_ATR_MULT = 1.5;
const SPEC_TIER_STOP_PCT = 8.0;

// Trailing-avg-loss window in months. Matches the backend default; kept
// as a named constant so it's visible next to the display copy.
const TRAILING_WINDOW_MONTHS = 12;

// Target Position Size cap buttons — same lineup as Position Sizer so
// the two pages stay in visual lockstep. Default is Overweight (12.5%)
// per the New Entry spec; every button stays selectable.
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

// Fallback NLV if journalLatest returns nothing — same value Position
// Sizer and Log Buy use so the UX matches when the journal is empty.
const FALLBACK_NLV = 100_000;

// ─────────────────────────────────────────────────────────────────────
// Pure math (unit-tested in new-entry.test.tsx)
// ─────────────────────────────────────────────────────────────────────

export interface NewEntryInputs {
  entry: number;          // Entry price, dollars
  atrPct: number;         // Daily ATR% of price (from priceLookup)
  nlv: number;            // Prior-day End NLV, dollars
  riskUnitPct: number;    // From SIZING_MODES[sizingMode].pct
  avgLossPct: number;     // Trailing-avg-loss % (positive number; caller
                          //   passes |avg_loss_pct| — the floor is
                          //   applied inside computeNewEntry).
  targetCapPct: number;   // Selected target-size button (2.5 … 20)
}

export interface NewEntryResults {
  denominatorPct: number;      // max(floor, |avgLoss|)
  denominatorFloored: boolean; // true when the floor bound
  formulaPct: number;          // risk_unit / denominator × 100
  capBound: boolean;           // true when target cap was tighter
  posSizePct: number;          // min(formula, cap) — the recommendation
  posDollars: number;          // posSizePct/100 × nlv
  shares: number;              // floor(posDollars / entry)
  stopPrice: number;           // entry × (1 - 1.5 × atrPct/100)
  stopPct: number;             // 1.5 × atrPct  (percent-of-price move)
  gapTailPctNlv: number;       // posSizePct × stopPct / 100
  gapTailDollars: number;      // gapTailPctNlv/100 × nlv
  isSpeculative: boolean;      // stopPct > SPEC_TIER_STOP_PCT
}

/**
 * Compute the full New Entry recommendation from validated numeric
 * inputs. Callers guard the "any input missing / bad" case by NOT
 * calling this — every field must be a finite positive number.
 */
export function computeNewEntry(i: NewEntryInputs): NewEntryResults {
  const denominatorPct = Math.max(AVG_LOSS_FLOOR_PCT, i.avgLossPct);
  const denominatorFloored = denominatorPct === AVG_LOSS_FLOOR_PCT
    && i.avgLossPct < AVG_LOSS_FLOOR_PCT;

  // Formula: risk_unit_% / denominator_% × 100 (see file header math).
  const formulaPct = (i.riskUnitPct / denominatorPct) * 100;

  // Cap = the target-size button. Cap wins when it's smaller.
  const capBound = i.targetCapPct <= formulaPct;
  const posSizePct = Math.min(formulaPct, i.targetCapPct);

  const posDollars = (posSizePct / 100) * i.nlv;
  const shares = Math.floor(posDollars / i.entry);

  // 1.5× ATR21 stop. atrPct is a percent-of-price ATR, so the stop
  // distance in percent-of-price is 1.5 × atrPct; the stop price is
  // entry × (1 - stopPct/100).
  const stopPct = HARD_STOP_ATR_MULT * i.atrPct;
  const stopPrice = i.entry * (1 - stopPct / 100);

  // Gap-tail scenario: if the market gaps through the stop and closes at
  // the stop distance below entry, the position takes stopPct of its
  // dollar value. As a fraction of NLV that's posSizePct × stopPct /
  // 100 (both are already percents).
  const gapTailPctNlv = (posSizePct * stopPct) / 100;
  const gapTailDollars = (gapTailPctNlv / 100) * i.nlv;

  return {
    denominatorPct,
    denominatorFloored,
    formulaPct,
    capBound,
    posSizePct,
    posDollars,
    shares,
    stopPrice,
    stopPct,
    gapTailPctNlv,
    gapTailDollars,
    isSpeculative: stopPct > SPEC_TIER_STOP_PCT,
  };
}

// ─────────────────────────────────────────────────────────────────────
// Small primitives
// ─────────────────────────────────────────────────────────────────────

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-[10px] uppercase tracking-[0.08em] font-semibold mb-1"
             style={{ color: "var(--ink-4)" }}>{label}</label>
      {children}
      {hint && (
        <div className="text-[10px] mt-1" style={{ color: "var(--ink-4)" }}>{hint}</div>
      )}
    </div>
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

      // Auto-mode = deriveAutoSizingMode (state + exit-ladder floor).
      // Matches Position Sizer + Log Buy so the three pages agree.
      setSizingMode(deriveAutoSizingMode(state, exits).idx);
      setSizingModeManual(false);

      const loss = (avgLoss as { avg_loss_pct?: number | null } | null)?.avg_loss_pct ?? null;
      const sample = (avgLoss as { sample_size?: number } | null)?.sample_size ?? 0;
      // Store as POSITIVE — the backend returns negatives; computeNewEntry
      // takes a positive magnitude to compare against the 4% floor.
      setAvgLossPct(typeof loss === "number" ? Math.abs(loss) : null);
      setAvgLossSample(sample);
    }).catch(err => log.error("new-entry", "mount fetch failed", err));
  }, []);

  // Debounced price + ATR lookup on ticker changes. Same 600ms cadence
  // Position Sizer uses; keeps the network cheap while the user types.
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

  // Downward-only manual override: after the user clicks a mode, clamp
  // to the auto index. This is what makes New Entry stricter than
  // Position Sizer — the user can't manually re-lift above what the
  // regime + exit-ladder permit.
  const handleModePick = useCallback((idx: 0 | 1 | 2) => {
    const clamped = clampManualToDownwardOnly(autoIdx, idx);
    setSizingMode(clamped);
    setSizingModeManual(clamped !== autoIdx);
  }, [autoIdx]);

  const handleResetToAuto = useCallback(() => {
    setSizingMode(autoIdx);
    setSizingModeManual(false);
  }, [autoIdx]);

  // Guard — only compute when every numeric input is a valid finite
  // positive number. Missing / bad → no verdict rendered.
  const entryNum = Number(entry);
  const canCompute =
    Number.isFinite(entryNum) && entryNum > 0
    && atrPct != null && Number.isFinite(atrPct) && atrPct > 0
    && nlv > 0
    && avgLossPct != null;

  const results = useMemo(() => {
    if (!canCompute) return null;
    return computeNewEntry({
      entry: entryNum,
      atrPct: atrPct!,
      nlv,
      riskUnitPct: SIZING_MODES[sizingMode].pct,
      avgLossPct: avgLossPct!,
      targetCapPct,
    });
  }, [canCompute, entryNum, atrPct, nlv, sizingMode, avgLossPct, targetCapPct]);

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
          <p className="mb-1"><strong>Formula:</strong></p>
          <ul className="list-disc ml-4 mb-2">
            <li>Denominator = <code>max(4.0%, |trailing 12-mo avg loss|)</code> — realized average loss, not the hard stop</li>
            <li>Position size % = <code>risk unit % / denominator % × 100</code></li>
            <li>Cap = the selected Target Position Size button (default Overweight 12.5%)</li>
          </ul>
          <p className="mb-1"><strong>Catastrophe backstop:</strong></p>
          <ul className="list-disc ml-4 mb-2">
            <li><strong>1.5× ATR21</strong> below entry — displayed, not fed into sizing</li>
            <li><strong>Gap tail</strong> = position size % × stop % (the % NLV lost if the market gaps through the stop)</li>
            <li>1.5× ATR21 &gt; 8% flags SPECULATIVE TIER (visual only; the denominator already shrinks size)</li>
          </ul>
          <p className="mb-1"><strong>Overrides:</strong></p>
          <ul className="list-disc ml-4 mb-2">
            <li>Sizing mode manual override is <strong>DOWNWARD ONLY</strong> — you may pick smaller than the regime allows, never larger</li>
            <li>Target Position Size buttons remain freely selectable in both directions</li>
            <li>Negative Trend Count surfaces an amber banner (SR8 cascade only); does not gate this recommendation</li>
          </ul>
          <p className="mb-2 text-[11px]" style={{ color: "var(--ink-4)" }}>
            Parallel to Position Sizer during evaluation. The old
            recommendation continues to be available there for
            side-by-side comparison.
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

      {/* Sizing-mode indicator (auto-derived, downward-only override) */}
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

      {/* Inputs */}
      <div className="grid gap-4 mb-4"
           style={{ gridTemplateColumns: "1fr 1fr" }}>
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
      </div>

      {/* Sizing mode override row (downward-only enforced by handleModePick) */}
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

      {/* Verdict */}
      {results ? (
        <div className="rounded-[14px] overflow-hidden mb-4"
             data-testid="new-entry-verdict"
             style={{ background: "var(--surface)", border: `2px solid ${navColor}`, boxShadow: "var(--card-shadow)" }}>
          <div className="px-5 py-4">
            <div className="text-[10px] uppercase tracking-[0.1em] font-bold mb-2"
                 style={{ color: navColor }}>
              Verdict
            </div>
            <div className="text-[22px] font-bold mb-1" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
              Buy {results.shares.toLocaleString()} shares
            </div>
            <div className="text-[12px] mb-3" style={{ color: "var(--ink-3)" }}>
              {results.posSizePct.toFixed(1)}% NLV
              {" · "}risk {(SIZING_MODES[sizingMode].pct).toFixed(2)}% NLV
              {" · "}gap tail {results.gapTailPctNlv.toFixed(2)}% NLV
              {" · "}{currentMode.label.split(" (")[0]} risk unit
            </div>
            <div className="text-[11px] grid gap-1" style={{ color: "var(--ink-4)" }}>
              <div>
                <strong>Position cost:</strong> {formatCurrency(results.posDollars, { decimals: 0 })}
                {" @ "}${entryNum.toFixed(2)}/share
              </div>
              <div>
                <strong>Formula:</strong> {SIZING_MODES[sizingMode].pct.toFixed(2)}% ÷ {results.denominatorPct.toFixed(2)}% × 100 = {results.formulaPct.toFixed(2)}%
                {results.denominatorFloored && (
                  <span style={{ color: "#a16207" }}>
                    {" "}(denominator floored at {AVG_LOSS_FLOOR_PCT}%)
                  </span>
                )}
              </div>
              <div>
                <strong>Cap:</strong> Target Position Size {targetCapPct}%
                {results.capBound
                  ? <span style={{ color: navColor }}> — BOUND (cap tighter than formula)</span>
                  : <span> — formula {results.formulaPct.toFixed(2)}% is under the cap</span>}
              </div>
            </div>
          </div>

          {/* Catastrophe backstop line */}
          <div className="px-5 py-3"
               style={{ background: "var(--bg)", borderTop: "1px solid var(--border)" }}>
            <div className="text-[10px] uppercase tracking-[0.08em] font-semibold mb-1"
                 style={{ color: "var(--ink-4)" }}>
              Catastrophe backstop
            </div>
            <div className="text-[12px]" style={{ color: "var(--ink-2)" }}>
              1.5× ATR21 stop @ ${results.stopPrice.toFixed(2)} (−{results.stopPct.toFixed(2)}% below entry).
              {" "}If gapped through stop:{" "}
              <strong style={{ color: "#e5484d" }}>
                −{results.gapTailPctNlv.toFixed(2)}% NLV
                {" "}({formatCurrency(results.gapTailDollars, { decimals: 0 })})
              </strong>
            </div>
            {results.isSpeculative && (
              <div className="mt-2 rounded-[8px] px-3 py-2 text-[11px] font-semibold"
                   data-testid="new-entry-speculative-warning"
                   style={{
                     background: "color-mix(in oklab, #e5484d 12%, var(--surface))",
                     border: "1px solid color-mix(in oklab, #e5484d 40%, var(--border))",
                     color: "#b91c1c",
                   }}>
                🎯 SPECULATIVE TIER — 1.5× ATR21 ({results.stopPct.toFixed(2)}%) exceeds {SPEC_TIER_STOP_PCT}%. Formula already shrinks size via denominator; treat as a smaller probe.
              </div>
            )}
          </div>
        </div>
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
                Denominator used: {Math.max(AVG_LOSS_FLOOR_PCT, avgLossPct).toFixed(2)}%
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
