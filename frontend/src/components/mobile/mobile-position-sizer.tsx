"use client";

import { useEffect, useMemo, useState } from "react";
import { ChevronDown, Lock } from "lucide-react";
import { Check } from "lucide-react";
import { api, getActivePortfolio } from "@/lib/api";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";
import { usePortfolio } from "@/lib/portfolio-context";
import { SIZING_MODES, mctStateToSizingMode } from "@/lib/sizing-mode";
import { MobileSelectSheet } from "./mobile-select-sheet";

/**
 * Phase 2 Step 3 — Mobile Position Sizer (Volatility-tab first cut).
 *
 * Replaces Phase 1's MOCK-based mock with real data hooks consuming the
 * SAME endpoints the desktop sizer uses. Volatility-mode "new trade"
 * flow only — Normal / Scale-In / Pyramid / Trim / Options remain
 * desktop-only as Tier 2 follow-ups.
 *
 * Mount fetch: journalLatest (equity = end_nlv) + rallyPrefix (MCT state
 * → auto sizing mode). Per-ticker debounced priceLookup auto-fills
 * entry + ATR. Computation runs live as inputs change — no Calculate
 * gate (anchor v6's "audit-result is live" design).
 *
 * Sizing math mirrors desktop volResults (volSizerMode === "new"
 * branch) field-for-field — riskBudget, atrRiskBudget, maxSharesVol,
 * effectiveStop / maxSharesTech, hard-cap 20%, target cap, limit
 * reason, riskPerShare with MA-stop-or-ATR fallback.
 *
 * Active-portfolio switch triggers window.location.reload() via the
 * shared usePortfolio() context, so this component doesn't need its
 * own reactivity wiring — the page remounts.
 */

// Locked to match desktop's VOL_PROFILES (position-sizer.tsx:43-47).
// Reordering would silently shift behavior — DEFAULT_VOL_PROFILE_INDEX
// keys off array position.
const VOL_PROFILES = [
  { key: "tight", label: "Tight", mult: 1.0 },
  { key: "normal", label: "Normal", mult: 1.25 },
  { key: "highvol", label: "High-Vol", mult: 1.5 },
] as const;

// Locked to match desktop's SIZE_OPTIONS (position-sizer.tsx:49-53).
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

const DEFAULT_VOL_PROFILE_INDEX = 0; // Tight — matches Phase 1 anchor
const DEFAULT_SIZE_INDEX = 3;        // Full (10%) — matches Phase 1 anchor

const HARD_CAP_PCT = 20;

export function MobilePositionSizer() {
  const { activePortfolio } = usePortfolio();

  // Inputs (user-editable)
  const [ticker, setTicker] = useState("");
  const [entryPrice, setEntryPrice] = useState("");
  const [maLevel, setMaLevel] = useState("");
  const [buffer, setBuffer] = useState("1.00");
  const [sizingMode, setSizingMode] = useState<0 | 1 | 2>(1); // overwritten on mount
  const [sizingModeManual, setSizingModeManual] = useState(false);
  const [volProfileIdx, setVolProfileIdx] = useState<number>(DEFAULT_VOL_PROFILE_INDEX);
  const [sizeIdx, setSizeIdx] = useState<number>(DEFAULT_SIZE_INDEX);

  // Fetched / lifecycle
  const [equity, setEquity] = useState<number | null>(null);
  const [atrPct, setAtrPct] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [priceError, setPriceError] = useState<string | null>(null);
  const [priceLoading, setPriceLoading] = useState(false);

  // Mount fetch — equity + MCT state for auto sizing mode. Same shape
  // as desktop's load effect; ignores the parts we don't need (open
  // trades, tradesOpenDetails, pyramid_rules — all are tab-specific to
  // Scale-In / Pyramid / Trim).
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
    ]).then(([j, rally]) => {
      if (cancelled) return;
      const endNlv = j ? parseFloat(String((j as { end_nlv?: number | string }).end_nlv ?? 0)) : 0;
      setEquity(Number.isFinite(endNlv) && endNlv > 0 ? endNlv : null);
      const stateStr = (rally as { state?: string } | null)?.state ?? null;
      // Auto-apply only if the user hasn't manually overridden — on mount
      // they couldn't have, but defending against a future re-run.
      setSizingMode((prev) => (sizingModeManual ? prev : mctStateToSizingMode(stateStr)));
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Ticker → priceLookup, debounced 600ms (matches desktop). On
  // success, auto-fills entryPrice + atrPct. On error, leaves the
  // inputs editable and surfaces the message near the ATR cell.
  useEffect(() => {
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
  }, [ticker]);

  // Derived inputs
  const entry = parseFloat(entryPrice) || 0;
  const ma = parseFloat(maLevel) || 0;
  const buf = parseFloat(buffer) || 1;
  const atr = atrPct ?? 0;
  const eq = equity ?? 0;
  const riskPct = SIZING_MODES[sizingMode].pct;
  const targetSize = SIZE_OPTIONS[sizeIdx].pct;
  const atrMultiplier = VOL_PROFILES[volProfileIdx].mult;

  // Live computation — mirrors desktop volResults (volSizerMode === "new").
  // Audit returns null when essential inputs are missing so the UI can
  // render "—" placeholders instead of computing on zero-divided data.
  const audit = useMemo(() => {
    if (entry <= 0 || eq <= 0 || atr <= 0) return null;

    const dailyRiskBudget = eq * (riskPct / 100);
    const atrRiskBudget = dailyRiskBudget * atrMultiplier;
    const atrDecimal = atr / 100;
    const maxSharesVol = Math.ceil(atrRiskBudget / (entry * atrDecimal));

    let maxSharesTech = Number.POSITIVE_INFINITY;
    let effectiveStop = 0;
    if (ma > 0) {
      effectiveStop = ma * (1 - buf / 100);
      if (effectiveStop < entry) {
        const rps = entry - effectiveStop;
        maxSharesTech = Math.ceil(dailyRiskBudget / rps);
      }
    }

    const maxSharesCap = Math.floor((eq * (HARD_CAP_PCT / 100)) / entry);
    const maxSharesTarget = Math.ceil((eq * (targetSize / 100)) / entry);

    const finalMaxShares = Math.min(maxSharesVol, maxSharesTech, maxSharesCap, maxSharesTarget);
    const finalMaxVal = finalMaxShares * entry;
    const finalPctNlv = eq > 0 ? (finalMaxVal / eq) * 100 : 0;

    let limitReason = "Volatility (ATR)";
    if (
      finalMaxShares === maxSharesTarget &&
      maxSharesTarget < Math.min(maxSharesVol, maxSharesTech, maxSharesCap)
    ) {
      limitReason = `Target Size (${targetSize}%)`;
    } else if (finalMaxShares === maxSharesCap) {
      limitReason = `Hard Cap (${HARD_CAP_PCT}%)`;
    } else if (finalMaxShares === maxSharesTech) {
      limitReason = `MA Support (${formatCurrency(ma)})`;
    }

    let riskPerShare: number;
    let stopForDisplay: number;
    if (effectiveStop > 0 && effectiveStop < entry) {
      riskPerShare = entry - effectiveStop;
      stopForDisplay = effectiveStop;
    } else {
      riskPerShare = entry * atrDecimal;
      stopForDisplay = Math.max(0, entry - riskPerShare);
    }
    const finalRiskDol = finalMaxShares * riskPerShare;
    const target2r = finalRiskDol * 2;

    return {
      shares: finalMaxShares,
      notional: finalMaxVal,
      pctOfNlv: finalPctNlv,
      totalRisk: finalRiskDol,
      stop: stopForDisplay,
      target2r,
      limitReason,
    };
  }, [entry, ma, buf, eq, atr, riskPct, atrMultiplier, targetSize]);

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

      {/* Mode chip — kept as the anchor v6 design contract dictates.
          Volatility is the only mode this step supports; other tabs
          (Normal / Scale-In / Pyramid / Trim / Options) ship as Tier 2
          follow-ups. No tap target until that lands. */}
      <div className="flex items-center justify-between rounded-m-md border-[0.5px] border-m-border bg-m-surface px-[14px] py-[10px]">
        <div className="flex items-center gap-2.5">
          <span className="text-[11px] font-medium text-m-text-dim">Mode</span>
          <span className="text-sm font-medium text-m-text">Volatility</span>
        </div>
        <ChevronDown size={14} strokeWidth={1.5} className="text-m-text-faint" aria-hidden="true" />
      </div>

      {/* Ticker card */}
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

      {/* 2×2 input grid */}
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
      <div className="flex items-baseline justify-between rounded-m-md border-[0.5px] border-m-border bg-m-surface px-[14px] py-[10px]">
        <span className="text-[11px] font-medium text-m-text-dim">ATR % (21D)</span>
        <span className="font-m-num text-xl font-medium tabular-nums tracking-[-0.01em] text-m-text">
          {atrDisplay}
        </span>
      </div>

      {/* Mode / Profile / Size picker tiles */}
      <div className="grid grid-cols-3 gap-2">
        <MobileSelectSheet
          triggerLabel="Mode"
          triggerValue={SIZING_MODES[sizingMode].key === "defense"
            ? "Defense"
            : SIZING_MODES[sizingMode].key === "normal"
              ? "Normal"
              : "Offense"}
          triggerSubValue={`${SIZING_MODES[sizingMode].pct.toFixed(2)}%`}
          triggerAccent
          triggerSelected
          sheetTitle="Sizing mode"
        >
          {(close) => (
            <div className="flex flex-col">
              {SIZING_MODES.map((m) => {
                const isActive = m.index === sizingMode;
                const displayLabel =
                  m.key === "defense" ? "Defense" : m.key === "normal" ? "Normal" : "Offense";
                return (
                  <button
                    key={m.key}
                    type="button"
                    role="option"
                    aria-selected={isActive}
                    onClick={() => {
                      setSizingMode(m.index);
                      setSizingModeManual(true);
                      close();
                    }}
                    className="flex min-h-[52px] items-center justify-between border-b-[0.5px] border-m-border px-5 py-3.5 text-left last:border-b-0"
                  >
                    <span className="flex flex-col">
                      <span className={`text-base ${isActive ? "font-medium" : ""} text-m-text`}>
                        {displayLabel}
                      </span>
                      <span className="font-m-num text-[12px] tabular-nums text-m-text-dim">
                        {m.pct.toFixed(2)}%
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

        <MobileSelectSheet
          triggerLabel="Profile"
          triggerValue={VOL_PROFILES[volProfileIdx].label}
          triggerSubValue={`${VOL_PROFILES[volProfileIdx].mult.toFixed(2)}×`}
          sheetTitle="ATR profile"
        >
          {(close) => (
            <div className="flex flex-col">
              {VOL_PROFILES.map((p, i) => {
                const isActive = i === volProfileIdx;
                return (
                  <button
                    key={p.key}
                    type="button"
                    role="option"
                    aria-selected={isActive}
                    onClick={() => {
                      setVolProfileIdx(i);
                      close();
                    }}
                    className="flex min-h-[52px] items-center justify-between border-b-[0.5px] border-m-border px-5 py-3.5 text-left last:border-b-0"
                  >
                    <span className="flex flex-col">
                      <span className={`text-base ${isActive ? "font-medium" : ""} text-m-text`}>
                        {p.label}
                      </span>
                      <span className="font-m-num text-[12px] tabular-nums text-m-text-dim">
                        {p.mult.toFixed(2)}× ATR
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

        <MobileSelectSheet
          triggerLabel="Size"
          triggerValue={SIZE_OPTIONS[sizeIdx].label}
          triggerSubValue={`${SIZE_OPTIONS[sizeIdx].pct}%`}
          sheetTitle="Target size"
        >
          {(close) => (
            <div className="flex flex-col">
              {SIZE_OPTIONS.map((o, i) => {
                const isActive = i === sizeIdx;
                return (
                  <button
                    key={o.label}
                    type="button"
                    role="option"
                    aria-selected={isActive}
                    onClick={() => {
                      setSizeIdx(i);
                      close();
                    }}
                    className="flex min-h-[52px] items-center justify-between border-b-[0.5px] border-m-border px-5 py-3.5 text-left last:border-b-0"
                  >
                    <span className="flex flex-col">
                      <span className={`text-base ${isActive ? "font-medium" : ""} text-m-text`}>
                        {o.label}
                      </span>
                      <span className="font-m-num text-[12px] tabular-nums text-m-text-dim">
                        {o.pct}% of NLV
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
      </div>

      {/* Audit result card */}
      <div className="rounded-m-xl border-[0.5px] border-m-border bg-m-surface px-5 pt-4 pb-3.5">
        <div className="flex items-baseline justify-between">
          <span className="text-[11px] font-medium text-m-text-dim">Audit result</span>
          {audit && (
            <span
              className={
                "text-[11px] font-medium " +
                (audit.pctOfNlv >= HARD_CAP_PCT ? "text-m-warn" : "text-m-text-muted")
              }
            >
              {audit.pctOfNlv.toFixed(1)}% of NLV
            </span>
          )}
        </div>
        <div className="mt-1.5 flex items-baseline gap-2">
          <span
            data-testid="audit-shares"
            className="font-m-num text-[38px] font-medium tabular-nums tracking-[-0.03em] text-m-text"
          >
            {audit ? audit.shares.toLocaleString() : "—"}
          </span>
          <span className="text-[15px] text-m-text-muted">
            shares
            {audit ? ` · ${formatCurrency(audit.notional, { decimals: 0 })}` : ""}
          </span>
        </div>
        {audit && (
          <div className="mt-1 text-[11px] text-m-text-dim">
            Limited by {audit.limitReason}
          </div>
        )}
        <div className="mt-3 grid grid-cols-3 gap-2.5 border-t-[0.5px] border-m-border pt-3">
          <Stat
            label="Total risk"
            value={audit ? formatCurrency(audit.totalRisk, { decimals: 0 }) : "—"}
            tone="down"
          />
          <Stat
            label="Stop"
            value={audit ? formatCurrency(audit.stop, { decimals: 0 }) : "—"}
          />
          <Stat
            label="2R target"
            value={audit ? formatCurrency(audit.target2r, { decimals: 0 }) : "—"}
            tone="up"
          />
        </div>
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

function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "up" | "down";
}) {
  const valueClass =
    tone === "down" ? "text-m-down" : tone === "up" ? "text-m-accent" : "text-m-text";
  return (
    <div>
      <div className="mb-0.5 text-[10px] text-m-text-dim">{label}</div>
      <div className={`font-m-num text-[13px] font-medium tabular-nums ${valueClass}`}>
        {value}
      </div>
    </div>
  );
}
