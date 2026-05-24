"use client";

import { useEffect, useMemo, useState } from "react";
import { ChevronDown, Lock } from "lucide-react";
import { Check } from "lucide-react";
import { api, getActivePortfolio } from "@/lib/api";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";
import { usePortfolio } from "@/lib/portfolio-context";
import { SIZING_MODES, mctStateToSizingMode } from "@/lib/sizing-mode";
import { computeVolatilitySizing, type VolSizerResults, type SizingScenario } from "@/lib/vol-sizer";
import { MobileSelectSheet } from "./mobile-select-sheet";

/**
 * Mobile Position Sizer — volatility flow on the shared vol-sizer lib.
 *
 * Mount fetch: journalLatest (equity = end_nlv) + rallyPrefix (MCT state
 * → auto sizing mode). Per-ticker debounced priceLookup auto-fills
 * entry + ATR. Computation runs live as inputs change — no Calculate
 * gate (anchor v6's "audit-result is live" design).
 *
 * Math delegates to frontend/src/lib/vol-sizer.ts: tech-stop sizing +
 * three fixed ATR-cushion scenarios (1× / 1.5× / 2×) + recommendation
 * + warning. The legacy Stock Volatility Profile picker is gone; the
 * three multipliers are baked into the lib.
 *
 * Active-portfolio switch triggers window.location.reload() via the
 * shared usePortfolio() context, so this component doesn't need its
 * own reactivity wiring — the page remounts.
 */

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

const DEFAULT_SIZE_INDEX = 3;        // Full (10%) — matches Phase 1 anchor

export function MobilePositionSizer() {
  const { activePortfolio } = usePortfolio();

  // Inputs (user-editable)
  const [ticker, setTicker] = useState("");
  const [entryPrice, setEntryPrice] = useState("");
  const [maLevel, setMaLevel] = useState("");
  const [buffer, setBuffer] = useState("1.00");
  const [sizingMode, setSizingMode] = useState<0 | 1 | 2>(1); // overwritten on mount
  const [sizingModeManual, setSizingModeManual] = useState(false);
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
  const tolPct = SIZING_MODES[sizingMode].pct;
  const targetSize = SIZE_OPTIONS[sizeIdx].pct;

  // Calculated stop banner derives from raw inputs (not the lib) so it
  // surfaces even when the lib bails (e.g. ma == 0 before the user has
  // entered one). The ATR-fraction annotation is only meaningful when
  // both atr and stop are positive.
  const calcStop = ma > 0 ? ma * (1 - buf / 100) : 0;
  const stopDistPct = entry > 0 && calcStop > 0 && calcStop < entry ? ((entry - calcStop) / entry) * 100 : 0;
  const calcAtrFraction = atr > 0 && stopDistPct > 0 ? stopDistPct / atr : null;

  // Live computation — delegates to the shared vol-sizer lib. Returns
  // null when essential inputs are missing so the UI renders "—"
  // placeholders instead of computing on zero-divided data. The lib's
  // input validators are stricter (require ma > 0 and stop < entry);
  // anything looser is caught here and short-circuits.
  const audit: VolSizerResults | null = useMemo(() => {
    if (entry <= 0 || eq <= 0 || atr <= 0 || ma <= 0) return null;
    if (calcStop >= entry) return null; // degenerate stop — UI will render "—"
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
  }, [entry, ma, buf, eq, atr, tolPct, targetSize, calcStop]);

  const rec = audit?.recommended ?? null;
  const recIsTechStop = audit?.recommendationReason === "tech_stop_safe";

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

      {/* Mode / Size picker tiles (Profile picker removed — the three
          ATR cushion scenarios are now shown side-by-side in the result
          card, eliminating the user-facing multiplier choice). */}
      <div className="grid grid-cols-2 gap-2">
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

      {/* Calculated Stop banner — surfaces even before the lib has all
          inputs, so the user sees the tech stop they're configuring. */}
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

      {/* Warning sub-banner: tech stop sits inside 1 ATR (or is invalid). */}
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

      {/* Tech Stop card */}
      {audit && (
        <ScenarioRow
          scenario={audit.techStop}
          entry={entry}
          targetSize={targetSize}
          isRecommended={!!recIsTechStop}
        />
      )}

      {/* ATR Cushion cards (stacked on mobile) */}
      {audit && (
        <div className="flex flex-col gap-2">
          {audit.atrScenarios.map((s, i) => (
            <ScenarioRow
              key={s.label}
              scenario={s}
              entry={entry}
              targetSize={targetSize}
              isRecommended={!recIsTechStop && i === 1}
            />
          ))}
        </div>
      )}

      {/* Verdict card */}
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

      {/* Placeholder when inputs not ready */}
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
  );
}

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
  entry,
  targetSize,
  isRecommended,
}: {
  scenario: SizingScenario;
  entry: number;
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
