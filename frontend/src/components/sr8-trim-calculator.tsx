"use client";

import { useState, useEffect, useMemo } from "react";
import type { EnrichedPosition } from "@/lib/positions";
import { formatCurrency } from "@/lib/format";
import {
  computeTrim,
  RULE_OPTIONS,
  type TrimRule,
  type SR7CushionTier,
} from "@/lib/sr8-trim-calc";

// Persists the user's last-typed NAV across page mounts so the
// calculator stays usable without re-entering. Key is namespaced so
// it doesn't collide with any other app-level storage.
const NAV_STORAGE_KEY = "mo-sr8-trim-nav";

function useLocalStorage(key: string, initial: string): [string, (v: string) => void] {
  const [value, setValue] = useState<string>(() => {
    if (typeof window === "undefined") return initial;
    try {
      const raw = window.localStorage.getItem(key);
      return raw ?? initial;
    } catch {
      return initial;
    }
  });
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      window.localStorage.setItem(key, value);
    } catch {
      /* quota / blocked; ignore */
    }
  }, [key, value]);
  return [value, setValue];
}

const MONO: React.CSSProperties = { fontFamily: "var(--font-jetbrains), monospace" };

function Label({ children }: { children: React.ReactNode }) {
  return (
    <label
      className="block text-[9px] uppercase tracking-[0.10em] font-semibold mb-1.5"
      style={{ color: "var(--ink-4)" }}
    >
      {children}
    </label>
  );
}

function StateChip({ state }: { state: string }) {
  // Tone matches existing app conventions; sr8-style chip from active-campaign.
  const tones: Record<string, { bg: string; fg: string; text: string }> = {
    "core-only": { bg: "color-mix(in oklab, #08a86b 12%, var(--surface))", fg: "#16a34a", text: "Core only" },
    "with-adds": { bg: "color-mix(in oklab, #6366f1 12%, var(--surface))", fg: "#4f46e5", text: "With ADDS" },
    "below-core": { bg: "color-mix(in oklab, #f59f00 12%, var(--surface))", fg: "#d97706", text: "Below core" },
    closed: { bg: "color-mix(in oklab, #e5484d 12%, var(--surface))", fg: "#dc2626", text: "Closed" },
    invalid: { bg: "var(--surface-2)", fg: "var(--ink-4)", text: "—" },
  };
  const tone = tones[state] || tones["invalid"];
  return (
    <span
      className="inline-block px-2 py-0.5 rounded-full text-[10px] font-semibold whitespace-nowrap"
      style={{ background: tone.bg, color: tone.fg }}
    >
      {tone.text}
    </span>
  );
}

function CushionTierBadge({ tier }: { tier: SR7CushionTier }) {
  const tones: Record<SR7CushionTier, { bg: string; fg: string; text: string }> = {
    gt50: { bg: "color-mix(in oklab, #08a86b 12%, var(--surface))", fg: "#16a34a", text: ">50% cushion" },
    "25to50": { bg: "color-mix(in oklab, #f59f00 12%, var(--surface))", fg: "#d97706", text: "25–50% cushion" },
    lt25: { bg: "color-mix(in oklab, #e5484d 12%, var(--surface))", fg: "#dc2626", text: "<25% cushion" },
  };
  const tone = tones[tier];
  return (
    <span
      className="inline-block px-2 py-0.5 rounded-full text-[10px] font-semibold whitespace-nowrap"
      style={{ background: tone.bg, color: tone.fg }}
    >
      {tone.text}
    </span>
  );
}

type Props = {
  /** Positions already filtered to sell_rule_tier='sr8' by the caller. */
  positions: EnrichedPosition[];
  /** When set, locks the position picker to this trade_id. ACS modal uses this. */
  preselectedTradeId?: string;
};

export function SR8TrimCalculator({ positions, preselectedTradeId }: Props) {
  const [selectedTradeId, setSelectedTradeId] = useState<string>(
    preselectedTradeId ?? positions[0]?.trade_id ?? "",
  );
  // Keep the local selection in sync when the preselect changes (modal
  // re-mounts with a different position).
  useEffect(() => {
    if (preselectedTradeId) setSelectedTradeId(preselectedTradeId);
  }, [preselectedTradeId]);

  const [navInput, setNavInput] = useLocalStorage(NAV_STORAGE_KEY, "");
  const [rule, setRule] = useState<TrimRule>("sr7");

  const position = useMemo(
    () => positions.find((p) => p.trade_id === selectedTradeId),
    [positions, selectedTradeId],
  );

  // Strip $, commas, whitespace before parseFloat. Accept "612636",
  // "$612,636", "612,636.00".
  const navRaw = navInput.replace(/[,$\s]/g, "");
  const nav = parseFloat(navRaw);
  const navValid = Number.isFinite(nav) && nav > 0;

  const result =
    position && navValid
      ? computeTrim({
          totalShares: position.shares,
          currentPrice: position.current_price,
          b1ReturnPct: position.b1_return_pct,
          nav,
          rule,
        })
      : null;

  // ─── empty / hint states ──────────────────────────────────────────────
  if (positions.length === 0) {
    return (
      <div
        className="rounded-[10px] p-6 text-[12px] text-center"
        style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-4)" }}
      >
        No positions currently classified as SR8.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5" style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="text-[13px]" style={{ color: "var(--ink-3)" }}>
        Computes recommended share count to sell when a sell rule fires on an SR8 position.
        Pure calculator — take the number to your broker.
      </div>

      {/* ─── Inputs row ────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <Label>Position</Label>
          {preselectedTradeId ? (
            <div
              className="h-[42px] px-3.5 rounded-[10px] flex items-center text-[13px]"
              style={{ background: "var(--surface-2)", border: "1px solid var(--border)", ...MONO }}
              aria-readonly
            >
              {position ? `${position.ticker} (${position.trade_id})` : "—"}
            </div>
          ) : (
            <select
              value={selectedTradeId}
              onChange={(e) => setSelectedTradeId(e.target.value)}
              className="w-full h-[42px] px-3 rounded-[10px] text-[13px] outline-none"
              style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)" }}
            >
              {positions.map((p) => (
                <option key={p.trade_id} value={p.trade_id}>
                  {p.ticker} — {p.shares} sh @ {formatCurrency(p.current_price)}
                </option>
              ))}
            </select>
          )}
        </div>

        <div>
          <Label>NAV ($)</Label>
          <input
            type="text"
            value={navInput}
            onChange={(e) => setNavInput(e.target.value)}
            placeholder="$612,636"
            className="w-full h-[42px] px-3.5 rounded-[10px] text-[13px] outline-none privacy-mask"
            style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", ...MONO }}
          />
        </div>

        <div>
          <Label>Sell Rule</Label>
          <select
            value={rule}
            onChange={(e) => setRule(e.target.value as TrimRule)}
            className="w-full h-[42px] px-3 rounded-[10px] text-[13px] outline-none"
            style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)" }}
          >
            {RULE_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* ─── Hint states ───────────────────────────────────────────────── */}
      {!position && (
        <div
          className="rounded-[10px] p-4 text-[12px]"
          style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-4)" }}
        >
          Select a position to begin.
        </div>
      )}

      {position && !navValid && (
        <div
          className="rounded-[10px] p-4 text-[12px]"
          style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-4)" }}
        >
          {navInput.trim() === "" ? "Enter NAV to compute trim." : "NAV must be a positive number."}
        </div>
      )}

      {position && position.current_price <= 0 && (
        <div
          className="rounded-[10px] p-4 text-[12px]"
          style={{
            background: "color-mix(in oklab, #e5484d 8%, var(--surface))",
            border: "1px solid color-mix(in oklab, #e5484d 30%, var(--border))",
            color: "#dc2626",
          }}
        >
          Live price unavailable for {position.ticker} — refresh the page.
        </div>
      )}

      {/* ─── Position state ────────────────────────────────────────────── */}
      {position && result && result.resultingState !== "invalid" && (
        <div className="rounded-[10px] overflow-hidden" style={{ border: "1px solid var(--border)" }}>
          <div
            className="px-4 py-2.5 text-[10px] uppercase tracking-[0.08em] font-semibold flex items-center justify-between"
            style={{ background: "var(--surface-2)", borderBottom: "1px solid var(--border)", color: "var(--ink-4)" }}
          >
            <span>Position state — {position.ticker}</span>
            {result.sr7CushionTier && <CushionTierBadge tier={result.sr7CushionTier} />}
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-3 p-4 text-[12px]">
            <div>
              <div className="text-[9px] uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>Total shares</div>
              <div style={{ ...MONO }} className="privacy-mask">{position.shares.toLocaleString()}</div>
            </div>
            <div>
              <div className="text-[9px] uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>Total value</div>
              <div style={{ ...MONO }} className="privacy-mask">{formatCurrency(result.totalValue)}</div>
            </div>
            <div>
              <div className="text-[9px] uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>NAV %</div>
              <div style={{ ...MONO }} className="privacy-mask">{result.totalNavPct.toFixed(1)}%</div>
            </div>
            <div>
              <div className="text-[9px] uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>B1 cushion</div>
              <div style={{ ...MONO }}>
                {result.currentCushionPct != null
                  ? `${result.currentCushionPct >= 0 ? "+" : ""}${result.currentCushionPct.toFixed(2)}%`
                  : "—"}
              </div>
            </div>

            <div>
              <div className="text-[9px] uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>Core target (15% NAV)</div>
              <div style={{ ...MONO }} className="privacy-mask">{formatCurrency(result.coreTargetValue)}</div>
            </div>
            <div>
              <div className="text-[9px] uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>Core target shares</div>
              <div style={{ ...MONO }} className="privacy-mask">{result.coreTargetShares.toLocaleString()}</div>
            </div>
            <div>
              <div className="text-[9px] uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>ADDS shares</div>
              <div style={{ ...MONO }} className="privacy-mask">{result.addsShares.toLocaleString()}</div>
            </div>
            <div>
              <div className="text-[9px] uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>Price</div>
              <div style={{ ...MONO }} className="privacy-mask">{formatCurrency(position.current_price)}</div>
            </div>
          </div>
        </div>
      )}

      {/* ─── Result ────────────────────────────────────────────────────── */}
      {position && result && result.resultingState !== "invalid" && (() => {
        // SR8 Quick / Quicksand are target-based: show the destination
        // NAV % and target share count so the user can sanity-check
        // the math against the NAV they entered.
        const targetPct = rule === "sr8-quick" ? 10 : rule === "sr8-quicksand" ? 5 : null;
        const targetShares = targetPct != null && navValid
          ? Math.floor((nav * (targetPct / 100)) / position.current_price)
          : null;
        const alreadyAtTarget = targetPct != null && result.trimShares === 0 && position.shares > 0;
        return (
          <div
            className="rounded-[10px] p-5"
            style={{
              background: "color-mix(in oklab, #6366f1 6%, var(--surface))",
              border: "1px solid color-mix(in oklab, #6366f1 30%, var(--border))",
            }}
          >
            <div className="text-[9px] uppercase tracking-[0.10em] font-semibold mb-1" style={{ color: "var(--ink-4)" }}>
              Recommendation
            </div>
            <div className="text-[28px] font-semibold tracking-tight" style={{ ...MONO, color: "var(--ink)" }}>
              SELL <span data-testid="trim-shares">{result.trimShares.toLocaleString()}</span> SHARES
            </div>
            {alreadyAtTarget && (
              <div className="text-[11px] mt-1" style={{ color: "var(--ink-4)" }} data-testid="already-at-target">
                Position already at or below {targetPct}% NAV target — nothing to trim.
              </div>
            )}
            {targetPct != null && targetShares != null && !alreadyAtTarget && (
              <div className="text-[11px] mt-1" style={{ color: "var(--ink-4)" }} data-testid="trim-target">
                Target: {targetPct}% NAV ({targetShares.toLocaleString()} shares)
              </div>
            )}
            <div className="text-[12px] mt-2 flex flex-wrap gap-x-4 gap-y-1" style={{ color: "var(--ink-3)" }}>
              <span>
                Intended: <span style={{ ...MONO }}>{result.intendedTrimShares.toLocaleString()}</span>
              </span>
            {result.coreFloorBinds && (
              <span
                className="inline-block px-2 py-0.5 rounded-full text-[10px] font-semibold whitespace-nowrap"
                style={{
                  background: "color-mix(in oklab, #f59f00 12%, var(--surface))",
                  color: "#d97706",
                }}
                data-testid="core-floor-binds-badge"
              >
                Core floor binds
              </span>
            )}
          </div>

          <div className="mt-4 pt-4 flex flex-wrap items-center gap-x-6 gap-y-2 text-[12px]"
               style={{ borderTop: "1px solid color-mix(in oklab, #6366f1 20%, var(--border))" }}>
            <span>
              <span className="text-[9px] uppercase tracking-[0.08em] mr-1.5" style={{ color: "var(--ink-4)" }}>Resulting</span>
              <span style={{ ...MONO }} className="privacy-mask">
                {result.resultingShares.toLocaleString()} sh · {formatCurrency(result.resultingValue)} · {result.resultingNavPct.toFixed(1)}% NAV
              </span>
            </span>
            <StateChip state={result.resultingState} />
          </div>
        </div>
        );
      })()}
    </div>
  );
}
