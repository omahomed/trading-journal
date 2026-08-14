// L-series exposure governor — shared classification + metadata.
//
// Migration 068 replaced the old ATH-anchored L1/L2/L3 "hard deck" model
// with a cycle-anchored L1 plus three IXIC-structural levels (L2/L3/L4).
// The Risk Manager page reads the composed state from GET /api/risk/levels
// (see api/main.py:get_risk_levels for the response contract). This module
// exposes the display metadata for each level so the page + any consumers
// render consistent trigger text, colors, and cap %.
//
// Old ATH-based HARD_DECKS are retained as LEGACY_HARD_DECKS ONLY for the
// Analytics scorecard classifier — that surface buckets historical days
// into L1_Aware / L1_Drifted style categories from a signed drawdown value
// and hasn't been reworked to the cycle-reference model yet. Do NOT reach
// for LEGACY_HARD_DECKS from any new surface; use the endpoint.

export type DeckLevel = "L0" | "L1" | "L2" | "L3" | "L4";

export interface LSeriesSpec {
  key: "L1" | "L2" | "L3" | "L4";
  cap_pct: number;               // Max gross exposure (80 / 60 / 40 / 20)
  action: string;
  trigger: string;
  color: string;
}

// Canonical L-series metadata. cap_pct + trigger match what the backend
// endpoint returns; the frontend uses these mostly for display styling +
// static reference in tests. Ordered shallow → deep (L1 to L4).
export const L_SERIES: readonly LSeriesSpec[] = [
  {
    key: "L1",
    cap_pct: 80,
    action: "Off margin",
    trigger: "NLV −7.5% from cycle reference",
    color: "#f59f00",
  },
  {
    key: "L2",
    cap_pct: 60,
    action: "Cap 60% gross",
    trigger: "IXIC close below 21 EMA + next-day undercut >1%",
    color: "#f97316",
  },
  {
    key: "L3",
    cap_pct: 40,
    action: "Cap 40% gross",
    trigger: "IXIC 2 consecutive closes below 21 EMA",
    color: "#ea580c",
  },
  {
    key: "L4",
    cap_pct: 20,
    action: "Cap 20% — SR8 holds only",
    trigger: "IXIC 2 consecutive closes below 50 SMA",
    color: "#dc2626",
  },
];

export const L_SERIES_META: Record<DeckLevel, { label: string; sub: string; color: string }> = {
  L0: { label: "L0", sub: "All Clear",              color: "#08a86b" },
  L1: { label: "L1", sub: "Off Margin",             color: "#f59f00" },
  L2: { label: "L2", sub: "Cap 60% Gross",          color: "#f97316" },
  L3: { label: "L3", sub: "Cap 40% Gross",          color: "#ea580c" },
  L4: { label: "L4", sub: "Cap 20% — SR8 Only",     color: "#dc2626" },
};

// Legacy meta the Command Center still renders while it stays on the
// ATH-drawdown classifier. Preserves the old sub copy so that surface
// doesn't accidentally flip to L-series wording without the L-series
// semantics behind it.
export const LEGACY_DECK_META: Record<"L0" | "L1" | "L2" | "L3", { label: string; sub: string; color: string }> = {
  L0: { label: "L0", sub: "All Clear",         color: "#08a86b" },
  L1: { label: "L1", sub: "Remove Margin",     color: "#f59f00" },
  L2: { label: "L2", sub: "Max 30% Invested",  color: "#f97316" },
  L3: { label: "L3", sub: "Go To Cash",        color: "#dc2626" },
};

// ── Legacy (Analytics scorecard only) ────────────────────────────────
// Historical ATH-drawdown classifier. NOT used by Risk Manager, NOT used
// by Dashboard, NOT used by Daily Journal — those all read the cycle-
// anchored endpoint. Kept alive because the Analytics scorecard buckets
// per-day snapshots (L1_Aware / L1_Drifted) from a signed drawdown, and
// converting that surface to the cycle-reference model is a separate,
// larger rework.

export interface LegacyDeckSpec {
  key: "L1" | "L2" | "L3";
  pct: number;
  action: string;
  color: string;
}

export const LEGACY_HARD_DECKS: readonly LegacyDeckSpec[] = [
  { key: "L1", pct: 7.5,  action: "Remove margin",     color: "#f59f00" },
  { key: "L2", pct: 12.5, action: "Max 30% invested",  color: "#f97316" },
  { key: "L3", pct: 15.0, action: "Go to cash",        color: "#dc2626" },
];

/**
 * Classify a signed OR magnitude drawdown into a legacy deck bucket.
 * Takes abs() so callers using either convention (signed like
 * dashboard_metrics vs. magnitude like Risk Manager's local math)
 * land on the same bucket. LEGACY: use for Analytics historical
 * bucketing only. Live risk state comes from GET /api/risk/levels.
 */
export function classifyLegacyDeck(ddPct: number | null | undefined): "L0" | "L1" | "L2" | "L3" {
  if (ddPct == null || Number.isNaN(ddPct)) return "L0";
  const abs = Math.abs(ddPct);
  if (abs >= LEGACY_HARD_DECKS[2].pct) return "L3";
  if (abs >= LEGACY_HARD_DECKS[1].pct) return "L2";
  if (abs >= LEGACY_HARD_DECKS[0].pct) return "L1";
  return "L0";
}

// Back-compat exports so existing imports keep resolving during the
// codebase-wide rewire. New code should import L_SERIES / L_SERIES_META
// / classifyLegacyDeck directly. DECK_META still points at the legacy
// meta so Command Center (and any other ATH-classifier consumers) render
// the old copy — the new L-series meta is L_SERIES_META and only surfaces
// where the endpoint is consumed. TODO: remove after every consumer is
// on the endpoint.
export const HARD_DECKS = LEGACY_HARD_DECKS;
export const DECK_META = LEGACY_DECK_META;
export const classifyDeck = classifyLegacyDeck;
