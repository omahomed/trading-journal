// Hard-deck classification shared by Risk Manager and Command Center.
// One source of truth so a threshold change here reflects on every page
// that renders a deck badge or status.

export type DeckLevel = "L0" | "L1" | "L2" | "L3";

export interface DeckSpec {
  key: "L1" | "L2" | "L3";
  pct: number;
  action: string;
  color: string;
}

export const HARD_DECKS: readonly DeckSpec[] = [
  { key: "L1", pct: 7.5, action: "Remove margin", color: "#f59f00" },
  { key: "L2", pct: 12.5, action: "Max 30% invested", color: "#f97316" },
  { key: "L3", pct: 15.0, action: "Go to cash", color: "#dc2626" },
];

export const DECK_META: Record<DeckLevel, { label: string; sub: string; color: string }> = {
  L0: { label: "L0",       sub: "All Clear",         color: "#08a86b" },
  L1: { label: "L1",       sub: "Remove Margin",     color: "#f59f00" },
  L2: { label: "L2",       sub: "Max 30% Invested",  color: "#f97316" },
  L3: { label: "L3",       sub: "Go To Cash",        color: "#dc2626" },
};

/**
 * Classify a drawdown percentage into a deck level.
 * Accepts either a positive magnitude (Risk Manager convention:
 * `(peak − current) / peak × 100`) or a signed value (dashboard_metrics
 * convention: negative when in drawdown). Takes the absolute value so
 * both callers land on the same bucket.
 */
export function classifyDeck(ddPct: number | null | undefined): DeckLevel {
  if (ddPct == null || Number.isNaN(ddPct)) return "L0";
  const abs = Math.abs(ddPct);
  if (abs >= HARD_DECKS[2].pct) return "L3";
  if (abs >= HARD_DECKS[1].pct) return "L2";
  if (abs >= HARD_DECKS[0].pct) return "L1";
  return "L0";
}
