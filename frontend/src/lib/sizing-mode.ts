// Sizing-mode mapping shared by Position Sizer + Log Buy.
//
// Risk-per-trade is derived from the V11 MCT state (replaces the legacy
// /api/market/mfactor MA-stack heuristic). The mapping below is fixed:
//
//   MCT State       →  Sizing Mode    →  Risk Per Trade
//   ────────────────────────────────────────────────────
//   CORRECTION      →  Defense        →  0.5%
//   RALLY MODE      →  Normal         →  0.75%
//   UPTREND         →  Offense        →  1.0%
//   POWERTREND      →  Offense        →  1.0%
//
// cap_at_100 does NOT enter this mapping. The 100% total-exposure ceiling
// is enforced separately by V11's exposure cap logic; per-trade sizing
// stays Offense even when the portfolio is capped.

export type MctState = "POWERTREND" | "UPTREND" | "RALLY MODE" | "CORRECTION";

export type SizingModeKey = "defense" | "normal" | "offense";

export interface SizingMode {
  key: SizingModeKey;
  label: string;
  /** Risk per trade as a percentage of equity. */
  pct: number;
  icon: string;
  /** Fixed display index — the position-sizer + log-buy state machines
      key off this number; reordering would silently shift behavior. */
  index: 0 | 1 | 2;
}

export const SIZING_MODES: readonly SizingMode[] = [
  { key: "defense", label: "Defense (0.50%)", pct: 0.5, icon: "🛡️", index: 0 },
  { key: "normal",  label: "Normal (0.75%)",  pct: 0.75, icon: "⚖️", index: 1 },
  { key: "offense", label: "Offense (1.00%)", pct: 1.0,  icon: "⚔️", index: 2 },
] as const;

/** Default fallback when MCT state is unknown / endpoint failed.
 *  Intentional: Normal (0.75%) is the safe middle ground — the user
 *  shouldn't be auto-promoted to Offense on a missing read, and Defense
 *  could over-restrict on a transient hiccup. */
const DEFAULT_INDEX = 1;

/** Map an MCT state string to the corresponding sizing-mode index.
 *  Unknown states (including null/undefined/legacy strings) fall back
 *  to the safe-middle default rather than guessing. */
export function mctStateToSizingMode(state: string | null | undefined): 0 | 1 | 2 {
  switch (state) {
    case "POWERTREND":
    case "UPTREND":
      return 2;  // Offense
    case "RALLY MODE":
      return 1;  // Normal
    case "CORRECTION":
      return 0;  // Defense
    default:
      return DEFAULT_INDEX;  // Unknown / null → Normal (safe middle)
  }
}

/** Human-readable label for the source of a sizing-mode pick. Used by
 *  Position Sizer's "Auto: Offense (from MCT POWERTREND)" / "Manual:
 *  Defense" indicator. */
export function describeMctSource(state: string | null | undefined): string {
  switch (state) {
    case "POWERTREND":
    case "UPTREND":
    case "RALLY MODE":
    case "CORRECTION":
      return `from MCT ${state}`;
    default:
      return "MCT state unknown";
  }
}
