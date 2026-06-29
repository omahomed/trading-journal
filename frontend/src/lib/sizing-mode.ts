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

export type SizingModeKey = "defense" | "normal" | "offense" | "pilot";

export interface SizingMode {
  key: SizingModeKey;
  label: string;
  /** Risk per trade as a percentage of equity. */
  pct: number;
  icon: string;
  /** Fixed lookup index — the position-sizer + log-buy state machines
      key off this number; reordering would silently shift behavior.
      Note: array index is NOT aggression order. Pilot (3) is the most
      conservative tier but lives at the end of the array so the original
      0/1/2 indices stay pinned to defense/normal/offense — keeps the
      "auto modes are 0|1|2" invariant intact. Use SIZING_MODES_DISPLAY
      below for left-to-right UI rendering by aggression. */
  index: 0 | 1 | 2 | 3;
}

export const SIZING_MODES: readonly SizingMode[] = [
  { key: "defense", label: "Defense (0.50%)", pct: 0.5,  icon: "🛡️", index: 0 },
  { key: "normal",  label: "Normal (0.75%)",  pct: 0.75, icon: "⚖️", index: 1 },
  { key: "offense", label: "Offense (1.00%)", pct: 1.0,  icon: "⚔️", index: 2 },
  // Manual-only tier. M Factor state mapping + exit-ladder floor will
  // never return index 3, so the auto path can't land you on Pilot —
  // it's strictly opt-in for "I want to be extra careful for reasons
  // the engine doesn't see" (earnings cluster, vacation, personal
  // cash needs). "Reset to auto" from Pilot drops to whatever rules
  // pick, never back to Pilot.
  { key: "pilot",   label: "Pilot (0.25%)",   pct: 0.25, icon: "✈️", index: 3 },
] as const;

/** UI-display order (left → right, most conservative → most aggressive):
 *  Pilot · Defense · Normal · Offense. Position Sizer + Log Buy both
 *  render radios in this order so the visual flow matches the
 *  aggression spectrum, while SIZING_MODES (canonical lookup) keeps
 *  the original 0/1/2 indices stable for backward compatibility. */
export const SIZING_MODES_DISPLAY: readonly SizingMode[] = [
  SIZING_MODES[3], // Pilot
  SIZING_MODES[0], // Defense
  SIZING_MODES[1], // Normal
  SIZING_MODES[2], // Offense
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

/** Minimal shape we need from the rally-prefix active_exits array.
 *  Matches the adapter's emitted dicts; extra fields are ignored. */
export interface ExitAlert {
  signal: string;
  severity?: string;
}

/** Exit-ladder ceiling on sizing mode. Logic mirrors the M Factor
 *  Exit Alerts taxonomy: a fired structural break should constrain
 *  NEW-trade sizing even when the engine state itself hasn't flipped
 *  to CORRECTION yet. Floor only — never lifts the mode above what
 *  the M Factor state would pick on its own. Watch states (no actual
 *  break confirmed) do NOT downshift; they're informational.
 *
 *    50 SMA Violation             → Defense (0)
 *    21 EMA Confirmed Break       → Normal  (1)
 *    21 EMA Violation             → Normal  (1)
 *    (anything else, incl Watch)  → no floor (returns 2)
 *
 *  Most-severe wins when multiple alerts are active: a 50 SMA
 *  Violation alongside a 21 EMA Confirmed Break floors at Defense,
 *  not Normal. */
export function exitLadderFloor(activeExits: readonly ExitAlert[] | null | undefined): {
  idx: 0 | 1 | 2;
  reason: string | null;
} {
  if (!activeExits || activeExits.length === 0) return { idx: 2, reason: null };
  for (const a of activeExits) {
    if (a?.signal === "50 SMA Violation") return { idx: 0, reason: "50 SMA Violation" };
  }
  for (const a of activeExits) {
    if (a?.signal === "21 EMA Confirmed Break" || a?.signal === "21 EMA Violation") {
      return { idx: 1, reason: a.signal };
    }
  }
  return { idx: 2, reason: null };
}

/** Combined auto-mode derivation: take the MORE conservative of
 *  (M Factor state → mode, exit-ladder floor). Returns the chosen
 *  index plus a breakdown the UI can use to explain the pick to the
 *  user ("Auto: Normal — from M Factor POWERTREND, downshifted by
 *  21 EMA Confirmed Break"). */
export function deriveAutoSizingMode(
  state: string | null | undefined,
  activeExits: readonly ExitAlert[] | null | undefined,
): {
  idx: 0 | 1 | 2;
  source: { stateIdx: 0 | 1 | 2; floor: { idx: 0 | 1 | 2; reason: string | null } };
} {
  const stateIdx = mctStateToSizingMode(state);
  const floor = exitLadderFloor(activeExits);
  const idx = Math.min(stateIdx, floor.idx) as 0 | 1 | 2;
  return { idx, source: { stateIdx, floor } };
}

/** Human-readable label for the source of a sizing-mode pick. Used by
 *  Position Sizer's "Auto: Offense (from M Factor POWERTREND)" /
 *  "Manual: Defense" indicator. The function NAME stays as
 *  describeMctSource — the engine internals are still called MCT —
 *  but the user-visible string says "M Factor".
 *
 *  Optional `floor` arg: when the exit-ladder downshifted the mode
 *  below what the state alone would have picked, the label says so
 *  ("from M Factor POWERTREND, downshifted by 21 EMA Confirmed Break")
 *  so the user understands WHY the auto-mode isn't what the state
 *  pill suggests. */
export function describeMctSource(
  state: string | null | undefined,
  floor?: { idx: 0 | 1 | 2; reason: string | null } | null,
): string {
  const base = (() => {
    switch (state) {
      case "POWERTREND":
      case "UPTREND":
      case "RALLY MODE":
      case "CORRECTION":
        return `from M Factor ${state}`;
      default:
        return "M Factor state unknown";
    }
  })();
  if (floor?.reason) {
    const stateIdx = mctStateToSizingMode(state);
    if (floor.idx < stateIdx) {
      return `${base}, downshifted by ${floor.reason}`;
    }
  }
  return base;
}
