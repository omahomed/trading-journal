// Sizing-mode mapping shared by Position Sizer, Log Buy, and New Entry.
//
// Risk-per-trade is derived from the V11 MCT state. The auto mapping is
// three tiers; Max is a fourth MANUAL-ONLY tier for conviction upshifts.
//
//   MCT State                →  Sizing Mode  →  Risk Per Trade
//   ────────────────────────────────────────────────────────────
//   POWERTREND               →  Offense      →  0.75%
//   UPTREND                  →  Normal       →  0.50%
//   UPTREND UNDER PRESSURE   →  Pilot        →  0.25%
//   RALLY MODE               →  Pilot        →  0.25%
//   CORRECTION               →  Pilot        →  0.25%
//   (no state maps here)     →  Max          →  1.00%  ← manual only
//
// Rationale for the retier: the old ladder (Defense 0.50 / Normal 0.75 /
// Offense 1.00, plus a manual-only Pilot 0.25) was calibrated to a
// hard-stop denominator; the New Entry model sizes against the trader's
// TRAILING REALIZED AVERAGE LOSS (~4.5% on the 2026 sample), which is a
// tighter denominator and needed proportionally smaller risk units to
// land the same shares recommendation. The Defense tier and the 1.0%
// tier are retired; the three surviving tiers are Pilot / Normal /
// Offense, indexed 0 / 1 / 2. Unknown-state fallback lands on Pilot
// (safest) rather than a middle-tier auto-pick — the redesign is
// intentionally conservative and the manual override is DOWNWARD-ONLY,
// so a middle-tier default would contradict that philosophy on a
// transient engine hiccup.
//
// cap_at_100 does NOT enter this mapping. The 100% total-exposure
// ceiling is enforced separately by V11's exposure cap logic; per-trade
// sizing stays wherever the state map lands.

export type MctState = "POWERTREND" | "UPTREND" | "UPTREND UNDER PRESSURE" | "RALLY MODE" | "CORRECTION";

export type SizingModeKey = "pilot" | "normal" | "offense" | "max";

/** Canonical index for a sizing tier. 0/1/2 map to Pilot / Normal /
 *  Offense — the tiers the MCT-state auto-picker can land on. Index 3
 *  is Max (1.00%), a MANUAL-ONLY conviction upshift the auto-picker
 *  never returns. Position Sizer + Log Buy expose Max as a clickable
 *  radio (their manual override is bidirectional). The New Entry
 *  downward-only clamp naturally excludes Max — see
 *  clampManualToDownwardOnly below. */
export type SizingModeIndex = 0 | 1 | 2 | 3;

/** Subset of SizingModeIndex the MCT-state auto-picker can return.
 *  Max (3) is manual-only and never auto-selected. */
export type AutoSizingModeIndex = 0 | 1 | 2;

export interface SizingMode {
  key: SizingModeKey;
  label: string;
  /** Risk per trade as a percentage of equity. */
  pct: number;
  icon: string;
  /** Canonical lookup index — position-sizer / log-buy / new-entry
      state machines key off this number. Array order equals aggression
      order (Pilot=0 → Normal=1 → Offense=2 → Max=3), so SIZING_MODES
      itself is also the UI-display order left-to-right. */
  index: SizingModeIndex;
}

export const SIZING_MODES: readonly SizingMode[] = [
  { key: "pilot",   label: "Pilot (0.25%)",   pct: 0.25, icon: "✈️", index: 0 },
  { key: "normal",  label: "Normal (0.50%)",  pct: 0.50, icon: "⚖️", index: 1 },
  { key: "offense", label: "Offense (0.75%)", pct: 0.75, icon: "⚔️", index: 2 },
  { key: "max",     label: "Max (1.00%)",     pct: 1.00, icon: "🔥", index: 3 },
] as const;

/** UI-display order (left → right, most conservative → most aggressive):
 *  Pilot · Normal · Offense · Max. Identical to SIZING_MODES since the
 *  array is already in aggression order — kept as a named export so
 *  callsites that render radios from SIZING_MODES_DISPLAY don't have to
 *  change. Max is a manual-only conviction upshift (2026-07-18); the
 *  MCT-state auto-picker only lands on Pilot / Normal / Offense. */
export const SIZING_MODES_DISPLAY: readonly SizingMode[] = SIZING_MODES;

/** Default fallback when MCT state is unknown / endpoint failed.
 *  Intentional: Pilot (0.25%). The redesign lands the auto path on
 *  Pilot for RALLY MODE / UPTREND UNDER PRESSURE / CORRECTION anyway,
 *  and the manual override is downward-only — so a middle-tier default
 *  on an engine hiccup would violate the "when in doubt, be smaller"
 *  invariant this model is built around. */
const DEFAULT_INDEX = 0;

/** Map an MCT state string to the corresponding sizing-mode index.
 *  Unknown states (including null/undefined/legacy strings) fall back
 *  to Pilot rather than guessing a middle tier. Never returns Max (3)
 *  — that tier is a manual-only conviction upshift. */
export function mctStateToSizingMode(state: string | null | undefined): AutoSizingModeIndex {
  switch (state) {
    case "POWERTREND":
      return 2;  // Offense (0.75%)
    case "UPTREND":
      return 1;  // Normal  (0.50%)
    case "UPTREND UNDER PRESSURE":
    case "RALLY MODE":
    case "CORRECTION":
      return 0;  // Pilot   (0.25%)
    default:
      return DEFAULT_INDEX;  // Unknown / null → Pilot (safe floor)
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
 *    50 SMA Violation             → Pilot   (0)   — most conservative
 *    21 EMA Confirmed Break       → Normal  (1)
 *    21 EMA Violation             → Normal  (1)
 *    (anything else, incl Watch)  → no floor (returns 2)
 *
 *  Most-severe wins when multiple alerts are active: a 50 SMA
 *  Violation alongside a 21 EMA Confirmed Break floors at Pilot,
 *  not Normal. (Old ladder called the strictest floor "Defense"; that
 *  tier is retired — Pilot now occupies index 0 and takes the same
 *  semantic role.) */
export function exitLadderFloor(activeExits: readonly ExitAlert[] | null | undefined): {
  idx: AutoSizingModeIndex;
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
 *  user ("Auto: Normal — from M Factor UPTREND, downshifted by
 *  21 EMA Confirmed Break"). */
export function deriveAutoSizingMode(
  state: string | null | undefined,
  activeExits: readonly ExitAlert[] | null | undefined,
): {
  idx: AutoSizingModeIndex;
  source: { stateIdx: AutoSizingModeIndex; floor: { idx: AutoSizingModeIndex; reason: string | null } };
} {
  const stateIdx = mctStateToSizingMode(state);
  const floor = exitLadderFloor(activeExits);
  const idx = Math.min(stateIdx, floor.idx) as AutoSizingModeIndex;
  return { idx, source: { stateIdx, floor } };
}

/** Human-readable label for the source of a sizing-mode pick. Used by
 *  Position Sizer / Log Buy / New Entry to render the "Auto: Offense
 *  (from M Factor POWERTREND)" / "Manual: Pilot" indicator. Function
 *  name stays as describeMctSource — the engine internals are still
 *  called MCT — but the user-visible string says "M Factor".
 *
 *  Optional `floor` arg: when the exit-ladder downshifted the mode
 *  below what the state alone would have picked, the label says so
 *  ("from M Factor UPTREND, downshifted by 21 EMA Confirmed Break")
 *  so the user understands WHY the auto-mode isn't what the state
 *  pill suggests. */
export function describeMctSource(
  state: string | null | undefined,
  floor?: { idx: AutoSizingModeIndex; reason: string | null } | null,
): string {
  const base = (() => {
    switch (state) {
      case "POWERTREND":
      case "UPTREND":
      case "UPTREND UNDER PRESSURE":
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

/** Downward-only manual override guard (New Entry semantics).
 *
 *  The New Entry page allows manual mode override but ONLY downward —
 *  the user may pick a tier smaller than the auto-selected one, never
 *  larger. This function returns the effective index given the
 *  auto-derived tier and a user-picked tier: whichever is more
 *  conservative wins. "Reset to auto" (in the calling component) simply
 *  discards the user pick and re-applies the auto value.
 *
 *  Position Sizer + Log Buy have their own longstanding upward-and-
 *  downward manual override — this helper is not used there, only in
 *  New Entry. */
export function clampManualToDownwardOnly(
  autoIdx: AutoSizingModeIndex,
  userIdx: SizingModeIndex,
): AutoSizingModeIndex {
  // Max (3) is a manual-only conviction upshift. Under the downward-
  // only rule (New Entry), clicking Max silently clamps back down to
  // the auto ceiling — the New Entry page doesn't expose an upshift
  // path. Position Sizer + Log Buy don't call this helper; their
  // manual override is bidirectional and Max is fully selectable there.
  return Math.min(autoIdx, userIdx) as AutoSizingModeIndex;
}
