"""V11 Market Cycle Tracker engine.

Stateless replay engine: takes a DataFrame of OHLC + indicators and produces
the bar-by-bar regime/exposure log plus a list of signal events. No database
access; no network. The signal vocabulary is sourced from
api.market_signals_vocab.

Per-bar processing order (must not be reordered without verifying canonical
signals — see Phase 2 spec for rationale):
  1. Ratchet reference high (only when both correction flags are False)
  2. Correction nullification (close > reference_high while correction_active)
  3. Correction declaration (close ≤ 7% threshold while correction_active=False)
  4. Anchored Violation detection (21 EMA, then 50 SMA) — fires regardless
     of in_correction; cap_at_100 set if correction_active=True; exposure cut
     deferred to V10 cascade resolution in phase 9.
  5. Streak/anchor maintenance (sets anchor on transition into below-MA streak,
     clears on close back above MA)
  6. Character Break / Confirmed Break / Trending Regime Break detection
     (gated on outside-in_correction; CB is single-bar, others on streak length)
  7. Rally hunt (Steps 0–4) when in_correction = True
  8. Post-Step-4 logic (V10 cascade-and-reset, Recovery, Steps 5/6/7, PT-OFF,
     Step 8 / PT-ON) when in_correction = False
  9. Derive state, build bar snapshot

The "two-flag correction model" is critical:
  - in_correction:        True only during rally-hunt phase (Steps 0–4 running);
                          flips False on STEP_4_LOW_ABOVE_21EMA_3BARS firing.
  - correction_active:    True from CORRECTION_DECLARED to CORRECTION_NULLIFIED.
  - cap_at_100 is set when a Violation fires inside correction_active=True
    regardless of in_correction; cleared on Rally Invalidation, macro
    nullification, or fresh STEP_0 after invalidation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd


# ============================================================================
# Constants
# ============================================================================

CORRECTION_DRAWDOWN = 0.07          # 7% close-below-reference triggers declaration
UNDERCUT = 0.01                     # ≥1% undercut of anchor low fires Violation
FTD_PCT_THRESHOLD = 0.01            # close % gain threshold to confirm an FTD
FTD_WINDOW_START = 4                # earliest rally_count where FTD eligible
FTD_WINDOW_END = 25                 # latest rally_count where FTD eligible

CONFIRMED_BREAK_BARS_BELOW_21 = 2   # consec closes below 21 EMA → Confirmed Break
TRB_BARS_BELOW_21 = 3               # consec closes below 21 EMA → Trending Regime Break
RECOVERY_BARS_LOW_ABOVE_21 = 3      # consec bars low > 21 EMA → Recovery

PT_ON_21_ABOVE_50_BARS = 5          # 21 EMA > 50 SMA for last N bars
PT_ON_LOW_ABOVE_21_BARS = 10        # low > 21 EMA for last N bars

RUNNING_MIN_LOOKBACK = 30           # while in_correction, track running min within this window

# Pink rally day threshold — close <= prev close but position-in-range > 0.5
# (close in upper half of bar's intraday range) qualifies as a rally day.
PINK_RALLY_DAY_POS_IN_RANGE = 0.5

# Step ladder exposure targets (for Steps 0–4 in rally-hunt phase)
STEP_LADDER_EXPOSURE = {0: 20, 1: 40, 2: 60, 3: 80, 4: 100}

# Exposure floor / ceilings for individual signals when V10 cascade doesn't apply
EXPOSURE_ON_CORRECTION_DECLARED = 0
EXPOSURE_ON_CHARACTER_BREAK = 50
EXPOSURE_ON_VIOLATION_21_NO_CASCADE = 50
EXPOSURE_ON_VIOLATION_50_NO_CASCADE = 0
EXPOSURE_ON_CONFIRMED_BREAK_NO_CASCADE = 30
EXPOSURE_ON_TRB_NO_CASCADE = 0
EXPOSURE_RECOVERY_TARGET = 100
EXPOSURE_V10_SOFT_RESET = 20
EXPOSURE_V10_FULL_INVALIDATION = 0
EXPOSURE_STEP_8 = 200

STATES = ("CORRECTION", "RALLY MODE", "UPTREND", "POWERTREND")

CASCADE_SIGNAL_TYPES = frozenset({
    "VIOLATION_21EMA",
    "CONFIRMED_BREAK_21EMA",
    "VIOLATION_50SMA",
    "TRENDING_REGIME_BREAK",
})


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class EngineConfig:
    initial_reference_high: Optional[float] = None
    initial_state: str = "POWERTREND"
    initial_exposure: int = 200
    initial_power_trend: bool = True
    correction_ever_declared: bool = True
    # If True, the reference-high ratchet runs from bar 0 instead of waiting
    # for the first nullification. Use this for full-history replays from
    # 2010 (where the seed is an ancient bar's high — without ratchet, the
    # 7% threshold stays anchored to that ancient value and no correction
    # ever declares). Leave False for stress-test configs that pass a
    # contemporaneous initial_reference_high (e.g., the Phase 2 canonical run
    # with seed 20,118.61).
    initial_ratchet_armed: bool = False


@dataclass
class SignalEvent:
    trade_date: date
    signal_type: str
    signal_label: str
    exposure_before: int
    exposure_after: int
    state_before: str
    state_after: str
    meta: dict = field(default_factory=dict)


@dataclass
class EngineResult:
    bars: pd.DataFrame
    signals: list[SignalEvent]
    final_state: dict


# ============================================================================
# Engine
# ============================================================================

class MCTEngine:
    """V11 Market Cycle Tracker engine. Pure-Python, stateless across runs."""

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()

    def run(self, history: pd.DataFrame) -> EngineResult:
        if history.empty:
            return EngineResult(bars=pd.DataFrame(), signals=[], final_state={})

        history = history.sort_values("trade_date").reset_index(drop=True)
        state = self._init_state()
        signals: list[SignalEvent] = []
        bar_records = []

        for i in range(len(history)):
            current = history.iloc[i]
            prev = history.iloc[i - 1] if i > 0 else None
            bar_signals = self._process_bar(i, current, prev, history, state)
            signals.extend(bar_signals)
            bar_records.append(self._bar_record(current, state, bar_signals))

        return EngineResult(
            bars=pd.DataFrame(bar_records),
            signals=signals,
            final_state=dict(state),
        )

    # ------------------------------------------------------------------------
    # State initialization
    # ------------------------------------------------------------------------

    def _init_state(self) -> dict:
        return {
            "exposure": self.config.initial_exposure,
            "power_trend": self.config.initial_power_trend,
            "reference_high": self.config.initial_reference_high,
            "correction_ever_declared": self.config.correction_ever_declared,

            # Ratchet gate: by default the seeded reference_high stays fixed
            # until the engine observes a complete correction cycle in this run
            # (declaration → nullification). After first nullification this
            # flips True and the ratchet starts updating reference_high on new
            # highs. For full-history replays the caller arms the ratchet up
            # front via EngineConfig.initial_ratchet_armed=True so the reference
            # climbs naturally from the 2010 seed.
            "ratchet_armed": self.config.initial_ratchet_armed,

            "in_correction": False,
            "correction_active": False,

            # Rally / FTD tracking
            "rally_active": False,
            "rally_day_idx": None,
            "rally_day_low": None,
            "running_min_low": None,
            "running_min_idx": None,
            "rally_count": 0,
            "ftd_close": None,
            "ftd_low": None,

            # Step ladder flags
            "step0_done": False,
            "step1_done": False,
            "step2_done": False,
            "step3_done": False,
            "step4_done": False,
            "step5_done": False,
            "step6_done": False,
            "step7_done": False,

            "cap_at_100": False,

            # Anchored Violation state
            "anchor_21_low": None,
            "anchor_50_low": None,
            "violation_21_fired": False,
            "violation_50_fired": False,

            # Streaks (running counts of consecutive bars meeting condition)
            "consec_below_21": 0,        # close < 21 EMA streak
            "consec_below_50": 0,        # close < 50 SMA streak
            "consec_low_above_21": 0,    # low > 21 EMA streak (Recovery / Step 4)
            "consec_low_above_50": 0,    # low > 50 SMA streak (Step 5)
            "consec_21_above_50": 0,     # 21 EMA > 50 SMA streak (Step 8 condition)

            # Single-fire flags (reset when their underlying streak ends)
            "character_break_fired": False,
            "confirmed_break_fired": False,
            "regime_break_fired": False,
        }

    # ------------------------------------------------------------------------
    # Per-bar pipeline
    # ------------------------------------------------------------------------

    def _process_bar(self, i, current, prev, history, state) -> list[SignalEvent]:
        bar_signals: list[SignalEvent] = []

        # Snapshot start-of-bar flags so step prerequisites only see prior-bar state
        # for steps that require persistence (Step 3 requires step2_done from prior
        # bar, etc.). Steps 0/1/2 use live flags (can chain within a bar).
        start_flags = {
            "step2_done": state["step2_done"],
            "step3_done": state["step3_done"],
            "step4_done": state["step4_done"],
            "step5_done": state["step5_done"],
            "step6_done": state["step6_done"],
        }

        # Phase 1: ratchet reference high
        self._phase_ratchet(current, state)

        # Phase 2: correction nullification
        self._phase_nullification(current, state, bar_signals)

        # Phase 3: correction declaration
        self._phase_declaration(current, state, bar_signals)

        # Phase 4: anchored Violation detection (uses anchor from PRIOR bar setup)
        # Cap_at_100 is set inside this phase. Exposure cut is deferred — the
        # final exposure is determined by V10 cascade resolution in phase 8/9
        # if applicable, or by individual cuts in phase 6 if no cascade applies.
        self._phase_violations(current, state, bar_signals)

        # Phase 5: streak & anchor maintenance for THIS bar
        # (sets anchor on transition into below-MA streak so subsequent bars
        # in the streak can detect Violation against it)
        self._phase_update_streaks(current, prev, state)

        # Phase 6: outside-correction exits — Character Break / Confirmed Break /
        # Trending Regime Break. CB is single-bar (transition); the others fire
        # at streak length thresholds. All only fire when not in_correction.
        self._phase_exits(current, prev, state, bar_signals)

        # Phase 7: rally hunt (Steps 0–4) when in_correction is True
        if state["in_correction"]:
            self._phase_rally_hunt(i, current, prev, history, state, bar_signals,
                                   start_flags)

        # Step 4 firing inside phase 7 may have flipped in_correction off; if so,
        # the rest of this bar's processing flows through phase 8 below.

        # Phase 8: post-Step-4 zone — V10 cascade resolution, Recovery,
        # Steps 5/6/7, Step 8 / PT-ON. in_correction may have flipped
        # during phase 7 (Step 4 fired) so re-check.
        if not state["in_correction"]:
            self._phase_post_step4(i, current, prev, history, state, bar_signals,
                                   start_flags)

        # Phase 9: PT-OFF always runs. Power-Trend can flip off during a rally
        # hunt phase (canonical 2/5/2026 fires PT-OFF the day after a V10 soft
        # reset, when 21 EMA dips below 50 SMA on a down close).
        self._phase_pt_off(current, prev, state, bar_signals)

        return bar_signals

    # ------------------------------------------------------------------------
    # Phase 1 — reference-high ratchet
    # ------------------------------------------------------------------------

    def _phase_ratchet(self, current, state):
        if state["in_correction"] or state["correction_active"]:
            return
        if not state["ratchet_armed"]:
            # Pre-first-nullification: respect the seeded reference_high.
            return
        h = float(current["high"])
        if state["reference_high"] is None or h > state["reference_high"]:
            state["reference_high"] = h

    # ------------------------------------------------------------------------
    # Phase 2 — correction nullification
    # ------------------------------------------------------------------------

    def _phase_nullification(self, current, state, bar_signals):
        if not state["correction_active"]:
            return
        if state["reference_high"] is None:
            return
        close = float(current["close"])
        if close > state["reference_high"]:
            before = state["exposure"]
            state["correction_active"] = False
            # Nullification also forces in_correction=False so the (in_correction=True,
            # correction_active=False) "illegal" state can never persist. Edge case:
            # nullification fires before STEP_4 (rally hunt didn't fully resolve).
            # Without this, the engine wedges in rally-hunt state forever and the
            # ratchet stays disabled (gated on both flags being False).
            state["in_correction"] = False
            state["ratchet_armed"] = True  # ratchet starts after first nullification
            had_cap = state["cap_at_100"]
            # Steps 5/6/7 retire so a future correction can re-fire them
            state["step5_done"] = False
            state["step6_done"] = False
            state["step7_done"] = False
            # Allow the next correction to be declared
            # (declaration gate is `not correction_active`, so this is automatic)

            bar_signals.append(self._signal(
                current, state, "CORRECTION_NULLIFIED",
                f"Close {close:.2f} > reference high {state['reference_high']:.2f}",
                exposure_before=before, exposure_after=state["exposure"],
                meta={"reference_high": state["reference_high"], "close": close},
            ))

            if had_cap:
                state["cap_at_100"] = False
                bar_signals.append(self._signal(
                    current, state, "CAP_AT_100_RELEASED",
                    "cap-at-100 released on correction nullification",
                    exposure_before=state["exposure"],
                    exposure_after=state["exposure"],
                    meta={},
                ))

    # ------------------------------------------------------------------------
    # Phase 3 — correction declaration
    # ------------------------------------------------------------------------

    def _phase_declaration(self, current, state, bar_signals):
        # Gate only on correction_active (not in_correction): a V10 soft reset
        # leaves the engine in_correction=True but correction_active=False, and
        # canonical 11/20/2025 fires a fresh declaration during exactly that
        # state. Declaration resets all rally state cleanly.
        if state["correction_active"]:
            return
        if state["reference_high"] is None:
            return
        threshold = state["reference_high"] * (1.0 - CORRECTION_DRAWDOWN)
        close = float(current["close"])
        if close > threshold:
            return

        before = state["exposure"]
        state["in_correction"] = True
        state["correction_active"] = True
        state["correction_ever_declared"] = True

        # Reset rally-hunt ladder
        state["rally_active"] = False
        state["rally_day_idx"] = None
        state["rally_day_low"] = None
        state["running_min_low"] = None
        state["running_min_idx"] = None
        state["rally_count"] = 0
        state["ftd_close"] = None
        state["ftd_low"] = None
        for s in range(8):
            state[f"step{s}_done"] = False
        # cap_at_100 is NOT auto-cleared on declaration (preserved across cycles
        # only via explicit reset rules — declaration of a new correction simply
        # opens a new correction_active window where Violations can re-set cap)

        # Exposure drops to 0% on macro correction declaration
        state["exposure"] = EXPOSURE_ON_CORRECTION_DECLARED

        bar_signals.append(self._signal(
            current, state, "CORRECTION_DECLARED",
            f"Close {close:.2f} ≤ {threshold:.2f} (7% from {state['reference_high']:.2f})",
            exposure_before=before, exposure_after=state["exposure"],
            meta={"reference_high": state["reference_high"], "threshold": threshold,
                  "close": close},
        ))

    # ------------------------------------------------------------------------
    # Phase 4 — anchored Violation detection
    # ------------------------------------------------------------------------

    def _phase_violations(self, current, state, bar_signals):
        low = float(current["low"])

        # 21 EMA Violation
        if state["anchor_21_low"] is not None and not state["violation_21_fired"]:
            anchor = state["anchor_21_low"]
            undercut_pct = (anchor - low) / anchor if anchor > 0 else 0.0
            if undercut_pct >= UNDERCUT:
                before = state["exposure"]
                state["violation_21_fired"] = True
                bar_signals.append(self._signal(
                    current, state, "VIOLATION_21EMA",
                    f"Low {low:.2f} undercuts anchor {anchor:.2f} by {undercut_pct*100:.2f}%",
                    exposure_before=before, exposure_after=state["exposure"],
                    meta={"anchor_low": anchor, "low": low,
                          "undercut_pct": undercut_pct},
                ))
                # Cap-at-100 if inside correction_active (regardless of in_correction)
                if state["correction_active"] and not state["cap_at_100"]:
                    state["cap_at_100"] = True
                    bar_signals.append(self._signal(
                        current, state, "CAP_AT_100_ACTIVATED",
                        "cap-at-100 set by 21 EMA violation inside correction window",
                        exposure_before=state["exposure"],
                        exposure_after=state["exposure"],
                        meta={"trigger": "VIOLATION_21EMA"},
                    ))

        # 50 SMA Violation (parallel structure)
        if state["anchor_50_low"] is not None and not state["violation_50_fired"]:
            anchor = state["anchor_50_low"]
            undercut_pct = (anchor - low) / anchor if anchor > 0 else 0.0
            if undercut_pct >= UNDERCUT:
                before = state["exposure"]
                state["violation_50_fired"] = True
                bar_signals.append(self._signal(
                    current, state, "VIOLATION_50SMA",
                    f"Low {low:.2f} undercuts 50-SMA anchor {anchor:.2f} by {undercut_pct*100:.2f}%",
                    exposure_before=before, exposure_after=state["exposure"],
                    meta={"anchor_low": anchor, "low": low,
                          "undercut_pct": undercut_pct},
                ))
                if state["correction_active"] and not state["cap_at_100"]:
                    state["cap_at_100"] = True
                    bar_signals.append(self._signal(
                        current, state, "CAP_AT_100_ACTIVATED",
                        "cap-at-100 set by 50 SMA violation inside correction window",
                        exposure_before=state["exposure"],
                        exposure_after=state["exposure"],
                        meta={"trigger": "VIOLATION_50SMA"},
                    ))

    # ------------------------------------------------------------------------
    # Phase 5 — streak & anchor maintenance
    # ------------------------------------------------------------------------

    def _phase_update_streaks(self, current, prev, state):
        close = float(current["close"])
        low = float(current["low"])
        ema_21 = float(current["ema_21"]) if pd.notna(current["ema_21"]) else None
        sma_50 = float(current["sma_50"]) if pd.notna(current["sma_50"]) else None

        # 21 EMA close streak + anchor management
        if ema_21 is not None:
            if close < ema_21:
                # Transition into streak: set anchor to THIS bar's low
                if state["consec_below_21"] == 0:
                    state["anchor_21_low"] = low
                    state["violation_21_fired"] = False
                state["consec_below_21"] += 1
            else:
                # Streak ended (or never started)
                state["consec_below_21"] = 0
                state["anchor_21_low"] = None
                state["violation_21_fired"] = False
                state["character_break_fired"] = False
                state["confirmed_break_fired"] = False
                state["regime_break_fired"] = False

            # low-above-21 streak (Recovery / Step 3 / Step 4)
            if low > ema_21:
                state["consec_low_above_21"] += 1
            else:
                state["consec_low_above_21"] = 0

        # 50 SMA close streak + anchor
        if sma_50 is not None:
            if close < sma_50:
                if state["consec_below_50"] == 0:
                    state["anchor_50_low"] = low
                    state["violation_50_fired"] = False
                state["consec_below_50"] += 1
            else:
                state["consec_below_50"] = 0
                state["anchor_50_low"] = None
                state["violation_50_fired"] = False

            # low-above-50 streak (Step 5)
            if low > sma_50:
                state["consec_low_above_50"] += 1
            else:
                state["consec_low_above_50"] = 0

        # 21 EMA > 50 SMA streak (PT-ON condition)
        if ema_21 is not None and sma_50 is not None:
            if ema_21 > sma_50:
                state["consec_21_above_50"] += 1
            else:
                state["consec_21_above_50"] = 0

    # ------------------------------------------------------------------------
    # Phase 6 — outside-in_correction exits
    # ------------------------------------------------------------------------

    def _phase_exits(self, current, prev, state, bar_signals):
        if state["in_correction"]:
            return

        # Character Break: single-bar transition into below-21 streak
        # (consec_below_21 just became 1 this bar). Only first bar of the streak.
        if (state["consec_below_21"] == 1
                and not state["character_break_fired"]):
            before = state["exposure"]
            # Drop to 50 (or to cap_at_100 floor of 100, whichever is lower).
            target = EXPOSURE_ON_CHARACTER_BREAK
            state["exposure"] = min(state["exposure"], target)
            state["character_break_fired"] = True
            # Steps 5/6/7 retire on CB so they can re-fire after Recovery
            # restores the stack (canonical 1/12/2026 STEP_7 re-fire and
            # 2/2/2026 STEP_7 re-fire follow this pattern).
            state["step5_done"] = False
            state["step6_done"] = False
            state["step7_done"] = False
            bar_signals.append(self._signal(
                current, state, "CHARACTER_BREAK",
                f"close < 21 EMA on transition; exposure cut to {target}%",
                exposure_before=before, exposure_after=state["exposure"],
                meta={"close": float(current["close"]),
                      "ema_21": float(current["ema_21"])},
            ))

        # Confirmed Break: consec_below_21 reaches CONFIRMED_BREAK_BARS_BELOW_21
        # (cascade trigger; exposure cut deferred to V10 resolution if applicable)
        if (state["consec_below_21"] == CONFIRMED_BREAK_BARS_BELOW_21
                and not state["confirmed_break_fired"]):
            before = state["exposure"]
            state["confirmed_break_fired"] = True
            bar_signals.append(self._signal(
                current, state, "CONFIRMED_BREAK_21EMA",
                f"{CONFIRMED_BREAK_BARS_BELOW_21} consecutive closes below 21 EMA",
                exposure_before=before, exposure_after=state["exposure"],
                meta={"consec_below_21": state["consec_below_21"]},
            ))

        # Trending Regime Break: consec_below_21 reaches TRB_BARS_BELOW_21
        if (state["consec_below_21"] == TRB_BARS_BELOW_21
                and not state["regime_break_fired"]):
            before = state["exposure"]
            state["regime_break_fired"] = True
            bar_signals.append(self._signal(
                current, state, "TRENDING_REGIME_BREAK",
                f"{TRB_BARS_BELOW_21} consecutive closes below 21 EMA",
                exposure_before=before, exposure_after=state["exposure"],
                meta={"consec_below_21": state["consec_below_21"]},
            ))

    # ------------------------------------------------------------------------
    # Phase 7 — rally hunt (Steps 0–4)
    # ------------------------------------------------------------------------

    def _phase_rally_hunt(self, i, current, prev, history, state, bar_signals,
                           start_flags):
        low = float(current["low"])
        close = float(current["close"])
        prev_close = float(prev["close"]) if prev is not None else close
        ema_21 = float(current["ema_21"]) if pd.notna(current["ema_21"]) else None

        # Track running min low while in rally hunt; this becomes rally_day_idx
        # when STEP_0 fires. Reset on entering in_correction (handled in
        # phase_declaration which clears running_min_*).
        if state["running_min_low"] is None or low < state["running_min_low"]:
            state["running_min_low"] = low
            state["running_min_idx"] = i

        # Rally invalidation: low < rally_day_low (only meaningful if rally active)
        if state["rally_active"] and state["rally_day_low"] is not None:
            if low < state["rally_day_low"]:
                self._fire_rally_invalidation(i, current, state, bar_signals,
                                              reason=f"low {low:.2f} < rally_day_low {state['rally_day_low']:.2f}")
                # Same bar can fire a fresh STEP_0 if conditions met (handled below)

        # ----- Step 0: Rally Day -----
        # V11 single rule: STEP_0 fires when the bar is either
        #   (a) an up day (close > prev close), OR
        #   (b) a "pink rally day" — down/flat close but with close in the
        #       upper half of the bar's intraday range (close - low) /
        #       (high - low) > 0.5. Captures bars that opened weak, made a
        #       new low, and closed near the top of the range.
        # Continuation-down bars (close in lower half of range) don't qualify.
        bar_high = float(current["high"])
        denom = bar_high - low
        position_in_range = (close - low) / denom if denom > 0 else 0.5
        up_day = prev is not None and close > prev_close
        pink_rally_day = (prev is not None
                          and close <= prev_close
                          and position_in_range > PINK_RALLY_DAY_POS_IN_RANGE)

        if (not state["step0_done"]
                and state["running_min_idx"] is not None
                and (up_day or pink_rally_day)):
            state["step0_done"] = True
            state["rally_active"] = True
            state["rally_day_idx"] = state["running_min_idx"]
            state["rally_day_low"] = state["running_min_low"]
            state["rally_count"] = i - state["rally_day_idx"] + 1

            before = state["exposure"]
            state["exposure"] = max(state["exposure"], STEP_LADDER_EXPOSURE[0])
            bar_signals.append(self._signal(
                current, state, "STEP_0_RALLY_DAY",
                f"rally day; rally_day_low={state['rally_day_low']:.2f}, "
                f"rally_count={state['rally_count']}",
                exposure_before=before, exposure_after=state["exposure"],
                meta={"rally_day_low": state["rally_day_low"],
                      "rally_day_idx": int(state["rally_day_idx"]),
                      "rally_count": state["rally_count"],
                      "trigger": "up_day" if up_day else "pink_rally_day",
                      "position_in_range": position_in_range},
            ))

        # Increment rally_count for bars after STEP_0 already fired
        elif state["rally_active"] and state["rally_day_idx"] is not None:
            state["rally_count"] = i - state["rally_day_idx"] + 1

        # ----- Step 1: Follow-Through Day -----
        if (state["step0_done"] and not state["step1_done"]
                and state["rally_count"] is not None
                and FTD_WINDOW_START <= state["rally_count"] <= FTD_WINDOW_END
                and prev is not None and prev_close > 0):
            pct_gain = (close - prev_close) / prev_close
            if pct_gain >= FTD_PCT_THRESHOLD:
                state["step1_done"] = True
                state["ftd_close"] = close
                state["ftd_low"] = low
                before = state["exposure"]
                state["exposure"] = max(state["exposure"], STEP_LADDER_EXPOSURE[1])
                bar_signals.append(self._signal(
                    current, state, "STEP_1_FTD",
                    f"FTD on Day {state['rally_count']}, close +{pct_gain*100:.2f}%",
                    exposure_before=before, exposure_after=state["exposure"],
                    meta={"rally_count": state["rally_count"],
                          "pct_gain": pct_gain,
                          "ftd_close": close, "ftd_low": low},
                ))

        # ----- Step 2: close > 21 EMA (can fire same bar as Step 1) -----
        if (state["step1_done"] and not state["step2_done"]
                and ema_21 is not None and close > ema_21):
            state["step2_done"] = True
            before = state["exposure"]
            state["exposure"] = max(state["exposure"], STEP_LADDER_EXPOSURE[2])
            bar_signals.append(self._signal(
                current, state, "STEP_2_CLOSE_ABOVE_21EMA",
                f"close {close:.2f} > 21 EMA {ema_21:.2f}",
                exposure_before=before, exposure_after=state["exposure"],
                meta={"close": close, "ema_21": ema_21},
            ))

        # ----- Step 3: low > 21 EMA (gated on step2_done at START of bar) -----
        if (start_flags["step2_done"] and not state["step3_done"]
                and ema_21 is not None and low > ema_21):
            state["step3_done"] = True
            before = state["exposure"]
            state["exposure"] = max(state["exposure"], STEP_LADDER_EXPOSURE[3])
            bar_signals.append(self._signal(
                current, state, "STEP_3_LOW_ABOVE_21EMA",
                f"low {low:.2f} > 21 EMA {ema_21:.2f}",
                exposure_before=before, exposure_after=state["exposure"],
                meta={"low": low, "ema_21": ema_21},
            ))

        # ----- Step 4: 3 consecutive bars low > 21 EMA -----
        if (start_flags["step3_done"] and not state["step4_done"]
                and state["consec_low_above_21"] >= RECOVERY_BARS_LOW_ABOVE_21):
            state["step4_done"] = True
            before = state["exposure"]
            state["exposure"] = max(state["exposure"], STEP_LADDER_EXPOSURE[4])
            bar_signals.append(self._signal(
                current, state, "STEP_4_LOW_ABOVE_21EMA_3BARS",
                f"low > 21 EMA for {state['consec_low_above_21']} consecutive bars",
                exposure_before=before, exposure_after=state["exposure"],
                meta={"consec_low_above_21": state["consec_low_above_21"]},
            ))
            # Exit rally-hunt phase. Anchor management resumes / exits become live
            # NEXT bar (in_correction flag flips at end of this bar's rally hunt).
            state["in_correction"] = False

        # ----- Post-FTD soft fail: close < ftd_low -----
        # Asymmetric vs. Rally Day invalidation by design (V11):
        #   Rally Day invalidation: ANY intraday low < rally_day_low → reset
        #     (rally is fragile, not yet confirmed)
        #   FTD soft-fail: requires CLOSE < ftd_low (FTD is confirmed; needs
        #     closing damage, not just intraday undercut)
        if (state["step1_done"] and state["ftd_low"] is not None
                and close < state["ftd_low"]):
            self._fire_post_ftd_soft_fail(current, state, bar_signals)

    def _fire_rally_invalidation(self, i, current, state, bar_signals, *, reason: str):
        before = state["exposure"]
        had_cap = state["cap_at_100"]
        # Full reset: exposure 0, ladder cleared, rally cleared, cap cleared
        state["exposure"] = 0
        for s in range(8):
            state[f"step{s}_done"] = False
        state["rally_active"] = False
        state["rally_day_idx"] = None
        state["rally_day_low"] = None
        # Seed a fresh running min at THIS bar so a same-bar STEP_0 can re-fire
        # when close > prev_close (canonical 4/9/2025 fires both signals same bar)
        state["running_min_low"] = float(current["low"])
        state["running_min_idx"] = i
        state["rally_count"] = 0
        state["ftd_close"] = None
        state["ftd_low"] = None
        state["cap_at_100"] = False  # rally invalidation clears cap
        # Make sure we're back in rally-hunt mode
        state["in_correction"] = True

        bar_signals.append(self._signal(
            current, state, "RALLY_INVALIDATED",
            reason,
            exposure_before=before, exposure_after=state["exposure"],
            meta={"reason": reason},
        ))
        if had_cap:
            bar_signals.append(self._signal(
                current, state, "CAP_AT_100_RELEASED",
                "cap-at-100 cleared on rally invalidation",
                exposure_before=state["exposure"],
                exposure_after=state["exposure"],
                meta={},
            ))

    def _fire_post_ftd_soft_fail(self, current, state, bar_signals):
        before = state["exposure"]
        # Reset Step 1 (and steps above it) so we can hunt for a fresh FTD
        # on the same rally cycle. rally_count keeps incrementing.
        state["step1_done"] = False
        state["step2_done"] = False
        state["step3_done"] = False
        state["step4_done"] = False
        state["ftd_close"] = None
        state["ftd_low"] = None
        # Exposure cut back to STEP_0 level
        state["exposure"] = STEP_LADDER_EXPOSURE[0]

        bar_signals.append(self._signal(
            current, state, "POST_FTD_SOFT_FAIL",
            f"close {float(current['close']):.2f} < ftd_low; resetting Step 1",
            exposure_before=before, exposure_after=state["exposure"],
            meta={"close": float(current["close"])},
        ))

    # ------------------------------------------------------------------------
    # Phase 8 — post-Step-4 zone
    # ------------------------------------------------------------------------

    def _phase_post_step4(self, i, current, prev, history, state, bar_signals,
                           start_flags):
        # ----- V10 cascade resolution -----
        # If any cascade-trigger signal fired this bar AND a rally is in progress
        # (rally_day_low set), apply soft or full reset based on low vs rally_day_low.
        if state["rally_day_low"] is not None and any(
            s.signal_type in CASCADE_SIGNAL_TYPES for s in bar_signals
        ):
            low = float(current["low"])
            if low > state["rally_day_low"]:
                self._fire_v10_soft_reset(i, current, state, bar_signals)
            else:
                self._fire_v10_full_invalidation(current, state, bar_signals)
            # V10 fully resolves the bar — no Recovery / Steps 5/6/7 / PT-ON
            return

        # If a non-cascade individual exit fired (e.g., CHARACTER_BREAK alone
        # set exposure to 50), let it stand. If a cascade signal fired but no
        # rally context (rally_day_low None), apply individual cuts as fallback.
        for sig in list(bar_signals):
            if sig.signal_type == "VIOLATION_21EMA" and state["rally_day_low"] is None:
                state["exposure"] = min(state["exposure"], EXPOSURE_ON_VIOLATION_21_NO_CASCADE)
            elif sig.signal_type == "VIOLATION_50SMA" and state["rally_day_low"] is None:
                state["exposure"] = min(state["exposure"], EXPOSURE_ON_VIOLATION_50_NO_CASCADE)
            elif sig.signal_type == "CONFIRMED_BREAK_21EMA" and state["rally_day_low"] is None:
                state["exposure"] = min(state["exposure"], EXPOSURE_ON_CONFIRMED_BREAK_NO_CASCADE)
            elif sig.signal_type == "TRENDING_REGIME_BREAK" and state["rally_day_low"] is None:
                state["exposure"] = min(state["exposure"], EXPOSURE_ON_TRB_NO_CASCADE)

        # ----- Recovery (runs regardless of step4_done; can lift after CB) -----
        if (state["consec_low_above_21"] >= RECOVERY_BARS_LOW_ABOVE_21
                and state["exposure"] < EXPOSURE_RECOVERY_TARGET):
            before = state["exposure"]
            state["exposure"] = EXPOSURE_RECOVERY_TARGET
            bar_signals.append(self._signal(
                current, state, "RECOVERY",
                f"low > 21 EMA for {state['consec_low_above_21']} bars; restore to 100%",
                exposure_before=before, exposure_after=state["exposure"],
                meta={"consec_low_above_21": state["consec_low_above_21"]},
            ))

        ema_8 = float(current["ema_8"]) if pd.notna(current["ema_8"]) else None
        ema_21 = float(current["ema_21"]) if pd.notna(current["ema_21"]) else None
        sma_50 = float(current["sma_50"]) if pd.notna(current["sma_50"]) else None
        sma_200 = float(current["sma_200"]) if pd.notna(current["sma_200"]) else None

        # ----- Step 5: low > 50 SMA × 3 consecutive bars -----
        # Steps 5/6/7 only fire while correction_active is True AND step4 was
        # done at START of bar — so they wait for the bar AFTER STEP_4 fires.
        # Canonical: STEP_4 on 12/24/2025, STEP_6 on 12/26 (next trading day),
        # STEP_7 on 12/29 (when 8 EMA finally tops 21 EMA).
        # Guard: don't re-fire on the same bar as CHARACTER_BREAK; Steps 5/6/7
        # re-fire AFTER a recovery, not in lockstep with the break.
        cb_fired_this_bar = any(
            s.signal_type == "CHARACTER_BREAK" for s in bar_signals
        )

        if (not state["step5_done"]
                and start_flags["step4_done"]
                and not cb_fired_this_bar
                and state["consec_low_above_50"] >= 3
                and state["correction_active"]):
            state["step5_done"] = True
            informational = state["cap_at_100"]
            before = state["exposure"]
            if not informational:
                state["exposure"] = max(state["exposure"], 120)
            label_suffix = " [INFORMATIONAL — cap_at_100]" if informational else ""
            bar_signals.append(self._signal(
                current, state, "STEP_5_LOW_ABOVE_50SMA_3BARS",
                f"low > 50 SMA for {state['consec_low_above_50']} bars" + label_suffix,
                exposure_before=before, exposure_after=state["exposure"],
                meta={"informational": informational,
                      "consec_low_above_50": state["consec_low_above_50"]},
            ))

        # ----- Step 6: 21 EMA > 50 SMA > 200 SMA -----
        if (not state["step6_done"] and start_flags["step4_done"]
                and not cb_fired_this_bar
                and state["correction_active"]
                and ema_21 is not None and sma_50 is not None and sma_200 is not None
                and ema_21 > sma_50 > sma_200):
            state["step6_done"] = True
            informational = state["cap_at_100"]
            before = state["exposure"]
            if not informational:
                state["exposure"] = max(state["exposure"], 140)
            label_suffix = " [INFORMATIONAL — cap_at_100]" if informational else ""
            bar_signals.append(self._signal(
                current, state, "STEP_6_MA_STACK_SLOW",
                "21 EMA > 50 SMA > 200 SMA" + label_suffix,
                exposure_before=before, exposure_after=state["exposure"],
                meta={"informational": informational,
                      "ema_21": ema_21, "sma_50": sma_50, "sma_200": sma_200},
            ))

        # ----- Step 7: 8 EMA > 21 EMA > 50 SMA > 200 SMA (full stack) -----
        # Gates on step6_done at START of bar — Step 7 is the strict version of
        # Step 6, so it can't fire same bar as Step 6. Canonical: Step 6 on
        # 12/26/2025, Step 7 on 12/29/2025 (next trading day).
        if (not state["step7_done"] and start_flags["step6_done"]
                and not cb_fired_this_bar
                and state["correction_active"]
                and ema_8 is not None and ema_21 is not None
                and sma_50 is not None and sma_200 is not None
                and ema_8 > ema_21 > sma_50 > sma_200):
            state["step7_done"] = True
            informational = state["cap_at_100"]
            before = state["exposure"]
            if not informational:
                state["exposure"] = max(state["exposure"], 160)
            label_suffix = " [INFORMATIONAL — cap_at_100]" if informational else ""
            bar_signals.append(self._signal(
                current, state, "STEP_7_MA_STACK_FULL",
                "8 EMA > 21 EMA > 50 SMA > 200 SMA" + label_suffix,
                exposure_before=before, exposure_after=state["exposure"],
                meta={"informational": informational,
                      "ema_8": ema_8, "ema_21": ema_21,
                      "sma_50": sma_50, "sma_200": sma_200},
            ))

        # ----- Step 8 / Power-Trend ON -----
        # Gated on: step4_done, not currently power_trend, no cap_at_100,
        # and the four canonical PT-ON conditions.
        if (state["step4_done"] and not state["power_trend"]
                and not state["cap_at_100"]
                and prev is not None
                and ema_21 is not None and sma_50 is not None
                and pd.notna(prev["sma_50"])
                and state["consec_21_above_50"] >= PT_ON_21_ABOVE_50_BARS
                and state["consec_low_above_21"] >= PT_ON_LOW_ABOVE_21_BARS
                and float(current["close"]) >= float(prev["close"])
                and sma_50 > float(prev["sma_50"])):
            state["power_trend"] = True
            before = state["exposure"]
            state["exposure"] = EXPOSURE_STEP_8
            bar_signals.append(self._signal(
                current, state, "STEP_8_POWERTREND_ON",
                "all 4 PT-ON conditions met; exposure → 200%",
                exposure_before=before, exposure_after=state["exposure"],
                meta={"consec_21_above_50": state["consec_21_above_50"],
                      "consec_low_above_21": state["consec_low_above_21"]},
            ))

    # ------------------------------------------------------------------------
    # Phase 9 — Power-Trend OFF (always runs)
    # ------------------------------------------------------------------------

    def _phase_pt_off(self, current, prev, state, bar_signals):
        if not state["power_trend"]:
            return
        if prev is None:
            return
        if pd.isna(current["ema_21"]) or pd.isna(current["sma_50"]):
            return
        if pd.isna(prev["ema_21"]) or pd.isna(prev["sma_50"]):
            return

        prev_ema_21 = float(prev["ema_21"])
        prev_sma_50 = float(prev["sma_50"])
        ema_21 = float(current["ema_21"])
        sma_50 = float(current["sma_50"])

        crossed_below = (prev_ema_21 >= prev_sma_50) and (ema_21 < sma_50)
        down_close = float(current["close"]) < float(prev["close"])

        if crossed_below and down_close:
            state["power_trend"] = False
            before = state["exposure"]
            bar_signals.append(self._signal(
                current, state, "POWERTREND_OFF",
                "21 EMA crossed below 50 SMA on down close",
                exposure_before=before, exposure_after=state["exposure"],
                meta={"ema_21": ema_21, "sma_50": sma_50,
                      "prev_ema_21": prev_ema_21, "prev_sma_50": prev_sma_50},
            ))

    # ------------------------------------------------------------------------
    # V10 cascade-and-reset
    # ------------------------------------------------------------------------

    def _fire_v10_soft_reset(self, i, current, state, bar_signals):
        before = state["exposure"]
        # Re-enter rally-hunt phase but preserve rally_day_low and cap
        state["in_correction"] = True
        state["exposure"] = EXPOSURE_V10_SOFT_RESET
        # Ladder reset to Step 0 done; 1-7 cleared
        state["step0_done"] = True
        state["step1_done"] = False
        state["step2_done"] = False
        state["step3_done"] = False
        state["step4_done"] = False
        state["step5_done"] = False
        state["step6_done"] = False
        state["step7_done"] = False
        # FTD count restarts: rally_day_idx → current bar
        state["rally_day_idx"] = i
        state["rally_count"] = 1
        # rally_day_low PRESERVED per V11 spec
        # cap_at_100 PRESERVED
        state["ftd_close"] = None
        state["ftd_low"] = None
        # Reset running min so a fresh STEP_0 detection can run if invalidated later
        state["running_min_low"] = float(current["low"])
        state["running_min_idx"] = i
        state["rally_active"] = True

        bar_signals.append(self._signal(
            current, state, "V10_SOFT_RESET",
            f"cascade fired; low {float(current['low']):.2f} > rally_day_low "
            f"{state['rally_day_low']:.2f}; exposure → {EXPOSURE_V10_SOFT_RESET}%",
            exposure_before=before, exposure_after=state["exposure"],
            meta={"rally_day_low": state["rally_day_low"],
                  "low": float(current["low"]),
                  "cap_at_100_preserved": state["cap_at_100"]},
        ))

    def _fire_v10_full_invalidation(self, current, state, bar_signals):
        before = state["exposure"]
        had_cap = state["cap_at_100"]
        state["in_correction"] = True
        state["exposure"] = EXPOSURE_V10_FULL_INVALIDATION
        for s in range(8):
            state[f"step{s}_done"] = False
        state["rally_active"] = False
        state["rally_day_idx"] = None
        state["rally_day_low"] = None
        state["running_min_low"] = float(current["low"])
        state["running_min_idx"] = None
        state["rally_count"] = 0
        state["ftd_close"] = None
        state["ftd_low"] = None
        state["cap_at_100"] = False  # full invalidation clears cap

        bar_signals.append(self._signal(
            current, state, "V10_FULL_INVALIDATION",
            f"cascade fired; low ≤ rally_day_low; full reset",
            exposure_before=before, exposure_after=state["exposure"],
            meta={"low": float(current["low"])},
        ))
        if had_cap:
            bar_signals.append(self._signal(
                current, state, "CAP_AT_100_RELEASED",
                "cap-at-100 cleared on full invalidation",
                exposure_before=state["exposure"],
                exposure_after=state["exposure"],
                meta={},
            ))

    # ------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------

    def _signal(self, current, state, signal_type, label, *,
                exposure_before, exposure_after, meta):
        return SignalEvent(
            trade_date=current["trade_date"],
            signal_type=signal_type,
            signal_label=label,
            exposure_before=int(exposure_before),
            exposure_after=int(exposure_after),
            state_before=self._derive_state(state),
            state_after=self._derive_state(state),
            meta=meta or {},
        )

    def _derive_state(self, state) -> str:
        if state["power_trend"]:
            return "POWERTREND"
        if state["step4_done"] and not state["in_correction"]:
            return "UPTREND"
        if state["rally_active"] and (
            state["step0_done"] or state["step1_done"]
            or state["step2_done"] or state["step3_done"]
        ):
            return "RALLY MODE"
        return "CORRECTION"

    def _bar_record(self, current, state, bar_signals) -> dict:
        ema_21 = float(current["ema_21"]) if pd.notna(current["ema_21"]) else None
        sma_50 = float(current["sma_50"]) if pd.notna(current["sma_50"]) else None
        close = float(current["close"])
        return {
            "trade_date": current["trade_date"],
            "open": float(current["open"]),
            "high": float(current["high"]),
            "low": float(current["low"]),
            "close": close,
            "vs21": (close - ema_21) / ema_21 if ema_21 else None,
            "vs50": (close - sma_50) / sma_50 if sma_50 else None,
            "ema_8": float(current["ema_8"]) if pd.notna(current["ema_8"]) else None,
            "ema_21": ema_21,
            "sma_50": sma_50,
            "sma_200": float(current["sma_200"]) if pd.notna(current["sma_200"]) else None,
            "state": self._derive_state(state),
            "exposure": int(state["exposure"]),
            "cap_at_100": state["cap_at_100"],
            "in_correction": state["in_correction"],
            "correction_active": state["correction_active"],
            "power_trend": state["power_trend"],
            "rally_count": state["rally_count"],
            "reference_high": state["reference_high"],
            "signals_summary": " | ".join(s.signal_type for s in bar_signals),
        }
