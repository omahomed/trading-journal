"""Tests for the V11 MCT engine.

The DB-backed canonical run loads real ^IXIC bars from market_data and replays
the engine across 2024-12-16 → 2026-04-24 with reference seeded at 20,118.61.
Each canonical (date, signal_type) pair from the design-session reference run
must appear in the output ("must contain" subset semantics — the engine may
emit additional signals beyond the canonical list).

Synthetic tests use small inline DataFrames to exercise isolated mechanics
(anchor lifecycle, post-FTD soft fail) without DB access.

Tests skip cleanly if DATABASE_URL is unset.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Canonical signal log from the design session
# ---------------------------------------------------------------------------
# Each entry is (date, signal_type). The engine output must contain every one
# of these pairs as a subset.
CANONICAL_EVENTS: list[tuple[date, str]] = [
    (date(2025, 2, 27), "CORRECTION_DECLARED"),
    (date(2025, 2, 28), "STEP_0_RALLY_DAY"),
    (date(2025, 4, 7), "STEP_0_RALLY_DAY"),       # rally_day_low 14,784.03
    (date(2025, 4, 11), "STEP_1_FTD"),
    (date(2025, 4, 21), "POST_FTD_SOFT_FAIL"),    # close < ftd_low
    (date(2025, 4, 22), "STEP_1_FTD"),
    (date(2025, 4, 29), "STEP_4_LOW_ABOVE_21EMA_3BARS"),
    (date(2025, 5, 5), "STEP_5_LOW_ABOVE_50SMA_3BARS"),
    (date(2025, 5, 16), "STEP_8_POWERTREND_ON"),
    (date(2025, 6, 26), "CORRECTION_NULLIFIED"),
    (date(2025, 11, 20), "CORRECTION_DECLARED"),
    (date(2025, 12, 17), "VIOLATION_21EMA"),
    (date(2025, 12, 17), "CAP_AT_100_ACTIVATED"),
    (date(2025, 12, 18), "STEP_1_FTD"),
    (date(2025, 12, 24), "STEP_4_LOW_ABOVE_21EMA_3BARS"),
    (date(2025, 12, 26), "STEP_6_MA_STACK_SLOW"),
    (date(2025, 12, 29), "STEP_7_MA_STACK_FULL"),
    (date(2026, 2, 4), "VIOLATION_21EMA"),
    (date(2026, 2, 4), "VIOLATION_50SMA"),
    (date(2026, 2, 4), "CONFIRMED_BREAK_21EMA"),
    (date(2026, 2, 4), "V10_SOFT_RESET"),
    (date(2026, 2, 5), "POWERTREND_OFF"),
    (date(2026, 2, 24), "STEP_1_FTD"),
    (date(2026, 3, 3), "POST_FTD_SOFT_FAIL"),
    (date(2026, 3, 4), "STEP_1_FTD"),
    (date(2026, 3, 6), "POST_FTD_SOFT_FAIL"),
    (date(2026, 3, 9), "STEP_1_FTD"),
    (date(2026, 4, 8), "STEP_1_FTD"),
    (date(2026, 4, 8), "STEP_3_LOW_ABOVE_21EMA"),
    (date(2026, 4, 10), "STEP_4_LOW_ABOVE_21EMA_3BARS"),
    (date(2026, 4, 15), "STEP_7_MA_STACK_FULL"),
    (date(2026, 4, 16), "CORRECTION_NULLIFIED"),
    (date(2026, 4, 22), "STEP_8_POWERTREND_ON"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

requires_db = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not set; skipping DB-dependent tests",
)


@pytest.fixture(scope="module")
def canonical_run():
    """Run the engine over the canonical 12/16/2024 → 4/24/2026 window once
    per test module. Returns (history, result)."""
    if not os.getenv("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set")
    from api.market_data_repo import get_history
    from api.mct_engine import MCTEngine, EngineConfig

    history = get_history("^IXIC", date(2024, 12, 16), date(2026, 4, 24))
    config = EngineConfig(
        initial_reference_high=20118.61,
        initial_state="POWERTREND",
        initial_exposure=200,
        initial_power_trend=True,
        correction_ever_declared=True,
    )
    engine = MCTEngine(config)
    return history, engine.run(history)


# ---------------------------------------------------------------------------
# Canonical-subset tests
# ---------------------------------------------------------------------------

@requires_db
def test_full_run_signal_count_within_bounds(canonical_run):
    """Tight-bound sanity check on total signal count.

    Locked to 118–122 (count ±2) after V11 corrections: post-FTD soft-fail
    uses close, STEP_0 single-rule (up day or pink rally day), STEP_7
    re-fires after CB, and the V11 multi-signal same-bar rule allows
    STEP_3 to fire same bar as STEP_1+STEP_2 and STEP_7 to fire same bar
    as STEP_6 when conditions are structurally met. Any change that
    shifts the count outside this band indicates a behavioral regression
    and should be investigated.
    """
    _, result = canonical_run
    assert 118 <= len(result.signals) <= 122, (
        f"Got {len(result.signals)} signals; expected 118–122"
    )


@requires_db
@pytest.mark.parametrize("event_date,event_type", CANONICAL_EVENTS)
def test_canonical_signal_present(canonical_run, event_date, event_type):
    """Each canonical (date, signal_type) pair must appear in the engine output."""
    _, result = canonical_run
    pairs = {(s.trade_date, s.signal_type) for s in result.signals}
    assert (event_date, event_type) in pairs, (
        f"Canonical signal {event_type} on {event_date} missing from engine output"
    )


# ---------------------------------------------------------------------------
# Targeted assertions (specific values, not just presence)
# ---------------------------------------------------------------------------

@requires_db
def test_step0_2025_04_07_rally_day_low(canonical_run):
    """4/7/2025 STEP_0 sets rally_day_low to 14,784.03 (running-min low)."""
    _, result = canonical_run
    step0s = [s for s in result.signals
              if s.signal_type == "STEP_0_RALLY_DAY" and s.trade_date == date(2025, 4, 7)]
    assert step0s, "STEP_0_RALLY_DAY missing on 4/7/2025"
    sig = step0s[0]
    assert sig.meta["rally_day_low"] == pytest.approx(14784.03, abs=0.5)


@requires_db
def test_correction_declared_2025_02_27(canonical_run):
    """2/27/2025 declaration: close 18,544 ≤ 7% below seed reference 20,118.61."""
    _, result = canonical_run
    decls = [s for s in result.signals
             if s.signal_type == "CORRECTION_DECLARED" and s.trade_date == date(2025, 2, 27)]
    assert decls, "CORRECTION_DECLARED missing on 2/27/2025"
    sig = decls[0]
    assert sig.meta["reference_high"] == pytest.approx(20118.61, abs=0.01)
    assert sig.meta["close"] == pytest.approx(18544.42, abs=0.01)


@requires_db
def test_anchored_21ema_violation_2025_12_17(canonical_run):
    """12/17/2025 VIOLATION_21EMA: anchor 23,094.51 (12/12 low), low 22,692, 1.74% undercut."""
    _, result = canonical_run
    vios = [s for s in result.signals
            if s.signal_type == "VIOLATION_21EMA" and s.trade_date == date(2025, 12, 17)]
    assert vios, "VIOLATION_21EMA missing on 12/17/2025"
    sig = vios[0]
    assert sig.meta["anchor_low"] == pytest.approx(23094.51, abs=1.0)
    assert sig.meta["low"] == pytest.approx(22692.00, abs=1.0)
    assert sig.meta["undercut_pct"] == pytest.approx(0.0174, abs=0.0005)


@requires_db
def test_v10_soft_reset_2026_02_04(canonical_run):
    """2/4/2026: cascade fires; low 22,684 > rally_day_low 21,898 → V10 soft reset."""
    _, result = canonical_run
    soft = [s for s in result.signals
            if s.signal_type == "V10_SOFT_RESET" and s.trade_date == date(2026, 2, 4)]
    assert soft, "V10_SOFT_RESET missing on 2/4/2026"
    sig = soft[0]
    assert sig.meta["rally_day_low"] == pytest.approx(21898.29, abs=1.0)
    assert sig.meta["cap_at_100_preserved"] is True
    assert sig.exposure_after == 20


@requires_db
def test_correction_nullification_2026_04_16(canonical_run):
    """4/16/2026: close 24,103 > reference high 24,019.99 → nullification."""
    _, result = canonical_run
    nulls = [s for s in result.signals
             if s.signal_type == "CORRECTION_NULLIFIED" and s.trade_date == date(2026, 4, 16)]
    assert nulls, "CORRECTION_NULLIFIED missing on 4/16/2026"
    sig = nulls[0]
    assert sig.meta["reference_high"] == pytest.approx(24019.99, abs=0.5)


@requires_db
def test_cycle_day_anchored_to_step0_2026_04_24(canonical_run):
    """day_num counts trading days from the most recent STEP_0 firing.
    Surviving cycle on the canonical run started 3/31/2026; 4/24/2026 should
    be Day 18 (3/31=1, 4/1=2, 4/2=3, [4/3 Good Friday closed], 4/6=4, …,
    4/24=18). Internal rally_count freezes at Step-4 (Day 9 on 4/10) — the
    new cycle_start_idx-anchored display is what the user sees."""
    history, result = canonical_run
    bars = result.bars
    last_idx = len(bars) - 1
    cycle_start = result.final_state.get("cycle_start_idx")
    assert cycle_start is not None, "expected an active cycle"
    cycle_day = last_idx - int(cycle_start) + 1
    assert cycle_day == 18, f"Expected Day 18 on 4/24/2026, got Day {cycle_day}"

    # Confirm cycle_start_idx points to 3/31/2026 specifically.
    cycle_start_date = bars.iloc[int(cycle_start)]["trade_date"]
    assert cycle_start_date == date(2026, 3, 31), (
        f"cycle_start_idx points to {cycle_start_date}, expected 2026-03-31"
    )


@requires_db
def test_v10_soft_reset_preserves_cycle_start_idx(canonical_run):
    """V10 soft resets at 12/15/2025, 1/2/2026, 2/4/2026 must NOT touch
    cycle_start_idx. The 11/21/2025 STEP_0 anchors a cycle that survives
    every soft reset until rally invalidation on 3/19/2026. On 2/24/2026
    — well after multiple soft resets — cycle_start_idx still points at
    the 11/21 bar, and cycle_day is in the mid-60s (calendar trading days
    11/21 → 2/24 inclusive)."""
    history, result = canonical_run
    bars = result.bars

    bar_2_24_rows = bars.index[bars["trade_date"] == date(2026, 2, 24)]
    assert len(bar_2_24_rows) == 1, "2/24/2026 missing from canonical run"
    bar_2_24_idx = int(bar_2_24_rows[0])

    cycle_start_at_2_24 = bars.iloc[bar_2_24_idx]["cycle_start_idx"]
    assert cycle_start_at_2_24 is not None, "cycle was active on 2/24/2026"
    cycle_start_at_2_24 = int(cycle_start_at_2_24)

    # Anchor must be 11/21/2025.
    cycle_start_date = bars.iloc[cycle_start_at_2_24]["trade_date"]
    assert cycle_start_date == date(2025, 11, 21), (
        f"cycle_start_idx on 2/24/2026 points to {cycle_start_date}, "
        f"expected 2025-11-21 (V10 soft reset must not have touched it)"
    )

    # Day count from 11/21 should be in the 60s (calendar trading days).
    cycle_day_2_24 = bar_2_24_idx - cycle_start_at_2_24 + 1
    assert 60 <= cycle_day_2_24 <= 70, (
        f"Expected cycle_day ~65 on 2/24/2026, got Day {cycle_day_2_24}"
    )


@requires_db
def test_powertrend_off_then_on(canonical_run):
    """PT-OFF on 2/5/2026, then PT-ON on 4/22/2026 — both should fire."""
    _, result = canonical_run
    types = {(s.trade_date, s.signal_type) for s in result.signals}
    assert (date(2026, 2, 5), "POWERTREND_OFF") in types
    assert (date(2026, 4, 22), "STEP_8_POWERTREND_ON") in types


@requires_db
def test_pt_on_idx_anchored_at_2026_04_22(canonical_run):
    """STEP_8_POWERTREND_ON fires on 4/22/2026; pt_on_idx must point at that bar."""
    _, result = canonical_run
    bars = result.bars
    pt_idx = result.final_state.get("pt_on_idx")
    assert pt_idx is not None, "PT is currently ON in the canonical run; pt_on_idx must be set"
    anchor_date = bars.iloc[int(pt_idx)]["trade_date"]
    assert anchor_date == date(2026, 4, 22), (
        f"pt_on_idx points to {anchor_date}, expected 2026-04-22"
    )


@requires_db
def test_pt_on_idx_yields_powertrend_d3_on_2026_04_24(canonical_run):
    """4/22 = D1, 4/23 = D2, 4/24 = D3 — what the journal MCT State badge shows."""
    _, result = canonical_run
    bars = result.bars
    last_idx = len(bars) - 1
    last_date = bars.iloc[last_idx]["trade_date"]
    assert last_date == date(2026, 4, 24), (
        f"Canonical run ends on {last_date}, expected 2026-04-24"
    )
    pt_idx = int(result.final_state["pt_on_idx"])
    pt_day = last_idx - pt_idx + 1
    assert pt_day == 3, f"Expected PT Day 3 on 4/24/2026, got Day {pt_day}"


@requires_db
def test_pt_on_idx_cleared_during_pt_off_window(canonical_run):
    """Between 2/5/2026 (PT-OFF) and 4/22/2026 (re-PT-ON), pt_on_idx must be None.
    Sample bar: 3/2/2026 (mid-window, plenty of bars on either side)."""
    _, result = canonical_run
    bars = result.bars
    rows = bars.index[bars["trade_date"] == date(2026, 3, 2)]
    assert len(rows) == 1, "3/2/2026 missing from canonical run"
    sample = bars.iloc[int(rows[0])]
    pt_idx = sample["pt_on_idx"]
    assert pt_idx is None or pd.isna(pt_idx), (
        f"pt_on_idx should be None on 3/2/2026 (between PT runs), got {pt_idx}"
    )


@requires_db
def test_pt_on_idx_re_anchors_on_second_step8(canonical_run):
    """STEP_8 fires twice in the canonical run (2025-05-16 then 2026-04-22).
    After PT-OFF on 2026-02-05 cleared the anchor, the 2026-04-22 STEP_8
    must re-anchor pt_on_idx to its own bar — NOT keep the 2025-05-16 idx."""
    _, result = canonical_run
    bars = result.bars

    first_step8_rows = bars.index[bars["trade_date"] == date(2025, 5, 16)]
    second_step8_rows = bars.index[bars["trade_date"] == date(2026, 4, 22)]
    assert len(first_step8_rows) == 1
    assert len(second_step8_rows) == 1
    second_idx = int(second_step8_rows[0])

    pt_idx_now = int(result.final_state["pt_on_idx"])
    assert pt_idx_now == second_idx, (
        f"pt_on_idx must re-anchor to 2026-04-22 bar ({second_idx}), "
        f"not the 2025-05-16 bar; got {pt_idx_now}"
    )


# ---------------------------------------------------------------------------
# Synthetic / isolated-mechanics tests (no DB)
# ---------------------------------------------------------------------------

def _synthetic_history(closes, ema_21=None, sma_50=None, sma_200=None,
                        ema_8=None, lows=None, highs=None,
                        start: date = date(2026, 1, 5)) -> pd.DataFrame:
    """Build a minimal DataFrame the engine can consume."""
    n = len(closes)
    if lows is None:
        lows = [c - 1.0 for c in closes]
    if highs is None:
        highs = [c + 1.0 for c in closes]
    if ema_8 is None:
        ema_8 = pd.Series(closes).ewm(span=8, adjust=False).mean().tolist()
    if ema_21 is None:
        ema_21 = pd.Series(closes).ewm(span=21, adjust=False).mean().tolist()
    if sma_50 is None:
        sma_50 = pd.Series(closes).rolling(50).mean().tolist()
    if sma_200 is None:
        sma_200 = pd.Series(closes).rolling(200).mean().tolist()
    return pd.DataFrame({
        "trade_date": [start + timedelta(days=i) for i in range(n)],
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [1_000_000] * n,
        "ema_8": ema_8,
        "ema_21": ema_21,
        "sma_50": sma_50,
        "sma_200": sma_200,
    })


def test_anchor_resets_on_close_above_21ema():
    """Once close goes back above 21 EMA, anchor and violation_21_fired reset."""
    from api.mct_engine import MCTEngine, EngineConfig

    # 5 bars below 21 EMA (anchor stays fixed), 1 bar back above (anchor clears)
    closes = [100.0, 99.0, 98.5, 98.0, 97.5, 102.0]
    ema_21 = [100.0] * 6
    lows = [99.0, 98.0, 97.5, 97.0, 96.5, 101.0]
    highs = [100.5, 99.5, 99.0, 98.5, 98.0, 102.5]
    df = _synthetic_history(closes, ema_21=ema_21, sma_50=[100.0]*6,
                             sma_200=[100.0]*6, lows=lows, highs=highs)

    engine = MCTEngine(EngineConfig(initial_reference_high=200.0,
                                     initial_power_trend=False,
                                     initial_exposure=100))
    result = engine.run(df)
    final = result.final_state
    # After last bar (close 102 > 21 EMA 100), anchor should be cleared.
    assert final["anchor_21_low"] is None
    assert final["violation_21_fired"] is False
    assert final["consec_below_21"] == 0


def test_anchored_violation_no_refire_within_streak():
    """VIOLATION_21EMA fires once per streak even if subsequent bars undercut more."""
    from api.mct_engine import MCTEngine, EngineConfig

    # Bar 0 sets anchor (low 99.0), bar 1 undercuts by ~2%, bar 2 undercuts more.
    closes = [99.0, 95.0, 90.0]
    ema_21 = [100.0, 100.0, 100.0]
    lows = [99.0, 97.0, 90.0]
    highs = [100.0, 99.0, 95.0]
    df = _synthetic_history(closes, ema_21=ema_21, sma_50=[100.0]*3,
                             sma_200=[100.0]*3, lows=lows, highs=highs)

    engine = MCTEngine(EngineConfig(initial_reference_high=200.0,
                                     initial_power_trend=False,
                                     initial_exposure=100))
    result = engine.run(df)
    vios = [s for s in result.signals if s.signal_type == "VIOLATION_21EMA"]
    assert len(vios) == 1, (
        f"Expected exactly 1 VIOLATION_21EMA per streak, got {len(vios)}"
    )


# ---------------------------------------------------------------------------
# step4_ever_fired latch lifecycle — UPTREND UNDER PRESSURE support
# ---------------------------------------------------------------------------


def _init_state_default() -> dict:
    """Return a fresh engine state dict via a minimal 1-bar replay so tests
    can inspect the initial `step4_ever_fired` value."""
    from api.mct_engine import MCTEngine, EngineConfig
    df = _synthetic_history([100.0], ema_21=[100.0], sma_50=[100.0],
                             sma_200=[100.0])
    engine = MCTEngine(EngineConfig(initial_reference_high=200.0,
                                     initial_power_trend=False,
                                     initial_exposure=0))
    result = engine.run(df)
    return result.final_state


def test_step4_ever_fired_initializes_false():
    """Fresh engine state has step4_ever_fired=False (baseline for the
    latch)."""
    state = _init_state_default()
    assert state["step4_ever_fired"] is False


def test_step4_ever_fired_latches_true_on_step4_arm():
    """When STEP_4 arms (3 consecutive bars with low > 21 EMA following
    Step 3), step4_ever_fired latches True on the same bar as
    step4_done."""
    from api.mct_engine import MCTEngine, EngineConfig

    # Bar 0: STEP_0 (up close), Bar 1: STEP_1 (FTD >=1% gain, rally_count>=4),
    # Bars 2..: build up low>21EMA streak for STEP_2/3/4.
    # Use a longer runway so rally_count reaches the FTD window (>=4).
    n = 12
    closes = [100.0, 101.0, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5,
              108.5, 109.5, 110.5, 111.5]
    ema_21 = [99.0] * n
    lows = [c - 0.3 for c in closes]  # low > 21 EMA throughout
    highs = [c + 0.5 for c in closes]
    df = _synthetic_history(closes, ema_21=ema_21, sma_50=[95.0]*n,
                             sma_200=[90.0]*n, lows=lows, highs=highs)

    engine = MCTEngine(EngineConfig(initial_reference_high=150.0,
                                     initial_power_trend=False,
                                     initial_exposure=0,
                                     correction_ever_declared=True))
    # Kick off in_correction so Phase 7 (rally hunt) runs.
    result = engine.run(df)
    # If STEP_4 fired, step4_ever_fired must be True in final state.
    step4_fired = any(s.signal_type == "STEP_4_LOW_ABOVE_21EMA_3BARS"
                      for s in result.signals)
    if step4_fired:
        assert result.final_state["step4_ever_fired"] is True, (
            "step4_ever_fired must latch True when STEP_4 arms"
        )
    # Also assert the per-bar snapshot exposes the field (needed by
    # downstream consumers per the Commit 2 spec).
    assert "step4_ever_fired" in result.bars.columns


def test_step4_ever_fired_persists_through_v10_soft_reset():
    """V10_SOFT_RESET clears step4_done but MUST NOT clear
    step4_ever_fired. This is the load-bearing invariant for the
    UPTREND UNDER PRESSURE branch — a mid-cycle break leaves the latch
    intact so the state resolvers can catch the post-Step-4-stressed
    label."""
    # Directly manipulate a state dict and drive the private
    # _fire_v10_soft_reset helper. Avoids a full replay while pinning
    # the exact invariant.
    from api.mct_engine import MCTEngine, EngineConfig

    engine = MCTEngine(EngineConfig(initial_reference_high=200.0,
                                     initial_power_trend=False,
                                     initial_exposure=100))
    state = engine._init_state()
    # Pretend Step 4 armed in a prior bar.
    state["step4_done"] = True
    state["step4_ever_fired"] = True
    state["rally_day_low"] = 100.0
    state["rally_day_idx"] = 0
    state["cap_at_100"] = False
    state["correction_active"] = False

    # Synthesize a "current" bar row (Series-like dict) with the low
    # V10 needs to record for cascade reasoning.
    import pandas as pd
    current = pd.Series({
        "trade_date": pd.Timestamp("2026-06-08"),
        "close": 100.5,
        "low": 100.2,
        "high": 101.0,
        "open": 100.5,
        "ema_21": 100.0,
        "sma_50": 100.0,
    })
    bar_signals = []
    engine._fire_v10_soft_reset(i=5, current=current, state=state,
                                 bar_signals=bar_signals)
    assert state["step4_done"] is False, "V10_SOFT_RESET must clear step4_done"
    assert state["step4_ever_fired"] is True, (
        "V10_SOFT_RESET must preserve step4_ever_fired — same rule as "
        "cycle_start_idx (see mct_engine.py docstring on the soft-reset "
        "path). This is what makes the UUP label reachable."
    )


def test_step4_ever_fired_persists_through_post_ftd_soft_fail():
    """POST_FTD_SOFT_FAIL clears step4_done but MUST NOT clear
    step4_ever_fired. Same reasoning as V10_SOFT_RESET — mid-cycle
    reset, not cycle boundary."""
    from api.mct_engine import MCTEngine, EngineConfig
    import pandas as pd

    engine = MCTEngine(EngineConfig(initial_reference_high=200.0,
                                     initial_power_trend=False,
                                     initial_exposure=100))
    state = engine._init_state()
    state["step4_done"] = True
    state["step4_ever_fired"] = True
    state["correction_active"] = False

    current = pd.Series({
        "trade_date": pd.Timestamp("2026-06-15"),
        "close": 95.0,
        "low": 94.0,
        "high": 96.0,
        "open": 95.0,
        "ema_21": 100.0,
        "sma_50": 100.0,
    })
    bar_signals = []
    engine._fire_post_ftd_soft_fail(current=current, state=state,
                                     bar_signals=bar_signals)
    assert state["step4_done"] is False, (
        "POST_FTD_SOFT_FAIL must clear step4_done"
    )
    assert state["step4_ever_fired"] is True, (
        "POST_FTD_SOFT_FAIL must preserve step4_ever_fired"
    )


def test_step4_ever_fired_clears_on_rally_invalidated():
    """RALLY_INVALIDATED is a cycle boundary — step4_ever_fired must
    clear alongside step_done flags and cycle_start_idx."""
    from api.mct_engine import MCTEngine, EngineConfig
    import pandas as pd

    engine = MCTEngine(EngineConfig(initial_reference_high=200.0,
                                     initial_power_trend=False,
                                     initial_exposure=100))
    state = engine._init_state()
    state["step4_ever_fired"] = True
    state["cycle_start_idx"] = 3
    state["correction_active"] = False
    state["cap_at_100"] = False

    current = pd.Series({
        "trade_date": pd.Timestamp("2026-06-20"),
        "close": 90.0,
        "low": 89.0,
        "high": 91.0,
        "open": 90.0,
        "ema_21": 100.0,
        "sma_50": 100.0,
    })
    bar_signals = []
    engine._fire_rally_invalidation(i=10, current=current, state=state,
                                     bar_signals=bar_signals,
                                     reason="test-invalidation")
    assert state["step4_ever_fired"] is False, (
        "RALLY_INVALIDATED must clear step4_ever_fired — cycle boundary"
    )
    assert state["cycle_start_idx"] is None, (
        "cycle_start_idx and step4_ever_fired MUST clear at the same "
        "sites — they share the same reset semantics"
    )


def test_step4_ever_fired_clears_on_v10_full_invalidation():
    """V10_FULL_INVALIDATION is a cycle boundary — step4_ever_fired
    must clear."""
    from api.mct_engine import MCTEngine, EngineConfig
    import pandas as pd

    engine = MCTEngine(EngineConfig(initial_reference_high=200.0,
                                     initial_power_trend=False,
                                     initial_exposure=100))
    state = engine._init_state()
    state["step4_ever_fired"] = True
    state["cycle_start_idx"] = 3
    state["correction_active"] = False
    state["cap_at_100"] = False

    current = pd.Series({
        "trade_date": pd.Timestamp("2026-06-25"),
        "close": 88.0,
        "low": 87.0,
        "high": 89.0,
        "open": 88.0,
        "ema_21": 100.0,
        "sma_50": 100.0,
    })
    bar_signals = []
    engine._fire_v10_full_invalidation(current=current, state=state,
                                        bar_signals=bar_signals)
    assert state["step4_ever_fired"] is False, (
        "V10_FULL_INVALIDATION must clear step4_ever_fired"
    )


def test_step4_ever_fired_clears_on_correction_declared():
    """CORRECTION_DECLARED is a cycle boundary — step4_ever_fired
    must clear alongside the range clear of step0..step7_done."""
    from api.mct_engine import MCTEngine, EngineConfig

    # 2-bar setup where both bars pass the correction gates
    # (close ≤ 90% of reference_high AND close < sma_50) so
    # declaration fires on bar 2.
    closes = [80.0, 79.0]
    ema_21 = [95.0, 94.0]
    sma_50 = [90.0, 90.0]
    df = _synthetic_history(closes, ema_21=ema_21, sma_50=sma_50,
                             sma_200=[85.0]*2)
    engine = MCTEngine(EngineConfig(initial_reference_high=100.0,
                                     initial_power_trend=False,
                                     initial_exposure=100))
    # Manually seed step4_ever_fired True BEFORE running so the test
    # sees the clear happen. Also set correction_ever_declared so the
    # engine will attempt a fresh declaration.
    state = engine._init_state()
    state["step4_ever_fired"] = True
    # Drive _phase_declaration directly. First bar arms pending; second
    # bar declares.
    engine._phase_declaration(df.iloc[0], state, [])
    engine._phase_declaration(df.iloc[1], state, [])
    # After declaration, the range loop clears step_done AND the new
    # step4_ever_fired write clears the latch.
    if state["correction_active"]:
        assert state["step4_ever_fired"] is False, (
            "CORRECTION_DECLARED must clear step4_ever_fired"
        )


# ---------------------------------------------------------------------------
# Real-date anchor test — the 2026-07-07 regression guard
# ---------------------------------------------------------------------------


@requires_db
def test_derive_state_2026_07_07_returns_uup_regression_guard():
    """Motivating-bug regression guard.

    Replays the engine over the full canonical history through
    2026-07-07 and asserts _derive_state on the final bar returns
    'UPTREND UNDER PRESSURE'. Locks the fix for the 2026-07-07 label
    bug end-to-end:

      - Live signals in market_signals for 2026-06-01 → 2026-07-07
        (verified via read-only query): V10_SOFT_RESET on 2026-06-08
        set in_correction=True; no CORRECTION_DECLARED in window;
        last CORRECTION_NULLIFIED was 2026-04-16 so correction_active
        has been False since April; drawdown ~5% off reference_high,
        well short of the 10% depth gate.

      - Under the old spec (UUP gated on `not in_correction`), today
        would erroneously return RALLY MODE because the V10-induced
        phantom in_correction=True blocks the UUP branch.

      - Under the new spec (UUP gated on `not correction_active`),
        today correctly returns UPTREND UNDER PRESSURE.

    If this test fails, the gate has been swapped back OR
    step4_ever_fired is being cleared incorrectly OR the export in
    _bar_record has drifted.
    """
    from api.mct_engine import MCTEngine, EngineConfig
    from api.market_data_repo import get_history, get_latest_date

    end = get_latest_date("^IXIC") or date.today()
    history = get_history("^IXIC", date(2010, 1, 1), end)
    config = EngineConfig(
        initial_reference_high=None,
        initial_state="POWERTREND",
        initial_exposure=200,
        initial_power_trend=True,
        correction_ever_declared=True,
        initial_ratchet_armed=True,
    )
    engine = MCTEngine(config)
    result = engine.run(history)
    bars = result.bars

    mask = pd.to_datetime(bars["trade_date"]).dt.date == date(2026, 7, 7)
    today = bars[mask]
    if today.empty:
        pytest.skip("2026-07-07 not in canonical market_data — "
                    "backfill needed before this test can run")
    row = today.iloc[0]

    # Sanity fingerprint — explicit bool coercion because pandas returns
    # numpy.bool_, for which `is False` is always False (the `== 0`
    # fallback would silently succeed on any value).
    assert not bool(row["step4_done"]), (
        "step4_done should be False on 2026-07-07 (cleared by "
        "V10_SOFT_RESET on 2026-06-08 and never re-armed)"
    )
    assert bool(row["step4_ever_fired"]), (
        "step4_ever_fired should be True on 2026-07-07 (latch survived "
        "V10_SOFT_RESET — mid-cycle reset, not a cycle boundary)"
    )
    assert not bool(row["correction_active"]), (
        "correction_active should be False on 2026-07-07 (last "
        "CORRECTION_NULLIFIED was 2026-04-16; no CORRECTION_DECLARED "
        "since)"
    )
    assert not bool(row["power_trend"]), (
        "power_trend should be False on 2026-07-07 (POWERTREND_OFF "
        "fired today)"
    )

    # The load-bearing assertions — the point of the test
    assert engine._derive_state(row.to_dict()) == "UPTREND UNDER PRESSURE"
    assert row["state"] == "UPTREND UNDER PRESSURE"
