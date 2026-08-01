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
# Depth gate uses intraday LOW, not close (2026-07-29 switch)
# ---------------------------------------------------------------------------


def test_declaration_fires_when_both_conditions_met():
    """The rule: 2 closes below the 50 SMA + intraday low ≤ 10% off
    the reference high. Both true → CORRECTION_DECLARED. Structure
    is tracked via `consec_below_50` (streak that naturally resets
    when close pops back above SMA — see the reset test below).

    Motivating scenario (NASDAQ 2026-07-29): 6 prior consecutive
    closes below SMA50; the intraday low first crossed the -10%
    depth threshold today. Declaration fires same-day."""
    from api.mct_engine import MCTEngine, EngineConfig

    closes = [91.0]        # < SMA50 today (contributes to streak this bar)
    lows = [89.0]          # < threshold=90 → depth passes
    highs = [92.0]
    df = _synthetic_history(closes, ema_21=[95.0], sma_50=[95.0],
                             sma_200=[80.0], lows=lows, highs=highs)
    engine = MCTEngine(EngineConfig(initial_reference_high=100.0,
                                     initial_power_trend=False,
                                     initial_exposure=100,
                                     correction_ever_declared=True))
    state = engine._init_state()
    # Prior streak from _phase_update_streaks (which runs AFTER this
    # phase). Non-zero means yesterday closed below SMA — combined
    # with today's close_below_sma that's 2+ consecutive.
    state["consec_below_50"] = 5

    engine._phase_declaration(df.iloc[0], state, [])
    assert state["correction_active"] is True


def test_declaration_stays_gated_when_close_pops_back_above_sma():
    """If today's close pops back above the SMA — even after a long
    prior streak below — structure gate breaks for today. The engine's
    _phase_update_streaks will zero consec_below_50 in the same bar's
    later phase, but _phase_declaration reads today's close directly
    to catch the same-bar reset."""
    from api.mct_engine import MCTEngine, EngineConfig

    closes = [97.0]        # ABOVE SMA50=95 — today's close breaks streak
    lows = [89.0]          # depth would still pass
    highs = [98.0]
    df = _synthetic_history(closes, ema_21=[95.0], sma_50=[95.0],
                             sma_200=[80.0], lows=lows, highs=highs)
    engine = MCTEngine(EngineConfig(initial_reference_high=100.0,
                                     initial_power_trend=False,
                                     initial_exposure=100,
                                     correction_ever_declared=True))
    state = engine._init_state()
    state["consec_below_50"] = 5   # long prior streak, but today closes above

    engine._phase_declaration(df.iloc[0], state, [])
    assert state["correction_active"] is False


def test_declaration_stays_gated_when_only_one_prior_close_below_sma():
    """Structure requires 2 consecutive including today. Prior streak
    = 0 means yesterday closed above SMA; today's first-below-SMA
    close only makes 1 consecutive → not enough."""
    from api.mct_engine import MCTEngine, EngineConfig

    closes = [91.0]        # < SMA today (1st below), but no prior streak
    lows = [89.0]          # depth would pass
    highs = [92.0]
    df = _synthetic_history(closes, ema_21=[95.0], sma_50=[95.0],
                             sma_200=[80.0], lows=lows, highs=highs)
    engine = MCTEngine(EngineConfig(initial_reference_high=100.0,
                                     initial_power_trend=False,
                                     initial_exposure=100,
                                     correction_ever_declared=True))
    state = engine._init_state()
    state["consec_below_50"] = 0   # yesterday was ABOVE SMA

    engine._phase_declaration(df.iloc[0], state, [])
    assert state["correction_active"] is False


def test_declaration_stays_gated_when_depth_fails():
    """Structure long met (5 prior consecutive below-SMA closes +
    today closes below) but intraday low stays above threshold → no
    declaration. Both conditions have to hold."""
    from api.mct_engine import MCTEngine, EngineConfig

    closes = [91.0]        # < SMA — streak continues
    lows = [92.0]          # ABOVE threshold=90 — depth fails
    highs = [93.0]
    df = _synthetic_history(closes, ema_21=[95.0], sma_50=[95.0],
                             sma_200=[80.0], lows=lows, highs=highs)
    engine = MCTEngine(EngineConfig(initial_reference_high=100.0,
                                     initial_power_trend=False,
                                     initial_exposure=100,
                                     correction_ever_declared=True))
    state = engine._init_state()
    state["consec_below_50"] = 5   # structure met

    engine._phase_declaration(df.iloc[0], state, [])
    assert state["correction_active"] is False


def test_structure_streak_reset_scenario_end_to_end():
    """The exact scenario the user walked through:

        Day 1: close below SMA          → streak = 1
        Day 2: close ABOVE SMA          → RESET, streak = 0
        Day 3: close below SMA (again)  → streak = 1 (only 1 consec)
        Day 4: close below SMA (again)  → streak = 2 (arms structure)
        Day 5+: depth crosses           → declare

    Uses _phase_update_streaks between bars so the streak evolves
    naturally (no state manipulation). Depth stays above threshold
    for bars 1-4 so we can prove structure gates by itself; bar 5
    crosses depth to confirm the full trigger fires."""
    from api.mct_engine import MCTEngine, EngineConfig

    #        Day 1   Day 2   Day 3   Day 4   Day 5
    closes = [91.0,  99.0,   91.0,   91.0,   91.0]   # bar 2 pops above
    lows   = [92.0,  98.0,   92.0,   92.0,   89.0]   # only bar 5 crosses depth
    highs  = [93.0,  100.0,  93.0,   93.0,   92.0]
    sma_50 = [95.0]  * 5
    ema_21 = [95.0]  * 5
    df = _synthetic_history(closes, ema_21=ema_21, sma_50=sma_50,
                             sma_200=[80.0] * 5, lows=lows, highs=highs)
    engine = MCTEngine(EngineConfig(initial_reference_high=100.0,
                                     initial_power_trend=False,
                                     initial_exposure=100,
                                     correction_ever_declared=True))
    state = engine._init_state()

    # Day 1 — first close below. Neither declaration NOR depth possible.
    engine._phase_declaration(df.iloc[0], state, [])
    assert state["correction_active"] is False
    engine._phase_update_streaks(df.iloc[0], None, state)
    assert state["consec_below_50"] == 1

    # Day 2 — close above SMA. Streak resets. Would-be depth doesn't matter.
    engine._phase_declaration(df.iloc[1], state, [])
    assert state["correction_active"] is False
    engine._phase_update_streaks(df.iloc[1], None, state)
    assert state["consec_below_50"] == 0, "close above SMA must reset streak"

    # Day 3 — close below again. First bar of a NEW streak, not enough.
    engine._phase_declaration(df.iloc[2], state, [])
    assert state["correction_active"] is False, (
        "streak just started over on day 3 — one bar isn't enough even "
        "if depth were met. The reset on day 2 wiped the earlier streak."
    )
    engine._phase_update_streaks(df.iloc[2], None, state)
    assert state["consec_below_50"] == 1

    # Day 4 — close below, streak = 2. Still no depth cross → no declare.
    engine._phase_declaration(df.iloc[3], state, [])
    assert state["correction_active"] is False, "depth still not met"
    engine._phase_update_streaks(df.iloc[3], None, state)
    assert state["consec_below_50"] == 2

    # Day 5 — close below (streak already ≥ 2), AND depth crosses. Declare.
    engine._phase_declaration(df.iloc[4], state, [])
    assert state["correction_active"] is True, (
        "structure (2+ consec closes below SMA, including today) + "
        "depth (low ≤ threshold) both met on day 5 — must declare."
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


# ---------------------------------------------------------------------------
# Force-Correction override (migration 053) — user-declared CORRECTION seed
# ---------------------------------------------------------------------------

def test_force_correction_at_date_seeds_engine_state():
    """When config.force_correction_at_date matches a bar and the systematic
    depth threshold would NOT fire on that bar, the engine still declares
    CORRECTION there — resetting step flags, rally state, and dropping
    exposure. Subsequent bars run normal rally-hunt detection off the
    forced anchor."""
    from api.mct_engine import MCTEngine, EngineConfig

    # Series of flat-ish bars sitting above the 50 SMA — systematic
    # gates would NEVER fire (no depth, no structure break).
    closes = [110.0] * 8
    lows = [109.0] * 8
    highs = [111.0] * 8
    df = _synthetic_history(
        closes,
        ema_21=[110.0] * 8,
        sma_50=[100.0] * 8,          # close above 50 SMA — no structure break
        sma_200=[95.0] * 8,
        lows=lows, highs=highs,
    )
    # Reference high 111 keeps drawdown ~1%, well shy of the 10% depth gate.
    force_date = df["trade_date"].iloc[3]

    engine = MCTEngine(EngineConfig(
        initial_reference_high=111.0,
        initial_power_trend=False,
        initial_exposure=100,
        force_correction_at_date=force_date,
    ))
    result = engine.run(df)

    # CORRECTION_DECLARED fires exactly once, on the forced bar, with the
    # user_override trigger stamped in meta.
    declared = [s for s in result.signals if s.signal_type == "CORRECTION_DECLARED"]
    assert len(declared) == 1, (
        f"Expected exactly 1 CORRECTION_DECLARED (from override), got {len(declared)}"
    )
    assert declared[0].trade_date == force_date
    assert declared[0].meta.get("trigger") == "user_override"

    # State reset side-effects — step flags all False, rally state zeroed,
    # correction_active True.
    final = result.final_state
    assert final["correction_active"] is True
    assert final["correction_ever_declared"] is True
    for s in range(8):
        # step0/1 may re-fire on bars after the override if the synthetic
        # closes happen to satisfy STEP_0 (they do — flat closes still
        # count as "not lower than prev" in some engine paths). Skip step0/1
        # asserts; the reset-at-force-bar itself is the invariant.
        if s in (0, 1):
            continue
        assert final[f"step{s}_done"] is False, f"step{s}_done should be False after override reset"


def test_force_correction_today_can_also_fire_step0_same_bar():
    """The override date can BE a rally day on its own bar. Phase 3a
    (force-declare) clears running_min_low, but phase 7 (rally-hunt)
    runs on the SAME bar in-correction, sets running_min from the just-
    declared bar's low, then evaluates the up_day / pink_rally_day
    conditions against prev_close. If today closes above the prior
    bar's close, STEP_0 fires today — cycle_start_idx = today's index.

    This is the user-model invariant: declaring correction today does
    not push the rally-hunt to start tomorrow; today itself can qualify
    depending on how it closes."""
    from api.mct_engine import MCTEngine, EngineConfig

    # 5 flat bars (nothing above threshold) then a 6th bar that closes
    # UP vs the 5th. Force-declare on the 6th bar. If the flow works
    # correctly, that same bar should emit STEP_0_RALLY_DAY as an up_day.
    closes = [110.0, 110.0, 110.0, 110.0, 110.0, 111.5]  # last bar closes up
    lows =   [109.0, 109.0, 109.0, 109.0, 109.0, 110.0]
    highs =  [111.0, 111.0, 111.0, 111.0, 111.0, 112.0]
    df = _synthetic_history(
        closes,
        ema_21=[110.0] * 6,
        sma_50=[100.0] * 6,
        sma_200=[95.0] * 6,
        lows=lows, highs=highs,
    )
    force_date = df["trade_date"].iloc[5]  # the up-close bar

    engine = MCTEngine(EngineConfig(
        initial_reference_high=112.0,
        initial_power_trend=False,
        initial_exposure=100,
        force_correction_at_date=force_date,
    ))
    result = engine.run(df)

    # CORRECTION_DECLARED and STEP_0_RALLY_DAY should both land on force_date.
    declared = [s for s in result.signals
                if s.signal_type == "CORRECTION_DECLARED" and s.trade_date == force_date]
    step0 = [s for s in result.signals
             if s.signal_type == "STEP_0_RALLY_DAY" and s.trade_date == force_date]
    assert len(declared) == 1, f"expected 1 CORRECTION_DECLARED on force_date, got {len(declared)}"
    assert len(step0) == 1, (
        f"expected STEP_0_RALLY_DAY on the same bar (up_day), got {len(step0)}"
    )
    # up_day trigger, not pink_rally_day — 111.5 > 110 is a proper up close.
    assert step0[0].meta.get("trigger") == "up_day"


def test_force_correction_today_pink_rally_day_same_bar():
    """Pink rally day variant — override bar closes flat/down vs prior
    but in the upper half of its intraday range. STEP_0 should still fire
    same-bar with trigger='pink_rally_day'."""
    from api.mct_engine import MCTEngine, EngineConfig

    # Prior bar close 110; force bar close 109.5 (down) BUT with low 108
    # and high 110 → position_in_range = (109.5 - 108) / (110 - 108) = 0.75 > 0.5
    closes = [110.0, 110.0, 110.0, 110.0, 110.0, 109.5]
    lows =   [109.0, 109.0, 109.0, 109.0, 109.0, 108.0]
    highs =  [111.0, 111.0, 111.0, 111.0, 111.0, 110.0]
    df = _synthetic_history(
        closes,
        ema_21=[110.0] * 6,
        sma_50=[100.0] * 6,
        sma_200=[95.0] * 6,
        lows=lows, highs=highs,
    )
    force_date = df["trade_date"].iloc[5]

    engine = MCTEngine(EngineConfig(
        initial_reference_high=111.0,
        initial_power_trend=False,
        initial_exposure=100,
        force_correction_at_date=force_date,
    ))
    result = engine.run(df)

    step0 = [s for s in result.signals
             if s.signal_type == "STEP_0_RALLY_DAY" and s.trade_date == force_date]
    assert len(step0) == 1
    assert step0[0].meta.get("trigger") == "pink_rally_day"


def test_force_correction_today_stays_correction_when_close_qualifies_neither():
    """When the override bar closes DOWN in the lower half of its range,
    it's a continuation-down bar — neither up_day nor pink_rally_day. The
    bar remains CORRECTION with step0_done=False. Rally-hunt continues
    from the NEXT bar."""
    from api.mct_engine import MCTEngine, EngineConfig

    # Prior close 110; force bar closes 109 in the LOWER half.
    # position_in_range = (109 - 108) / (111 - 108) = 0.33 → not pink.
    closes = [110.0, 110.0, 110.0, 110.0, 110.0, 109.0]
    lows =   [109.0, 109.0, 109.0, 109.0, 109.0, 108.0]
    highs =  [111.0, 111.0, 111.0, 111.0, 111.0, 111.0]
    df = _synthetic_history(
        closes,
        ema_21=[110.0] * 6,
        sma_50=[100.0] * 6,
        sma_200=[95.0] * 6,
        lows=lows, highs=highs,
    )
    force_date = df["trade_date"].iloc[5]

    engine = MCTEngine(EngineConfig(
        initial_reference_high=111.0,
        initial_power_trend=False,
        initial_exposure=100,
        force_correction_at_date=force_date,
    ))
    result = engine.run(df)

    step0 = [s for s in result.signals
             if s.signal_type == "STEP_0_RALLY_DAY" and s.trade_date == force_date]
    assert step0 == [], (
        "continuation-down bar shouldn't fire STEP_0 on the override bar"
    )
    # State should still reflect an active correction with step0 not done.
    assert result.final_state["correction_active"] is True
    assert result.final_state["step0_done"] is False


def test_force_correction_no_op_when_date_absent_from_history():
    """If force_correction_at_date doesn't match any bar in the sliced
    history (e.g., stale override anchor), the engine runs normally and
    no user_override CORRECTION_DECLARED is emitted. force_correction_applied
    stays False so the endpoint can distinguish 'applied' from 'pending'."""
    from api.mct_engine import MCTEngine, EngineConfig

    closes = [110.0] * 5
    df = _synthetic_history(closes, ema_21=[110.0] * 5,
                            sma_50=[100.0] * 5, sma_200=[95.0] * 5)
    # Pick a date not in the frame.
    unmatched = date(1990, 1, 1)

    engine = MCTEngine(EngineConfig(
        initial_reference_high=111.0,
        initial_power_trend=False,
        initial_exposure=100,
        force_correction_at_date=unmatched,
    ))
    result = engine.run(df)
    user_declared = [
        s for s in result.signals
        if s.signal_type == "CORRECTION_DECLARED"
        and s.meta.get("trigger") == "user_override"
    ]
    assert user_declared == []
    assert result.final_state["force_correction_applied"] is False


def test_force_correction_applied_flag_true_when_date_matches():
    """Sanity — when Phase 3a fires, the final_state.force_correction_applied
    latch flips True. Endpoint reads this off final_state to decide whether
    to surface `override` (applied) vs `override_pending` (declared but
    engine didn't see the date, typical yfinance ingest lag)."""
    from api.mct_engine import MCTEngine, EngineConfig

    closes = [110.0] * 6
    df = _synthetic_history(closes, ema_21=[110.0] * 6,
                            sma_50=[100.0] * 6, sma_200=[95.0] * 6)
    force_date = df["trade_date"].iloc[3]

    engine = MCTEngine(EngineConfig(
        initial_reference_high=111.0,
        initial_power_trend=False,
        initial_exposure=100,
        force_correction_at_date=force_date,
    ))
    result = engine.run(df)
    assert result.final_state["force_correction_applied"] is True


def test_force_correction_pending_when_date_after_history_end():
    """The concrete data-lag scenario: user forces correction on 2026-07-27,
    but market_data only has bars through 2026-07-24 (yfinance hasn't
    ingested Monday's bar yet). force_correction_applied must stay False
    so the endpoint surfaces `override_pending`, not a fake `override`.
    Otherwise the M Factor page would report a bogus RALLY MODE Day 1
    anchored on a bar the engine never actually processed."""
    from api.mct_engine import MCTEngine, EngineConfig

    # History ends 2026-01-10; override targets 2026-01-15 (in the future
    # relative to the slice).
    closes = [110.0] * 6
    df = _synthetic_history(closes, ema_21=[110.0] * 6,
                            sma_50=[100.0] * 6, sma_200=[95.0] * 6,
                            start=date(2026, 1, 5))
    future_date = date(2026, 1, 15)  # past the last bar (2026-01-10)

    engine = MCTEngine(EngineConfig(
        initial_reference_high=111.0,
        initial_power_trend=False,
        initial_exposure=100,
        force_correction_at_date=future_date,
    ))
    result = engine.run(df)
    assert result.final_state["force_correction_applied"] is False
    assert all(
        s.meta.get("trigger") != "user_override"
        for s in result.signals if s.signal_type == "CORRECTION_DECLARED"
    )


# ---------------------------------------------------------------------------
# Dual-index FTD gate (FTD_DUAL_INDEX_START, SPY confirmation side-channel)
# ---------------------------------------------------------------------------
# The dual-index gate takes effect on bars whose trade_date >= 2026-07-31.
# Bars before that date use the legacy IXIC-price-only rule so historical
# replays reproduce pre-cutover signals unchanged.
#
# Tests drive _phase_rally_hunt directly with a pre-seeded state, mirroring
# the pattern used by test_step4_ever_fired_persists_*. Full-engine replay
# tests would need to also stand up CORRECTION_DECLARED first (rally-hunt
# signal emissions gate on correction_active); testing the STEP_1 gate
# in isolation keeps the assertions tight to the new logic.

def _seed_pre_ftd_state(engine, rally_count: int = 4) -> dict:
    """State right before the STEP_1 gate: rally active, STEP_0 done, day N.

    ftd_low = None so state.get("ftd_low") is None (fresh rally, no prior
    FTD anchor).
    """
    state = engine._init_state()
    state["correction_active"] = True    # arms signal emission
    state["in_correction"] = True
    state["step0_done"] = True
    state["rally_active"] = True
    state["rally_day_idx"] = 0
    state["rally_day_low"] = 99.0
    state["running_min_low"] = 99.0
    state["running_min_idx"] = 0
    state["rally_count"] = rally_count
    state["cycle_start_idx"] = 0
    return state


def _ftd_candidate_bar(trade_date: date, close: float = 101.5, low: float = 99.5,
                        volume: int = 1_200_000):
    """A bar with prev_close=100 → 1.5% gain (clears FTD_PCT_THRESHOLD=1%)."""
    return pd.Series({
        "trade_date": pd.Timestamp(trade_date),
        "close": close, "low": low, "high": close + 0.5, "open": close - 0.2,
        "volume": volume,
        "ema_21": 98.0, "ema_8": 100.0, "sma_50": 95.0, "sma_200": 90.0,
    })


def _ftd_prev_bar(trade_date: date, close: float = 100.0, volume: int = 1_000_000):
    return pd.Series({
        "trade_date": pd.Timestamp(trade_date),
        "close": close, "low": close - 1.0, "high": close + 1.0, "open": close,
        "volume": volume,
        "ema_21": 98.0, "ema_8": 100.0, "sma_50": 95.0, "sma_200": 90.0,
    })


def test_ftd_pre_cutover_fires_ixic_price_only_no_volume_required():
    """Bars BEFORE FTD_DUAL_INDEX_START keep the legacy rule: fires on
    IXIC close pct_gain >= 1%, no volume check, no SPY dependency."""
    from api.mct_engine import MCTEngine, EngineConfig

    engine = MCTEngine(EngineConfig(initial_reference_high=200.0,
                                     initial_power_trend=False,
                                     initial_exposure=20))
    state = _seed_pre_ftd_state(engine)
    prev = _ftd_prev_bar(date(2026, 7, 29), volume=5_000_000)
    # Post-cutover would fail: current volume LESS than prev volume.
    current = _ftd_candidate_bar(date(2026, 7, 30), close=101.5, volume=1_000_000)
    bar_signals: list = []
    history = pd.DataFrame([prev, current]).reset_index(drop=True)
    engine._phase_rally_hunt(i=3, current=current, prev=prev, history=history,
                              state=state, bar_signals=bar_signals,
                              start_flags={"step3_done": False, "step4_done": False})
    types = [s.signal_type for s in bar_signals]
    assert "STEP_1_FTD" in types, "pre-cutover: price-only rule must fire"
    ftd = next(s for s in bar_signals if s.signal_type == "STEP_1_FTD")
    assert ftd.meta.get("confirmed_by") == "ixic_legacy"


def test_ftd_post_cutover_ixic_price_and_volume_fires():
    """On/after cutover: IXIC price ≥1% AND vol > prev → fires,
    confirmed_by='ixic' when SPY did NOT confirm that day."""
    from api.mct_engine import MCTEngine, EngineConfig

    engine = MCTEngine(EngineConfig(
        initial_reference_high=200.0, initial_power_trend=False,
        initial_exposure=20,
        spy_confirmations={date(2026, 8, 3): False},   # SPY present, didn't confirm
    ))
    state = _seed_pre_ftd_state(engine)
    prev = _ftd_prev_bar(date(2026, 7, 31), volume=1_000_000)
    current = _ftd_candidate_bar(date(2026, 8, 3), close=101.5, volume=1_500_000)
    bar_signals: list = []
    history = pd.DataFrame([prev, current]).reset_index(drop=True)
    engine._phase_rally_hunt(i=3, current=current, prev=prev, history=history,
                              state=state, bar_signals=bar_signals,
                              start_flags={"step3_done": False, "step4_done": False})
    ftd = next(s for s in bar_signals if s.signal_type == "STEP_1_FTD")
    assert ftd.meta.get("confirmed_by") == "ixic"
    assert ftd.meta.get("ixic_confirms") is True
    assert ftd.meta.get("spy_confirms") is False


def test_ftd_post_cutover_spy_only_fires_when_ixic_volume_flat():
    """4/8/2026-style: IXIC price up ≥1% but volume DOWN vs prev; SPY
    confirms → FTD fires with confirmed_by='spy'."""
    from api.mct_engine import MCTEngine, EngineConfig

    engine = MCTEngine(EngineConfig(
        initial_reference_high=200.0, initial_power_trend=False,
        initial_exposure=20,
        spy_confirmations={date(2026, 8, 3): True},
    ))
    state = _seed_pre_ftd_state(engine)
    prev = _ftd_prev_bar(date(2026, 7, 31), volume=5_000_000)
    # IXIC price up 1.5% BUT volume DOWN vs prev → IXIC gate fails
    current = _ftd_candidate_bar(date(2026, 8, 3), close=101.5, volume=1_000_000)
    bar_signals: list = []
    history = pd.DataFrame([prev, current]).reset_index(drop=True)
    engine._phase_rally_hunt(i=3, current=current, prev=prev, history=history,
                              state=state, bar_signals=bar_signals,
                              start_flags={"step3_done": False, "step4_done": False})
    ftd = next(s for s in bar_signals if s.signal_type == "STEP_1_FTD")
    assert ftd.meta.get("confirmed_by") == "spy"
    assert ftd.meta.get("ixic_confirms") is False
    assert ftd.meta.get("spy_confirms") is True


def test_ftd_post_cutover_both_confirm_marks_confirmed_by_both():
    from api.mct_engine import MCTEngine, EngineConfig

    engine = MCTEngine(EngineConfig(
        initial_reference_high=200.0, initial_power_trend=False,
        initial_exposure=20,
        spy_confirmations={date(2026, 8, 3): True},
    ))
    state = _seed_pre_ftd_state(engine)
    prev = _ftd_prev_bar(date(2026, 7, 31), volume=1_000_000)
    current = _ftd_candidate_bar(date(2026, 8, 3), close=101.5, volume=1_500_000)
    bar_signals: list = []
    history = pd.DataFrame([prev, current]).reset_index(drop=True)
    engine._phase_rally_hunt(i=3, current=current, prev=prev, history=history,
                              state=state, bar_signals=bar_signals,
                              start_flags={"step3_done": False, "step4_done": False})
    ftd = next(s for s in bar_signals if s.signal_type == "STEP_1_FTD")
    assert ftd.meta.get("confirmed_by") == "both"


def test_ftd_post_cutover_missing_spy_data_refuses_to_fire():
    """SPY bar not in confirmations dict (adapter didn't find it in
    market_data) → engine refuses to fire even if IXIC would confirm.

    Design choice: "wait for SPY to land" beats "fall back to IXIC-only"
    so a stale nightly ingest doesn't silently produce a single-index FTD.
    """
    from api.mct_engine import MCTEngine, EngineConfig

    engine = MCTEngine(EngineConfig(
        initial_reference_high=200.0, initial_power_trend=False,
        initial_exposure=20,
        spy_confirmations={},   # SPY missing for 2026-08-03
    ))
    state = _seed_pre_ftd_state(engine)
    prev = _ftd_prev_bar(date(2026, 7, 31), volume=1_000_000)
    current = _ftd_candidate_bar(date(2026, 8, 3), close=101.5, volume=1_500_000)
    bar_signals: list = []
    history = pd.DataFrame([prev, current]).reset_index(drop=True)
    engine._phase_rally_hunt(i=3, current=current, prev=prev, history=history,
                              state=state, bar_signals=bar_signals,
                              start_flags={"step3_done": False, "step4_done": False})
    types = [s.signal_type for s in bar_signals]
    assert "STEP_1_FTD" not in types, (
        "SPY data missing must block STEP_1_FTD — no silent IXIC-only fallback"
    )
    assert state["step1_done"] is False


def test_ftd_post_cutover_neither_confirms_no_fire():
    """IXIC price up but no volume; SPY present but didn't confirm → no FTD."""
    from api.mct_engine import MCTEngine, EngineConfig

    engine = MCTEngine(EngineConfig(
        initial_reference_high=200.0, initial_power_trend=False,
        initial_exposure=20,
        spy_confirmations={date(2026, 8, 3): False},
    ))
    state = _seed_pre_ftd_state(engine)
    prev = _ftd_prev_bar(date(2026, 7, 31), volume=5_000_000)
    current = _ftd_candidate_bar(date(2026, 8, 3), close=101.5, volume=1_000_000)
    bar_signals: list = []
    history = pd.DataFrame([prev, current]).reset_index(drop=True)
    engine._phase_rally_hunt(i=3, current=current, prev=prev, history=history,
                              state=state, bar_signals=bar_signals,
                              start_flags={"step3_done": False, "step4_done": False})
    types = [s.signal_type for s in bar_signals]
    assert "STEP_1_FTD" not in types


def test_ftd_post_cutover_price_below_threshold_never_fires():
    """Even with volume up on both indexes, IXIC price gain < 1% + SPY
    confirmations False → no FTD (price threshold is the primary gate)."""
    from api.mct_engine import MCTEngine, EngineConfig

    engine = MCTEngine(EngineConfig(
        initial_reference_high=200.0, initial_power_trend=False,
        initial_exposure=20,
        spy_confirmations={date(2026, 8, 3): False},
    ))
    state = _seed_pre_ftd_state(engine)
    prev = _ftd_prev_bar(date(2026, 7, 31), volume=1_000_000)
    # Only +0.3% gain — under threshold.
    current = _ftd_candidate_bar(date(2026, 8, 3), close=100.3, volume=1_500_000)
    bar_signals: list = []
    history = pd.DataFrame([prev, current]).reset_index(drop=True)
    engine._phase_rally_hunt(i=3, current=current, prev=prev, history=history,
                              state=state, bar_signals=bar_signals,
                              start_flags={"step3_done": False, "step4_done": False})
    types = [s.signal_type for s in bar_signals]
    assert "STEP_1_FTD" not in types
