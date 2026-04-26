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
    (date(2025, 4, 11), "STEP_1_FTD"),
    (date(2025, 4, 16), "POST_FTD_SOFT_FAIL"),
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
    (date(2026, 3, 4), "STEP_1_FTD"),
    (date(2026, 3, 5), "POST_FTD_SOFT_FAIL"),
    (date(2026, 3, 9), "STEP_1_FTD"),
    (date(2026, 4, 8), "STEP_1_FTD"),
    (date(2026, 4, 10), "STEP_4_LOW_ABOVE_21EMA_3BARS"),
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
    """Loose-bound sanity check on total signal count.

    Pinned to 100–180 based on the V11 first-implementation run (133 signals).
    Adjust the upper bound as the engine is iterated to remove noise; tighten
    to an exact equality once the canonical V11 reference is locked.
    """
    _, result = canonical_run
    assert 100 <= len(result.signals) <= 180, (
        f"Got {len(result.signals)} signals; expected 100–180"
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
def test_powertrend_off_then_on(canonical_run):
    """PT-OFF on 2/5/2026, then PT-ON on 4/22/2026 — both should fire."""
    _, result = canonical_run
    types = {(s.trade_date, s.signal_type) for s in result.signals}
    assert (date(2026, 2, 5), "POWERTREND_OFF") in types
    assert (date(2026, 4, 22), "STEP_8_POWERTREND_ON") in types


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
