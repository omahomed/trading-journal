"""Regression tests for the SR8 activation-anchor fix (2026-07-18).

Before the fix, monitor.analyze()'s Quick/QS target_dollars used
live NLV — so a position whose NAV grew past its activation NAV
got trim targets in shares that exceeded held shares → the trim
signal read as "already at target" → cores went undefended on
valid signals.

The fix anchors target_dollars to sr8_activation_nlv passed by the
API caller. Live NLV remains only for the display-only
current_pct_nlv metric.

Tests here mock mors.mors_backtest.run so we exercise ONLY the
target-anchoring math (no yfinance, no CSV, no cascade replay).
"""
from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest


def _mock_backtest_result(current_tier: str, current_price: float, log_signal: str):
    """Shape the return dict of mors_backtest.run() the way monitor.analyze()
    consumes it. Just enough to drive the tier + price + last-signal logic."""
    log_df = pd.DataFrame([
        {"Date": pd.Timestamp("2026-06-26"), "Signal": log_signal, "Phase": 2},
    ])
    return {
        "log": log_df,
        "exit_px": current_price,
        "exit_date": pd.Timestamp("2026-06-26"),
        "current_tier_label": current_tier,
    }


# ─────────────────────────────────────────────────────────────────────────
# BE regression (from the spec):
#   activation 4/29 NAV=$430,249, core=224 shs
#   Quick fires 6/26, NAV=$805,679 (up 87%), price ~$288
#   OLD formula: 10% × 805679 / 288 ≈ 279 shs target → 224 held → no trim
#   NEW formula: 10% × 430249 / 288 ≈ 149 shs target → 75 shs trim ✓
# ─────────────────────────────────────────────────────────────────────────

def test_be_case_anchored_target_produces_valid_trim():
    """The exact case the spec cited. Verifies the fix produces a
    non-zero trim on a Quick signal that the pre-fix formula silently
    ignored."""
    from mors.monitor import analyze

    pos = {
        "ticker": "BE",
        "b1_date": "2026-01-15",
        "b1_price": 40.0,
        "shares_held": 224,
        "avg_price": 60.0,
    }
    live_nlv = 805_679.0
    activation_nlv = 430_249.0
    current_price = 288.0

    with patch("mors.monitor.run", return_value=_mock_backtest_result(
        current_tier="QUICK", current_price=current_price, log_signal="QUICK",
    )):
        r = analyze(pos, nlv=live_nlv, refresh=False, activation_nlv=activation_nlv)

    # target_dollars = activation_nlv × 10% = $43,024.90
    assert r["target_dollars"] == pytest.approx(43_024.9, abs=0.01)
    # delta_dollars = held$ − target$ = 224*288 − 43024.9 = 64,512 − 43,024.9 = 21,487
    assert r["delta_dollars"] > 20_000
    # delta_shares ≈ 21487 / 288 = 74.6 → rounds to 75 (or 74/75 in floor/round)
    assert r["delta_shares"] in (74, 75)
    # Anchor source badge
    assert r["anchor_source"] == "activation"
    assert r["activation_nlv"] == activation_nlv


def test_be_case_live_nav_fallback_shows_the_bug_it_used_to_hide():
    """When activation_nlv is NOT provided, the formula falls back to live
    NAV — reproducing the pre-fix behavior. This test locks in the fallback
    contract: fallback rows are visibly worse (target > held → 0 trim) and
    labeled anchor_source='live_fallback' so operators can spot legacy
    positions still in need of backfill."""
    from mors.monitor import analyze

    pos = {
        "ticker": "BE",
        "b1_date": "2026-01-15",
        "b1_price": 40.0,
        "shares_held": 224,
        "avg_price": 60.0,
    }
    live_nlv = 805_679.0
    current_price = 288.0

    with patch("mors.monitor.run", return_value=_mock_backtest_result(
        current_tier="QUICK", current_price=current_price, log_signal="QUICK",
    )):
        r = analyze(pos, nlv=live_nlv, refresh=False, activation_nlv=None)

    # Live-NAV target = 805679 × 10% = $80,567.90 → 279 shs at $288.
    # Held = 224 → target > held → held_value < target_dollars → delta_dollars = 0.
    assert r["target_dollars"] == pytest.approx(80_567.9, abs=0.01)
    # 224 × 288 = 64,512 < 80,567.90 → delta clamps at 0.
    assert r["delta_dollars"] == 0
    assert r["delta_shares"] == 0
    assert r["anchor_source"] == "live_fallback"


def test_mu_case_small_nav_drift_targets_close_to_live_answer():
    """Anti-regression from the spec: on a campaign where NAV barely
    moved between activation and today, the anchored target should be
    within a few shares of what live-NAV would have produced. Confirms
    the fix doesn't distort the calm-drift case."""
    from mors.monitor import analyze

    pos = {
        "ticker": "MU",
        "b1_date": "2026-01-01",
        "b1_price": 400.0,
        "shares_held": 116,
        "avg_price": 500.0,
    }
    activation_nlv = 551_423.0
    live_nlv = 553_000.0  # ~0.3% drift
    current_price = 900.0

    with patch("mors.monitor.run", return_value=_mock_backtest_result(
        current_tier="QUICK", current_price=current_price, log_signal="QUICK",
    )):
        r_anchored = analyze(pos, nlv=live_nlv, refresh=False, activation_nlv=activation_nlv)
        r_fallback = analyze(pos, nlv=live_nlv, refresh=False, activation_nlv=None)

    # Deltas differ by no more than a couple of shares — the small drift
    # doesn't distort trim quantities on this fixture.
    assert abs(r_anchored["delta_shares"] - r_fallback["delta_shares"]) <= 2


def test_quicksand_uses_5_pct_of_activation_nlv():
    """Same anchor logic for Quicksand — the 5% NAV target destination."""
    from mors.monitor import analyze

    pos = {
        "ticker": "BE",
        "b1_date": "2026-01-15",
        "b1_price": 40.0,
        "shares_held": 149,
        "avg_price": 200.0,
    }

    with patch("mors.monitor.run", return_value=_mock_backtest_result(
        current_tier="QUICKSAND", current_price=288.0, log_signal="QUICKSAND",
    )):
        r = analyze(pos, nlv=805_679.0, refresh=False, activation_nlv=430_249.0)

    # target = 430249 × 5% = $21,512.45
    assert r["target_dollars"] == pytest.approx(21_512.45, abs=0.01)
    # held$ = 149 × 288 = 42,912 → delta = 42912 − 21512.45 = 21,399.55
    assert r["delta_dollars"] == pytest.approx(21_399.55, abs=0.01)
    # delta_shares ≈ 74 (21399.55 / 288)
    assert r["delta_shares"] in (74, 75)


def test_grateful_dead_target_is_zero_regardless_of_anchor():
    """GD terminates the campaign — target is 0 regardless of anchor."""
    from mors.monitor import analyze

    pos = {
        "ticker": "BE",
        "b1_date": "2026-01-15",
        "b1_price": 40.0,
        "shares_held": 75,
        "avg_price": 200.0,
    }

    with patch("mors.monitor.run", return_value=_mock_backtest_result(
        current_tier="GD", current_price=200.0, log_signal="GD",
    )):
        r = analyze(pos, nlv=805_679.0, refresh=False, activation_nlv=430_249.0)

    assert r["target_dollars"] == 0.0
    assert r["terminated"] is True


def test_green_tier_never_sells_regardless_of_anchor():
    """GREEN is a REBUILD target, not a trim floor. Should return 0
    delta even if held << target."""
    from mors.monitor import analyze

    pos = {
        "ticker": "BE",
        "b1_date": "2026-01-15",
        "b1_price": 40.0,
        "shares_held": 200,
        "avg_price": 200.0,
    }

    with patch("mors.monitor.run", return_value=_mock_backtest_result(
        current_tier="GREEN", current_price=200.0, log_signal="GREEN",
    )):
        r = analyze(pos, nlv=805_679.0, refresh=False, activation_nlv=430_249.0)

    assert r["delta_dollars"] == 0.0
    assert r["delta_shares"] == 0
