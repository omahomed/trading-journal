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
# BE regression (from the spec, adapted to 2026-08-13 SR8 doctrine):
#   activation 4/29 NAV=$430,249, core=224 shs
#   Quick fires 6/26, NAV=$805,679 (up 87%), price ~$288
#   Post-2026-08-13 (5% Quick target):
#     ANCHORED : 5% × 430249 / 288 ≈  74 shs target → trim 224−74 = 150 shs
#     LIVE-NAV : 5% × 805679 / 288 ≈ 139 shs target → trim 224−139 =  85 shs
#   The critical property: anchored trims MORE than live-nav (the fix's
#   whole point). Under the old 10% doctrine live-nav gave 279 target
#   → 0 trim; under 5% the fallback trims a smaller-but-nonzero count.
# ─────────────────────────────────────────────────────────────────────────

def test_be_case_anchored_target_produces_valid_trim():
    """The BE spec fixture under the 2026-08-13 5% Quick doctrine.
    Verifies the anchored target drives a substantial trim."""
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

    # target_dollars = activation_nlv × 5% = $21,512.45
    assert r["target_dollars"] == pytest.approx(21_512.45, abs=0.01)
    # delta_dollars = held$ − target$ = 224*288 − 21512.45 = 42,999.55
    assert r["delta_dollars"] == pytest.approx(42_999.55, abs=0.01)
    # delta_shares ≈ 42999.55 / 288 = 149.30 (round to 149 or 150)
    assert r["delta_shares"] in (149, 150)
    # Anchor source badge
    assert r["anchor_source"] == "activation"
    assert r["activation_nlv"] == activation_nlv


def test_be_case_live_nav_fallback_still_trims_less_than_anchored():
    """When activation_nlv is NOT provided, the formula falls back to live
    NAV — reproducing the pre-fix behavior. Under the 2026-08-13 5% Quick
    target the fallback still fires a trim (unlike the old 10% doctrine
    where target > held → silent no-op), but its trim quantity is smaller
    than the anchored answer. The important properties: anchor_source=
    'live_fallback' surfaces the flag, and the fallback trims LESS than
    the anchored path — showing the anchor's teeth even under smaller
    cascade destinations."""
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
    activation_nlv = 430_249.0

    with patch("mors.monitor.run", return_value=_mock_backtest_result(
        current_tier="QUICK", current_price=current_price, log_signal="QUICK",
    )):
        r_fallback = analyze(pos, nlv=live_nlv, refresh=False, activation_nlv=None)
        r_anchored = analyze(pos, nlv=live_nlv, refresh=False,
                             activation_nlv=activation_nlv)

    # Live-NAV target = 805679 × 5% = $40,283.95 → 139 shs at $288.
    assert r_fallback["target_dollars"] == pytest.approx(40_283.95, abs=0.01)
    # Held = 224 → delta = 224*288 − 40283.95 = 24,228.05
    assert r_fallback["delta_dollars"] == pytest.approx(24_228.05, abs=0.01)
    # delta_shares ≈ 84 (24228.05 / 288)
    assert r_fallback["delta_shares"] in (84, 85)
    assert r_fallback["anchor_source"] == "live_fallback"

    # The fix's teeth: anchored trims MORE than the live-NAV fallback,
    # correcting the under-defended core.
    assert r_anchored["delta_dollars"] > r_fallback["delta_dollars"]
    assert r_anchored["delta_shares"] > r_fallback["delta_shares"]


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


def test_quicksand_uses_2_5_pct_of_activation_nlv():
    """Same anchor logic for Quicksand — the 2.5% NAV target destination
    under the 2026-08-13 cascade doctrine."""
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

    # target = 430249 × 2.5% = $10,756.225
    assert r["target_dollars"] == pytest.approx(10_756.225, abs=0.01)
    # held$ = 149 × 288 = 42,912 → delta = 42912 − 10756.225 = 32,155.775
    assert r["delta_dollars"] == pytest.approx(32_155.775, abs=0.01)
    # delta_shares ≈ 111.65 (32155.775 / 288) → 112 rounded
    assert r["delta_shares"] in (111, 112)


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
