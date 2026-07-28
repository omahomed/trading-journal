"""Regression: MCT stamp paths must apply the active Force Correction
override so trading_journal rows and Journal Log's MCT State badge
match what /api/market/rally-prefix (and the M Factor page) render.

Bug the tests lock: when the user declared a Force Correction via
the M Factor override UI, the M Factor page correctly showed
CORRECTION but the Daily Journal / Journal Log row for the same day
kept stamping the systematic "UPTREND UNDER PRESSURE" state. Cause:
_compute_mct_state_with_day_num, _compute_trend_count, and
_heal_recent_mct_stamps each called run_engine WITHOUT passing
force_correction_at_date — the override was only wired into
rally_prefix. Every save-time or heal-time stamper silently used
the systematic engine and clobbered the visible override on the
next Journal Log load.

The fix: a single _current_override_date() read-only helper is
plumbed into every run_engine call in the stamp/heal chain. These
tests verify (a) the helper returns the override date when active,
None when not; (b) the stampers pass it through to run_engine.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest


def test_current_override_date_returns_none_when_no_override(monkeypatch):
    """No active override → None. Stamp path then runs the systematic
    engine exactly as it did before the override existed."""
    import api.main as main
    import db_layer
    monkeypatch.setattr(db_layer, "get_active_mct_override", lambda: None)
    assert main._current_override_date() is None


def test_current_override_date_returns_parsed_date_when_active(monkeypatch):
    """Active override → the CT-date portion of activated_date_ct as a
    datetime.date, ready to hand to EngineConfig.force_correction_at_date."""
    import api.main as main
    import db_layer
    monkeypatch.setattr(db_layer, "get_active_mct_override", lambda: {
        "activated_date_ct": "2026-07-27",
        "reason": "manual — pre-IBD 10% threshold",
    })
    assert main._current_override_date() == date(2026, 7, 27)


def test_current_override_date_swallows_lookup_failures(monkeypatch, capsys):
    """DB hiccup during the override lookup must NOT break the stamp
    path — every caller is best-effort and returning None here means
    "fall back to systematic engine," matching pre-override behavior."""
    import api.main as main
    import db_layer
    def boom():
        raise RuntimeError("connection reset")
    monkeypatch.setattr(db_layer, "get_active_mct_override", boom)
    assert main._current_override_date() is None
    # Logged for debuggability, but the error is contained.
    assert "lookup failed" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _compute_mct_state_with_day_num — override plumbing through the save-time
# MCT stamper. Verifies the arg reaches run_engine; the engine itself has
# its own coverage (see tests/test_mct_engine.py for the Phase 3a hook).
# ---------------------------------------------------------------------------


def _fake_engine_result(state_name="POWERTREND", trend_count=None):
    """Minimal EngineResult stand-in — one bar with the fields the
    stamper reads. Whatever state we hand it flows through unchanged so
    the assertion focuses on 'was force_correction_at_date passed?'
    without depending on the engine's internal logic."""
    from api.mct_engine import EngineResult
    bars = pd.DataFrame([{
        "trade_date": pd.Timestamp("2026-07-27"),
        "state": state_name,
        "cycle_start_idx": 0,
        "pt_on_idx": 0,
        "rally_active": False,
    }])
    return EngineResult(bars=bars, signals=[], final_state={})


def test_mct_stamp_passes_override_date_to_engine(monkeypatch):
    """Live regression: with an active override, the MCT stamper must
    invoke run_engine with force_correction_at_date=<override_date>.
    Otherwise Journal Log's stamped state diverges from M Factor."""
    import api.main as main
    import db_layer
    monkeypatch.setattr(db_layer, "get_active_mct_override", lambda: {
        "activated_date_ct": "2026-07-27",
        "reason": "manual",
    })

    called_with = {}
    def fake_run_engine(symbol="^IXIC", as_of=None, force_correction_at_date=None):
        called_with["force_correction_at_date"] = force_correction_at_date
        called_with["as_of"] = as_of
        return _fake_engine_result()
    monkeypatch.setattr("api.mct_endpoint_adapter.run_engine", fake_run_engine)
    monkeypatch.setattr("api.market_data_updater.update_if_needed",
                        lambda symbol="^IXIC": None)

    main._compute_mct_state_with_day_num("2026-07-27")
    assert called_with["force_correction_at_date"] == date(2026, 7, 27)


def test_mct_stamp_omits_override_when_none_active(monkeypatch):
    """No active override → force_correction_at_date=None. Confirms the
    stamper doesn't accidentally leak a stale override date from an
    earlier request into a fresh systematic run."""
    import api.main as main
    import db_layer
    monkeypatch.setattr(db_layer, "get_active_mct_override", lambda: None)

    called_with = {}
    def fake_run_engine(symbol="^IXIC", as_of=None, force_correction_at_date=None):
        called_with["force_correction_at_date"] = force_correction_at_date
        return _fake_engine_result()
    monkeypatch.setattr("api.mct_endpoint_adapter.run_engine", fake_run_engine)
    monkeypatch.setattr("api.market_data_updater.update_if_needed",
                        lambda symbol="^IXIC": None)

    main._compute_mct_state_with_day_num("2026-07-27")
    assert called_with["force_correction_at_date"] is None


def test_trend_count_stamp_passes_override_date_to_engine(monkeypatch):
    """Trend count stamper (mirror of the MCT stamper) must apply the
    override too — otherwise Journal Log's Trend column reads the
    systematic count while M Factor shows the reset-on-override count."""
    import api.main as main
    import db_layer
    monkeypatch.setattr(db_layer, "get_active_mct_override", lambda: {
        "activated_date_ct": "2026-07-27",
        "reason": "manual",
    })

    called_with = {}
    def fake_run_engine(symbol="^IXIC", as_of=None, force_correction_at_date=None):
        called_with["force_correction_at_date"] = force_correction_at_date
        return _fake_engine_result()
    monkeypatch.setattr("api.mct_endpoint_adapter.run_engine", fake_run_engine)
    # to_rally_prefix_response is called next; return a minimal payload so the
    # stamper doesn't traceback while extracting trend_count.
    monkeypatch.setattr("api.mct_endpoint_adapter.to_rally_prefix_response",
                        lambda result: {"trend_count": None})
    monkeypatch.setattr("api.market_data_updater.update_if_needed",
                        lambda symbol="^IXIC": None)

    main._compute_trend_count("2026-07-27")
    assert called_with["force_correction_at_date"] == date(2026, 7, 27)
