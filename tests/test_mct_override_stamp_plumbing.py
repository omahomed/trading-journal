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


# ---------------------------------------------------------------------------
# _heal_recent_mct_stamps — STRICT NULL-ONLY (immutability contract)
# ---------------------------------------------------------------------------
#
# Regression 2026-07-30: a prior "re-stamp on stale" variant silently
# overwrote historical stamps (07-27 CORRECTION, 07-28 RALLY MODE Day 1)
# after a Force Correction override auto-cleared on 07-29. Journal Log
# stamps became mutable — the exact opposite of what an audit trail
# should be. Reverted to NULL-only; these tests lock the immutability
# contract so it can't erode again.


def test_heal_never_overwrites_a_stamped_row(monkeypatch):
    """The immutability contract: any row with a non-NULL market_cycle
    stamp is left alone forever. Doesn't matter if the engine's current
    output disagrees (override activated / cleared, rule tweak, market
    data revision) — the historical stamp represents what the engine
    said at save time, and that's the audit trail's job."""
    import api.main as main
    import db_layer

    # Override active — under the OLD "re-stamp on stale" heal, this
    # would rewrite the stored UUP stamp to CORRECTION. Under the new
    # NULL-only heal, the stamp is left alone.
    monkeypatch.setattr(db_layer, "get_active_mct_override", lambda: {
        "activated_date_ct": "2026-07-27",
        "reason": "manual",
    })

    from api.mct_engine import EngineResult
    engine_bars = pd.DataFrame([{
        "trade_date": pd.Timestamp("2026-07-27"),
        "state": "CORRECTION",     # engine says one thing
        "cycle_start_idx": pd.NA,
        "pt_on_idx": pd.NA,
        "rally_active": False,
    }])
    monkeypatch.setattr(
        "api.mct_endpoint_adapter.run_engine",
        lambda symbol="^IXIC", as_of=None, force_correction_at_date=None:
            EngineResult(bars=engine_bars, signals=[], final_state={}),
    )
    monkeypatch.setattr("api.market_data_updater.update_if_needed",
                        lambda symbol="^IXIC": None)

    writes = []
    monkeypatch.setattr(db_layer, "update_journal_mct_state",
                        lambda *a, **kw: (writes.append(("mct",) + a), 1)[1])
    monkeypatch.setattr(db_layer, "update_journal_trend_state",
                        lambda *a, **kw: (writes.append(("trend",) + a), 1)[1])

    # Row already stamped with a DIFFERENT state (from an earlier save).
    df = pd.DataFrame([{
        "day": pd.Timestamp("2026-07-27"),
        "market_cycle": "UPTREND UNDER PRESSURE",   # stored says another thing
        "mct_display_day_num": 81,
        "trend_count": -33,
    }])

    main._heal_recent_mct_stamps("CanSlim", df)

    assert writes == [], (
        "heal wrote to a non-NULL row — immutability contract broken. "
        "Historical stamps must survive engine-output changes."
    )


def test_heal_fills_null_state_when_engine_has_the_bar(monkeypatch):
    """NULL-heal is still legitimate: a row saved before market_data
    ingested the bar has NULL stamps; when the bar shows up later,
    heal fills them in. Only NULL columns are touched."""
    import api.main as main
    import db_layer

    monkeypatch.setattr(db_layer, "get_active_mct_override", lambda: None)

    from api.mct_engine import EngineResult
    engine_bars = pd.DataFrame([{
        "trade_date": pd.Timestamp("2026-07-27"),
        "state": "UPTREND UNDER PRESSURE",
        "cycle_start_idx": pd.NA,
        "pt_on_idx": pd.NA,
        "rally_active": False,
    }])
    monkeypatch.setattr(
        "api.mct_endpoint_adapter.run_engine",
        lambda symbol="^IXIC", as_of=None, force_correction_at_date=None:
            EngineResult(bars=engine_bars, signals=[], final_state={}),
    )
    monkeypatch.setattr("api.market_data_updater.update_if_needed",
                        lambda symbol="^IXIC": None)

    # _compute_mct_state_with_day_num is the real code path the heal
    # calls — stub the engine underneath so it returns UUP.
    captured_mct = {}
    def fake_update_mct(portfolio, day_str, state, day_num):
        captured_mct.update({"day": day_str, "state": state, "day_num": day_num})
        return 1
    monkeypatch.setattr(db_layer, "update_journal_mct_state", fake_update_mct)

    captured_trend = {}
    def fake_update_trend(portfolio, day_str, trend):
        captured_trend.update({"day": day_str, "trend": trend})
        return 1
    monkeypatch.setattr(db_layer, "update_journal_trend_state", fake_update_trend)
    # _compute_trend_count is called by the heal for trend-NULL rows.
    monkeypatch.setattr(main, "_compute_trend_count", lambda day_str: -35)

    # Row saved with NULL MCT stamps (typical when market_data hadn't
    # caught up at save time).
    df = pd.DataFrame([{
        "day": pd.Timestamp("2026-07-27"),
        "market_cycle": None,          # NULL — heal fills
        "mct_display_day_num": None,   # NULL — heal fills
        "trend_count": None,           # NULL — heal fills
    }])

    main._heal_recent_mct_stamps("CanSlim", df)

    assert captured_mct.get("state") == "UPTREND UNDER PRESSURE"
    assert captured_trend.get("trend") == -35


# ---------------------------------------------------------------------------
# DB-layer immutability guard — belt-and-suspenders
# ---------------------------------------------------------------------------
#
# Even if a caller (or an old deployed version of the heal) tries to
# overwrite a non-NULL MCT stamp, the DB write function itself refuses
# unless force=True is passed explicitly. Only the user-triggered
# restamp_mct endpoint uses force=True; every other caller defaults to
# NULL-only.


def test_update_journal_mct_state_default_signature_is_null_only():
    """The write function's default (force=False) must not overwrite a
    stamped value. Grep-based check on the SQL — the WHERE clause must
    include a NULL-or-empty guard on market_cycle when force is False."""
    import re
    src = (_REPO_ROOT / "db_layer.py").read_text()
    # Slice update_journal_mct_state body
    m = re.search(
        r"def update_journal_mct_state\(.*?\n(.*?)(?=\ndef )",
        src, re.DOTALL,
    )
    assert m, "update_journal_mct_state not found in db_layer.py"
    body = m.group(0)

    # Signature must accept force kwarg with default False.
    assert "force: bool = False" in body, (
        "update_journal_mct_state no longer has the force kwarg — the "
        "immutability guard depends on it. Default must be False."
    )
    # The default (non-force) SQL must have the NULL-or-empty guard.
    assert "market_cycle IS NULL OR market_cycle = ''" in body, (
        "the default (force=False) UPDATE SQL must include the NULL-or-"
        "empty guard on market_cycle. Otherwise any caller can overwrite "
        "a stamped row silently — the historical incident this test "
        "exists to prevent."
    )


def test_update_journal_trend_state_default_signature_is_null_only():
    """Mirror check for trend_count."""
    import re
    src = (_REPO_ROOT / "db_layer.py").read_text()
    m = re.search(
        r"def update_journal_trend_state\(.*?\n(.*?)(?=\ndef |\Z)",
        src, re.DOTALL,
    )
    assert m, "update_journal_trend_state not found in db_layer.py"
    body = m.group(0)

    assert "force: bool = False" in body, (
        "update_journal_trend_state must accept force kwarg (default False)"
    )
    assert "trend_count IS NULL" in body, (
        "the default (force=False) UPDATE SQL must include a "
        "trend_count IS NULL guard"
    )


# Ensure the repo-root path (imported at top of file via _REPO_ROOT) is
# consistent with the pattern used by other contract tests.
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent


def test_heal_leaves_partial_row_alone_when_only_trend_is_null(monkeypatch):
    """If MCT state is stamped but trend_count is NULL, heal fills
    ONLY the trend_count. Never touches the already-stamped MCT."""
    import api.main as main
    import db_layer

    monkeypatch.setattr(db_layer, "get_active_mct_override", lambda: None)

    from api.mct_engine import EngineResult
    engine_bars = pd.DataFrame([{
        "trade_date": pd.Timestamp("2026-07-27"),
        "state": "CORRECTION",   # different from stored, but stored isn't NULL
        "cycle_start_idx": pd.NA,
        "pt_on_idx": pd.NA,
        "rally_active": False,
    }])
    monkeypatch.setattr(
        "api.mct_endpoint_adapter.run_engine",
        lambda symbol="^IXIC", as_of=None, force_correction_at_date=None:
            EngineResult(bars=engine_bars, signals=[], final_state={}),
    )
    monkeypatch.setattr("api.market_data_updater.update_if_needed",
                        lambda symbol="^IXIC": None)

    mct_writes = []
    monkeypatch.setattr(db_layer, "update_journal_mct_state",
                        lambda *a, **kw: (mct_writes.append(a), 1)[1])
    trend_writes = []
    monkeypatch.setattr(db_layer, "update_journal_trend_state",
                        lambda *a, **kw: (trend_writes.append(a), 1)[1])
    monkeypatch.setattr(main, "_compute_trend_count", lambda day_str: -33)

    df = pd.DataFrame([{
        "day": pd.Timestamp("2026-07-27"),
        "market_cycle": "UPTREND UNDER PRESSURE",  # stamped — leave alone
        "mct_display_day_num": 81,                 # stamped — leave alone
        "trend_count": None,                       # NULL — heal fills
    }])

    main._heal_recent_mct_stamps("CanSlim", df)

    assert mct_writes == [], "MCT state was stamped — must not be touched"
    assert len(trend_writes) == 1, "trend_count was NULL — should be filled"


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
