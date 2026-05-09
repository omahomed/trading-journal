"""Regression tests for string-NaN persistence (Phase 2, Commit 1).

The bug: pandas Series .get() returns the cell value when the key exists,
not the default. So `row.get("rule", "")` for a row with rule=NaN returns
np.nan, not "". And np.nan is truthy under `or`, so `str(row.get("rule", "")
or "")` evaluates to `str(np.nan)` → the literal string "nan", which gets
written to the DB.

Producers fixed in this commit:
  S2 — log_buy scale-in (api/main.py:2819-2820)
  S4 — log_sell           (api/main.py:3041-3042)
  S5 — exercise_option stock-side scale-in (api/main.py:3406-3407)
  B1 — exercise_option option-side notes concat (api/main.py:3297)
  B2 — exercise_option stock-side notes concat (api/main.py:3398)
  S3 — set_trade_grade defensive cleanup (api/main.py:2909-2914) — re-anchor
       protection (not a producer, but blocked any future regression).

The fix: db_layer.clean_text_value normalizes None / NaN / empty strings /
literal sentinels ('nan'/'none'/'null', case-insensitive) to None so the
DB stores NULL instead of garbage strings.

These tests are the contract guard: any future change that bypasses
clean_text_value at one of the producer sites will surface here.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import jwt
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import db_layer


_TEST_SECRET = "test-secret-not-for-prod"
_TEST_USER_ID = "test-user"


def _auth_headers() -> dict[str, str]:
    token = jwt.encode({"sub": _TEST_USER_ID}, _TEST_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# D1: Unit tests on clean_text_value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("inp,expected", [
    # Null-likes
    (None,       None),
    (np.nan,     None),
    (pd.NA,      None),
    # Empty / whitespace
    ("",         None),
    ("   ",      None),
    ("\t\n ",    None),
    # Literal sentinels (case-insensitive, stripped)
    ("nan",      None),
    ("NaN",      None),
    ("NAN",      None),
    ("  nan  ",  None),
    ("none",     None),
    ("None",     None),
    ("NONE",     None),
    ("null",     None),
    ("Null",     None),
    ("NULL",     None),
    # Real values pass through stripped
    ("real value",     "real value"),
    ("  trimmed  ",    "trimmed"),
    # Numeric coercion (defensive — shouldn't reach the helper but must not crash)
    (123,              "123"),
    (45.67,            "45.67"),
    (0,                "0"),
    # The string "0" is a real value, not a sentinel
    ("0",              "0"),
])
def test_clean_text_value(inp, expected):
    assert db_layer.clean_text_value(inp) == expected


def test_clean_text_value_does_not_crash_on_unhashable():
    """pd.isna raises TypeError on lists/dicts; helper must swallow it and
    fall through to str-coercion + sentinel check."""
    # A list is truthy but pd.isna(list) raises. We expect the helper to
    # str-coerce it rather than crash.
    result = db_layer.clean_text_value([1, 2])
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# D2 + D3: Integration tests via FastAPI TestClient
# ---------------------------------------------------------------------------


@pytest.fixture
def buy_sell_stubs(monkeypatch):
    """Yield (state, client) for log_buy / log_sell tests.

    The fixture stubs db_layer reads/writes so the endpoints run without a
    DB. Each saved summary_row dict is captured in state["saved_summaries"]
    for inspection.
    """
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)

    import api.main as main

    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    state: dict[str, Any] = {
        "summary_df": pd.DataFrame(),
        "details_df": pd.DataFrame(),
        "saved_summaries": [],
        "saved_with_closures": [],
        "saved_details": [],
    }

    monkeypatch.setattr(db_layer, "load_summary",
                        lambda *a, **kw: state["summary_df"])
    monkeypatch.setattr(db_layer, "load_details",
                        lambda *a, **kw: state["details_df"])
    monkeypatch.setattr(main, "_normalize_trades", lambda df: df)
    monkeypatch.setattr(db_layer, "load_strategies",
                        lambda *a, **kw: [{"name": "CanSlim"}])

    def fake_save_summary_row(portfolio, row_dict):
        state["saved_summaries"].append({
            "portfolio": portfolio,
            "row": dict(row_dict),
        })
        return 1
    monkeypatch.setattr(db_layer, "save_summary_row", fake_save_summary_row)

    def fake_save_with_closures(portfolio, trade_id, summary_row, closures):
        state["saved_with_closures"].append({
            "portfolio": portfolio,
            "trade_id": trade_id,
            "summary_row": dict(summary_row),
        })
        return 1
    monkeypatch.setattr(db_layer, "save_summary_with_closures",
                        fake_save_with_closures)

    detail_counter = {"v": 0}
    def fake_save_detail(portfolio_name, row_dict):
        detail_counter["v"] += 1
        state["saved_details"].append(dict(row_dict))
        return detail_counter["v"]
    monkeypatch.setattr(db_layer, "save_detail_row", fake_save_detail)

    monkeypatch.setattr(db_layer, "generate_unique_trx_id",
                        lambda portfolio, trade_id, prefix: f"{prefix}1")
    monkeypatch.setattr(db_layer, "log_audit", lambda *a, **kw: None)

    # validate_post_edit_lifo isn't called by buy/sell, but stub for safety.
    monkeypatch.setattr(main, "validate_post_edit_lifo",
                        lambda *a, **kw: None)

    if hasattr(main.limiter, "enabled"):
        original_enabled = main.limiter.enabled
        main.limiter.enabled = False
    else:
        original_enabled = None

    client = TestClient(main.app, headers=_auth_headers())
    try:
        yield state, client
    finally:
        if original_enabled is not None:
            main.limiter.enabled = original_enabled


def _summary_row_with_nan() -> dict[str, Any]:
    """A trades_summary row whose user-entered text fields are np.nan.

    Snake-case keys match _normalize_trades output.
    """
    return {
        "trade_id": "202605-001",
        "ticker": "AAPL",
        "status": "OPEN",
        "open_date": pd.Timestamp("2026-05-01"),
        "closed_date": None,
        "shares": 100.0,
        "avg_entry": 200.0,
        "avg_exit": 0.0,
        "total_cost": 20000.0,
        "realized_pl": 0.0,
        "unrealized_pl": 0.0,
        "return_pct": 0.0,
        "rule": np.nan,        # the corruption-trigger
        "buy_notes": np.nan,   # the corruption-trigger
        "sell_rule": np.nan,
        "sell_notes": np.nan,
        "notes": np.nan,
        "risk_budget": 500.0,
        "stop_loss": 195.0,
        "instrument_type": "STOCK",
        "multiplier": 1.0,
        "strategy": "CanSlim",
    }


def test_log_buy_scale_in_with_nan_existing_row_writes_null_not_nan(buy_sell_stubs):
    """S2 producer guard.

    Setup: existing summary row has rule=NaN, buy_notes=NaN. POST a scale-in
    buy without rule/notes in the body. Pre-fix, the code did
        str(row.get("rule", "") or rule or "")
    which resolved to str(np.nan) = "nan" and persisted to DB. Post-fix,
    db.clean_text_value returns None and the DB stores NULL.
    """
    state, client = buy_sell_stubs
    state["summary_df"] = pd.DataFrame([_summary_row_with_nan()])

    r = client.post("/api/trades/buy", json={
        "portfolio": "CanSlim",
        "action_type": "scalein",
        "trade_id": "202605-001",
        "ticker": "AAPL",
        "shares": 50,
        "price": 210.0,
        "stop_loss": 195.0,
        # IMPORTANT: rule and notes intentionally omitted from body so the
        # code falls through to row.get(...) on the NaN-bearing row.
        "date": "2026-05-08",
        "time": "10:00",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert "error" not in body, body

    assert state["saved_summaries"], "Expected a save_summary_row call"
    saved = state["saved_summaries"][-1]["row"]

    assert saved.get("Rule") is None, \
        f"Rule should be None (NULL), got {saved.get('Rule')!r}"
    assert saved.get("Buy_Notes") is None, \
        f"Buy_Notes should be None (NULL), got {saved.get('Buy_Notes')!r}"
    # Belt + suspenders: explicitly assert the literal "nan" string never appears.
    assert saved.get("Rule") != "nan"
    assert saved.get("Buy_Notes") != "nan"


def test_log_buy_scale_in_with_real_body_overrides_nan_existing(buy_sell_stubs):
    """S2 behavior preserved: when the body supplies rule/notes, those win
    over the NaN existing row. Verifies the `or` fallback chain still works
    after the helper rewrite."""
    state, client = buy_sell_stubs
    state["summary_df"] = pd.DataFrame([_summary_row_with_nan()])

    r = client.post("/api/trades/buy", json={
        "portfolio": "CanSlim",
        "action_type": "scalein",
        "trade_id": "202605-001",
        "ticker": "AAPL",
        "shares": 50,
        "price": 210.0,
        "stop_loss": 195.0,
        "rule": "br1.3 Cup w/o Handle",
        "notes": "Adding to winner",
        "date": "2026-05-08",
        "time": "10:00",
    })
    assert r.status_code == 200, r.text
    saved = state["saved_summaries"][-1]["row"]
    # Body's rule wins over NaN existing rule
    assert saved.get("Rule") == "br1.3 Cup w/o Handle"
    # Body's notes wins over NaN existing buy_notes
    assert saved.get("Buy_Notes") == "Adding to winner"


def test_log_buy_scale_in_with_nan_body_falls_back_to_clean_existing(buy_sell_stubs):
    """S2: if body has no rule but the existing row has a real rule, the
    fallback should pick up the real rule (helper is transparent for non-NaN)."""
    state, client = buy_sell_stubs
    clean_row = _summary_row_with_nan()
    clean_row["rule"] = "br1.3 Cup w/o Handle"
    clean_row["buy_notes"] = "Initial entry"
    state["summary_df"] = pd.DataFrame([clean_row])

    r = client.post("/api/trades/buy", json={
        "portfolio": "CanSlim",
        "action_type": "scalein",
        "trade_id": "202605-001",
        "ticker": "AAPL",
        "shares": 50,
        "price": 210.0,
        "stop_loss": 195.0,
        "date": "2026-05-08",
        "time": "10:00",
    })
    assert r.status_code == 200, r.text
    saved = state["saved_summaries"][-1]["row"]
    assert saved.get("Rule") == "br1.3 Cup w/o Handle"
    assert saved.get("Buy_Notes") == "Initial entry"


def test_log_sell_with_nan_existing_row_writes_null_not_nan(buy_sell_stubs):
    """S4 producer guard. Existing row has rule=NaN, buy_notes=NaN.
    Pre-fix: str(row.get("rule", "") or "") = "nan". Post-fix: None."""
    state, client = buy_sell_stubs
    state["summary_df"] = pd.DataFrame([_summary_row_with_nan()])
    state["details_df"] = pd.DataFrame([{
        "trade_id": "202605-001",
        "ticker": "AAPL",
        "action": "BUY",
        "date": pd.Timestamp("2026-05-01 09:30:00"),
        "shares": 100.0,
        "amount": 200.0,
        "trx_id": "B1",
        "instrument_type": "STOCK",
        "multiplier": 1.0,
    }])

    r = client.post("/api/trades/sell", json={
        "portfolio": "CanSlim",
        "trade_id": "202605-001",
        "shares": 50,
        "price": 220.0,
        "rule": "sr1.1 Profit target",
        "notes": "Half off",
        "date": "2026-05-08",
        "time": "10:00",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert "error" not in body, body

    # Inline save_summary_row at line 3054 is the protected site.
    assert state["saved_summaries"], "Expected a save_summary_row call"
    inline_saved = state["saved_summaries"][-1]["row"]

    assert inline_saved.get("Rule") is None, \
        f"Rule should be None (was NaN-on-existing), got {inline_saved.get('Rule')!r}"
    assert inline_saved.get("Buy_Notes") is None, \
        f"Buy_Notes should be None, got {inline_saved.get('Buy_Notes')!r}"
    # Belt + suspenders against any future bypass
    assert inline_saved.get("Rule") != "nan"
    assert inline_saved.get("Buy_Notes") != "nan"


def test_set_trade_grade_with_nan_existing_row_writes_null_not_nan(buy_sell_stubs):
    """S3 defensive cleanup guard. set_trade_grade was a re-anchorer (not a
    producer): if a literal 'nan' string ever lived in the DB, this code
    re-persisted it via `row.get("rule") or None` ('nan' is truthy).
    Post-fix: clean_text_value strips it."""
    state, client = buy_sell_stubs
    polluted_row = _summary_row_with_nan()
    polluted_row["rule"] = "nan"           # simulating a hypothetical pre-existing string-NaN
    polluted_row["buy_notes"] = "NaN"
    polluted_row["sell_rule"] = "null"
    polluted_row["sell_notes"] = "None"
    polluted_row["notes"] = "  nan  "
    state["summary_df"] = pd.DataFrame([polluted_row])

    r = client.post("/api/trades/grade", json={
        "portfolio": "CanSlim",
        "trade_id": "202605-001",
        "grade": 4,
    })
    assert r.status_code == 200, r.text
    saved = state["saved_summaries"][-1]["row"]

    for field in ("Rule", "Buy_Notes", "Sell_Rule", "Sell_Notes", "Notes"):
        assert saved.get(field) is None, \
            f"{field} should be None (sentinel-stripped), got {saved.get(field)!r}"


# ---------------------------------------------------------------------------
# D4: exercise_option (scale-in stock-side + option-side notes concat)
# ---------------------------------------------------------------------------


@pytest.fixture
def exercise_stubs(monkeypatch):
    """Yield (state, client) for exercise_option tests.

    Mirrors the scaffolding in tests/test_exercise_option.py — the endpoint
    composes 4 _in_txn helpers inside one atomic_transaction. We capture
    every dict passed to _save_summary_with_closures_in_txn so the test
    can inspect what the producer sites built.
    """
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)

    import api.main as main

    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    state: dict[str, Any] = {
        "summary_df": pd.DataFrame(),
        "details_df": pd.DataFrame(),
        "trade_ids_for_month": [],
        "details_inserted": [],
        "summaries_written": [],
        "audit_logs": [],
    }

    class _FakeCache:
        def __init__(self, fetch):
            self._fetch = fetch
        def __call__(self, *args, **kwargs):
            return self._fetch(*args, **kwargs)
        def clear(self):
            pass

    monkeypatch.setattr(db_layer, "load_summary",
                        _FakeCache(lambda *a, **kw: state["summary_df"]))
    monkeypatch.setattr(db_layer, "load_details",
                        _FakeCache(lambda *a, **kw: state["details_df"]))
    monkeypatch.setattr(db_layer, "load_all_trade_ids_for_month",
                        lambda p, ym: state["trade_ids_for_month"])
    monkeypatch.setattr(main, "_normalize_trades", lambda df: df)

    class FakeCursor:
        def __init__(self):
            self._next = None
        def execute(self, sql, params=None):
            if "SELECT id FROM portfolios" in sql:
                self._next = (1,)
            else:
                self._next = None
        def fetchone(self):
            return self._next
        def fetchall(self):
            return []
        def close(self):
            pass

    @contextmanager
    def fake_atomic_transaction():
        yield (None, FakeCursor())
    monkeypatch.setattr(db_layer, "atomic_transaction", fake_atomic_transaction)

    trx_counter: dict[str, int] = {}
    def fake_gen_trx(cur, portfolio_id, trade_id, prefix):
        trx_counter[prefix] = trx_counter.get(prefix, 0) + 1
        return f"{prefix}{trx_counter[prefix]}"
    monkeypatch.setattr(db_layer, "_generate_unique_trx_id_in_txn", fake_gen_trx)

    detail_counter = {"v": 0}
    def fake_save_detail(cur, portfolio_id, row):
        detail_counter["v"] += 1
        state["details_inserted"].append({**row, "_id": detail_counter["v"]})
        return detail_counter["v"]
    monkeypatch.setattr(db_layer, "_save_detail_row_in_txn", fake_save_detail)

    summary_counter = {"v": 100}
    def fake_save_summary(cur, portfolio_id, trade_id, summary, closures):
        summary_counter["v"] += 1
        state["summaries_written"].append({
            "trade_id": trade_id,
            "summary": dict(summary),
            "_id": summary_counter["v"],
        })
        return summary_counter["v"]
    monkeypatch.setattr(db_layer, "_save_summary_with_closures_in_txn",
                        fake_save_summary)

    monkeypatch.setattr(db_layer, "_log_audit_in_txn",
                        lambda *a, **kw: state["audit_logs"].append(a))

    if hasattr(main.limiter, "enabled"):
        original_enabled = main.limiter.enabled
        main.limiter.enabled = False
    else:
        original_enabled = None

    client = TestClient(main.app, headers=_auth_headers())
    try:
        yield state, client
    finally:
        if original_enabled is not None:
            main.limiter.enabled = original_enabled


def _opt_summary_row(rule=np.nan, buy_notes=np.nan, notes=np.nan,
                     sell_rule=np.nan, sell_notes=np.nan):
    """An OPEN OPTION trade row whose user-text fields default to NaN."""
    return {
        "trade_id": "202604-001",
        "ticker": "AMZN 270115 $270C",
        "status": "OPEN",
        "instrument_type": "OPTION",
        "multiplier": 100,
        "shares": 2,
        "avg_entry": 35.63,
        "open_date": "2026-04-01",
        "stop_loss": None,
        "rule": rule,
        "buy_notes": buy_notes,
        "notes": notes,
        "sell_rule": sell_rule,
        "sell_notes": sell_notes,
        "risk_budget": 7126,
    }


def _stock_existing_row(rule=np.nan, buy_notes=np.nan, notes=np.nan):
    """An OPEN STOCK trade row for the same underlying, used to test
    exercise_option's scale-in branch."""
    return {
        "trade_id": "202605-005",
        "ticker": "AMZN",
        "status": "OPEN",
        "instrument_type": "STOCK",
        "multiplier": 1,
        "shares": 50,
        "avg_entry": 300.0,
        "open_date": "2026-05-01",
        "stop_loss": 280.0,
        "rule": rule,
        "buy_notes": buy_notes,
        "notes": notes,
        "sell_rule": None,
        "sell_notes": None,
        "risk_budget": 1000,
    }


def _details_row(trade_id, ticker, action, date, shares, amount, trx_id,
                 instrument_type, multiplier):
    return {
        "trade_id": trade_id, "ticker": ticker, "action": action,
        "date": date, "shares": shares, "amount": amount, "trx_id": trx_id,
        "instrument_type": instrument_type, "multiplier": multiplier,
    }


def test_exercise_option_scale_in_with_nan_existing_stock_row(exercise_stubs):
    """S5 + B2 producer guards. Existing stock trade has rule=NaN, buy_notes=NaN,
    notes=NaN. exercise_option scales into it. The captured stock_summary_row
    must have Rule=None, Buy_Notes=None, and Notes that does NOT contain 'nan'."""
    state, client = exercise_stubs
    state["summary_df"] = pd.DataFrame([
        _opt_summary_row(rule="opt rule", buy_notes="opt notes"),
        _stock_existing_row(rule=np.nan, buy_notes=np.nan, notes=np.nan),
    ])
    state["details_df"] = pd.DataFrame([
        _details_row("202604-001", "AMZN 270115 $270C", "BUY",
                     "2026-04-01 09:30:00", 2, 35.63, "B1", "OPTION", 100),
        _details_row("202605-005", "AMZN", "BUY",
                     "2026-05-01 09:30:00", 50, 300.0, "B1", "STOCK", 1),
    ])

    r = client.post("/api/trades/exercise-option", json={
        "portfolio": "CanSlim",
        "trade_id": "202604-001",
        "date": "2026-05-09",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("status") == "ok", f"unexpected response: {body}"
    assert body["stock_was_new"] is False, "should have scaled into existing AMZN"

    # Two summaries written: option (index 0) and stock scale-in (index 1).
    assert len(state["summaries_written"]) == 2
    stock_summary = state["summaries_written"][1]["summary"]

    assert stock_summary.get("Rule") is None, \
        f"Rule should be None, got {stock_summary.get('Rule')!r}"
    assert stock_summary.get("Buy_Notes") is None, \
        f"Buy_Notes should be None, got {stock_summary.get('Buy_Notes')!r}"

    # B2: the merged Notes string must not embed "nan" anywhere.
    notes = stock_summary.get("Notes") or ""
    assert "nan" not in notes.lower().split(), \
        f"Notes contains 'nan' substring: {notes!r}"
    # The auto-link should still be present (proves merge happened).
    assert "Scaled in via exercise" in notes


def test_exercise_option_option_side_with_nan_notes(exercise_stubs):
    """B1 producer guard. Option-side existing notes is NaN. The auto-note
    concat must NOT produce 'nan\\nExercised on ...'."""
    state, client = exercise_stubs
    # Option row has notes=NaN; no existing stock so we hit the new-stock branch
    # (which already uses literal "" — but the option-side notes path runs
    # first and is what we're guarding).
    state["summary_df"] = pd.DataFrame([
        _opt_summary_row(notes=np.nan),
    ])
    state["details_df"] = pd.DataFrame([
        _details_row("202604-001", "AMZN 270115 $270C", "BUY",
                     "2026-04-01 09:30:00", 2, 35.63, "B1", "OPTION", 100),
    ])
    state["trade_ids_for_month"] = ["202605-001"]

    r = client.post("/api/trades/exercise-option", json={
        "portfolio": "CanSlim",
        "trade_id": "202604-001",
        "date": "2026-05-09",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("status") == "ok", f"unexpected response: {body}"

    # First summary written is the option-side recompute.
    opt_summary = state["summaries_written"][0]["summary"]
    notes = opt_summary.get("Notes") or ""
    assert "nan" not in notes.lower().split(), \
        f"Option Notes contains 'nan': {notes!r}"
    # The auto-link should still be present.
    assert "Exercised on" in notes


# ---------------------------------------------------------------------------
# D5: Direct unit tests at the producer-line semantics
# ---------------------------------------------------------------------------


def test_log_buy_scale_in_producer_semantics():
    """Direct exercise of the post-fix expression at api/main.py:2819-2820 —
    confirms behavior for the four corner combinations of NaN/real on body
    and existing-row inputs."""
    clean = db_layer.clean_text_value

    # (existing_rule, body_rule) -> expected
    cases = [
        (np.nan,           "",            None),  # both empty/NaN -> None
        (np.nan,           "real body",   "real body"),
        ("real existing",  "",            "real existing"),
        ("real existing",  "body wins?",  "real existing"),  # existing wins per code
        ("nan",            "real body",   "real body"),  # sentinel falls through
    ]
    for existing, body, expected in cases:
        # Mirror the post-fix expression: prefer existing, fall back to body.
        result = clean(existing) or clean(body)
        assert result == expected, f"({existing!r}, {body!r}) -> {result!r}, expected {expected!r}"


def test_log_sell_producer_semantics():
    """Direct exercise of the post-fix expression at api/main.py:3041-3042."""
    clean = db_layer.clean_text_value

    # (existing_rule_in_row) -> expected (no body-fallback; sell only reads existing)
    assert clean(np.nan) is None
    assert clean("nan") is None
    assert clean("") is None
    assert clean("real rule") == "real rule"


def test_exercise_scale_in_producer_semantics():
    """Direct exercise of api/main.py:3406-3407."""
    clean = db_layer.clean_text_value
    assert clean(np.nan) is None
    assert clean("nan") is None
    assert clean("real value") == "real value"


def test_exercise_notes_concat_semantics():
    """Direct exercise of api/main.py:3297 + 3398. With NaN input, the
    helper returns None and the `or ""` fallback keeps the string concat
    safe (no 'nan' embedded in the merged notes)."""
    clean = db_layer.clean_text_value
    auto_note = "Exercised on 2026-05-09 — converted to AMZN stock position"

    # NaN existing notes -> empty preserved -> auto-note alone
    preserved = clean(np.nan) or ""
    merged = f"{preserved}\n{auto_note}".strip() if preserved else auto_note
    assert "nan" not in merged.lower().split()
    assert merged == auto_note

    # Real existing notes -> concat with newline separator
    preserved = clean("Closing into earnings") or ""
    merged = f"{preserved}\n{auto_note}".strip() if preserved else auto_note
    assert merged.startswith("Closing into earnings\n")
    assert merged.endswith(auto_note)
