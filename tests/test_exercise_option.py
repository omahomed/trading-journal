"""Tests for the exercise-option endpoint (POST /api/trades/exercise-option).

The endpoint composes 4 _in_txn helpers from db_layer inside one
atomic_transaction. We stub the helpers (mirroring the dashboard_metrics
test pattern) so tests run without a database, and assert on what the
endpoint passes to each helper. Atomicity is verified at the contract
level: when a mid-txn helper raises, the response carries the error
and downstream helpers (audit log, post-commit cache clears) never run.

Test invocation uses FastAPI's TestClient, mirroring tests/test_ibkr_nav.py
— the rate-limiter is disabled per-test so a tight loop never 429s.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import jwt
import pandas as pd
import pytest
from fastapi.testclient import TestClient


_TEST_SECRET = "test-secret-not-for-prod"
_TEST_USER_ID = "test-user"


def _auth_headers() -> dict[str, str]:
    token = jwt.encode({"sub": _TEST_USER_ID}, _TEST_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stubbed(monkeypatch):
    """Yield (state, client) — configure DB stubs via state, exercise via client.

    state has these knobs:
      summary_df          — what db.load_summary returns (post-normalize)
      details_df          — what db.load_details returns (post-normalize)
      trade_ids_for_month — what db.load_all_trade_ids_for_month returns
      fail_on             — name of helper to raise inside (atomicity test)

    state captures these write-side observations:
      details_inserted    — list of detail dicts passed to _save_detail_row_in_txn
      summaries_written   — list of {trade_id, summary, closures}
      trx_ids_generated   — list of (trade_id, prefix, returned_id)
      audit_logs          — list of {action, trade_id, ticker, details}
      caches_cleared      — list of cache names (verifies post-commit cleanup)
    """
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)

    # Import after env is set so middleware closes over the right secret.
    import api.main as main
    import db_layer

    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    state: dict[str, Any] = {
        "summary_df": pd.DataFrame(),
        "details_df": pd.DataFrame(),
        "trade_ids_for_month": [],
        "fail_on": None,
        "details_inserted": [],
        "summaries_written": [],
        "trx_ids_generated": [],
        "audit_logs": [],
        "caches_cleared": [],
    }

    # Replace the ttl_cache-wrapped readers with cache-bypassing fakes.
    # _TTLCache stores entries with a numeric ttl; constructing one with
    # ttl=None blows up inside __call__ (`now + self.ttl`), so we use a
    # minimal stand-in that always re-evaluates and tracks .clear() calls
    # via state so the atomicity test can assert post-commit cleanup.
    class _FakeCache:
        def __init__(self, fetch, name):
            self._fetch = fetch
            self._name = name
        def __call__(self, *args, **kwargs):
            return self._fetch(*args, **kwargs)
        def clear(self):
            state["caches_cleared"].append(self._name)
    monkeypatch.setattr(db_layer, "load_summary",
                        _FakeCache(lambda *a, **kw: state["summary_df"], "summary"))
    monkeypatch.setattr(db_layer, "load_details",
                        _FakeCache(lambda *a, **kw: state["details_df"], "details"))
    monkeypatch.setattr(db_layer, "load_all_trade_ids_for_month",
                        lambda p, ym: state["trade_ids_for_month"])

    # Bypass _normalize_trades — return the dataframe as-is. Test fixtures use
    # snake_case column names so the endpoint's downstream code reads them
    # without further mangling.
    monkeypatch.setattr(main, "_normalize_trades", lambda df: df)

    # Fake atomic_transaction that yields (None, FakeCursor) where the cursor
    # only handles the SELECT portfolios query the endpoint runs first.
    class FakeCursor:
        def __init__(self):
            self._next = None
        def execute(self, sql, params=None):
            if "SELECT id FROM portfolios" in sql:
                self._next = (1,)  # portfolio_id = 1
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

    # Stub the four _in_txn helpers — these are what the endpoint composes.
    trx_counter: dict[str, int] = {}
    def fake_gen_trx(cur, portfolio_id, trade_id, prefix):
        if state["fail_on"] == f"gen_trx_{prefix}":
            raise RuntimeError(f"simulated trx_id failure for prefix {prefix}")
        trx_counter[prefix] = trx_counter.get(prefix, 0) + 1
        trx_id = f"{prefix}{trx_counter[prefix]}"
        state["trx_ids_generated"].append((trade_id, prefix, trx_id))
        return trx_id
    monkeypatch.setattr(db_layer, "_generate_unique_trx_id_in_txn", fake_gen_trx)

    detail_counter = {"v": 0}
    def fake_save_detail(cur, portfolio_id, row):
        if state["fail_on"] == "save_detail":
            raise RuntimeError("simulated detail save failure")
        detail_counter["v"] += 1
        state["details_inserted"].append({**row, "_detail_id": detail_counter["v"]})
        return detail_counter["v"]
    monkeypatch.setattr(db_layer, "_save_detail_row_in_txn", fake_save_detail)

    summary_counter = {"v": 100}
    def fake_save_summary(cur, portfolio_id, trade_id, summary, closures):
        if state["fail_on"] == "save_summary":
            raise RuntimeError("simulated summary save failure")
        summary_counter["v"] += 1
        state["summaries_written"].append({
            "trade_id": trade_id,
            "summary": dict(summary),
            "closures": list(closures),
            "_summary_id": summary_counter["v"],
        })
        return summary_counter["v"]
    monkeypatch.setattr(db_layer, "_save_summary_with_closures_in_txn", fake_save_summary)

    def fake_log_audit(cur, portfolio_id, action, trade_id, ticker, details, username='User'):
        state["audit_logs"].append({
            "action": action, "trade_id": trade_id,
            "ticker": ticker, "details": details, "username": username,
        })
    monkeypatch.setattr(db_layer, "_log_audit_in_txn", fake_log_audit)

    # Cache clears are tracked by _FakeCache.clear() above — no extra
    # patching needed (the .clear() calls go through the same instance).

    # Disable rate limiting for the test session.
    original_enabled = getattr(main.limiter, "enabled", True)
    if hasattr(main.limiter, "enabled"):
        main.limiter.enabled = False

    client = TestClient(main.app, headers=_auth_headers())
    try:
        yield state, client
    finally:
        if hasattr(main.limiter, "enabled"):
            main.limiter.enabled = original_enabled


def _summary_df(rows: list[dict]) -> pd.DataFrame:
    """Build a trades_summary DataFrame in the shape _normalize_trades produces."""
    cols = ["trade_id", "ticker", "status", "instrument_type", "multiplier",
            "shares", "avg_entry", "open_date", "stop_loss", "rule",
            "buy_notes", "notes", "sell_rule", "sell_notes", "risk_budget"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def _details_df(rows: list[dict]) -> pd.DataFrame:
    """Build a trades_details DataFrame in the post-normalize shape."""
    cols = ["trade_id", "ticker", "action", "date", "shares", "amount",
            "trx_id", "instrument_type", "multiplier"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_single_buy_exercise_into_new_stock_trade(stubbed):
    """Canonical example from the design report: 2 contracts of AMZN $270C
    bought at $35.63. Exercise → new AMZN stock position 200 sh @ $305.63."""
    state, client = stubbed
    state["summary_df"] = _summary_df([{
        "trade_id": "202604-001", "ticker": "AMZN 270115 $270C",
        "status": "OPEN", "instrument_type": "OPTION", "multiplier": 100,
        "shares": 2, "avg_entry": 35.63, "open_date": "2026-04-01",
        "stop_loss": None, "rule": "Breakout", "buy_notes": "thesis: AI tailwind",
        "notes": "", "risk_budget": 7126,
    }])
    state["details_df"] = _details_df([{
        "trade_id": "202604-001", "ticker": "AMZN 270115 $270C",
        "action": "BUY", "date": "2026-04-01 09:30:00",
        "shares": 2, "amount": 35.63, "trx_id": "B1",
    }])
    state["trade_ids_for_month"] = ["202605-001", "202605-002"]

    r = client.post("/api/trades/exercise-option", json={
        "portfolio": "CanSlim",
        "trade_id": "202604-001",
        "date": "2026-05-01",
        "notes": "Exercising before expiry",
    })

    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok", f"unexpected response: {body}"
    assert body["option_trade_id"] == "202604-001"
    assert body["stock_trade_id"] == "202605-003"  # next after 001, 002
    assert body["stock_was_new"] is True
    assert body["contracts_exercised"] == 2.0
    assert body["shares_acquired"] == 200.0
    assert body["stock_entry_price"] == 305.63

    # Option SELL detail
    opt_sell = state["details_inserted"][0]
    assert opt_sell["Trade_ID"] == "202604-001"
    assert opt_sell["Action"] == "SELL"
    assert opt_sell["Shares"] == 2.0
    assert opt_sell["Amount"] == 35.63  # weighted_avg = single buy's premium
    assert opt_sell["Value"] == round(2 * 35.63 * 100, 2)
    assert opt_sell["Rule"] == ""  # locked decision: empty
    assert opt_sell["Notes"] == "Exercising before expiry"
    assert opt_sell["Trx_ID"] == "S1"
    assert opt_sell["Multiplier"] == 100.0

    # Stock BUY detail
    stock_buy = state["details_inserted"][1]
    assert stock_buy["Trade_ID"] == "202605-003"
    assert stock_buy["Ticker"] == "AMZN"  # parsed underlying
    assert stock_buy["Action"] == "BUY"
    assert stock_buy["Shares"] == 200.0
    assert stock_buy["Amount"] == 305.63
    assert stock_buy["Value"] == 200 * 305.63
    assert stock_buy["Stop_Loss"] is None  # locked: NULL, user fills via Edit
    assert stock_buy["Rule"] == ""
    assert stock_buy["Trx_ID"] == "B1"  # new trade → B prefix
    assert stock_buy["Multiplier"] == 1.0

    # 3 summary writes: option recompute (CLOSED), stock placeholder, stock recompute
    assert len(state["summaries_written"]) == 3
    opt_sum = state["summaries_written"][0]
    assert opt_sum["trade_id"] == "202604-001"
    assert opt_sum["summary"]["Status"] == "CLOSED"
    assert opt_sum["summary"]["Realized_PL"] == 0  # by-construction zero
    # Auto-note appended to option summary.notes
    assert "Exercised on 2026-05-01" in opt_sum["summary"]["Notes"]
    assert "202605-003" in opt_sum["summary"]["Notes"]
    # Preserved fields from existing option summary survive recompute
    assert opt_sum["summary"]["Rule"] == "Breakout"
    assert opt_sum["summary"]["Buy_Notes"] == "thesis: AI tailwind"

    stock_placeholder = state["summaries_written"][1]
    assert stock_placeholder["trade_id"] == "202605-003"
    assert stock_placeholder["summary"]["Ticker"] == "AMZN"

    stock_final = state["summaries_written"][2]
    assert stock_final["trade_id"] == "202605-003"
    assert stock_final["summary"]["Status"] == "OPEN"
    assert stock_final["summary"]["Shares"] == 200.0
    assert "Created via exercise of option trade 202604-001" in stock_final["summary"]["Notes"]

    # Audit log
    assert len(state["audit_logs"]) == 1
    assert state["audit_logs"][0]["action"] == "EXERCISE"
    assert state["audit_logs"][0]["trade_id"] == "202604-001"

    # Caches cleared post-commit
    assert "details" in state["caches_cleared"]
    assert "summary" in state["caches_cleared"]


def test_multi_buy_option_uses_weighted_average_premium(stubbed):
    """Two BUY lots at different premiums — weighted_avg = LIFO remaining
    cost / contracts. Spec example: 2@$30 + 3@$40 = remaining 5 contracts
    at avg ($60+$120)/5 = $36."""
    state, client = stubbed
    state["summary_df"] = _summary_df([{
        "trade_id": "202604-002", "ticker": "AAPL 260620 $190C",
        "status": "OPEN", "instrument_type": "OPTION", "multiplier": 100,
        "shares": 5, "avg_entry": 36.0, "open_date": "2026-04-01",
    }])
    state["details_df"] = _details_df([
        {"trade_id": "202604-002", "ticker": "AAPL 260620 $190C",
         "action": "BUY", "date": "2026-04-01 09:30:00",
         "shares": 2, "amount": 30.0, "trx_id": "B1"},
        {"trade_id": "202604-002", "ticker": "AAPL 260620 $190C",
         "action": "BUY", "date": "2026-04-05 10:00:00",
         "shares": 3, "amount": 40.0, "trx_id": "A1"},
    ])
    state["trade_ids_for_month"] = []

    r = client.post("/api/trades/exercise-option", json={
        "portfolio": "CanSlim", "trade_id": "202604-002",
        "date": "2026-05-01",
    })

    body = r.json()
    assert body.get("status") == "ok", f"unexpected: {body}"
    assert body["contracts_exercised"] == 5.0
    assert body["shares_acquired"] == 500.0  # 5 × 100
    # weighted_avg = (2*30 + 3*40) / 5 = 36.0; entry = 190 + 36 = 226
    assert body["stock_entry_price"] == 226.0

    opt_sell = state["details_inserted"][0]
    assert opt_sell["Shares"] == 5.0
    assert opt_sell["Amount"] == 36.0


def test_exercise_into_existing_stock_scales_in(stubbed):
    """An existing OPEN AMZN stock trade exists. Exercise should scale in
    via add-on (A_n trx_id), not create a new trade."""
    state, client = stubbed
    state["summary_df"] = _summary_df([
        {"trade_id": "202604-001", "ticker": "AMZN 270115 $270C",
         "status": "OPEN", "instrument_type": "OPTION", "multiplier": 100,
         "shares": 2, "avg_entry": 35.63, "open_date": "2026-04-01"},
        {"trade_id": "202603-005", "ticker": "AMZN",  # existing stock trade
         "status": "OPEN", "instrument_type": "STOCK", "multiplier": 1,
         "shares": 100, "avg_entry": 280.0, "open_date": "2026-03-15",
         "stop_loss": 270.0, "rule": "RS leader", "buy_notes": "B1 thesis",
         "notes": "watching for breakout"},
    ])
    state["details_df"] = _details_df([
        {"trade_id": "202604-001", "ticker": "AMZN 270115 $270C",
         "action": "BUY", "date": "2026-04-01 09:30:00",
         "shares": 2, "amount": 35.63, "trx_id": "B1"},
        {"trade_id": "202603-005", "ticker": "AMZN",
         "action": "BUY", "date": "2026-03-15 10:00:00",
         "shares": 100, "amount": 280.0, "trx_id": "B1"},
    ])

    r = client.post("/api/trades/exercise-option", json={
        "portfolio": "CanSlim", "trade_id": "202604-001",
        "date": "2026-05-01", "notes": "",
    })

    body = r.json()
    assert body.get("status") == "ok", f"unexpected: {body}"
    assert body["stock_trade_id"] == "202603-005"  # existing trade, not new
    assert body["stock_was_new"] is False

    # Stock BUY uses A prefix (add-on to existing trade)
    stock_buy = state["details_inserted"][1]
    assert stock_buy["Trx_ID"] == "A1"
    assert stock_buy["Trade_ID"] == "202603-005"
    assert stock_buy["Shares"] == 200.0
    assert stock_buy["Amount"] == 305.63

    # Only 2 summary writes (no placeholder): option recompute + stock recompute
    assert len(state["summaries_written"]) == 2
    stock_sum = state["summaries_written"][1]
    assert stock_sum["trade_id"] == "202603-005"
    # Existing user fields preserved
    assert stock_sum["summary"]["Stop_Loss"] == 270.0
    assert stock_sum["summary"]["Rule"] == "RS leader"
    assert stock_sum["summary"]["Buy_Notes"] == "B1 thesis"
    # Cross-link appended to .notes
    assert "watching for breakout" in stock_sum["summary"]["Notes"]
    assert "Scaled in via exercise of option trade 202604-001" in stock_sum["summary"]["Notes"]
    # LIFO sees both buys: 100 @ $280 + 200 @ $305.63
    assert stock_sum["summary"]["Shares"] == 300.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_when_trade_not_open(stubbed):
    state, client = stubbed
    state["summary_df"] = _summary_df([{
        "trade_id": "202604-001", "ticker": "AMZN 270115 $270C",
        "status": "CLOSED", "instrument_type": "OPTION", "multiplier": 100,
        "shares": 0, "avg_entry": 35.63, "open_date": "2026-04-01",
    }])
    state["details_df"] = _details_df([])

    r = client.post("/api/trades/exercise-option", json={
        "portfolio": "CanSlim", "trade_id": "202604-001",
        "date": "2026-05-01",
    })

    body = r.json()
    assert "error" in body
    assert "not open" in body["error"].lower()
    assert state["details_inserted"] == []
    assert state["summaries_written"] == []


def test_rejects_when_trade_is_stock_not_option(stubbed):
    state, client = stubbed
    state["summary_df"] = _summary_df([{
        "trade_id": "202604-001", "ticker": "AAPL",
        "status": "OPEN", "instrument_type": "STOCK", "multiplier": 1,
        "shares": 100, "avg_entry": 195.0, "open_date": "2026-04-01",
    }])
    state["details_df"] = _details_df([])

    r = client.post("/api/trades/exercise-option", json={
        "portfolio": "CanSlim", "trade_id": "202604-001",
        "date": "2026-05-01",
    })

    body = r.json()
    assert "error" in body
    assert "options" in body["error"].lower()
    assert state["details_inserted"] == []


def test_rejects_when_no_contracts_held(stubbed):
    """Option has been fully sold prior — LIFO walk yields 0 held."""
    state, client = stubbed
    state["summary_df"] = _summary_df([{
        "trade_id": "202604-001", "ticker": "AMZN 270115 $270C",
        "status": "OPEN", "instrument_type": "OPTION", "multiplier": 100,
        "shares": 0, "avg_entry": 35.63, "open_date": "2026-04-01",
    }])
    state["details_df"] = _details_df([
        {"trade_id": "202604-001", "ticker": "AMZN 270115 $270C",
         "action": "BUY", "date": "2026-04-01 09:30:00",
         "shares": 2, "amount": 35.63, "trx_id": "B1"},
        {"trade_id": "202604-001", "ticker": "AMZN 270115 $270C",
         "action": "SELL", "date": "2026-04-15 14:00:00",
         "shares": 2, "amount": 38.0, "trx_id": "S1"},
    ])

    r = client.post("/api/trades/exercise-option", json={
        "portfolio": "CanSlim", "trade_id": "202604-001",
        "date": "2026-05-01",
    })

    body = r.json()
    assert "error" in body
    assert "no contracts" in body["error"].lower()
    assert state["details_inserted"] == []


def test_rejects_unparseable_ticker(stubbed):
    """Ticker that's marked OPTION but doesn't fit the readable format —
    defensive against legacy / corrupt rows."""
    state, client = stubbed
    state["summary_df"] = _summary_df([{
        "trade_id": "202604-001", "ticker": "WEIRD_FORMAT",
        "status": "OPEN", "instrument_type": "OPTION", "multiplier": 100,
        "shares": 2, "avg_entry": 35.63, "open_date": "2026-04-01",
    }])
    state["details_df"] = _details_df([
        {"trade_id": "202604-001", "ticker": "WEIRD_FORMAT",
         "action": "BUY", "date": "2026-04-01 09:30:00",
         "shares": 2, "amount": 35.63, "trx_id": "B1"},
    ])

    r = client.post("/api/trades/exercise-option", json={
        "portfolio": "CanSlim", "trade_id": "202604-001",
        "date": "2026-05-01",
    })

    body = r.json()
    assert "error" in body
    assert "parse" in body["error"].lower()
    assert state["details_inserted"] == []


# ---------------------------------------------------------------------------
# Atomicity
# ---------------------------------------------------------------------------


def test_failure_in_stock_save_summary_aborts_audit_and_caches(stubbed):
    """If the stock-side summary save raises mid-transaction, the response
    surfaces the error AND downstream steps (audit log, post-commit summary
    cache clear) never run. This is the contract that holds even though the
    underlying atomic_transaction is mocked: the endpoint's control flow
    short-circuits via exception, never reaches the audit + post-commit
    cache lines."""
    state, client = stubbed
    state["summary_df"] = _summary_df([{
        "trade_id": "202604-001", "ticker": "AMZN 270115 $270C",
        "status": "OPEN", "instrument_type": "OPTION", "multiplier": 100,
        "shares": 2, "avg_entry": 35.63, "open_date": "2026-04-01",
    }])
    state["details_df"] = _details_df([
        {"trade_id": "202604-001", "ticker": "AMZN 270115 $270C",
         "action": "BUY", "date": "2026-04-01 09:30:00",
         "shares": 2, "amount": 35.63, "trx_id": "B1"},
    ])
    state["trade_ids_for_month"] = []
    state["fail_on"] = "save_summary"  # FIRST summary write raises

    r = client.post("/api/trades/exercise-option", json={
        "portfolio": "CanSlim", "trade_id": "202604-001",
        "date": "2026-05-01",
    })

    body = r.json()
    assert "error" in body
    assert "simulated summary save failure" in body["error"]
    # The audit log MUST not have run — it's the last step inside the txn.
    assert state["audit_logs"] == []
    # The SUMMARY cache clear is post-commit only — its absence is the
    # signal that commit didn't run. (The DETAILS cache may be cleared
    # mid-txn when the LIFO walk re-reads from DB, regardless of outcome.)
    assert "summary" not in state["caches_cleared"]
