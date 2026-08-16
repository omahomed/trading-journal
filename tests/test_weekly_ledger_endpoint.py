"""Tests for GET /api/weekly-ledger + PUT /notes + PATCH retro-notes.

Migration 069 — Weekly Ledger. The endpoint composes ledger rows + stats
+ YTD-avg benchmark + page-level free-text note in one shot. Tests stub
db.list_portfolios + db.get_db_connection so they run without Postgres.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from datetime import date, timedelta

import jwt
import pytest
from fastapi.testclient import TestClient


_TEST_SECRET = "test-secret-not-for-prod"


def _auth_headers() -> dict[str, str]:
    token = jwt.encode({"sub": "test-user"}, _TEST_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


# ── Fake DB connection ────────────────────────────────────────────────
# Only implements the specific SQL shapes get_weekly_ledger uses. Every
# query the endpoint fires is dispatched via _match_sql on a substring
# key; unmatched queries raise so a test hitting an unexpected path
# fails loudly.

@dataclass
class _FakeCursor:
    state: dict[str, Any]
    _rowset: list[tuple] = field(default_factory=list)
    _returning: tuple | None = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, sql: str, params: tuple = ()):  # noqa: D401
        key = _match_sql(sql)
        if key == "ledger":
            self._rowset = self.state.get("ledger_rows", [])
        elif key == "ytd_weekly_counts":
            self._rowset = self.state.get("ytd_counts", [])
        elif key == "select_note":
            note = self.state.get("note", "")
            self._rowset = [(note,)] if note else []
        elif key == "upsert_note":
            note = params[2]
            self.state["upserted_note"] = {"portfolio_id": params[0],
                                           "week_start": params[1],
                                           "note": note}
            from datetime import datetime as _dt
            self._returning = (
                42, params[1] if isinstance(params[1], str)
                else params[1].isoformat(),
                note, _dt(2026, 8, 15, 0, 0, 0),
            )
        elif key == "update_retro_notes":
            self.state["patched_retro_notes"] = {
                "text": params[0], "detail_id": params[1],
            }
            self._returning = self.state.get("patch_return")
        else:
            raise AssertionError(f"unexpected SQL: {sql[:120]}")

    def fetchall(self):
        return list(self._rowset)

    def fetchone(self):
        if self._returning is not None:
            r, self._returning = self._returning, None
            return r
        return self._rowset[0] if self._rowset else None


def _match_sql(sql: str) -> str:
    s = " ".join(sql.split())
    if "FROM trades_details td LEFT JOIN trades_summary" in s:
        return "ledger"
    if "date_trunc('week', td.date)" in s:
        return "ytd_weekly_counts"
    if "FROM weekly_ledger_notes" in s and "SELECT note" in s:
        return "select_note"
    if "INSERT INTO weekly_ledger_notes" in s:
        return "upsert_note"
    if "UPDATE trades_details" in s and "retro_notes" in s:
        return "update_retro_notes"
    return "unknown"


class _FakeConn:
    def __init__(self, state):
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def cursor(self, cursor_factory=None):
        return _FakeCursor(self.state)

    def commit(self):
        self.state["committed"] = self.state.get("committed", 0) + 1


@pytest.fixture
def ledger_client(monkeypatch):
    """TestClient with db.list_portfolios + db.get_db_connection stubbed."""
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)
    import api.main as main
    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    state: dict[str, Any] = {
        "portfolios": [{"id": 1, "name": "CanSlim"}],
        "ledger_rows": [],
        "ytd_counts": [],
        "note": "",
    }

    monkeypatch.setattr(main.db, "list_portfolios",
                        lambda: state["portfolios"])
    monkeypatch.setattr(main.db, "get_db_connection",
                        lambda *a, **kw: _FakeConn(state))

    tc = TestClient(main.app, headers=_auth_headers())
    tc.state = state  # type: ignore[attr-defined]
    return tc


def _row(detail_id=1, ticker="AAPL", action="BUY", trx_id="B1",
         day=date(2026, 8, 12), shares=100, amount=-15000, value=15000,
         row_rule="br3.2", realized_pl=None, retro_notes="",
         instrument_type="STOCK", multiplier=1, buy_rule="br3.2",
         sell_rule=None, status="OPEN"):
    """Shape one ledger tuple in the exact column order the endpoint SELECTs."""
    return (detail_id, "202608-001", ticker, action, trx_id, day,
            shares, amount, value, row_rule, realized_pl, retro_notes,
            instrument_type, multiplier, buy_rule, sell_rule, status)


# ── GET /api/weekly-ledger — Monday snap ────────────────────────────

def test_week_start_snaps_to_monday(ledger_client):
    """Any day-of-week in a request snaps to Monday of that ISO week.
    Aug 15 2026 = Saturday → Monday Aug 10."""
    r = ledger_client.get("/api/weekly-ledger",
                          params={"portfolio": "CanSlim",
                                  "week_start": "2026-08-15"})
    assert r.status_code == 200
    d = r.json()
    assert d["week_start"] == "2026-08-10"
    assert d["week_end"] == "2026-08-14"


def test_monday_input_stays_monday(ledger_client):
    r = ledger_client.get("/api/weekly-ledger",
                          params={"portfolio": "CanSlim",
                                  "week_start": "2026-08-10"})
    assert r.status_code == 200
    assert r.json()["week_start"] == "2026-08-10"


def test_invalid_date_returns_422(ledger_client):
    r = ledger_client.get("/api/weekly-ledger",
                          params={"portfolio": "CanSlim", "week_start": "nope"})
    assert r.status_code == 422


def test_unknown_portfolio_returns_404(ledger_client):
    r = ledger_client.get("/api/weekly-ledger",
                          params={"portfolio": "Ghost",
                                  "week_start": "2026-08-10"})
    assert r.status_code == 404


# ── Stats math ────────────────────────────────────────────────────

def test_stats_from_ledger(ledger_client):
    """Buys/sells split, unique_tickers, net_realized (SELLs only),
    avg_per_day = total / 5."""
    ledger_client.state["ledger_rows"] = [
        _row(detail_id=1, ticker="AAPL", action="BUY", trx_id="B1"),
        _row(detail_id=2, ticker="AAPL", action="BUY", trx_id="A1"),
        _row(detail_id=3, ticker="NVDA", action="BUY", trx_id="B1"),
        _row(detail_id=4, ticker="AAPL", action="SELL", trx_id="S1",
             realized_pl=250.0),
        _row(detail_id=5, ticker="TSLA", action="SELL", trx_id="S1",
             realized_pl=-100.0),
    ]
    r = ledger_client.get("/api/weekly-ledger",
                          params={"portfolio": "CanSlim",
                                  "week_start": "2026-08-10"})
    stats = r.json()["stats"]
    assert stats["total_transactions"] == 5
    assert stats["buys"] == 3
    assert stats["sells"] == 2
    assert stats["unique_tickers"] == 3
    assert stats["net_realized"] == 150.0
    assert stats["avg_per_day"] == 1.0   # 5 / 5


def test_empty_week_returns_zero_stats(ledger_client):
    r = ledger_client.get("/api/weekly-ledger",
                          params={"portfolio": "CanSlim",
                                  "week_start": "2026-08-10"})
    stats = r.json()["stats"]
    assert stats["total_transactions"] == 0
    assert stats["buys"] == 0
    assert stats["sells"] == 0
    assert stats["net_realized"] == 0.0


# ── YTD-avg benchmark ────────────────────────────────────────────

def test_ytd_avg_computes_from_weekly_counts(ledger_client):
    """avg = sum(counts) / weeks_counted. current_vs_avg_pct = signed
    delta of this week vs the avg."""
    ledger_client.state["ledger_rows"] = [
        _row(detail_id=i) for i in range(20)  # 20 this week
    ]
    ledger_client.state["ytd_counts"] = [
        (date(2026, 1, 5), 10), (date(2026, 1, 12), 8),
        (date(2026, 1, 19), 12), (date(2026, 1, 26), 10),
    ]
    d = ledger_client.get("/api/weekly-ledger",
                          params={"portfolio": "CanSlim",
                                  "week_start": "2026-08-10"}).json()
    ytd = d["ytd_avg"]
    assert ytd["weeks_counted"] == 4
    assert ytd["avg_transactions"] == 10.0
    # 20 vs 10 = +100%
    assert ytd["current_vs_avg_pct"] == 100.0


def test_ytd_avg_null_when_no_history(ledger_client):
    """First week of the year → no prior weeks → avg is null, delta null."""
    ledger_client.state["ledger_rows"] = [_row()]
    ytd = ledger_client.get("/api/weekly-ledger",
                            params={"portfolio": "CanSlim",
                                    "week_start": "2026-08-10"}).json()["ytd_avg"]
    assert ytd["weeks_counted"] == 0
    assert ytd["avg_transactions"] is None
    assert ytd["current_vs_avg_pct"] is None


# ── Row shape ─────────────────────────────────────────────────────

def test_row_shape_and_price_derivation(ledger_client):
    """price = value / shares when both are present; falls back to
    amount/shares otherwise."""
    ledger_client.state["ledger_rows"] = [
        _row(detail_id=42, ticker="MSFT", shares=10, amount=-1000, value=1000),
    ]
    row = ledger_client.get("/api/weekly-ledger",
                            params={"portfolio": "CanSlim",
                                    "week_start": "2026-08-10"}).json()["rows"][0]
    assert row["detail_id"] == 42
    assert row["ticker"] == "MSFT"
    assert row["price"] == 100.0
    assert row["shares"] == 10.0


# ── Notes ─────────────────────────────────────────────────────────

def test_note_defaults_to_empty_string(ledger_client):
    d = ledger_client.get("/api/weekly-ledger",
                          params={"portfolio": "CanSlim",
                                  "week_start": "2026-08-10"}).json()
    assert d["note"] == ""


def test_note_returned_when_set(ledger_client):
    ledger_client.state["note"] = "themes: chased entries, oversized adds"
    d = ledger_client.get("/api/weekly-ledger",
                          params={"portfolio": "CanSlim",
                                  "week_start": "2026-08-10"}).json()
    assert d["note"] == "themes: chased entries, oversized adds"


def test_put_note_upserts(ledger_client):
    r = ledger_client.put("/api/weekly-ledger/notes",
                          json={"portfolio": "CanSlim",
                                "week_start": "2026-08-12",   # Wed → snaps
                                "note": "watch overactivity"})
    assert r.status_code == 200
    body = r.json()
    assert body["note"] == "watch overactivity"
    assert body["week_start"] == "2026-08-10"     # snapped to Monday
    upserted = ledger_client.state["upserted_note"]
    assert upserted["note"] == "watch overactivity"
    assert str(upserted["week_start"]) == "2026-08-10"


def test_put_note_rejects_invalid_date(ledger_client):
    r = ledger_client.put("/api/weekly-ledger/notes",
                          json={"portfolio": "CanSlim",
                                "week_start": "nope", "note": ""})
    assert r.status_code == 422


# ── Retro notes patch ────────────────────────────────────────────

def test_patch_retro_notes_writes(ledger_client):
    ledger_client.state["patch_return"] = (99, "why I closed early")
    r = ledger_client.patch("/api/trades/details/99/retro-notes",
                            json={"retro_notes": "why I closed early"})
    assert r.status_code == 200
    body = r.json()
    assert body["retro_notes"] == "why I closed early"
    assert ledger_client.state["patched_retro_notes"]["text"] == "why I closed early"


def test_patch_retro_notes_404_when_row_missing(ledger_client):
    ledger_client.state["patch_return"] = None
    r = ledger_client.patch("/api/trades/details/99/retro-notes",
                            json={"retro_notes": "x"})
    assert r.status_code == 404
