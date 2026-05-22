"""Diagnostic-logging guard for _compute_ticker_atr_pct exception path.

The function returns 0.0 on any yfinance exception. The 0.0 silently
weights the ticker out of any portfolio_heat snapshot that includes it.
Without the log line added in feat/portfolio-heat-batch-lookup, an
operator couldn't tell whether a snapshot's heat number was suppressed
by a yfinance outage — the saved row looks fine, the dropped tickers
look like they contributed 0.

This test pins the diagnostic so future refactors don't silently drop
the log. Behavior is unchanged (still returns 0.0); only observability
matters.
"""
from __future__ import annotations

import pytest


def test_compute_ticker_atr_pct_logs_exception_path(capsys, monkeypatch):
    """Force a yfinance exception → verify (a) return value stays 0.0
    and (b) a diagnostic line lands on stdout containing the ticker
    name + exception type."""
    import api.main as main
    import yfinance as yf

    class _ExplodingTicker:
        def __init__(self, t): self.t = t
        def history(self, **_):
            raise RuntimeError("simulated yfinance outage")

    monkeypatch.setattr(yf, "Ticker", _ExplodingTicker)

    out = main._compute_ticker_atr_pct("FAKETICKER", as_of_date="2026-05-22")
    assert out == 0.0

    captured = capsys.readouterr()
    # Diagnostic line shape:
    #   [_compute_ticker_atr_pct] FAKETICKER (as_of=2026-05-22) silently returned 0.0 due to: RuntimeError: simulated yfinance outage
    assert "_compute_ticker_atr_pct" in captured.out
    assert "FAKETICKER" in captured.out
    assert "RuntimeError" in captured.out
    assert "silently returned 0.0" in captured.out


def test_compute_ticker_atr_pct_no_log_on_clean_empty(capsys, monkeypatch):
    """Empty df / sparse df should still return 0.0 BUT not log —
    those are expected paths (recent IPO, weekend, etc.), not failure
    modes worth surfacing in operator logs."""
    import api.main as main
    import yfinance as yf
    import pandas as pd

    class _EmptyTicker:
        def __init__(self, t): self.t = t
        def history(self, **_):
            return pd.DataFrame()

    monkeypatch.setattr(yf, "Ticker", _EmptyTicker)

    out = main._compute_ticker_atr_pct("RECENT_IPO")
    assert out == 0.0
    # No log line — empty df isn't an exception, just a sparse-data signal.
    captured = capsys.readouterr()
    assert "silently returned 0.0" not in captured.out
