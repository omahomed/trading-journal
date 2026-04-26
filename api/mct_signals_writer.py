"""Persistence layer for MCT engine signals.

Writes SignalEvent records to the market_signals table with idempotent
INSERT ... ON CONFLICT (trade_date, signal_type) DO NOTHING semantics.

Validation: every event's signal_type must be in ALLOWED_SIGNAL_TYPES.
Validation runs before the database round-trip so a bad event halts the
whole batch — no partial writes followed by a rollback.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Iterable

from psycopg2.extras import execute_values, Json

from db_layer import get_db_connection
from api.market_signals_vocab import ALLOWED_SIGNAL_TYPES
from api.mct_engine import MCTEngine, EngineConfig, SignalEvent
from api.market_data_repo import get_history, get_latest_date


def write_signals(events: Iterable[SignalEvent]) -> int:
    """Persist signal events. Returns the count of NEW rows actually inserted.

    Pre-existing (trade_date, signal_type) pairs are skipped via ON CONFLICT;
    the return value is the row count from the RETURNING clause, so re-runs
    of the same batch return 0.

    Raises ValueError on the first event with an unknown signal_type.
    """
    events = list(events)
    if not events:
        return 0

    # Validate up front so we don't half-write a batch.
    for ev in events:
        if ev.signal_type not in ALLOWED_SIGNAL_TYPES:
            raise ValueError(
                f"Unknown signal_type {ev.signal_type!r} on {ev.trade_date} "
                f"(allowed: {sorted(ALLOWED_SIGNAL_TYPES)})"
            )

    rows = [
        (
            ev.trade_date,
            ev.signal_type,
            ev.signal_label,
            ev.exposure_before,
            ev.exposure_after,
            ev.state_before,
            ev.state_after,
            Json(ev.meta or {}),
        )
        for ev in events
    ]

    sql = """
        INSERT INTO market_signals (
            trade_date, signal_type, signal_label,
            exposure_before, exposure_after,
            state_before, state_after, meta
        ) VALUES %s
        ON CONFLICT (trade_date, signal_type) DO NOTHING
        RETURNING id
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=500)
            inserted = cur.fetchall()
        conn.commit()
    return len(inserted)


def write_signals_for_date_range(
    start_date: date,
    end_date: date,
    *,
    symbol: str = "^IXIC",
    initial_reference_high: float | None = None,
    initial_state: str = "POWERTREND",
    initial_exposure: int = 200,
    initial_power_trend: bool = True,
    correction_ever_declared: bool = True,
) -> dict:
    """Load history, run the engine over the range, persist signals.

    Returns: {"events_emitted", "rows_inserted", "first_date", "last_date",
              "engine_final_state"}.
    """
    history = get_history(symbol, start_date, end_date)
    if history.empty:
        return {
            "events_emitted": 0,
            "rows_inserted": 0,
            "first_date": None,
            "last_date": None,
            "engine_final_state": None,
        }

    config = EngineConfig(
        initial_reference_high=initial_reference_high,
        initial_state=initial_state,
        initial_exposure=initial_exposure,
        initial_power_trend=initial_power_trend,
        correction_ever_declared=correction_ever_declared,
    )
    engine = MCTEngine(config)
    result = engine.run(history)
    inserted = write_signals(result.signals)

    return {
        "events_emitted": len(result.signals),
        "rows_inserted": inserted,
        "first_date": history["trade_date"].iloc[0],
        "last_date": history["trade_date"].iloc[-1],
        "engine_final_state": result.final_state,
    }
