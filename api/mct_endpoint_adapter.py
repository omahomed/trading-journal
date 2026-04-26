"""Adapter translating MCTEngine EngineResult into the legacy JSON shapes
returned by the three market endpoints.

Goal of this module: keep the existing API contracts intact during Phase 3
cutover so the frontend continues rendering. Phase 4 will redesign the
frontend to consume V11-native fields directly.

Pure functions; no side effects (the helper that runs the engine lives
here too as a convenience for the endpoints — it reads market_data via
market_data_repo but doesn't write).
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

import pandas as pd

from api.mct_engine import (
    MCTEngine,
    EngineConfig,
    EngineResult,
    SignalEvent,
)
from api.market_data_repo import get_history, get_latest_date


# ============================================================================
# Engine runner (with default config) — convenience used by all three endpoints
# ============================================================================

# Backfill from 2010-01-01: the V11 engine ratchets reference_high after the
# first nullification, so seeding from 2010 produces the structurally correct
# present-day reference high without cherry-picking a mid-stretch seed.
HISTORY_START = date(2010, 1, 1)


def _default_config(initial_reference_high: Optional[float]) -> EngineConfig:
    return EngineConfig(
        initial_reference_high=initial_reference_high,
        initial_state="POWERTREND",
        initial_exposure=200,
        initial_power_trend=True,
        correction_ever_declared=True,
        # Production replay starts from 2010, so the ratchet must run from
        # bar 0 — otherwise the 2010 seed never updates and corrections
        # never declare. Phase 2's stress-test seeded a contemporaneous
        # value and intentionally kept ratchet_armed=False until first
        # nullification.
        initial_ratchet_armed=True,
    )


def run_engine(symbol: str = "^IXIC", as_of: Optional[date] = None) -> EngineResult:
    """Run the V11 engine over market_data history through `as_of` (or latest).

    Reads from market_data (no yfinance round-trip). Returns EngineResult.
    Empty result if market_data has no rows for `symbol`.
    """
    if as_of is None:
        latest = get_latest_date(symbol)
        if latest is None:
            return EngineResult(bars=pd.DataFrame(), signals=[], final_state={})
        as_of = latest

    history = get_history(symbol, HISTORY_START, as_of)
    if history.empty:
        return EngineResult(bars=pd.DataFrame(), signals=[], final_state={})

    seed = float(history["high"].iloc[0])
    return MCTEngine(_default_config(seed)).run(history)


# ============================================================================
# Helpers
# ============================================================================

# Legacy ladder labels and per-step exposure targets (regardless of cap state).
_STEP_LABELS = [
    "Rally Day", "Follow-Through Day", "Close > 21 EMA",
    "Low > 21 EMA", "Low > 21 EMA (3 days)", "Low > 50 SMA (3 days)",
    "21 EMA > 50 SMA > 200 SMA", "8 EMA > 21 EMA > 50 SMA > 200 SMA",
    "Power-Trend ON",
]
_LEGACY_STEP_EXPOSURES = [20, 40, 60, 80, 100, 120, 140, 160, 200]


def _step_done_flags(state: dict) -> list[bool]:
    return [
        bool(state.get(f"step{i}_done", False)) for i in range(8)
    ] + [bool(state.get("power_trend", False))]


def _highest_step(state: dict) -> int:
    flags = _step_done_flags(state)
    achieved = [i for i, f in enumerate(flags) if f]
    return max(achieved) if achieved else -1


def _entry_ladder(state: dict) -> list[dict]:
    flags = _step_done_flags(state)
    return [
        {"step": i, "label": _STEP_LABELS[i], "achieved": flags[i],
         "exposure": _LEGACY_STEP_EXPOSURES[i]}
        for i in range(9)
    ]


def _state_name(state: dict) -> str:
    if state.get("power_trend"):
        return "POWERTREND"
    if state.get("step4_done") and not state.get("in_correction"):
        return "UPTREND"
    if state.get("rally_active") and any(
        state.get(f"step{i}_done") for i in range(4)
    ):
        return "RALLY MODE"
    return "CORRECTION"


def _isodate(d) -> Optional[str]:
    if d is None:
        return None
    if hasattr(d, "isoformat"):
        return d.isoformat()
    return str(d)[:10]


def _find_reference_high_date(bars: pd.DataFrame, ref_high: Optional[float]) -> Optional[str]:
    """Most recent bar whose high equals reference_high (within $0.01)."""
    if not ref_high or bars.empty:
        return None
    matches = bars[
        (bars["high"] >= ref_high - 0.01) & (bars["high"] <= ref_high + 0.01)
    ]
    if matches.empty:
        return None
    return _isodate(matches["trade_date"].iloc[-1])


def _power_trend_on_since(bars: pd.DataFrame) -> Optional[str]:
    """Walk backward from the last bar while power_trend is True; the first
    bar of that contiguous True run is the start date. None if not in PT."""
    if bars.empty or "power_trend" not in bars.columns:
        return None
    pt = bars["power_trend"].astype(bool).values
    if not pt[-1]:
        return None
    i = len(pt) - 1
    while i > 0 and pt[i - 1]:
        i -= 1
    return _isodate(bars["trade_date"].iloc[i])


def _latest_ftd_date(signals: list[SignalEvent]) -> Optional[str]:
    for sig in reversed(signals):
        if sig.signal_type == "STEP_1_FTD":
            return _isodate(sig.trade_date)
    return None


def _build_active_exits(state: dict, bars: pd.DataFrame) -> list[dict]:
    """Legacy active_exits — derived only from consec_below_21 + close vs sma_50.
    The richer V11 violation vocabulary is intentionally not exposed here;
    Phase 4 will replace this with V11-native exit signals."""
    exits = []
    consec = int(state.get("consec_below_21") or 0)
    if consec >= 2:
        exits.append({
            "signal": "21 EMA Confirmed Break",
            "detail": f"{consec} consecutive closes below 21 EMA",
            "target": "30%",
            "severity": "SERIOUS",
        })
    elif consec == 1:
        exits.append({
            "signal": "21 EMA Watch",
            "detail": "1 close below 21 EMA — watching for confirmation",
            "target": "50%",
            "severity": "WARNING",
        })

    if not bars.empty:
        last = bars.iloc[-1]
        sma_50 = last.get("sma_50")
        close = last.get("close")
        if pd.notna(sma_50) and pd.notna(close) and float(close) < float(sma_50):
            exits.append({
                "signal": "50 SMA Violation",
                "detail": "Price below 50 SMA",
                "target": "0%",
                "severity": "CRITICAL",
            })
    return exits


# ============================================================================
# Public translators
# ============================================================================

def to_rally_prefix_response(result: EngineResult) -> dict[str, Any]:
    """Translate to the legacy /api/market/rally-prefix shape."""
    if result.bars.empty:
        return {"prefix": "", "error": "No data"}

    state = result.final_state
    bars = result.bars
    last = bars.iloc[-1]

    state_name = _state_name(state)
    rally_active = bool(state.get("rally_active"))

    # day_num is anchored to cycle_start_idx (the bar where the current
    # rally cycle's STEP_0 fired). Survives V10 soft resets and post-Step-4
    # transitions — only resets on RALLY_INVALIDATED / CORRECTION_DECLARED /
    # V10_FULL_INVALIDATION. Distinct from the engine's internal rally_count,
    # which freezes after STEP_4 because rally-hunt logic stops running.
    cycle_start_idx = state.get("cycle_start_idx")
    if cycle_start_idx is not None and rally_active:
        latest_idx = len(bars) - 1
        cycle_day = latest_idx - int(cycle_start_idx) + 1
    else:
        cycle_day = 0

    if rally_active and cycle_day > 0:
        prefix = f"Day {cycle_day}: "
    else:
        prefix = "CORRECTION: "

    close = float(last["close"])
    ema_8 = float(last["ema_8"]) if pd.notna(last["ema_8"]) else 0.0
    ema_21 = float(last["ema_21"]) if pd.notna(last["ema_21"]) else 0.0
    sma_50 = float(last["sma_50"]) if pd.notna(last["sma_50"]) else 0.0
    sma_200 = float(last["sma_200"]) if pd.notna(last["sma_200"]) else 0.0

    ref_high = float(state.get("reference_high") or 0.0)
    drawdown = ((close - ref_high) / ref_high * 100) if ref_high > 0 else 0.0

    return {
        "prefix": prefix,
        "day_num": cycle_day,
        "state": state_name,
        "entry_step": _highest_step(state),
        "entry_exposure": int(state.get("exposure") or 0),
        "price": round(close, 2),
        "ema8": round(ema_8, 2),
        "ema21": round(ema_21, 2),
        "sma50": round(sma_50, 2),
        "sma200": round(sma_200, 2),
        "reference_high": round(ref_high, 2),
        "reference_high_date": _find_reference_high_date(bars, ref_high),
        "drawdown_pct": round(drawdown, 2),
        "consecutive_below_21": int(state.get("consec_below_21") or 0),
        "active_exits": _build_active_exits(state, bars),
        "low_above_21_streak": int(state.get("consec_low_above_21") or 0),
        "low_above_50_streak": int(state.get("consec_low_above_50") or 0),
        "stack_8_21": ema_8 > ema_21,
        "stack_21_50": ema_21 > sma_50,
        "stack_50_200": sma_50 > sma_200,
        "entry_ladder": _entry_ladder(state),
        "ftd_date": _latest_ftd_date(result.signals),
        "data_as_of": _isodate(last["trade_date"]),
        "power_trend_on_since": _power_trend_on_since(bars),
    }


def to_market_state_for_journal(result: EngineResult) -> dict[str, Any]:
    """Compact V11 snapshot used by /api/journal/* paths.

    Does NOT include market_window (deprecated as of Phase 3a).
    """
    state = result.final_state
    return {
        "market_state": _state_name(state),
        "exposure_ceiling": int(state.get("exposure") or 0),
        "cap_at_100": bool(state.get("cap_at_100")),
    }
