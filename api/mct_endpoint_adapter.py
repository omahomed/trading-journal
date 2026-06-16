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

# Step-index → canonical signal_type from market_signals_vocab. The Entry
# Ladder UI asks "did Step N fire in the current cycle?" — answer lives
# in the signal log, not in the engine's live step_done flags (which it
# recycles across cycles, e.g., resets 5/6/7 on CORRECTION_NULLIFIED).
_STEP_SIGNAL_TYPE = {
    0: "STEP_0_RALLY_DAY",
    1: "STEP_1_FTD",
    2: "STEP_2_CLOSE_ABOVE_21EMA",
    3: "STEP_3_LOW_ABOVE_21EMA",
    4: "STEP_4_LOW_ABOVE_21EMA_3BARS",
    5: "STEP_5_LOW_ABOVE_50SMA_3BARS",
    6: "STEP_6_MA_STACK_SLOW",
    7: "STEP_7_MA_STACK_FULL",
}


def _step_done_flags(
    state: dict,
    signals: Optional[list[SignalEvent]] = None,
    cycle_start_date: Optional[date] = None,
) -> list[bool]:
    """Per-step validity for the Entry Ladder, under the sum-of-valid-
    steps exposure model.

    Step 0  (Rally Day)        — latched: state["step0_done"]
    Step 1  (FTD)              — latched: state["step1_done"]
    Steps 2-7                  — LIVE: state["live_step_valid"][s] —
                                 re-evaluated against the current bar's
                                 close/low and MA stack by the engine's
                                 _phase_exposure_recompute. The checkmarks
                                 render "true right now", not "fired at
                                 some point in this cycle".
    Step 8  (Power-Trend ON)   — latched: state["power_trend"]

    The `signals` / `cycle_start_date` parameters are preserved for
    backward-compat with callers but are no longer consulted — the engine
    now exposes per-bar live validity directly on state.

    Defensive fallback: if `live_step_valid` is missing (e.g. a stale
    EngineResult from a build before this field was added), fall back to
    the latched step_done flag for that step. Doesn't happen in normal
    operation — the engine populates the dict every bar.
    """
    live = state.get("live_step_valid") or {}
    flags = [
        bool(state.get("step0_done", False)),    # Step 0 — latched
        bool(state.get("step1_done", False)),    # Step 1 — latched
        bool(live.get(2, state.get("step2_done", False))),  # Step 2 — live
        bool(live.get(3, state.get("step3_done", False))),  # Step 3 — live
        bool(live.get(4, state.get("step4_done", False))),  # Step 4 — live
        bool(live.get(5, state.get("step5_done", False))),  # Step 5 — live
        bool(live.get(6, state.get("step6_done", False))),  # Step 6 — live
        bool(live.get(7, state.get("step7_done", False))),  # Step 7 — live
        bool(state.get("power_trend", False)),   # Step 8 — latched
    ]
    return flags


def _highest_step(
    state: dict,
    signals: Optional[list[SignalEvent]] = None,
    cycle_start_date: Optional[date] = None,
) -> int:
    flags = _step_done_flags(state, signals, cycle_start_date)
    achieved = [i for i, f in enumerate(flags) if f]
    return max(achieved) if achieved else -1


def _entry_ladder(
    state: dict,
    signals: Optional[list[SignalEvent]] = None,
    cycle_start_date: Optional[date] = None,
) -> list[dict]:
    flags = _step_done_flags(state, signals, cycle_start_date)
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


# Lookback windows for the volatility-regime metrics. 200 bars matches the
# Webster heuristic the FTD threshold rule references. 21 is the period the
# rest of the app uses for ATR% (see _compute_ticker_atr_pct in api/main.py:587
# — same TR definition, same SMA averaging, same `SMA(TR) / SMA(Low)` ratio).
# Sharing the formula keeps the M Factor chip and the daily journal's per-
# ticker ATR figures consistent.
_VOL_REGIME_LOOKBACK = 200
_ATR_PERIOD = 21


def _avg_up_day_pct(bars: pd.DataFrame) -> Optional[float]:
    """Average close-over-close % gain across UP DAYS ONLY over the last
    _VOL_REGIME_LOOKBACK bars. Down/flat days are excluded.

    Returns the mean expressed as a percent (e.g. 0.92 means 0.92%) or
    None if fewer than 20 up-day samples are in the window (avoids a
    noisy reading on thin data).

    Aligns with the Webster volatility-regime call: avg ≥ 1.0 = HIGH
    volatility (FTD threshold would step to 1.25%); avg < 1.0 = LOW
    (FTD threshold stays at the current 1.0%). Informational only — the
    engine still hard-codes FTD_PCT_THRESHOLD = 0.01.
    """
    if bars.empty or "close" not in bars.columns:
        return None
    window = bars.tail(_VOL_REGIME_LOOKBACK + 1)  # need one extra for prev_close
    closes = window["close"].astype(float).values
    if len(closes) < 21:  # 20 up-day samples needed, so at least 21 closes
        return None
    rets = (closes[1:] - closes[:-1]) / closes[:-1]
    up_rets = rets[rets > 0]
    if len(up_rets) < 20:
        return None
    return float(up_rets.mean() * 100.0)


def _atr_pct(bars: pd.DataFrame) -> Optional[float]:
    """ATR%(21) computed the same way the daily journal's ticker snapshot
    computes it: ATR% = SMA(TR, 21) / SMA(Low, 21) * 100. See
    _compute_ticker_atr_pct in api/main.py:587 — identical TR definition,
    identical 21-bar SMA averaging, identical SMA-of-Low denominator (NOT
    latest close). Sharing the formula keeps this card and the daily
    journal's per-ticker ATR figures apples-to-apples for the user's
    indicator workflow.

    Provided alongside avg_up_day_pct for the empirical side-by-side —
    ATR mixes intraday range + gap moves + bearish bars, so it's a
    broader volatility measure than the up-day-only average.
    Informational only.
    """
    if bars.empty or not {"high", "low", "close"}.issubset(bars.columns):
        return None
    window = bars.tail(_ATR_PERIOD + 1)  # one extra bar for prev_close on the first TR
    if len(window) < _ATR_PERIOD + 1:
        return None
    high = window["high"].astype(float)
    low = window["low"].astype(float)
    close = window["close"].astype(float)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    sma_tr = float(tr.tail(_ATR_PERIOD).mean())
    sma_low = float(low.tail(_ATR_PERIOD).mean())
    if sma_low <= 0:
        return None
    return (sma_tr / sma_low) * 100.0


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
    cycle_start_date: Optional[str] = None
    cycle_start_date_obj: Optional[date] = None
    if cycle_start_idx is not None and rally_active:
        latest_idx = len(bars) - 1
        cycle_day = latest_idx - int(cycle_start_idx) + 1
        idx = int(cycle_start_idx)
        if 0 <= idx < len(bars):
            raw_td = bars["trade_date"].iloc[idx]
            cycle_start_date_obj = pd.Timestamp(raw_td).date()
            cycle_start_date = cycle_start_date_obj.isoformat()
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

    avg_up = _avg_up_day_pct(bars)
    atr_p = _atr_pct(bars)

    return {
        "prefix": prefix,
        "day_num": cycle_day,
        "state": state_name,
        "entry_step": _highest_step(state, result.signals, cycle_start_date_obj),
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
        "entry_ladder": _entry_ladder(state, result.signals, cycle_start_date_obj),
        "ftd_date": _latest_ftd_date(result.signals),
        "data_as_of": _isodate(last["trade_date"]),
        "power_trend_on_since": _power_trend_on_since(bars),
        # V11 surfaces consume cap_at_100 to render the "capped at 100%"
        # indicator on the tape pill and the MCT page.
        "cap_at_100": bool(state.get("cap_at_100")),
        # cycle_start_date — the date STEP_0 fired for the current cycle.
        # None when no rally is active. Used by the MCT page header to show
        # "Cycle started: YYYY-MM-DD (Day N)".
        "cycle_start_date": cycle_start_date,
        # Volatility-regime metrics (informational; engine does NOT consume
        # them — FTD_PCT_THRESHOLD is still the hard-coded 0.01 / 1.0%).
        # avg_up_day_pct: Webster heuristic — mean % gain on up days only
        # over the last 200 bars. ≥1.0 = HIGH volatility regime (would step
        # FTD threshold to 1.25%); <1.0 = LOW (stays at 1.0%).
        # atr_pct: ATR(14) as % of latest close, side-by-side for the
        # empirical comparison. ATR is broader than up-only avg.
        "avg_up_day_pct": None if avg_up is None else round(avg_up, 3),
        "atr_pct": None if atr_p is None else round(atr_p, 3),
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
