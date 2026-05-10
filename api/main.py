"""
MO Money — FastAPI backend
Wraps the database layer and supporting modules so the React frontend
can fetch real data via REST endpoints.
"""

import sys
import os

# Add parent directory to path so we can import existing modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Sentry must init before any application code runs so it can hook uncaught
# exceptions from the first request onward. Env var set on Railway; missing
# DSN is a no-op (SDK initializes but never sends).
import sentry_sdk
sentry_sdk.init(
    dsn=os.environ.get("SENTRY_DSN"),
    traces_sample_rate=0.1,
    environment=os.environ.get("RAILWAY_ENVIRONMENT_NAME", "development"),
    send_default_pii=False,
)

from fastapi import FastAPI, Query, Body, Request, Depends, HTTPException
import io
import math
import re
from functools import lru_cache

import psycopg2
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from datetime import datetime, date
import pandas as pd
import jwt
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Import existing modules (needed by the middleware below).
import db_layer as db
import nlv_service
from trade_calc import (
    compute_lifo_summary,
    compute_trade_risk,
    is_option_ticker,
    multiplier_for_ticker,
    normalize_journal_columns as _normalize_journal,
    validate_post_edit_lifo,
)
from tickers import parse_option_ticker


def _rate_limit_key(request: Request) -> str:
    """Rate-limit per authenticated user; fall back to IP if unauthenticated."""
    uid = getattr(request.state, "user_id", None)
    return uid or get_remote_address(request)


# slowapi is opt-in: only endpoints with @limiter.limit(...) are rate-limited.
# No global default so we don't accidentally throttle normal navigation.
limiter = Limiter(key_func=_rate_limit_key)

app = FastAPI(title="MO Money API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — allow React dev server + Vercel production
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.vercel\.app|https://motrading\.net|http://localhost:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# JWT AUTH MIDDLEWARE
# ============================================================
# next-auth on the frontend mints an HS256 JWT in the session callback using
# AUTH_SECRET (shared with Railway). Every request to /api/* except /api/health
# must arrive with Authorization: Bearer <token>. We verify with the same
# secret, pull `sub` (user UUID), and stash it on request.state.user_id so
# handlers can scope DB queries by owner.
AUTH_SECRET = os.environ.get("AUTH_SECRET")
if not AUTH_SECRET:
    print("[AUTH] AUTH_SECRET not set — every /api/* call will be rejected (except /api/health).")

# Owner / founder UUID. Single-tenant app (Tier 1 multi-tenancy is the future
# state, not current). Endpoints under /api/admin/* additionally require the
# authenticated user_id to match this UUID — bearer auth alone isn't enough.
# Sourced from migration 002's seed; kept in code so the gate works even when
# the DB layer is unreachable. Override via env var for staging-as-non-owner.
FOUNDER_USER_ID = os.environ.get(
    "FOUNDER_USER_ID", "d7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a"
)

# Paths that remain reachable without a bearer token.
_PUBLIC_PATHS = {"/api/health", "/api/healthz", "/"}

# Must mirror the CORSMiddleware regex above. CORS headers don't automatically
# propagate onto responses an inner middleware returns early (known Starlette
# quirk), so we attach them ourselves on every 4xx/5xx we emit from here.
_CORS_ORIGIN_RE = re.compile(r"https://.*\.vercel\.app|https://motrading\.net|http://localhost:\d+")


def _cors_headers_for(request: Request) -> dict:
    origin = request.headers.get("origin", "")
    if origin and _CORS_ORIGIN_RE.fullmatch(origin):
        return {
            "access-control-allow-origin": origin,
            "access-control-allow-credentials": "true",
            "vary": "Origin",
        }
    return {}


def _reject(request: Request, status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(
        {"detail": detail}, status_code=status_code,
        headers=_cors_headers_for(request),
    )


@app.middleware("http")
async def jwt_auth_middleware(request: Request, call_next):
    path = request.url.path

    # CORS preflight always passes through.
    if request.method == "OPTIONS":
        return await call_next(request)

    if path in _PUBLIC_PATHS or not path.startswith("/api/"):
        return await call_next(request)

    auth_header = request.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        return _reject(request, 401, "Missing bearer token")

    if not AUTH_SECRET:
        return _reject(request, 500, "Server auth not configured")

    token = auth_header[7:].strip()
    try:
        payload = jwt.decode(token, AUTH_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        return _reject(request, 401, "Token expired")
    except jwt.InvalidTokenError as e:
        return _reject(request, 401, f"Invalid token: {e}")

    user_id = payload.get("sub")
    if not user_id:
        return _reject(request, 401, "Token missing user id")

    request.state.user_id = user_id

    # Tell db_layer which user this request belongs to. get_db_connection()
    # reads this ContextVar and does `SET app.user_id = <uuid>` on each new
    # connection, which is what Postgres RLS policies filter by. Reset in
    # finally so the ContextVar doesn't leak into any background tasks.
    ctx_token = db.current_user_id.set(user_id)
    try:
        return await call_next(request)
    finally:
        db.current_user_id.reset(ctx_token)


def get_user_id(request: Request) -> str:
    """FastAPI dependency: pull the authenticated user_id set by the middleware."""
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user_id


# Required secrets are expected in the runtime environment (Railway service
# variables in production; .env / shell export locally). No values are
# hardcoded here — every secret must come from the environment so rotation
# doesn't require a code change and nothing leaks to git history.
_REQUIRED_ENV = {
    "R2": ["R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME", "R2_PUBLIC_URL"],
    "IBKR": ["IBKR_FLEX_TOKEN", "IBKR_FLEX_QUERY_ID"],
}
for _group, _keys in _REQUIRED_ENV.items():
    _missing = [k for k in _keys if not os.environ.get(k)]
    if _missing:
        print(f"[{_group}] Missing env vars: {', '.join(_missing)} — related endpoints will fail until set.")

try:
    import r2_storage as r2
    _r2_ep = os.environ.get("R2_ENDPOINT_URL", "")
    print(f"[R2] Module loaded, endpoint: {_r2_ep[:40] if _r2_ep else 'NOT CONFIGURED'}")
except Exception as _r2_err:
    print(f"[R2] Import failed: {_r2_err}")
    r2 = None

def _is_r2_available():
    if r2 is None:
        return False
    try:
        cfg = r2._get_r2_config()
        return bool(cfg.get("endpoint_url"))
    except Exception:
        return False


def _df_to_records(df: pd.DataFrame) -> list:
    """Convert DataFrame to list of dicts, handling NaN and Timestamps."""
    if df.empty:
        return []
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df.fillna("").to_dict(orient="records")


# ============================================================
# JOURNAL ENDPOINTS
# ============================================================
@app.get("/api/journal/latest")
def journal_latest(portfolio: str = "CanSlim", before: str = ""):
    """Get the most recent journal entry (NLV, daily change, etc.).

    `before` (YYYY-MM-DD, optional): when set, returns the latest entry
    strictly before that date. Used by the Daily Routine form so editing a
    past date pulls the correct *prior* day's NLV as the baseline for the
    Daily % calculation — without this, editing yesterday's entry would
    diff against yesterday's own prior-saved value (typically the
    estimated NLV the user is trying to overwrite), producing a meaningless
    delta.
    """
    df = db.load_journal(portfolio)
    if df.empty:
        return {"error": "No journal data"}
    df = _normalize_journal(df)
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    if before:
        cutoff = pd.to_datetime(str(before).strip()[:10], errors="coerce")
        if pd.notna(cutoff):
            df = df[df["day"] < cutoff]
    if df.empty:
        return {"error": "No journal data"}
    df = df.sort_values("day", ascending=False)
    row = df.iloc[0].to_dict()
    for k, v in row.items():
        if isinstance(v, pd.Timestamp):
            row[k] = v.strftime("%Y-%m-%d")
        elif pd.isna(v):
            row[k] = None
        elif hasattr(v, 'item'):  # numpy types
            row[k] = v.item()
    return row


@app.get("/api/journal/history")
def journal_history(portfolio: str = "CanSlim", days: int = 365):
    """Get journal history for equity curve."""
    df = db.load_journal(portfolio)
    if df.empty:
        return []
    df = _normalize_journal(df)
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    df = df.sort_values("day")

    # Self-heal MCT badge fields for recent rows whose stamp is NULL because
    # they were saved before the engine had their bar (typical: today's
    # entry saved while market_data still ends at yesterday). One engine
    # replay, one DB write per healable row, no work in the common case.
    _heal_recent_mct_stamps(portfolio, df)

    # Clean numeric columns
    for c in ["beg_nlv", "end_nlv", "cash_change", "daily_dollar_change",
              "daily_pct_change", "pct_invested", "spy", "nasdaq",
              "portfolio_heat", "spy_atr", "nasdaq_atr"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    # mct_display_day_num is nullable INT — keep NaN distinct from 0 so the
    # frontend can hide the "D{N}" suffix when no day count exists (e.g.
    # CORRECTION rows, or legacy rows pre-dating migration 015).
    if "mct_display_day_num" in df.columns:
        df["mct_display_day_num"] = pd.to_numeric(df["mct_display_day_num"], errors="coerce")

    # Filter to requested days
    if days > 0:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
        df = df[df["day"] >= cutoff]

    # Compute TWR equity curve
    df["adjusted_beg"] = df["beg_nlv"] + df["cash_change"]
    mask = df["adjusted_beg"] != 0
    df["daily_return"] = 0.0
    df.loc[mask, "daily_return"] = (df.loc[mask, "end_nlv"] - df.loc[mask, "adjusted_beg"]) / df.loc[mask, "adjusted_beg"]
    df["twr_curve"] = (1 + df["daily_return"]).cumprod()
    df["portfolio_ltd"] = (df["twr_curve"] - 1) * 100

    # SPY/NDX benchmarks — LTD (cumulative) + daily % change
    if "spy" in df.columns:
        spy_start = df["spy"].iloc[0] if df["spy"].iloc[0] > 0 else 1
        df["spy_ltd"] = (df["spy"] / spy_start - 1) * 100
        df["spy_daily_pct"] = df["spy"].pct_change().fillna(0) * 100
    if "nasdaq" in df.columns:
        ndx_start = df["nasdaq"].iloc[0] if df["nasdaq"].iloc[0] > 0 else 1
        df["ndx_ltd"] = (df["nasdaq"] / ndx_start - 1) * 100
        df["ndx_daily_pct"] = df["nasdaq"].pct_change().fillna(0) * 100

    cols = ["day", "end_nlv", "beg_nlv", "daily_pct_change", "daily_dollar_change",
            "daily_return", "pct_invested", "portfolio_ltd",
            "spy_ltd", "ndx_ltd", "spy_daily_pct", "ndx_daily_pct",
            "spy", "nasdaq", "portfolio_heat", "score", "cash_change",
            "market_window", "market_cycle", "mct_display_day_num",
            "market_notes", "market_action",
            "spy_atr", "nasdaq_atr",
            "highlights", "lowlights", "mistakes", "top_lesson"]
    available_cols = [c for c in cols if c in df.columns]
    return _df_to_records(df[available_cols])


@app.get("/api/journal/mct-state-by-date-range")
def journal_mct_state_by_date_range(start_date: str, end_date: str):
    """Per-day MCT V11 state for a journal date range.

    Replays the V11 engine over full history through end_date (warmup is needed
    so the ratchet, correction flags, and rally cycle anchors converge to their
    structurally correct values), then slices the bar log to [start_date,
    end_date] inclusive.

    Response: {"states": [{trade_date, state, exposure_ceiling, cap_at_100,
                          cycle_day, display_day_num, in_correction,
                          correction_active, power_trend}, ...]}

    display_day_num is what the journal UI's MCT State badge appends as
    "D{N}" — anchored differently per state:
      POWERTREND: bars since STEP_8_POWERTREND_ON (pt_on_idx) — owner studies
        PT durations historically, so this maps cleanly to a separate ref DB
      UPTREND / RALLY MODE: bars since cycle STEP_0 (cycle_start_idx) —
        continuous count across the cycle's rally→uptrend transition
      CORRECTION: None (no day count rendered)
    """
    try:
        from datetime import datetime as _dt
        from api.mct_endpoint_adapter import run_engine

        try:
            start = _dt.strptime(start_date.strip()[:10], "%Y-%m-%d").date()
            end = _dt.strptime(end_date.strip()[:10], "%Y-%m-%d").date()
        except (ValueError, AttributeError):
            return {"error": "start_date and end_date required as YYYY-MM-DD",
                    "states": []}

        if start > end:
            return {"error": "start_date must be <= end_date", "states": []}

        result = run_engine("^IXIC", as_of=end)
        if result.bars.empty:
            return {"states": []}

        bars = result.bars
        # Bars index is RangeIndex(0, N) matching the engine's per-bar i — so
        # row.name preserves the cycle_start_idx / pt_on_idx semantics after
        # we filter.
        trade_dates = pd.to_datetime(bars["trade_date"]).dt.date
        mask = (trade_dates >= start) & (trade_dates <= end)
        sliced = bars[mask]

        states = []
        for orig_idx, row in sliced.iterrows():
            cycle_start_idx = row["cycle_start_idx"]
            pt_on_idx = row.get("pt_on_idx")
            rally_active = bool(row["rally_active"])
            state_name = row["state"]

            if (rally_active and cycle_start_idx is not None
                    and not pd.isna(cycle_start_idx)):
                cycle_day = int(orig_idx) - int(cycle_start_idx) + 1
            else:
                cycle_day = 0

            display_day_num: int | None
            if state_name == "POWERTREND" and pt_on_idx is not None and not pd.isna(pt_on_idx):
                display_day_num = int(orig_idx) - int(pt_on_idx) + 1
            elif state_name in ("UPTREND", "RALLY MODE") and cycle_day > 0:
                display_day_num = cycle_day
            else:
                display_day_num = None

            td = row["trade_date"]
            states.append({
                "trade_date": td.isoformat() if hasattr(td, "isoformat") else str(td)[:10],
                "state": state_name,
                "exposure_ceiling": int(row["exposure"]),
                "cap_at_100": bool(row["cap_at_100"]),
                "cycle_day": cycle_day,
                "display_day_num": display_day_num,
                "in_correction": bool(row["in_correction"]),
                "correction_active": bool(row["correction_active"]),
                "power_trend": bool(row["power_trend"]),
            })

        return {"states": states}
    except Exception as e:
        return {"error": str(e), "states": []}


def _compute_ticker_atr_pct(ticker: str, as_of_date: str = "") -> float:
    """Compute 21-period ATR% = SMA(TR, 21) / SMA(Low, 21) * 100."""
    import yfinance as yf
    try:
        if as_of_date:
            end_dt = pd.Timestamp(as_of_date) + pd.Timedelta(days=1)
            start_dt = pd.Timestamp(as_of_date) - pd.Timedelta(days=60)
            df = yf.Ticker(ticker).history(start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"))
        else:
            df = yf.Ticker(ticker).history(period="45d")
        if df.empty or len(df) < 21:
            return 0.0
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        sma_tr = float(tr.tail(21).mean())
        sma_low = float(df["Low"].tail(21).mean())
        if sma_low <= 0:
            return 0.0
        return round((sma_tr / sma_low) * 100, 4)
    except Exception:
        return 0.0


def _compute_portfolio_heat(portfolio: str, as_of_date: str, equity: float) -> float:
    """Portfolio heat = sum(weight% * atr%/100) for all open positions."""
    try:
        summary_df = db.load_summary(portfolio)
        if summary_df.empty or equity <= 0:
            return 0.0
        summary_df = _normalize_trades(summary_df)
        status_col = "status" if "status" in summary_df.columns else "Status"
        open_df = summary_df[summary_df[status_col].str.upper() == "OPEN"]
        if open_df.empty:
            return 0.0
        heat = 0.0
        for _, row in open_df.iterrows():
            ticker = str(row.get("ticker", "")).strip()
            total_cost = float(row.get("total_cost", 0) or 0)
            if not ticker or total_cost <= 0:
                continue
            atr_pct = _compute_ticker_atr_pct(ticker, as_of_date)
            weight_pct = (total_cost / equity) * 100
            heat += weight_pct * (atr_pct / 100)
        return round(heat, 4)
    except Exception:
        return 0.0


def _compute_cycle_state(as_of_date: str = "") -> str:
    """Compute NASDAQ cycle state for a given date (or today if empty).

    Returns one of: POWERTREND / UPTREND / RALLY MODE / CORRECTION / "".
    Delegates to the /api/market/rally-prefix route handler, which is
    just a regular function we can call directly with an as_of_date.
    """
    try:
        result = rally_prefix(as_of_date)
        if isinstance(result, dict):
            return str(result.get("state", "") or "")
        return ""
    except Exception:
        return ""


def _compute_mct_state_with_day_num(as_of_date: str = "") -> tuple[str, int | None]:
    """Compute (state_name, display_day_num) for a given date, snapshot-style.

    Used by /api/journal/edit at save time to stamp both fields into the
    trading_journal row, so the Daily Journal page can render its MCT badge
    directly from the row instead of replaying the engine on every visit.

    Anchoring rules mirror /api/journal/mct-state-by-date-range — the same
    logic used by the dynamic endpoint the journal page used to call:
      POWERTREND          → bars since pt_on_idx (Power-Trend ON anchor)
      UPTREND / RALLY MODE → bars since cycle_start_idx (cycle STEP_0 anchor)
      CORRECTION          → None (no day count)

    Strict bar match: if the engine has no bar for `as_of_date` exactly, we
    return ("", None) rather than stamping the previous trading day's state
    on the requested date. The old "fall through to the last bar ≤ as_of"
    behavior caused the journal to write yesterday's `day_num` onto a row
    saved before the day's market_data was ingested — so a Tuesday save run
    on Wednesday morning could stamp `POWERTREND D5` on Wednesday because
    Tuesday was D5.

    Empty / unparseable date or engine failure → ("", None) so the caller
    can persist NULL without breaking the journal save; the backfill script
    or a later re-save reconciles the row once market_data catches up.
    """
    try:
        from datetime import datetime as _dt
        from api.mct_endpoint_adapter import run_engine

        as_of = None
        if as_of_date:
            try:
                as_of = _dt.strptime(as_of_date.strip()[:10], "%Y-%m-%d").date()
            except (ValueError, AttributeError):
                as_of = None

        # Best-effort: pull today's ^IXIC bar from yfinance before the engine
        # reads market_data. Without this, a Daily Routine save run after
        # market close (but before any external ingest cron) sees a stale
        # market_data, fails the strict bar match below, and stamps NULL.
        try:
            from api.market_data_updater import update_if_needed
            update_if_needed("^IXIC")
        except Exception:
            pass

        result = run_engine("^IXIC", as_of=as_of)
        if result.bars.empty:
            return ("", None)

        bars = result.bars
        # Require an exact bar match when the caller pinned a date. Anything
        # else (no rows ≤ as_of, or the last bar predates as_of) means the
        # engine is stale relative to the requested day — refuse to guess.
        if as_of is not None:
            trade_dates = pd.to_datetime(bars["trade_date"]).dt.date
            mask = trade_dates == as_of
            if not mask.any():
                return ("", None)
            row = bars[mask].iloc[-1]
            orig_idx = int(bars[mask].index[-1])
        else:
            row = bars.iloc[-1]
            orig_idx = int(bars.index[-1])

        state_name = str(row["state"])
        cycle_start_idx = row.get("cycle_start_idx")
        pt_on_idx = row.get("pt_on_idx")
        rally_active = bool(row.get("rally_active"))

        cycle_day = 0
        if (rally_active and cycle_start_idx is not None
                and not pd.isna(cycle_start_idx)):
            cycle_day = orig_idx - int(cycle_start_idx) + 1

        if state_name == "POWERTREND" and pt_on_idx is not None and not pd.isna(pt_on_idx):
            display_day_num: int | None = orig_idx - int(pt_on_idx) + 1
        elif state_name in ("UPTREND", "RALLY MODE") and cycle_day > 0:
            display_day_num = cycle_day
        else:
            display_day_num = None

        return (state_name, display_day_num)
    except Exception:
        return ("", None)


def _heal_recent_mct_stamps(portfolio: str, df: pd.DataFrame, lookback_days: int = 14) -> None:
    """Backfill NULL mct_display_day_num / market_cycle on recent journal rows.

    The save-time stamper in _compute_mct_state_with_day_num intentionally
    persists NULL when the engine has no bar for the requested date — the
    common cause is "user logged today's journal before market_data ingested
    today's bar." When the bar lands later, those rows would stay NULL
    forever without an explicit re-save. This helper runs once per
    /api/journal/history call: it locates NULL rows in the last
    `lookback_days`, replays the engine once to get every cached bar's
    state, and stamps any row whose date now has a bar. In-memory df is
    patched so the response reflects the fresh values without a second
    DB read.

    Bounded lookback (default 14 days) keeps this cheap on every page
    load — older NULLs go through scripts/backfill_mct_state.py for a
    full historical sweep.
    """
    if df.empty or "mct_display_day_num" not in df.columns:
        return

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
    needs_heal = df[
        (df["day"] >= cutoff)
        & (
            df["mct_display_day_num"].isna()
            | (df.get("market_cycle", pd.Series(dtype=object)).fillna("").astype(str) == "")
        )
    ]
    if needs_heal.empty:
        return

    try:
        from api.mct_endpoint_adapter import run_engine
        # Refresh market_data first so the bar_index built below reflects
        # today's bar — otherwise the per-row short-circuit `if as_of not in
        # bar_index.index: continue` skips today and the heal never fires.
        # (The same refresh inside _compute_mct_state_with_day_num doesn't
        # help here because we'd already have skipped before calling it.)
        try:
            from api.market_data_updater import update_if_needed
            update_if_needed("^IXIC")
        except Exception:
            pass
        result = run_engine("^IXIC")
        if result.bars.empty:
            return
        bars = result.bars.copy()
        bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.date
        bar_index = bars.set_index("trade_date")
    except Exception:
        return

    for _, row in needs_heal.iterrows():
        day_value = row["day"]
        if pd.isna(day_value):
            continue
        as_of = day_value.date() if hasattr(day_value, "date") else day_value
        if as_of not in bar_index.index:
            continue  # engine still doesn't have this bar — skip
        day_str = as_of.strftime("%Y-%m-%d")
        state, day_num = _compute_mct_state_with_day_num(day_str)
        if not state:
            continue

        # Persist via the targeted helper — save_journal_entry rewrites
        # every column and would clobber NLV/notes when called with a
        # partial dict.
        try:
            db.update_journal_mct_state(portfolio, day_str, state, day_num)
        except Exception:
            continue

        df.loc[df["day"] == day_value, "market_cycle"] = state
        df.loc[df["day"] == day_value, "mct_display_day_num"] = day_num


@app.post("/api/journal/edit")
def journal_edit(entry: dict):
    """Update or insert a journal entry. Preserves existing values for fields not sent."""
    try:
        portfolio = entry.pop("portfolio", "CanSlim")
        day = entry.get("day")

        # Load existing entry to preserve fields not being edited
        existing = {}
        df = db.load_journal(portfolio)
        if not df.empty:
            df = _normalize_journal(df)
            df["day"] = pd.to_datetime(df["day"], errors="coerce").dt.strftime("%Y-%m-%d")
            match = df[df["day"] == str(day).strip()[:10]]
            if not match.empty:
                row = match.iloc[0]
                existing = {
                    "ending_nlv": float(row.get("end_nlv", 0) or 0),
                    "beginning_nlv": float(row.get("beg_nlv", 0) or 0),
                    "cash_flow": float(row.get("cash_change", 0) or 0),
                    "daily_dollar_change": float(row.get("daily_dollar_change", 0) or 0),
                    "daily_percent_change": float(row.get("daily_pct_change", 0) or 0),
                    "percent_invested": float(row.get("pct_invested", 0) or 0),
                    "spy_close": float(row.get("spy", 0) or 0),
                    "nasdaq_close": float(row.get("nasdaq", 0) or 0),
                    "market_window": str(row.get("market_window", "") or ""),
                    "market_cycle": str(row.get("market_cycle", "") or ""),
                    "mct_display_day_num": (
                        int(row["mct_display_day_num"])
                        if "mct_display_day_num" in row
                        and row["mct_display_day_num"] is not None
                        and not pd.isna(row["mct_display_day_num"])
                        else None
                    ),
                    "market_notes": str(row.get("market_notes", "") or ""),
                    "market_action": str(row.get("market_action", "") or ""),
                    "portfolio_heat": float(row.get("portfolio_heat", 0) or 0),
                    "spy_atr": float(row.get("spy_atr", 0) or 0),
                    "nasdaq_atr": float(row.get("nasdaq_atr", 0) or 0),
                    "score": int(row.get("score", 0) or 0),
                    "highlights": str(row.get("highlights", "") or ""),
                    "lowlights": str(row.get("lowlights", "") or ""),
                    "mistakes": str(row.get("mistakes", "") or ""),
                    "top_lesson": str(row.get("top_lesson", "") or ""),
                    "nlv_source": str(row.get("nlv_source", "") or "manual"),
                    "holdings_source": str(row.get("holdings_source", "") or "manual"),
                    "status": db.clean_text_value(row.get("status")),
                    "above_21ema": int(row.get("above_21ema", 0) or 0),
                }

        # Merge: use sent values, fall back to existing
        def _f(key, db_key, default=0):
            if key in entry and entry[key] not in (None, ""):
                return float(entry[key])
            return existing.get(db_key, default)

        def _s(key, db_key, default=""):
            if key in entry and entry[key] is not None:
                return str(entry[key])
            return existing.get(db_key, default)

        # Constrain nlv_source / holdings_source to the allowed enum-equivalent
        # values; anything else (including empty/None from older clients)
        # collapses to 'manual'. Tracked independently — a row can be
        # ibkr_auto for NLV and ibkr_override for Holdings (or vice versa).
        nlv_source_in = str(entry.get("nlv_source") or existing.get("nlv_source", "manual")).strip()
        if nlv_source_in not in ("manual", "ibkr_auto", "ibkr_override"):
            nlv_source_in = "manual"
        holdings_source_in = str(entry.get("holdings_source") or existing.get("holdings_source", "manual")).strip()
        if holdings_source_in not in ("manual", "ibkr_auto", "ibkr_override"):
            holdings_source_in = "manual"

        journal_entry = {
            "portfolio_id": portfolio,
            "day": day,
            "ending_nlv": _f("end_nlv", "ending_nlv"),
            "beginning_nlv": _f("beg_nlv", "beginning_nlv"),
            "cash_flow": _f("cash_change", "cash_flow"),
            "daily_dollar_change": _f("daily_dollar_change", "daily_dollar_change"),
            "daily_percent_change": _f("daily_pct_change", "daily_percent_change"),
            "percent_invested": _f("pct_invested", "percent_invested"),
            "spy_close": _f("spy", "spy_close"),
            "nasdaq_close": _f("nasdaq", "nasdaq_close"),
            "market_window": _s("market_window", "market_window"),
            "market_cycle": _s("market_cycle", "market_cycle"),
            "mct_display_day_num": (
                int(entry["mct_display_day_num"])
                if entry.get("mct_display_day_num") not in (None, "")
                else existing.get("mct_display_day_num")
            ),
            "market_notes": _s("market_notes", "market_notes"),
            "market_action": _s("market_action", "market_action"),
            "portfolio_heat": _f("portfolio_heat", "portfolio_heat"),
            "spy_atr": _f("spy_atr", "spy_atr"),
            "nasdaq_atr": _f("nasdaq_atr", "nasdaq_atr"),
            "score": int(_f("score", "score")),
            "highlights": _s("highlights", "highlights"),
            "lowlights": _s("lowlights", "lowlights"),
            "mistakes": _s("mistakes", "mistakes"),
            "top_lesson": _s("top_lesson", "top_lesson"),
            "nlv_source": nlv_source_in,
            "holdings_source": holdings_source_in,
            # PRESERVATION: status + above_21ema. Without these keys,
            # save_journal_entry binds status='U' and above_21ema=0 on every
            # edit — wiping any prior user-entered value. Body wins, falls
            # back to existing-row value, then schema default.
            "status": (
                db.clean_text_value(entry.get("status"))
                or existing.get("status")
                or "U"
            ),
            "above_21ema": int(
                entry["above_21ema"]
                if entry.get("above_21ema") not in (None, "")
                else (existing.get("above_21ema") or 0)
            ),
        }

        # Auto-compute missing market/risk metrics.
        # market_window is deprecated as of MCT V11 Phase 3a — no longer auto-filled.
        # Existing values are preserved if the caller sends them; new entries get NULL.
        day_str = str(day).strip()[:10]
        # Single engine replay yields both the cycle state and the
        # display_day_num the badge appends ("POWERTREND D3" etc.). Snapshot
        # both into the row so the Daily Journal page can render the badge
        # without re-running the engine on every visit.
        if not journal_entry["market_cycle"] or journal_entry["mct_display_day_num"] is None:
            mct_state, mct_day_num = _compute_mct_state_with_day_num(day_str)
            if not journal_entry["market_cycle"]:
                journal_entry["market_cycle"] = mct_state
            if journal_entry["mct_display_day_num"] is None:
                journal_entry["mct_display_day_num"] = mct_day_num
        if not journal_entry["spy_atr"]:
            journal_entry["spy_atr"] = _compute_ticker_atr_pct("SPY", day_str)
        if not journal_entry["nasdaq_atr"]:
            journal_entry["nasdaq_atr"] = _compute_ticker_atr_pct("^IXIC", day_str)
        if not journal_entry["portfolio_heat"]:
            equity = journal_entry["ending_nlv"]
            journal_entry["portfolio_heat"] = _compute_portfolio_heat(portfolio, day_str, equity)

        row_id = db.save_journal_entry(journal_entry)
        return {"status": "ok", "id": row_id}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/api/journal/restamp-mct")
def restamp_mct(payload: dict):
    """Force-recompute MCT state + day_num for a single journal row.

    Bypasses the "only stamp when missing" gate in /api/journal/edit so the
    UI can unstick a row whose original save persisted NULL (engine had no
    bar at the time). Idempotent — re-running with a date the engine still
    can't resolve returns 'no_bar' without touching the row.
    """
    portfolio = payload.get("portfolio", "CanSlim")
    day = payload.get("day")
    if not day:
        return {"status": "error", "detail": "day required"}
    day_str = str(day).strip()[:10]
    state, day_num = _compute_mct_state_with_day_num(day_str)
    if not state:
        return {"status": "no_bar", "detail": "engine has no bar for this date yet"}
    try:
        db.update_journal_mct_state(portfolio, day_str, state, day_num)
        return {"status": "ok", "market_cycle": state, "mct_display_day_num": day_num}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.delete("/api/journal/delete")
def journal_delete(portfolio: str = Query("CanSlim"), day: str = Query(...)):
    """Delete a single journal entry for the given portfolio and day."""
    try:
        deleted_id = db.delete_journal_entry(portfolio, day)
        if deleted_id:
            return {"status": "ok", "id": deleted_id}
        return {"status": "not_found", "detail": f"No entry for {day} in {portfolio}"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/api/journal/backfill-metrics")
@limiter.limit("2/minute")
def journal_backfill_metrics(request: Request, body: dict = Body(...)):
    """Backfill missing market_window, portfolio_heat, spy_atr, nasdaq_atr
    for existing journal entries. Only updates rows where these fields are
    empty/zero; preserves any existing non-zero values."""
    try:
        portfolio = body.get("portfolio", "CanSlim")
        start_date = body.get("start_date", "")
        end_date = body.get("end_date", "")
        force = bool(body.get("force", False))

        df = db.load_journal(portfolio)
        if df.empty:
            return {"status": "ok", "updated": 0, "checked": 0, "message": "No entries"}

        df = _normalize_journal(df)
        df["day"] = pd.to_datetime(df["day"], errors="coerce")
        df = df.sort_values("day")

        if start_date:
            df = df[df["day"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["day"] <= pd.Timestamp(end_date)]

        updated = 0
        checked = 0
        errors = []

        for _, row in df.iterrows():
            checked += 1
            day_str = row["day"].strftime("%Y-%m-%d")

            existing_mw = str(row.get("market_window", "") or "")
            existing_cycle = str(row.get("market_cycle", "") or "")
            existing_heat = float(row.get("portfolio_heat", 0) or 0)
            existing_spy_atr = float(row.get("spy_atr", 0) or 0)
            existing_ndx_atr = float(row.get("nasdaq_atr", 0) or 0)

            # market_window deprecated as of Phase 3a — backfill no longer touches it.
            need_cycle = force or not existing_cycle
            need_heat = force or existing_heat == 0
            need_spy_atr = force or existing_spy_atr == 0
            need_ndx_atr = force or existing_ndx_atr == 0

            if not any([need_cycle, need_heat, need_spy_atr, need_ndx_atr]):
                continue

            try:
                # Sanitize source-of-truth flags to the allowed enum values.
                # Defends against any pre-existing pollution and keeps the
                # CHECK constraint on trading_journal happy.
                nlv_source_in = str(row.get("nlv_source") or "manual").strip()
                if nlv_source_in not in ("manual", "ibkr_auto", "ibkr_override"):
                    nlv_source_in = "manual"
                holdings_source_in = str(row.get("holdings_source") or "manual").strip()
                if holdings_source_in not in ("manual", "ibkr_auto", "ibkr_override"):
                    holdings_source_in = "manual"

                journal_entry = {
                    "portfolio_id": portfolio,
                    "day": day_str,
                    "ending_nlv": float(row.get("end_nlv", 0) or 0),
                    "beginning_nlv": float(row.get("beg_nlv", 0) or 0),
                    "cash_flow": float(row.get("cash_change", 0) or 0),
                    "daily_dollar_change": float(row.get("daily_dollar_change", 0) or 0),
                    "daily_percent_change": float(row.get("daily_pct_change", 0) or 0),
                    "percent_invested": float(row.get("pct_invested", 0) or 0),
                    "spy_close": float(row.get("spy", 0) or 0),
                    "nasdaq_close": float(row.get("nasdaq", 0) or 0),
                    "market_window": existing_mw,  # deprecated; preserved as-is
                    "market_cycle": _compute_cycle_state(day_str) if need_cycle else existing_cycle,
                    "market_notes": str(row.get("market_notes", "") or ""),
                    "market_action": str(row.get("market_action", "") or ""),
                    "portfolio_heat": _compute_portfolio_heat(portfolio, day_str, float(row.get("end_nlv", 0) or 0)) if need_heat else existing_heat,
                    "spy_atr": _compute_ticker_atr_pct("SPY", day_str) if need_spy_atr else existing_spy_atr,
                    "nasdaq_atr": _compute_ticker_atr_pct("^IXIC", day_str) if need_ndx_atr else existing_ndx_atr,
                    "score": int(row.get("score", 0) or 0),
                    "highlights": str(row.get("highlights", "") or ""),
                    "lowlights": str(row.get("lowlights", "") or ""),
                    "mistakes": str(row.get("mistakes", "") or ""),
                    "top_lesson": str(row.get("top_lesson", "") or ""),
                    # PRESERVATION: 5 fields journal_backfill_metrics doesn't
                    # recompute. Without these keys, save_journal_entry's
                    # rewrite-every-column UPDATE binds NULL/DEFAULT — silently
                    # wiping user-entered values on every backfill run.
                    "status": db.clean_text_value(row.get("status")) or "U",
                    "above_21ema": int(row.get("above_21ema", 0) or 0),
                    "mct_display_day_num": (
                        int(row["mct_display_day_num"])
                        if "mct_display_day_num" in row
                        and row["mct_display_day_num"] is not None
                        and not pd.isna(row["mct_display_day_num"])
                        else None
                    ),
                    "nlv_source": nlv_source_in,
                    "holdings_source": holdings_source_in,
                }
                db.save_journal_entry(journal_entry)
                updated += 1
            except Exception as row_err:
                errors.append(f"{day_str}: {str(row_err)}")

        return {
            "status": "ok",
            "portfolio": portfolio,
            "checked": checked,
            "updated": updated,
            "errors": errors[:5] if errors else [],
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ============================================================
# TRADE ENDPOINTS
# ============================================================
def _normalize_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize trade column names to lowercase."""
    rename = {
        "Trade_ID": "trade_id", "Ticker": "ticker", "Status": "status",
        "Open_Date": "open_date", "Closed_Date": "closed_date",
        "Shares": "shares", "Avg_Entry": "avg_entry", "Avg_Exit": "avg_exit",
        "Total_Cost": "total_cost", "Realized_PL": "realized_pl",
        "Unrealized_PL": "unrealized_pl", "Return_Pct": "return_pct",
        "Rule": "rule", "Buy_Notes": "buy_notes", "Sell_Rule": "sell_rule",
        "Sell_Notes": "sell_notes", "Risk_Budget": "risk_budget", "Grade": "grade",
        "BE_Stop_Moved_At": "be_stop_moved_at",
        "Last_Updated": "last_updated",
        "Action": "action", "Date": "date", "Amount": "amount",
        "Value": "value", "Notes": "notes", "Stop_Loss": "stop_loss",
        "Trx_ID": "trx_id", "_DB_ID": "detail_id",
        "Instrument_Type": "instrument_type", "Multiplier": "multiplier",
        "Strategy": "strategy",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    # Also handle already-lowercase columns (from DB mode)
    return df


@app.get("/api/trades/open")
def trades_open(portfolio: str = "CanSlim"):
    """Get all open positions."""
    df = db.load_summary(portfolio)
    if df.empty:
        return []
    df = _normalize_trades(df)
    status_col = "status" if "status" in df.columns else "Status"
    open_df = df[df[status_col].str.upper() == "OPEN"].copy()
    return _df_to_records(open_df)


@app.get("/api/trades/closed")
def trades_closed(portfolio: str = "CanSlim", limit: int = 50):
    """Get recent closed trades."""
    df = db.load_summary(portfolio)
    if df.empty:
        return []
    df = _normalize_trades(df)
    status_col = "status" if "status" in df.columns else "Status"
    closed = df[df[status_col].str.upper() == "CLOSED"].copy()
    if "closed_date" in closed.columns:
        closed["closed_date"] = pd.to_datetime(closed["closed_date"], errors="coerce")
        closed = closed.sort_values("closed_date", ascending=False)
    return _df_to_records(closed.head(limit))


@app.get("/api/trades/details/{trade_id}")
def trade_details(trade_id: str, portfolio: str = "CanSlim"):
    """Get all transactions for a trade campaign."""
    df = db.load_details(portfolio, trade_id=trade_id)
    if df.empty:
        return []
    df = _normalize_trades(df)
    return _df_to_records(df)


@app.get("/api/trades/open/details")
def trades_open_details(portfolio: str = "CanSlim"):
    """Get all transactions for open trades (for stop loss, pyramid info)
    plus the lot_closures for those trades. Frontend uses lot_closures to
    render per-row realized P&L without re-walking LIFO client-side; falls
    back to its own walk for trades whose closures aren't yet backfilled.

    Response shape: {"details": [...], "lot_closures": [...]}.
    """
    summary_df = db.load_summary(portfolio)
    if summary_df.empty:
        return {"details": [], "lot_closures": []}
    summary_df = _normalize_trades(summary_df)
    status_col = "status" if "status" in summary_df.columns else "Status"
    open_ids = summary_df[summary_df[status_col].str.upper() == "OPEN"]["trade_id"].tolist()
    if not open_ids:
        return {"details": [], "lot_closures": []}
    details_df = db.load_details(portfolio)
    if details_df.empty:
        return {"details": [], "lot_closures": []}
    details_df = _normalize_trades(details_df)
    filtered = details_df[details_df["trade_id"].isin(open_ids)].copy()
    if "date" in filtered.columns:
        filtered["date"] = pd.to_datetime(filtered["date"], errors="coerce")
        filtered = filtered.sort_values(["trade_id", "date"])

    # Closures for the same set of trades — one query, batch-filtered to
    # the open_ids we just returned details for. Avoids N+1; bounded payload.
    closures_df = db.load_lot_closures(portfolio, trade_ids=open_ids)

    return {
        "details": _df_to_records(filtered),
        "lot_closures": _df_to_records(closures_df),
    }


@app.get("/api/trades/recent")
def trades_recent(portfolio: str = "CanSlim", limit: int = 20):
    """Get most recent trade transactions (buys + sells) plus the
    lot_closures for the trades that appear in the result. Same
    enrichment shape as /api/trades/open/details.

    Response shape: {"details": [...], "lot_closures": [...]}.
    """
    df = db.load_details(portfolio)
    if df.empty:
        return {"details": [], "lot_closures": []}
    df = _normalize_trades(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date", ascending=False)
    sliced = df.head(limit)

    # Closures for the trade_ids that survived the LIMIT slice. Bounds the
    # payload to the same trades the details cover.
    recent_trade_ids = sliced["trade_id"].dropna().astype(str).unique().tolist()
    closures_df = db.load_lot_closures(portfolio, trade_ids=recent_trade_ids)

    return {
        "details": _df_to_records(sliced),
        "lot_closures": _df_to_records(closures_df),
    }


# ============================================================
# LIVE PRICES
# ============================================================
import re as _re

def _is_option_ticker(ticker):
    # Thin alias kept for legacy call sites — canonical impl lives in
    # trade_calc.is_option_ticker so the LIFO engine and price routing share
    # one ticker-pattern definition.
    return is_option_ticker(ticker)

def _to_occ_symbol(readable_ticker):
    try:
        m = _re.match(r'^(\S+)\s+(\d{6})\s+\$([0-9.]+)(C|P)$', readable_ticker.strip())
        if not m:
            return None
        underlying = m.group(1)
        expiry = m.group(2)
        strike = float(m.group(3))
        put_call = m.group(4)
        strike_int = int(strike * 1000)
        return f"{underlying}{expiry}{put_call}{strike_int:08d}"
    except Exception:
        return None


def _fetch_historical_closes(tickers: list[str], target_date) -> dict[str, float]:
    """Return {ticker: close} for a specific past date via yfinance.

    Used by /api/prices/batch when the caller passes a `date` param. Skips
    option tickers (yfinance historical for options is unreliable and the
    Daily Routine form only ever requests SPY/^IXIC). Falls back to the
    most recent close on or before target_date if the exact day isn't a
    trading day (handles weekends/holidays gracefully).
    """
    import yfinance as yf
    out: dict[str, float] = {}
    if not tickers:
        return out
    # Pull a small window (target ± a week) so weekend/holiday lookups still
    # land on the last trading session — single-day yfinance queries can
    # return empty when the requested day is a non-trading day.
    start = (pd.Timestamp(target_date) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(target_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for t in tickers:
        if _is_option_ticker(t):
            continue
        try:
            df = yf.Ticker(t).history(start=start, end=end, auto_adjust=False)
            if df.empty:
                continue
            df = df[df.index.date <= target_date]
            if df.empty:
                continue
            out[t] = round(float(df["Close"].iloc[-1]), 2)
        except Exception:
            continue
    return out


@app.get("/api/prices/lookup")
@limiter.limit("30/minute")
def price_lookup(request: Request, ticker: str = ""):
    """Get live price + ATR for a single ticker. Used by Position Sizer and Log Buy."""
    if not ticker.strip():
        return {"error": "No ticker provided"}
    import yfinance as yf

    try:
        t = ticker.strip().upper()
        stock = yf.Ticker(t)
        df = stock.history(period="40d")
        if df.empty:
            return {"error": f"No data for {t}"}

        price = float(df["Close"].iloc[-1])

        # ATR calculation — matches Streamlit: ATR% = SMA(TR,21) / SMA(Low,21) * 100
        atr_pct = 5.0
        atr_21 = 0.0
        if len(df) >= 21:
            tr = pd.concat([
                df["High"] - df["Low"],
                (df["High"] - df["Close"].shift(1)).abs(),
                (df["Low"] - df["Close"].shift(1)).abs(),
            ], axis=1).max(axis=1)
            sma_tr = float(tr.tail(21).mean())
            sma_low = float(df["Low"].tail(21).mean())
            atr_21 = sma_tr
            if sma_low > 0:
                atr_pct = (sma_tr / sma_low) * 100

        return {
            "ticker": t,
            "price": round(price, 2),
            "atr": round(atr_21, 2),
            "atr_pct": round(atr_pct, 2),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/charts/ohlcv/{ticker}")
@limiter.limit("20/minute")
def chart_ohlcv(request: Request, ticker: str, start: str = "", end: str = "", period: str = "6mo", interval: str = "1d"):
    """Get OHLCV candlestick data for lightweight-charts."""
    import yfinance as yf
    try:
        t = ticker.strip().upper()
        # Map friendly interval names
        interval_map = {"1d": "1d", "1wk": "1wk", "1mo": "1mo", "daily": "1d", "weekly": "1wk", "monthly": "1mo"}
        yf_interval = interval_map.get(interval, "1d")
        stock = yf.Ticker(t)
        if start and end:
            df = stock.history(start=start, end=end, interval=yf_interval)
        else:
            df = stock.history(period=period, interval=yf_interval)
        if df.empty:
            return {"error": f"No data for {t}"}

        candles = []
        for idx, row in df.iterrows():
            ts = int(idx.timestamp()) if hasattr(idx, 'timestamp') else 0
            candles.append({
                "time": ts,
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row.get("Volume", 0)),
            })
        return {"ticker": t, "candles": candles}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/prices/batch")
@limiter.limit("10/minute")
def batch_prices(request: Request, tickers: str = "", portfolio: str = "",
                 date: str = ""):
    """Get prices for a comma-separated list of tickers.
    Supports both stock tickers and readable option format ('LUMN 260717 $8C').
    Returns dict of {readable_ticker: price}.

    `date` (YYYY-MM-DD, optional): when set to a *past* date, returns that
    date's close from yfinance instead of the live price. Used by the Daily
    Routine form when editing a past day so SPY/NDX/etc. reflect that day's
    close, not the live price the user happens to be looking at later.
    Today/future dates fall through to the live path. Manual_price overrides
    are NOT layered in historical mode — they only apply to live snapshots.

    Delegates to the shared PriceProvider in live mode so the Dashboard NLV
    and Active Campaign Current columns are guaranteed to agree — both go
    through the same yfinance path.

    When portfolio is provided (live mode only), manual_price overrides on
    open positions in that portfolio take precedence over the yfinance
    result for matching tickers.
    """
    if not tickers.strip():
        return {}

    # Historical mode: if date is set and is strictly before today (in UTC,
    # which is conservative — yfinance's "today" close isn't finalized until
    # post-close anyway), pull the close for that date.
    if date:
        try:
            target = pd.to_datetime(str(date).strip()[:10], errors="coerce").date()
        except Exception:
            target = None
        if target is not None and target < datetime.now().date():
            return _fetch_historical_closes(
                [t.strip() for t in tickers.split(",") if t.strip()],
                target,
            )

    from price_providers import get_price_provider

    ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]
    yf_to_readable: dict[str, str] = {}
    yf_symbols: list[str] = []
    for t in ticker_list:
        # yfinance can't reliably resolve OCC option symbols, so skip the
        # fetch for options entirely. They stay in ticker_list and get
        # priced by the manual_price override layer below when portfolio
        # is set; options without a manual_price are omitted from the
        # response, matching the prior behavior when the yfinance fetch
        # failed.
        if _is_option_ticker(t):
            continue
        yf_symbols.append(t)
        yf_to_readable[t] = t

    if not yf_symbols:
        # No yfinance work to do, but overrides may still apply if portfolio
        # is set and the requested tickers map to open positions.
        if not portfolio:
            return {}

    try:
        live = (get_price_provider().get_current_prices(yf_symbols)
                if yf_symbols else {})
        # Re-key by the readable ticker so callers can look up by the format
        # they know (e.g. "LUMN 260717 $8C" instead of the OCC-encoded symbol).
        result: dict[str, float] = {
            yf_to_readable.get(yf_sym, yf_sym): price
            for yf_sym, price in live.items()
        }

        # Layer manual_price overrides for open positions in the requested
        # portfolio. Keyed by upper-cased readable ticker.
        if portfolio:
            try:
                summary_df = db.load_summary(portfolio, status="OPEN")
            except Exception:
                summary_df = None
            if summary_df is not None and not summary_df.empty:
                manual_col = (
                    "Manual_Price" if "Manual_Price" in summary_df.columns
                    else ("manual_price" if "manual_price" in summary_df.columns
                          else None)
                )
                ticker_col = "Ticker" if "Ticker" in summary_df.columns else "ticker"
                if manual_col is not None:
                    requested_upper = {t.upper(): t for t in ticker_list}
                    for _, row in summary_df.iterrows():
                        mp = row.get(manual_col)
                        # Filter None AND pandas NaN — load_summary's
                        # Decimal-to-numeric conversion turns DB NULLs into
                        # NaN, which slips past `mp is None` and survives
                        # `float()` + `<= 0`, then crashes the response on
                        # starlette's allow_nan=False JSON encoder.
                        if pd.isna(mp):
                            continue
                        try:
                            mp_f = float(mp)
                        except (TypeError, ValueError):
                            continue
                        if not math.isfinite(mp_f) or mp_f <= 0:
                            continue
                        tkr = str(row.get(ticker_col, "") or "").upper()
                        if tkr in requested_upper:
                            # Re-key with the caller's original casing.
                            result[requested_upper[tkr]] = mp_f
        return result
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# MARKET ENDPOINTS
# ============================================================

# How far past the latest ingested bar we'll project day_num forward when
# the caller asks for a date the data feed hasn't caught up to. Bounded so a
# stuck cron can't silently inflate the count by weeks. 5 trading days = one
# week, which covers the realistic data-lag window (Mon morning before the
# overnight ingest runs, weekend-only outage, etc.) without being absurd.
_RALLY_PREFIX_MAX_PROJECTION_DAYS = 5


def _project_rally_prefix_for_data_lag(response: dict, requested_as_of) -> dict:
    """Project day_num forward when the requested date is past the latest
    ingested bar. The MCT engine reads from market_data; if today's bar
    hasn't ingested yet (cron lag, market still open, etc.), the engine
    reports yesterday's day_num and the Daily Routine prefix lags by one
    trading day.

    Pure function — takes the rally-prefix response dict + the user's
    requested date and returns either the original dict or a mutated copy
    with day_num + prefix bumped forward. Skipped during corrections /
    inactive cycles since day_num is meaningless there.

    Caps offset at _RALLY_PREFIX_MAX_PROJECTION_DAYS. Counts Mon-Fri only —
    market holidays aren't tracked here, accept the rare ±1 error around
    them rather than maintain a holiday calendar in the endpoint layer.
    """
    if requested_as_of is None:
        return response
    if response.get("state") not in ("UPTREND", "POWERTREND", "RALLY MODE"):
        return response
    if (response.get("day_num") or 0) <= 0:
        return response
    data_as_of_iso = response.get("data_as_of")
    if not data_as_of_iso:
        return response

    try:
        from datetime import date as _date, timedelta as _td
        data_dt = _date.fromisoformat(data_as_of_iso)
    except ValueError:
        return response

    if requested_as_of <= data_dt:
        return response

    offset = 0
    cursor = data_dt + _td(days=1)
    while cursor <= requested_as_of and offset < _RALLY_PREFIX_MAX_PROJECTION_DAYS:
        if cursor.weekday() < 5:  # Mon-Fri
            offset += 1
        cursor += _td(days=1)

    if offset == 0:
        return response

    new_day_num = response["day_num"] + offset
    # Preserve the prefix shape — only rebuild it if it was the standard
    # "Day N: " form. Custom or empty prefixes pass through untouched.
    new_prefix = response.get("prefix", "")
    if new_prefix.startswith("Day "):
        new_prefix = f"Day {new_day_num}: "

    return {
        **response,
        "day_num": new_day_num,
        "prefix": new_prefix,
        # Diagnostic flags so callers / future debugging can tell a projected
        # number from a real one. Not consumed by the routine UI; surfaced
        # for inspection in /api/admin/* paths and tests.
        "day_num_projected": True,
        "day_num_projection_offset": offset,
    }


@app.get("/api/market/rally-prefix")
def rally_prefix(as_of_date: str = ""):
    """Get rally day prefix for Market/Global Notes (e.g., 'Day 14 POWERTREND:').

    V11 implementation: replays the MCT engine over market_data history and
    translates the result into the legacy response shape. If as_of_date is
    provided (YYYY-MM-DD), slices history to bars on or before that date.

    When as_of_date sits past the latest ingested market_data bar (typical
    case: opening Daily Routine before today's bar ingests), the response
    is projected forward — day_num is bumped by the trading-day delta,
    prefix is rebuilt accordingly, and `day_num_projected: true` flags the
    synthesis. See _project_rally_prefix_for_data_lag for the exact rule.
    """
    try:
        from datetime import date as _date, datetime as _dt
        from api.mct_endpoint_adapter import run_engine, to_rally_prefix_response

        as_of: _date | None = None
        if as_of_date:
            try:
                as_of = _dt.strptime(as_of_date.strip()[:10], "%Y-%m-%d").date()
            except ValueError:
                as_of = None

        result = run_engine("^IXIC", as_of=as_of)
        response = to_rally_prefix_response(result)
        return _project_rally_prefix_for_data_lag(response, as_of)
    except Exception as e:
        return {"prefix": "", "error": str(e)}


@lru_cache(maxsize=1)
def _get_2025_historical_rally():
    """Build the 4/22/2025 historical reference line under the new
    convention (Day 1 = FTD, baseline = day-before-FTD close).

    Anchored on the 2025-04-22 IXIC FTD. Uses ^IXIC (Nasdaq Composite) —
    must match the index used by the user line so the comparison is
    apples-to-apples. Hardcoded to ^IXIC here even when the live
    rally-data endpoint receives a different `index` query parameter
    (the historical reference is a fixed point of comparison, not a
    function of the user's chosen index).

    The data is immutable (past prices don't change) so we cache the
    result for the lifetime of the worker process. A worker restart
    re-fetches; on yfinance failure the caller catches and serves
    historical_rally_2025 = None (graceful degradation).

    Returns: list[{day, date, close, pct}] of length 25, where Day 1
    is 2025-04-22 (the FTD itself) and Day 25 is 2025-05-27.
    """
    import yfinance as yf
    df = yf.Ticker("^IXIC").history(start="2025-04-15", end="2025-06-15")
    if df.empty:
        raise RuntimeError("No data returned for ^IXIC 2025 historical fetch")

    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    ftd_ts = pd.Timestamp("2025-04-22")
    ftd_mask = df.index.date >= ftd_ts.date()
    df_pre = df[~ftd_mask]
    df_post = df[ftd_mask]
    if df_pre.empty or df_post.empty:
        raise RuntimeError("Missing pre- or post-FTD bars in 2025 historical fetch")

    day_before_ftd_close = float(df_pre.iloc[-1]["Close"])
    out = []
    for i in range(0, min(len(df_post), 25)):
        row = df_post.iloc[i]
        pct = ((float(row["Close"]) / day_before_ftd_close) - 1) * 100
        out.append({
            "day": i + 1,
            "date": df_post.index[i].strftime("%Y-%m-%d"),
            "close": round(float(row["Close"]), 2),
            "pct": round(pct, 2),
        })
    return out


@app.get("/api/market/rally-data")
@limiter.limit("25/minute")
def rally_data(request: Request, ftd_date: str = "", index: str = "^IXIC"):
    """Fetch index closes from FTD through Day 25 for Rally Context chart.

    Day numbering: Day 1 = the FTD itself; baseline for the % gain column
    is the day-before-FTD close (so Day 1's pct shows the FTD's own
    session gain). Returns up to 25 trading days starting at the FTD.

    Response also includes `historical_rally_2025` — the 4/22/2025 IXIC
    reference line under the same convention — so the frontend can
    overlay it without re-fetching. This field is `None` if the
    historical fetch fails; the user line still renders.
    """
    if not ftd_date:
        return {"error": "ftd_date required"}
    try:
        import yfinance as yf
        ftd_ts = pd.Timestamp(ftd_date)
        end_ts = ftd_ts + pd.Timedelta(days=40)
        today_ts = pd.Timestamp.now().normalize()
        if end_ts > today_ts:
            end_ts = today_ts + pd.Timedelta(days=1)

        df = yf.Ticker(index).history(start=ftd_ts - pd.Timedelta(days=5), end=end_ts)
        if df.empty:
            return {"error": "No data from yfinance"}

        df.index = df.index.tz_localize(None) if df.index.tz else df.index
        ftd_mask = df.index.date >= ftd_ts.date()
        df_pre = df[~ftd_mask]
        df_post = df[ftd_mask]
        if df_post.empty:
            return {"error": f"No data on or after {ftd_date}"}
        if df_pre.empty:
            return {"error": f"No data on the trading day before {ftd_date} "
                             f"(needed as baseline for the % gain column)"}

        # Baseline is the day-before-FTD close, not the FTD close.
        # This means Day 1 (the FTD itself) carries its OWN session gain
        # in the pct column rather than a +0.00% no-op.
        day_before_ftd_close = float(df_pre.iloc[-1]["Close"])
        points = []
        for i in range(0, min(len(df_post), 25)):
            row = df_post.iloc[i]
            pct = ((float(row["Close"]) / day_before_ftd_close) - 1) * 100
            points.append({
                "day": i + 1,
                "date": df_post.index[i].strftime("%Y-%m-%d"),
                "close": round(float(row["Close"]), 2),
                "low": round(float(row["Low"]), 2),
                "pct": round(pct, 2),
            })

        # Historical 4/22/2025 reference line. Cached after first call;
        # if the underlying yfinance fetch fails, degrade gracefully so
        # the user line still renders.
        try:
            historical_rally_2025 = _get_2025_historical_rally()
        except Exception as e:
            print(f"[rally-data] 2025 historical fetch failed: {e}")
            historical_rally_2025 = None

        return {
            "day_before_ftd_close": round(day_before_ftd_close, 2),
            "points": points,
            "historical_rally_2025": historical_rally_2025,
        }
    except Exception as e:
        return {"error": str(e)}


# /api/market/mfactor was the V10 MA-stack snapshot endpoint that fed
# Position Sizer + Log Buy's sizing-mode picker. Both surfaces now derive
# their sizing mode from V11 MCT state (rallyPrefix.state) via the
# mctStateToSizingMode helper in @/lib/sizing-mode. Endpoint deleted.


@app.get("/api/market/signals")
@limiter.limit("30/minute")
def get_recent_market_signals(request: Request, days: int = 30, signal_type: str = ""):
    """Last N days of V11 MCT engine signals from market_signals.

    Powers the MCT page's Recent Signal Log (Section 4). Reads the V11
    schema (trade_date, signal_type, signal_label, exposure_before,
    exposure_after, state_before, state_after, meta) from the table
    rebuilt by migration 010. Sorted desc by trade_date then id.
    """
    try:
        rows = db.load_v11_market_signals(
            days=days,
            signal_type=signal_type or None,
        )
        signals = []
        for r in rows:
            td = r["trade_date"]
            signals.append({
                "trade_date": td.isoformat() if hasattr(td, "isoformat") else str(td)[:10],
                "signal_type": r["signal_type"],
                "signal_label": r["signal_label"],
                "exposure_before": r["exposure_before"],
                "exposure_after": r["exposure_after"],
                "state_before": r["state_before"],
                "state_after": r["state_after"],
                "meta": r["meta"],
            })
        return {"signals": signals}
    except Exception as e:
        return {"error": str(e), "signals": []}


# ============================================================
# CONFIG ENDPOINTS
# ============================================================
@app.get("/api/config/{key}")
def get_config(key: str):
    """Get a config value from app_config."""
    try:
        val = db.get_config(key)
        return {"key": key, "value": val}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# HEALTH
# ============================================================
@app.get("/api/trades/lessons")
def get_trade_lessons(portfolio: str = "CanSlim"):
    """Get all trade lesson notes for a portfolio."""
    try:
        lessons = db.get_trade_lessons(portfolio)
        return {"lessons": {k: {"note": v[0], "category": v[1]} for k, v in lessons.items()}}
    except Exception as e:
        return {"lessons": {}, "error": str(e)}


@app.post("/api/trades/lessons")
def save_trade_lesson(entry: dict):
    """Save a trade lesson note."""
    try:
        portfolio = entry.get("portfolio", "CanSlim")
        trade_id = entry.get("trade_id", "")
        note = entry.get("note", "")
        category = entry.get("category", "")
        ok = db.save_trade_lesson(portfolio, trade_id, note, category)
        return {"status": "ok" if ok else "error"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/api/health")
def health():
    """Health check."""
    try:
        ok = db.test_connection()
        return {"status": "ok" if ok else "db_error", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/api/healthz")
def healthz():
    """Lightweight liveness probe — does not check downstream dependencies. Use /api/health for readiness."""
    return {"status": "ok"}


# ============================================================
# PORTFOLIOS — multi-tenant CRUD
# ============================================================
def _serialize_portfolio(row: dict) -> dict:
    """Make a portfolio row JSON-safe (dates → ISO, Decimals → float)."""
    if row.get("created_at") is not None:
        row["created_at"] = row["created_at"].isoformat()
    if row.get("reset_date") is not None:
        row["reset_date"] = row["reset_date"].isoformat()
    if row.get("starting_capital") is not None:
        row["starting_capital"] = float(row["starting_capital"])
    if row.get("cash_balance") is not None:
        row["cash_balance"] = float(row["cash_balance"])
    return row


def _serialize_cash_tx(row: dict) -> dict:
    """Make a cash_transactions row JSON-safe."""
    if row.get("date") is not None:
        row["date"] = row["date"].isoformat()
    if row.get("created_at") is not None:
        row["created_at"] = row["created_at"].isoformat()
    if row.get("amount") is not None:
        row["amount"] = float(row["amount"])
    return row


def _ensure_user_owns_portfolio(portfolio_id: int) -> bool:
    """RLS will hide portfolios the user doesn't own, so list_portfolios()
    returning no match for a given id implies 404."""
    return any(p["id"] == portfolio_id for p in db.list_portfolios())


@app.get("/api/strategies")
def list_strategies_endpoint(active: bool = True):
    """Return rows from the strategies lookup table (Migration 019).

    `active` (default true) filters to is_active=true — what the Log Buy
    dropdown wants. Pass `?active=false` to include disabled strategies
    (used by Phase 2 admin UI). Strategies are global, not per-user, so
    this endpoint is safe to call without a portfolio context.
    """
    try:
        rows = db.load_strategies(active_only=active)
        return [_serialize_strategy(r) for r in rows]
    except Exception as e:
        return {"error": str(e)}


def _serialize_strategy(r: dict) -> dict:
    """Shared serialization for strategy rows. Centralized so GET, POST,
    and PUT all return the same shape (name, description, color,
    is_active, created_at-as-isoformat)."""
    return {
        "name": r["name"],
        "description": r.get("description"),
        "color": r["color"],
        "is_active": r["is_active"],
        "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
    }


# Phase 2 — strategies CRUD (founder-gated). Single-tenant for now;
# strategies are global, so a non-founder writing them would mutate state
# everyone sees. Mirrors the gate already in place on /api/admin/* paths.
@app.post("/api/strategies")
@limiter.limit("5/minute")
def create_strategy_endpoint(request: Request, body: dict = Body(...)):
    """Create a new strategy. Founder-only.

    Body: { name, color, description?, is_active? }. Validates hex color
    server-side regardless of what the client sends — frontend mirrors the
    same regex but never trust the client. Returns the persisted row, or
    an {error} body the frontend reads.
    """
    user_id = getattr(request.state, "user_id", None)
    if user_id != FOUNDER_USER_ID:
        return {"error": "forbidden_not_admin"}
    try:
        row = db.create_strategy(
            name=body.get("name", ""),
            color=body.get("color", ""),
            description=body.get("description"),
            is_active=bool(body.get("is_active", True)),
        )
        return _serialize_strategy(row)
    except ValueError as e:
        return {"error": str(e)}
    except psycopg2.errors.UniqueViolation:
        return {"error": f"Strategy '{body.get('name', '')}' already exists"}
    except Exception as e:
        return {"error": str(e)}


@app.put("/api/strategies/{name}")
@limiter.limit("10/minute")
def update_strategy_endpoint(name: str, request: Request, body: dict = Body(...)):
    """Update a strategy's description, color, or is_active. Founder-only.

    Name is NOT updatable here — renaming would cascade through every
    trades_summary.strategy via the FK's ON UPDATE CASCADE, which is fine
    for the DB but a UX/audit minefield (every audit row's "Strategy: X"
    blob would be retroactively wrong). Defer rename to a future explicit
    "rename strategy" flow.
    """
    user_id = getattr(request.state, "user_id", None)
    if user_id != FOUNDER_USER_ID:
        return {"error": "forbidden_not_admin"}
    try:
        # Pass through only the recognised keys so a typo'd body field
        # doesn't silently no-op (caller sees the unchanged row and can
        # diagnose). The helper itself ignores unrecognised keys.
        fields = {k: body[k] for k in ("description", "color", "is_active") if k in body}
        row = db.update_strategy(name, **fields)
        if row is None:
            return {"error": f"Strategy '{name}' not found"}
        return _serialize_strategy(row)
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


# Phase 2 — retroactive tagging (NOT founder-gated). Per-trade ops; any
# authenticated owner of the portfolio can retag their own campaigns.
@app.patch("/api/trades/{trade_id}/strategy")
@limiter.limit("30/minute")
def patch_trade_strategy_endpoint(trade_id: str, request: Request, body: dict = Body(...)):
    """Retag a single campaign with a new strategy.

    Body: { strategy: string, portfolio?: string }. Strategy is validated
    against the active list (matches log_buy's contract — you can't tag a
    trade with an inactive strategy via this endpoint, even if the row
    already references one).
    """
    portfolio = body.get("portfolio", "CanSlim")
    strategy = (body.get("strategy") or "").strip()
    if not strategy:
        return {"error": "Missing strategy"}
    try:
        valid = {s["name"] for s in db.load_strategies(active_only=True)}
        if strategy not in valid:
            return {"error": f"Unknown strategy: {strategy}"}
        ok = db.update_trade_strategy(portfolio, trade_id, strategy)
        if not ok:
            return {"error": f"Trade {trade_id} not found in {portfolio}"}
        try:
            db.log_audit(portfolio, "STRATEGY_TAG", trade_id, "",
                         f"strategy → {strategy}", username="web")
        except Exception:
            pass
        return {"ok": True, "trade_id": trade_id, "strategy": strategy}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/trades/bulk-strategy")
@limiter.limit("5/minute")
def bulk_set_trade_strategy_endpoint(request: Request, body: dict = Body(...)):
    """Retag many campaigns at once.

    Body: { trade_ids: [string], strategy: string, portfolio?: string }.

    Failure semantics (per Phase 2 design):
      - Strategy validation failure → reject ENTIRE batch up-front
        (no partial commits). One up-front lookup against the active set.
      - Missing trade_id (subset of trade_ids not found) → commit the
        valid ones and return failed: [trade_ids] for the missing ones.
        The single UPDATE … WHERE trade_id = ANY(%s) is atomic for the
        rows it does match, and missing_ids are computed from the
        RETURNING set vs the input.
    """
    portfolio = body.get("portfolio", "CanSlim")
    strategy = (body.get("strategy") or "").strip()
    trade_ids = body.get("trade_ids") or []
    if not strategy:
        return {"error": "Missing strategy"}
    if not isinstance(trade_ids, list) or not trade_ids:
        return {"error": "Missing trade_ids"}
    try:
        # All-or-nothing on strategy validation — if the strategy is bad,
        # the whole batch is rejected before any UPDATE runs.
        valid = {s["name"] for s in db.load_strategies(active_only=True)}
        if strategy not in valid:
            return {"error": f"Unknown strategy: {strategy}"}
        updated, missing = db.bulk_update_trade_strategy(portfolio, trade_ids, strategy)
        try:
            db.log_audit(portfolio, "STRATEGY_BULK_TAG", "",
                         "", f"strategy → {strategy}: {updated} updated, "
                         f"{len(missing)} missing", username="web")
        except Exception:
            pass
        return {"ok": True, "updated": updated, "failed": missing, "strategy": strategy}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/portfolios")
def list_portfolios_endpoint():
    """Return portfolios owned by the authenticated user. Empty list for a
    brand-new user until they create one. The frontend uses the empty state
    to render the onboarding screen."""
    try:
        rows = db.list_portfolios()
        return [_serialize_portfolio(r) for r in rows]
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/portfolios")
@limiter.limit("5/minute")
def create_portfolio_endpoint(request: Request, body: dict = Body(...)):
    """Create a new portfolio for the authenticated user. Name is required
    and unique per user; starting_capital and reset_date are optional."""
    try:
        name = body.get("name", "")
        starting_capital = body.get("starting_capital")
        reset_date = body.get("reset_date") or None
        row = db.create_portfolio(name, starting_capital=starting_capital, reset_date=reset_date)
        return _serialize_portfolio(row)
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


@app.put("/api/portfolios/{portfolio_id}")
@limiter.limit("30/minute")
def update_portfolio_endpoint(portfolio_id: int, request: Request, body: dict = Body(...)):
    """Update a portfolio the authenticated user owns. Only the fields
    present in the body are modified (name, starting_capital, reset_date)."""
    try:
        row = db.update_portfolio(
            portfolio_id,
            name=body.get("name"),
            starting_capital=body.get("starting_capital"),
            reset_date=body.get("reset_date") or None,
        )
        if row is None:
            return {"error": "Portfolio not found"}
        return _serialize_portfolio(row)
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


@app.delete("/api/portfolios/{portfolio_id}")
@limiter.limit("10/minute")
def delete_portfolio_endpoint(portfolio_id: int, request: Request):
    """Delete a portfolio the authenticated user owns. Cascades to all
    trades, journal entries, and snapshots under it — irreversible."""
    try:
        deleted = db.delete_portfolio(portfolio_id)
        if not deleted:
            return {"error": "Portfolio not found"}
        return {"status": "ok"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/portfolios/{portfolio_id}/nlv")
@limiter.limit("30/minute")
def get_portfolio_nlv(portfolio_id: int, request: Request):
    """Live NLV snapshot: cash + Σ(open position market values).

    Live prices come from the configured PriceProvider (yfinance today).
    Positions with unresolved prices fall back to cost basis and are flagged
    with price_unavailable: true so the UI can show a warning.
    """
    try:
        rows = db.list_portfolios()
        match = next((r for r in rows if r["id"] == portfolio_id), None)
        if match is None:
            return {"error": "Portfolio not found"}
        return nlv_service.compute_nlv(portfolio_id, match["name"])
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/portfolios/{portfolio_id}/returns")
@limiter.limit("30/minute")
def get_portfolio_returns(portfolio_id: int, request: Request):
    """LTD + YTD P&L in dollars and percent.

    Derived from the cash_transactions ledger (net contributions) and the
    live NLV. YTD is only meaningful when the portfolio started this year
    OR when an EOD snapshot exists for a prior year-end (Phase 4); until
    then it reports ytd_available = false so the UI can show '—'.
    """
    try:
        rows = db.list_portfolios()
        match = next((r for r in rows if r["id"] == portfolio_id), None)
        if match is None:
            return {"error": "Portfolio not found"}
        return nlv_service.compute_returns(portfolio_id, match["name"], match)
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/portfolios/{portfolio_id}/twr-returns")
@limiter.limit("30/minute")
def get_portfolio_twr_returns(portfolio_id: int, request: Request):
    """Time-weighted LTD + YTD chained from daily journal returns.

    Complements /returns: that endpoint answers 'total profit as a % of
    contributions' (snapshot, ignores cash-flow timing); this one answers
    'what compound return did the strategy produce' by chaining daily
    returns with the flow-at-start-of-day Modified Dietz formula. Both
    coexist so the dashboard headline can show TWR while a future tile
    can still surface the snapshot ratio if needed.
    """
    try:
        rows = db.list_portfolios()
        match = next((r for r in rows if r["id"] == portfolio_id), None)
        if match is None:
            return {"error": "Portfolio not found"}
        return nlv_service.compute_twr_returns(match["name"])
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/portfolios/{portfolio_id}/dashboard-metrics")
@limiter.limit("30/minute")
def get_dashboard_metrics(portfolio_id: int, request: Request):
    """Aggregated read view powering the dashboard. journal.end_nlv is
    the single source of truth for nlv / drawdown / exposure / cash.
    See nlv_service.dashboard_metrics for the field-level contract."""
    try:
        rows = db.list_portfolios()
        match = next((r for r in rows if r["id"] == portfolio_id), None)
        if match is None:
            return {"error": "Portfolio not found"}
        return nlv_service.dashboard_metrics(portfolio_id, match["name"])
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/portfolios/{portfolio_id}/cash-transactions")
@limiter.limit("60/minute")
def list_cash_transactions_endpoint(portfolio_id: int, request: Request,
                                    limit: int = 50,
                                    exclude_trade_rows: bool = False):
    """Return recent cash_transactions for a portfolio, newest first. Used
    by the Settings cash-activity view.

    exclude_trade_rows=true filters buy/sell at the SQL layer so backdated
    deposits/withdrawals can't be hidden behind hundreds of recent trades.
    """
    try:
        if not _ensure_user_owns_portfolio(portfolio_id):
            return {"error": "Portfolio not found"}
        rows = db.list_cash_transactions(
            portfolio_id, limit=limit, exclude_trade_rows=exclude_trade_rows,
        )
        return [_serialize_cash_tx(r) for r in rows]
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/portfolios/{portfolio_id}/cash-transactions")
@limiter.limit("30/minute")
def create_cash_transaction_endpoint(portfolio_id: int, request: Request, body: dict = Body(...)):
    """Insert a user-initiated cash transaction — deposit, withdrawal, or
    broker reconcile.

    Body shape:
        {
          "source":  "deposit" | "withdraw" | "reconcile",
          "amount":  <positive number — sign is derived from source>,
          "date":    "<ISO date|datetime>",  (optional, defaults to now)
          "note":    "<free text>"           (optional)
        }

    For `source = 'reconcile'`, amount is interpreted as the user's ACTUAL
    broker cash balance. The system computes (actual - system) and writes a
    signed reconcile row for the delta, absorbing commissions, interest, and
    any other drift in a single line without per-trade fee tracking.

    Buy/sell sources are reserved for save_detail_row and rejected here.
    """
    try:
        if not _ensure_user_owns_portfolio(portfolio_id):
            return {"error": "Portfolio not found"}

        source = (body.get("source") or "").lower()
        if source not in ("deposit", "withdraw", "reconcile"):
            return {"error": "source must be deposit, withdraw, or reconcile"}

        # body.get("amount") could be 0 — using `or 0` would mask it. Read raw
        # and reject only None/non-numeric so reconcile with broker_balance=0
        # is accepted (rare but legal: just-funded margin account).
        raw_amount = body.get("amount")
        if raw_amount is None:
            return {"error": "amount is required"}
        try:
            amount = float(raw_amount)
        except (TypeError, ValueError):
            return {"error": "amount must be numeric"}
        # Deposit/withdraw amounts must be positive (the source determines the
        # sign). Reconcile takes the user's actual broker cash balance, which
        # can be negative (margin) or zero — must not be filtered here or
        # users with margin debits can never reconcile.
        if source != "reconcile" and amount <= 0:
            return {"error": "amount must be greater than zero"}

        note = body.get("note") or None
        date = body.get("date") or None  # None → db_layer defaults to now()

        if source == "deposit":
            signed = amount
        elif source == "withdraw":
            signed = -amount
        else:  # reconcile — amount is the user's actual broker cash balance
            current = db.get_cash_balance(portfolio_id)
            signed = amount - current
            if abs(signed) < 0.01:
                # No drift to record; surface that to the user instead of
                # silently inserting a zero-dollar row.
                return {"status": "noop", "delta": 0.0, "message": "Already in sync with broker"}
            if note is None:
                note = f"Reconcile: system had ${current:,.2f}, broker has ${amount:,.2f}"

        row = db.insert_cash_transaction(
            portfolio_id, signed, source, date=date, note=note,
        )
        return _serialize_cash_tx(row)
    except Exception as e:
        return {"error": str(e)}


_INITIAL_CAPITAL_NOTE_PREFIX = "Initial capital"


def _is_user_editable_cash_tx(row: dict) -> tuple[bool, str | None]:
    """Cash rows fall into three buckets re: user-driven mutation:
      - buy/sell: managed by save_detail_row; edit the trade itself, not this row
      - 'Initial capital' deposit: managed by _sync_initial_deposit; edit
        starting_capital + reset_date in portfolio settings instead
      - everything else (deposit/withdraw/reconcile, non-initial): freely editable
    Returns (editable, reason_when_not_editable).
    """
    src = (row.get("source") or "").lower()
    if src in ("buy", "sell"):
        return False, "Buy/sell rows are managed by the trade itself — edit the trade to change them."
    note = row.get("note") or ""
    if src == "deposit" and note.startswith(_INITIAL_CAPITAL_NOTE_PREFIX):
        return False, "Initial capital is managed via Starting capital + Reset date on the portfolio."
    return True, None


@app.patch("/api/portfolios/{portfolio_id}/cash-transactions/{tx_id}")
@limiter.limit("30/minute")
def update_cash_transaction_endpoint(portfolio_id: int, tx_id: int,
                                     request: Request, body: dict = Body(...)):
    """Patch amount / date / note on an existing deposit/withdraw/reconcile row.
    Source is immutable. Buy/sell + initial-capital rows are protected.

    Body (all optional, only present fields are written):
        { "amount": <positive>, "date": "<ISO>", "note": "<text|null>" }

    For withdraw rows, the stored amount is signed-negative — the caller passes
    a positive number and we re-apply the sign based on the existing source.
    """
    try:
        if not _ensure_user_owns_portfolio(portfolio_id):
            return {"error": "Portfolio not found"}

        existing = db.get_cash_transaction(tx_id)
        if existing is None or existing.get("portfolio_id") != portfolio_id:
            return {"error": "Transaction not found"}

        editable, reason = _is_user_editable_cash_tx(existing)
        if not editable:
            return {"error": reason}

        signed_amount = None
        if "amount" in body and body["amount"] is not None:
            try:
                amt = float(body["amount"])
            except (TypeError, ValueError):
                return {"error": "amount must be numeric"}
            if amt <= 0:
                return {"error": "amount must be greater than zero"}
            src = (existing.get("source") or "").lower()
            signed_amount = -amt if src == "withdraw" else amt

        date = body.get("date") if "date" in body else None
        note = body.get("note") if "note" in body else None

        row = db.update_cash_transaction(
            tx_id, amount=signed_amount, date=date, note=note,
        )
        if row is None:
            return {"error": "Transaction not found"}
        return _serialize_cash_tx(row)
    except Exception as e:
        return {"error": str(e)}


@app.delete("/api/portfolios/{portfolio_id}/cash-transactions/{tx_id}")
@limiter.limit("30/minute")
def delete_cash_transaction_endpoint(portfolio_id: int, tx_id: int, request: Request):
    """Delete a deposit/withdraw/reconcile row. Buy/sell + initial-capital
    rows are protected (same rules as PATCH)."""
    try:
        if not _ensure_user_owns_portfolio(portfolio_id):
            return {"error": "Portfolio not found"}

        existing = db.get_cash_transaction(tx_id)
        if existing is None or existing.get("portfolio_id") != portfolio_id:
            return {"error": "Transaction not found"}

        editable, reason = _is_user_editable_cash_tx(existing)
        if not editable:
            return {"error": reason}

        ok = db.delete_cash_transaction(tx_id)
        return {"status": "ok"} if ok else {"error": "Transaction not found"}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# ADMIN — CONFIG WRITE
# ============================================================
@app.post("/api/config/{key}")
def set_config(key: str, body: dict):
    """Set a config value."""
    try:
        value = body.get("value")
        value_type = body.get("value_type")
        category = body.get("category")
        description = body.get("description")
        user = body.get("user", "admin")
        ok = db.set_config(key, value, value_type=value_type, category=category,
                           description=description, user=user)
        if ok:
            return {"status": "ok"}
        return {"status": "error", "detail": "set_config returned False"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ============================================================
# ADMIN — DASHBOARD EVENTS
# ============================================================
@app.delete("/api/events/auto-cleanup")
def cleanup_auto_events():
    """Remove all auto-generated events (e.g. RESET_DATE) from dashboard_events."""
    try:
        with db.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM dashboard_events WHERE auto_generated = TRUE")
                deleted = cur.rowcount
                conn.commit()
        return {"status": "ok", "deleted": deleted}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/events")
def list_events(scope: str = "CanSlim"):
    """List dashboard events."""
    try:
        df = db.load_dashboard_events(scope=scope)
        if df.empty:
            return []
        return _df_to_records(df)
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/events")
def add_event(body: dict):
    """Add a dashboard event."""
    try:
        ok = db.save_dashboard_event(
            event_date=body["event_date"],
            label=body["label"],
            category=body.get("category", "market"),
            notes=body.get("notes", ""),
            scope=body.get("scope", "CanSlim"),
            auto_generated=False,
            user=body.get("user", "admin"),
        )
        return {"status": "ok" if ok else "error"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.put("/api/events/{event_id}")
def update_event(event_id: int, body: dict):
    """Update a dashboard event."""
    try:
        ok = db.update_dashboard_event(
            event_id=event_id,
            event_date=body.get("event_date"),
            label=body.get("label"),
            category=body.get("category"),
            notes=body.get("notes"),
            user=body.get("user", "admin"),
        )
        return {"status": "ok" if ok else "error"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.delete("/api/events/{event_id}")
def delete_event(event_id: int, user: str = "admin"):
    """Delete a dashboard event."""
    try:
        ok = db.delete_dashboard_event(event_id, user=user)
        return {"status": "ok" if ok else "error"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ============================================================
# ADMIN — AUDIT TRAIL
# ============================================================
@app.get("/api/audit")
def audit_trail(limit: int = 100, action_filter: str = None):
    """Get recent audit trail entries."""
    try:
        if not hasattr(db, "load_recent_audit_entries"):
            return []
        df = db.load_recent_audit_entries(limit=limit, action_filter=action_filter)
        if df.empty:
            return []
        return _df_to_records(df)
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# ADMIN — DATA CLEANUP
# ============================================================
@app.post("/api/admin/cleanup-marketsurge")
@limiter.limit("5/minute")
def cleanup_marketsurge(request: Request, body: dict):
    """Clean up duplicate MarketSurge images."""
    try:
        if not hasattr(db, "cleanup_duplicate_marketsurge_images"):
            return {"error": "Cleanup function not available"}
        dry_run = body.get("dry_run", True)
        result = db.cleanup_duplicate_marketsurge_images(dry_run=dry_run, user="admin")
        return result
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/admin/rebuild-mct-signals")
@limiter.limit("3/minute")
def rebuild_mct_signals(request: Request):
    """Atomically DELETE + re-INSERT all market_signals rows from a fresh
    full-history engine run.

    Use after engine logic changes that obsolete previously persisted signals
    (write_signals' ON CONFLICT DO NOTHING semantics can't notice when a
    signal's date shifts). One transaction — table is never empty mid-rebuild.
    """
    try:
        from datetime import date as _date
        from api.mct_engine import MCTEngine
        from api.market_data_repo import get_history, get_latest_date
        from api.mct_endpoint_adapter import _default_config
        from api.mct_signals_writer import rebuild_signals

        symbol = "^IXIC"
        latest = get_latest_date(symbol)
        if latest is None:
            return {"error": "no market_data rows for symbol"}

        history = get_history(symbol, _date(2010, 1, 1), latest)
        if history.empty:
            return {"error": "empty history"}

        config = _default_config(float(history["high"].iloc[0]))
        result = MCTEngine(config).run(history)
        summary = rebuild_signals(result.signals)

        first = result.signals[0].trade_date if result.signals else None
        last = result.signals[-1].trade_date if result.signals else None
        return {
            "deleted": summary["deleted"],
            "inserted": summary["inserted"],
            "events_emitted": summary["events_emitted"],
            "first_signal_date": first.isoformat() if first else None,
            "last_signal_date": last.isoformat() if last else None,
            "bars_processed": len(result.bars),
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}


# ============================================================
# AI COACH — STREAMING CHAT
# ============================================================
from fastapi.responses import StreamingResponse
import json as _json

def _build_trade_context(portfolio: str, scope: str = "recent", n: int = 10):
    """Build trade data context for AI coach prompts."""
    parts = []
    df_s = db.load_summary(portfolio)
    df_d = db.load_details(portfolio)
    if not df_s.empty:
        df_s = _normalize_trades(df_s)
        closed = df_s[df_s["status"].str.upper() == "CLOSED"].copy()
        if "closed_date" in closed.columns:
            closed["closed_date"] = pd.to_datetime(closed["closed_date"], errors="coerce")
            closed = closed.sort_values("closed_date", ascending=False)
        open_t = df_s[df_s["status"].str.upper() == "OPEN"].copy()
        show = closed.head(n) if scope == "recent" else closed
        if not show.empty:
            cols = ["trade_id", "ticker", "open_date", "closed_date", "shares",
                    "avg_entry", "avg_exit", "realized_pl", "return_pct",
                    "rule", "sell_rule", "stop_loss"]
            use = [c for c in cols if c in show.columns]
            parts.append(f"=== CLOSED TRADES ({len(show)}) ===")
            parts.append(show[use].to_csv(index=False))
        if not open_t.empty:
            cols = ["trade_id", "ticker", "open_date", "shares",
                    "avg_entry", "total_cost", "unrealized_pl", "rule", "stop_loss"]
            use = [c for c in cols if c in open_t.columns]
            parts.append(f"\n=== OPEN TRADES ({len(open_t)}) ===")
            parts.append(open_t[use].to_csv(index=False))
    return "\n".join(parts)


def _build_journal_context(portfolio: str, n: int = 14):
    """Build journal context for AI coach prompts."""
    df = db.load_journal(portfolio)
    if df.empty:
        return "No journal data available."
    df = _normalize_journal(df)
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    df = df.sort_values("day", ascending=False).head(n)
    parts = [f"=== JOURNAL ENTRIES (last {len(df)} days) ==="]
    for _, row in df.iterrows():
        day = row["day"].strftime("%Y-%m-%d") if pd.notna(row.get("day")) else "?"
        entry = f"\n--- {day} ---"
        for field in ["status", "market_window", "market_action", "market_notes",
                      "daily_dollar_change", "daily_pct_change", "end_nlv", "pct_invested",
                      "portfolio_heat", "score", "highlights", "lowlights", "mistakes", "top_lesson"]:
            val = row.get(field)
            if val is not None and str(val).strip() and str(val) != "nan":
                entry += f"\n  {field}: {val}"
        parts.append(entry)
    return "\n".join(parts)


def _build_stats_context(portfolio: str):
    """Build aggregate stats for AI coach."""
    parts = ["=== PORTFOLIO STATS ==="]
    df_s = db.load_summary(portfolio)
    if not df_s.empty:
        df_s = _normalize_trades(df_s)
        closed = df_s[df_s["status"].str.upper() == "CLOSED"]
        open_t = df_s[df_s["status"].str.upper() == "OPEN"]
        if not closed.empty and "realized_pl" in closed.columns:
            pl = pd.to_numeric(closed["realized_pl"], errors="coerce").fillna(0)
            wins = pl[pl > 0]
            losses = pl[pl < 0]
            parts.append(f"Total closed trades: {len(closed)}")
            parts.append(f"Open positions: {len(open_t)}")
            if len(closed) > 0:
                parts.append(f"Win rate: {len(wins)}/{len(closed)} ({len(wins)/len(closed)*100:.1f}%)")
            parts.append(f"Total realized P&L: ${pl.sum():,.2f}")
            if len(wins) > 0:
                parts.append(f"Avg win: ${wins.mean():,.2f}")
            if len(losses) > 0:
                parts.append(f"Avg loss: ${losses.mean():,.2f}")
            parts.append(f"Best trade: ${pl.max():,.2f}")
            parts.append(f"Worst trade: ${pl.min():,.2f}")
    df_j = db.load_journal(portfolio)
    if not df_j.empty:
        df_j = _normalize_journal(df_j)
        for c in ["end_nlv"]:
            if c in df_j.columns:
                df_j[c] = pd.to_numeric(df_j[c], errors="coerce").fillna(0)
        df_j["day"] = pd.to_datetime(df_j["day"], errors="coerce")
        latest = df_j.sort_values("day", ascending=False).iloc[0]
        parts.append(f"\nLatest NLV: ${float(latest.get('end_nlv', 0)):,.0f}")
    return "\n".join(parts)


COACH_SYSTEM_PROMPT = """You are MO's personal AI trading coach. You analyze his CANSLIM trading journal data and provide actionable insights.

TRADING SYSTEM CONTEXT:
- MO trades the CANSLIM / IBD growth stock strategy
- Buy rules: Base breakouts, volume events (HVE/HVSI), moving average reclaims, pullbacks, gap-ups, trendline breaks
- Sell rules: Capital protection (stop loss), selling into strength, portfolio management, change of character, breakout failure
- Position sizing: Risk-based with stop losses and risk budgets
- Transaction IDs: B1/B2 = initial buys, A1/A2 = add-on buys, S1/S2 = sells
- Trade IDs format: YYYYMM-NNN (e.g., 202602-001)

ANALYSIS GUIDELINES:
- Be specific — reference actual tickers, dates, and numbers from the data
- Focus on patterns, not individual trades (unless asked)
- Highlight both strengths and areas for improvement
- Frame feedback constructively — you're a coach, not a critic
- When discussing sell discipline, compare actual exit vs. stop loss and optimal exit
- Keep responses concise and actionable — use bullet points
- Use dollar amounts and percentages to quantify insights
- If the data is insufficient to answer, say so honestly

Current portfolio: CanSlim
"""


@app.post("/api/coach/chat")
@limiter.limit("20/minute")
def coach_chat(request: Request, body: dict):
    """AI Coach chat endpoint — streams response via SSE."""
    try:
        import anthropic
    except ImportError:
        return {"error": "anthropic package not installed"}

    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return {"error": "Anthropic API key not configured"}

        client = anthropic.Anthropic(api_key=api_key)
        user_msg = body.get("message", "")
        preset = body.get("preset")
        portfolio = body.get("portfolio", "CanSlim")

        # Build context based on preset or free-form
        if preset == "recent":
            context = _build_trade_context(portfolio, "all")
            if not user_msg:
                user_msg = """Review my recent closed trades. For each trade, analyze:
1. Entry quality — was the buy rule appropriate? Was timing good?
2. Exit execution — did I follow my sell rules? Did I sell too early or too late?
3. Position sizing — was the risk budget reasonable?
Then give me an overall grade (A-F) and your top 3 actionable recommendations."""
        elif preset == "mistakes":
            context = _build_journal_context(portfolio, 30)
            if not user_msg:
                user_msg = """Analyze my journal entries looking for recurring mistake patterns.
Group the mistakes into categories and rank them by frequency and cost.
For each pattern: what the mistake is, how often it occurs, what it costs, and a specific fix.
Also look at my lowlights entries for additional patterns."""
        elif preset == "weekly":
            context = _build_journal_context(portfolio, 14)
            if not user_msg:
                user_msg = """Give me a coaching summary for the current trading week. Cover:
1. P&L performance and equity trend
2. What I did well (from highlights)
3. What needs work (from lowlights/mistakes)
4. Market conditions and how I adapted
5. One key lesson or focus for next week"""
        elif preset == "behavior":
            context = _build_trade_context(portfolio, "all") + "\n\n" + _build_journal_context(portfolio, 30) + "\n\n" + _build_stats_context(portfolio)
            if not user_msg:
                user_msg = """Analyze my trading behavior patterns. Look at:
1. Do I add to winners or average down on losers?
2. Am I cutting losses quickly or holding too long?
3. Am I selling winners too early?
4. Position sizing patterns
5. Time patterns — am I more profitable on certain setups?
Give me a behavioral profile and 3 specific things to work on."""
        else:
            # Free-form: include stats + trades + journal
            lower = user_msg.lower()
            journal_days = 30
            if any(w in lower for w in ["all", "history", "overall", "pattern", "behavior", "trend", "year"]):
                journal_days = 90
            elif any(w in lower for w in ["week", "recent", "last", "today", "yesterday"]):
                journal_days = 14
            context = _build_stats_context(portfolio) + "\n\n" + _build_trade_context(portfolio, "all") + "\n\n" + _build_journal_context(portfolio, journal_days)

        today_str = date.today().strftime("%Y-%m-%d")
        system = COACH_SYSTEM_PROMPT + f"\nToday's date: {today_str}"
        full_msg = f"{context}\n\n{user_msg}"

        def stream_gen():
            try:
                with client.messages.stream(
                    model="claude-sonnet-4-6",
                    max_tokens=2048,
                    system=system,
                    messages=[{"role": "user", "content": full_msg}],
                ) as stream:
                    for text in stream.text_stream:
                        yield f"data: {_json.dumps({'text': text})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {_json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(stream_gen(), media_type="text/event-stream")
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# ============================================================
# TRADE WRITE ENDPOINTS (Log Buy / Log Sell)
# ============================================================
@app.get("/api/trades/next-id")
def next_trade_id(portfolio: str = "CanSlim", date: str = ""):
    """Generate next available trade ID for a given month.

    Counts BOTH active and soft-deleted summaries when computing
    max(seq) + 1, so a deleted trade_id is never recycled to a new
    campaign. Gaps in the visible sequence are expected and recoverable —
    any missing number corresponds to a soft-deleted summary, queryable
    via WHERE deleted_at IS NOT NULL.
    """
    try:
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        ym = pd.Timestamp(date).strftime("%Y%m")
        existing = db.load_all_trade_ids_for_month(portfolio, ym)
        seqs = []
        for x in existing:
            try:
                if "-" in x:
                    seqs.append(int(x.split("-")[-1]))
            except:
                pass
        next_seq = (max(seqs) + 1) if seqs else 1
        return {"trade_id": f"{ym}-{next_seq:03d}"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/ibkr/status")
def ibkr_status():
    """Check if IBKR credentials are configured."""
    token = os.environ.get("IBKR_FLEX_TOKEN", "")
    query_id = os.environ.get("IBKR_FLEX_QUERY_ID", "")
    nav_query_id = os.environ.get("IBKR_NAV_FLEX_QUERY_ID", "")
    # Also check all env var keys that contain IBKR (case-insensitive)
    ibkr_vars = {k: f"{v[:4]}..." for k, v in os.environ.items() if "IBKR" in k.upper() or "FLEX" in k.upper()}
    return {
        "token_set": bool(token),
        "query_id_set": bool(query_id),
        "nav_query_id_set": bool(nav_query_id),
        "token_preview": f"{token[:4]}...{token[-4:]}" if len(token) > 8 else "(empty)",
        "env_keys_found": ibkr_vars,
        "total_env_vars": len(os.environ),
    }


@app.get("/api/ibkr/nav-for-date")
@limiter.limit("12/minute")
def ibkr_nav_for_date(request: Request, date: str = ""):
    """Pull broker-reported NAV for a single day from IBKR Flex Query.

    Powers Daily Routine's auto-fill of End NLV. Always returns 200 OK; on
    failure the body has {success: false, error, message} so the frontend can
    render a fallback banner without HTTP-error parsing. The frontend
    discriminates on `error` (e.g. ibkr_not_configured vs no_data_for_date)
    and shows `message` to the user verbatim.

    Query: ?date=YYYY-MM-DD (optional). If omitted, defaults to the last
    completed trading day per ibkr_flex._last_completed_trading_day().
    """
    try:
        import ibkr_flex
        target = date.strip() or None
        result = ibkr_flex.fetch_nav_for_date(target_date=target)
        return {
            "success": True,
            "nav": result["nav"],
            "cash_balance": result["cash_balance"],
            "position_value": result["position_value"],
            "currency": result.get("currency", "USD"),
            "account": result.get("account", ""),
            "report_date": result["date"],
            "source": "ibkr_flex_query",
        }
    except Exception as e:
        # ibkr_flex.FlexQueryError carries a stable code + human message;
        # anything else gets bucketed as 'unknown_error' so the frontend
        # always has both fields to surface.
        code = getattr(e, "code", "unknown_error")
        msg = getattr(e, "message", str(e)) or str(e)
        return {"success": False, "error": code, "message": msg}


@app.get("/api/admin/ibkr/raw-nav-debug")
@limiter.limit("6/minute")
def ibkr_raw_nav_debug(request: Request):
    """ADMIN diagnostic — return the raw IBKR NAV report structure.

    Use this when /api/ibkr/nav-for-date returns no_data_for_date for what
    looks like a valid date and we need to see what the report actually
    contains. The response surfaces every attribute on the EquitySummary
    rows so we can identify field-name drift (e.g. the parser keys on
    `reportDate` but the report uses `toDate`).

    Founder-gated: bearer auth alone isn't enough — request.state.user_id
    must equal FOUNDER_USER_ID. Returns 200 OK + {success: false, error}
    on every failure mode (config / auth / network / parse) so a single
    response shape covers all cases.

    Output redactions:
      - accountId attributes are masked in raw_xml_first_2000_chars
      - the inspector reports presence of attrs, not their values
      - the IBKR token is never echoed (it's a query param, not response)
    """
    user_id = getattr(request.state, "user_id", None)
    if user_id != FOUNDER_USER_ID:
        # Distinct error code so a logged-in non-owner sees a clear reason
        # rather than a generic 401 (the bearer auth already passed).
        return {
            "success": False,
            "error": "forbidden_not_admin",
            "message": "This endpoint is owner-only.",
        }

    try:
        import ibkr_flex
        import xml.etree.ElementTree as _ET
        token = os.environ.get("IBKR_FLEX_TOKEN", "").strip()
        query_id = os.environ.get("IBKR_NAV_FLEX_QUERY_ID", "").strip()
        if not token or not query_id:
            return {
                "success": False,
                "error": "ibkr_not_configured",
                "message": "Set IBKR_FLEX_TOKEN and IBKR_NAV_FLEX_QUERY_ID.",
            }

        xml_root = ibkr_flex._fetch_nav_xml(query_id, token)
        # Re-serialise the parsed tree (rather than echoing the raw HTTP body)
        # so we can guarantee the redaction pass runs on a stable canonical
        # form. ET.tostring keeps attribute order in CPython 3.8+.
        raw_xml = _ET.tostring(xml_root, encoding="unicode")
        redacted = ibkr_flex.redact_nav_xml(raw_xml)

        info = ibkr_flex.inspect_nav_xml(xml_root)
        return {
            "success": True,
            "raw_xml_first_2000_chars": redacted[:2000],
            "raw_xml_total_chars": len(redacted),
            **info,
        }
    except Exception as e:
        code = getattr(e, "code", "unknown_error")
        msg = getattr(e, "message", str(e)) or str(e)
        return {"success": False, "error": code, "message": msg}


@app.get("/api/admin/drift-scan")
@limiter.limit("12/minute")
def drift_scan_endpoint(
    request: Request,
    portfolio: str | None = Query(None, description="Optional portfolio name; default scans all"),
    check_id: str | None = Query(None, description="Optional single-check id; default runs all 12"),
    limit_samples: int = Query(10, ge=1, le=50, description="Sample rows per check (1-50)"),
):
    """ADMIN — run drift checks against the database.

    Catalog-driven consistency scan. Each entry in api/drift_checks.DRIFT_CHECKS
    is a SQL query for a known invariant (rule mismatch, orphan summaries,
    LIFO drift, tripwires for Migration 022 CHECK constraints, etc.). The
    runner returns per-check (violation_count, samples) plus a summary.

    Founder-gated. Same {error: forbidden_not_admin} 200-OK contract as the
    other /api/admin/* endpoints — bearer auth alone isn't enough.

    Read-only: every check is a SELECT, the runner rolls back its implicit
    transaction at the end. No writes ever land.
    """
    user_id = getattr(request.state, "user_id", None)
    if user_id != FOUNDER_USER_ID:
        return {"error": "forbidden_not_admin"}

    # Validate check_id up-front so a typo returns 400 instead of an
    # empty-checks response that looks like nothing ran.
    from api.drift_checks import DRIFT_CHECKS_BY_ID, run_drift_scan
    if check_id is not None and check_id not in DRIFT_CHECKS_BY_ID:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown check_id: {check_id}",
        )

    portfolio_id: int | None = None
    if portfolio:
        # Resolve the name → int FK once. Unknown name = 400 (vs scanning
        # everything), since silently treating "typo'd portfolio" as
        # "scan all" would surprise the caller.
        try:
            with db.get_db_connection() as _conn:
                with _conn.cursor() as _cur:
                    _cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio,))
                    row = _cur.fetchone()
                    if row is None:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Unknown portfolio: {portfolio}",
                        )
                    portfolio_id = int(row[0])
        except HTTPException:
            raise
        except Exception as e:
            return {"error": f"db_unavailable: {e}"}

    try:
        with db.get_db_connection() as conn:
            return run_drift_scan(
                conn,
                portfolio_id=portfolio_id,
                portfolio_name=portfolio,
                check_id=check_id,
                sample_limit=limit_samples,
            )
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/trades/import")
@limiter.limit("3/minute")
def import_ibkr_trades(request: Request):
    """Pull today's trade confirmations from IBKR Flex Query."""
    try:
        import ibkr_flex
        df, debug, err = ibkr_flex.pull_ibkr_trades(consolidate=True)
        if err:
            return {"error": err, "debug": debug or {}}
        if df.empty:
            return {"status": "ok", "trades": [], "count": 0,
                    "message": "No trades found in IBKR report", "debug": debug or {}}
        # Convert to list of dicts for JSON response
        records = df.to_dict(orient="records")
        # Clean numpy types
        for r in records:
            for k, v in r.items():
                if hasattr(v, 'item'):
                    r[k] = v.item()
                elif pd.isna(v) if not isinstance(v, str) else False:
                    r[k] = None
        return {"status": "ok", "trades": records, "count": len(records), "debug": debug or {}}
    except Exception as e:
        return {"error": str(e)}


TRX_ID_MAX_RETRIES = 5


def _save_detail_with_unique_trx_id(
    portfolio: str,
    trade_id: str,
    prefix: str,
    detail_row: dict,
    *,
    given_trx_id: str = "",
) -> tuple[int, str]:
    """Save a detail row, regenerating its trx_id on UniqueViolation.

    First attempt uses given_trx_id when non-empty (preserves a client-
    supplied value if one came in); otherwise asks db.generate_unique_trx_id.
    Every retry regenerates via the helper. Bounded at TRX_ID_MAX_RETRIES so
    a runaway never spins forever.

    Race-safety leans on the partial unique index `unique_trx_id_per_trade`
    added in migration 018 — UNIQUE (portfolio_id, trade_id, trx_id) WHERE
    deleted_at IS NULL. Active rows only; soft-deleted rows are outside the
    index's scope so a deleted trx_id is reusable for a new active row.
    The index raises UniqueViolation on a concurrent duplicate active
    insert, which this loop catches and retries. Without the index (pre-
    migration-018), this function CAN still produce duplicates under
    concurrency; that's the deploy-window risk we knowingly accept,
    mitigated by the advisory lock in db.generate_unique_trx_id and by
    re-running scripts/dedupe_trx_ids.py post-migration.

    Returns (detail_id, final_trx_id).
    """
    last_err: Exception | None = None
    for attempt in range(TRX_ID_MAX_RETRIES):
        if attempt == 0 and given_trx_id:
            detail_row["Trx_ID"] = given_trx_id
        else:
            detail_row["Trx_ID"] = db.generate_unique_trx_id(portfolio, trade_id, prefix)
        try:
            detail_id = db.save_detail_row(portfolio, detail_row)
            return detail_id, detail_row["Trx_ID"]
        except psycopg2.errors.UniqueViolation as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Failed to generate unique trx_id for trade {trade_id} after "
        f"{TRX_ID_MAX_RETRIES} attempts. Last error: {last_err}"
    )


@app.post("/api/trades/buy")
@limiter.limit("10/minute")
def log_buy(request: Request, body: dict):
    """Log a buy transaction. Creates/updates summary + inserts detail row."""
    try:
        portfolio = body.get("portfolio", "CanSlim")
        action_type = body.get("action_type", "new")  # "new" or "scalein"
        ticker = body.get("ticker", "").upper()
        trade_id = body.get("trade_id", "")
        shares = float(body.get("shares") or 0)
        price = float(body.get("price") or 0)
        stop_loss = float(body.get("stop_loss") or 0)
        rule = body.get("rule", "")
        notes = body.get("notes", "")
        date_str = body.get("date", datetime.now().strftime("%Y-%m-%d"))
        time_str = body.get("time", datetime.now().strftime("%H:%M"))
        client_trx_id = body.get("trx_id", "")
        # Strategy (Migration 019). Defaults to CanSlim — matches the DB
        # column DEFAULT and the user's primary strategy. For new buys we
        # validate the value against the active strategies; for scale-ins
        # we ignore the body entirely and inherit from the parent campaign
        # (same defense-in-depth pattern as instrument_type below).
        strategy = (body.get("strategy") or "CanSlim").strip() or "CanSlim"

        if not ticker or not trade_id or shares <= 0 or price <= 0:
            return {"error": "Missing required fields: ticker, trade_id, shares, price"}

        # Validate strategy on new buys only. Scale-in trusts the parent.
        if action_type == "new":
            valid_strategy_names = {s["name"] for s in db.load_strategies(active_only=True)}
            if strategy not in valid_strategy_names:
                return {"error": f"Unknown strategy: {strategy}"}

        # Detect equity options from ticker shape (`SYMBOL YYMMDD $STRIKE C|P`)
        # and apply the standard 100× contract multiplier so cost basis and
        # downstream P&L reflect notional dollars, not premium-per-contract.
        instrument_type = 'OPTION' if is_option_ticker(ticker) else 'STOCK'
        multiplier = 100.0 if instrument_type == 'OPTION' else 1.0
        value = shares * price * multiplier
        date_time = f"{date_str} {time_str}:00"

        # Determine trx_id prefix: first BUY on a trade uses 'B'; any
        # subsequent BUY (add-on) uses 'A'. The numeric suffix is assigned
        # by _save_detail_with_unique_trx_id at insert time (collision-safe
        # via db.generate_unique_trx_id + retry-on-conflict).
        trx_prefix = "B"
        if not client_trx_id:
            df_d = db.load_details(portfolio)
            if not df_d.empty:
                df_d = _normalize_trades(df_d)
                existing_txns = df_d[df_d["trade_id"] == trade_id]
                if not existing_txns.empty and (existing_txns["action"].str.upper() == "BUY").any():
                    trx_prefix = "A"

        # Build summary row first (FK requires summary before detail).
        # risk_budget = initial $ at risk at entry (shares * (entry - stop)).
        # Populated here because direct-log paths (e.g. IBKR Quick Log) skip
        # the sizing calculator that historically set this column.
        if action_type == "new":
            # Group 7-1: route risk_budget through compute_trade_risk even
            # for a single-BUY new campaign so log_buy has one canonical
            # source of truth for the formula. Mathematically equivalent to
            # calc_risk_budget(shares, price, stop_loss, multiplier) here.
            new_buy_df = pd.DataFrame([{
                "date": pd.to_datetime(date_time), "action": "BUY",
                "shares": shares, "amount": price, "stop_loss": stop_loss,
            }])
            summary_row = {
                "Trade_ID": trade_id, "Ticker": ticker, "Status": "OPEN",
                "Open_Date": date_str, "Shares": shares,
                "Avg_Entry": price, "Total_Cost": value,
                "Stop_Loss": stop_loss, "Rule": rule, "Buy_Notes": notes,
                "Risk_Budget": compute_trade_risk(new_buy_df, multiplier),
                "Instrument_Type": instrument_type, "Multiplier": multiplier,
                "Strategy": strategy,
            }
        else:
            # Scale-in: load existing summary and update
            df_s = db.load_summary(portfolio)
            df_s = _normalize_trades(df_s)
            existing = df_s[df_s["trade_id"] == trade_id]
            if not existing.empty:
                row = existing.iloc[0]
                # Inherit instrument_type from the existing campaign so a
                # scale-in can never flip a stock trade into an option (or
                # vice-versa). Falls back to the autodetected value if the row
                # pre-dates Migration 016.
                existing_instr = str(row.get("instrument_type") or "").upper() or instrument_type
                existing_mult = float(row.get("multiplier") or 0) or multiplier
                instrument_type = existing_instr
                multiplier = existing_mult
                # Inherit strategy from the parent campaign (same reason as
                # instrument_type — a scale-in must never reclassify the
                # campaign). Falls back to 'CanSlim' for legacy rows that
                # pre-date Migration 019.
                strategy = str(row.get("strategy") or "").strip() or "CanSlim"
                value = shares * price * multiplier
                old_shares = float(row.get("shares", 0))
                old_entry = float(row.get("avg_entry", 0))
                old_cost = float(row.get("total_cost", 0))
                new_total_shares = old_shares + shares
                new_total_cost = old_cost + value
                new_avg_entry = (new_total_cost / new_total_shares / multiplier) if new_total_shares > 0 else price
                effective_stop = float(stop_loss if stop_loss > 0 else row.get("stop_loss", 0) or 0)
                # Group 7-1: risk_budget is the holistic Trade Risk $ over the
                # full post-insert BUY set (existing details + this new lot),
                # walked LIFO per-lot with per-lot stops, each contribution
                # floored at 0. Replaces the prior additive logic
                # (existing_rb + new_lot_rb) which inflated risk on every
                # scale-in and never reflected lots whose stops had moved up.
                df_d_existing = db.load_details(portfolio)
                if df_d_existing.empty:
                    existing_txns = pd.DataFrame()
                else:
                    df_d_existing = _normalize_trades(df_d_existing)
                    existing_txns = df_d_existing[df_d_existing["trade_id"] == trade_id]
                new_buy_df = pd.DataFrame([{
                    "date": pd.to_datetime(date_time),
                    "action": "BUY",
                    "shares": shares,
                    "amount": price,
                    "stop_loss": effective_stop,
                }])
                post_insert = pd.concat([existing_txns, new_buy_df], ignore_index=True) \
                    if not existing_txns.empty else new_buy_df
                holistic_risk = compute_trade_risk(post_insert, multiplier)
                summary_row = {
                    "Trade_ID": trade_id, "Ticker": ticker, "Status": "OPEN",
                    "Open_Date": str(row.get("open_date", date_str))[:10],
                    "Shares": float(new_total_shares),
                    "Avg_Entry": float(round(new_avg_entry, 4)),
                    "Total_Cost": float(round(new_total_cost, 2)),
                    "Stop_Loss": effective_stop,
                    "Rule": db.clean_text_value(row.get("rule")) or db.clean_text_value(rule),
                    "Buy_Notes": db.clean_text_value(notes) or db.clean_text_value(row.get("buy_notes")),
                    "Risk_Budget": holistic_risk,
                    "Instrument_Type": instrument_type, "Multiplier": multiplier,
                    "Strategy": strategy,
                }
            else:
                # Parent campaign not found — treat as a new campaign with the
                # body-supplied strategy. Validate same as the "new" branch.
                valid_strategy_names = {s["name"] for s in db.load_strategies(active_only=True)}
                if strategy not in valid_strategy_names:
                    return {"error": f"Unknown strategy: {strategy}"}
                # Same canonical compute_trade_risk path as the "new" branch.
                orphan_new_buy_df = pd.DataFrame([{
                    "date": pd.to_datetime(date_time), "action": "BUY",
                    "shares": shares, "amount": price, "stop_loss": stop_loss,
                }])
                summary_row = {
                    "Trade_ID": trade_id, "Ticker": ticker, "Status": "OPEN",
                    "Open_Date": date_str, "Shares": shares,
                    "Avg_Entry": price, "Total_Cost": value,
                    "Stop_Loss": stop_loss, "Rule": rule, "Buy_Notes": notes,
                    "Risk_Budget": compute_trade_risk(orphan_new_buy_df, multiplier),
                    "Instrument_Type": instrument_type, "Multiplier": multiplier,
                    "Strategy": strategy,
                }

        summary_id = db.save_summary_row(portfolio, summary_row)

        # Save detail row (after summary so FK constraint is satisfied).
        # Trx_ID is assigned by the helper — the placeholder here is overwritten.
        detail_row = {
            "Trade_ID": trade_id, "Ticker": ticker, "Action": "BUY",
            "Date": date_time, "Shares": shares, "Amount": price,
            "Value": value, "Rule": rule, "Notes": notes,
            "Stop_Loss": stop_loss, "Trx_ID": "",
            "Instrument_Type": instrument_type, "Multiplier": multiplier,
        }
        detail_id, trx_id = _save_detail_with_unique_trx_id(
            portfolio, trade_id, trx_prefix, detail_row, given_trx_id=client_trx_id,
        )

        # Audit trail
        try:
            db.log_audit(portfolio, "BUY", trade_id, ticker,
                         f"{trx_id}: {shares} shs @ ${price:.2f}", username="web")
        except Exception:
            pass

        return {"status": "ok", "detail_id": detail_id, "summary_id": summary_id, "trx_id": trx_id}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/trades/grade")
def set_trade_grade(body: dict = Body(...)):
    """Set or clear the 1-5 star grade on a trade campaign (by trade_id).
    Pass grade=null to clear."""
    try:
        portfolio = body.get("portfolio", "CanSlim")
        trade_id = str(body.get("trade_id", "")).strip()
        if not trade_id:
            return {"error": "Missing trade_id"}
        raw = body.get("grade", None)
        grade_val = None
        if raw is not None and str(raw).strip() != "":
            try:
                g = int(raw)
                if not (1 <= g <= 5):
                    return {"error": "grade must be 1-5 or null"}
                grade_val = g
            except (ValueError, TypeError):
                return {"error": "grade must be integer 1-5"}

        # Load existing summary row so we don't blow away other fields
        df_s = db.load_summary(portfolio)
        df_s = _normalize_trades(df_s)
        existing = df_s[df_s["trade_id"] == trade_id]
        if existing.empty:
            return {"error": f"Trade {trade_id} not found"}
        row = existing.iloc[0]

        summary_row = {
            "Trade_ID": trade_id,
            "Ticker": str(row.get("ticker", "")),
            "Status": str(row.get("status", "OPEN")),
            "Open_Date": str(row.get("open_date", ""))[:10] if row.get("open_date") else None,
            "Closed_Date": str(row.get("closed_date", ""))[:10] if row.get("closed_date") else None,
            "Shares": float(row.get("shares", 0) or 0),
            "Avg_Entry": float(row.get("avg_entry", 0) or 0),
            "Avg_Exit": float(row.get("avg_exit", 0) or 0),
            "Total_Cost": float(row.get("total_cost", 0) or 0),
            "Realized_PL": float(row.get("realized_pl", 0) or 0),
            "Unrealized_PL": float(row.get("unrealized_pl", 0) or 0),
            "Return_Pct": float(row.get("return_pct", 0) or 0),
            "Sell_Rule": db.clean_text_value(row.get("sell_rule")),
            "Notes": db.clean_text_value(row.get("notes")),
            "Stop_Loss": row.get("stop_loss") or None,
            "Rule": db.clean_text_value(row.get("rule")),
            "Buy_Notes": db.clean_text_value(row.get("buy_notes")),
            "Sell_Notes": db.clean_text_value(row.get("sell_notes")),
            "Risk_Budget": float(row.get("risk_budget", 0) or 0),
            "Grade": grade_val,
        }
        summary_id = db.save_summary_row(portfolio, summary_row)
        return {"status": "ok", "trade_id": trade_id, "grade": grade_val, "summary_id": summary_id}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/trades/sell")
@limiter.limit("10/minute")
def log_sell(request: Request, body: dict):
    """Log a sell transaction. Updates summary via LIFO + inserts detail row."""
    try:
        portfolio = body.get("portfolio", "CanSlim")
        trade_id = body.get("trade_id", "")
        shares = float(body.get("shares") or 0)
        price = float(body.get("price") or 0)
        rule = body.get("rule", "")
        notes = body.get("notes", "")
        date_str = body.get("date") or datetime.now().strftime("%Y-%m-%d")
        time_str = body.get("time") or datetime.now().strftime("%H:%M")
        client_trx_id = body.get("trx_id", "")
        grade_raw = body.get("grade", None)

        if not trade_id or shares <= 0 or price <= 0:
            return {"error": "Missing required fields: trade_id, shares, price"}

        # Load existing summary to get ticker and validate shares
        df_s = db.load_summary(portfolio)
        df_s = _normalize_trades(df_s)
        existing = df_s[df_s["trade_id"] == trade_id]
        if existing.empty:
            return {"error": f"Trade {trade_id} not found"}

        row = existing.iloc[0]
        ticker = row.get("ticker", "")
        current_shares = float(row.get("shares", 0))

        if shares > current_shares:
            return {"error": f"Cannot sell {shares} shares — only {current_shares} held"}

        # Inherit instrument_type/multiplier from the existing campaign so the
        # LIFO realized_pl + total_cost recompute use the right unit. Falls
        # back to autodetect for legacy rows that pre-date Migration 016.
        instrument_type = str(row.get("instrument_type") or "").upper() \
            or ('OPTION' if is_option_ticker(ticker) else 'STOCK')
        multiplier = float(row.get("multiplier") or 0) \
            or (100.0 if instrument_type == 'OPTION' else 1.0)

        value = shares * price * multiplier
        date_time = f"{date_str} {time_str}:00"

        # Save detail row. Trx_ID is assigned by the helper (always 'S{n}'
        # for SELLs — no SA/SB branching in live code; legacy SA/SB rows
        # are preserved by the regex-based suffix scan in the helper).
        # The placeholder Trx_ID below is overwritten before the INSERT.
        detail_row = {
            "Trade_ID": trade_id, "Ticker": ticker, "Action": "SELL",
            "Date": date_time, "Shares": shares, "Amount": price,
            "Value": value, "Rule": rule, "Notes": notes,
            "Realized_PL": 0, "Trx_ID": "",
            "Instrument_Type": instrument_type, "Multiplier": multiplier,
        }
        detail_id, trx_id = _save_detail_with_unique_trx_id(
            portfolio, trade_id, "S", detail_row, given_trx_id=client_trx_id,
        )

        # LIFO recalculation: reload all details and recompute summary
        df_d = db.load_details(portfolio)
        df_d = _normalize_trades(df_d)
        txns = df_d[df_d["trade_id"] == trade_id].copy()
        txns["date"] = pd.to_datetime(txns["date"], errors="coerce")
        txns = txns.dropna(subset=["date"]).sort_values("date")

        inventory = []
        total_realized = 0.0
        for _, tx in txns.iterrows():
            action = str(tx.get("action", "")).upper()
            tx_shares = float(tx.get("shares", 0))
            tx_price = float(tx.get("amount", 0))
            if action == "BUY":
                inventory.append({"price": tx_price, "shares": tx_shares})
            elif action == "SELL":
                to_sell = tx_shares
                while to_sell > 0 and inventory:
                    last = inventory[-1]
                    take = min(to_sell, last["shares"])
                    total_realized += (tx_price - last["price"]) * take
                    last["shares"] -= take
                    to_sell -= take
                    if last["shares"] < 0.0001:
                        inventory.pop()

        remaining_shares = sum(lot["shares"] for lot in inventory)
        remaining_cost = sum(lot["shares"] * lot["price"] for lot in inventory)
        avg_entry = remaining_cost / remaining_shares if remaining_shares > 0 else float(row.get("avg_entry", 0))

        # Compute avg_exit from all sells
        sells = txns[txns["action"].str.upper() == "SELL"]
        total_sell_val = float((sells["shares"].astype(float) * sells["amount"].astype(float)).sum())
        total_sell_shs = float(sells["shares"].astype(float).sum())
        avg_exit = total_sell_val / total_sell_shs if total_sell_shs > 0 else 0.0

        is_closed = remaining_shares < 0.01
        buys = txns[txns["action"].str.upper() == "BUY"]
        total_cost = float((buys["shares"].astype(float) * buys["amount"].astype(float)).sum())
        total_buy_shs = float(buys["shares"].astype(float).sum())
        # Return % is multiplier-invariant (ratio cancels). Apply multiplier
        # only to absolute dollars (Total_Cost, Realized_PL).
        return_pct = (total_realized / total_cost * 100) if is_closed and total_cost > 0 else 0.0
        cost_to_report = remaining_cost if not is_closed else total_cost

        summary_row = {
            "Trade_ID": trade_id, "Ticker": str(ticker),
            "Status": "CLOSED" if is_closed else "OPEN",
            "Open_Date": str(row.get("open_date", ""))[:10],
            "Closed_Date": date_str if is_closed else None,
            "Shares": float(remaining_shares if not is_closed else total_buy_shs),
            "Avg_Entry": float(round(avg_entry, 4)),
            "Avg_Exit": float(round(avg_exit, 4)) if avg_exit > 0 else 0.0,
            "Total_Cost": float(round(cost_to_report * multiplier, 2)),
            "Realized_PL": float(round(total_realized * multiplier, 2)),
            "Return_Pct": float(round(return_pct, 4)),
            "Sell_Rule": rule,
            "Sell_Notes": notes,
            "Rule": db.clean_text_value(row.get("rule")),
            "Buy_Notes": db.clean_text_value(row.get("buy_notes")),
            "Stop_Loss": row.get("stop_loss"),
            "Risk_Budget": row.get("risk_budget"),
            "Instrument_Type": instrument_type, "Multiplier": multiplier,
        }
        if grade_raw is not None and str(grade_raw).strip() != "":
            try:
                g = int(grade_raw)
                if 1 <= g <= 5:
                    summary_row["Grade"] = g
            except (ValueError, TypeError):
                pass
        summary_id = db.save_summary_row(portfolio, summary_row)

        # Audit trail
        try:
            db.log_audit(portfolio, "SELL", trade_id, ticker,
                         f"{trx_id}: {shares} shs @ ${price:.2f} | P&L: ${total_realized:.2f}", username="web")
        except Exception:
            pass

        # Re-run LIFO via the shared recompute path so lot_closures gets
        # populated for fresh SELLs too. The inline LIFO above already wrote
        # the correct summary; this second pass produces the same summary
        # numbers (idempotent) and adds the per-pair closure rows the inline
        # path doesn't touch. TODO step 5: kill the inline LIFO duplication
        # and have this endpoint use _recompute_summary_lifo exclusively.
        try:
            _recompute_summary_lifo(portfolio, trade_id, ticker)
        except Exception as e:
            print(f"[lot_closures] post-SELL recompute failed for {trade_id}: {e}")

        return {
            "status": "ok", "detail_id": detail_id, "summary_id": summary_id,
            "trx_id": trx_id, "realized_pl": round(total_realized * multiplier, 2),
            "remaining_shares": round(remaining_shares, 4),
            "is_closed": is_closed,
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/trades/exercise-option")
@limiter.limit("10/minute")
def exercise_option(request: Request, body: dict = Body(...)):
    """Exercise an OPEN option position into the underlying stock.

    Closes ALL currently-held contracts on the option trade (LIFO walk over
    open BUYs) and either scales into an existing OPEN stock trade for the
    underlying, or opens a new stock trade if none exists. The whole
    sequence — option SELL detail, option summary recompute, stock detail
    insert, stock summary upsert, audit log — runs inside one
    atomic_transaction. A failure at any step rolls back every write.

    Body: { portfolio: str, trade_id: str, date: 'YYYY-MM-DD', notes?: str }

    Math (per-contract premium → per-share stock cost basis):
        contracts_held         = LIFO remaining shares on the option trade
        weighted_avg_premium   = LIFO remaining cost / contracts_held  ($/share)
        shares_acquired        = contracts_held × multiplier            (typ. 100×)
        stock_entry_price      = strike + weighted_avg_premium
        stock_total_cost_basis = shares_acquired × stock_entry_price
                                = (premium × contracts × multiplier) + (strike × shares)

    The option SELL is priced at weighted_avg_premium so the option-side
    realized P&L is exactly $0 by construction — the cost basis migrates
    to the stock position rather than being realized. Cash-ledger impact
    nets to −strike × shares (the strike payment), correct under
    physical-settlement assumptions.
    """
    try:
        portfolio = body.get("portfolio", "CanSlim")
        option_trade_id = str(body.get("trade_id", "") or "").strip()
        date_str = str(body.get("date", "") or "").strip() \
                   or datetime.now().strftime("%Y-%m-%d")
        user_notes = str(body.get("notes", "") or "").strip()

        if not option_trade_id:
            return {"error": "Missing trade_id"}

        # Pre-flight reads of the SUMMARY only — used for early validation
        # (trade exists, is OPEN, is OPTION) before opening a transaction
        # and for the existing-stock-trade lookup. Details are re-read
        # INSIDE the txn block below so the LIFO walk that produces
        # contracts_held / weighted_avg_premium sees the latest committed
        # state (avoids a stale-snapshot race against a parallel writer
        # that committed during the pre-flight phase).
        df_s = db.load_summary(portfolio)
        df_s = _normalize_trades(df_s)
        opt_summary = df_s[df_s["trade_id"] == option_trade_id]
        if opt_summary.empty:
            return {"error": f"Trade {option_trade_id} not found"}

        opt_row = opt_summary.iloc[0]
        opt_status = str(opt_row.get("status", "") or "").upper()
        opt_instrument = str(opt_row.get("instrument_type") or "").upper()
        opt_ticker = str(opt_row.get("ticker", "") or "")

        if opt_status != "OPEN":
            return {"error": "Trade is not open"}
        # Treat legacy rows that pre-date Migration 016 as OPTION when their
        # ticker matches the readable option format. Otherwise reject — the
        # exercise flow only makes sense for an option position.
        if opt_instrument != "OPTION" and not is_option_ticker(opt_ticker):
            return {"error": "Only options can be exercised"}

        parsed = parse_option_ticker(opt_ticker)
        if parsed is None:
            return {"error": f"Cannot parse option ticker '{opt_ticker}' "
                             f"(expected: SYMBOL YYMMDD $STRIKEC|P)"}
        underlying = parsed["underlying"]
        strike = float(parsed["strike"])

        opt_multiplier = float(opt_row.get("multiplier") or 0) or 100.0

        # Look up an existing OPEN stock trade for the underlying so we know
        # whether to scale-in or open a new campaign. Match on ticker AND
        # instrument_type=STOCK so we never collide with a same-ticker option
        # row (e.g. AAPL stock + AAPL option both open).
        df_s_norm = df_s
        existing_stock = df_s_norm[
            (df_s_norm["ticker"] == underlying)
            & (df_s_norm["status"].astype(str).str.upper() == "OPEN")
        ]
        # instrument_type may be missing/NULL on legacy rows — treat anything
        # that isn't explicitly OPTION as a stock candidate.
        if "instrument_type" in existing_stock.columns:
            existing_stock = existing_stock[
                existing_stock["instrument_type"].astype(str).str.upper() != "OPTION"
            ]
        scale_into_existing = not existing_stock.empty
        if scale_into_existing:
            stock_trade_id = str(existing_stock.iloc[0].get("trade_id", ""))
            stock_existing_row = existing_stock.iloc[0]
        else:
            # Generate a fresh trade_id using the same soft-delete-safe path
            # next_trade_id uses (load_all_trade_ids_for_month sees deleted
            # rows so we never recycle).
            ym = pd.Timestamp(date_str).strftime("%Y%m")
            existing_ids = db.load_all_trade_ids_for_month(portfolio, ym)
            seqs = []
            for x in existing_ids:
                try:
                    if "-" in x:
                        seqs.append(int(x.split("-")[-1]))
                except Exception:
                    pass
            next_seq = (max(seqs) + 1) if seqs else 1
            stock_trade_id = f"{ym}-{next_seq:03d}"
            stock_existing_row = None

        date_time = f"{date_str} {datetime.now().strftime('%H:%M')}:00"

        # === ATOMIC SECTION ===
        # All mutating writes live inside one atomic_transaction. Any exception
        # below rolls back every write. The details read + LIFO walk also
        # happen inside so contracts_held / weighted_avg_premium reflect the
        # latest committed state — not a stale snapshot from a parallel writer
        # that committed during the pre-flight phase. The advisory locks taken
        # inside _generate_unique_trx_id_in_txn additionally serialize
        # concurrent writers to the same trade_id.
        with db.atomic_transaction() as (_conn, cur):
            cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio,))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Portfolio '{portfolio}' not found")
            portfolio_id = row[0]

            # LIFO walk on a fresh details read — load_details is ttl_cached,
            # so .clear() forces a real round-trip and we don't accidentally
            # recompute against the pre-flight cached snapshot. The early
            # return below is safe: no writes have happened yet, so the
            # context-manager exit is an empty no-op commit.
            db.load_details.clear()
            df_d = db.load_details(portfolio)
            df_d = _normalize_trades(df_d)
            opt_txns = df_d[df_d["trade_id"] == option_trade_id].copy()
            opt_txns["date"] = pd.to_datetime(opt_txns["date"], errors="coerce")
            opt_txns = opt_txns.dropna(subset=["date"]).sort_values("date")

            inventory = []
            for _, tx in opt_txns.iterrows():
                action = str(tx.get("action", "")).upper()
                tx_shares = float(tx.get("shares", 0) or 0)
                tx_price = float(tx.get("amount", 0) or 0)
                if action == "BUY":
                    inventory.append({"price": tx_price, "shares": tx_shares})
                elif action == "SELL":
                    to_sell = tx_shares
                    while to_sell > 0 and inventory:
                        last = inventory[-1]
                        take = min(to_sell, last["shares"])
                        last["shares"] -= take
                        to_sell -= take
                        if last["shares"] < 0.0001:
                            inventory.pop()
            contracts_held = sum(lot["shares"] for lot in inventory)
            held_cost = sum(lot["shares"] * lot["price"] for lot in inventory)
            if contracts_held <= 0:
                return {"error": "No contracts currently held"}
            weighted_avg_premium = held_cost / contracts_held

            # Derived stock-side numbers (depend on the LIFO walk output above).
            shares_acquired = contracts_held * opt_multiplier
            stock_entry_price = strike + weighted_avg_premium
            stock_total_cost = shares_acquired * stock_entry_price
            opt_total_premium_dollars = held_cost * opt_multiplier  # for audit detail

            # --- 1. Option SELL detail ---
            opt_sell_value = contracts_held * weighted_avg_premium * opt_multiplier
            opt_sell_trx_id = db._generate_unique_trx_id_in_txn(
                cur, portfolio_id, option_trade_id, "S",
            )
            opt_detail_row = {
                "Trade_ID": option_trade_id, "Ticker": opt_ticker, "Action": "SELL",
                "Date": date_time,
                "Shares": float(contracts_held),
                "Amount": float(round(weighted_avg_premium, 4)),
                "Value": float(round(opt_sell_value, 2)),
                "Rule": "",  # locked decision: empty (no SELL_RULES expansion)
                "Notes": user_notes,
                "Realized_PL": 0,
                "Trx_ID": opt_sell_trx_id,
                "Instrument_Type": "OPTION",
                "Multiplier": opt_multiplier,
            }
            opt_detail_id = db._save_detail_row_in_txn(cur, portfolio_id, opt_detail_row)

            # --- 2. Recompute option summary + closures ---
            # Re-walk all option details (including the SELL we just inserted)
            # via compute_lifo_summary. The pure function returns the LIFO
            # fields; we layer on the preserved/derived columns.
            opt_txns_after = pd.concat([
                opt_txns,
                pd.DataFrame([{
                    "trade_id": option_trade_id,
                    "ticker": opt_ticker,
                    "action": "SELL",
                    "date": pd.Timestamp(date_time),
                    "shares": float(contracts_held),
                    "amount": float(weighted_avg_premium),
                    "trx_id": opt_sell_trx_id,
                }]),
            ], ignore_index=True)
            opt_lifo_result = compute_lifo_summary(
                opt_txns_after, option_trade_id, opt_ticker,
                multiplier=opt_multiplier, with_closures=True,
            )
            opt_lifo_summary, opt_closures = opt_lifo_result
            if opt_lifo_summary is None:
                # Defensive — would only happen with corrupt detail rows.
                raise RuntimeError("LIFO recompute returned None for option side")

            # Preserve user-set fields from the existing summary; append the
            # exercise breadcrumb to .notes per locked decision #2.
            preserved_notes = db.clean_text_value(opt_row.get("notes")) or ""
            auto_note = (f"Exercised on {date_str} — converted to {underlying} "
                         f"stock position {stock_trade_id}")
            new_opt_notes = f"{preserved_notes}\n{auto_note}".strip() if preserved_notes else auto_note

            opt_summary_row = {
                **opt_lifo_summary,
                "Notes": new_opt_notes,
                "Stop_Loss": opt_row.get("stop_loss"),
                "Rule": opt_row.get("rule"),
                "Buy_Notes": opt_row.get("buy_notes"),
                "Sell_Rule": opt_row.get("sell_rule"),
                "Sell_Notes": opt_row.get("sell_notes"),
                "Risk_Budget": float(opt_row.get("risk_budget") or 0),
                "Instrument_Type": "OPTION",
                "Multiplier": opt_multiplier,
            }
            db._save_summary_with_closures_in_txn(
                cur, portfolio_id, option_trade_id, opt_summary_row, opt_closures,
            )

            # --- 3. Stock summary placeholder (FK precondition for detail) ---
            # When opening a new stock trade, the trades_summary row must
            # exist before we INSERT the BUY detail (FK constraint). Insert a
            # minimal placeholder now; the LIFO recompute below UPDATEs it
            # with the real numbers.
            if not scale_into_existing:
                cross_link = (f"Created via exercise of option trade "
                              f"{option_trade_id} ({opt_ticker})")
                placeholder = {
                    "Trade_ID": stock_trade_id, "Ticker": underlying,
                    "Status": "OPEN", "Open_Date": date_str,
                    "Shares": 0, "Avg_Entry": 0, "Total_Cost": 0,
                    "Stop_Loss": None, "Rule": "",
                    "Buy_Notes": "", "Notes": cross_link,
                    "Risk_Budget": 0,
                    "Instrument_Type": "STOCK", "Multiplier": 1,
                }
                db._save_summary_with_closures_in_txn(
                    cur, portfolio_id, stock_trade_id, placeholder, [],
                )

            # --- 4. Stock BUY detail ---
            # Prefix B for the first BUY on a new trade; A for an add-on
            # (scale-in) on an existing trade. Mirrors log_buy's convention.
            stock_trx_prefix = "A" if scale_into_existing else "B"
            stock_buy_trx_id = db._generate_unique_trx_id_in_txn(
                cur, portfolio_id, stock_trade_id, stock_trx_prefix,
            )
            stock_detail_row = {
                "Trade_ID": stock_trade_id, "Ticker": underlying, "Action": "BUY",
                "Date": date_time,
                "Shares": float(shares_acquired),
                "Amount": float(round(stock_entry_price, 4)),
                "Value": float(round(stock_total_cost, 2)),
                "Rule": "",  # locked decision: blank, user fills via Edit
                "Notes": "",
                "Stop_Loss": None,  # locked decision: NULL, user fills via Edit
                "Trx_ID": stock_buy_trx_id,
                "Instrument_Type": "STOCK",
                "Multiplier": 1.0,
            }
            stock_detail_id = db._save_detail_row_in_txn(
                cur, portfolio_id, stock_detail_row,
            )

            # --- 5. Recompute stock summary + closures ---
            # Reload all stock details for this trade and run LIFO on the full
            # set (existing + newly inserted BUY). For new trades the inventory
            # is just our single BUY, so LIFO is trivially the BUY's cost.
            if scale_into_existing:
                stock_existing_txns = df_d[df_d["trade_id"] == stock_trade_id].copy()
                stock_existing_txns["date"] = pd.to_datetime(
                    stock_existing_txns["date"], errors="coerce",
                )
                stock_existing_txns = stock_existing_txns.dropna(subset=["date"])
            else:
                stock_existing_txns = pd.DataFrame()
            stock_txns_after = pd.concat([
                stock_existing_txns,
                pd.DataFrame([{
                    "trade_id": stock_trade_id,
                    "ticker": underlying,
                    "action": "BUY",
                    "date": pd.Timestamp(date_time),
                    "shares": float(shares_acquired),
                    "amount": float(stock_entry_price),
                    "trx_id": stock_buy_trx_id,
                }]),
            ], ignore_index=True)
            stock_lifo_result = compute_lifo_summary(
                stock_txns_after, stock_trade_id, underlying,
                multiplier=1.0, with_closures=True,
            )
            stock_lifo_summary, stock_closures = stock_lifo_result
            if stock_lifo_summary is None:
                raise RuntimeError("LIFO recompute returned None for stock side")

            # Preserve scale-in row's user fields when present; new-trade row
            # uses the placeholder defaults + cross-link note.
            if scale_into_existing:
                existing_notes = db.clean_text_value(stock_existing_row.get("notes")) or ""
                scale_link = (f"Scaled in via exercise of option trade "
                              f"{option_trade_id} on {date_str}")
                merged_notes = f"{existing_notes}\n{scale_link}".strip() if existing_notes else scale_link
                stock_summary_row = {
                    **stock_lifo_summary,
                    "Notes": merged_notes,
                    "Stop_Loss": stock_existing_row.get("stop_loss"),
                    "Rule": db.clean_text_value(stock_existing_row.get("rule")),
                    "Buy_Notes": db.clean_text_value(stock_existing_row.get("buy_notes")),
                    "Sell_Rule": stock_existing_row.get("sell_rule"),
                    "Sell_Notes": stock_existing_row.get("sell_notes"),
                    "Risk_Budget": float(stock_existing_row.get("risk_budget") or 0),
                    "Instrument_Type": "STOCK",
                    "Multiplier": 1.0,
                }
            else:
                cross_link = (f"Created via exercise of option trade "
                              f"{option_trade_id} ({opt_ticker})")
                stock_summary_row = {
                    **stock_lifo_summary,
                    "Notes": cross_link,
                    "Stop_Loss": None,
                    "Rule": "",
                    "Buy_Notes": "",
                    "Sell_Rule": None,
                    "Sell_Notes": None,
                    "Risk_Budget": 0,
                    "Instrument_Type": "STOCK",
                    "Multiplier": 1.0,
                }
            db._save_summary_with_closures_in_txn(
                cur, portfolio_id, stock_trade_id, stock_summary_row, stock_closures,
            )

            # --- 6. Audit log entry ---
            audit_detail = (
                f"{opt_sell_trx_id}: exercised {contracts_held:g} contract(s) "
                f"of {opt_ticker} (premium ${opt_total_premium_dollars:.2f}) → "
                f"{stock_buy_trx_id}: {shares_acquired:g} {underlying} shares "
                f"@ ${stock_entry_price:.2f} on trade {stock_trade_id}"
            )
            db._log_audit_in_txn(
                cur, portfolio_id, "EXERCISE", option_trade_id, opt_ticker,
                audit_detail, "web",
            )

        # Cache invalidation only after successful commit.
        db.load_details.clear()
        db.load_summary.clear()

        return {
            "status": "ok",
            "option_trade_id": option_trade_id,
            "stock_trade_id": stock_trade_id,
            "stock_was_new": not scale_into_existing,
            "contracts_exercised": float(contracts_held),
            "shares_acquired": float(shares_acquired),
            "stock_entry_price": float(round(stock_entry_price, 4)),
        }
    except Exception as e:
        return {"error": str(e)}


@app.put("/api/trades/edit-transaction")
@limiter.limit("15/minute")
def edit_transaction_endpoint(request: Request, body: dict = Body(...)):
    """Edit an existing transaction detail row."""
    try:
        detail_id = body.get("detail_id")
        portfolio = body.get("portfolio", "CanSlim")
        trade_id = body.get("trade_id", "")
        ticker = body.get("ticker", "")

        if not detail_id:
            return {"error": "detail_id is required"}

        # Resolve the multiplier from the existing detail row so an edit can't
        # collapse an option's notional back to per-contract premium just
        # because the form forgot to send it. Falls back to ticker-pattern
        # autodetect for legacy rows. We also capture the row's trade_id and
        # ticker as fallbacks so an omitted client param can't blank them out
        # on the row or cause the post-edit recompute to silently skip.
        df_d = db.load_details(portfolio)
        multiplier = 1.0
        existing_trade_id = ""
        existing_ticker = ""
        existing_trx_id = ""
        if not df_d.empty:
            df_d = _normalize_trades(df_d)
            existing = df_d[df_d.get("detail_id", df_d.index) == detail_id] if "detail_id" in df_d.columns else df_d.iloc[0:0]
            if not existing.empty:
                m = existing.iloc[0].get("multiplier")
                if m is not None and float(m) > 0:
                    multiplier = float(m)
                existing_trade_id = str(existing.iloc[0].get("trade_id", "") or "")
                existing_ticker = str(existing.iloc[0].get("ticker", "") or "")
                existing_trx_id = str(existing.iloc[0].get("trx_id", "") or "")
        # Client value wins; fall back to whatever was on the row.
        effective_trade_id = trade_id or existing_trade_id
        effective_ticker = ticker or existing_ticker
        if multiplier == 1.0 and is_option_ticker(effective_ticker):
            multiplier = 100.0

        # Validate trx_id (if changing): must not collide with another row in
        # the same trade. The frontend's edit form makes the field readOnly so
        # the client shouldn't be sending changed values, but treat it as a
        # belt-and-suspenders check — reject early with a friendly error
        # rather than relying on the migration-018 UNIQUE constraint to fail
        # the UPDATE midway.
        client_trx_id = str(body.get("trx_id", "") or "").strip()
        if client_trx_id and client_trx_id != existing_trx_id and not df_d.empty \
                and "trx_id" in df_d.columns and "detail_id" in df_d.columns:
            try:
                collision = df_d[
                    (df_d["trade_id"] == effective_trade_id)
                    & (df_d["trx_id"] == client_trx_id)
                    & (df_d["detail_id"] != int(detail_id))
                ]
                if not collision.empty:
                    return {"error": f"trx_id '{client_trx_id}' already used by another row in this trade"}
            except (TypeError, ValueError):
                pass  # malformed detail_id — fall through; later DB layer catches it

        shares = float(body.get("shares") or 0)
        amount = float(body.get("amount") or 0)
        proposed_date = str(body.get("date", "") or "")
        proposed_action = str(body.get("action", "") or "").upper()

        # LIFO-safety check: simulate the post-edit state and reject before
        # we commit if the edit would leave any SELL shares unmatched. The
        # recompute path that runs after a successful UPDATE silently drops
        # unmatched sells, producing wrong realized_pl with no error surface
        # — six production trades carried bad data because of this gap.
        if effective_trade_id and not df_d.empty:
            txns_for_trade = df_d[df_d["trade_id"] == effective_trade_id] \
                if "trade_id" in df_d.columns else df_d.iloc[0:0]
            err = validate_post_edit_lifo(
                txns_for_trade, int(detail_id),
                proposed_action, shares, amount, proposed_date,
            )
            if err:
                return {"error": err}

        # Recompute value server-side so detail.value stays consistent with
        # shares × amount × multiplier regardless of what the form posted.
        value = round(shares * amount * multiplier, 2)

        row_dict = {
            "Trade_ID": effective_trade_id,
            "Ticker": effective_ticker,
            "Action": body.get("action", ""),
            "Date": body.get("date", ""),
            "Shares": shares,
            "Amount": amount,
            "Value": value,
            "Rule": body.get("rule", ""),
            "Notes": body.get("notes", ""),
            "Stop_Loss": body.get("stop_loss", 0),
            "Trx_ID": body.get("trx_id", ""),
        }

        db.update_detail_row(portfolio, detail_id, row_dict)

        # Mirror canonical detail-row fields (earliest BUY's rule/notes/
        # stop_loss; latest SELL's rule/notes on CLOSED) to trades_summary
        # BEFORE the recompute. The recompute's preservation block then
        # reads the just-mirrored values instead of the stale pre-edit
        # ones — fixes the c0435ee interaction where edits to detail
        # rule/notes/stop_loss were locked out of summary.
        try:
            if effective_trade_id:
                db.mirror_detail_edit_to_summary(portfolio, effective_trade_id)
        except Exception as e:
            print(f"[edit_transaction] mirror to summary failed for {effective_trade_id}: {e}")
            try:
                db.log_audit(portfolio, "MIRROR_FAILED", effective_trade_id,
                             effective_ticker, f"detail {detail_id}: {e}",
                             username="web")
            except Exception:
                pass

        # Recompute the campaign summary so avg_entry / realized_pl /
        # return_pct reflect the edited detail. Without this the face card
        # keeps stale numbers (e.g. edit a buy price after the sell already
        # closed the trade — the card still shows the pre-edit P&L).
        try:
            if effective_trade_id:
                _recompute_summary_lifo(portfolio, effective_trade_id, effective_ticker)
        except Exception:
            pass

        try:
            db.log_audit(portfolio, "EDIT", effective_trade_id, row_dict.get("Trx_ID", ""),
                         f"Transaction {detail_id} edited", username="web")
        except Exception:
            pass

        return {"status": "ok", "detail_id": detail_id}
    except Exception as e:
        return {"error": str(e)}


@app.delete("/api/trades/transaction")
@limiter.limit("15/minute")
def delete_transaction_endpoint(request: Request,
                                detail_id: int = Query(...),
                                trade_id: str = Query(""),
                                ticker: str = Query(""),
                                portfolio: str = Query("CanSlim")):
    """Soft-delete a single transaction detail row, then recompute its
    campaign's LIFO summary so avg_entry / realized_pl / status reflect
    the removal. If the deletion empties the campaign, the summary is
    cleaned up by _recompute_summary_lifo."""
    try:
        if not detail_id:
            return {"error": "detail_id is required"}

        # Look up the row's trade_id/ticker before deleting so the recompute
        # can fire even when the client doesn't pass them in the query string.
        # Client values still win when supplied; row values are the fallback.
        effective_trade_id = trade_id
        effective_ticker = ticker
        # Pre-initialize so the LIFO validator below can rely on df_d being
        # defined even when the inner load fails (the bare except below would
        # otherwise leave it unbound).
        df_d = pd.DataFrame()
        try:
            df_d = db.load_details(portfolio)
            if not df_d.empty:
                df_d = _normalize_trades(df_d)
                if "detail_id" in df_d.columns:
                    row = df_d[df_d["detail_id"] == int(detail_id)]
                    if not row.empty:
                        if not effective_trade_id:
                            effective_trade_id = str(row.iloc[0].get("trade_id", "") or "")
                        if not effective_ticker:
                            effective_ticker = str(row.iloc[0].get("ticker", "") or "")
        except Exception:
            # Lookup failure is non-fatal — the delete below will still run,
            # and if we end up without a trade_id the recompute is skipped.
            pass

        # LIFO-safety check (same rationale as edit_transaction_endpoint):
        # reject deletions that would leave SELL shares unmatched, before we
        # soft-delete the row.
        if effective_trade_id and not df_d.empty:
            txns_for_trade = df_d[df_d["trade_id"] == effective_trade_id] \
                if "trade_id" in df_d.columns else df_d.iloc[0:0]
            err = validate_post_edit_lifo(
                txns_for_trade, int(detail_id),
                "DELETE", 0.0, 0.0, "",
            )
            if err:
                return {"error": err}

        try:
            db.delete_detail_row(portfolio, int(detail_id))
        except ValueError as e:
            return {"error": str(e)}

        if effective_trade_id:
            try:
                _recompute_summary_lifo(portfolio, effective_trade_id, effective_ticker)
            except Exception:
                # Summary recompute failure shouldn't roll back the delete —
                # the row is gone; the worst case is a stale summary that
                # the next edit/recompute will heal.
                pass

        try:
            db.log_audit(portfolio, "DELETE_TXN", effective_trade_id, "",
                         f"Transaction {detail_id} deleted", username="web")
        except Exception:
            pass

        return {"status": "ok", "detail_id": detail_id}
    except Exception as e:
        return {"error": str(e)}


def _recompute_summary_lifo(portfolio: str, trade_id: str, ticker: str, fallback_open_date: str = "") -> None:
    """Recompute a trade campaign's summary from its remaining detail rows
    using LIFO and replace its lot_closures rows. If no details remain,
    deletes the summary and any orphan closures. Shared helper used by
    delete-by-date cleanup."""
    df_d = db.load_details(portfolio)
    if df_d.empty:
        db.delete_trade(portfolio, trade_id)
        _safe_delete_lot_closures(portfolio, trade_id)
        return
    df_d = _normalize_trades(df_d)
    txns = df_d[df_d["trade_id"] == trade_id]
    # Resolve multiplier from the campaign's detail rows (Migration 016). Falls
    # back to ticker-pattern autodetect for any pre-migration row that still
    # has the default 1× multiplier.
    instrument_type = 'STOCK'
    multiplier = 1.0
    if not txns.empty:
        if "multiplier" in txns.columns:
            mults = pd.to_numeric(txns["multiplier"], errors="coerce").dropna()
            if not mults.empty and float(mults.max()) > 1:
                multiplier = float(mults.max())
        if "instrument_type" in txns.columns:
            types = txns["instrument_type"].dropna().astype(str).str.upper().unique().tolist()
            if 'OPTION' in types:
                instrument_type = 'OPTION'
        if multiplier == 1.0 and is_option_ticker(ticker):
            multiplier = 100.0
            instrument_type = 'OPTION'
    result = compute_lifo_summary(
        txns, trade_id, ticker, fallback_open_date,
        multiplier=multiplier, with_closures=True,
    )
    summary_row, closures = result
    if summary_row is None:
        db.delete_trade(portfolio, trade_id)
        _safe_delete_lot_closures(portfolio, trade_id)
        return
    summary_row["Instrument_Type"] = instrument_type
    summary_row["Multiplier"] = multiplier
    # Preserve user-entered fields that LIFO doesn't compute. compute_lifo_summary
    # only returns LIFO-derived fields (status, shares, avg_entry, etc.), so passing
    # its output directly to save_summary_with_closures would bind NULL/DEFAULT to
    # rule, buy_notes, sell_rule, sell_notes, risk_budget, stop_loss — wiping user
    # metadata on every sell/edit/delete/rebuild.
    try:
        df_s_existing = db.load_summary(portfolio)
        if not df_s_existing.empty:
            df_s_existing = _normalize_trades(df_s_existing)
            existing_match = df_s_existing[df_s_existing["trade_id"] == trade_id]
            if not existing_match.empty:
                existing_row = existing_match.iloc[0]
                for snake, pascal in (("rule", "Rule"),
                                      ("buy_notes", "Buy_Notes"),
                                      ("sell_rule", "Sell_Rule"),
                                      ("sell_notes", "Sell_Notes"),
                                      ("risk_budget", "Risk_Budget"),
                                      ("stop_loss", "Stop_Loss"),
                                      ("notes", "Notes")):
                    val = existing_row.get(snake)
                    if pd.notna(val):
                        summary_row[pascal] = val
    except Exception as e:
        print(f"[recompute] preserve-existing-fields failed for {trade_id}: {e}")
    # Try the combined write first so summary + closures land together.
    # Falls back to summary-only if lot_closures isn't there yet (deploy ran
    # before migration 017) or if the closures phase fails — summary is the
    # high-stakes write; closures self-heal on the next recompute. Failures
    # are logged loudly so we don't silently ship a permanently-stale state.
    try:
        db.save_summary_with_closures(portfolio, trade_id, summary_row, closures)
    except Exception as e:
        print(f"[lot_closures] save_summary_with_closures failed for {trade_id}: {e}. "
              f"Falling back to summary-only write.")
        db.save_summary_row(portfolio, summary_row)


def _safe_delete_lot_closures(portfolio: str, trade_id: str) -> None:
    """Best-effort cleanup of lot_closures rows when the parent trade is gone.
    Tolerates a missing table (deploy-before-migrate) by logging and moving on."""
    try:
        db.delete_lot_closures_for_trade(portfolio, trade_id)
    except Exception as e:
        print(f"[lot_closures] delete_lot_closures_for_trade failed for {trade_id}: {e}")


@app.delete("/api/trades/delete-transactions-by-date")
def delete_transactions_by_date(date: str = Query(...), portfolio: str = Query("CanSlim")):
    """Delete every trades_details row on the given date (YYYY-MM-DD) and
    cascade-clean affected campaigns: rows with no remaining details get
    their summary deleted; the rest get LIFO-recomputed. Use for "undo"
    after an IBKR import mishap."""
    try:
        day = str(date).strip()[:10]
        if not day:
            return {"error": "date required (YYYY-MM-DD)"}

        df_d = db.load_details(portfolio)
        if df_d.empty:
            return {"status": "ok", "deleted": 0, "trade_ids": []}
        df_d = _normalize_trades(df_d)
        df_d["date_str"] = df_d["date"].astype(str).str[:10]
        rows_today = df_d[df_d["date_str"] == day]
        if rows_today.empty:
            return {"status": "ok", "deleted": 0, "trade_ids": []}

        # Capture the affected (trade_id, ticker) pairs BEFORE deleting,
        # so we can recompute/clean-up their summaries afterward.
        affected = rows_today[["trade_id", "ticker", "detail_id"]].copy()
        ticker_by_tid = {}
        for _, r in affected.iterrows():
            tid = str(r["trade_id"])
            if tid not in ticker_by_tid:
                ticker_by_tid[tid] = str(r.get("ticker", ""))

        deleted = 0
        for did in affected["detail_id"].dropna().tolist():
            try:
                db.delete_detail_row(portfolio, int(did))
                deleted += 1
            except Exception:
                pass

        for tid, ticker in ticker_by_tid.items():
            try:
                _recompute_summary_lifo(portfolio, tid, ticker)
            except Exception:
                pass

        try:
            db.log_audit(portfolio, "DELETE", "", "",
                         f"Deleted {deleted} txn(s) for {day} — affected: {', '.join(ticker_by_tid.keys())}",
                         username="web")
        except Exception:
            pass

        return {"status": "ok", "deleted": deleted, "trade_ids": list(ticker_by_tid.keys())}
    except Exception as e:
        return {"error": str(e)}


@app.delete("/api/trades/delete")
def delete_trade_endpoint(trade_id: str = Query(...), portfolio: str = Query("CanSlim")):
    """Permanently delete a trade and all its transactions."""
    try:
        if not trade_id:
            return {"error": "trade_id is required"}
        db.delete_trade(portfolio, trade_id)
        try:
            db.log_audit(portfolio, "DELETE", trade_id, "",
                         f"Campaign permanently deleted", username="web")
        except Exception:
            pass
        return {"status": "ok", "trade_id": trade_id}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/trades/manual-price")
def set_trade_manual_price(body: dict):
    """Set or clear the manual_price override on an open trades_summary row.

    Body:
        portfolio:    "CanSlim" | ...
        trade_id:     "202602-001"
        manual_price: <number> | null    (null clears the override)

    yfinance can't reliably resolve OCC option symbols, so the live price for
    options is unstable. The override lets the user pin a per-position price
    that nlv_service + /api/prices/batch prefer over the live result. Cleared
    automatically when the user blanks the field on the ACS row.
    """
    try:
        portfolio = body.get("portfolio") or "CanSlim"
        trade_id = (body.get("trade_id") or "").strip()
        if not trade_id:
            return {"error": "trade_id is required"}

        raw = body.get("manual_price")
        manual_price: float | None
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            manual_price = None
        else:
            try:
                manual_price = float(raw)
            except (TypeError, ValueError):
                return {"error": "manual_price must be numeric or null"}
            if manual_price <= 0:
                return {"error": "manual_price must be greater than zero"}

        updated = db.set_manual_price(portfolio, trade_id, manual_price)
        if updated is None:
            return {"error": "Trade not found or manual_price column missing"}
        try:
            db.load_summary.clear()
        except Exception:
            pass
        try:
            note = (
                f"manual_price cleared" if manual_price is None
                else f"manual_price set to {manual_price}"
            )
            db.log_audit(portfolio, "MANUAL_PRICE", trade_id, "", note, username="web")
        except Exception:
            pass

        # Serialize the timestamp for JSON.
        if updated.get("manual_price_set_at") is not None:
            updated["manual_price_set_at"] = updated["manual_price_set_at"].isoformat()
        if updated.get("manual_price") is not None:
            updated["manual_price"] = float(updated["manual_price"])
        return {"status": "ok", **updated}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/trades/flag-be")
def flag_be_rule(body: dict):
    """Manually set or clear the be_stop_moved_at flag for a trade without
    changing the stop price. Used to backfill trades where the user already
    moved stop to BE before the auto-detect was wired up.
    """
    try:
        portfolio = body.get("portfolio", "CanSlim")
        trade_id = body.get("trade_id", "")
        flagged = bool(body.get("flagged", True))
        if not trade_id:
            return {"error": "trade_id is required"}
        with db.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio,))
                row = cur.fetchone()
                if not row:
                    return {"error": f"Portfolio '{portfolio}' not found"}
                portfolio_id = row[0]
                if flagged:
                    cur.execute("""
                        UPDATE trades_summary
                        SET be_stop_moved_at = COALESCE(be_stop_moved_at, NOW())
                        WHERE portfolio_id = %s AND trade_id = %s
                          AND deleted_at IS NULL
                    """, (portfolio_id, trade_id))
                else:
                    cur.execute("""
                        UPDATE trades_summary
                        SET be_stop_moved_at = NULL
                        WHERE portfolio_id = %s AND trade_id = %s
                          AND deleted_at IS NULL
                    """, (portfolio_id, trade_id))
                rowcount = cur.rowcount
                conn.commit()
        try:
            db.load_summary.clear()
        except Exception:
            pass
        try:
            db.log_audit(portfolio, "BE_FLAG", trade_id, "",
                         f"BE rule flag {'set' if flagged else 'cleared'}",
                         username="web")
        except Exception:
            pass
        return {"status": "ok", "trade_id": trade_id, "flagged": flagged, "updated": rowcount}
    except Exception as e:
        return {"error": str(e)}


@app.put("/api/trades/update-stops")
def update_trade_stops(body: dict):
    """Update stop loss across every open lot of a trade. If the new stop is
    within 0.5% of avg_entry AND the current price is ≥ 10% above avg_entry,
    stamp be_stop_moved_at as the +10% BE rule being applied. If the stop
    later moves off BE, clear the flag.
    """
    try:
        portfolio = body.get("portfolio", "CanSlim")
        trade_id = body.get("trade_id", "")
        new_stop = float(body.get("new_stop") or 0)
        if not trade_id or new_stop <= 0:
            return {"error": "trade_id and new_stop (>0) are required"}

        df_s = db.load_summary(portfolio)
        if df_s is None or df_s.empty:
            return {"error": "No trades found"}
        df_s = _normalize_trades(df_s)
        match = df_s[df_s["trade_id"] == trade_id]
        if match.empty:
            return {"error": f"Trade {trade_id} not found"}
        row = match.iloc[0]
        ticker = str(row.get("ticker", "")).strip()
        avg_entry = float(row.get("avg_entry", 0) or 0)

        # Fetch current price to detect the +10% BE rule condition. Failure
        # falls back to no auto-flagging (user can still set via sell rule).
        current_price = 0.0
        if ticker:
            try:
                import yfinance as yf
                if _is_option_ticker(ticker):
                    occ = _to_occ_symbol(ticker)
                    fetch_sym = occ or ticker
                else:
                    fetch_sym = ticker
                data = yf.download(fetch_sym, period="1d", progress=False, auto_adjust=False)["Close"]
                if not data.empty:
                    current_price = float(data.iloc[-1])
            except Exception:
                pass

        # BE rule detection: new stop within 0.5% of avg_entry AND price
        # ≥ avg_entry × 1.10. Tolerant window handles small rounding.
        be_applied = False
        be_cleared = False
        if avg_entry > 0:
            stop_near_be = abs(new_stop - avg_entry) / avg_entry <= 0.005
            price_up_10 = current_price >= avg_entry * 1.10
            be_applied = stop_near_be and price_up_10
            # If we're moving the stop AWAY from BE (to a higher or lower
            # value outside the tolerance), clear any prior BE flag.
            be_cleared = not stop_near_be

        # Update all open-lot stop_loss values in trades_details + the
        # trades_summary.stop_loss mirror, plus the BE flag.
        updated_lots = db.update_trade_stops(portfolio, trade_id, new_stop,
                                             be_applied=be_applied,
                                             be_cleared=be_cleared)
        try:
            db.log_audit(portfolio, "STOP_UPDATE", trade_id, ticker,
                         f"Stop → ${new_stop:.2f} across {updated_lots} lot(s)" +
                         (" · BE rule applied" if be_applied else ""),
                         username="web")
        except Exception:
            pass
        return {"status": "ok", "trade_id": trade_id, "updated_lots": updated_lots,
                "be_applied": be_applied, "current_price": round(current_price, 4)}
    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}


# ============================================================
# FUNDAMENTALS ENDPOINT
# ============================================================
@app.get("/api/fundamentals/{trade_id}")
def get_fundamentals(trade_id: str, portfolio: str = "CanSlim"):
    """Get extracted MarketSurge fundamentals for a trade."""
    try:
        funds = db.get_trade_fundamentals(portfolio, trade_id)
        # Convert Decimal to float for JSON serialization
        for f in funds:
            for k, v in f.items():
                if hasattr(v, 'as_tuple'):  # Decimal
                    f[k] = float(v)
                elif hasattr(v, 'isoformat'):  # datetime
                    f[k] = v.isoformat()
        return funds or []
    except Exception as e:
        return {"error": str(e)}


@app.delete("/api/fundamentals/{trade_id}")
def delete_fundamentals(trade_id: str, portfolio: str = "CanSlim"):
    """Hard-delete all extracted MarketSurge fundamentals rows for a trade.
    Used to re-extract from a new screenshot or to clear a bad extraction.
    """
    try:
        with db.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio,))
                row = cur.fetchone()
                if not row:
                    return {"error": f"Portfolio '{portfolio}' not found"}
                portfolio_id = row[0]
                cur.execute(
                    "DELETE FROM trade_fundamentals WHERE portfolio_id = %s AND trade_id = %s",
                    (portfolio_id, trade_id),
                )
                deleted = cur.rowcount
                conn.commit()
        try:
            db.get_trade_fundamentals.clear()
        except Exception:
            pass
        return {"status": "ok", "deleted": deleted}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# R2 IMAGE ENDPOINTS
# ============================================================
from fastapi import UploadFile, File, Form

@app.get("/api/images/{trade_id}")
def get_trade_images(trade_id: str, portfolio: str = "CanSlim"):
    """Get all image metadata for a trade."""
    try:
        images = db.get_trade_images(portfolio, trade_id)
        # Build viewable URLs — use public R2 CDN
        R2_PUBLIC = (os.environ.get("R2_PUBLIC_URL") or "").rstrip("/")
        if images:
            for img in images:
                key = img.get("image_url", "")
                if key and key.startswith("http"):
                    img["view_url"] = key
                elif key:
                    img["view_url"] = f"{R2_PUBLIC}/{key}"
                else:
                    img["view_url"] = ""
        return images or []
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/images/upload")
@limiter.limit("5/minute")
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
    portfolio: str = Form("CanSlim"),
    trade_id: str = Form(...),
    ticker: str = Form(...),
    image_type: str = Form(...),
):
    """Upload a trade image to R2 and save metadata to DB.
    For marketsurge images, also extracts fundamentals via Claude Vision."""
    if not _is_r2_available():
        return {"error": "R2 storage not configured"}
    try:
        # Read file content
        content = await file.read()
        file_like = io.BytesIO(content)
        file_like.name = file.filename or "upload.png"

        # For MarketSurge screenshots, save as 'entry' type so it appears
        # in Entry Charts, and also run AI extraction
        save_type = "entry" if image_type == "marketsurge" else image_type

        # Upload to R2
        object_key = r2.upload_image(file_like, portfolio, trade_id, ticker, save_type)
        if not object_key:
            return {"error": "Upload to R2 failed"}

        # Save metadata to DB
        image_id = db.save_trade_image(portfolio, trade_id, ticker, save_type, object_key, file.filename)

        # Extract fundamentals from MarketSurge screenshots
        fundamentals = None
        if image_type == "marketsurge":
            try:
                import vision_extract
                extracted = vision_extract.extract_fundamentals(content, file.filename or "image.png")
                if extracted:
                    db.save_trade_fundamentals(portfolio, trade_id, ticker, extracted, image_id)
                    fundamentals = extracted
                    print(f"[Vision] Extracted fundamentals for {ticker} ({trade_id})")
                else:
                    print(f"[Vision] No data extracted for {ticker}")
            except Exception as ve:
                print(f"[Vision] Extraction failed: {ve}")

        return {"status": "ok", "image_id": image_id, "object_key": object_key, "fundamentals": fundamentals}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/snapshots/upload")
@limiter.limit("5/minute")
async def upload_eod_snapshot(
    request: Request,
    file: UploadFile = File(...),
    portfolio: str = Form("CanSlim"),
    day: str = Form(...),
    snapshot_type: str = Form(...),  # "dashboard" or "campaign"
):
    """Upload an end-of-day snapshot (PNG) to R2 tied to a journal day."""
    if not _is_r2_available():
        return {"error": "R2 storage not configured"}
    try:
        content = await file.read()
        file_like = io.BytesIO(content)
        file_like.name = file.filename or f"{snapshot_type}.png"

        # Use synthetic trade_id: EOD-2026-04-20
        trade_id = f"EOD-{day}"
        ticker = snapshot_type.upper()
        image_type = f"eod_{snapshot_type}"

        object_key = r2.upload_image(file_like, portfolio, trade_id, ticker, image_type)
        if not object_key:
            return {"error": "Upload to R2 failed"}

        image_id = db.save_trade_image(portfolio, trade_id, ticker, image_type, object_key, file.filename)
        return {"status": "ok", "image_id": image_id, "object_key": object_key}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/snapshots/{day}")
def list_eod_snapshots(day: str, portfolio: str = "CanSlim"):
    """List EOD snapshots for a specific day."""
    try:
        trade_id = f"EOD-{day}"
        images = db.get_trade_images(portfolio, trade_id)
        R2_PUBLIC = (os.environ.get("R2_PUBLIC_URL") or "").rstrip("/")
        if images:
            for img in images:
                key = img.get("image_url", "")
                if key and str(key).startswith("http"):
                    img["view_url"] = key
                elif key:
                    img["view_url"] = f"{R2_PUBLIC}/{key}"
                else:
                    img["view_url"] = ""
        return images or []
    except Exception as e:
        return {"error": str(e)}


@app.delete("/api/images/{image_id}")
def delete_image(image_id: int):
    """Delete a trade image from R2 and DB."""
    try:
        # Get the image URL from DB before deleting
        if _is_r2_available():
            url = db.delete_trade_image_by_id(image_id)
            if url and not url.startswith("http"):
                r2.delete_image(url)
        else:
            db.delete_trade_image_by_id(image_id)
        return {"status": "ok"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/r2/status")
def r2_status():
    """Check R2 availability."""
    return {"available": _is_r2_available()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
