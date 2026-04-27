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
    calc_risk_budget,
    compute_lifo_summary,
    normalize_journal_columns as _normalize_journal,
)


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

# Paths that remain reachable without a bearer token.
_PUBLIC_PATHS = {"/api/health", "/"}

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
def journal_latest(portfolio: str = "CanSlim"):
    """Get the most recent journal entry (NLV, daily change, etc.)."""
    df = db.load_journal(portfolio)
    if df.empty:
        return {"error": "No journal data"}
    df = _normalize_journal(df)
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
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

    # Clean numeric columns
    for c in ["beg_nlv", "end_nlv", "cash_change", "daily_dollar_change",
              "daily_pct_change", "pct_invested", "spy", "nasdaq",
              "portfolio_heat", "spy_atr", "nasdaq_atr"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

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
            "market_window", "market_cycle", "market_notes", "market_action",
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
        }

        # Auto-compute missing market/risk metrics.
        # market_window is deprecated as of MCT V11 Phase 3a — no longer auto-filled.
        # Existing values are preserved if the caller sends them; new entries get NULL.
        day_str = str(day).strip()[:10]
        if not journal_entry["market_cycle"]:
            journal_entry["market_cycle"] = _compute_cycle_state(day_str)
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
    """Get all transactions for open trades (for stop loss, pyramid info)."""
    summary_df = db.load_summary(portfolio)
    if summary_df.empty:
        return []
    summary_df = _normalize_trades(summary_df)
    status_col = "status" if "status" in summary_df.columns else "Status"
    open_ids = summary_df[summary_df[status_col].str.upper() == "OPEN"]["trade_id"].tolist()
    if not open_ids:
        return []
    details_df = db.load_details(portfolio)
    if details_df.empty:
        return []
    details_df = _normalize_trades(details_df)
    filtered = details_df[details_df["trade_id"].isin(open_ids)].copy()
    if "date" in filtered.columns:
        filtered["date"] = pd.to_datetime(filtered["date"], errors="coerce")
        filtered = filtered.sort_values(["trade_id", "date"])
    return _df_to_records(filtered)


@app.get("/api/trades/recent")
def trades_recent(portfolio: str = "CanSlim", limit: int = 20):
    """Get most recent trade transactions (buys + sells)."""
    df = db.load_details(portfolio)
    if df.empty:
        return []
    df = _normalize_trades(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date", ascending=False)
    return _df_to_records(df.head(limit))


# ============================================================
# LIVE PRICES
# ============================================================
import re as _re

def _is_option_ticker(ticker):
    return bool(ticker and '$' in ticker and _re.search(r'\d{6}', ticker))

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
def batch_prices(request: Request, tickers: str = "", portfolio: str = ""):
    """Get live prices for a comma-separated list of tickers.
    Supports both stock tickers and readable option format ('LUMN 260717 $8C').
    Returns dict of {readable_ticker: price}.

    Delegates to the shared PriceProvider so the Dashboard NLV and Active
    Campaign Current columns are guaranteed to agree — both go through the
    same yfinance path.

    When portfolio is provided, manual_price overrides on open positions in
    that portfolio take precedence over the yfinance result for matching
    tickers. Without portfolio, behavior is unchanged (yfinance only).
    """
    if not tickers.strip():
        return {}
    from price_providers import get_price_provider

    ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]
    yf_to_readable: dict[str, str] = {}
    yf_symbols: list[str] = []
    for t in ticker_list:
        if _is_option_ticker(t):
            occ = _to_occ_symbol(t)
            if occ:
                yf_symbols.append(occ)
                yf_to_readable[occ] = t
        else:
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
# NLV SHADOW (read-only comparison vs manually-entered NLV)
# ============================================================
@app.get("/api/nlv/shadow-today")
def nlv_shadow_today(portfolio: str = "CanSlim"):
    """Compute today's NLV from yesterday's journal entry + current position
    prices + today's trade cash flows. Read-only; does not write to DB.

    Formula:
        yesterday_cash = yesterday.End_NLV × (1 − yesterday.% Invested / 100)
        today_cash     = yesterday_cash + today.Cash_Change + today_trade_flow
                         (buys subtract, sells add)
        computed_nlv   = today_cash + Σ(open_shares × today_price)

    Returned alongside manually-entered NLV (if any) so the UI can show the
    delta. This is a read-only shadow endpoint — rollback is just deleting it.
    """
    import yfinance as yf
    from datetime import datetime
    try:
        import pytz
        central = pytz.timezone('America/Chicago')
        today = datetime.now(central).date()
    except Exception:
        today = datetime.now().date()

    try:
        journal = db.load_journal(portfolio)
        if journal is None or journal.empty:
            return {"error": "No journal entries found", "portfolio": portfolio}

        journal = journal.copy()
        journal['Day'] = pd.to_datetime(journal['Day']).dt.date

        prior = journal[journal['Day'] < today].sort_values('Day')
        if prior.empty:
            return {"error": "No prior-day journal entry to anchor from", "portfolio": portfolio}
        prior_row = prior.iloc[-1]

        prior_day = prior_row['Day']
        yesterday_end_nlv = float(prior_row['End NLV'] or 0)
        yesterday_pct_invested = float(prior_row['% Invested'] or 0)
        yesterday_cash = yesterday_end_nlv * (1 - yesterday_pct_invested / 100)

        today_rows = journal[journal['Day'] == today]
        today_cash_change = 0.0
        manual_nlv = None
        if not today_rows.empty:
            today_row = today_rows.iloc[0]
            today_cash_change = float(today_row.get('Cash -/+') or 0)
            mv = today_row.get('End NLV')
            if mv is not None and float(mv) > 0:
                manual_nlv = float(mv)

        # Today's trade cash flows (BUY subtracts, SELL adds). For options,
        # trades_details.Value is shares*amount (no 100x), so we recompute
        # with the contract multiplier to match actual broker cash impact.
        details = db.load_details(portfolio)
        today_trade_flow = 0.0
        if details is not None and not details.empty:
            details = details.copy()
            details['Date'] = pd.to_datetime(details['Date']).dt.date
            today_trades = details[details['Date'] == today]
            for _, row in today_trades.iterrows():
                t_ticker = str(row.get('Ticker') or '').strip()
                shares_d = float(row.get('Shares') or 0)
                amount_d = float(row.get('Amount') or 0)
                mult_d = 100.0 if _is_option_ticker(t_ticker) else 1.0
                value = shares_d * amount_d * mult_d
                action = str(row.get('Action') or '').upper()
                if action == 'BUY':
                    today_trade_flow -= value
                elif action == 'SELL':
                    today_trade_flow += value

        today_cash = yesterday_cash + today_cash_change + today_trade_flow

        # Value open positions at today's price. Options need OCC conversion
        # (yfinance can't resolve "LUMN 260717 $8C") plus the 100x contract
        # multiplier — same logic Active Campaign Summary uses via batch_prices.
        summary = db.load_summary(portfolio, status='OPEN')
        holdings_value = 0.0
        breakdown = []
        missing_prices = []
        if summary is not None and not summary.empty:
            tickers = [str(t).strip() for t in summary['Ticker'].tolist() if t]
            yf_symbols = []
            yf_to_readable = {}
            for t in tickers:
                if _is_option_ticker(t):
                    occ = _to_occ_symbol(t)
                    if occ:
                        yf_symbols.append(occ)
                        yf_to_readable[occ] = t
                else:
                    yf_symbols.append(t)
                    yf_to_readable[t] = t

            prices: dict = {}
            if yf_symbols:
                try:
                    data = yf.download(yf_symbols, period='1d', progress=False, auto_adjust=False)['Close']
                    if len(yf_symbols) == 1:
                        if not data.empty:
                            readable = yf_to_readable.get(yf_symbols[0], yf_symbols[0])
                            prices[readable] = float(data.iloc[-1])
                    else:
                        last = data.iloc[-1]
                        for yf_sym in yf_symbols:
                            if yf_sym in last.index and not pd.isna(last[yf_sym]):
                                readable = yf_to_readable.get(yf_sym, yf_sym)
                                prices[readable] = float(last[yf_sym])
                except Exception:
                    pass

            for _, pos in summary.iterrows():
                ticker = str(pos['Ticker']).strip()
                shares = float(pos.get('Shares') or 0)
                price = prices.get(ticker)
                if price is None:
                    missing_prices.append(ticker)
                    price = 0.0
                # Options trade per-share but each contract represents 100
                # underlying shares — apply the 100x multiplier so the shadow
                # matches how brokers/Active Campaign Summary report value.
                multiplier = 100.0 if _is_option_ticker(ticker) else 1.0
                value = shares * price * multiplier
                holdings_value += value
                breakdown.append({
                    "ticker": ticker,
                    "shares": shares,
                    "price": round(price, 4),
                    "multiplier": multiplier,
                    "value": round(value, 2),
                })

        computed_nlv = today_cash + holdings_value

        diff = None
        diff_pct = None
        if manual_nlv is not None and manual_nlv > 0:
            diff = manual_nlv - computed_nlv
            diff_pct = (diff / manual_nlv) * 100

        return {
            "portfolio": portfolio,
            "as_of": today.isoformat(),
            "prior_day": prior_day.isoformat() if hasattr(prior_day, 'isoformat') else str(prior_day),
            "yesterday_end_nlv": round(yesterday_end_nlv, 2),
            "yesterday_pct_invested": round(yesterday_pct_invested, 2),
            "yesterday_cash": round(yesterday_cash, 2),
            "today_cash_change": round(today_cash_change, 2),
            "today_trade_flow": round(today_trade_flow, 2),
            "today_cash": round(today_cash, 2),
            "today_holdings_value": round(holdings_value, 2),
            "computed_nlv": round(computed_nlv, 2),
            "manual_nlv": round(manual_nlv, 2) if manual_nlv is not None else None,
            "diff": round(diff, 2) if diff is not None else None,
            "diff_pct": round(diff_pct, 4) if diff_pct is not None else None,
            "position_breakdown": breakdown,
            "missing_prices": missing_prices,
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}


# ============================================================
# MARKET ENDPOINTS
# ============================================================
@app.get("/api/market/rally-prefix")
def rally_prefix(as_of_date: str = ""):
    """Get rally day prefix for Market/Global Notes (e.g., 'Day 14 POWERTREND:').

    V11 implementation: replays the MCT engine over market_data history and
    translates the result into the legacy response shape. If as_of_date is
    provided (YYYY-MM-DD), slices history to bars on or before that date.
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
        return to_rally_prefix_response(result)
    except Exception as e:
        return {"prefix": "", "error": str(e)}


@app.get("/api/market/rally-data")
@limiter.limit("25/minute")
def rally_data(request: Request, ftd_date: str = "", index: str = "^IXIC"):
    """Fetch index closes from FTD through Day 25 for Rally Context chart."""
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
        df_post = df[ftd_mask]
        if df_post.empty:
            return {"error": f"No data on or after {ftd_date}"}

        day0_close = float(df_post.iloc[0]["Close"])
        points = []
        for i in range(1, min(len(df_post), 26)):
            row = df_post.iloc[i]
            pct = ((float(row["Close"]) / day0_close) - 1) * 100
            points.append({
                "day": i,
                "date": df_post.index[i].strftime("%Y-%m-%d"),
                "close": round(float(row["Close"]), 2),
                "low": round(float(row["Low"]), 2),
                "pct": round(pct, 2),
            })

        return {"day0_close": round(day0_close, 2), "points": points}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/market/mfactor")
@limiter.limit("30/minute")
def market_mfactor(request: Request):
    """Get current M Factor state for NASDAQ + SPY with full MA stack."""
    try:
        import yfinance as yf

        result = {}
        for ticker, label in [("^IXIC", "nasdaq"), ("SPY", "spy")]:
            df = yf.Ticker(ticker).history(period="1y")
            if df.empty:
                continue
            df["21EMA"] = df["Close"].ewm(span=21, adjust=False).mean()
            df["50SMA"] = df["Close"].rolling(window=50).mean()
            df["200SMA"] = df["Close"].rolling(window=200).mean()
            curr = df.iloc[-1]

            price = float(curr["Close"])
            ema21 = float(curr["21EMA"]) if pd.notna(curr["21EMA"]) else 0
            sma50 = float(curr["50SMA"]) if pd.notna(curr["50SMA"]) else 0
            sma200 = float(curr["200SMA"]) if pd.notna(curr["200SMA"]) else 0

            above_21 = bool(price > ema21) if ema21 > 0 else False
            above_50 = bool(price > sma50) if sma50 > 0 else False
            above_200 = bool(price > sma200) if sma200 > 0 else False

            # Powertrend: Low > 21 EMA for 3+ consecutive days
            low_above_21_streak = 0
            for i in range(len(df) - 1, -1, -1):
                row = df.iloc[i]
                if pd.notna(row.get("21EMA")) and row["Low"] > row["21EMA"]:
                    low_above_21_streak += 1
                else:
                    break
            is_powertrend = low_above_21_streak >= 3 and above_21 and above_50

            # Individual state
            if is_powertrend:
                state = "POWERTREND"
            elif above_21:
                state = "OPEN"
            elif above_50:
                state = "NEUTRAL"
            else:
                state = "CLOSED"

            # % distance from MAs
            d21 = ((price - ema21) / ema21 * 100) if ema21 > 0 else 0
            d50 = ((price - sma50) / sma50 * 100) if sma50 > 0 else 0
            d200 = ((price - sma200) / sma200 * 100) if sma200 > 0 else 0

            result[label] = {
                "price": round(price, 2),
                "ema21": round(ema21, 2),
                "sma50": round(sma50, 2),
                "sma200": round(sma200, 2),
                "above_21ema": above_21,
                "above_50sma": above_50,
                "above_200sma": above_200,
                "d21": round(d21, 2),
                "d50": round(d50, 2),
                "d200": round(d200, 2),
                "state": state,
                "powertrend_streak": low_above_21_streak,
            }

        # Combined state
        ns = result.get("nasdaq", {}).get("state", "CLOSED")
        ss = result.get("spy", {}).get("state", "CLOSED")
        if ns == "POWERTREND" or ss == "POWERTREND":
            combined = "POWERTREND"
        elif ns == "CLOSED" and ss == "CLOSED":
            combined = "CLOSED"
        elif ns in ["NEUTRAL", "CLOSED"] or ss in ["NEUTRAL", "CLOSED"]:
            combined = "NEUTRAL"
        else:
            combined = "OPEN"

        result["combined_state"] = combined
        return result
    except Exception as e:
        return {"error": str(e)}


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

        try:
            amount = float(body.get("amount") or 0)
        except (TypeError, ValueError):
            return {"error": "amount must be numeric"}
        if amount <= 0:
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
    """Generate next available trade ID for a given month."""
    try:
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        ym = pd.Timestamp(date).strftime("%Y%m")
        df_s = db.load_summary(portfolio)
        df_s = _normalize_trades(df_s)
        existing = df_s[df_s["trade_id"].str.startswith(ym)]["trade_id"].tolist() if not df_s.empty else []
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
    # Also check all env var keys that contain IBKR (case-insensitive)
    ibkr_vars = {k: f"{v[:4]}..." for k, v in os.environ.items() if "IBKR" in k.upper() or "FLEX" in k.upper()}
    return {
        "token_set": bool(token),
        "query_id_set": bool(query_id),
        "token_preview": f"{token[:4]}...{token[-4:]}" if len(token) > 8 else "(empty)",
        "env_keys_found": ibkr_vars,
        "total_env_vars": len(os.environ),
    }


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
        trx_id = body.get("trx_id", "")

        if not ticker or not trade_id or shares <= 0 or price <= 0:
            return {"error": "Missing required fields: ticker, trade_id, shares, price"}

        value = shares * price
        date_time = f"{date_str} {time_str}:00"

        # Determine trx_id if not provided
        if not trx_id:
            df_d = db.load_details(portfolio)
            if not df_d.empty:
                df_d = _normalize_trades(df_d)
                existing_txns = df_d[df_d["trade_id"] == trade_id]
                buy_count = len(existing_txns[existing_txns["action"].str.upper() == "BUY"]) if not existing_txns.empty else 0
                trx_id = f"B{buy_count + 1}" if buy_count == 0 else f"A{buy_count}"
            else:
                trx_id = "B1"

        # Build summary row first (FK requires summary before detail).
        # risk_budget = initial $ at risk at entry (shares * (entry - stop)).
        # Populated here because direct-log paths (e.g. IBKR Quick Log) skip
        # the sizing calculator that historically set this column.
        if action_type == "new":
            summary_row = {
                "Trade_ID": trade_id, "Ticker": ticker, "Status": "OPEN",
                "Open_Date": date_str, "Shares": shares,
                "Avg_Entry": price, "Total_Cost": value,
                "Stop_Loss": stop_loss, "Rule": rule, "Buy_Notes": notes,
                "Risk_Budget": calc_risk_budget(shares, price, stop_loss),
            }
        else:
            # Scale-in: load existing summary and update
            df_s = db.load_summary(portfolio)
            df_s = _normalize_trades(df_s)
            existing = df_s[df_s["trade_id"] == trade_id]
            if not existing.empty:
                row = existing.iloc[0]
                old_shares = float(row.get("shares", 0))
                old_entry = float(row.get("avg_entry", 0))
                old_cost = float(row.get("total_cost", 0))
                new_total_shares = old_shares + shares
                new_total_cost = old_cost + value
                new_avg_entry = new_total_cost / new_total_shares if new_total_shares > 0 else price
                effective_stop = float(stop_loss if stop_loss > 0 else row.get("stop_loss", 0) or 0)
                existing_rb = float(row.get("risk_budget", 0) or 0)
                added_rb = calc_risk_budget(shares, price, effective_stop)
                new_rb = existing_rb + added_rb if existing_rb > 0 or added_rb > 0 else 0.0
                summary_row = {
                    "Trade_ID": trade_id, "Ticker": ticker, "Status": "OPEN",
                    "Open_Date": str(row.get("open_date", date_str))[:10],
                    "Shares": float(new_total_shares),
                    "Avg_Entry": float(round(new_avg_entry, 4)),
                    "Total_Cost": float(round(new_total_cost, 2)),
                    "Stop_Loss": effective_stop,
                    "Rule": str(row.get("rule", "") or rule or ""),
                    "Buy_Notes": str(notes or row.get("buy_notes", "") or ""),
                    "Risk_Budget": round(new_rb, 2),
                }
            else:
                summary_row = {
                    "Trade_ID": trade_id, "Ticker": ticker, "Status": "OPEN",
                    "Open_Date": date_str, "Shares": shares,
                    "Avg_Entry": price, "Total_Cost": value,
                    "Stop_Loss": stop_loss, "Rule": rule, "Buy_Notes": notes,
                    "Risk_Budget": calc_risk_budget(shares, price, stop_loss),
                }

        summary_id = db.save_summary_row(portfolio, summary_row)

        # Save detail row (after summary so FK constraint is satisfied)
        detail_row = {
            "Trade_ID": trade_id, "Ticker": ticker, "Action": "BUY",
            "Date": date_time, "Shares": shares, "Amount": price,
            "Value": value, "Rule": rule, "Notes": notes,
            "Stop_Loss": stop_loss, "Trx_ID": trx_id,
        }
        detail_id = db.save_detail_row(portfolio, detail_row)

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
            "Sell_Rule": row.get("sell_rule") or None,
            "Notes": row.get("notes") or None,
            "Stop_Loss": row.get("stop_loss") or None,
            "Rule": row.get("rule") or None,
            "Buy_Notes": row.get("buy_notes") or None,
            "Sell_Notes": row.get("sell_notes") or None,
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
        trx_id = body.get("trx_id", "")
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

        value = shares * price
        date_time = f"{date_str} {time_str}:00"

        # Generate trx_id if not provided
        if not trx_id:
            df_d = db.load_details(portfolio)
            if not df_d.empty:
                df_d = _normalize_trades(df_d)
                existing_txns = df_d[df_d["trade_id"] == trade_id]
                sell_count = len(existing_txns[existing_txns["action"].str.upper() == "SELL"]) if not existing_txns.empty else 0
                trx_id = f"S{sell_count + 1}"
            else:
                trx_id = "S1"

        # Save detail row
        detail_row = {
            "Trade_ID": trade_id, "Ticker": ticker, "Action": "SELL",
            "Date": date_time, "Shares": shares, "Amount": price,
            "Value": value, "Rule": rule, "Notes": notes,
            "Realized_PL": 0, "Trx_ID": trx_id,
        }
        detail_id = db.save_detail_row(portfolio, detail_row)

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
        return_pct = (total_realized / total_cost * 100) if is_closed and total_cost > 0 else 0.0

        summary_row = {
            "Trade_ID": trade_id, "Ticker": str(ticker),
            "Status": "CLOSED" if is_closed else "OPEN",
            "Open_Date": str(row.get("open_date", ""))[:10],
            "Closed_Date": date_str if is_closed else None,
            "Shares": float(remaining_shares if not is_closed else total_buy_shs),
            "Avg_Entry": float(round(avg_entry, 4)),
            "Avg_Exit": float(round(avg_exit, 4)) if avg_exit > 0 else 0.0,
            "Total_Cost": float(round(remaining_cost if not is_closed else total_cost, 2)),
            "Realized_PL": float(round(total_realized, 2)),
            "Return_Pct": float(round(return_pct, 4)),
            "Sell_Rule": rule,
            "Sell_Notes": notes,
            "Rule": str(row.get("rule", "") or ""),
            "Buy_Notes": str(row.get("buy_notes", "") or ""),
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

        return {
            "status": "ok", "detail_id": detail_id, "summary_id": summary_id,
            "trx_id": trx_id, "realized_pl": round(total_realized, 2),
            "remaining_shares": round(remaining_shares, 4),
            "is_closed": is_closed,
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

        if not detail_id:
            return {"error": "detail_id is required"}

        row_dict = {
            "Trade_ID": trade_id,
            "Ticker": body.get("ticker", ""),
            "Action": body.get("action", ""),
            "Date": body.get("date", ""),
            "Shares": body.get("shares", 0),
            "Amount": body.get("amount", 0),
            "Value": body.get("value", 0),
            "Rule": body.get("rule", ""),
            "Notes": body.get("notes", ""),
            "Stop_Loss": body.get("stop_loss", 0),
            "Trx_ID": body.get("trx_id", ""),
        }

        db.update_detail_row(portfolio, detail_id, row_dict)

        # Recompute the campaign summary so avg_entry / realized_pl /
        # return_pct reflect the edited detail. Without this the face card
        # keeps stale numbers (e.g. edit a buy price after the sell already
        # closed the trade — the card still shows the pre-edit P&L).
        try:
            if trade_id:
                _recompute_summary_lifo(portfolio, trade_id, body.get("ticker", ""))
        except Exception:
            pass

        try:
            db.log_audit(portfolio, "EDIT", trade_id, row_dict.get("Trx_ID", ""),
                         f"Transaction {detail_id} edited", username="web")
        except Exception:
            pass

        return {"status": "ok", "detail_id": detail_id}
    except Exception as e:
        return {"error": str(e)}


def _recompute_summary_lifo(portfolio: str, trade_id: str, ticker: str, fallback_open_date: str = "") -> None:
    """Recompute a trade campaign's summary from its remaining detail rows
    using LIFO. If no details remain, deletes the summary entirely. Shared
    helper used by delete-by-date cleanup."""
    df_d = db.load_details(portfolio)
    if df_d.empty:
        db.delete_trade(portfolio, trade_id)
        return
    df_d = _normalize_trades(df_d)
    txns = df_d[df_d["trade_id"] == trade_id]
    summary_row = compute_lifo_summary(txns, trade_id, ticker, fallback_open_date)
    if summary_row is None:
        db.delete_trade(portfolio, trade_id)
        return
    db.save_summary_row(portfolio, summary_row)


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
