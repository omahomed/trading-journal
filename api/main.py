"""
MO Money — FastAPI backend
Wraps the database layer and supporting modules so the React frontend
can fetch real data via REST endpoints.
"""

import sys
import os
import time
from typing import Any

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
    compute_matching_summary,
    compute_open_inventory,
    compute_trade_risk,
    is_option_ticker,
    multiplier_for_ticker,
    normalize_journal_columns as _normalize_journal,
    validate_post_edit_matching,
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
# BACKGROUND: daily Sell Rule tier reconcile (b1_max_return_pct)
# ============================================================
# The persistent Sell Rule tier is max(current_b1_return, stored peak). The
# frontend promotes the stored peak live, but only while the app is open, so a
# leader that peaks >50% while the app is closed can mis-tier as SR11 on the
# pullback — and SR11 (BE stop-out) vs SR8 (RS-defended core) prescribe
# OPPOSITE sell actions. This in-process task recomputes the close-basis peak
# for every open equity position and raises any stale stored peak. The DB
# guard (db.update_b1_max_return_pct) only ever RAISES, never lowers.
#
# Runs ~90s after boot (so every Railway deploy heals immediately) and every
# 24h thereafter. Single uvicorn worker → exactly one scheduler. yfinance + DB
# work runs in a thread so it never blocks the event loop. Disabled under
# pytest and via DISABLE_B1_RECONCILE=1.
import asyncio

_B1_RECONCILE_INTERVAL_S = 24 * 60 * 60
_B1_RECONCILE_STARTUP_DELAY_S = 90


async def _b1_reconcile_loop():
    from b1_reconcile import reconcile_open_positions

    await asyncio.sleep(_B1_RECONCILE_STARTUP_DELAY_S)
    while True:
        try:
            summary = await asyncio.to_thread(
                reconcile_open_positions, None, True, False, 0.3
            )
            c = summary["counters"]
            print(f"[b1_reconcile] {c['raised']} raised · {c['unchanged']} current · "
                  f"{c['skipped_no_data']} no-data · {c['errors']} errors "
                  f"(of {summary['total']} open)")
        except Exception as exc:  # never let a bad run kill the loop
            print(f"[b1_reconcile] run failed: {exc}")
        await asyncio.sleep(_B1_RECONCILE_INTERVAL_S)


@app.on_event("startup")
async def _start_b1_reconcile():
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("DISABLE_B1_RECONCILE"):
        return
    if not os.environ.get("DATABASE_URL"):
        print("[b1_reconcile] DATABASE_URL not set — scheduler disabled.")
        return
    asyncio.create_task(_b1_reconcile_loop())
    print("[b1_reconcile] daily Sell Rule tier reconcile scheduled.")


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


def _load_robinhood_module():
    """Lazy import of the Robinhood importer script.

    Kept lazy (module-scoped after first call) so the FastAPI process
    doesn't pay the parse cost unless the endpoints are actually hit.
    scripts/ isn't on the normal PYTHONPATH; we push it once and reuse.
    """
    import sys as _sys
    from pathlib import Path as _Path
    scripts_dir = _Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in _sys.path:
        _sys.path.insert(0, str(scripts_dir))
    import import_robinhood_csv as _rh  # noqa: E402  (deferred by design)
    return _rh


def _serialize_rh_campaign(c) -> dict:
    """Campaign → JSON. Matches the fields the frontend Preview table
    binds to; keeps the raw txn objects out (each carries the full raw
    CSV row and would balloon the payload)."""
    return {
        "ticker": c.ticker,
        "instrument_type": c.instrument_type,
        "multiplier": c.multiplier,
        "open_date": c.open_date.isoformat(),
        "shares_remaining": round(c.shares_remaining, 6),
        "status": c.status,
        "txn_count": len(c.txns),
        # Same-day partial fills aggregated per (date, action) tuple —
        # what the user actually sees on their Trade Journal after import.
        "buys": sum(1 for t in c.txns if t.action == "BUY"),
        "sells": sum(1 for t in c.txns if t.action == "SELL"),
        "option_meta": c.option_meta,
    }


def _serialize_rh_cash(c) -> dict:
    return {
        "date": c.date.isoformat(),
        "amount": c.amount,
        "source": c.source,
        "note": c.note,
    }


def _parse_rh_body(body: dict) -> tuple[str, "date", str]:
    """Shared body-validation for the two Robinhood endpoints."""
    from datetime import datetime as _dt
    csv_text = body.get("csv_text") or ""
    since_str = body.get("since") or "2026-01-01"
    portfolio = (body.get("portfolio") or "").strip()
    if not csv_text or not portfolio:
        raise HTTPException(status_code=400, detail="csv_text and portfolio are required")
    try:
        since_d = _dt.strptime(since_str.strip()[:10], "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail=f"bad since date: {since_str}")
    return csv_text, since_d, portfolio


@app.post("/api/imports/robinhood/preview")
def robinhood_import_preview(request: Request, body: dict = Body(...)):
    """Dry-run parse of a Robinhood CSV. Returns the same report the CLI
    prints, structured for the /import-trades preview panel.

    Body: { csv_text: str, since: 'YYYY-MM-DD', portfolio: str }
    Returns: {
      portfolio, since, counts, pre_cutoff, existing_trades,
      warnings, equity_campaigns[], option_campaigns[], cash_rows[]
    }

    No DB writes. Duplicate detection queries trades_summary for the
    target portfolio + cutoff so the frontend can warn the user before
    they hit Commit.
    """
    csv_text, since_d, portfolio = _parse_rh_body(body)
    rh = _load_robinhood_module()

    raw_rows = rh.read_csv_from_text(csv_text)
    kept_rows, pre_cutoff = rh.filter_by_date(raw_rows, since_d)
    counts = rh.classify_counts(kept_rows)
    warnings: list[str] = []
    equity_campaigns = rh.reconstruct_equity_campaigns(kept_rows, warnings)
    option_campaigns = rh.reconstruct_option_campaigns(kept_rows, warnings)
    cash_rows = rh.extract_cash_rows(kept_rows, warnings)
    warnings.extend(rh.collect_short_option_warnings(kept_rows))

    # Duplicate detection — count existing trades in the target portfolio
    # since the cutoff. Non-fatal on failure (portfolio might not exist yet).
    existing_trades = 0
    try:
        with db.get_db_connection() as conn, conn.cursor() as cur:
            portfolio_id, _uid = rh._resolve_portfolio(cur, portfolio)
            existing_trades = rh.check_existing_trades(cur, portfolio_id, since_d)
    except Exception as e:
        print(f"[robinhood_preview] existing_trades check skipped: {e}")

    return {
        "portfolio": portfolio,
        "since": since_d.isoformat(),
        "counts": counts,
        "pre_cutoff": pre_cutoff,
        "existing_trades": existing_trades,
        "warnings": warnings,
        "equity_campaigns": [_serialize_rh_campaign(c) for c in equity_campaigns],
        "option_campaigns": [_serialize_rh_campaign(c) for c in option_campaigns],
        "cash_rows": [_serialize_rh_cash(c) for c in cash_rows],
    }


@app.post("/api/imports/robinhood/commit")
def robinhood_import_commit(request: Request, body: dict = Body(...)):
    """Actually write the campaigns + cash rows parsed from a Robinhood CSV.

    Same parsing pipeline as /preview, then runs the script's write path
    inside a single transaction. Rolls back on any exception mid-write.

    Body: { csv_text, since, portfolio, strategy?, reset_cash_ledger? }
    Returns: { written: {summary, details, cash}, reset_count?, warnings[] }
    """
    csv_text, since_d, portfolio = _parse_rh_body(body)
    strategy = str(body.get("strategy") or "LongTerm")
    reset_cash_ledger = bool(body.get("reset_cash_ledger", False))
    rh = _load_robinhood_module()

    raw_rows = rh.read_csv_from_text(csv_text)
    kept_rows, _pre_cutoff = rh.filter_by_date(raw_rows, since_d)
    warnings: list[str] = []
    equity_campaigns = rh.reconstruct_equity_campaigns(kept_rows, warnings)
    option_campaigns = rh.reconstruct_option_campaigns(kept_rows, warnings)
    cash_rows = rh.extract_cash_rows(kept_rows, warnings)
    warnings.extend(rh.collect_short_option_warnings(kept_rows))

    try:
        with db.get_db_connection() as conn, conn.cursor() as cur:
            portfolio_id, user_id = rh._resolve_portfolio(cur, portfolio)

            reset_count = None
            if reset_cash_ledger:
                reset_count = rh.reset_cash_ledger(cur, portfolio_id, user_id)

            s_count, d_count = rh.write_campaigns(
                cur, portfolio_id, user_id, portfolio,
                equity_campaigns + option_campaigns,
                strategy,
            )
            c_count = rh.write_cash_rows(cur, portfolio_id, user_id, cash_rows)

            conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import commit failed: {e}")

    return {
        "portfolio": portfolio,
        "since": since_d.isoformat(),
        "reset_count": reset_count,
        "written": {"summary": s_count, "details": d_count, "cash": c_count},
        "warnings": warnings,
    }


@app.get("/api/mindset/traps")
def mindset_traps(request: Request, portfolio: str = "CanSlim", weeks: int = 8):
    """Aggregate behavior tags over the last N weeks.

    Powers two views from a single call:
      1. Recurring Traps strip on Weekly Retro — the top 3 tags by
         total_count over the window.
      2. Trader Mindset page — heat map (tag x week), per-tag trend
         lines, and per-tag drill-through to the specific trades that
         triggered each fire.

    Response shape:
      {
        "portfolio": "CanSlim",
        "weeks": 8,
        "weeks_included": [{"week_start": "2026-05-11"}, ...],  // oldest→newest
        "traps": [
          {
            "tag": "FOMO Entry",
            "total_count": 6,
            "series": [{"week_start": "2026-05-11", "count": 2}, ...],
            "trades": [
              {"ticker": "NVDA", "week_start": "2026-05-11",
               "retro_id": 42, "grade": "C (Sloppy)",
               "notes": "chased after breakout"},
              ...
            ]
          },
          ...
        ]
      }

    Sort: traps by total_count desc, then tag asc for stable tie-break.
    Each trap's series contains exactly `weeks` entries — weeks with no
    fires get {count: 0} so the heat map has a consistent column count.
    """
    try:
        w = max(1, min(int(weeks or 8), 52))
    except (TypeError, ValueError):
        w = 8

    from datetime import date, timedelta
    today = date.today()
    # Anchor at this week's Monday, then walk back w-1 Mondays.
    days_since_mon = (today.weekday())  # Mon=0
    this_monday = today - timedelta(days=days_since_mon)
    week_starts = [this_monday - timedelta(weeks=i) for i in range(w - 1, -1, -1)]
    earliest_monday = week_starts[0]

    try:
        with db.get_db_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.id AS retro_id, r.week_start,
                       ptg.ticker, ptg.grade, ptg.notes,
                       jsonb_array_elements_text(ptg.behaviors) AS tag
                  FROM weekly_retros r
                  JOIN portfolios p ON p.id = r.portfolio_id
                  JOIN weekly_retro_ticker_grades ptg ON ptg.weekly_retro_id = r.id
                 WHERE p.name = %s
                   AND r.deleted_at IS NULL
                   AND r.week_start >= %s
                """,
                (portfolio, earliest_monday),
            )
            rows = cur.fetchall()
    except Exception as e:
        print(f"[mindset_traps] query failed: {e}")
        return {"error": str(e), "portfolio": portfolio, "weeks": w,
                "weeks_included": [], "traps": []}

    # Aggregate per tag. Trip counts + per-week counts + drill-through
    # trades list. Weeks with no fires get count=0 via the pre-seeded
    # dict so the heat map has consistent columns.
    per_tag: dict = {}
    for row in rows:
        # dict-style row via RealDictCursor OR positional — support both.
        if isinstance(row, dict):
            retro_id = row["retro_id"]
            wk = row["week_start"]
            ticker = row["ticker"]
            grade = row.get("grade") or ""
            notes = row.get("notes") or ""
            tag = row["tag"]
        else:
            retro_id, wk, ticker, grade, notes, tag = row
            grade = grade or ""
            notes = notes or ""
        tag_s = str(tag).strip()
        if not tag_s:
            continue
        entry = per_tag.setdefault(tag_s, {
            "tag": tag_s,
            "total_count": 0,
            "series_map": {ws.isoformat(): 0 for ws in week_starts},
            "trades": [],
        })
        entry["total_count"] += 1
        wk_iso = wk.isoformat() if hasattr(wk, "isoformat") else str(wk)[:10]
        if wk_iso in entry["series_map"]:
            entry["series_map"][wk_iso] += 1
        entry["trades"].append({
            "ticker": ticker,
            "week_start": wk_iso,
            "retro_id": retro_id,
            "grade": grade,
            "notes": notes,
        })

    traps = []
    for tag, e in per_tag.items():
        traps.append({
            "tag": tag,
            "total_count": e["total_count"],
            "series": [
                {"week_start": ws.isoformat(), "count": e["series_map"][ws.isoformat()]}
                for ws in week_starts
            ],
            "trades": sorted(e["trades"], key=lambda t: (t["week_start"], t["ticker"])),
        })
    traps.sort(key=lambda x: (-x["total_count"], x["tag"]))

    return {
        "portfolio": portfolio,
        "weeks": w,
        "weeks_included": [{"week_start": ws.isoformat()} for ws in week_starts],
        "traps": traps,
    }


@app.get("/api/portfolio/heat-preview")
def portfolio_heat_preview(portfolio: str = "CanSlim"):
    """Live Portfolio Heat snapshot for the Daily Routine tile.

    Recomputes _compute_portfolio_heat against the latest saved end_nlv so
    the Daily Routine card can preview "what my risk looks like right now"
    before the user finalises today's save. If the user hasn't saved a
    journal row yet, returns 0 with nlv_used=0. If yfinance is offline,
    _compute_portfolio_heat returns 0 (same silent-fail contract as the
    stamp path).
    """
    try:
        df = db.load_journal(portfolio)
        if df.empty:
            return {"heat": 0.0, "nlv_used": 0.0, "portfolio": portfolio}
        df = _normalize_journal(df)
        df["day"] = pd.to_datetime(df["day"], errors="coerce")
        df = df.sort_values("day", ascending=False)
        equity = float(df.iloc[0].get("end_nlv", 0) or 0)
        if equity <= 0:
            return {"heat": 0.0, "nlv_used": 0.0, "portfolio": portfolio}
        heat = _compute_portfolio_heat(portfolio, "", equity)
        return {"heat": heat, "nlv_used": equity, "portfolio": portfolio}
    except Exception as e:
        print(f"[portfolio_heat_preview] handler failed: {e}")
        return {"heat": 0.0, "nlv_used": 0.0, "portfolio": portfolio, "error": str(e)}


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
    # trend_count is nullable SMALLINT (Migration 043) — same NaN-vs-0
    # discipline as mct_display_day_num above. 0 is a valid Step-4 arm
    # bar; NaN means pre-first-Step-4 or market-closed day journaled for
    # NLV only. Frontend renders — for NaN, +N/−N for signed values.
    if "trend_count" in df.columns:
        df["trend_count"] = pd.to_numeric(df["trend_count"], errors="coerce")

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

    cols = ["id", "day", "end_nlv", "beg_nlv", "daily_pct_change", "daily_dollar_change",
            "daily_return", "pct_invested", "portfolio_ltd",
            "spy_ltd", "ndx_ltd", "spy_daily_pct", "ndx_daily_pct",
            "spy", "nasdaq", "portfolio_heat", "score", "cash_change",
            "market_window", "market_cycle", "mct_display_day_num",
            "trend_count",
            "market_notes", "market_action",
            "spy_atr", "nasdaq_atr",
            "highlights", "lowlights", "mistakes", "top_lesson",
            # Phase 7 — id surfaces the daily journal row's PK so TagPicker,
            # NotesRail, and SnapshotGallery on the daily report have an
            # entity_id to bind to. daily_thoughts is the rich-text body
            # for the new Daily Thoughts editor (migration 031).
            "daily_thoughts"]
    available_cols = [c for c in cols if c in df.columns]
    return _df_to_records(df[available_cols])


# ─────────────────────────────────────────────────────────────────────
# Realized Equity Curve — closed-positions-only cumulative P&L by date.
#
# Source: lot_closures.realized_pl (multiplier-scaled at write time —
# do NOT re-scale) summed by lot_closures.closed_at::date. Sister to
# /api/journal/history but movement is event-driven (only on closes)
# instead of mark-to-market daily.
#
# Caveat: a small set of legacy "deferred" trades may lack lot_closures
# rows even though trades_summary.realized_pl is correct (per migration
# 017/033). v1 accepts this — the curve is the durable closure record.
# ─────────────────────────────────────────────────────────────────────

def _realized_curve_baseline_nlv(portfolio: str, start_date: pd.Timestamp) -> tuple[float, str]:
    """Resolve the baseline NLV for the realized curve at `start_date`.

    Preference order:
      1. trading_journal row ON start_date     → beg_nlv (open-of-day NLV)
      2. latest trading_journal row BEFORE it  → end_nlv (close-of-prior-day)
      3. portfolios.starting_capital
      4. (0.0, "none") — caller then renders 0% for every point
    """
    try:
        journal = db.load_journal(portfolio)
    except Exception:
        journal = pd.DataFrame()

    if not journal.empty:
        # db.load_journal returns capitalized column names ("Day", "Beg_NLV",
        # "End_NLV"); _normalize_journal lowercases them to match what the
        # rest of this module reads. /api/journal/history applies the same
        # normalization at line 265 — we mirror that here so a deployed
        # portfolio with real journal rows doesn't KeyError on "day".
        j = _normalize_journal(journal.copy())
        if "day" in j.columns:
            j["day"] = pd.to_datetime(j["day"], errors="coerce")
            j = j.dropna(subset=["day"])
            on_start = j[j["day"].dt.normalize() == start_date]
            if not on_start.empty:
                beg = pd.to_numeric(on_start.iloc[0].get("beg_nlv"), errors="coerce")
                if pd.notna(beg) and beg > 0:
                    return float(beg), "journal"
            before_start = j[j["day"].dt.normalize() < start_date].sort_values("day")
            if not before_start.empty:
                end = pd.to_numeric(before_start.iloc[-1].get("end_nlv"), errors="coerce")
                if pd.notna(end) and end > 0:
                    return float(end), "journal"

    # Fallback: portfolios.starting_capital. Read directly — there's no
    # db.get_portfolio_by_name() helper and list_portfolios() is RLS-scoped
    # to the current user, which we may not have a session for in unit
    # tests. A small direct SELECT keeps the path testable via monkeypatch
    # of db.get_db_connection.
    try:
        with db.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT starting_capital FROM portfolios WHERE name = %s",
                    (portfolio,),
                )
                row = cur.fetchone()
                if row is not None and row[0] is not None:
                    sc = float(row[0])
                    if sc > 0:
                        return sc, "starting_capital"
    except Exception as e:
        print(f"[realized_curve] starting_capital lookup failed: {e}")

    return 0.0, "none"


def _compute_realized_curve(
    closures: pd.DataFrame,
    start_nlv: float,
    start_date: pd.Timestamp,
) -> tuple[list[dict[str, Any]], float, int]:
    """Group lot_closures.realized_pl by closed_at::date, prefix-sum,
    return (series, total_realized_pl, closed_count).

    One series point per date that had at least one closure — gaps are
    intentional; the frontend stair-steps / forward-fills between them
    onto its date axis. Pre-start closures are dropped entirely (the
    curve begins at 0 on start_date; no merged "carry" point).
    """
    if closures is None or closures.empty:
        return [], 0.0, 0
    df = closures.copy()
    df["closed_at"] = pd.to_datetime(df["closed_at"], errors="coerce")
    df = df.dropna(subset=["closed_at"])
    df = df[df["closed_at"].dt.normalize() >= start_date]
    if df.empty:
        return [], 0.0, 0
    df["realized_pl"] = pd.to_numeric(df["realized_pl"], errors="coerce").fillna(0)
    df["day"] = df["closed_at"].dt.strftime("%Y-%m-%d")
    by_day = df.groupby("day", as_index=False)["realized_pl"].sum().sort_values("day")

    series: list[dict[str, Any]] = []
    cum = 0.0
    for _, r in by_day.iterrows():
        cum += float(r["realized_pl"])
        pct = (cum / start_nlv * 100.0) if start_nlv > 0 else 0.0
        series.append({
            "day": str(r["day"]),
            "cum_realized_pl": round(cum, 2),
            "cum_realized_pct": round(pct, 2),
        })
    return series, round(cum, 2), int(len(df))


@app.get("/api/realized/curve")
@limiter.limit("60/minute")
def realized_curve(
    request: Request,
    portfolio: str = "CanSlim",
    start: str = "2026-01-01",
):
    """Cumulative realized P&L by close date for a portfolio.

    Movement is event-driven: the series has one point per date that had
    at least one SELL closure (lot_closures row) on or after `start`. The
    frontend stair-steps between points. Multiplier is already applied at
    write time (trade_calc.py:189) — options notional is correct.

    Query:
      portfolio (str): portfolio name (default "CanSlim").
      start     (date, default 2026-01-01): inclusive lower bound on
                closure dates. The curve begins at 0 P&L on this date;
                pre-start closures are not folded into a carry.

    Response:
      {
        "series": [ { "day": "YYYY-MM-DD",
                      "cum_realized_pl": float,
                      "cum_realized_pct": float }, ... ],
        "summary": {
          "total_realized_pl": float,   # final cum value (0 when empty)
          "realized_pct":      float,   # total / start_nlv × 100
          "closed_count":      int,     # # of lot_closures rows in range
          "start_nlv":         float,   # baseline used for the % anchor
          "start_date":        "YYYY-MM-DD",
          "baseline_source":   "journal" | "starting_capital" | "none"
        }
      }
    """
    try:
        start_date = pd.Timestamp(start).normalize()
    except Exception:
        return {"error": "start must be a date in YYYY-MM-DD form"}

    start_nlv, baseline_source = _realized_curve_baseline_nlv(portfolio, start_date)

    try:
        closures = db.load_lot_closures(portfolio)
    except Exception as e:
        print(f"[realized_curve] load_lot_closures failed for portfolio={portfolio}: {e}")
        return {"error": str(e)}

    series, total, count = _compute_realized_curve(closures, start_nlv, start_date)
    pct = (total / start_nlv * 100.0) if start_nlv > 0 else 0.0

    return {
        "series": series,
        "summary": {
            "total_realized_pl": round(total, 2),
            "realized_pct":      round(pct, 2),
            "closed_count":      count,
            "start_nlv":         round(start_nlv, 2),
            "start_date":        start_date.strftime("%Y-%m-%d"),
            "baseline_source":   baseline_source,
        },
    }


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
            elif state_name in ("UPTREND", "UPTREND UNDER PRESSURE", "RALLY MODE") and cycle_day > 0:
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
        print(f"[journal_mct_state_by_date_range] handler failed: {e}")
        return {"error": str(e), "states": []}


def _compute_ticker_atr_pct(ticker: str, as_of_date: str = "") -> float:
    """Compute 21-period ATR% = SMA(TR, 21) / SMA(Low, 21) * 100.

    Returns 0.0 on any failure mode (empty/sparse history, yfinance
    exception, zero-volatility series). Callers (the journal snapshot
    path in particular) use the return value to weight position
    contribution to portfolio_heat, so a silent 0 here effectively
    drops the ticker from the snapshot. Log the exception path so an
    operator can grep production logs to identify silently-dropped
    tickers after the fact — behavior unchanged, observability only.
    """
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
    except Exception as e:
        print(f"[_compute_ticker_atr_pct] {ticker} (as_of={as_of_date or 'live'}) "
              f"silently returned 0.0 due to: {type(e).__name__}: {e}")
        return 0.0


def _compute_portfolio_heat(portfolio: str, as_of_date: str, equity: float) -> float:
    """Portfolio heat = sum(weight% * atr%/100) for stock positions only.

    Weight uses shares * last_close (market value) instead of total_cost
    (cost basis) so the number matches the Portfolio Heat page and Active
    Campaign's POS SIZE. Cost basis grossly understated heat on winners
    (e.g., MU at +137% showed ~half its true weight). Falls back to cost
    basis when yfinance returns nothing so a partial outage still stamps
    a reasonable number.

    Option positions are excluded — yfinance has no ATR data for OCC option
    tickers, so they always contributed 0 heat anyway. Excluding them keeps
    the metric a clean "volatility check on equity exposure" rather than a
    diluted average across instruments with incommensurate risk profiles.

    yfinance fetch is inlined (single history call per ticker yields both
    atr_pct and last_close). Kept _compute_ticker_atr_pct untouched — its
    other callers (SPY/^IXIC) only need ATR.
    """
    import yfinance as yf
    try:
        summary_df = db.load_summary(portfolio)
        if summary_df.empty or equity <= 0:
            return 0.0
        summary_df = _normalize_trades(summary_df)
        status_col = "status" if "status" in summary_df.columns else "Status"
        open_df = summary_df[summary_df[status_col].str.upper() == "OPEN"]
        if open_df.empty:
            return 0.0
        # Filter to stocks. Prefer the instrument_type column (Migration 016).
        # Fall back to the option-ticker regex pattern matching the canonical
        # isOptionRow helper on the frontend (perf-heatmap.tsx:73-75) so any
        # pre-016 legacy row without instrument_type set is still filtered.
        if "instrument_type" in open_df.columns:
            open_df = open_df[open_df["instrument_type"].astype(str).str.upper() == "STOCK"]
        else:
            option_pattern = r"^\S+\s+\d{6}\s+\$[0-9.]+(C|P)$"
            open_df = open_df[~open_df["ticker"].astype(str).str.match(option_pattern)]
        if open_df.empty:
            return 0.0
        heat = 0.0
        for _, row in open_df.iterrows():
            ticker = str(row.get("ticker", "")).strip()
            shares = float(row.get("shares", 0) or 0)
            total_cost = float(row.get("total_cost", 0) or 0)
            if not ticker or total_cost <= 0:
                continue

            atr_pct = 0.0
            last_close = 0.0
            try:
                if as_of_date:
                    end_dt = pd.Timestamp(as_of_date) + pd.Timedelta(days=1)
                    start_dt = pd.Timestamp(as_of_date) - pd.Timedelta(days=60)
                    df = yf.Ticker(ticker).history(
                        start=start_dt.strftime("%Y-%m-%d"),
                        end=end_dt.strftime("%Y-%m-%d"),
                    )
                else:
                    df = yf.Ticker(ticker).history(period="45d")
                if not df.empty and len(df) >= 21:
                    tr = pd.concat([
                        df["High"] - df["Low"],
                        (df["High"] - df["Close"].shift(1)).abs(),
                        (df["Low"] - df["Close"].shift(1)).abs(),
                    ], axis=1).max(axis=1)
                    sma_tr = float(tr.tail(21).mean())
                    sma_low = float(df["Low"].tail(21).mean())
                    if sma_low > 0:
                        atr_pct = round((sma_tr / sma_low) * 100, 4)
                    last_close = float(df["Close"].iloc[-1])
            except Exception as e:
                print(f"[portfolio_heat] {ticker} yfinance fetch failed: "
                      f"{type(e).__name__}: {e}")

            # Market value with cost-basis fallback so a partial yfinance
            # outage still stamps a reasonable weight.
            market_value = shares * last_close if (shares > 0 and last_close > 0) else total_cost
            weight_pct = (market_value / equity) * 100
            heat += weight_pct * (atr_pct / 100)
        return round(heat, 4)
    except Exception as e:
        print(f"[portfolio_heat] heat aggregation failed: {e}")
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
    except Exception as e:
        print(f"[mct_cycle] rally_prefix lookup failed: {e}")
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
        elif state_name in ("UPTREND", "UPTREND UNDER PRESSURE", "RALLY MODE") and cycle_day > 0:
            display_day_num = cycle_day
        else:
            display_day_num = None

        return (state_name, display_day_num)
    except Exception as e:
        print(f"[mct_state] state + day_num resolve failed: {e}")
        return ("", None)


def _compute_trend_count(as_of_date: str = "") -> int | None:
    """Compute signed Trend Count for a given date, snapshot-style.

    Wraps run_engine → to_rally_prefix_response so the save-time stamper reads
    the SAME value the M Factor banner shows — no reimplementation of the
    Tier-1 leg math here. If the adapter's payload has trend_count = None
    (pre-first-Step-4 in the replay, or no bar for as_of), we return None
    and the caller persists NULL.

    Mirrors the strict-bar-match discipline of _compute_mct_state_with_day_num:
    same yfinance top-up + same "empty on failure" contract, so a save run
    before market_data ingests today's bar stamps NULL rather than yesterday's
    value.
    """
    try:
        from datetime import datetime as _dt
        from api.mct_endpoint_adapter import run_engine, to_rally_prefix_response

        as_of = None
        if as_of_date:
            try:
                as_of = _dt.strptime(as_of_date.strip()[:10], "%Y-%m-%d").date()
            except (ValueError, AttributeError):
                as_of = None

        # Same market_data top-up as the MCT stamper — a Daily Routine save
        # right after close but before the ingest cron would otherwise land
        # in the "no bar for as_of" branch and stamp NULL forever.
        try:
            from api.market_data_updater import update_if_needed
            update_if_needed("^IXIC")
        except Exception:
            pass

        result = run_engine("^IXIC", as_of=as_of)
        if result.bars.empty:
            return None

        # Strict bar match — if the engine has nothing for as_of, do not
        # fall through to the prior trading day. Same rationale as the MCT
        # stamper above: NULL beats "yesterday's number stamped on today."
        if as_of is not None:
            trade_dates = pd.to_datetime(result.bars["trade_date"]).dt.date
            if not (trade_dates == as_of).any():
                return None

        payload = to_rally_prefix_response(result)
        raw = payload.get("trend_count")
        return int(raw) if raw is not None else None
    except Exception as e:
        print(f"[trend_count] compute failed: {e}")
        return None


def _heal_recent_mct_stamps(portfolio: str, df: pd.DataFrame, lookback_days: int = 14) -> None:
    """Backfill NULL mct_display_day_num / market_cycle / trend_count on
    recent journal rows.

    The save-time stampers (_compute_mct_state_with_day_num and
    _compute_trend_count) intentionally persist NULL when the engine has
    no bar for the requested date — the common cause is "user logged
    today's journal before market_data ingested today's bar." When the
    bar lands later, those rows would stay NULL forever without an
    explicit re-save. This helper runs once per /api/journal/history
    call: it locates NULL rows in the last `lookback_days`, replays the
    engine once to get every cached bar's state, and stamps any row
    whose date now has a bar. In-memory df is patched so the response
    reflects the fresh values without a second DB read.

    Bounded lookback (default 14 days) keeps this cheap on every page
    load — older NULLs go through scripts/backfill_mct_state.py and
    scripts/backfill_trend_count.py for a full historical sweep.
    """
    if df.empty or "mct_display_day_num" not in df.columns:
        return

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
    trend_col = df.get("trend_count", pd.Series(dtype=object))
    needs_heal = df[
        (df["day"] >= cutoff)
        & (
            df["mct_display_day_num"].isna()
            | (df.get("market_cycle", pd.Series(dtype=object)).fillna("").astype(str) == "")
            | trend_col.isna()
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
    except Exception as e:
        print(f"[mct_heal] engine setup failed: {e}")
        return

    for _, row in needs_heal.iterrows():
        day_value = row["day"]
        if pd.isna(day_value):
            continue
        as_of = day_value.date() if hasattr(day_value, "date") else day_value
        if as_of not in bar_index.index:
            continue  # engine still doesn't have this bar — skip
        day_str = as_of.strftime("%Y-%m-%d")

        # MCT state heal — same behavior as before. Skips the update if
        # this row already has a state (only trend_count needs healing).
        mct_null = (
            pd.isna(row.get("mct_display_day_num"))
            or not str(row.get("market_cycle") or "").strip()
        )
        if mct_null:
            state, day_num = _compute_mct_state_with_day_num(day_str)
            if state:
                try:
                    db.update_journal_mct_state(portfolio, day_str, state, day_num)
                    df.loc[df["day"] == day_value, "market_cycle"] = state
                    df.loc[df["day"] == day_value, "mct_display_day_num"] = day_num
                except Exception as e:
                    print(f"[mct_heal] update_journal_mct_state failed for {day_str}: {e}")

        # Trend Count heal — mirror of the MCT stamp path. Same engine
        # replay was already done above so the bar exists; recompute the
        # signed count and persist via the targeted helper so unrelated
        # columns (NLV, notes) aren't touched.
        trend_null = pd.isna(row.get("trend_count"))
        if trend_null:
            trend_count = _compute_trend_count(day_str)
            if trend_count is not None:
                try:
                    db.update_journal_trend_state(portfolio, day_str, trend_count)
                    df.loc[df["day"] == day_value, "trend_count"] = trend_count
                except Exception as e:
                    print(f"[trend_heal] update_journal_trend_state failed for {day_str}: {e}")


@app.post("/api/journal/edit")
def journal_edit(entry: dict):
    """Update or insert a journal entry. Preserves existing values for fields not sent."""
    try:
        portfolio = entry.pop("portfolio", "CanSlim")
        day = entry.get("day")

        # Load existing entry to preserve fields not being edited
        existing = {}
        existing_row_present = False
        df = db.load_journal(portfolio)
        if not df.empty:
            df = _normalize_journal(df)
            df["day"] = pd.to_datetime(df["day"], errors="coerce").dt.strftime("%Y-%m-%d")
            match = df[df["day"] == str(day).strip()[:10]]
            if not match.empty:
                existing_row_present = True
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
                    # Phase 7 — rich-text body for the Daily Thoughts editor
                    # (migration 031). Preserved on every edit so a partial
                    # PUT from another surface (e.g., Daily Routine) doesn't
                    # wipe the page's prose.
                    "daily_thoughts": str(row.get("daily_thoughts", "") or ""),
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
            "trend_count": (
                int(entry["trend_count"])
                if entry.get("trend_count") not in (None, "")
                else existing.get("trend_count")
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
            "daily_thoughts": _s("daily_thoughts", "daily_thoughts"),
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

        # Auto-compute missing market/risk metrics — SNAPSHOT-AT-WRITE contract.
        # market_window is deprecated as of MCT V11 Phase 3a — no longer auto-filled.
        #
        # These four branches (market_cycle/mct_display_day_num, spy_atr,
        # nasdaq_atr, portfolio_heat) only fire on FRESH INSERTs, never on
        # edits of pre-existing rows. The as_of in every compute is `day_str`,
        # which equals "today" by construction when Daily Routine saves the
        # current day — so the snapshot is correct at write time.
        #
        # On edit of an existing row whose stored value happens to be 0 or
        # NULL, we deliberately PRESERVE that value rather than recomputing
        # against "today's" data. Recomputing would silently rewrite the
        # snapshot using inputs (current open positions, current ATR window)
        # that don't match the row's historical date, producing wrong
        # numbers. User-initiated gap-fill goes through
        # /api/journal/backfill-metrics (which is explicit and date-aware);
        # user-initiated override goes through the Manage Logs edit form
        # (which always sends an explicit value through the payload).
        day_str = str(day).strip()[:10]
        if not existing_row_present:
            # Single engine replay yields both the cycle state and the
            # display_day_num the badge appends ("POWERTREND D3" etc.).
            # Snapshot both into the row so the Daily Journal page can
            # render the badge without re-running the engine on every visit.
            if not journal_entry["market_cycle"] or journal_entry["mct_display_day_num"] is None:
                mct_state, mct_day_num = _compute_mct_state_with_day_num(day_str)
                if not journal_entry["market_cycle"]:
                    journal_entry["market_cycle"] = mct_state
                if journal_entry["mct_display_day_num"] is None:
                    journal_entry["mct_display_day_num"] = mct_day_num
            # Trend Count — separate engine payload read, same snapshot
            # discipline as the MCT stamp above. Explicit `is None` check
            # so that 0 (a Step-4 arm bar's legit signed count) is not
            # mistakenly re-stamped. Fresh insert only; edits preserve
            # whatever's already in the row (backfill handled offline).
            if journal_entry.get("trend_count") is None:
                journal_entry["trend_count"] = _compute_trend_count(day_str)
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


# ─────────────────────────────────────────────────────────────────────────────
# Batch journal save — Daily Routine multi-portfolio entrypoint.
#
# Atomic write of N journal rows in a single PG transaction. Designed for the
# multi-portfolio Daily Routine redesign: one save, N rows, all-or-nothing.
# Coexists with /api/journal/edit (which remains the single-row entry path
# used by Manage Logs and the Daily Report Card).
#
# Shared-vs-per-portfolio field split mirrors the investigation report's
# verified categorization (6 shared + 9 per-portfolio + 1 per-portfolio
# ambiguous-but-data-derived = market_action / "actions").
#
# Snapshot semantics on auto-compute fields match the recent leak-fix in
# journal_edit: market_cycle, mct_display_day_num, spy_atr, nasdaq_atr,
# portfolio_heat fire ONLY for rows that are net-new on this save. For rows
# being overwritten via force_overwrite=true, those columns are preserved
# from the existing row (not recomputed against today's data).
# ─────────────────────────────────────────────────────────────────────────────


_BATCH_EDIT_INSERT_SQL = """
    INSERT INTO trading_journal (
        user_id, portfolio_id, day, status, market_window, market_cycle,
        mct_display_day_num, trend_count, above_21ema,
        cash_change, beg_nlv, end_nlv, daily_dollar_change,
        daily_pct_change, pct_invested, spy, nasdaq,
        market_notes, market_action, portfolio_heat,
        spy_atr, nasdaq_atr, score,
        highlights, lowlights, mistakes, top_lesson,
        nlv_source, holdings_source, daily_thoughts
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    RETURNING id
"""

_BATCH_EDIT_UPDATE_SQL = """
    UPDATE trading_journal
       SET status = %s, market_window = %s, market_cycle = %s,
           mct_display_day_num = %s, trend_count = %s, above_21ema = %s,
           cash_change = %s, beg_nlv = %s, end_nlv = %s,
           daily_dollar_change = %s, daily_pct_change = %s,
           pct_invested = %s, spy = %s, nasdaq = %s,
           market_notes = %s, market_action = %s, portfolio_heat = %s,
           spy_atr = %s, nasdaq_atr = %s, score = %s,
           highlights = %s, lowlights = %s, mistakes = %s, top_lesson = %s,
           nlv_source = %s, holdings_source = %s, daily_thoughts = %s
     WHERE id = %s
     RETURNING id
"""


def _coerce_nlv_source(v) -> str:
    """Constrain to {manual, ibkr_auto, ibkr_override}; default 'manual'."""
    s = str(v or "manual").strip()
    return s if s in ("manual", "ibkr_auto", "ibkr_override") else "manual"


@app.post("/api/journal/batch-edit")
def journal_batch_edit(body: dict = Body(...)):
    """Save N journal rows atomically in a single PG transaction.

    Request shape:
      {
        "day": "YYYY-MM-DD",
        "shared": { spy, nasdaq, market_notes, score, highlights, mistakes,
                    nlv_source?, holdings_source? },
        "portfolios": [
          { "portfolio": <name>, "end_nlv": <num>, "total_holdings": <num>,
            "cash_change": <num>, "actions": <str>,
            "pct_invested": <num>, "daily_dollar_change": <num>,
            "daily_pct_change": <num> },
          ...
        ],
        "force_overwrite": false
      }

    Responses:
      200 — {status: "ok", rows_written: N, portfolios: [names]}
      404 — unknown portfolio name
      409 — conflict with force_overwrite=false (conflicting_portfolios listed)
      422 — validation errors (errors list with portfolio/field/message)
      500 — any unhandled exception; transaction is rolled back
    """
    # 1. Top-level validation.
    day = body.get("day")
    if not day:
        return JSONResponse(status_code=422, content={
            "status": "invalid",
            "errors": [{"portfolio": None, "field": "day", "message": "Required"}],
        })
    day_str = str(day).strip()[:10]

    portfolios = body.get("portfolios") or []
    if not isinstance(portfolios, list) or not portfolios:
        return JSONResponse(status_code=422, content={
            "status": "invalid",
            "errors": [{"portfolio": None, "field": "portfolios",
                        "message": "Required (must be non-empty array)"}],
        })

    shared = body.get("shared") or {}
    force_overwrite = bool(body.get("force_overwrite", False))

    # 2. Per-portfolio field validation.
    errors: list[dict] = []
    for pf in portfolios:
        name = pf.get("portfolio")
        if not name:
            errors.append({"portfolio": None, "field": "portfolio",
                           "message": "Required"})
            continue
        if pf.get("end_nlv") is None:
            errors.append({"portfolio": name, "field": "end_nlv",
                           "message": "Required"})
        if pf.get("total_holdings") is None:
            errors.append({"portfolio": name, "field": "total_holdings",
                           "message": "Required"})
    if errors:
        return JSONResponse(status_code=422,
                            content={"status": "invalid", "errors": errors})

    # 3. Open the transaction. Everything below this point either commits
    # together or rolls back together. db.get_db_connection() sets app.user_id
    # + ROLE app_runtime per the auth context, so RLS is in effect.
    try:
        with db.get_db_connection() as conn:
            with conn.cursor() as cur:
                # 3a. Resolve every portfolio_id by name. Fail-fast 404 on
                # any unknown portfolio (RLS already scopes by user).
                portfolio_meta: dict[str, dict] = {}
                for pf in portfolios:
                    name = pf["portfolio"]
                    cur.execute(
                        "SELECT id, user_id FROM portfolios WHERE name = %s",
                        (name,),
                    )
                    row = cur.fetchone()
                    if not row:
                        return JSONResponse(status_code=404, content={
                            "status": "not_found",
                            "detail": f"Portfolio '{name}' not found",
                        })
                    portfolio_meta[name] = {
                        "id": row[0], "user_id": str(row[1]),
                    }

                # 3b. Pre-flight conflict check (force_overwrite=false only).
                # Returns the full list of conflicting portfolios so the UI can
                # tell the user exactly which need overwrite confirmation.
                if not force_overwrite:
                    conflicts = []
                    for pf in portfolios:
                        pid = portfolio_meta[pf["portfolio"]]["id"]
                        cur.execute(
                            "SELECT 1 FROM trading_journal "
                            "WHERE portfolio_id = %s AND day = %s "
                            "  AND deleted_at IS NULL",
                            (pid, day_str),
                        )
                        if cur.fetchone() is not None:
                            conflicts.append(pf["portfolio"])
                    if conflicts:
                        return JSONResponse(status_code=409, content={
                            "status": "exists",
                            "detail": ("Rows already exist for some portfolios "
                                       "on this date. Check Force Overwrite to "
                                       "replace."),
                            "conflicting_portfolios": conflicts,
                        })

                # 3c. Shared field resolution (computed once, reused for every
                # row's INSERT/UPDATE).
                shared_spy = float(shared.get("spy") or 0)
                shared_ndx = float(shared.get("nasdaq") or 0)
                shared_market_notes = str(shared.get("market_notes") or "")
                shared_score = int(float(shared.get("score") or 0))
                shared_highlights = str(shared.get("highlights") or "")
                shared_mistakes = str(shared.get("mistakes") or "")
                shared_nlv_source = _coerce_nlv_source(
                    shared.get("nlv_source", "manual"))
                shared_holdings_source = _coerce_nlv_source(
                    shared.get("holdings_source", "manual"))

                # 3d. Per-row save loop. Each row gets:
                #   - Its own portfolio_id + user_id
                #   - beg_nlv from THIS portfolio's prior end_nlv
                #   - Snapshot fields (market_cycle, ATRs, heat) auto-computed
                #     ONLY when no existing row (snapshot semantics from the
                #     /api/journal/edit leak fix; applied per-row here).
                written: list[str] = []
                for pf in portfolios:
                    name = pf["portfolio"]
                    meta = portfolio_meta[name]
                    pid, uid = meta["id"], meta["user_id"]

                    # Resolve beg_nlv for THIS portfolio = prior day's end_nlv.
                    cur.execute(
                        "SELECT end_nlv FROM trading_journal "
                        "WHERE portfolio_id = %s AND day < %s "
                        "  AND deleted_at IS NULL "
                        "ORDER BY day DESC LIMIT 1",
                        (pid, day_str),
                    )
                    prev_row = cur.fetchone()
                    beg_nlv = float(prev_row[0]) if prev_row else 0.0

                    # Existence check for snapshot gating + UPDATE-vs-INSERT
                    # branch. Reads the snapshot fields so preserved values
                    # don't get recomputed against today's data.
                    cur.execute(
                        "SELECT id, portfolio_heat, spy_atr, nasdaq_atr, "
                        "       market_cycle, mct_display_day_num, trend_count "
                        "  FROM trading_journal "
                        " WHERE portfolio_id = %s AND day = %s "
                        "   AND deleted_at IS NULL",
                        (pid, day_str),
                    )
                    existing_row = cur.fetchone()
                    existing_row_present = existing_row is not None

                    end_nlv = float(pf["end_nlv"])
                    total_holdings = float(pf["total_holdings"])
                    cash_change = float(pf.get("cash_change") or 0)
                    actions = str(pf.get("actions") or "")
                    pct_invested = float(pf.get("pct_invested") or 0)
                    daily_dollar_change = float(
                        pf.get("daily_dollar_change") or 0)
                    daily_pct_change = float(pf.get("daily_pct_change") or 0)

                    # Snapshot fields — preserve on overwrite, compute on insert.
                    if existing_row_present:
                        portfolio_heat = float(existing_row[1] or 0)
                        spy_atr = float(existing_row[2] or 0)
                        nasdaq_atr = float(existing_row[3] or 0)
                        market_cycle = existing_row[4] or ""
                        mct_display_day_num = existing_row[5]
                        trend_count = existing_row[6]
                    else:
                        market_cycle, mct_display_day_num = (
                            _compute_mct_state_with_day_num(day_str))
                        market_cycle = market_cycle or ""
                        # trend_count uses its own engine payload read (single
                        # int) — mirrors the MCT snapshot discipline right
                        # above it. NULL on failure / no-bar (matches backfill).
                        trend_count = _compute_trend_count(day_str)
                        spy_atr = _compute_ticker_atr_pct("SPY", day_str)
                        nasdaq_atr = _compute_ticker_atr_pct(
                            "^IXIC", day_str)
                        portfolio_heat = _compute_portfolio_heat(
                            name, day_str, end_nlv)

                    # Defaults for fields not surfaced by the batch shape
                    # but still part of the schema. Match save_journal_entry's
                    # defaults so the row looks identical to a Daily-Routine-
                    # saved single-portfolio row.
                    status_val = "U"
                    market_window = "Open"
                    above_21ema = 0
                    lowlights = ""
                    top_lesson = ""
                    daily_thoughts = ""

                    if existing_row_present:
                        cur.execute(_BATCH_EDIT_UPDATE_SQL, (
                            status_val, market_window, market_cycle,
                            mct_display_day_num, trend_count, above_21ema,
                            cash_change, beg_nlv, end_nlv,
                            daily_dollar_change, daily_pct_change,
                            pct_invested, shared_spy, shared_ndx,
                            shared_market_notes, actions, portfolio_heat,
                            spy_atr, nasdaq_atr, shared_score,
                            shared_highlights, lowlights, shared_mistakes,
                            top_lesson, shared_nlv_source,
                            shared_holdings_source, daily_thoughts,
                            existing_row[0],
                        ))
                    else:
                        cur.execute(_BATCH_EDIT_INSERT_SQL, (
                            uid, pid, day_str, status_val, market_window,
                            market_cycle, mct_display_day_num, trend_count,
                            above_21ema,
                            cash_change, beg_nlv, end_nlv,
                            daily_dollar_change, daily_pct_change,
                            pct_invested, shared_spy, shared_ndx,
                            shared_market_notes, actions, portfolio_heat,
                            spy_atr, nasdaq_atr, shared_score,
                            shared_highlights, lowlights, shared_mistakes,
                            top_lesson, shared_nlv_source,
                            shared_holdings_source, daily_thoughts,
                        ))

                    written.append(name)

                conn.commit()
                # Invalidate the load_journal memoize cache so subsequent
                # reads see the freshly-written rows.
                db.load_journal.clear()
                return {
                    "status": "ok",
                    "rows_written": len(written),
                    "portfolios": written,
                }
    except Exception as e:
        # Any exception aborts the with-block; psycopg2's context manager
        # rolls back automatically. Surface a 500 with the detail so the
        # UI can show the user a useful error.
        return JSONResponse(status_code=500, content={
            "status": "error", "detail": str(e),
        })


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


# ============================================================
# WEEKLY RETROS (Migration 025 — Phase 0)
# ============================================================
# Move weekly retros out of localStorage into Postgres so subsequent phases
# (tags, snapshots, cross-entity search) have a real entity to attach to.
# Endpoints follow the project's body: dict + RLS + slowapi conventions.
# Business errors return HTTP 200 with {"error": "..."} — auth/rate-limit
# remain HTTP 401/429 as middleware enforces.

def _parse_week_start(raw) -> date:
    """Parse a YYYY-MM-DD string into a date. Raises ValueError on bad
    input — caller surfaces as {"error": "bad week_start"}."""
    if not raw:
        raise ValueError("week_start required")
    s = str(raw).strip()[:10]
    return datetime.strptime(s, "%Y-%m-%d").date()


@app.get("/api/weekly-retros")
def weekly_retro_get(
    portfolio: str = Query("CanSlim"),
    week_start: str = Query(...),
):
    """Return the live retro for a given portfolio + Monday, or
    {"error": "not_found"} if no row exists. The frontend treats
    not_found as "fresh blank retro" — not a UI error."""
    try:
        ws = _parse_week_start(week_start)
    except ValueError as e:
        return {"error": f"bad week_start: {e}"}
    try:
        row = db.load_weekly_retro(portfolio, ws)
        if row is None:
            return {"error": "not_found"}
        return row
    except Exception as e:
        print(f"[weekly_retro_get] handler failed: {e}")
        return {"error": str(e)}


@app.get("/api/weekly-retros/list")
def weekly_retro_list(
    portfolio: str = Query("CanSlim"),
):
    """Wrapped envelope for the Phase 6 NotesRail.

    Response: { "weeks": [...], "ytd_stats": {...} }. Synthetic empty rows
    cover Mondays without a saved retro so the rail's sparkline + grade-dot
    grid is continuous. See db.list_weekly_retros_rail for the field-level
    contract.

    Phase 6 broke the previous bare-array shape — the only consumer (the
    Review History tab) was deleted in the same commit, so this is a
    coordinated cutover with no parallel consumers to support.
    """
    try:
        return db.list_weekly_retros_rail(portfolio)
    except Exception as e:
        print(f"[weekly_retro_list] handler failed: {e}")
        return {"error": str(e)}


@app.get("/api/pinned-routes")
@limiter.limit("60/minute")
def pinned_routes_list(request: Request):
    """Return the caller's currently-pinned routes, FIFO-ordered
    (oldest pin first). RLS scopes to the calling user.

    Response: { routes: [{ route_path: str, pinned_at: iso8601 }, ...] }

    pinned_at is preserved across pin/unpin/repin cycles (toggle_pin_route
    revives rows instead of inserting fresh ones), so first-pinned-first
    ordering is stable regardless of toggle history.
    """
    try:
        rows = db.list_pinned_routes()
        # Normalize pinned_at to ISO-8601 for the JSON response. psycopg2
        # returns datetime objects; FastAPI's default encoder would str()
        # them but we serialize explicitly so the contract is pinned.
        routes = [
            {"route_path": r["route_path"], "pinned_at": r["pinned_at"].isoformat()}
            for r in rows
        ]
        return {"routes": routes}
    except Exception as e:
        print(f"[pinned_routes_list] handler failed: {e}")
        return {"error": str(e)}


@app.post("/api/pinned-routes/toggle")
@limiter.limit("60/minute")
def pinned_routes_toggle(request: Request, body: dict = Body(...)):
    """Idempotent pin/unpin for a single route_path. Mirrors the
    /api/pins/toggle contract used by Phase 6 NotesRail (pinned_entities)
    but keyed on route_path strings instead of (entity_type, entity_id).

    Body: { route_path: "/log-buy" }
    Returns: { pinned: bool } — the NEW state after the toggle.

    Idempotency contract: the server tracks pin/unpin via soft-delete on
    pinned_routes (Migration 042). Sending the same request twice
    deterministically returns to the pre-call state, no bouncing. Revival
    on re-pin preserves pinned_at so FIFO ordering is stable.

    RLS scopes to the calling user — toggling another user's pin is
    isolated (the SELECT can't see other users' rows; the INSERT path
    creates the caller's own row).
    """
    route_path = (body or {}).get("route_path")
    if not isinstance(route_path, str) or not route_path:
        return {"error": "route_path must be a non-empty string"}
    try:
        pinned = db.toggle_pin_route(route_path)
        return {"pinned": pinned}
    except ValueError as e:
        print(f"[pinned_routes_toggle] handler failed: {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"[pinned_routes_toggle] handler failed: {e}")
        return {"error": str(e)}


@app.post("/api/pins/toggle")
@limiter.limit("60/minute")
def pins_toggle(request: Request, body: dict = Body(...)):
    """Idempotent pin toggle for the Phase 6 NotesRail.

    Body: { entity_type: "weekly_retro" | "daily_journal", entity_id: int }
    Returns: { pinned: bool } — the NEW state after the toggle.

    Idempotency contract: the server tracks pin/unpin via soft-delete on
    pinned_entities (Migration 029). Sending the same request twice
    deterministically returns to the pre-call state, no bouncing.

    RLS scopes to the calling user — toggling another user's pin is a
    no-op (cur.fetchone() returns None and we INSERT a row owned by the
    caller; their attempt to "unpin" someone else's pin actually creates
    their own).
    """
    entity_type = (body or {}).get("entity_type")
    entity_id = (body or {}).get("entity_id")
    if entity_type not in ("weekly_retro", "daily_journal"):
        return {"error": "invalid_entity_type"}
    try:
        entity_id = int(entity_id)
    except (TypeError, ValueError):
        return {"error": "entity_id must be an integer"}
    try:
        pinned = db.toggle_pin(entity_type, entity_id)
        return {"pinned": pinned}
    except ValueError as e:
        print(f"[pins_toggle] handler failed: {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"[pins_toggle] handler failed: {e}")
        return {"error": str(e)}


@app.put("/api/weekly-retros")
@limiter.limit("30/minute")
def weekly_retro_upsert(request: Request, body: dict = Body(...)):
    """Upsert a retro keyed by (portfolio, week_start). Body shape:

      { portfolio, week_start, week_grade?, best_decision?,
        worst_decision?, rule_change?, rule_change_text?,
        ticker_grades?: { ticker: { grade, behaviors[], behavior, notes } },
        execution_grade?, process_grade?, pnl_grade?,
        overall_override?, reviewed_at? }

    Phase 4.6 fields:
    - execution_grade / process_grade / pnl_grade: 3-axis grading
    - overall_override: when False (default) AND all 3 axes are non-null,
      the server recomputes week_grade from the axes and ignores the
      client-supplied value (defense vs poisoned overall)
    - reviewed_at: ISO string sets the reviewed lock; null clears it.
      Changing graded fields while reviewed_at is non-null on both
      existing + incoming raises a 409.

    Returns the persisted row (with id) on success so the frontend can
    attach Phase 1 tags immediately after first save. A soft-deleted row
    for the same key is REVIVED, not duplicated, so attached tag IDs
    survive an accidental delete-then-recreate."""
    portfolio = body.get("portfolio") or "CanSlim"
    try:
        ws = _parse_week_start(body.get("week_start"))
    except ValueError as e:
        return {"error": f"bad week_start: {e}"}

    week_grade = body.get("week_grade")
    if week_grade == "":
        week_grade = None

    # Phase 4.6: normalize axis grades + override flag + reviewed_at.
    def _norm_grade(key: str) -> str | None:
        v = body.get(key)
        if v == "" or v is None:
            return None
        return str(v)

    execution_grade = _norm_grade("execution_grade")
    process_grade = _norm_grade("process_grade")
    pnl_grade = _norm_grade("pnl_grade")
    overall_override = bool(body.get("overall_override", False))
    reviewed_at = body.get("reviewed_at")
    if reviewed_at == "":
        reviewed_at = None

    tg_in = body.get("ticker_grades")
    if tg_in is not None and not isinstance(tg_in, dict):
        return {"error": "ticker_grades must be an object"}

    try:
        return db.upsert_weekly_retro(
            portfolio,
            ws,
            week_grade=week_grade,
            best_decision=str(body.get("best_decision") or ""),
            worst_decision=str(body.get("worst_decision") or ""),
            rule_change=bool(body.get("rule_change", False)),
            rule_change_text=str(body.get("rule_change_text") or ""),
            # Phase 3: HTML body of the Weekly Thoughts editor. Frontend
            # sanitizes via DOMPurify before sending; the column is plain
            # TEXT and accepts any string. Defaults to '' when absent.
            weekly_thoughts=str(body.get("weekly_thoughts") or ""),
            ticker_grades=tg_in or {},
            execution_grade=execution_grade,
            process_grade=process_grade,
            pnl_grade=pnl_grade,
            overall_override=overall_override,
            reviewed_at=reviewed_at,
        )
    except db.WeeklyRetroLockedError as e:
        # Phase 4.6 review lock — surfaced as 409 Conflict so the
        # frontend can re-render the locked state and surface the
        # un-review affordance.
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        print(f"[weekly_retro_upsert] handler failed: {e}")
        return {"error": str(e)}
    except psycopg2.errors.CheckViolation as e:
        # Most likely the Monday CHECK or grade vocab CHECK firing — give a
        # concrete message instead of leaking the trigger text. Phase 4.6:
        # any of the 4 grade columns can trip vocab — preserve the
        # column-specific "invalid week_grade" message for backward compat,
        # use "invalid grade" for the new axis columns.
        msg = str(e).lower()
        if "monday" in msg or "isodow" in msg:
            return {"error": "week_start must be a Monday"}
        if "week_grade" in msg:
            return {"error": "invalid week_grade"}
        if "grade" in msg:
            return {"error": "invalid grade"}
        return {"error": "constraint violation"}
    except Exception as e:
        print(f"[weekly_retro_upsert] handler failed: {e}")
        return {"error": str(e)}


@app.delete("/api/weekly-retros/{retro_id}")
@limiter.limit("10/minute")
def weekly_retro_delete(retro_id: int, request: Request):
    """Soft-delete a retro (sets deleted_at = NOW()). Children are left
    intact so a future upsert for the same week revives the row with its
    full picture."""
    try:
        ok = db.soft_delete_weekly_retro(retro_id)
        if not ok:
            return {"error": "not_found"}
        return {"status": "ok", "id": retro_id}
    except Exception as e:
        print(f"[weekly_retro_delete] handler failed: {e}")
        return {"error": str(e)}


# ============================================================
# TAG SYSTEM (Migration 026 — Phase 1)
# ============================================================
# User-created, portfolio-scoped, polymorphic tags. Phase 1 mounts on Weekly
# Retro only; daily journals (Phase 7) and trade summaries (Phase 8) reuse
# the same endpoints with different entity_type values. Standard project
# conventions: body: dict + Body(...), no Pydantic, HTTP 200 with
# {"error": "..."} for business errors, slowapi rate-limits on writes.
#
# Closed color palette and the max-10-per-entity cap are enforced API-side
# so the wire contract is stable regardless of frontend client.

_TAG_VALID_COLORS = {"rose", "amber", "emerald", "sky", "violet"}
_TAG_VALID_ENTITY_TYPES = {"weekly_retro", "daily_journal", "trades_summary"}
_TAG_MAX_PER_ENTITY = 10


@app.get("/api/tags")
def list_tags_endpoint(portfolio: str = Query("CanSlim")):
    """List the user's live tags for a portfolio. RLS scopes to the caller;
    no separate user filter is needed."""
    try:
        return db.load_tags(portfolio)
    except Exception as e:
        print(f"[list_tags_endpoint] handler failed: {e}")
        return {"error": str(e)}


@app.post("/api/tags")
@limiter.limit("10/minute")
def create_tag_endpoint(request: Request, body: dict = Body(...)):
    """Create a new tag. Body: { portfolio, name, color }. Color must be
    in the closed palette (rose|amber|emerald|sky|violet). Case-insensitive
    name collision returns {"error": "tag_name_exists"} so the frontend
    can surface a clean message instead of the raw IntegrityError."""
    portfolio = body.get("portfolio") or "CanSlim"
    name = (body.get("name") or "").strip()
    color = body.get("color") or ""
    if not name:
        return {"error": "name required"}
    if color not in _TAG_VALID_COLORS:
        return {"error": "invalid_color"}
    try:
        return db.create_tag(portfolio, name, color)
    except ValueError as e:
        print(f"[create_tag_endpoint] handler failed: {e}")
        return {"error": str(e)}
    except psycopg2.errors.UniqueViolation:
        return {"error": "tag_name_exists"}
    except Exception as e:
        print(f"[create_tag_endpoint] handler failed: {e}")
        return {"error": str(e)}


@app.patch("/api/tags/{tag_id}")
@limiter.limit("30/minute")
def update_tag_endpoint(tag_id: int, request: Request, body: dict = Body(...)):
    """Patch a tag's name and/or color. Whitelisted to those two fields;
    unknown body keys are silently ignored. Returns the updated row, or
    {"error": "not_found"} if the tag is missing or already soft-deleted."""
    fields: dict = {}
    if "name" in body:
        fields["name"] = body["name"]
    if "color" in body:
        if body["color"] not in _TAG_VALID_COLORS:
            return {"error": "invalid_color"}
        fields["color"] = body["color"]
    try:
        row = db.update_tag(tag_id, **fields)
        if row is None:
            return {"error": "not_found"}
        return row
    except ValueError as e:
        print(f"[update_tag_endpoint] handler failed: {e}")
        return {"error": str(e)}
    except psycopg2.errors.UniqueViolation:
        return {"error": "tag_name_exists"}
    except Exception as e:
        print(f"[update_tag_endpoint] handler failed: {e}")
        return {"error": str(e)}


@app.delete("/api/tags/{tag_id}")
@limiter.limit("10/minute")
def delete_tag_endpoint(tag_id: int, request: Request):
    """Soft-delete a tag. Assignments are left in place but become
    invisible (load_tag_assignments filters tags.deleted_at IS NULL).
    Un-deleting reactivates every historical assignment."""
    try:
        ok = db.soft_delete_tag(tag_id)
        if not ok:
            return {"error": "not_found"}
        return {"status": "ok", "id": tag_id}
    except Exception as e:
        print(f"[delete_tag_endpoint] handler failed: {e}")
        return {"error": str(e)}


@app.get("/api/tags/assignments")
def list_tag_assignments_endpoint(
    entity_type: str = Query(...),
    entity_id: int = Query(...),
):
    """List the tags currently attached to one entity. Joined with each
    tag's display fields (tag_name, tag_color) so the frontend doesn't
    need a second fetch."""
    if entity_type not in _TAG_VALID_ENTITY_TYPES:
        return {"error": "invalid_entity_type"}
    try:
        return db.load_tag_assignments(entity_type, entity_id)
    except Exception as e:
        print(f"[list_tag_assignments_endpoint] handler failed: {e}")
        return {"error": str(e)}


@app.post("/api/tags/assignments")
@limiter.limit("30/minute")
def create_tag_assignment_endpoint(request: Request, body: dict = Body(...)):
    """Attach a tag to an entity. Body: { tag_id, entity_type, entity_id }.
    Idempotent: re-attaching a tag is a no-op; re-attaching a previously-
    detached tag REVIVES the soft-deleted assignment (preserves id).

    Enforces the hard cap of 10 live assignments per (entity_type,
    entity_id) — counted via count_live_tag_assignments. The cap doesn't
    block restoring an already-counted assignment (idempotent path checks
    happen first inside the helper)."""
    tag_id = body.get("tag_id")
    entity_type = body.get("entity_type")
    entity_id = body.get("entity_id")
    if not isinstance(tag_id, int):
        return {"error": "tag_id required"}
    if entity_type not in _TAG_VALID_ENTITY_TYPES:
        return {"error": "invalid_entity_type"}
    if not isinstance(entity_id, int):
        return {"error": "entity_id required"}
    try:
        # Cap check: count current live assignments. Reject if >= 10 AND
        # this would be a NEW or REVIVED row (not a no-op re-attach of a
        # tag already in the count). The cleanest test: if the cap is hit
        # AND no live assignment for this tag exists, reject.
        if db.count_live_tag_assignments(entity_type, entity_id) >= _TAG_MAX_PER_ENTITY:
            existing = [a for a in db.load_tag_assignments(entity_type, entity_id)
                        if a["tag_id"] == tag_id]
            if not existing:
                return {"error": "tag_limit_reached"}
        return db.create_tag_assignment(tag_id, entity_type, entity_id)
    except ValueError as e:
        print(f"[create_tag_assignment_endpoint] handler failed: {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"[create_tag_assignment_endpoint] handler failed: {e}")
        return {"error": str(e)}


@app.delete("/api/tags/assignments/{assignment_id}")
@limiter.limit("30/minute")
def delete_tag_assignment_endpoint(assignment_id: int, request: Request):
    """Detach a tag from an entity (soft-delete the assignment). Returns
    not_found if the row doesn't exist or has already been detached."""
    try:
        ok = db.soft_delete_tag_assignment(assignment_id)
        if not ok:
            return {"error": "not_found"}
        return {"status": "ok", "id": assignment_id}
    except Exception as e:
        print(f"[delete_tag_assignment_endpoint] handler failed: {e}")
        return {"error": str(e)}


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
        "Match_Method": "match_method",
        "Stop_Ladder": "stop_ladder",
        "Strategy": "strategy",
        "B1_Entry_Price": "b1_entry_price",
        "B1_Max_Return_Pct": "b1_max_return_pct",
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


@app.get("/api/campaigns/review")
def campaigns_review(portfolio: str = "CanSlim", since: str = "2026-01-01"):
    """Closed campaigns for the Campaign Review page. One row per campaign
    (no B1/A1/S1 detail expansion) enriched with:
      - initial_risk_dollars (from B1 detail: (price - stop) * shares * mult)
      - r_multiple (realized_pl / initial_risk_dollars)
      - grade (from trades_summary.grade)
      - lesson_note + lesson_category (from trade_lessons)
      - has_add_ons (any Trx_ID starting with 'A' among the trade's details)

    Filtered to campaigns with closed_date >= since (default 2026-01-01),
    which naturally covers "2025 opens that closed in 2026". The frontend
    handles further filtering (ticker/rule/instrument/lesson/date/grade)
    client-side."""
    try:
        df_s = db.load_summary(portfolio)
        if df_s.empty:
            return []
        df_s = _normalize_trades(df_s)

        status_col = "status" if "status" in df_s.columns else "Status"
        closed = df_s[df_s[status_col].str.upper() == "CLOSED"].copy()
        if closed.empty:
            return []

        if "closed_date" in closed.columns:
            closed["closed_date"] = pd.to_datetime(closed["closed_date"], errors="coerce")
            since_ts = pd.to_datetime(since, errors="coerce")
            if pd.notna(since_ts):
                closed = closed[closed["closed_date"] >= since_ts]
        if closed.empty:
            return []

        in_scope_ids = set(closed["trade_id"].astype(str))
        details_by_trade: dict = {}
        df_d = db.load_details(portfolio)
        if not df_d.empty:
            df_d = _normalize_trades(df_d)
            df_d = df_d[df_d["trade_id"].astype(str).isin(in_scope_ids)]
            if "date" in df_d.columns:
                df_d["date"] = pd.to_datetime(df_d["date"], errors="coerce")
            for tid, group in df_d.groupby("trade_id"):
                details_by_trade[str(tid)] = group.sort_values("date")

        lessons = db.get_trade_lessons(portfolio) or {}

        rows = []
        for _, r in closed.iterrows():
            tid = str(r.get("trade_id", ""))
            g = details_by_trade.get(tid)

            initial_risk = 0.0
            r_multiple = None
            has_add_ons = False

            if g is not None and not g.empty:
                buys = g[g["action"].astype(str).str.upper() == "BUY"] if "action" in g.columns else g.iloc[0:0]
                b1 = None
                if not buys.empty:
                    trx_series = buys["trx_id"].astype(str).str.upper() if "trx_id" in buys.columns else None
                    if trx_series is not None:
                        b1_rows = buys[trx_series == "B1"]
                        b1 = b1_rows.iloc[0] if not b1_rows.empty else buys.iloc[0]
                    else:
                        b1 = buys.iloc[0]

                if b1 is not None:
                    try:
                        b1_price = float(b1.get("amount") or 0)
                        b1_stop = float(b1.get("stop_loss") or 0)
                        b1_shares = float(b1.get("shares") or 0)
                        b1_mult = float(b1.get("multiplier") or 1) or 1.0
                        if b1_price > 0 and b1_stop > 0 and b1_shares > 0 and b1_stop < b1_price:
                            initial_risk = (b1_price - b1_stop) * b1_shares * b1_mult
                    except (TypeError, ValueError):
                        pass

                if "trx_id" in g.columns:
                    trx_ids = g["trx_id"].astype(str).str.upper()
                    has_add_ons = bool((trx_ids.str.startswith("A")).any())

            realized_pl = float(r.get("realized_pl") or 0)
            if initial_risk > 0:
                r_multiple = realized_pl / initial_risk

            note, category = lessons.get(tid, ("", ""))

            grade_val = r.get("grade")
            grade_out: int | None
            if grade_val is None or (isinstance(grade_val, float) and pd.isna(grade_val)):
                grade_out = None
            else:
                try:
                    grade_out = int(grade_val)
                except (TypeError, ValueError):
                    grade_out = None

            closed_dt = r.get("closed_date")
            open_dt = r.get("open_date")

            rows.append({
                "trade_id": tid,
                "ticker": str(r.get("ticker", "")),
                "open_date": open_dt.isoformat() if hasattr(open_dt, "isoformat") else str(open_dt or ""),
                "closed_date": closed_dt.isoformat() if hasattr(closed_dt, "isoformat") else str(closed_dt or ""),
                "realized_pl": realized_pl,
                "return_pct": float(r.get("return_pct") or 0),
                "initial_risk_dollars": initial_risk,
                "r_multiple": r_multiple,
                "grade": grade_out,
                "lesson_note": note,
                "lesson_category": category,
                "instrument_type": str(r.get("instrument_type", "STOCK") or "STOCK"),
                "has_add_ons": has_add_ons,
                "rule": str(r.get("rule", "") or ""),
                "sell_rule": str(r.get("sell_rule", "") or ""),
                "shares": float(r.get("shares") or 0),
                "avg_entry": float(r.get("avg_entry") or 0),
                "avg_exit": float(r.get("avg_exit") or 0),
            })

        rows.sort(key=lambda x: x.get("closed_date") or "", reverse=True)
        return rows
    except Exception as e:
        import traceback
        print(
            f"[campaigns_review] handler failed: {e}\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        return {"error": str(e)}


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


# In-memory cache for /api/prices/lookup. Each entry is (timestamp, payload).
# 5-minute TTL is enough to coalesce the 24-call burst Portfolio Heat fires
# per page load down to ~24 calls per 5 minutes — well under yfinance's
# undocumented per-IP throttle. Errors are NOT cached; transient failures
# retry cleanly on the next page load.
_atr_cache: dict[str, tuple[float, dict]] = {}
_ATR_CACHE_TTL_S = 300


@app.get("/api/prices/lookup")
@limiter.limit("30/minute")
def price_lookup(request: Request, ticker: str = ""):
    """Get live price + ATR for a single ticker. Used by Position Sizer and Log Buy."""
    if not ticker.strip():
        raise HTTPException(status_code=400, detail="No ticker provided")
    import yfinance as yf

    t = ticker.strip().upper()
    cached = _atr_cache.get(t)
    if cached is not None and (time.time() - cached[0]) < _ATR_CACHE_TTL_S:
        return cached[1]

    try:
        stock = yf.Ticker(t)
        df = stock.history(period="40d")
        if df.empty:
            raise HTTPException(status_code=503, detail=f"Price data unavailable for {t}")

        price = float(df["Close"].iloc[-1])

        # ATR calculation — matches Streamlit: ATR% = SMA(TR,21) / SMA(Low,21) * 100.
        # When fewer than 21 bars are available we now return 0.0 instead of the
        # legacy 5.0 default — aligns with _compute_ticker_atr_pct so the snapshot
        # and live paths agree on sparse-history tickers.
        atr_pct = 0.0
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

        result = {
            "ticker": t,
            "price": round(price, 2),
            "atr": round(atr_21, 2),
            "atr_pct": round(atr_pct, 2),
        }
        _atr_cache[t] = (time.time(), result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Price data unavailable for {t}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# /api/prices/lookup-batch — many-tickers-in-one-call variant of the live
# price + ATR lookup. Designed for Portfolio Heat: one HTTP call covering N
# tickers instead of N sequential calls. The previous sequential pattern
# would burn through /api/prices/lookup's 30/minute rate limit on
# portfolios with >30 stock positions (or any portfolio when the user had
# also recently used Log Buy / Position Sizer, which share that endpoint).
# The user's three oldest CanSlim positions were consistently flagged
# "unavailable" because they were always the last to be requested in the
# sequential loop and so always the ones that hit 429.
#
# Status vocabulary lets the frontend distinguish:
#   - "ok"     — atr_pct is a number; render normally
#   - "empty"  — yfinance returned no rows; price/atr_pct null
#   - "sparse" — yfinance returned <21 bars (e.g., recent IPO); price
#                surfaced as the last close, but atr_pct=0.0 because
#                21-period ATR isn't meaningful
#   - "error"  — exception during fetch; price/atr_pct null. Logged
#                server-side so a transient yfinance outage can be
#                diagnosed from operator logs.
#
# Reuses the in-process _atr_cache to short-circuit repeated lookups
# within the 5-minute TTL — the cache is per-ticker keyed and shared
# with the single-ticker endpoint above. A cached entry counts against
# neither yfinance nor the rate-limit slot.
# ─────────────────────────────────────────────────────────────────────────────


_BATCH_MAX_TICKERS = 50


@app.get("/api/prices/lookup-batch")
@limiter.limit("10/minute")
def price_lookup_batch(request: Request, tickers: str = ""):
    """Batch live price + ATR for a comma-separated ticker list.

    Returns one result per ticker (deduped, normalized to upper-case).
    Per-ticker failures don't fail the whole call — each result carries
    a `status` field. Rate-limit slot usage is 1 per call regardless of
    ticker count, so this is preferable to fan-out from the client.
    """
    if not tickers.strip():
        raise HTTPException(status_code=400, detail="No tickers provided")
    import yfinance as yf

    # Normalize + dedupe (preserving first-seen order so callers can rely
    # on the response shape matching their request shape when no dupes
    # were sent).
    seen: set[str] = set()
    normalized: list[str] = []
    for raw in tickers.split(","):
        t = raw.strip().upper()
        if not t or t in seen:
            continue
        seen.add(t)
        normalized.append(t)

    if not normalized:
        raise HTTPException(status_code=400, detail="No valid tickers in request")
    if len(normalized) > _BATCH_MAX_TICKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many tickers ({len(normalized)}); max {_BATCH_MAX_TICKERS} per call",
        )

    results: list[dict] = []
    for t in normalized:
        # Cache short-circuit. Wraps the cached single-ticker result with
        # status='ok' so the batch shape is uniform.
        cached = _atr_cache.get(t)
        if cached is not None and (time.time() - cached[0]) < _ATR_CACHE_TTL_S:
            cached_payload = cached[1]
            results.append({
                "ticker": t,
                "price": cached_payload.get("price"),
                "atr_pct": cached_payload.get("atr_pct"),
                "status": "ok",
            })
            continue

        try:
            df = yf.Ticker(t).history(period="40d")
        except Exception as e:
            print(f"[price_lookup_batch] yfinance threw for {t}: "
                  f"{type(e).__name__}: {e}")
            results.append({
                "ticker": t, "price": None, "atr_pct": None, "status": "error",
            })
            continue

        if df.empty:
            results.append({
                "ticker": t, "price": None, "atr_pct": None, "status": "empty",
            })
            continue

        price = float(df["Close"].iloc[-1])

        if len(df) < 21:
            # Sparse history (recent IPO etc.). Surface last close but flag
            # so the UI can render an explanatory badge.
            results.append({
                "ticker": t, "price": round(price, 2),
                "atr_pct": 0.0, "status": "sparse",
            })
            continue

        # Full ATR computation — identical formula to /api/prices/lookup.
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        sma_tr = float(tr.tail(21).mean())
        sma_low = float(df["Low"].tail(21).mean())
        atr_pct = (sma_tr / sma_low) * 100 if sma_low > 0 else 0.0
        atr_21 = sma_tr

        payload = {
            "ticker": t,
            "price": round(price, 2),
            "atr": round(atr_21, 2),
            "atr_pct": round(atr_pct, 2),
        }
        _atr_cache[t] = (time.time(), payload)
        results.append({
            "ticker": t,
            "price": payload["price"],
            "atr_pct": payload["atr_pct"],
            "status": "ok",
        })

    return {"results": results}


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
        print(f"[chart_ohlcv] handler failed: {e}")
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

    ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        return {}
    try:
        return _fetch_live_prices_with_manual_overlay(ticker_list, portfolio)
    except Exception as e:
        print(f"[batch_prices] handler failed: {e}")
        return {"error": str(e)}


def _fetch_live_prices_with_manual_overlay(
    ticker_list: list[str], portfolio: str = "",
) -> dict[str, float]:
    """Shared live-price fetch + manual_price overlay.

    Mirrors batch_prices' live-path behavior 1:1 — extracted so other
    endpoints (e.g. /api/analytics/add-effectiveness) can reuse the
    same price semantics without duplicating the provider call + the
    manual_price overlay. Do NOT add a second copy of this logic.

    Behavior:
      * yfinance can't reliably resolve OCC option symbols, so options
        skip the live fetch. They get priced solely by the manual_price
        overlay when portfolio is set; options without a manual_price
        are omitted (same shape as a yfinance miss).
      * Result is re-keyed to the caller's readable ticker, including
        the user's original casing.
      * manual_price overlay loads trades_summary at status='OPEN' and
        replaces the live value for any matching open position with a
        finite positive Manual_Price/manual_price (matching the same
        NaN / type guards as batch_prices).

    Returns {readable_ticker: price}. May raise — caller wraps.
    """
    from price_providers import get_price_provider

    yf_to_readable: dict[str, str] = {}
    yf_symbols: list[str] = []
    for t in ticker_list:
        if _is_option_ticker(t):
            continue
        yf_symbols.append(t)
        yf_to_readable[t] = t

    if not yf_symbols and not portfolio:
        # No yfinance work and no overlay would apply.
        return {}

    live = (get_price_provider().get_current_prices(yf_symbols)
            if yf_symbols else {})
    result: dict[str, float] = {
        yf_to_readable.get(yf_sym, yf_sym): price
        for yf_sym, price in live.items()
    }

    if portfolio:
        try:
            summary_df = db.load_summary(portfolio, status="OPEN")
        except Exception as e:
            print(f"[prices_batch] manual_price overlay load failed: {e}")
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
                        result[requested_upper[tkr]] = mp_f
    return result


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
    if response.get("state") not in ("UPTREND", "UPTREND UNDER PRESSURE", "POWERTREND", "RALLY MODE"):
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
        print(f"[rally_data] handler failed: {e}")
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
        print(f"[get_recent_market_signals] handler failed: {e}")
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
        print(f"[get_config] handler failed: {e}")
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
def list_strategies_endpoint(active: bool = True, portfolio: str | None = None):
    """Return rows from the strategies lookup table (Migration 019; scoped
    by Migration 038).

    `active` (default true) filters to is_active=true — what the Log Buy
    dropdown wants. Pass `?active=false` to include disabled strategies
    (used by Phase 2 admin UI).

    `portfolio` (optional) scopes the result to strategies allowed in that
    portfolio. Strategies with NULL allowed_portfolio_names are visible
    everywhere; an explicit list narrows visibility. Omit the param for
    admin / cross-portfolio views that want every strategy regardless of
    portfolio scoping.
    """
    try:
        rows = db.load_strategies(active_only=active, portfolio_name=portfolio)
        return [_serialize_strategy(r) for r in rows]
    except Exception as e:
        print(f"[list_strategies_endpoint] handler failed: {e}")
        return {"error": str(e)}


def _serialize_strategy(r: dict) -> dict:
    """Shared serialization for strategy rows. Centralized so GET, POST,
    and PUT all return the same shape (name, description, color,
    is_active, created_at-as-isoformat, allowed_portfolio_names)."""
    return {
        "name": r["name"],
        "description": r.get("description"),
        "color": r["color"],
        "is_active": r["is_active"],
        "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
        "allowed_portfolio_names": r.get("allowed_portfolio_names"),
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
        print(f"[create_strategy_endpoint] handler failed: {e}")
        return {"error": str(e)}
    except psycopg2.errors.UniqueViolation:
        return {"error": f"Strategy '{body.get('name', '')}' already exists"}
    except Exception as e:
        print(f"[create_strategy_endpoint] handler failed: {e}")
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
        print(f"[update_strategy_endpoint] handler failed: {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"[update_strategy_endpoint] handler failed: {e}")
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
        print(f"[patch_trade_strategy_endpoint] handler failed: {e}")
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
        print(f"[bulk_set_trade_strategy_endpoint] handler failed: {e}")
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
        print(f"[list_portfolios_endpoint] handler failed: {e}")
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
        print(f"[create_portfolio_endpoint] handler failed: {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"[create_portfolio_endpoint] handler failed: {e}")
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
        print(f"[update_portfolio_endpoint] handler failed: {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"[update_portfolio_endpoint] handler failed: {e}")
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
        print(f"[delete_portfolio_endpoint] handler failed: {e}")
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
        print(f"[get_portfolio_nlv] handler failed: {e}")
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
        print(f"[get_portfolio_returns] handler failed: {e}")
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
        print(f"[get_portfolio_twr_returns] handler failed: {e}")
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
        print(f"[get_dashboard_metrics] handler failed: {e}")
        return {"error": str(e)}


@app.get("/api/analytics/weekly-metrics")
@limiter.limit("30/minute")
def get_weekly_metrics(request: Request,
                       portfolio: str = "CanSlim",
                       week_start: str = ""):
    """Phase 5 — performance tiles for the Weekly Retro page.

    Reuses the same TWR / NLV-delta math as Period Review's weekly
    aggregation (the formulas live in nlv_service, the per-day return
    series mirrors /api/journal/history). All metrics are computed as
    of week_end (Friday of the requested week); historical weeks are
    stable because the underlying journal + trades_summary are immutable.

    Query: portfolio (name), week_start (YYYY-MM-DD, the Monday).
    """
    if not week_start:
        return {"error": "week_start (YYYY-MM-DD) is required"}
    try:
        return nlv_service.weekly_metrics(portfolio, week_start)
    except Exception as e:
        print(f"[get_weekly_metrics] handler failed: {e}")
        return {"error": str(e)}


@app.get("/api/analytics/add-effectiveness")
@limiter.limit("30/minute")
def add_effectiveness(
    request: Request,
    portfolio: str = Query(...),
    start: str = Query(""),
    end: str = Query(""),
    strategy: str = Query(""),
):
    """Group scale-in (add) buys by their own buy rule over a date window
    and report effectiveness.

    An "add" = a trades_details row with trx_id starting 'A' (the prefix
    log_buy assigns to every non-opening BUY — see api/main.py log_buy
    handler), action='BUY', deleted_at IS NULL, in the requested portfolio,
    with detail-row date inside [start, end] (both inclusive).

    Per-add metrics reuse existing functions verbatim — no LIFO or price-
    fetch reimplementation:
      * Realized P&L → SUM(lot_closures.realized_pl WHERE buy_trx_id =
        add.trx_id) — already multiplier-scaled by save_summary_with_closures.
      * Open shares → compute_open_inventory(campaign_txns), filtered to
        the lot whose trx_id matches this add.
      * Unrealized P&L → (current_price − add.amount) × open_shares ×
        multiplier. Current price comes from one batched call to
        _fetch_live_prices_with_manual_overlay (the same path /api/prices/
        batch uses), including manual_price overlay for open positions.
      * Extension at add + pyramid-up vs average-down classification →
        replay _walk_inventory on txns with date < add.date to derive
        the pre-add open lots; blended_cost =
        Σ(lot.shares × lot.price) / Σ(lot.shares).
        extension_pct = (add.price − blended_cost) / blended_cost × 100.
        Ties (add.price >= blended_cost) classify as pyramid-up; strictly
        below = average-down.

    Aggregates per rule (grouped by the DETAIL row's rule, NOT the
    summary's rule — an add can carry a different rule than the opening
    buy, and using the summary rule would erase the point of the view):
      rule, add_count, realized_pl, unrealized_pl, closed_count
      (adds with ≥1 lot_closures row), win_rate, avg_realized_per_add,
      avg_extension_at_add.

    Headline totals + a discipline guardrail (average_down_count and a
    list of those add identifiers — normally 0; surfaces exceptions or
    mis-tagged rule rows).

    Query:
      portfolio: required, name (e.g. "CanSlim")
      start, end: YYYY-MM-DD inclusive; empty/omitted → unbounded on that side
      strategy: filter trades_summary.strategy; empty or "all" → no filter
    """
    try:
        # Parse window — empty params mean unbounded on that side.
        try:
            start_ts = pd.Timestamp(start) if start else pd.Timestamp.min
        except Exception:
            return {"error": f"Invalid start date: {start!r} (expected YYYY-MM-DD)"}
        try:
            # End is INCLUSIVE — extend a date-only "end" to end-of-day so
            # rows logged later that day (with HH:MM:SS) still fall inside.
            end_ts = (pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)) \
                if end else pd.Timestamp.max
        except Exception:
            return {"error": f"Invalid end date: {end!r} (expected YYYY-MM-DD)"}

        df_s = db.load_summary(portfolio)
        if df_s.empty:
            return _empty_add_effectiveness_response(portfolio, start, end, strategy)
        df_s = _normalize_trades(df_s)

        # Strategy filter — "all" or empty means no filter. Case-insensitive.
        strategy_filter = (strategy or "").strip()
        if strategy_filter and strategy_filter.lower() != "all":
            if "strategy" not in df_s.columns:
                return _empty_add_effectiveness_response(portfolio, start, end, strategy)
            df_s = df_s[
                df_s["strategy"].astype(str).str.lower() == strategy_filter.lower()
            ].copy()
        if df_s.empty:
            return _empty_add_effectiveness_response(portfolio, start, end, strategy)

        in_scope_trade_ids: set[str] = set(df_s["trade_id"].astype(str))
        # Summary lookup: trade_id → {ticker, multiplier}
        summary_lookup: dict[str, dict[str, Any]] = {}
        for _, srow in df_s.iterrows():
            tid = str(srow.get("trade_id") or "")
            if not tid:
                continue
            mult_raw = srow.get("multiplier")
            try:
                mult = float(mult_raw) if mult_raw is not None and not pd.isna(mult_raw) else 1.0
            except (TypeError, ValueError):
                mult = 1.0
            if mult <= 0:
                mult = 1.0
            summary_lookup[tid] = {
                "ticker": str(srow.get("ticker") or "").strip(),
                "multiplier": mult,
            }

        df_d = db.load_details(portfolio)
        if df_d.empty:
            return _empty_add_effectiveness_response(portfolio, start, end, strategy)
        df_d = _normalize_trades(df_d)
        df_d["date_parsed"] = pd.to_datetime(df_d["date"], errors="coerce")
        # Drop rows whose date couldn't be parsed — they'd silently slip the
        # window filter (NaT < anything → False; <= anything → False).
        df_d = df_d.dropna(subset=["date_parsed"])

        # Identify ADD rows (the spec: trx_id LIKE 'A%' + BUY + in scope +
        # inside window). deleted_at is already filtered by load_details.
        action_col = df_d["action"].astype(str).str.upper()
        trx_col = df_d["trx_id"].astype(str)
        adds = df_d[
            df_d["trade_id"].astype(str).isin(in_scope_trade_ids)
            & (action_col == "BUY")
            & trx_col.str.startswith("A")
            & (df_d["date_parsed"] >= start_ts)
            & (df_d["date_parsed"] <= end_ts)
        ].copy()

        if adds.empty:
            return _empty_add_effectiveness_response(portfolio, start, end, strategy)

        # ONE batched price fetch for every distinct in-scope ticker. The
        # window restricts adds, not tickers — campaigns with adds in the
        # window may have open shares that need a current price for
        # unrealized P&L. Use the in-scope set, not just adds' tickers.
        distinct_tickers = sorted({
            v["ticker"] for v in summary_lookup.values() if v["ticker"]
        })
        try:
            prices = _fetch_live_prices_with_manual_overlay(
                distinct_tickers, portfolio,
            ) if distinct_tickers else {}
        except Exception as e:
            print(f"[add_effectiveness] price fetch failed: {e}")
            prices = {}

        # lot_closures for in-scope trades — one DB round-trip.
        try:
            df_lc = db.load_lot_closures(
                portfolio, trade_ids=list(in_scope_trade_ids),
            )
        except Exception as e:
            print(f"[add_effectiveness] load_lot_closures failed: {e}")
            df_lc = pd.DataFrame(columns=[
                "trade_id", "buy_trx_id", "realized_pl",
            ])
        # Index closures by (trade_id, buy_trx_id) → (sum_pl, count).
        closure_index: dict[tuple[str, str], tuple[float, int]] = {}
        if df_lc is not None and not df_lc.empty:
            for (tid, btid), grp in df_lc.groupby(["trade_id", "buy_trx_id"]):
                closure_index[(str(tid), str(btid))] = (
                    float(grp["realized_pl"].sum()),
                    int(len(grp)),
                )

        # Pre-compute the full-campaign open inventory once per in-scope
        # trade_id (cheaper than re-walking per-add for open-share lookup).
        open_inv_by_trade: dict[str, dict[str, float]] = {}
        # Pre-bucket details by trade_id so we don't repeatedly filter
        # the global DataFrame in the per-add loop.
        details_by_trade: dict[str, pd.DataFrame] = {}
        for tid, grp in df_d.groupby(df_d["trade_id"].astype(str)):
            details_by_trade[tid] = grp
            if tid in in_scope_trade_ids:
                try:
                    inv = compute_open_inventory(grp)
                except Exception as e:
                    print(f"[add_effectiveness] compute_open_inventory failed for {tid}: {e}")
                    inv = []
                by_trx: dict[str, float] = {}
                for lot in inv:
                    by_trx[str(lot.get("trx_id", "") or "")] = float(
                        lot.get("shares", 0) or 0,
                    )
                open_inv_by_trade[tid] = by_trx

        # Per-rule accumulator
        rules_acc: dict[str, dict[str, Any]] = {}
        average_down_list: list[dict[str, Any]] = []
        total_adds_with_closure = 0
        total_wins = 0

        for _, add_row in adds.iterrows():
            tid = str(add_row.get("trade_id") or "")
            trx_id = str(add_row.get("trx_id") or "")
            try:
                add_price = float(add_row.get("amount") or 0)
            except (TypeError, ValueError):
                add_price = 0.0
            add_date = add_row["date_parsed"]
            rule_raw = add_row.get("rule")
            rule = (
                str(rule_raw).strip()
                if rule_raw is not None and not pd.isna(rule_raw) and str(rule_raw).strip()
                else "(no rule)"
            )

            summ = summary_lookup.get(tid, {})
            ticker = str(summ.get("ticker") or "")
            multiplier = float(summ.get("multiplier") or 1.0)

            # Pre-add blended cost via _walk_inventory on the slice of
            # campaign txns strictly before this add. Reuses the canonical
            # walker — no parallel implementation.
            campaign_txns = details_by_trade.get(tid)
            blended_cost = 0.0
            blended_shares = 0.0
            if campaign_txns is not None and not campaign_txns.empty:
                pre_add = campaign_txns[campaign_txns["date_parsed"] < add_date]
                if not pre_add.empty:
                    try:
                        pre_inv = compute_open_inventory(pre_add)
                    except Exception as e:
                        print(f"[add_effectiveness] pre-add walk failed for {tid}/{trx_id}: {e}")
                        pre_inv = []
                    for lot in pre_inv:
                        s = float(lot.get("shares", 0) or 0)
                        p = float(lot.get("price", 0) or 0)
                        blended_shares += s
                        blended_cost += s * p
                    if blended_shares > 0:
                        blended_cost = blended_cost / blended_shares
                    else:
                        blended_cost = 0.0

            # Classify. Spec: ties (add.price >= blended_cost) = pyramid-up.
            # Adds with no prior open lots (blended_shares == 0) shouldn't
            # happen given trx_id 'A%' definition, but if encountered
            # default to pyramid-up (treat as if add opened the campaign).
            extension_pct: float | None = None
            is_average_down = False
            if blended_cost > 0:
                extension_pct = (add_price - blended_cost) / blended_cost * 100.0
                is_average_down = add_price < blended_cost

            # Realized P&L for this add — sum of multiplier-scaled
            # lot_closures rows whose buy_trx_id matches.
            realized_pl, closure_rows = closure_index.get((tid, trx_id), (0.0, 0))

            # Open shares for THIS add (filter the precomputed inventory).
            open_shares = float(
                open_inv_by_trade.get(tid, {}).get(trx_id, 0.0),
            )

            # Unrealized P&L = (current_price - add_price) × open_shares × mult.
            current_price = float(prices.get(ticker, 0) or 0)
            unrealized_pl = (
                (current_price - add_price) * open_shares * multiplier
                if open_shares > 0 and current_price > 0 and add_price > 0
                else 0.0
            )

            # Accumulate per-rule.
            acc = rules_acc.setdefault(rule, {
                "rule": rule,
                "add_count": 0,
                "realized_pl": 0.0,
                "unrealized_pl": 0.0,
                "closed_count": 0,
                "win_count": 0,
                "ext_sum": 0.0,
                "ext_count": 0,
            })
            acc["add_count"] += 1
            acc["realized_pl"] += realized_pl
            acc["unrealized_pl"] += unrealized_pl
            if closure_rows > 0:
                acc["closed_count"] += 1
                total_adds_with_closure += 1
                if realized_pl > 0:
                    acc["win_count"] += 1
                    total_wins += 1
            if extension_pct is not None:
                acc["ext_sum"] += extension_pct
                acc["ext_count"] += 1
            if is_average_down:
                average_down_list.append({
                    "trade_id": tid, "trx_id": trx_id, "ticker": ticker,
                    "rule": rule,
                    "add_price": round(add_price, 4),
                    "blended_cost_pre_add": round(blended_cost, 4),
                })

        # Build per-rule output rows.
        rule_rows: list[dict[str, Any]] = []
        total_adds = 0
        total_realized = 0.0
        total_unrealized = 0.0
        for r in rules_acc.values():
            cc = r["closed_count"]
            win_rate = (r["win_count"] / cc) if cc > 0 else 0.0
            avg_realized = (r["realized_pl"] / cc) if cc > 0 else 0.0
            avg_ext = (r["ext_sum"] / r["ext_count"]) if r["ext_count"] > 0 else 0.0
            rule_rows.append({
                "rule": r["rule"],
                "add_count": r["add_count"],
                "realized_pl": round(r["realized_pl"], 2),
                "unrealized_pl": round(r["unrealized_pl"], 2),
                "closed_count": cc,
                "win_rate": round(win_rate, 4),
                "avg_realized_per_add": round(avg_realized, 2),
                "avg_extension_at_add": round(avg_ext, 4),
            })
            total_adds += r["add_count"]
            total_realized += r["realized_pl"]
            total_unrealized += r["unrealized_pl"]
        # Sort by add_count desc, then realized_pl desc for ties.
        rule_rows.sort(
            key=lambda r: (-r["add_count"], -r["realized_pl"]),
        )

        return {
            "rules": rule_rows,
            "totals": {
                "total_adds": total_adds,
                "total_realized_pl": round(total_realized, 2),
                "total_unrealized_pl": round(total_unrealized, 2),
                "overall_win_rate": round(
                    (total_wins / total_adds_with_closure)
                    if total_adds_with_closure > 0 else 0.0,
                    4,
                ),
                "avg_realized_per_add": round(
                    (total_realized / total_adds_with_closure)
                    if total_adds_with_closure > 0 else 0.0,
                    2,
                ),
            },
            "discipline": {
                "average_down_count": len(average_down_list),
                "average_downs": average_down_list,
            },
            "window": {
                "portfolio": portfolio,
                "start": start or None,
                "end": end or None,
                "strategy": (strategy or None) if (strategy and strategy.lower() != "all") else None,
            },
        }
    except Exception as e:
        print(f"[add_effectiveness] handler failed: {e}")
        return {"error": str(e)}


def _empty_add_effectiveness_response(
    portfolio: str, start: str, end: str, strategy: str,
) -> dict[str, Any]:
    """Zero-state response used by add_effectiveness when there are no
    in-scope campaigns or no adds in the window. Preserves the response
    shape so frontend callers can render an empty-state view without
    special-casing."""
    return {
        "rules": [],
        "totals": {
            "total_adds": 0,
            "total_realized_pl": 0.0,
            "total_unrealized_pl": 0.0,
            "overall_win_rate": 0.0,
            "avg_realized_per_add": 0.0,
        },
        "discipline": {
            "average_down_count": 0,
            "average_downs": [],
        },
        "window": {
            "portfolio": portfolio,
            "start": start or None,
            "end": end or None,
            "strategy": (strategy or None) if (strategy and strategy.lower() != "all") else None,
        },
    }


# ─────────────────────────────────────────────────────────────────────
# SR8 Cascade Monitor — wraps mors.monitor.analyze() per position in
# the user's portfolio that the Active Campaign view classifies as
# SR8-tier, and returns hold/trim/exit recommendations.
#
# Architecture:
#   - Source of truth: mors/monitor.py + mors/mors_backtest.py
#     (Python engine that replays the weekly cascade from each
#     position's B1 date through today).
#   - On-demand price fetching: mors_backtest.ensure_data() auto-pulls
#     from yfinance if the per-ticker CSV is missing. First request
#     on a fresh Railway container is slow (~1s per ticker); subsequent
#     hits use the on-disk cache for the container lifetime.
#   - Positions source: trades_summary rows in the active portfolio
#     with shares > 0, classified SR8 via the same effective-max as the
#     frontend: max(live B1 return, stored b1_max_return_pct) >= 50.
#     Live prices come from _fetch_live_prices_with_manual_overlay so
#     the selection matches the campaign table exactly (incl. manual-
#     price overlays). See positions.ts:154-170 + sell-rule.ts:8-13.
# ─────────────────────────────────────────────────────────────────────

_MORS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mors")
_SR8_SPY_CSV_PATH = os.path.join(_MORS_DIR, "data", "SPY_daily.csv")

# Persistent-peak threshold that defines the SR8 tier. Mirrors
# classifySellRuleTier(b1_return_pct >= 50) in frontend/src/lib/sell-rule.ts.
_SR8_TIER_THRESHOLD = 50.0


def _sr8_load_db_positions(portfolio: str) -> list[dict[str, Any]]:
    """Open positions in `portfolio` that are SR8-tier — synthesizes the
    MORS analyze() input shape from trades_summary + trades_details.

    Selection mirrors the frontend classifier 1:1:
      positions.ts:154-170 — effectiveMax = max(live B1 return, stored
        b1_max_return_pct), null-tolerant on either side.
      sell-rule.ts:8-13 — SR8 iff effectiveMax >= 50.
    Live prices come from _fetch_live_prices_with_manual_overlay so the
    selection matches the campaign exactly (manual-price overlay
    included). Using only the stored peak misses fresh SR8 promotions
    whose b1_max hasn't been heal-written yet (the
    active-campaign.tsx:246-263 auto-promote effect can lag here).

    Options are excluded (cascade engine is stock-only). Positions
    missing a first BUY (rare History-import rows) or a valid B1 price
    are skipped — MORS needs both to replay the cascade.

    Returns [] on any DB error so the endpoint surfaces an empty page
    rather than a 500.
    """
    try:
        summary_df = db.load_summary(portfolio)
    except Exception as e:
        print(f"[sr8_monitor] load_summary failed for portfolio={portfolio}: {e}")
        return []
    if summary_df.empty:
        return []
    summary_df = _normalize_trades(summary_df)

    status_col = "status" if "status" in summary_df.columns else "Status"
    open_df = summary_df[summary_df[status_col].astype(str).str.upper() == "OPEN"].copy()
    if open_df.empty:
        return []

    open_df["_shares_num"] = pd.to_numeric(open_df.get("shares"), errors="coerce").fillna(0)
    open_df["_b1_max_num"] = pd.to_numeric(open_df.get("b1_max_return_pct"), errors="coerce")
    open_df["_b1_entry_num"] = pd.to_numeric(open_df.get("b1_entry_price"), errors="coerce")
    open_df["_avg_entry_num"] = pd.to_numeric(open_df.get("avg_entry"), errors="coerce").fillna(0)

    # Pre-filter to viable candidates: held shares, valid B1 entry, not
    # an option. We need the live-price overlay BEFORE filtering on the
    # 50% threshold — otherwise stale stored peaks would silently drop
    # SR8 names whose b1_max hasn't been heal-written yet.
    candidate = open_df[
        (open_df["_shares_num"] > 0)
        & (open_df["_b1_entry_num"].notna())
        & (open_df["_b1_entry_num"] > 0)
    ].copy()
    if "instrument_type" in candidate.columns:
        candidate = candidate[candidate["instrument_type"].astype(str).str.upper() != "OPTION"]
    if candidate.empty:
        return []

    # Same price source the frontend uses (batch_prices / campaign rows)
    # so SR8 selection matches the campaign table exactly. If the live
    # fetch fails (yfinance hiccup, no network), we fall back to the
    # stored peak alone — no worse than the prior behavior.
    ticker_list = [
        str(t).upper().strip()
        for t in candidate["ticker"].tolist()
        if isinstance(t, str) and t.strip()
    ]
    try:
        live_prices = _fetch_live_prices_with_manual_overlay(ticker_list, portfolio)
    except Exception as e:
        print(f"[sr8_monitor] live price fetch failed (falling back to stored peak): {e}")
        live_prices = {}
    # Normalize the live-price map to upper-case ticker keys so we
    # always look it up the same way regardless of provider casing.
    live_by_upper: dict[str, float] = {}
    for k, v in (live_prices or {}).items():
        try:
            kv = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(kv) or kv <= 0:
            continue
        live_by_upper[str(k).upper().strip()] = kv

    def _effective_max(row) -> float:
        """Mirror of positions.ts:154-170 + sell-rule.ts:8-13.
        Returns NaN when neither side resolves so .notna() drops the row.
        """
        stored = row["_b1_max_num"]
        stored_val = float(stored) if pd.notna(stored) else None
        b1 = float(row["_b1_entry_num"])
        tkr_up = str(row.get("ticker") or "").upper().strip()
        live = live_by_upper.get(tkr_up)
        live_val: float | None = None
        if live is not None and b1 > 0:
            live_val = (live - b1) / b1 * 100.0
        if live_val is not None and stored_val is not None:
            return max(live_val, stored_val)
        if live_val is not None:
            return live_val
        if stored_val is not None:
            return stored_val
        return float("nan")

    candidate["_effective_max"] = candidate.apply(_effective_max, axis=1)
    sr8_df = candidate[
        candidate["_effective_max"].notna()
        & (candidate["_effective_max"] >= _SR8_TIER_THRESHOLD)
    ]
    if sr8_df.empty:
        return []

    # First BUY date per trade_id — one details query for the whole portfolio.
    try:
        details_df = db.load_details(portfolio)
    except Exception as e:
        print(f"[sr8_monitor] load_details failed for portfolio={portfolio}: {e}")
        return []
    if details_df.empty:
        return []
    details_df = _normalize_trades(details_df)
    buys = details_df[details_df["action"].astype(str).str.upper() == "BUY"].copy()
    buys["date"] = pd.to_datetime(buys["date"], errors="coerce")
    buys = buys.dropna(subset=["date"]).sort_values(["trade_id", "date"])
    first_buy_dates = buys.groupby("trade_id")["date"].first().to_dict()

    positions: list[dict[str, Any]] = []
    for _, row in sr8_df.iterrows():
        trade_id = row.get("trade_id")
        ticker = str(row.get("ticker") or "").upper().strip()
        if not ticker:
            continue
        b1_date_val = first_buy_dates.get(trade_id)
        if b1_date_val is None:
            continue
        b1_date_iso = (
            b1_date_val.strftime("%Y-%m-%d")
            if hasattr(b1_date_val, "strftime")
            else str(b1_date_val)
        )
        b1_entry = row["_b1_entry_num"]
        if pd.isna(b1_entry) or float(b1_entry) <= 0:
            continue
        positions.append({
            "ticker": ticker,
            "b1_date": b1_date_iso,
            "b1_price": float(b1_entry),
            "shares_held": float(row["_shares_num"]),
            "avg_price": float(row["_avg_entry_num"]),
        })
    return positions


def _sr8_fetched_at_iso() -> str:
    """SPY CSV mtime as a freshness anchor for the snapshot's 'as of'
    label. Empty string when the CSV doesn't exist yet (first request
    will fetch it; subsequent requests will resolve mtime correctly)."""
    try:
        if os.path.exists(_SR8_SPY_CSV_PATH):
            return datetime.fromtimestamp(
                os.path.getmtime(_SR8_SPY_CSV_PATH)
            ).isoformat()
    except Exception:
        pass
    return ""


def _sr8_fetch_failed_row(pos: dict[str, Any], err_msg: str) -> dict[str, Any]:
    """Shape an entry for a position whose analyze() raised. Mirrors
    the design's 'fetch failed' UI state — non-null shares/avg/b1 so
    the row still renders meaningful info, null current_price + 0
    derived metrics."""
    return {
        "ticker": str(pos.get("ticker") or ""),
        "b1_date": str(pos.get("b1_date") or ""),
        "b1_price": float(pos.get("b1_price") or 0),
        "shares_held": float(pos.get("shares_held") or 0),
        "avg_price": float(pos.get("avg_price") or 0),
        "current_price": None,
        "current_dollars": 0.0,
        "current_pct_nlv": 0.0,
        "current_tier": "GREEN",  # safe default for failed-fetch rows
        "tier_pct_nlv": 0.0,
        "target_dollars": 0.0,
        "delta_dollars": 0.0,
        "delta_shares": 0,
        "unreal_dollars": 0.0,
        "unreal_pct": 0.0,
        "last_signal": "",
        "last_signal_date": "",
        "last_bar_date": "",
        "signal_today": False,
        "terminated": False,
        "phase": 1,
        "is_action": False,
        "early_warn": False,
        "fetch_failed": True,
        "fetch_error": err_msg[:200],
    }


def _sr8_enrich(pos: dict[str, Any], r: dict[str, Any]) -> dict[str, Any]:
    """Augment monitor.analyze()'s return dict with the b1_date/b1_price
    pass-through fields + the design-additions (is_action, early_warn,
    fetch_failed=False). Coerces date objects to ISO strings so the
    response is JSON-serializable."""
    # Action gate: surface anything that's NOT Green and still above its
    # floor (delta_dollars > AT_TARGET_TOL = $500). Previously we gated
    # on signal_today, which dropped a trim out of ActionRow once its
    # firing weekly bar passed — a position sitting in QUICK for a week
    # with the trim un-acted-on would disappear from "Action needed"
    # until GD finally fired. signal_today stays on the payload for any
    # downstream caller that wants the "fired today" hint, but the
    # action surface is now the tier+floor truth: Quick/QS/GD over floor
    # = act; Green or at-floor = hold.
    tier_up = str(r.get("current_tier") or "GREEN").upper()
    is_green_tier = tier_up in ("GREEN", "GREEN(SUB-ENTRY)")
    is_action = bool(r.get("terminated")) or (
        (not is_green_tier)
        and float(r.get("delta_dollars") or 0) > 500.0  # AT_TARGET_TOL in monitor.py
    )
    # Early-warning: held position within 2 points BELOW its tier target.
    # Design's earlyWarn definition (README:113-115); a small UI hint
    # the engine doesn't compute but is cheap to derive here.
    tier = float(r.get("tier_pct_nlv") or 0)
    cur_pct = float(r.get("current_pct_nlv") or 0)
    early_warn = (not is_action) and tier > 0 and 0 <= (tier - cur_pct) <= 2

    def _iso(v: Any) -> str:
        if v is None:
            return ""
        if hasattr(v, "isoformat"):
            return v.isoformat()
        return str(v)

    return {
        "ticker": str(r.get("ticker") or pos.get("ticker") or ""),
        "b1_date": str(pos.get("b1_date") or ""),
        "b1_price": float(pos.get("b1_price") or 0),
        "shares_held": float(r.get("shares_held") or 0),
        "avg_price": float(r.get("avg_price") or 0),
        "current_price": float(r.get("current_price") or 0),
        "current_dollars": float(r.get("current_dollars") or 0),
        "current_pct_nlv": float(r.get("current_pct_nlv") or 0),
        # current_tier is the live cascade tier from the ratchet —
        # distinct from last_signal (which is the latest emission in the
        # log, can be ENTRY for newly-entered positions). The frontend
        # binds the Signal badge to this so SNDK reads GREEN, not ENTRY.
        "current_tier": str(r.get("current_tier") or "GREEN"),
        "tier_pct_nlv": tier,
        "target_dollars": float(r.get("target_dollars") or 0),
        "delta_dollars": float(r.get("delta_dollars") or 0),
        "delta_shares": int(r.get("delta_shares") or 0),
        "unreal_dollars": float(r.get("unreal_dollars") or 0),
        "unreal_pct": float(r.get("unreal_pct") or 0),
        "last_signal": str(r.get("last_signal") or ""),
        "last_signal_date": _iso(r.get("last_signal_date")),
        "last_bar_date": _iso(r.get("last_bar_date")),
        "signal_today": bool(r.get("signal_today")),
        "terminated": bool(r.get("terminated")),
        "phase": int(r.get("phase") or 1),
        "is_action": is_action,
        "early_warn": bool(early_warn),
        "fetch_failed": False,
        "fetch_error": "",
    }


def _sr8_run_monitor(nlv: float, portfolio: str = "CanSlim", refresh: bool = False) -> dict[str, Any]:
    """Pull SR8 positions for `portfolio` from the trades DB, run
    mors.monitor.analyze() per position, and assemble the response
    payload. Errors per-position bucket into fetch_failed rows so a
    single bad ticker doesn't break the page."""
    from mors.monitor import analyze as mors_analyze

    positions = _sr8_load_db_positions(portfolio)
    rows: list[dict[str, Any]] = []
    for pos in positions:
        try:
            r = mors_analyze(pos, nlv, refresh=refresh)
            rows.append(_sr8_enrich(pos, r))
        except Exception as e:
            ticker = str(pos.get("ticker") or "")
            print(f"[sr8_monitor] analyze failed for {ticker}: {e}")
            rows.append(_sr8_fetch_failed_row(pos, f"{type(e).__name__}: {e}"))

    actionable = [r for r in rows if r["is_action"] and not r["fetch_failed"]]
    priced = [r for r in rows if not r["fetch_failed"]]
    # Tier distribution replaces the obsolete 20-cas / 15-cas breakdown.
    # Counts positions per live cascade tier (the ratchet's current state).
    # GREEN(sub-entry) folds into "green" for display parity.
    def _tier_key(t: str) -> str:
        u = (t or "").upper()
        if u in ("GREEN", "GREEN(SUB-ENTRY)"): return "green"
        if u == "QUICK": return "quick"
        if u in ("QUICKSAND", "QS"): return "quicksand"
        if u in ("GD", "TERMINATED"): return "gd"
        return "green"  # defensive default
    tier_counts = {"green": 0, "quick": 0, "quicksand": 0, "gd": 0}
    for r in priced:
        tier_counts[_tier_key(r["current_tier"])] += 1
    at_risk_pct = sum(r["current_pct_nlv"] for r in actionable)
    to_trim_dollars = sum(r["delta_dollars"] for r in actionable)

    return {
        "summary": {
            "total_positions": len(rows),
            "flagged_count": len(actionable),
            "at_risk_pct": round(at_risk_pct, 2),
            "to_trim_dollars": round(to_trim_dollars, 2),
            "tier_breakdown": tier_counts,
        },
        "positions": rows,
        "meta": {
            "fetched_at": _sr8_fetched_at_iso(),
            "nlv": nlv,
            "portfolio": portfolio,
        },
    }


@app.get("/api/analytics/trailing-avg-loss")
@limiter.limit("30/minute")
def trailing_avg_loss(
    request: Request,
    portfolio: str = "CanSlim",
    window_months: int = 12,
):
    """Trailing realized-loss aggregate for the New Entry sizing model.

    New Entry sizes against the trader's *realized* average loss, not the
    hard stop — because the trader exits whole positions fast and the hard
    stop only prices the rare gap scenario. Formula on the client:

        denominator_% = max(4.0%, |avg_loss_pct|)
        position_size_% = risk_unit_% / denominator_%

    The 4% floor is applied CLIENT-side so this endpoint returns the raw
    aggregate. That lets callers distinguish "no data" (sample_size=0,
    pcts=null) from "data present but tighter than floor".

    Aggregation:
        AVG(return_pct)  over CLOSED equity campaigns with return_pct < 0
                         closed within the trailing `window_months`,
                         portfolio-scoped, deleted rows excluded.
        Median via PERCENTILE_CONT(0.5).

    Equity-only (instrument_type = 'STOCK'): options are excluded because
    a bought long call going to zero registers as a -100% return that
    doesn't reflect the trader's actual exit behavior — the New Entry
    model is sizing STOCK entries, and a -99% option loss would skew the
    denominator and shrink stock sizing to zero.

    RLS: the portfolio name → id lookup and the aggregation SELECT both
    run under the caller's app.user_id (set on the connection by the
    tenant middleware), so cross-tenant reads can't leak.

    Response:
        {
            "portfolio": "CanSlim",
            "window_months": 12,
            "avg_loss_pct":    -4.58,   // negative; null if sample_size=0
            "median_loss_pct": -3.70,   // negative; null if sample_size=0
            "sample_size":     84,
            "as_of":           "2026-07-13"
        }
    """
    from datetime import date as _date
    if window_months <= 0 or window_months > 120:
        raise HTTPException(
            status_code=400,
            detail=f"window_months must be in (0, 120]; got {window_months}",
        )
    try:
        df = db.load_summary(portfolio)
    except Exception as e:
        print(f"[trailing_avg_loss] load_summary failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    as_of = _date.today().isoformat()
    empty_response = {
        "portfolio": portfolio,
        "window_months": window_months,
        "avg_loss_pct": None,
        "median_loss_pct": None,
        "sample_size": 0,
        "as_of": as_of,
    }
    if df is None or df.empty:
        return empty_response

    df = _normalize_trades(df)
    status_col = "status" if "status" in df.columns else "Status"
    inst_col = "instrument_type" if "instrument_type" in df.columns else None
    closed_col = "closed_date" if "closed_date" in df.columns else None
    return_col = "return_pct" if "return_pct" in df.columns else None
    if not (closed_col and return_col and inst_col):
        # Schema older than what the endpoint requires — treat as no data.
        return empty_response

    cutoff = pd.Timestamp(_date.today()) - pd.DateOffset(months=window_months)
    mask = (
        df[status_col].astype(str).str.upper().eq("CLOSED")
        & df[inst_col].astype(str).str.upper().eq("STOCK")
        & pd.to_numeric(df[return_col], errors="coerce").lt(0)
        & pd.to_datetime(df[closed_col], errors="coerce").ge(cutoff)
    )
    losers = df[mask]
    if losers.empty:
        return empty_response

    losses = pd.to_numeric(losers[return_col], errors="coerce").dropna()
    if losses.empty:
        return empty_response

    return {
        "portfolio": portfolio,
        "window_months": window_months,
        "avg_loss_pct": round(float(losses.mean()), 4),
        "median_loss_pct": round(float(losses.median()), 4),
        "sample_size": int(losses.size),
        "as_of": as_of,
    }


@app.get("/api/sr8/monitor")
@limiter.limit("30/minute")
def sr8_monitor(request: Request, nlv: float = Query(..., gt=0), portfolio: str = "CanSlim"):
    """Daily SR8 cascade monitor — per-position hold/trim/exit decisions.

    Wraps mors.monitor.analyze() across every open position in `portfolio`
    that classifies as SR8 — max(live B1 return, stored b1_max) >= 50,
    matching the campaign classifier (see _sr8_load_db_positions).
    Reads weekly-cached prices from mors/data/ (auto-fetched from yfinance
    if missing). Returns a single payload the SR8 Monitor frontend page
    renders directly.

    Query:
      nlv (float, > 0): current Net Liq Value driving cascade math.
        Edits to NLV on the page re-call this endpoint.
      portfolio (str): portfolio name — same convention as /api/trades/open.

    Response: see _sr8_run_monitor(). Stable shape; fetch failures
    bucket per-position into rows with `fetch_failed: true`.
    """
    try:
        return _sr8_run_monitor(nlv, portfolio=portfolio, refresh=False)
    except Exception as e:
        print(f"[sr8_monitor] handler failed: {e}")
        return {"error": str(e)}


@app.post("/api/sr8/refresh")
@limiter.limit("2/minute")
def sr8_refresh(request: Request, body: dict = Body(...)):
    """Force-refresh weekly cached prices from yfinance, then return
    the updated monitor payload. Heavier than GET /api/sr8/monitor —
    rate-limited to 2/min to discourage abuse.

    Body:
      nlv (float, > 0): same as GET /api/sr8/monitor's query param.
      portfolio (str, optional): defaults to "CanSlim".
    """
    try:
        nlv = float(body.get("nlv") or 0)
        portfolio = str(body.get("portfolio") or "CanSlim")
        if nlv <= 0:
            return {"error": "nlv must be a positive number"}
        return _sr8_run_monitor(nlv, portfolio=portfolio, refresh=True)
    except Exception as e:
        print(f"[sr8_refresh] handler failed: {e}")
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
        print(f"[list_cash_transactions_endpoint] handler failed: {e}")
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
        print(f"[create_cash_transaction_endpoint] handler failed: {e}")
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
        print(f"[update_cash_transaction_endpoint] handler failed: {e}")
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
        print(f"[delete_cash_transaction_endpoint] handler failed: {e}")
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
        print(f"[cleanup_auto_events] handler failed: {e}")
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
        print(f"[list_events] handler failed: {e}")
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
        print(f"[audit_trail] handler failed: {e}")
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
        print(f"[cleanup_marketsurge] handler failed: {e}")
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
        print(f"[rebuild_mct_signals] handler failed: {e}")
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
- Sell rules: Capital protection (stop loss), selling into strength, portfolio management, change of character, failed breakout
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
        print(f"[coach_chat] handler failed: {e}")
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
            except Exception:
                pass
        next_seq = (max(seqs) + 1) if seqs else 1
        return {"trade_id": f"{ym}-{next_seq:03d}"}
    except Exception as e:
        print(f"[next_trade_id] handler failed: {e}")
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
        print(f"[drift_scan_endpoint] handler failed: {e}")
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
        print(f"[import_ibkr_trades] handler failed: {e}")
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


# Phase 2 B-1: per-SELL matching-method stamping. log_sell and the
# exercise_option option-leg SELL read the live MATCH_METHOD env var
# at request time and write the resolved value into trades_details.
# Per-call resolution (not module-load) keeps tests cleanly isolated
# via monkeypatch.setenv.
_VALID_MATCH_METHODS = ("LIFO", "HCFO")


def _resolve_match_method() -> str:
    """Read MATCH_METHOD env. Default 'LIFO'. Invalid → ValueError.

    Empty string and unset both resolve to 'LIFO'. Case-insensitive on
    the env value. Whitespace stripped. Any other value raises so a
    typo in deployment config surfaces loudly instead of silently
    landing as NULL.
    """
    val = os.environ.get("MATCH_METHOD", "LIFO").strip().upper()
    if not val:
        return "LIFO"
    if val not in _VALID_MATCH_METHODS:
        raise ValueError(
            f"MATCH_METHOD={val!r} is invalid. Must be one of "
            f"{_VALID_MATCH_METHODS} (or unset for LIFO default)."
        )
    return val


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
        if client_trx_id:
            raise HTTPException(
                status_code=422,
                detail=(
                    "trx_id is server-assigned and cannot be supplied in the "
                    "buy request body. Omit the field; the server will return "
                    "the assigned trx_id in the response."
                ),
            )
        # Strategy (Migration 019). Defaults to CanSlim — matches the DB
        # column DEFAULT and the user's primary strategy. For new buys we
        # validate the value against the active strategies; for scale-ins
        # we ignore the body entirely and inherit from the parent campaign
        # (same defense-in-depth pattern as instrument_type below).
        strategy = (body.get("strategy") or "CanSlim").strip() or "CanSlim"

        # Migration 044 — optional Scale-Out Stops ladder. Shape:
        #   { "legs": [ {"pct": 3, "shares": N1}, {"pct": 5, "shares": N2},
        #               {"pct": 7, "shares": N3} ] }
        # Locked at [3, 5, 7]. sum(leg_shares) must equal `shares` on the
        # request. When absent (single-stop mode) column stays NULL and
        # every legacy read path uses stop_loss as it always has. When
        # present, stop_loss is set to leg 1's price (the first-firing
        # stop) so Trade Journal / Risk Manager / Portfolio Heat keep
        # showing a coherent primary stop until Phase 2 makes them
        # ladder-aware. Restricted to new-campaign buys (B1): scale-ins
        # inherit the parent's plan; a ladder on an add-on has no
        # meaning in the user's workflow.
        raw_ladder = body.get("stop_ladder")
        stop_ladder = None
        if raw_ladder is not None and action_type != "new":
            raise HTTPException(
                status_code=422,
                detail="stop_ladder is only allowed on new-campaign buys (B1). Omit on scale-in.",
            )
        if raw_ladder is not None:
            legs = raw_ladder.get("legs") if isinstance(raw_ladder, dict) else None
            if not isinstance(legs, list) or len(legs) != 3:
                raise HTTPException(status_code=422, detail="stop_ladder.legs must be a list of exactly 3 entries")
            expected_pcts = [3, 5, 7]
            cleaned_legs = []
            leg_shares_sum = 0
            for i, leg in enumerate(legs):
                if not isinstance(leg, dict):
                    raise HTTPException(status_code=422, detail=f"stop_ladder.legs[{i}] must be an object")
                pct = leg.get("pct")
                leg_shares = leg.get("shares")
                if pct != expected_pcts[i]:
                    raise HTTPException(status_code=422, detail=f"stop_ladder.legs[{i}].pct must be {expected_pcts[i]}")
                try:
                    leg_shares_int = int(leg_shares)
                except (TypeError, ValueError):
                    raise HTTPException(status_code=422, detail=f"stop_ladder.legs[{i}].shares must be an integer")
                if leg_shares_int < 0:
                    raise HTTPException(status_code=422, detail=f"stop_ladder.legs[{i}].shares must be >= 0")
                cleaned_legs.append({"pct": pct, "shares": leg_shares_int})
                leg_shares_sum += leg_shares_int
            if leg_shares_sum != int(shares):
                raise HTTPException(
                    status_code=422,
                    detail=f"stop_ladder leg shares sum ({leg_shares_sum}) must equal shares ({int(shares)})",
                )
            stop_ladder = {"legs": cleaned_legs}
            # Primary stop = first-firing leg (−3% by convention).
            stop_loss = round(price * (1 - cleaned_legs[0]["pct"] / 100), 4)

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
        # Migration 044: attach ladder only when caller supplied one so
        # legacy paths keep hitting the "column absent → NULL" branch in
        # _save_detail_row_in_txn.
        if stop_ladder is not None:
            detail_row["Stop_Ladder"] = stop_ladder
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
    except HTTPException:
        # Let FastAPI render the actual HTTP status (e.g. the 422 strict-mode
        # reject above). The catch-all below would otherwise wrap it into a
        # 200 with {"error": str(e)} and drop the status code.
        raise
    except Exception as e:
        print(f"[log_buy] handler failed: {e}")
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
        print(f"[set_trade_grade] handler failed: {e}")
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
        if client_trx_id:
            raise HTTPException(
                status_code=422,
                detail=(
                    "trx_id is server-assigned and cannot be supplied in the "
                    "sell request body. Omit the field; the server will return "
                    "the assigned trx_id in the response."
                ),
            )
        # Phase 2 B-1: stamp the SELL with the current MATCH_METHOD.
        # Resolved once per request so a mid-walk env flip can't desync
        # the row's stamp from the inline LIFO/HCFO walk below.
        match_method = _resolve_match_method()
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
            "Match_Method": match_method,
        }
        detail_id, trx_id = _save_detail_with_unique_trx_id(
            portfolio, trade_id, "S", detail_row, given_trx_id=client_trx_id,
        )

        # Phase 2 B-2: sole summary writer is _recompute_summary_matching.
        # Sell_Rule + Sell_Notes (and Grade) come from the request body and
        # override the preserve-existing-fields loop inside the recompute,
        # which would otherwise carry over the prior SELL's metadata.
        overrides: dict[str, Any] = {"Sell_Rule": rule, "Sell_Notes": notes}
        if grade_raw is not None and str(grade_raw).strip() != "":
            try:
                g = int(grade_raw)
                if 1 <= g <= 5:
                    overrides["Grade"] = g
            except (ValueError, TypeError):
                pass

        try:
            summary = _recompute_summary_matching(
                portfolio, trade_id, ticker, overrides=overrides,
            )
        except Exception as e:
            print(f"[lot_closures] post-SELL recompute failed for {trade_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail=(
                    f"SELL row saved (detail_id={detail_id}, trx_id={trx_id}) "
                    f"but summary recompute failed: {type(e).__name__}: {e}. "
                    f"Trigger a manual recompute via Trade Manager → Edit any "
                    f"transaction → Save."
                ),
            )

        # Audit trail. Pulls realized_pl from the recompute summary so the
        # message reflects the SELL row's incremental contribution (CLOSED
        # trades show full lifetime realized_pl; OPEN trades show the
        # cumulative figure — same value the prior inline path emitted).
        try:
            realized_for_audit = float(summary["Realized_PL"]) if summary else 0.0
            db.log_audit(portfolio, "SELL", trade_id, ticker,
                         f"{trx_id}: {shares} shs @ ${price:.2f} | P&L: ${realized_for_audit:.2f}", username="web")
        except Exception:
            pass

        return {
            "status": "ok",
            "detail_id": detail_id,
            "trx_id": trx_id,
            "realized_pl": float(summary["Realized_PL"]) if summary else 0.0,
            "remaining_shares": float(summary["Shares"]) if (summary and summary["Status"] == "OPEN") else 0.0,
            "is_closed": bool(summary and summary["Status"] == "CLOSED"),
        }
    except HTTPException:
        # Let FastAPI render the actual HTTP status (e.g. the 422 strict-mode
        # reject above). The catch-all below would otherwise wrap it into a
        # 200 with {"error": str(e)} and drop the status code.
        raise
    except Exception as e:
        print(f"[log_sell] handler failed: {e}")
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

            # Phase 2 B-2: shared inventory walker. The per-SELL LIFO/HCFO
            # switch lives inside compute_open_inventory → _walk_inventory;
            # the call site only consumes the post-walk aggregates.
            opt_inventory = compute_open_inventory(opt_txns)
            contracts_held = sum(lot["shares"] for lot in opt_inventory)
            held_cost = sum(lot["shares"] * lot["price"] for lot in opt_inventory)
            if contracts_held <= 0:
                return {"error": "No contracts currently held"}
            weighted_avg_premium = held_cost / contracts_held

            # Derived stock-side numbers (depend on the LIFO walk output above).
            shares_acquired = contracts_held * opt_multiplier
            stock_entry_price = strike + weighted_avg_premium
            stock_total_cost = shares_acquired * stock_entry_price
            opt_total_premium_dollars = held_cost * opt_multiplier  # for audit detail

            # --- 1. Option SELL detail ---
            # Phase 2 B-1: stamp the option-leg SELL with the current method.
            # Stock-leg BUY (below) intentionally NOT stamped — BUY rows stay
            # NULL by convention; per-SELL stamping is the rule, not per-row.
            opt_match_method = _resolve_match_method()
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
                "Match_Method": opt_match_method,
            }
            opt_detail_id = db._save_detail_row_in_txn(cur, portfolio_id, opt_detail_row)

            # --- 2. Recompute option summary + closures ---
            # Re-walk all option details (including the SELL we just inserted)
            # via compute_matching_summary. The pure function returns the LIFO
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
            opt_lifo_result = compute_matching_summary(
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
            stock_lifo_result = compute_matching_summary(
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
        print(f"[exercise_option] handler failed: {e}")
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

        # Strictness: trx_id is server-assigned and immutable after creation.
        # The frontend's edit form makes the field readOnly and echoes the
        # existing value back to the server, so a matching client value is
        # tolerated. Any divergence is a strict 422 — preventing the
        # pre-df6141a renumbering pattern from re-introducing scrambles.
        client_trx_id = str(body.get("trx_id", "") or "").strip()
        if client_trx_id and client_trx_id != existing_trx_id:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"trx_id is server-assigned and immutable. Existing value "
                    f"'{existing_trx_id}' cannot be changed to '{client_trx_id}'."
                ),
            )

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
            err = validate_post_edit_matching(
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
            # trx_id is immutable post-creation — strict-mode check above
            # already rejects mismatched client values; explicitly pin to
            # existing_trx_id so an inadvertent code-path change can't drift.
            "Trx_ID": existing_trx_id,
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
        # The detail-row update already committed at update_detail_row above,
        # so a recompute failure means the row is saved but the summary is
        # stale — raise 500 (mirroring log_sell at L4172-4185) instead of
        # swallowing, so the operator sees the divergence immediately.
        try:
            if effective_trade_id:
                _recompute_summary_matching(portfolio, effective_trade_id, effective_ticker)
        except Exception as e:
            print(f"[edit_transaction] post-edit recompute failed for {effective_trade_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Detail row edit saved (detail_id={detail_id}) but summary "
                    f"recompute failed: {type(e).__name__}: {e}. Trigger a manual "
                    f"recompute via Trade Manager → Edit any transaction → Save."
                ),
            )

        try:
            db.log_audit(portfolio, "EDIT", effective_trade_id, row_dict.get("Trx_ID", ""),
                         f"Transaction {detail_id} edited", username="web")
        except Exception:
            pass

        return {"status": "ok", "detail_id": detail_id}
    except HTTPException:
        # Let FastAPI render the actual HTTP status (e.g. the 422 strict-mode
        # reject above). The catch-all below would otherwise wrap it into a
        # 200 with {"error": str(e)} and drop the status code.
        raise
    except Exception as e:
        print(f"[edit_transaction_endpoint] handler failed: {e}")
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
    cleaned up by _recompute_summary_matching."""
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
            err = validate_post_edit_matching(
                txns_for_trade, int(detail_id),
                "DELETE", 0.0, 0.0, "",
            )
            if err:
                return {"error": err}

        try:
            db.delete_detail_row(portfolio, int(detail_id))
        except ValueError as e:
            print(f"[delete_transaction_endpoint] handler failed: {e}")
            return {"error": str(e)}

        if effective_trade_id:
            try:
                _recompute_summary_matching(portfolio, effective_trade_id, effective_ticker)
            except Exception as e:
                # Summary recompute failure shouldn't roll back the delete —
                # the row is gone; the worst case is a stale summary that
                # the next edit/recompute will heal.
                print(f"[delete_transaction] post-delete recompute failed for {effective_trade_id}: {e}")
                pass

        try:
            db.log_audit(portfolio, "DELETE_TXN", effective_trade_id, "",
                         f"Transaction {detail_id} deleted", username="web")
        except Exception:
            pass

        return {"status": "ok", "detail_id": detail_id}
    except Exception as e:
        print(f"[delete_transaction_endpoint] handler failed: {e}")
        return {"error": str(e)}


def _recompute_summary_matching(
    portfolio: str,
    trade_id: str,
    ticker: str,
    fallback_open_date: str = "",
    overrides: dict[str, Any] | None = None,
) -> dict | None:
    """Recompute a trade campaign's summary from its remaining detail rows
    using the canonical per-SELL matching walker and replace its
    lot_closures rows. If no details remain, deletes the summary and any
    orphan closures and returns None.

    Phase 2 B-2: log_sell is now the sole writer through this helper
    for fresh SELLs (the inline summary save was removed). edit / delete
    paths continue to use this helper as before.

    `overrides`: optional dict whose entries are written into the
    summary_row AFTER the preserve-existing-fields loop, so body-supplied
    Sell_Rule / Sell_Notes / Grade win over the previous SELL's
    persisted values. Other callers omit overrides → no change in
    behavior.

    Returns the summary_row that was written, or None when the trade was
    deleted (no detail rows remain). Existing callers ignore the return
    value; log_sell uses it to build its response payload.
    """
    df_d = db.load_details(portfolio)
    if df_d.empty:
        db.delete_trade(portfolio, trade_id)
        _safe_delete_lot_closures(portfolio, trade_id)
        return None
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
    result = compute_matching_summary(
        txns, trade_id, ticker, fallback_open_date,
        multiplier=multiplier, with_closures=True,
    )
    summary_row, closures = result
    if summary_row is None:
        db.delete_trade(portfolio, trade_id)
        _safe_delete_lot_closures(portfolio, trade_id)
        return None
    summary_row["Instrument_Type"] = instrument_type
    summary_row["Multiplier"] = multiplier
    # Preserve user-entered fields that LIFO doesn't compute. compute_matching_summary
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
    # Phase 2 B-2: overrides run AFTER preservation so log_sell's body-
    # supplied Sell_Rule / Sell_Notes / Grade win over the prior SELL's
    # persisted values. Other callers omit overrides → no-op branch.
    if overrides:
        for k, v in overrides.items():
            summary_row[k] = v
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
    return summary_row


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
            except Exception as e:
                print(f"[delete_by_date] per-detail delete failed for detail_id={did}: {e}")
                pass

        for tid, ticker in ticker_by_tid.items():
            try:
                _recompute_summary_matching(portfolio, tid, ticker)
            except Exception as e:
                print(f"[delete_by_date] post-batch recompute failed for {tid}: {e}")
                pass

        try:
            db.log_audit(portfolio, "DELETE", "", "",
                         f"Deleted {deleted} txn(s) for {day} — affected: {', '.join(ticker_by_tid.keys())}",
                         username="web")
        except Exception:
            pass

        return {"status": "ok", "deleted": deleted, "trade_ids": list(ticker_by_tid.keys())}
    except Exception as e:
        print(f"[delete_transactions_by_date] handler failed: {e}")
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
        print(f"[delete_trade_endpoint] handler failed: {e}")
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
        print(f"[set_trade_manual_price] handler failed: {e}")
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
        print(f"[flag_be_rule] handler failed: {e}")
        return {"error": str(e)}


@app.post("/api/trades/{trade_id}/update-b1-max")
def update_b1_max(trade_id: str, body: dict = Body(...)):
    """Persist the running peak B1 return % for a position (migration 036).

    Drives the Sell Rule column's persistent tier — see SELL_RULES (sr8)
    in frontend/src/lib/trade-rules.ts. Frontend fires this fire-and-
    forget whenever it observes a current B1 return % above the stored
    max; the SQL guard inside db.update_b1_max_return_pct enforces
    monotonic non-decrease so multi-tab races and bad-faith inputs
    can't lower the stored peak.

    Body:
        portfolio:   "CanSlim" | ...  (defaults to CanSlim)
        new_max_pct: <number>          (finite; NaN / Inf rejected)

    Response:
        { "stored_max_pct": float | None, "was_updated": bool }
        - was_updated=true   → SQL wrote a higher value
        - was_updated=false  → stored already >= new; no-op (safe to retry)
        - stored_max_pct     → value AFTER the call, for client sync
    """
    try:
        portfolio = (body.get("portfolio") or "CanSlim").strip() or "CanSlim"
        if not trade_id or not trade_id.strip():
            raise HTTPException(status_code=404, detail="trade_id is required")

        raw = body.get("new_max_pct")
        if raw is None:
            raise HTTPException(status_code=422, detail="new_max_pct is required")
        try:
            new_max_pct = float(raw)
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="new_max_pct must be numeric")
        if not math.isfinite(new_max_pct):
            raise HTTPException(status_code=422, detail="new_max_pct must be finite")

        result = db.update_b1_max_return_pct(portfolio, trade_id.strip(), new_max_pct)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")

        if result.get("was_updated"):
            try:
                db.load_summary.clear()
            except Exception:
                pass

        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"[update_b1_max] handler failed: {e}")
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
            except Exception as e:
                print(f"[stop_update] yfinance fetch failed for {ticker}: {e}")
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
        print(f"[update_trade_stops] handler failed: {e}")
        return {"error": str(e), "trace": traceback.format_exc()}


@app.put("/api/trades/update-ladder")
def update_trade_ladder(body: dict):
    """Replace the Scale-Out Stops ladder on the B1 row of a trade.

    Validates the same shape as log_buy — 3 legs, pcts locked at
    [3, 5, 7], sum(leg_shares) must equal B1's original share count so
    the ladder still accounts for every share on the first buy. Doesn't
    touch stop_loss (leg 1 price is derived from B1 amount × 0.97 which
    is immutable); doesn't touch scale-in rows.
    """
    try:
        portfolio = body.get("portfolio", "CanSlim")
        trade_id = body.get("trade_id", "")
        raw_ladder = body.get("stop_ladder")
        if not trade_id or not isinstance(raw_ladder, dict):
            return {"error": "trade_id and stop_ladder are required"}

        legs = raw_ladder.get("legs")
        if not isinstance(legs, list) or len(legs) != 3:
            raise HTTPException(status_code=422, detail="stop_ladder.legs must be a list of exactly 3 entries")
        expected_pcts = [3, 5, 7]
        cleaned_legs = []
        leg_shares_sum = 0
        for i, leg in enumerate(legs):
            if not isinstance(leg, dict):
                raise HTTPException(status_code=422, detail=f"stop_ladder.legs[{i}] must be an object")
            pct = leg.get("pct")
            leg_shares = leg.get("shares")
            if pct != expected_pcts[i]:
                raise HTTPException(status_code=422, detail=f"stop_ladder.legs[{i}].pct must be {expected_pcts[i]}")
            try:
                leg_shares_int = int(leg_shares)
            except (TypeError, ValueError):
                raise HTTPException(status_code=422, detail=f"stop_ladder.legs[{i}].shares must be an integer")
            if leg_shares_int < 0:
                raise HTTPException(status_code=422, detail=f"stop_ladder.legs[{i}].shares must be >= 0")
            cleaned_legs.append({"pct": pct, "shares": leg_shares_int})
            leg_shares_sum += leg_shares_int

        # Cross-check against B1's share count on this trade.
        df_d = db.load_details(portfolio)
        if df_d is None or df_d.empty:
            return {"error": f"No details found for trade {trade_id}"}
        df_d = _normalize_trades(df_d)
        buys = df_d[(df_d["trade_id"] == trade_id) & (df_d["action"].str.upper() == "BUY")].copy()
        if buys.empty:
            return {"error": f"No BUY rows found for trade {trade_id}"}
        buys["date"] = pd.to_datetime(buys["date"], errors="coerce")
        buys = buys.sort_values("date")
        b1_shares = int(float(buys.iloc[0].get("shares") or 0))
        if leg_shares_sum != b1_shares:
            raise HTTPException(
                status_code=422,
                detail=f"stop_ladder leg shares sum ({leg_shares_sum}) must equal B1 shares ({b1_shares})",
            )

        updated = db.update_trade_ladder(portfolio, trade_id, {"legs": cleaned_legs})
        if updated == 0:
            return {"error": f"B1 row not found for trade {trade_id}"}
        try:
            db.log_audit(portfolio, "LADDER_UPDATE", trade_id, str(buys.iloc[0].get("ticker") or ""),
                         f"Ladder → {cleaned_legs[0]['shares']}/{cleaned_legs[1]['shares']}/{cleaned_legs[2]['shares']}",
                         username="web")
        except Exception:
            pass
        return {"status": "ok", "trade_id": trade_id}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[update_trade_ladder] handler failed: {e}")
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
        print(f"[get_fundamentals] handler failed: {e}")
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
        print(f"[delete_fundamentals] handler failed: {e}")
        return {"error": str(e)}


# ============================================================
# R2 IMAGE ENDPOINTS
# ============================================================
from fastapi import UploadFile, File, Form, BackgroundTasks

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
        print(f"[get_trade_images] handler failed: {e}")
        return {"error": str(e)}


def _run_marketsurge_vision_extract(
    content: bytes,
    filename: str,
    portfolio: str,
    trade_id: str,
    ticker: str,
    image_id: int,
) -> None:
    """Background task: pull fundamentals out of a MarketSurge screenshot
    via Claude Vision and persist them. Runs AFTER the upload response
    returns so the Vision API latency (5-15s typical, sometimes hangs)
    never blocks the user's submit. Failures here only mean fundamentals
    didn't extract — the image itself is already in R2 + DB."""
    try:
        import vision_extract
        extracted = vision_extract.extract_fundamentals(content, filename)
        if extracted:
            db.save_trade_fundamentals(portfolio, trade_id, ticker, extracted, image_id)
            print(f"[Vision] Extracted fundamentals for {ticker} ({trade_id})")
        else:
            # Route to stderr so Railway flags this prominently — silent
            # stdout prints are how the stale-model bug went unnoticed.
            print(f"[Vision] No data extracted for {ticker} ({trade_id})", file=sys.stderr)
    except Exception as ve:
        import traceback
        print(
            f"[Vision] Extraction failed for {ticker} ({trade_id}): {ve}\n"
            f"{traceback.format_exc()}",
            file=sys.stderr,
        )


@app.post("/api/images/upload")
@limiter.limit("5/minute")
async def upload_image(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    portfolio: str = Form("CanSlim"),
    trade_id: str = Form(...),
    ticker: str = Form(...),
    image_type: str = Form(...),
):
    """Upload a trade image to R2 and save metadata to DB.

    For marketsurge images, fundamentals extraction (Claude Vision OCR)
    runs as a FastAPI background task so the upload response returns
    once R2 + DB metadata are saved. The Vision call can take 5-15s on a
    good day and has been observed to hang indefinitely — running it
    inline blocks the Log Buy submit and the user sees "Saving…" until
    they refresh (which kills the in-flight upload). Decoupling means
    the chart is durably saved before we touch Vision at all."""
    if not _is_r2_available():
        return {"error": "R2 storage not configured"}
    try:
        content = await file.read()
        file_like = io.BytesIO(content)
        file_like.name = file.filename or "upload.png"

        # MarketSurge screenshots are saved as 'entry' type so they appear
        # in Entry Charts alongside the user's manually attached charts.
        save_type = "entry" if image_type == "marketsurge" else image_type

        object_key = r2.upload_image(file_like, portfolio, trade_id, ticker, save_type)
        if not object_key:
            return {"error": "Upload to R2 failed"}

        image_id = db.save_trade_image(portfolio, trade_id, ticker, save_type, object_key, file.filename)

        # Schedule Vision OCR off the request path. Response returns now;
        # fundamentals land in trade_fundamentals when Vision finishes.
        if image_type == "marketsurge":
            background_tasks.add_task(
                _run_marketsurge_vision_extract,
                content,
                file.filename or "image.png",
                portfolio,
                trade_id,
                ticker,
                image_id,
            )

        # `fundamentals` is always null in the immediate response — the
        # frontend never relied on the synchronous value (the upload page
        # just acks success). Fundamentals are read from
        # /api/trades/{id}/fundamentals on demand once they're persisted.
        return {"status": "ok", "image_id": image_id, "object_key": object_key, "fundamentals": None}
    except Exception as e:
        print(f"[upload_image] handler failed: {e}")
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
    """Upload an end-of-day snapshot (PNG) to R2 tied to a journal day.

    Phase 7: user-uploaded notes are now first-class captures under
    /api/daily-journals/{id}/captures. Calls with snapshot_type="note"
    are rejected here so legacy frontends can't silently keep writing
    to the old surface. Dashboard / campaign uploads remain accepted —
    those are auto-generated EOD content (cron / IBKR sync paths)."""
    if not _is_r2_available():
        return {"error": "R2 storage not configured"}
    if str(snapshot_type or "").lower() == "note":
        raise HTTPException(
            status_code=410,
            detail={
                "error": "endpoint_retired",
                "message": "User-uploaded notes moved to Daily Captures. "
                           "Use POST /api/daily-journals/{id}/captures.",
            },
        )
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
    except HTTPException:
        raise
    except Exception as e:
        print(f"[upload_eod_snapshot] handler failed: {e}")
        return {"error": str(e)}


@app.get("/api/snapshots/{day}")
def list_eod_snapshots(day: str, portfolio: str = "CanSlim"):
    """List EOD snapshots for a specific day.

    Phase 7: 'eod_note' rows are filtered out — they were migrated to
    daily_journal_captures by migration 032 and now render in the Daily
    Captures section instead. The legacy rows still exist in
    trade_images (no soft-delete column there; see migration 032 for
    rationale) but the endpoint hides them so they don't double-render."""
    try:
        trade_id = f"EOD-{day}"
        images = db.get_trade_images(portfolio, trade_id)
        R2_PUBLIC = (os.environ.get("R2_PUBLIC_URL") or "").rstrip("/")
        if images:
            images = [img for img in images
                      if (img.get("image_type") or "").lower() != "eod_note"]
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
        print(f"[list_eod_snapshots] handler failed: {e}")
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
        print(f"[delete_image] handler failed: {e}")
        return {"error": str(e)}


@app.get("/api/r2/status")
def r2_status():
    """Check R2 availability."""
    return {"available": _is_r2_available()}


# ============================================================
# WEEKLY RETRO SNAPSHOTS (Migration 028 — Phase 4)
# ============================================================
# URL-grouped under /api/weekly-retros/{retro_id}/snapshots even though the
# code lives in the R2 endpoint section here, to share the UploadFile
# imports. Browser fetches bytes directly from R2 via the view_url returned
# in row dicts; backend is not in the serving hot path.

import uuid as _uuid

# Audit §5: 15MB upload cap, image MIME whitelist. Bumped from 5MB
# (Phase 2 housekeeping) ahead of T2-4 mobile detail consumers — iPhone
# HEIC→JPEG conversions can exceed 5MB at high quality, and the larger
# headroom de-risks the mobile upload flow without inviting absurd sizes.
_SNAPSHOT_MAX_BYTES = 15 * 1024 * 1024
_SNAPSHOT_ALLOWED_MIMES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
_SNAPSHOT_MIME_TO_EXT = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
}


@app.post("/api/weekly-retros/{retro_id}/snapshots")
@limiter.limit("10/minute")
async def upload_weekly_retro_snapshot(
    retro_id: int,
    request: Request,
    file: UploadFile = File(...),
    portfolio: str = Form("CanSlim"),
):
    """Upload an image snapshot attached to a weekly retro.

    Rejects:
      - non-image MIME (allow: png/jpeg/gif/webp) → 415
      - size > 15MB → 413
      - retro not owned by caller (RLS miss) → 404
    Bytes upload to R2 via r2.upload_blob; metadata row INSERTed into
    weekly_retro_snapshots. Returns row dict with precomputed view_url.
    """
    if not _is_r2_available():
        return {"error": "R2 storage not configured"}

    mime = (file.content_type or "").lower()
    if mime not in _SNAPSHOT_ALLOWED_MIMES:
        raise HTTPException(
            status_code=415,
            detail={"error": "unsupported_media_type",
                    "allowed": sorted(_SNAPSHOT_ALLOWED_MIMES)},
        )

    content = await file.read()
    if len(content) > _SNAPSHOT_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"error": "file_too_large", "limit_bytes": _SNAPSHOT_MAX_BYTES},
        )

    ext = _SNAPSHOT_MIME_TO_EXT.get(mime, "bin")
    object_key = f"weekly_retros/{retro_id}/{_uuid.uuid4().hex}.{ext}"
    file_like = io.BytesIO(content)
    storage_ref = r2.upload_blob(file_like, object_key, content_type=mime)
    if not storage_ref:
        return {"error": "Upload to R2 failed"}

    row = db.save_weekly_retro_snapshot(
        portfolio, retro_id, storage_ref,
        file_name=file.filename, mime_type=mime,
        file_size_bytes=len(content),
    )
    if row is None:
        # Retro not owned by caller (RLS miss). R2 bytes are orphaned;
        # the future sweep job reclaims by stuffed-key scan.
        raise HTTPException(status_code=404, detail={"error": "retro_not_found"})
    return row


@app.get("/api/weekly-retros/{retro_id}/snapshots")
def list_weekly_retro_snapshots_endpoint(retro_id: int, portfolio: str = Query("CanSlim")):
    """List all live snapshots for a retro, ordered by sort_order then
    created_at. 404 if the retro is missing or not owned by the caller."""
    rows = db.list_weekly_retro_snapshots(portfolio, retro_id)
    if rows is None:
        raise HTTPException(status_code=404, detail={"error": "retro_not_found"})
    return rows


@app.delete("/api/weekly-retros/snapshots/{snapshot_id}")
@limiter.limit("30/minute")
def delete_weekly_retro_snapshot(snapshot_id: int, request: Request):
    """Soft-delete a snapshot. RLS scopes the UPDATE to the current
    tenant — a cross-tenant snapshot_id misses and returns 404 (NOT 403,
    to avoid leaking existence). R2 bytes are NOT deleted synchronously;
    the future cleanup sweep handles tombstone reclamation."""
    ok = db.soft_delete_weekly_retro_snapshot(snapshot_id)
    if not ok:
        raise HTTPException(status_code=404, detail={"error": "snapshot_not_found"})
    return {"deleted": True, "id": snapshot_id}


# ============================================================
# PHASE 4.1 — INLINE THOUGHTS IMAGES
# ============================================================
# Separate from the gallery snapshots: these are <img> tags pasted /
# dragged inline into the Weekly Thoughts editor body. No DB row — the
# editor's HTML already encodes which image exists (as <img src=R2URL>).
# Future cleanup is by R2-prefix scan vs current weekly_thoughts HTML.
#
# Reuses Phase 4's MIME/size constants and r2.upload_blob helper.


@app.post("/api/weekly-retros/{retro_id}/thoughts-images")
@limiter.limit("20/minute")
async def upload_weekly_thoughts_image(
    retro_id: int,
    request: Request,
    file: UploadFile = File(...),
    portfolio: str = Form("CanSlim"),
):
    """Upload an inline image for the Weekly Thoughts editor. No DB row —
    the editor body's HTML is the source of truth for which images exist.
    Returns {"view_url": "..."} so the editor can swap the optimistic
    blob URL for the real R2 URL."""
    if not _is_r2_available():
        return {"error": "R2 storage not configured"}

    mime = (file.content_type or "").lower()
    if mime not in _SNAPSHOT_ALLOWED_MIMES:
        raise HTTPException(
            status_code=415,
            detail={"error": "unsupported_media_type",
                    "allowed": sorted(_SNAPSHOT_ALLOWED_MIMES)},
        )

    content = await file.read()
    if len(content) > _SNAPSHOT_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"error": "file_too_large", "limit_bytes": _SNAPSHOT_MAX_BYTES},
        )

    # Ownership check — fail closed before touching R2 so an unauthorized
    # caller can't burn bucket quota by upload-then-orphan.
    if not db.verify_retro_ownership(portfolio, retro_id):
        raise HTTPException(status_code=404, detail={"error": "retro_not_found"})

    ext = _SNAPSHOT_MIME_TO_EXT.get(mime, "bin")
    # Distinct prefix from gallery snapshots so future R2 cleanup can
    # differentiate inline images (referenced by editor HTML) from
    # gallery rows (referenced by weekly_retro_snapshots).
    object_key = f"weekly_retros/{retro_id}/thoughts/{_uuid.uuid4().hex}.{ext}"
    file_like = io.BytesIO(content)
    storage_ref = r2.upload_blob(file_like, object_key, content_type=mime)
    if not storage_ref:
        return {"error": "Upload to R2 failed"}

    r2_public = (os.environ.get("R2_PUBLIC_URL") or "").rstrip("/")
    view_url = f"{r2_public}/{storage_ref}" if r2_public else storage_ref
    return {"view_url": view_url}


# ============================================================
# DAILY JOURNAL CAPTURES (Migration 031 — Phase 7)
# ============================================================
# Image attachments on daily journal entries. URL-grouped under
# /api/daily-journals/{journal_id}/captures (mirrors the weekly-retros
# snapshot endpoints above). Reuses the _SNAPSHOT_* MIME/size constants
# and r2.upload_blob helper from the weekly-retros block.


@app.post("/api/daily-journals/{journal_id}/captures")
@limiter.limit("10/minute")
async def upload_daily_journal_capture(
    journal_id: int,
    request: Request,
    file: UploadFile = File(...),
    portfolio: str = Form("CanSlim"),
):
    """Upload an image capture attached to a daily journal entry.
    Mirrors POST /api/weekly-retros/{retro_id}/snapshots.

    Rejects:
      - non-image MIME (allow: png/jpeg/gif/webp) → 415
      - size > 15MB → 413
      - journal not owned by caller (RLS miss) → 404
    """
    if not _is_r2_available():
        return {"error": "R2 storage not configured"}

    mime = (file.content_type or "").lower()
    if mime not in _SNAPSHOT_ALLOWED_MIMES:
        raise HTTPException(
            status_code=415,
            detail={"error": "unsupported_media_type",
                    "allowed": sorted(_SNAPSHOT_ALLOWED_MIMES)},
        )

    content = await file.read()
    if len(content) > _SNAPSHOT_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"error": "file_too_large", "limit_bytes": _SNAPSHOT_MAX_BYTES},
        )

    ext = _SNAPSHOT_MIME_TO_EXT.get(mime, "bin")
    object_key = f"daily_journal/{journal_id}/{_uuid.uuid4().hex}.{ext}"
    file_like = io.BytesIO(content)
    storage_ref = r2.upload_blob(file_like, object_key, content_type=mime)
    if not storage_ref:
        return {"error": "Upload to R2 failed"}

    row = db.save_daily_journal_capture(
        portfolio, journal_id, storage_ref,
        file_name=file.filename, mime_type=mime,
        file_size_bytes=len(content),
    )
    if row is None:
        # Journal not owned by caller. R2 bytes orphaned; future sweep
        # reclaims by stuffed-key scan.
        raise HTTPException(status_code=404, detail={"error": "journal_not_found"})
    return row


@app.get("/api/daily-journals/{journal_id}/captures")
def list_daily_journal_captures_endpoint(journal_id: int, portfolio: str = Query("CanSlim")):
    """List all live captures for the journal entry. 404 if the journal
    row is missing or not owned by the caller."""
    rows = db.list_daily_journal_captures(portfolio, journal_id)
    if rows is None:
        raise HTTPException(status_code=404, detail={"error": "journal_not_found"})
    return rows


@app.delete("/api/daily-journals/captures/{capture_id}")
@limiter.limit("30/minute")
def delete_daily_journal_capture(capture_id: int, request: Request):
    """Soft-delete a capture. RLS scopes the UPDATE to the current
    tenant — a cross-tenant capture_id misses and returns 404 (NOT 403,
    to avoid leaking existence)."""
    ok = db.soft_delete_daily_journal_capture(capture_id)
    if not ok:
        raise HTTPException(status_code=404, detail={"error": "capture_not_found"})
    return {"deleted": True, "id": capture_id}


@app.post("/api/daily-journals/{journal_id}/thoughts-images")
@limiter.limit("20/minute")
async def upload_daily_thoughts_image(
    journal_id: int,
    request: Request,
    file: UploadFile = File(...),
    portfolio: str = Form("CanSlim"),
):
    """Upload an inline image for the Daily Thoughts editor. No DB row —
    the editor body's HTML is the source of truth for which images exist
    (parallel to the Phase 4.1 weekly thoughts surface)."""
    if not _is_r2_available():
        return {"error": "R2 storage not configured"}

    mime = (file.content_type or "").lower()
    if mime not in _SNAPSHOT_ALLOWED_MIMES:
        raise HTTPException(
            status_code=415,
            detail={"error": "unsupported_media_type",
                    "allowed": sorted(_SNAPSHOT_ALLOWED_MIMES)},
        )

    content = await file.read()
    if len(content) > _SNAPSHOT_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"error": "file_too_large", "limit_bytes": _SNAPSHOT_MAX_BYTES},
        )

    # Ownership check before R2 — same fail-closed pattern as
    # upload_weekly_thoughts_image.
    if not db.verify_daily_journal_ownership(portfolio, journal_id):
        raise HTTPException(status_code=404, detail={"error": "journal_not_found"})

    ext = _SNAPSHOT_MIME_TO_EXT.get(mime, "bin")
    object_key = f"daily_journal/{journal_id}/thoughts/{_uuid.uuid4().hex}.{ext}"
    file_like = io.BytesIO(content)
    storage_ref = r2.upload_blob(file_like, object_key, content_type=mime)
    if not storage_ref:
        return {"error": "Upload to R2 failed"}

    r2_public = (os.environ.get("R2_PUBLIC_URL") or "").rstrip("/")
    view_url = f"{r2_public}/{storage_ref}" if r2_public else storage_ref
    return {"view_url": view_url}


@app.get("/api/daily-journals/list")
@limiter.limit("60/minute")
def list_daily_journals_rail_endpoint(request: Request, portfolio: str = Query("CanSlim")):
    """Wrapped rail envelope for the Daily Report's NotesRail. Mirrors
    GET /api/weekly-retros/list in shape (items, ytd_stats), but
    semantics-wise each item is a single daily journal entry rather than
    a synthetic week."""
    return db.list_daily_journals_rail(portfolio)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
