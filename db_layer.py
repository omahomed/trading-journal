# db_layer.py - PostgreSQL abstraction layer for trading journal

import json
import math
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from psycopg2.extensions import register_adapter, adapt
import numpy as np
import pandas as pd
import os
import re
import threading
import zlib
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
import time
from decimal import Decimal


# ============================================
# psycopg2 numpy scalar adapter
# ============================================
# psycopg2 has no built-in adapter for numpy scalars. Without this,
# values like np.float64 (returned by DataFrame.iloc[0].get(...) for any
# numeric column) fall through to psycopg2's repr() fallback. Under
# numpy 2.0+ that repr is "np.float64(123.45)" — Postgres parses that
# as schema "np", function "float64"(...) and raises InvalidSchemaName.
#
# The fix: convert numpy scalars to their native Python equivalent via
# .item() and delegate to psycopg2's standard adapter chain for that
# native type. NaN is mapped to NULL (matches the pd.isna() → None
# behaviour the per-row _clean helpers in this file were already doing).
def _adapt_numpy_scalar(val):
    py_val = val.item()
    if isinstance(py_val, float) and math.isnan(py_val):
        return adapt(None)
    return adapt(py_val)


for _np_type in (np.float64, np.float32, np.float16,
                 np.int64, np.int32, np.int16, np.int8,
                 np.uint64, np.uint32, np.uint16, np.uint8,
                 np.bool_):
    register_adapter(_np_type, _adapt_numpy_scalar)


# ============================================
# TTL CACHE (Streamlit-free replacement for @st.cache_data)
# ============================================
# Tiny in-process memoizer that mirrors the API db_layer was using:
#   @ttl_cache(ttl=30)
#   def f(...): ...
#   f.clear()                # invalidate everything
# Accepts arbitrary kwargs (e.g. show_spinner) and ignores any it doesn't
# recognise, so we can drop it in over the old `@ttl_cache(ttl=N,
# show_spinner=False)` calls without touching them. Cache keys are the
# argument tuple — all our cached functions take hashable scalars.
class _TTLCache:
    def __init__(self, func, ttl):
        self.func = func
        self.ttl = ttl
        self.cache: dict[tuple, tuple] = {}  # key -> (value, expires_at)
        self.lock = threading.Lock()
        wraps(func)(self)

    def __call__(self, *args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        now = time.time()
        with self.lock:
            hit = self.cache.get(key)
            if hit is not None and hit[1] > now:
                return hit[0]
        value = self.func(*args, **kwargs)
        with self.lock:
            self.cache[key] = (value, now + self.ttl)
        return value

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()


def ttl_cache(ttl: float, **_unused):
    def decorator(func):
        return _TTLCache(func, ttl=ttl)
    return decorator


# ============================================
# VALUE SANITIZATION
# ============================================

def clean_text_value(val) -> str | None:
    """Normalize user-entered text values before persisting.

    Returns None for None, pandas/numpy NaN, empty/whitespace strings, and
    the literal sentinels 'nan'/'none'/'null' (case-insensitive, trimmed).
    Otherwise returns the stripped str(val).

    This is the canonical conversion at the DataFrame->dict boundary for
    text columns in trades_summary, trades_details, trading_journal.
    Without it, str(np.nan) writes the literal string 'nan' to the DB.
    """
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    if not s:
        return None
    if s.lower() in ('nan', 'none', 'null'):
        return None
    return s


# ============================================
# TENANT CONTEXT
# ============================================
# Row-level security in Postgres filters every query by user_id = the session
# variable `app.user_id`. We carry the current request's user_id in a
# ContextVar, then SET it on every new DB connection below. FastAPI middleware
# in api/main.py populates this from the verified JWT. If unset (e.g. CLI
# scripts, legacy Streamlit paths), connections run with no user_id and RLS
# makes every tenant table return zero rows — safe failure mode.
current_user_id: ContextVar[str | None] = ContextVar("current_user_id", default=None)

# ============================================
# CONNECTION CONFIGURATION
# ============================================
def get_db_config():
    """
    Load database configuration from environment or defaults.
    Priority:
      1. Streamlit secrets (for Streamlit Cloud)
      2. Environment variables
      3. Local defaults (for development)
    """
    # Check environment variable first (Railway, Docker, etc.)
    if os.getenv('DATABASE_URL'):
        return {'dsn': os.getenv('DATABASE_URL')}

    # Local development defaults
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'trading_journal'),
        'user': os.getenv('DB_USER', os.getenv('USER', 'postgres')),
        'password': os.getenv('DB_PASSWORD', '')
    }

@contextmanager
def get_db_connection(max_retries=3, retry_delay=1):
    """
    Context manager for database connections with retry logic.
    Ensures connections are properly closed and auto-recovers from transient errors.

    Args:
        max_retries: Maximum number of connection attempts (default: 3)
        retry_delay: Seconds to wait between retries (default: 1, exponential backoff)
    """
    config = get_db_config()
    conn = None
    last_error = None

    for attempt in range(max_retries):
        try:
            if 'dsn' in config:
                conn = psycopg2.connect(config['dsn'])
            else:
                conn = psycopg2.connect(**config)

            # Apply tenant context to this connection before anything reads.
            # SET (not SET LOCAL) is session-scoped, so it survives commits and
            # rollbacks for the lifetime of the connection.
            # Then SET ROLE app_runtime to drop BYPASSRLS so RLS actually
            # enforces. Migrations run as neondb_owner and skip both SETs.
            uid = current_user_id.get()
            if uid:
                with conn.cursor() as _cur:
                    _cur.execute("SET app.user_id = %s", (uid,))
                    _cur.execute("SET ROLE app_runtime")
                conn.commit()

            yield conn
            return  # Success, exit retry loop
        except psycopg2.OperationalError as e:
            last_error = e
            if attempt < max_retries - 1:  # Not last attempt
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"Database connection failed (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                # Last attempt failed, log and re-raise
                print(f"Database connection error after {max_retries} attempts: {e}")
                raise
        finally:
            if conn:
                conn.close()


# ============================================
# INPUT VALIDATION
# ============================================
def validate_trade_id(trade_id):
    """Validate and sanitize trade ID."""
    if not trade_id:
        raise ValueError("Trade_ID cannot be empty")
    trade_id_str = str(trade_id).strip()
    if len(trade_id_str) > 50:
        raise ValueError(f"Trade_ID too long (max 50 chars): {trade_id_str}")
    if len(trade_id_str) == 0:
        raise ValueError("Trade_ID cannot be empty after trimming")
    return trade_id_str


def validate_shares(shares):
    """Validate share quantity."""
    try:
        shares_num = float(shares)
        if shares_num <= 0:
            raise ValueError(f"Shares must be positive: {shares}")
        if shares_num > 1_000_000:
            raise ValueError(f"Shares seems unreasonably large: {shares_num}")
        return shares_num
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid shares value: {shares}") from e


def validate_price(price, field_name="Price"):
    """Validate price/amount."""
    try:
        price_num = float(price)
        if price_num < 0:
            raise ValueError(f"{field_name} cannot be negative: {price}")
        if price_num > 1_000_000:
            raise ValueError(f"{field_name} seems unreasonably large: {price_num}")
        return price_num
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid {field_name}: {price}") from e


def validate_portfolio_name(portfolio_name):
    """Validate portfolio name."""
    if not portfolio_name:
        raise ValueError("Portfolio name cannot be empty")
    portfolio_str = str(portfolio_name).strip()
    if len(portfolio_str) > 50:
        raise ValueError(f"Portfolio name too long (max 50 chars): {portfolio_str}")
    return portfolio_str


# ============================================
# TRANSACTION HELPERS
# ============================================
@contextmanager
def atomic_transaction():
    """
    Context manager for atomic database transactions.
    Ensures all-or-nothing: commits if successful, rolls back if error.

    Usage:
        with atomic_transaction() as (conn, cur):
            cur.execute("INSERT ...")
            cur.execute("UPDATE ...")
            # Commits automatically on success
            # Rolls back automatically on exception
    """
    with get_db_connection() as conn:
        cur = conn.cursor()
        try:
            yield conn, cur
            conn.commit()  # Commit if no exceptions
        except Exception as e:
            conn.rollback()  # Rollback on any error
            print(f"Transaction rolled back due to error: {e}")
            raise
        finally:
            cur.close()


# ============================================
# CORE READ OPERATIONS (Replace load_data)
# ============================================
@ttl_cache(ttl=30, show_spinner=False)  # Cache for 30 seconds
def load_summary(portfolio_name, status=None):
    """
    Load trades summary for a portfolio.
    Replaces: load_data(SUMMARY_FILE)

    Args:
        portfolio_name: 'CanSlim', 'TQQQ Strategy', or '457B Plan'
        status: Optional filter ('OPEN', 'CLOSED', or None for all)

    Returns:
        pandas.DataFrame
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Migration-tolerance for the manual_price columns (012). Prod and
            # staging may temporarily run without them if a deploy lands before
            # the migration. Detect once per call so older DBs don't 500 the
            # entire load_summary path.
            try:
                cur.execute(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name = 'trades_summary' "
                    "AND column_name = 'manual_price'"
                )
                has_manual_price = cur.fetchone() is not None
            except Exception:
                has_manual_price = False
            manual_price_select = (
                's.manual_price AS "Manual_Price",\n                    '
                's.manual_price_set_at AS "Manual_Price_Set_At",\n                    '
                if has_manual_price else ''
            )
            # Migration-tolerance for migration 036. Code deploy may
            # briefly precede migration apply; absent the column, fall
            # back to NULL so the SELECT keeps working.
            try:
                cur.execute(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name = 'trades_summary' "
                    "AND column_name = 'b1_max_return_pct'"
                )
                has_b1_max = cur.fetchone() is not None
            except Exception:
                has_b1_max = False
            b1_max_select = (
                's.b1_max_return_pct AS "B1_Max_Return_Pct",'
                if has_b1_max else
                'NULL::numeric AS "B1_Max_Return_Pct",'
            )
            query = f"""
                SELECT
                    s.trade_id AS "Trade_ID",
                    s.ticker AS "Ticker",
                    s.status AS "Status",
                    s.open_date AS "Open_Date",
                    s.closed_date AS "Closed_Date",
                    s.shares AS "Shares",
                    s.avg_entry AS "Avg_Entry",
                    s.avg_exit AS "Avg_Exit",
                    s.total_cost AS "Total_Cost",
                    s.realized_pl AS "Realized_PL",
                    s.unrealized_pl AS "Unrealized_PL",
                    s.return_pct AS "Return_Pct",
                    s.sell_rule AS "Sell_Rule",
                    s.notes AS "Notes",
                    s.amount AS "Amount",
                    s.value AS "Value",
                    s.stop_loss AS "Stop_Loss",
                    s.rule AS "Rule",
                    s.buy_notes AS "Buy_Notes",
                    s.sell_notes AS "Sell_Notes",
                    s.risk_budget AS "Risk_Budget",
                    s.grade AS "Grade",
                    s.instrument_type AS "Instrument_Type",
                    s.multiplier AS "Multiplier",
                    s.strategy AS "Strategy",
                    {b1_max_select}
                    {manual_price_select}s.be_stop_moved_at AS "BE_Stop_Moved_At",
                    s.last_updated AS "Last_Updated",
                    COALESCE(
                        (SELECT d.rule
                         FROM trades_details d
                         WHERE d.trade_id = s.trade_id
                           AND d.portfolio_id = s.portfolio_id
                           AND d.action = 'BUY'
                           AND d.deleted_at IS NULL
                         ORDER BY d.date ASC
                         LIMIT 1),
                        s.rule
                    ) AS "Buy_Rule",
                    (SELECT d.amount
                     FROM trades_details d
                     WHERE d.trade_id = s.trade_id
                       AND d.portfolio_id = s.portfolio_id
                       AND d.action = 'BUY'
                       AND d.deleted_at IS NULL
                     ORDER BY d.date ASC, d.id ASC
                     LIMIT 1) AS "B1_Entry_Price"
                FROM trades_summary s
                JOIN portfolios p ON s.portfolio_id = p.id
                WHERE p.name = %s
                  AND s.deleted_at IS NULL
            """
            params = [portfolio_name]

            if status:
                query += " AND s.status = %s"
                params.append(status)

            query += " ORDER BY s.open_date DESC"

            cur.execute(query, params)
            try:
                columns = [desc[0] for desc in cur.description]
            except (IndexError, TypeError) as col_err:
                import traceback
                error_msg = f"ERROR getting columns in load_summary:\n{traceback.format_exc()}\ncur.description = {cur.description}"
                print(error_msg)
                with open('/tmp/db_layer_error.log', 'a') as f:
                    f.write(f"\n{'='*60}\n{error_msg}\n")
                raise
            rows = cur.fetchall()
            df = pd.DataFrame(rows, columns=columns)

            # Convert Decimal columns to float (PostgreSQL returns Decimal for NUMERIC columns)
            if not df.empty:
                from decimal import Decimal
                for col in df.columns:
                    # Check if any non-null value in the column is a Decimal
                    sample = df[col].dropna()
                    if len(sample) > 0 and isinstance(sample.iloc[0], Decimal):
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # Convert timestamps to match CSV format
                for col in ['Open_Date', 'Closed_Date']:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')

            return df


def load_all_trade_ids_for_month(portfolio_name, ym):
    """Return every trade_id in trades_summary for the given portfolio whose
    Trade_ID starts with the YYYYMM prefix `ym`, INCLUDING soft-deleted rows.

    This intentionally bypasses the deleted_at filter that load_summary
    applies. Used by api/main.py:next_trade_id so generated trade_ids are
    computed against the full historical sequence — recycling a soft-deleted
    id silently merges a new campaign into a tombstoned summary row, which
    is exactly the failure mode this helper exists to prevent.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.trade_id
                FROM trades_summary s
                JOIN portfolios p ON s.portfolio_id = p.id
                WHERE p.name = %s
                  AND s.trade_id LIKE %s
                """,
                (portfolio_name, f"{ym}-%"),
            )
            return [r[0] for r in cur.fetchall()]


@ttl_cache(ttl=30, show_spinner=False)  # Cache for 30 seconds
def load_details(portfolio_name, trade_id=None):
    """
    Load transaction details for a portfolio.
    Replaces: load_data(DETAILS_FILE)

    Args:
        portfolio_name: Portfolio name
        trade_id: Optional filter for specific trade

    Returns:
        pandas.DataFrame
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT
                    d.id AS "_DB_ID",
                    d.trade_id AS "Trade_ID",
                    d.ticker AS "Ticker",
                    d.action AS "Action",
                    d.date AS "Date",
                    d.shares AS "Shares",
                    d.amount AS "Amount",
                    d.value AS "Value",
                    d.rule AS "Rule",
                    d.notes AS "Notes",
                    d.realized_pl AS "Realized_PL",
                    d.stop_loss AS "Stop_Loss",
                    d.trx_id AS "Trx_ID",
                    d.exec_grade AS "Exec_Grade",
                    d.behavior_tag AS "Behavior_Tag",
                    d.retro_notes AS "Retro_Notes",
                    d.instrument_type AS "Instrument_Type",
                    d.multiplier AS "Multiplier",
                    d.match_method AS "Match_Method",
                    d.stop_ladder AS "Stop_Ladder"
                FROM trades_details d
                JOIN portfolios p ON d.portfolio_id = p.id
                WHERE p.name = %s
                  AND d.deleted_at IS NULL
            """
            params = [portfolio_name]

            if trade_id:
                query += " AND d.trade_id = %s"
                params.append(trade_id)

            query += " ORDER BY d.date, d.action, d.id"

            cur.execute(query, params)
            try:
                columns = [desc[0] for desc in cur.description]
            except (IndexError, TypeError) as col_err:
                import traceback
                error_msg = f"ERROR getting columns in load_details:\n{traceback.format_exc()}\ncur.description = {cur.description}"
                print(error_msg)
                with open('/tmp/db_layer_error.log', 'a') as f:
                    f.write(f"\n{'='*60}\n{error_msg}\n")
                raise
            rows = cur.fetchall()
            df = pd.DataFrame(rows, columns=columns)

            # Convert Decimal columns to float (PostgreSQL returns Decimal for NUMERIC columns)
            if not df.empty:
                from decimal import Decimal
                for col in df.columns:
                    # Check if any non-null value in the column is a Decimal
                    sample = df[col].dropna()
                    if len(sample) > 0 and isinstance(sample.iloc[0], Decimal):
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

            return df


@ttl_cache(ttl=30, show_spinner=False)  # Cache for 30 seconds
def load_journal(portfolio_name, start_date=None, end_date=None):
    """
    Load trading journal entries.
    Replaces: load_data(JOURNAL_FILE)

    Args:
        portfolio_name: Portfolio name
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        pandas.DataFrame
    """
    try:
        with get_db_connection() as conn:
            # Use cursor to execute query with parameters, then pandas to read results
            with conn.cursor() as cur:
                # Detect whether market_cycle column exists so we can include it
                # in the SELECT without failing on older DBs that haven't run the
                # add_market_cycle_column.sql migration yet.
                try:
                    cur.execute(
                        "SELECT 1 FROM information_schema.columns "
                        "WHERE table_name = 'trading_journal' AND column_name = 'market_cycle'"
                    )
                    has_market_cycle = cur.fetchone() is not None
                except Exception:
                    has_market_cycle = False

                # Same gate for mct_display_day_num (migration 015) — newer DBs
                # have it, older ones won't and SELECTing it would 500 the call.
                try:
                    cur.execute(
                        "SELECT 1 FROM information_schema.columns "
                        "WHERE table_name = 'trading_journal' AND column_name = 'mct_display_day_num'"
                    )
                    has_mct_day_num = cur.fetchone() is not None
                except Exception:
                    has_mct_day_num = False

                # Same gate for trend_count (migration 043).
                try:
                    cur.execute(
                        "SELECT 1 FROM information_schema.columns "
                        "WHERE table_name = 'trading_journal' AND column_name = 'trend_count'"
                    )
                    has_trend_count = cur.fetchone() is not None
                except Exception:
                    has_trend_count = False

                cycle_select = 'j.market_cycle AS "Market Cycle",\n                        ' if has_market_cycle else ''
                # Aliased as snake_case directly so normalize_journal_columns
                # leaves it alone (the rename map only handles Pascal-case keys).
                day_num_select = 'j.mct_display_day_num AS "mct_display_day_num",\n                        ' if has_mct_day_num else ''
                trend_count_select = 'j.trend_count AS "trend_count",\n                        ' if has_trend_count else ''
                # Phase 7 daily_thoughts column (migration 031). Detection-gated
                # so a DB still on pre-031 doesn't 500 the journal_history call.
                try:
                    cur.execute(
                        "SELECT 1 FROM information_schema.columns "
                        "WHERE table_name = 'trading_journal' AND column_name = 'daily_thoughts'"
                    )
                    has_daily_thoughts = cur.fetchone() is not None
                except Exception:
                    has_daily_thoughts = False
                daily_thoughts_select = 'j.daily_thoughts AS "daily_thoughts",\n                        ' if has_daily_thoughts else ''
                # j.id is required by the Phase 7 rail/tag/capture mounts in
                # daily-report-card.tsx (TagPicker.entity_id, capture parent
                # FK, NotesRail row id). Snake-case alias passes through
                # normalize_journal_columns untouched.
                query = f"""
                    SELECT
                        j.id AS "id",
                        j.day AS "Day",
                        j.status AS "Status",
                        j.market_window AS "Market Window",
                        {cycle_select}{day_num_select}{trend_count_select}{daily_thoughts_select}j.above_21ema AS "> 21e",
                        j.cash_change AS "Cash -/+",
                        j.beg_nlv AS "Beg NLV",
                        j.end_nlv AS "End NLV",
                        j.daily_dollar_change AS "Daily $ Change",
                        j.daily_pct_change AS "Daily % Change",
                        j.pct_invested AS "% Invested",
                        j.spy AS "SPY",
                        j.nasdaq AS "Nasdaq",
                        j.market_notes AS "Market_Notes",
                        j.market_action AS "Market_Action",
                        j.score AS "Score",
                        j.highlights AS "Highlights",
                        j.lowlights AS "Lowlights",
                        j.mistakes AS "Mistakes",
                        j.top_lesson AS "Top_Lesson",
                        j.nlv_source AS "nlv_source",
                        j.holdings_source AS "holdings_source"
                    FROM trading_journal j
                    JOIN portfolios p ON j.portfolio_id = p.id
                    WHERE p.name = %s
                """
                # Build WHERE clause with safe string substitution
                # Note: Using direct substitution instead of params due to psycopg2/DSN compatibility issue
                # Safe because portfolio_name is from controlled selectbox with known values
                where_parts = [f"p.name = '{portfolio_name}'", "j.deleted_at IS NULL"]

                if start_date:
                    where_parts.append(f"j.day >= '{start_date}'")

                if end_date:
                    where_parts.append(f"j.day <= '{end_date}'")

                query = query.replace("WHERE p.name = %s", "WHERE " + " AND ".join(where_parts))
                query += " ORDER BY j.day DESC"

                # Execute query
                cur.execute(query)

                # Get column names
                try:
                    columns = [desc[0] for desc in cur.description]
                except (IndexError, TypeError) as col_err:
                    import traceback
                    error_msg = f"ERROR getting columns in load_journal:\n{traceback.format_exc()}\ncur.description = {cur.description}"
                    print(error_msg)
                    with open('/tmp/db_layer_error.log', 'a') as f:
                        f.write(f"\n{'='*60}\n{error_msg}\n")
                    raise

                # Fetch all rows
                rows = cur.fetchall()

                # Create DataFrame
                df = pd.DataFrame(rows, columns=columns)

                # Convert Decimal columns to float (PostgreSQL returns Decimal for NUMERIC columns)
                if not df.empty:
                    from decimal import Decimal
                    try:
                        for col in df.columns:
                            # Check if any non-null value in the column is a Decimal
                            sample = df[col].dropna()
                            if len(sample) > 0 and isinstance(sample.iloc[0], Decimal):
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                    except Exception as decimal_err:
                        import traceback
                        error_msg = f"ERROR in load_journal Decimal conversion for column '{col}':\n{traceback.format_exc()}"
                        print(error_msg)
                        with open('/tmp/db_layer_error.log', 'a') as f:
                            f.write(f"\n{'='*60}\n{error_msg}\n")
                        # Continue anyway - column might be usable as-is

                    df['Day'] = pd.to_datetime(df['Day'], errors='coerce')

                    # Add Portfolio_Heat, SPY_ATR, Nasdaq_ATR if columns exist in DB
                    if 'Portfolio_Heat' not in df.columns:
                        try:
                            heat_query = f"""
                                SELECT j.day, j.portfolio_heat, j.spy_atr, j.nasdaq_atr
                                FROM trading_journal j
                                JOIN portfolios p ON j.portfolio_id = p.id
                                WHERE p.name = '{portfolio_name}'
                            """
                            cur.execute(heat_query)
                            heat_rows = cur.fetchall()
                            if heat_rows:
                                heat_df = pd.DataFrame(heat_rows, columns=['Day', 'Portfolio_Heat', 'SPY_ATR', 'Nasdaq_ATR'])
                                heat_df['Day'] = pd.to_datetime(heat_df['Day'], errors='coerce')
                                for _col in ['Portfolio_Heat', 'SPY_ATR', 'Nasdaq_ATR']:
                                    heat_df[_col] = pd.to_numeric(heat_df[_col], errors='coerce').fillna(0)
                                df = df.merge(heat_df, on='Day', how='left')
                                for _col in ['Portfolio_Heat', 'SPY_ATR', 'Nasdaq_ATR']:
                                    df[_col] = df[_col].fillna(0)
                            else:
                                df['Portfolio_Heat'] = 0.0
                                df['SPY_ATR'] = 0.0
                                df['Nasdaq_ATR'] = 0.0
                        except:
                            df['Portfolio_Heat'] = 0.0
                            df['SPY_ATR'] = 0.0
                            df['Nasdaq_ATR'] = 0.0

                return df
    except Exception as e:
        import traceback
        error_msg = f"ERROR in load_journal:\nPortfolio: {portfolio_name}\nException: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(error_msg)
        with open('/tmp/db_journal_error.log', 'w') as f:
            f.write(error_msg)
        raise  # Re-raise so load_data() catches it and falls back to CSV


def set_manual_price(portfolio_name, trade_id, manual_price):
    """Set or clear the manual price override on an open trades_summary row.

    manual_price=None clears the override (UI uses this when the user blanks
    the cell). manual_price_set_at is bumped on every write so the UI can
    surface staleness.

    Returns the updated {trade_id, manual_price, manual_price_set_at} dict,
    or None when the row isn't found / not owned by the current user (RLS).
    Quietly no-ops when the manual_price columns don't exist yet — the same
    migration-tolerance applied to load_summary so a code deploy that lands
    before migration 012 doesn't 500 the endpoint.
    """
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = 'trades_summary' "
                "AND column_name = 'manual_price'"
            )
            if cur.fetchone() is None:
                return None

            cur.execute(
                "SELECT id FROM portfolios WHERE name = %s", (portfolio_name,)
            )
            row = cur.fetchone()
            if row is None:
                return None
            portfolio_id = row["id"]

            if manual_price is None:
                cur.execute(
                    "UPDATE trades_summary "
                    "SET manual_price = NULL, manual_price_set_at = NULL "
                    "WHERE portfolio_id = %s AND trade_id = %s "
                    "AND deleted_at IS NULL "
                    "RETURNING trade_id, manual_price, manual_price_set_at",
                    (portfolio_id, trade_id),
                )
            else:
                cur.execute(
                    "UPDATE trades_summary "
                    "SET manual_price = %s, manual_price_set_at = NOW() "
                    "WHERE portfolio_id = %s AND trade_id = %s "
                    "AND deleted_at IS NULL "
                    "RETURNING trade_id, manual_price, manual_price_set_at",
                    (manual_price, portfolio_id, trade_id),
                )
            updated = cur.fetchone()
            conn.commit()
            return dict(updated) if updated else None


def update_b1_max_return_pct(portfolio_name, trade_id, new_max_pct):
    """Idempotent guard for the persistent Sell Rule tier (migration 036).

    UPDATEs trades_summary.b1_max_return_pct only if the stored value is
    NULL or strictly less than new_max_pct. Auto-promote on observation;
    never auto-demote — that semantic lives in the SQL itself so multi-tab
    races and bad-faith input can't lower the stored peak.

    Returns:
      {"stored_max_pct": float | None, "was_updated": bool}
        — stored_max_pct is the value AFTER the (no-)update, so callers
          can sync their local state without a follow-up read.
        — was_updated=False on equal-or-lower input (no SQL write).
      None when the trade_id isn't found in the given portfolio.

    Migration-tolerance: if the b1_max_return_pct column doesn't exist
    yet (deploy raced migration apply), returns
    {"stored_max_pct": None, "was_updated": False} so callers degrade
    gracefully instead of 500ing.
    """
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = 'trades_summary' "
                "AND column_name = 'b1_max_return_pct'"
            )
            if cur.fetchone() is None:
                return {"stored_max_pct": None, "was_updated": False}

            cur.execute(
                "SELECT id FROM portfolios WHERE name = %s", (portfolio_name,)
            )
            row = cur.fetchone()
            if row is None:
                return None
            portfolio_id = row["id"]

            # Conditional UPDATE in a single statement. The WHERE clause
            # encodes the idempotent guard: write only when the new value
            # exceeds the stored one (or stored is NULL). RETURNING tells
            # us whether a write happened.
            cur.execute(
                "UPDATE trades_summary "
                "SET b1_max_return_pct = %s "
                "WHERE portfolio_id = %s AND trade_id = %s "
                "  AND deleted_at IS NULL "
                "  AND (b1_max_return_pct IS NULL OR b1_max_return_pct < %s) "
                "RETURNING b1_max_return_pct",
                (new_max_pct, portfolio_id, trade_id, new_max_pct),
            )
            updated = cur.fetchone()
            conn.commit()
            if updated is not None:
                return {
                    "stored_max_pct": float(updated["b1_max_return_pct"]),
                    "was_updated": True,
                }

            # No write — either no such trade, or stored already >= new.
            # Read back the current value to distinguish.
            cur.execute(
                "SELECT b1_max_return_pct FROM trades_summary "
                "WHERE portfolio_id = %s AND trade_id = %s "
                "  AND deleted_at IS NULL",
                (portfolio_id, trade_id),
            )
            row = cur.fetchone()
            if row is None:
                return None
            stored = row["b1_max_return_pct"]
            return {
                "stored_max_pct": float(stored) if stored is not None else None,
                "was_updated": False,
            }


# ============================================
# CORE WRITE OPERATIONS (Replace secure_save)
# ============================================
# ============================================
# trades_summary UPDATE column whitelist (Commit 6)
# ============================================
# Maps PascalCase dict keys to snake_case DB columns. save_summary_row's
# UPDATE path binds ONLY the columns whose keys are present in row_dict —
# omitted keys leave the existing DB value untouched (partial-dict-safe).
# This map documents the legitimate UPDATE surface; new columns require a
# deliberate edit here before they can be written via save_summary_row.
#
# Trade_ID is the lookup key, never updated via this map.
# deleted_at, created_at, last_updated are managed by the schema/triggers.
_TRADES_SUMMARY_UPDATE_COLUMNS = {
    "Ticker":          "ticker",
    "Status":          "status",
    "Open_Date":       "open_date",
    "Closed_Date":     "closed_date",
    "Shares":          "shares",
    "Avg_Entry":       "avg_entry",
    "Avg_Exit":        "avg_exit",
    "Total_Cost":      "total_cost",
    "Realized_PL":     "realized_pl",
    "Unrealized_PL":   "unrealized_pl",
    "Return_Pct":      "return_pct",
    "Sell_Rule":       "sell_rule",
    "Notes":           "notes",
    "Stop_Loss":       "stop_loss",
    "Rule":            "rule",
    "Buy_Notes":       "buy_notes",
    "Sell_Notes":      "sell_notes",
    "Risk_Budget":     "risk_budget",
    "Grade":           "grade",
    "Instrument_Type": "instrument_type",
    "Multiplier":      "multiplier",
    "Strategy":        "strategy",
}

# Columns added in newer migrations (013+); removed from the working dict
# on legacy-schema fallback so a DB that hasn't run those migrations can
# still UPDATE successfully via the existing try/except retry pattern.
_TRADES_SUMMARY_LEGACY_EXCLUDED = frozenset((
    "Grade", "Instrument_Type", "Multiplier", "Strategy",
))


def _build_summary_update_set_clauses(row_dict):
    """Build (set_clauses, params_list, unknown_keys) for save_summary_row's
    UPDATE path. Only columns whose PascalCase keys appear in row_dict are
    bound — omitted keys leave the existing DB value untouched.

    Honors explicit None as a deliberate "set to NULL". Distinguishes
    via key-presence check, not value check.

    Grade gets special 1-5 validation (returns None for invalid values).
    Other columns pass through unchanged (clean_value already ran upstream
    on every dict value).

    Returns:
        set_clauses: list of "col = %s" strings
        params_list: list of bound values matching set_clauses order
        unknown_keys: set of dict keys that weren't in the column whitelist
                      (caller decides whether to warn / ignore)
    """
    set_clauses = []
    params_list = []
    for key, col in _TRADES_SUMMARY_UPDATE_COLUMNS.items():
        if key not in row_dict:
            continue
        val = row_dict[key]
        if key == "Grade" and val is not None:
            try:
                g = int(val)
                val = g if 1 <= g <= 5 else None
            except (ValueError, TypeError):
                val = None
        set_clauses.append(f"{col} = %s")
        params_list.append(val)

    unknown_keys = (
        set(row_dict.keys())
        - set(_TRADES_SUMMARY_UPDATE_COLUMNS.keys())
        - {"Trade_ID"}
    )
    return set_clauses, params_list, unknown_keys


def save_summary_row(portfolio_name, row_dict):
    """
    Insert or update a single summary row.
    Replaces: Adding row to df_s and calling secure_save()

    Args:
        portfolio_name: Portfolio name
        row_dict: Dictionary with column values

    Returns:
        int: ID of inserted/updated row
    """
    # Clean NaT/NaN values and numpy types for SQL compatibility
    def clean_value(val):
        if val is None:
            return None
        try:
            if pd.isna(val) or str(val).strip() == 'NaT':
                return None
        except (TypeError, ValueError):
            pass
        # Convert numpy types to native Python types
        if hasattr(val, 'item'):  # np.float64, np.int64, etc.
            return val.item()
        # Defense-in-depth: literal 'nan'/'none'/'null' on string values
        # (mirrors clean_text_value's sentinel check from Commit 1).
        if isinstance(val, str) and val.strip().lower() in ('nan', 'none', 'null'):
            return None
        return val

    # Sanitize all values in row_dict to prevent numpy types reaching psycopg2
    row_dict = {k: clean_value(v) for k, v in row_dict.items()}

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Get portfolio_id
            cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
            result = cur.fetchone()
            if not result:
                raise ValueError(f"Portfolio '{portfolio_name}' not found")
            portfolio_id = result[0]

            # Check if trade exists. Soft-deleted rows are excluded so a
            # recycled trade_id never silently UPDATEs a tombstoned summary —
            # the INSERT branch runs instead, where the schema's
            # unique_trade_per_portfolio constraint surfaces the duplicate
            # as a loud IntegrityError. See fix/trade-id-soft-delete-safety.
            cur.execute(
                "SELECT id FROM trades_summary "
                "WHERE portfolio_id = %s AND trade_id = %s AND deleted_at IS NULL",
                (portfolio_id, row_dict.get('Trade_ID'))
            )
            existing = cur.fetchone()

            # Grade is optional: omitting it from row_dict should leave the
            # existing DB value untouched. Only write when the caller passes
            # 'Grade' explicitly (including None to clear it).
            grade_val = row_dict.get('Grade', '__unset__')
            update_grade = grade_val != '__unset__'
            grade_clean = None
            if update_grade and grade_val is not None:
                try:
                    g = int(grade_val)
                    grade_clean = g if 1 <= g <= 5 else None
                except (ValueError, TypeError):
                    grade_clean = None

            # instrument_type/multiplier (Migration 016) only get written when
            # the caller passes them. Omitting both leaves the existing DB
            # values untouched, which matters for legacy callers that never
            # supply them (status updates, grade-only edits, etc.).
            instrument_type_val = row_dict.get('Instrument_Type')
            multiplier_val = row_dict.get('Multiplier')
            update_instrument = instrument_type_val is not None or multiplier_val is not None

            # strategy (Migration 019) follows the same opt-in pattern as
            # instrument_type — only written when the caller passes it.
            # Legacy paths (status updates, grade-only edits, etc.) leave
            # the existing DB value alone.
            strategy_val = row_dict.get('Strategy')
            update_strategy = strategy_val is not None

            if existing:
                # UPDATE existing trade — whitelist mode (Commit 6).
                # Only columns whose PascalCase keys are present in row_dict
                # are bound; omitted keys leave the existing DB value
                # untouched (partial-dict-safe). Try with newer migrations'
                # columns first; on failure (legacy schema), drop those
                # columns and retry.
                try:
                    set_clauses, params_list, unknown_keys = \
                        _build_summary_update_set_clauses(row_dict)
                    if unknown_keys:
                        print(f"[save_summary_row] ignored unknown columns: {sorted(unknown_keys)}")
                    if not set_clauses:
                        raise ValueError(
                            "save_summary_row: dict has no columns to UPDATE"
                        )
                    update_query = (
                        f"UPDATE trades_summary SET {', '.join(set_clauses)} "
                        "WHERE id = %s RETURNING id"
                    )
                    params_list.append(existing[0])
                    cur.execute(update_query, tuple(params_list))
                except ValueError:
                    # Empty-clause guard above — propagate up rather than
                    # masking with the legacy-fallback retry.
                    raise
                except Exception:
                    # DB missing newer-migration columns (Migration 013+ —
                    # grade, instrument_type, multiplier, strategy) — drop
                    # them from the working dict and retry.
                    conn.rollback()
                    fallback_row = {
                        k: v for k, v in row_dict.items()
                        if k not in _TRADES_SUMMARY_LEGACY_EXCLUDED
                    }
                    set_clauses, params_list, _ = \
                        _build_summary_update_set_clauses(fallback_row)
                    if not set_clauses:
                        raise ValueError(
                            "save_summary_row: dict has no columns to UPDATE "
                            "(legacy fallback)"
                        )
                    update_query = (
                        f"UPDATE trades_summary SET {', '.join(set_clauses)} "
                        "WHERE id = %s RETURNING id"
                    )
                    params_list.append(existing[0])
                    cur.execute(update_query, tuple(params_list))
            else:
                # INSERT new trade — try with grade + instrument_type, fall
                # back without (legacy schema where Migration 016 hasn't run).
                try:
                    insert_cols = [
                        "portfolio_id", "trade_id", "ticker", "status", "open_date", "closed_date",
                        "shares", "avg_entry", "avg_exit", "total_cost", "realized_pl", "unrealized_pl",
                        "return_pct", "sell_rule", "notes", "stop_loss", "rule",
                        "buy_notes", "sell_notes", "risk_budget", "grade",
                    ]
                    insert_vals = [
                        portfolio_id,
                        row_dict.get('Trade_ID'),
                        row_dict.get('Ticker'),
                        row_dict.get('Status', 'OPEN'),
                        clean_value(row_dict.get('Open_Date')),
                        clean_value(row_dict.get('Closed_Date')),
                        row_dict.get('Shares', 0),
                        row_dict.get('Avg_Entry', 0),
                        row_dict.get('Avg_Exit', 0),
                        row_dict.get('Total_Cost', 0),
                        row_dict.get('Realized_PL', 0),
                        row_dict.get('Unrealized_PL', 0),
                        row_dict.get('Return_Pct', 0),
                        row_dict.get('Sell_Rule'),
                        row_dict.get('Notes'),
                        row_dict.get('Stop_Loss'),
                        row_dict.get('Rule'),
                        row_dict.get('Buy_Notes'),
                        row_dict.get('Sell_Notes'),
                        row_dict.get('Risk_Budget', 0),
                        grade_clean,
                    ]
                    if update_instrument:
                        insert_cols += ["instrument_type", "multiplier"]
                        insert_vals += [
                            instrument_type_val or 'STOCK',
                            multiplier_val if multiplier_val is not None else 1,
                        ]
                    if update_strategy:
                        insert_cols.append("strategy")
                        insert_vals.append(strategy_val)
                    placeholders = ", ".join(["%s"] * len(insert_vals))
                    insert_query = (
                        f"INSERT INTO trades_summary ({', '.join(insert_cols)}) "
                        f"VALUES ({placeholders}) RETURNING id"
                    )
                    cur.execute(insert_query, tuple(insert_vals))
                except Exception:
                    conn.rollback()
                    insert_query = """
                        INSERT INTO trades_summary (
                            portfolio_id, trade_id, ticker, status, open_date, closed_date,
                            shares, avg_entry, avg_exit, total_cost, realized_pl, unrealized_pl,
                            return_pct, sell_rule, notes, stop_loss, rule, buy_notes, sell_notes, risk_budget
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        RETURNING id
                    """
                    cur.execute(insert_query, (
                        portfolio_id,
                        row_dict.get('Trade_ID'),
                        row_dict.get('Ticker'),
                        row_dict.get('Status', 'OPEN'),
                        clean_value(row_dict.get('Open_Date')),
                        clean_value(row_dict.get('Closed_Date')),
                        row_dict.get('Shares', 0),
                        row_dict.get('Avg_Entry', 0),
                        row_dict.get('Avg_Exit', 0),
                        row_dict.get('Total_Cost', 0),
                        row_dict.get('Realized_PL', 0),
                        row_dict.get('Unrealized_PL', 0),
                        row_dict.get('Return_Pct', 0),
                        row_dict.get('Sell_Rule'),
                        row_dict.get('Notes'),
                        row_dict.get('Stop_Loss'),
                        row_dict.get('Rule'),
                        row_dict.get('Buy_Notes'),
                        row_dict.get('Sell_Notes'),
                        row_dict.get('Risk_Budget', 0),
                    ))

            row_id = cur.fetchone()[0]
            conn.commit()

            # Clear cache so next load gets fresh data
            load_summary.clear()

            return row_id


def _save_detail_row_in_txn(cur, portfolio_id, row_dict):
    """Insert a trades_details row + emit its cash_transactions ledger row,
    using the caller's cursor. Returns the new detail row id.

    Caller owns the transaction (commit/rollback) and is responsible for
    invalidating the load_details cache after the outer commit. Sanitizes
    numpy-typed values in row_dict so the SQL adapter sees plain Python
    scalars.

    Extracted from save_detail_row so multi-write endpoints (e.g. exercise-
    option) can compose this with other writes inside one atomic_transaction.
    The public save_detail_row wrapper preserves the original single-call
    semantics for existing callers (log_buy, log_sell, IBKR import, etc.).
    """
    def _clean(val):
        if val is None:
            return None
        if hasattr(val, 'item'):
            return val.item()
        try:
            if pd.isna(val):
                return None
        except (TypeError, ValueError):
            pass
        return val
    row_dict = {k: _clean(v) for k, v in row_dict.items()}

    insert_cols = [
        "portfolio_id", "trade_id", "ticker", "action", "date", "shares", "amount", "value",
        "rule", "notes", "realized_pl", "stop_loss", "trx_id",
        "exec_grade", "behavior_tag", "retro_notes",
    ]
    insert_vals = [
        portfolio_id,
        row_dict.get('Trade_ID'),
        row_dict.get('Ticker'),
        row_dict.get('Action'),
        row_dict.get('Date'),
        row_dict.get('Shares'),
        row_dict.get('Amount'),
        row_dict.get('Value'),
        row_dict.get('Rule'),
        row_dict.get('Notes'),
        row_dict.get('Realized_PL', 0),
        row_dict.get('Stop_Loss'),
        row_dict.get('Trx_ID'),
        row_dict.get('Exec_Grade'),
        row_dict.get('Behavior_Tag'),
        row_dict.get('Retro_Notes'),
    ]
    # Migration 016: persist instrument_type + multiplier when caller
    # passes them. Defaults (STOCK / 1) on the column take over when
    # omitted, so legacy callers stay working.
    if 'Instrument_Type' in row_dict or 'Multiplier' in row_dict:
        insert_cols += ["instrument_type", "multiplier"]
        insert_vals += [
            row_dict.get('Instrument_Type') or 'STOCK',
            row_dict.get('Multiplier') if row_dict.get('Multiplier') is not None else 1,
        ]
    # Migration 041 / Phase 2 B-1: per-SELL match_method stamp. Caller
    # (log_sell, exercise_option) sets this for SELL rows. BUY callers
    # and legacy/test callers omit the key → column lands NULL, which
    # the CHECK constraint (LIFO/HCFO/NULL) explicitly allows.
    if 'Match_Method' in row_dict:
        insert_cols += ["match_method"]
        insert_vals += [row_dict.get('Match_Method')]
    # Migration 044: Position Sizer scale-out ladder. Optional 3-leg
    # staged exit. Only BUY rows carry a ladder; scale-in/sell/legacy
    # callers omit the key and the column stays NULL.
    if 'Stop_Ladder' in row_dict:
        ladder = row_dict.get('Stop_Ladder')
        insert_cols += ["stop_ladder"]
        insert_vals += [json.dumps(ladder) if ladder is not None else None]
    placeholders = ", ".join(["%s"] * len(insert_vals))
    insert_query = (
        f"INSERT INTO trades_details ({', '.join(insert_cols)}) "
        f"VALUES ({placeholders}) RETURNING id"
    )
    cur.execute(insert_query, tuple(insert_vals))

    row_id = cur.fetchone()[0]

    # Emit cash_transactions row for the NLV ledger (BUY/SELL only).
    # Runs inside the same transaction so the trade and its cash
    # movement commit together or not at all.
    _emit_trade_cash_tx(
        cur, portfolio_id, row_id,
        action=row_dict.get('Action'),
        date=row_dict.get('Date'),
        value=row_dict.get('Value'),
    )

    return row_id


def save_detail_row(portfolio_name, row_dict):
    """
    Insert a transaction detail row.

    Args:
        portfolio_name: Portfolio name
        row_dict: Dictionary with column values

    Returns:
        int: ID of inserted row
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
            result = cur.fetchone()
            if not result:
                raise ValueError(f"Portfolio '{portfolio_name}' not found")
            portfolio_id = result[0]

            row_id = _save_detail_row_in_txn(cur, portfolio_id, row_dict)

            conn.commit()

            # Clear cache so next load gets fresh data
            load_details.clear()

            return row_id


def update_detail_row(portfolio_name, detail_id, row_dict):
    """
    Update an existing transaction detail row.

    Args:
        portfolio_name: Portfolio name
        detail_id: The id of the trades_details row to update
        row_dict: Dictionary with column values to update

    Returns:
        bool: True if successful
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Try to find portfolio - check both exact match and partial match
            cur.execute("SELECT id, name FROM portfolios WHERE name = %s", (portfolio_name,))
            result = cur.fetchone()

            if not result:
                # Try partial match with name mappings
                name_mappings = {
                    'CanSlim (Main)': ['CanSlim', 'CanSlim (Main)', 'Canslim'],
                    'TQQQ Strategy': ['TQQQ', 'TQQQ Strategy'],
                    '457B Plan': ['457B', '457B Plan']
                }

                for display_name, db_names in name_mappings.items():
                    if portfolio_name in db_names or any(name in portfolio_name for name in db_names):
                        for db_name in db_names:
                            cur.execute("SELECT id, name FROM portfolios WHERE name = %s", (db_name,))
                            result = cur.fetchone()
                            if result:
                                break
                    if result:
                        break

            if not result:
                raise ValueError(f"Portfolio '{portfolio_name}' not found")

            portfolio_id = result[0]

            # Verify the detail row belongs to this portfolio
            cur.execute(
                "SELECT id FROM trades_details WHERE id = %s AND portfolio_id = %s",
                (detail_id, portfolio_id)
            )
            if not cur.fetchone():
                raise ValueError(f"Detail row {detail_id} not found for portfolio '{portfolio_name}'")

            update_query = """
                UPDATE trades_details
                SET trade_id = %s, ticker = %s, action = %s, date = %s,
                    shares = %s, amount = %s, value = %s, rule = %s,
                    notes = %s, stop_loss = %s, trx_id = %s
                WHERE id = %s
            """
            cur.execute(update_query, (
                row_dict.get('Trade_ID'),
                row_dict.get('Ticker'),
                row_dict.get('Action'),
                row_dict.get('Date'),
                row_dict.get('Shares'),
                row_dict.get('Amount'),
                row_dict.get('Value'),
                row_dict.get('Rule'),
                row_dict.get('Notes'),
                row_dict.get('Stop_Loss'),
                row_dict.get('Trx_ID'),
                detail_id
            ))

            # Sync the cash_tx row for this trade: delete the old one and
            # re-emit with the new values. Simpler than UPDATE and handles
            # edge cases (action flipped BUY↔SELL, value changed) uniformly.
            cur.execute(
                "DELETE FROM cash_transactions WHERE trade_detail_id = %s",
                (detail_id,),
            )
            _emit_trade_cash_tx(
                cur, portfolio_id, detail_id,
                action=row_dict.get('Action'),
                date=row_dict.get('Date'),
                value=row_dict.get('Value'),
            )

            conn.commit()

            # Clear cache so next load gets fresh data
            load_details.clear()
            load_summary.clear()

            return True


def _emit_trade_cash_tx(cur, portfolio_id, detail_id, *, action, date, value):
    """Insert the cash_transactions row that mirrors a BUY/SELL trade detail.

    Non-BUY/SELL actions (defensive — current code only writes BUY or SELL)
    produce no ledger row. Zero-value trades are also skipped — they don't
    move cash so there's nothing to record.
    """
    action_upper = str(action or "").upper()
    if action_upper not in ("BUY", "SELL"):
        return
    try:
        value_num = float(value or 0)
    except (TypeError, ValueError):
        return
    if value_num <= 0:
        return
    cash_amount = -value_num if action_upper == "BUY" else value_num
    cur.execute(
        "INSERT INTO cash_transactions "
        "(portfolio_id, date, amount, source, trade_detail_id) "
        "VALUES (%s, %s, %s, %s, %s)",
        (portfolio_id, date, cash_amount, action_upper.lower(), detail_id),
    )


def delete_detail_row(portfolio_name, detail_id):
    """
    Delete a transaction detail row.

    Args:
        portfolio_name: Portfolio name
        detail_id: The id of the trades_details row to delete

    Returns:
        bool: True if successful
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Try to find portfolio - check both exact match and partial match
            cur.execute("SELECT id, name FROM portfolios WHERE name = %s", (portfolio_name,))
            result = cur.fetchone()

            if not result:
                # Try partial match with name mappings
                name_mappings = {
                    'CanSlim (Main)': ['CanSlim', 'CanSlim (Main)', 'Canslim'],
                    'TQQQ Strategy': ['TQQQ', 'TQQQ Strategy'],
                    '457B Plan': ['457B', '457B Plan']
                }

                for display_name, db_names in name_mappings.items():
                    if portfolio_name in db_names or any(name in portfolio_name for name in db_names):
                        for db_name in db_names:
                            cur.execute("SELECT id, name FROM portfolios WHERE name = %s", (db_name,))
                            result = cur.fetchone()
                            if result:
                                break
                    if result:
                        break

            if not result:
                raise ValueError(f"Portfolio '{portfolio_name}' not found")

            portfolio_id = result[0]

            # Soft-delete: stamp deleted_at instead of removing the row, so
            # an accidental delete is reversible by clearing the column.
            # Only stamps live rows — re-deleting a soft-deleted row is a
            # no-op that we treat as "not found" below.
            cur.execute(
                "UPDATE trades_details SET deleted_at = NOW() "
                "WHERE id = %s AND portfolio_id = %s AND deleted_at IS NULL",
                (detail_id, portfolio_id)
            )

            if cur.rowcount == 0:
                raise ValueError(f"Detail row {detail_id} not found for portfolio '{portfolio_name}'")

            # Remove the linked cash_tx row so cash_balance no longer counts
            # this trade. If the trade is ever restored (deleted_at → NULL),
            # the restore flow will need to re-emit the cash_tx — no restore
            # UI exists yet, so this is a forward problem.
            cur.execute(
                "DELETE FROM cash_transactions WHERE trade_detail_id = %s",
                (detail_id,),
            )

            conn.commit()

            # Clear cache
            load_details.clear()
            load_summary.clear()

            return True


def _log_audit_in_txn(cur, portfolio_id, action, trade_id, ticker, details, username='User'):
    """Insert an audit_trail row using the supplied cursor. Caller owns
    the transaction. Used by multi-write endpoints (e.g. exercise-option)
    that need the audit row to commit-or-rollback alongside the trade
    writes it describes."""
    cur.execute(
        """
        INSERT INTO audit_trail (portfolio_id, username, action, trade_id, ticker, details)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (portfolio_id, username, action, trade_id, ticker, details),
    )


def log_audit(portfolio_name, action, trade_id, ticker, details, username='User'):
    """
    Log an audit trail entry.
    Replaces: log_audit_trail()
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
            result = cur.fetchone()
            if not result:
                raise ValueError(f"Portfolio '{portfolio_name}' not found")
            portfolio_id = result[0]

            _log_audit_in_txn(cur, portfolio_id, action, trade_id, ticker, details, username)

            conn.commit()


# ============================================
# TRANSACTION WRAPPER FOR LIFO SYNC
# ============================================
def sync_trade_summary(portfolio_name, trade_id, update_data):
    """
    Update summary table after LIFO calculation.
    Called by update_campaign_summary() after math is done.

    Args:
        portfolio_name: Portfolio name
        trade_id: Trade to update
        update_data: Dict with calculated fields (shares, avg_entry, total_cost, realized_pl, status)
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
            result = cur.fetchone()
            if not result:
                raise ValueError(f"Portfolio '{portfolio_name}' not found")
            portfolio_id = result[0]

            cur.execute("""
                UPDATE trades_summary
                SET shares = %s,
                    avg_entry = %s,
                    total_cost = %s,
                    realized_pl = %s,
                    status = %s
                WHERE portfolio_id = %s AND trade_id = %s
            """, (
                update_data.get('shares', 0),
                update_data.get('avg_entry', 0),
                update_data.get('total_cost', 0),
                update_data.get('realized_pl', 0),
                update_data.get('status', 'OPEN'),
                portfolio_id,
                trade_id
            ))

            conn.commit()

            # Clear cache so next load gets fresh data
            load_summary.clear()


# ============================================
# BULK DELETE OPERATIONS
# ============================================
def delete_trade(portfolio_name, trade_id):
    """
    Soft-delete a trade and all its transactions; hard-delete related images
    and fundamentals so a reused trade_id doesn't resurrect prior artifacts.
    Stamps deleted_at on both the summary row and every detail row; the rows
    stay in the table but become invisible to load_summary / load_details.
    To restore, clear deleted_at on the affected rows.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
            result = cur.fetchone()
            if not result:
                raise ValueError(f"Portfolio '{portfolio_name}' not found")
            portfolio_id = result[0]

            # Soft-delete details first (child rows) then summary.
            cur.execute("""
                UPDATE trades_details SET deleted_at = NOW()
                WHERE portfolio_id = %s AND trade_id = %s
                  AND deleted_at IS NULL
            """, (portfolio_id, trade_id))

            cur.execute("""
                UPDATE trades_summary SET deleted_at = NOW()
                WHERE portfolio_id = %s AND trade_id = %s
                  AND deleted_at IS NULL
            """, (portfolio_id, trade_id))

            # Hard-delete lot_closures for this trade. Mirrors the cleanup
            # the recompute path does when a campaign empties out (see
            # _safe_delete_lot_closures in api/main.py). Joins this txn so
            # closures + summary stamp + details stamp commit atomically.
            # lot_closures uses hard-delete throughout (no deleted_at
            # column per migration 017) — DELETE is the right paradigm.
            cur.execute(
                "DELETE FROM lot_closures "
                "WHERE portfolio_id = %s AND trade_id = %s",
                (portfolio_id, trade_id),
            )

            # Hard-delete images and fundamentals — trade_fundamentals.image_id
            # is ON DELETE SET NULL, so clear fundamentals first or they'd be
            # orphaned with NULL image_id instead of removed. Neither table has
            # a deleted_at column, so a soft-delete pattern doesn't apply here.
            cur.execute("""
                DELETE FROM trade_fundamentals
                WHERE portfolio_id = %s AND trade_id = %s
            """, (portfolio_id, trade_id))

            cur.execute("""
                DELETE FROM trade_images
                WHERE portfolio_id = %s AND trade_id = %s
            """, (portfolio_id, trade_id))

            conn.commit()

            # Clear caches so next load gets fresh data
            load_summary.clear()
            load_details.clear()
            try:
                get_trade_images.clear()
            except Exception:
                pass
            try:
                get_trade_fundamentals.clear()
            except Exception:
                pass


def save_summary_with_closures(portfolio_name, trade_id, summary_row, closures):
    """Atomically persist the summary row and replace the trade's lot_closures.

    All writes — trades_summary UPSERT, lot_closures DELETE, lot_closures
    INSERT — run inside a single atomic_transaction. Either everything
    commits or everything rolls back; no half-committed state where summary
    is updated but closures are stale (or vice versa).

    Why we inline the summary upsert here instead of calling save_summary_row:
    save_summary_row owns its own connection + commit + a conn.rollback() in
    its legacy-schema fallback path. Calling it from inside this transaction
    would either commit early (breaking atomicity) or roll back the closures
    work alongside its own retry. Inlining the recompute-specific upsert is
    the cheaper fix — we accept ~30 lines of SQL duplication in exchange
    for true single-transaction semantics. The duplicated SQL targets the
    modern schema only (grade + instrument_type both present); migration 017
    can't be applied to a DB that's missing those columns anyway.

    Field-write behavior matches the existing save_summary_row recompute
    call site: LIFO-derived columns + Instrument_Type + Multiplier are
    written; Sell_Rule / Notes / Stop_Loss / Rule / Buy_Notes / Sell_Notes /
    Risk_Budget take whatever's in summary_row (None if absent). Grade,
    manual_price, and be_stop_moved_at are NEVER touched here — the
    recompute path doesn't own those columns.

    `closures` is the list returned by compute_matching_summary(..., with_closures=True);
    pass `[]` for an open-only trade (the DELETE still clears any prior closures).

    Returns the summary row id.
    """
    with atomic_transaction() as (_conn, cur):
        cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
        result = cur.fetchone()
        if not result:
            raise ValueError(f"Portfolio '{portfolio_name}' not found")
        portfolio_id = result[0]

        summary_id = _save_summary_with_closures_in_txn(
            cur, portfolio_id, trade_id, summary_row, closures,
        )

    # Cache invalidation only after the transaction has committed.
    load_summary.clear()
    return summary_id


def _save_summary_with_closures_in_txn(cur, portfolio_id, trade_id, summary_row, closures):
    """Inner-transaction body of save_summary_with_closures: UPSERT the
    trades_summary row and replace its lot_closures using the supplied
    cursor. Returns the summary row id.

    Caller owns the transaction (commit/rollback) and is responsible for
    invalidating the load_summary cache after the outer commit.

    Field-write semantics match the public save_summary_with_closures —
    see that function's docstring for the full contract on which columns
    this writes vs. preserves.
    """
    # Defense-in-depth: sanitize user-prose text columns at the writer
    # boundary. Callers may pass raw row.get(...) values (e.g.,
    # exercise_option's option-side path); writer never trusts. Reuses
    # the canonical clean_text_value from Commit 1 — the existing
    # 'nan'/'none'/'null'/NaN sentinels are normalized to None here.
    summary_row = dict(summary_row)  # don't mutate caller's dict
    for col in ('Rule', 'Buy_Notes', 'Sell_Rule', 'Sell_Notes', 'Notes'):
        if col in summary_row:
            summary_row[col] = clean_text_value(summary_row[col])

    trade_id_for_summary = summary_row.get('Trade_ID')

    # --- 1. Summary upsert ---
    cur.execute(
        "SELECT id FROM trades_summary WHERE portfolio_id = %s AND trade_id = %s",
        (portfolio_id, trade_id_for_summary),
    )
    existing = cur.fetchone()

    if existing:
        cur.execute(
            """
            UPDATE trades_summary SET
                ticker = %s, status = %s, open_date = %s, closed_date = %s,
                shares = %s, avg_entry = %s, avg_exit = %s, total_cost = %s,
                realized_pl = %s, unrealized_pl = %s, return_pct = %s,
                sell_rule = %s, notes = %s, stop_loss = %s, rule = %s,
                buy_notes = %s, sell_notes = %s, risk_budget = %s,
                instrument_type = %s, multiplier = %s
            WHERE id = %s
            RETURNING id
            """,
            (
                summary_row.get('Ticker'),
                summary_row.get('Status', 'OPEN'),
                summary_row.get('Open_Date'),
                summary_row.get('Closed_Date'),
                summary_row.get('Shares', 0),
                summary_row.get('Avg_Entry', 0),
                summary_row.get('Avg_Exit', 0),
                summary_row.get('Total_Cost', 0),
                summary_row.get('Realized_PL', 0),
                summary_row.get('Unrealized_PL', 0),
                summary_row.get('Return_Pct', 0),
                summary_row.get('Sell_Rule'),
                summary_row.get('Notes'),
                summary_row.get('Stop_Loss'),
                summary_row.get('Rule'),
                summary_row.get('Buy_Notes'),
                summary_row.get('Sell_Notes'),
                summary_row.get('Risk_Budget', 0),
                summary_row.get('Instrument_Type', 'STOCK'),
                summary_row.get('Multiplier', 1),
                existing[0],
            ),
        )
    else:
        cur.execute(
            """
            INSERT INTO trades_summary (
                portfolio_id, trade_id, ticker, status, open_date, closed_date,
                shares, avg_entry, avg_exit, total_cost, realized_pl, unrealized_pl,
                return_pct, sell_rule, notes, stop_loss, rule, buy_notes, sell_notes,
                risk_budget, instrument_type, multiplier
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            RETURNING id
            """,
            (
                portfolio_id,
                trade_id_for_summary,
                summary_row.get('Ticker'),
                summary_row.get('Status', 'OPEN'),
                summary_row.get('Open_Date'),
                summary_row.get('Closed_Date'),
                summary_row.get('Shares', 0),
                summary_row.get('Avg_Entry', 0),
                summary_row.get('Avg_Exit', 0),
                summary_row.get('Total_Cost', 0),
                summary_row.get('Realized_PL', 0),
                summary_row.get('Unrealized_PL', 0),
                summary_row.get('Return_Pct', 0),
                summary_row.get('Sell_Rule'),
                summary_row.get('Notes'),
                summary_row.get('Stop_Loss'),
                summary_row.get('Rule'),
                summary_row.get('Buy_Notes'),
                summary_row.get('Sell_Notes'),
                summary_row.get('Risk_Budget', 0),
                summary_row.get('Instrument_Type', 'STOCK'),
                summary_row.get('Multiplier', 1),
            ),
        )
    summary_id = cur.fetchone()[0]

    # --- 2. Replace lot_closures (DELETE-then-INSERT) ---
    cur.execute(
        "DELETE FROM lot_closures WHERE portfolio_id = %s AND trade_id = %s",
        (portfolio_id, trade_id),
    )

    if closures:
        cur.executemany(
            """
            INSERT INTO lot_closures (
                portfolio_id, trade_id, sell_trx_id, buy_trx_id,
                shares, buy_price, sell_price, multiplier,
                realized_pl, closed_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    portfolio_id,
                    trade_id,
                    c["sell_trx_id"],
                    c["buy_trx_id"],
                    c["shares"],
                    c["buy_price"],
                    c["sell_price"],
                    c["multiplier"],
                    c["realized_pl"],
                    c["closed_at"],
                )
                for c in closures
            ],
        )

    return summary_id


def delete_lot_closures_for_trade(portfolio_name, trade_id):
    """Hard-delete every lot_closures row for one trade.

    Standalone primitive used by the recompute empty-txns branch in
    api/main.py (when the trade is being deleted because no detail rows
    remain). Whole-portfolio deletes don't need a bulk version — the
    portfolio_id FK on lot_closures is ON DELETE CASCADE, so dropping a
    portfolio cleans its closures automatically.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
            result = cur.fetchone()
            if not result:
                raise ValueError(f"Portfolio '{portfolio_name}' not found")
            portfolio_id = result[0]
            cur.execute(
                "DELETE FROM lot_closures WHERE portfolio_id = %s AND trade_id = %s",
                (portfolio_id, trade_id),
            )
            conn.commit()


def load_lot_closures(portfolio_name, trade_id=None, trade_ids=None):
    """Load lot_closures rows for a portfolio, optionally filtered.

    Args:
        portfolio_name: Portfolio name (e.g. 'CanSlim')
        trade_id: If given, return only rows for this single trade
        trade_ids: If given (and trade_id is None), return rows for this set
                   of trades — used by batch endpoints to fetch closures for
                   the same slice of trades they're returning details for.
        If neither filter is given, returns ALL closures for the portfolio.

    Returns:
        DataFrame with columns: trade_id, buy_trx_id, sell_trx_id, shares,
        buy_price, sell_price, multiplier, realized_pl, closed_at.

    No deleted_at filter — lot_closures has no soft-delete column. The
    recompute path keeps rows current via DELETE-then-INSERT in
    save_summary_with_closures, so any row present is authoritative.
    Empty DataFrame returned when no rows match (or for the 6 deferred
    trades that haven't been backfilled).
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
            result = cur.fetchone()
            if not result:
                raise ValueError(f"Portfolio '{portfolio_name}' not found")
            portfolio_id = result[0]

            base_query = """
                SELECT trade_id, buy_trx_id, sell_trx_id,
                       shares, buy_price, sell_price,
                       multiplier, realized_pl, closed_at
                FROM lot_closures
                WHERE portfolio_id = %s
            """
            params = [portfolio_id]

            if trade_id is not None:
                base_query += " AND trade_id = %s"
                params.append(trade_id)
            elif trade_ids is not None:
                if not trade_ids:
                    # Empty list filter → no rows can match. Skip the SQL
                    # round-trip and return an empty frame with the right
                    # columns so callers can iterate without special cases.
                    return pd.DataFrame(columns=[
                        "trade_id", "buy_trx_id", "sell_trx_id",
                        "shares", "buy_price", "sell_price",
                        "multiplier", "realized_pl", "closed_at",
                    ])
                placeholders = ",".join(["%s"] * len(trade_ids))
                base_query += f" AND trade_id IN ({placeholders})"
                params.extend(trade_ids)

            base_query += " ORDER BY trade_id, closed_at, id"

            cur.execute(base_query, tuple(params))
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]

    df = pd.DataFrame(rows, columns=columns)

    # Coerce Decimal columns to float so JSON serialization downstream
    # produces numbers, not strings. Same pattern load_summary uses.
    if not df.empty:
        for col in ("shares", "buy_price", "sell_price", "multiplier", "realized_pl"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# Canonical trx_id format. DRY anchor for strict-mode validators on the
# import (migrate_csv_to_postgres.py) and API surface (api/main.py log_buy
# / log_sell / edit_transaction). Generators like generate_unique_trx_id
# below build their own per-prefix regex; consumers that just need to
# decide "is this string a legitimate trx_id?" import this constant.
#
# Allowed forms (surveyed against 1453 active production rows):
#   B1, B2, ...           initial BUYs                                (542 rows)
#   A1, A2, ...           add-on BUYs                                 (294 rows)
#   S1, S2, ...           SELLs                                       (307 rows)
#   SA1, SA2, ...         legacy SELL-of-add                          (134 rows)
#   SB1, SB2, ...         legacy SELL-of-base                         (134 rows)
#   {base}-Auto           IBKR auto-import marker — dead convention,  ( 13 rows)
#                         preserved for backward-compat; no current
#                         code path generates it.
#   {base}-N              dedupe-script survivor (N >= 2), see        ( 24 rows)
#                         scripts/dedupe_trx_ids.py:163-168
#   B0-Pre                pre-existing-position marker                (  1 row)
#
# Alternation ordering inside the regex is longest-first deliberately —
# `S` is a prefix of `SA` and `SB`, so listing `SB|SA` before `S` avoids
# regex backtracking that would otherwise misclassify `SA1` as `S` + `A1`.
# Same reason `B0-Pre` is its own top-level alternative: the literal hyphen
# would otherwise need to be excluded from the `{base}-Auto|-\d+` branch.
#
# Numeric suffix is `[1-9]\d*` — generator counts from 1, and a production
# survey confirmed no standalone B0/A0/S0/SA0/SB0 rows exist. B0 is
# reserved exclusively for the `B0-Pre` literal.
TRX_ID_PATTERN = re.compile(
    r'^(?:B0-Pre|(?:SB|SA|B|A|S)[1-9]\d*(?:-Auto|-\d+)?)$'
)


def generate_unique_trx_id(portfolio_name: str, trade_id: str, prefix: str) -> str:
    """Return the lowest-numbered unused trx_id matching ^{prefix}\\d+$ in
    (portfolio_id, trade_id), e.g. 'B2' or 'S5'. Skips gaps already filled.

    Concurrency — read this carefully:
      An advisory lock keyed on (portfolio_id, crc32(trade_id)) is held
      during the SELECT to serialize concurrent generators on the same
      trade. The lock auto-releases at this function's commit, BEFORE
      the caller's INSERT. Two concurrent callers can therefore still
      receive the same trx_id in the (small) window between this
      function returning and either caller's INSERT landing.

      The actual race-safety guarantee is the partial unique index
      `unique_trx_id_per_trade` on trades_details (migration 018):
      UNIQUE (portfolio_id, trade_id, trx_id) WHERE deleted_at IS NULL.
      Active rows only — soft-deleted rows are outside the index's
      scope, matching this helper's existing-trx_id scan (which already
      filters deleted_at IS NULL). A duplicate active INSERT raises
      psycopg2.errors.UniqueViolation, which the caller catches and
      retries by calling this helper again. The advisory lock is a
      contention reducer that makes retries rare; it is NOT a
      correctness guarantee on its own.

      In short: the returned trx_id is a best guess, not a reservation.

    Why the regex (not LIKE 'prefix%'): we want pure {prefix}{digits}
    matches only. For prefix='S', LIKE 'S%' would also match legacy
    'SA1' / 'SB2' rows and incorrectly think S1 is taken when it isn't.
    The ^{prefix}\\d+$ regex isolates the integer-suffix variant.

    Soft-deleted rows (deleted_at IS NOT NULL) don't count as taken —
    a deleted 'S2' frees up that suffix for reuse.

    Examples (in a trade with existing trx_ids B1, A1, A2, S1, SA1):
        generate_unique_trx_id(portfolio, trade, 'B') -> 'B2'
        generate_unique_trx_id(portfolio, trade, 'A') -> 'A3'
        generate_unique_trx_id(portfolio, trade, 'S') -> 'S2'
        generate_unique_trx_id(portfolio, trade, 'SA') -> 'SA2'
    """
    with atomic_transaction() as (_conn, cur):
        cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
        result = cur.fetchone()
        if not result:
            raise ValueError(f"Portfolio '{portfolio_name}' not found")
        portfolio_id = result[0]

        return _generate_unique_trx_id_in_txn(cur, portfolio_id, trade_id, prefix)


def _generate_unique_trx_id_in_txn(cur, portfolio_id, trade_id, prefix):
    """Inner-transaction body of generate_unique_trx_id: take the per-trade
    advisory lock, scan existing active trx_ids matching ^{prefix}\\d+$,
    and return the lowest unused integer suffix as '{prefix}{n}'.

    Caller owns the transaction. Same race-safety contract as the public
    helper — the returned trx_id is a best guess, not a reservation, and
    the caller's INSERT must be ready to handle a UniqueViolation by
    retrying.

    The advisory lock is keyed on (portfolio_id, crc32(trade_id)) and
    auto-releases at the caller's commit/rollback. Multi-write callers
    (e.g. exercise-option) holding the same outer transaction across two
    generate_*_in_txn calls for the SAME trade_id will serialize
    correctly; calls for DIFFERENT trade_ids never contend because the
    lock key includes trade_id.
    """
    # crc32 is deterministic across processes (unlike Python's built-in hash,
    # which is randomized by PYTHONHASHSEED). Mask to int32 signed range so
    # pg_advisory_xact_lock(int4, int4) accepts the value.
    lock_key = zlib.crc32(trade_id.encode()) & 0x7FFFFFFF
    prefix_re = re.compile(rf"^{re.escape(prefix)}(\d+)$")

    cur.execute(
        "SELECT pg_advisory_xact_lock(%s, %s)",
        (portfolio_id, lock_key),
    )

    cur.execute(
        """
        SELECT trx_id FROM trades_details
        WHERE portfolio_id = %s
          AND trade_id = %s
          AND deleted_at IS NULL
          AND trx_id ~ %s
        """,
        (portfolio_id, trade_id, prefix_re.pattern),
    )
    used: set[int] = set()
    for (trx_id,) in cur.fetchall():
        m = prefix_re.match(trx_id)
        if m:
            used.add(int(m.group(1)))

    n = 1
    while n in used:
        n += 1
    return f"{prefix}{n}"


def update_trade_stops(portfolio_name, trade_id, new_stop,
                       be_applied=False, be_cleared=False):
    """
    Apply a new stop loss to every open lot of a trade and mirror it on the
    summary row. Optionally set or clear the BE rule flag.

    Args:
        portfolio_name: Portfolio name ('CanSlim' etc.)
        trade_id: Trade ID to update
        new_stop: New stop price (> 0)
        be_applied: If True, stamp be_stop_moved_at = NOW() on the summary
        be_cleared: If True, clear be_stop_moved_at (stop moved off BE)

    Returns:
        Number of detail rows updated
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
            result = cur.fetchone()
            if not result:
                raise ValueError(f"Portfolio '{portfolio_name}' not found")
            portfolio_id = result[0]

            # Update all detail rows for open lots of this trade. Also
            # clear stop_ladder on every BUY row (Phase 3): when the user
            # sets a manual single stop, they're promoting the ladder
            # plan to a global stop — the ladder is over. Clearing on all
            # BUY rows is defensive (Phase 1 only writes ladder to B1)
            # and idempotent for non-laddered trades (SET to NULL is a
            # no-op when already NULL).
            cur.execute("""
                UPDATE trades_details
                SET stop_loss = %s,
                    stop_ladder = NULL
                WHERE portfolio_id = %s AND trade_id = %s
                  AND action = 'BUY' AND deleted_at IS NULL
            """, (new_stop, portfolio_id, trade_id))
            updated = cur.rowcount

            # Mirror to summary + optional BE flag update
            if be_applied:
                cur.execute("""
                    UPDATE trades_summary
                    SET stop_loss = %s,
                        be_stop_moved_at = NOW()
                    WHERE portfolio_id = %s AND trade_id = %s
                      AND deleted_at IS NULL
                """, (new_stop, portfolio_id, trade_id))
            elif be_cleared:
                cur.execute("""
                    UPDATE trades_summary
                    SET stop_loss = %s,
                        be_stop_moved_at = NULL
                    WHERE portfolio_id = %s AND trade_id = %s
                      AND deleted_at IS NULL
                """, (new_stop, portfolio_id, trade_id))
            else:
                cur.execute("""
                    UPDATE trades_summary
                    SET stop_loss = %s
                    WHERE portfolio_id = %s AND trade_id = %s
                      AND deleted_at IS NULL
                """, (new_stop, portfolio_id, trade_id))

            conn.commit()
            load_summary.clear()
            load_details.clear()
            return updated


def update_trade_ladder(portfolio_name: str, trade_id: str, ladder: dict) -> int:
    """Replace the stop_ladder JSONB on the B1 (earliest BUY) row of a
    trade. Used by the Trade Manager → Stop Loss Adjustment "Edit ladder"
    flow. The caller validates ladder shape (locked pcts [3,5,7], leg
    shares sum to B1 shares) before invoking. Returns 1 if B1 was found
    and updated, 0 otherwise.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
            result = cur.fetchone()
            if not result:
                raise ValueError(f"Portfolio '{portfolio_name}' not found")
            portfolio_id = result[0]

            # B1 = earliest BUY by (date ASC, id ASC). Ladder is B1-only
            # by convention; scale-in rows never carry one.
            cur.execute("""
                UPDATE trades_details
                SET stop_ladder = %s
                WHERE id = (
                    SELECT id FROM trades_details
                    WHERE portfolio_id = %s AND trade_id = %s
                      AND action = 'BUY' AND deleted_at IS NULL
                    ORDER BY date ASC, id ASC
                    LIMIT 1
                )
            """, (json.dumps(ladder), portfolio_id, trade_id))
            updated = cur.rowcount
            conn.commit()
            load_details.clear()
            return updated


def mirror_detail_edit_to_summary(portfolio_name: str, trade_id: str) -> None:
    """Mirror canonical detail-row fields to trades_summary.

    Convention:
      - earliest BUY (date ASC, id ASC) wins for summary.rule, buy_notes,
        stop_loss
      - latest SELL (date DESC, id DESC) on a CLOSED campaign wins for
        summary.sell_rule, sell_notes
      - OPEN campaign with partial sells: sell_rule/sell_notes left alone
        (those edits are detail-level only)

    Called from edit_transaction_endpoint AFTER update_detail_row so the
    helper sees the post-edit state, and BEFORE _recompute_summary_matching so
    the recompute's preservation block reads the just-mirrored values
    instead of the stale pre-edit ones.

    Idempotent: re-running with no detail changes is a no-op.
    No-op if the trade has no BUY rows (e.g. mid-delete state).
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
            result = cur.fetchone()
            if not result:
                raise ValueError(f"Portfolio '{portfolio_name}' not found")
            portfolio_id = result[0]

            # Earliest BUY — canonical row for summary.rule/buy_notes/stop_loss.
            # date ASC + id ASC tiebreak matches load_summary's Buy_Rule
            # projection conceptually (date ASC) with deterministic insertion-
            # order tiebreak for same-day rows.
            cur.execute("""
                SELECT rule, notes, stop_loss
                  FROM trades_details
                 WHERE portfolio_id = %s AND trade_id = %s
                   AND action = 'BUY' AND deleted_at IS NULL
                 ORDER BY date ASC, id ASC
                 LIMIT 1
            """, (portfolio_id, trade_id))
            first_buy = cur.fetchone()

            # Latest SELL on a CLOSED campaign — canonical row for
            # summary.sell_rule/sell_notes. OPEN campaign with partial sells
            # returns no row (status filter), so the SELL UPDATE is skipped
            # and detail-level rule/notes stay detail-only.
            cur.execute("""
                SELECT d.rule, d.notes
                  FROM trades_details d
                  JOIN trades_summary s
                    ON s.portfolio_id = d.portfolio_id
                   AND s.trade_id = d.trade_id
                 WHERE d.portfolio_id = %s AND d.trade_id = %s
                   AND d.action = 'SELL' AND d.deleted_at IS NULL
                   AND s.deleted_at IS NULL
                   AND s.status = 'CLOSED'
                 ORDER BY d.date DESC, d.id DESC
                 LIMIT 1
            """, (portfolio_id, trade_id))
            latest_sell = cur.fetchone()

            if first_buy is not None:
                cur.execute("""
                    UPDATE trades_summary
                       SET rule = %s,
                           buy_notes = %s,
                           stop_loss = %s
                     WHERE portfolio_id = %s AND trade_id = %s
                       AND deleted_at IS NULL
                """, (
                    clean_text_value(first_buy[0]),
                    clean_text_value(first_buy[1]),
                    first_buy[2],
                    portfolio_id, trade_id,
                ))

            if latest_sell is not None:
                cur.execute("""
                    UPDATE trades_summary
                       SET sell_rule = %s,
                           sell_notes = %s
                     WHERE portfolio_id = %s AND trade_id = %s
                       AND deleted_at IS NULL
                """, (
                    clean_text_value(latest_sell[0]),
                    clean_text_value(latest_sell[1]),
                    portfolio_id, trade_id,
                ))

            conn.commit()
            load_summary.clear()


# ============================================
# UTILITY: Query Cross-Portfolio
# ============================================
def get_all_open_trades():
    """
    Get all open trades across all portfolios.
    Example of cross-portfolio query.
    """
    with get_db_connection() as conn:
        query = """
            SELECT s.*, p.name as portfolio_name
            FROM trades_summary s
            JOIN portfolios p ON s.portfolio_id = p.id
            WHERE s.status = 'OPEN'
              AND s.deleted_at IS NULL
            ORDER BY p.name, s.open_date DESC
        """
        return pd.read_sql(query, conn)


# ============================================
# MARKET SIGNALS OPERATIONS
# ============================================

@ttl_cache(ttl=600, show_spinner=False)
def load_market_signals(symbol=None, days=30):
    """Load market signals for display (cached 10 minutes)."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT symbol, signal_date, close_price, daily_change_pct,
                       market_exposure, position_allocation, buy_switch,
                       distribution_count, above_21ema, above_50ma,
                       buy_signals, sell_signals, analyzed_at
                FROM market_signals
                WHERE signal_date >= CURRENT_DATE - INTERVAL '%s days'
            """
            params = [days]
            if symbol:
                query += " AND symbol = %s"
                params.append(symbol)
            query += " ORDER BY signal_date DESC, symbol"

            cur.execute(query, params)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            df = pd.DataFrame(rows, columns=columns)

            if not df.empty:
                df['signal_date'] = pd.to_datetime(df['signal_date'])
                df['analyzed_at'] = pd.to_datetime(df['analyzed_at'])
                from decimal import Decimal
                for col in df.columns:
                    sample = df[col].dropna()
                    if len(sample) > 0 and isinstance(sample.iloc[0], Decimal):
                        df[col] = pd.to_numeric(df[col], errors='coerce')
            return df


def save_market_signal(signal_dict):
    """Insert or update market signal row."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM market_signals WHERE symbol = %s AND signal_date = %s",
                (signal_dict['symbol'], signal_dict['signal_date'])
            )
            existing = cur.fetchone()

            if existing:
                update_query = """
                    UPDATE market_signals
                    SET close_price=%s, daily_change_pct=%s, market_exposure=%s,
                        position_allocation=%s, buy_switch=%s, distribution_count=%s,
                        above_21ema=%s, above_50ma=%s, buy_signals=%s, sell_signals=%s,
                        analyzed_at=CURRENT_TIMESTAMP
                    WHERE id=%s RETURNING id
                """
                cur.execute(update_query, (
                    signal_dict.get('close_price'),
                    signal_dict.get('daily_change_pct'),
                    signal_dict.get('market_exposure', 0),
                    signal_dict.get('position_allocation', 0),
                    signal_dict.get('buy_switch', False),
                    signal_dict.get('distribution_count', 0),
                    signal_dict.get('above_21ema', False),
                    signal_dict.get('above_50ma', False),
                    signal_dict.get('buy_signals'),
                    signal_dict.get('sell_signals'),
                    existing[0]
                ))
            else:
                insert_query = """
                    INSERT INTO market_signals (
                        symbol, signal_date, close_price, daily_change_pct,
                        market_exposure, position_allocation, buy_switch,
                        distribution_count, above_21ema, above_50ma,
                        buy_signals, sell_signals
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id
                """
                cur.execute(insert_query, (
                    signal_dict['symbol'],
                    signal_dict['signal_date'],
                    signal_dict.get('close_price'),
                    signal_dict.get('daily_change_pct'),
                    signal_dict.get('market_exposure', 0),
                    signal_dict.get('position_allocation', 0),
                    signal_dict.get('buy_switch', False),
                    signal_dict.get('distribution_count', 0),
                    signal_dict.get('above_21ema', False),
                    signal_dict.get('above_50ma', False),
                    signal_dict.get('buy_signals'),
                    signal_dict.get('sell_signals')
                ))

            row_id = cur.fetchone()[0]
            conn.commit()
            load_market_signals.clear()
            return row_id


def get_latest_signal_date(symbol):
    """Get most recent signal date for a symbol."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT MAX(signal_date) FROM market_signals WHERE symbol = %s",
                (symbol,)
            )
            result = cur.fetchone()
            return result[0] if result and result[0] else None


def load_v11_market_signals(days=30, signal_type=None):
    """Load V11 MCT engine signals from market_signals (post-migration 010 schema).

    Distinct from load_market_signals() above, which targets the orphaned V10
    schema (market_exposure, buy_switch, distribution_count, ...) — those columns
    no longer exist on the table after migration 010 dropped + recreated it.
    The V10 reader is dead code pending cleanup.

    Returns rows ordered by trade_date desc, then id desc as a tie-break so a
    multi-event day reads back in deterministic insertion order. Each row is a
    dict ready to JSON-serialize except trade_date, which the caller must coerce.
    """
    import json as _json
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT trade_date, signal_type, signal_label,
                       exposure_before, exposure_after,
                       state_before, state_after, meta
                FROM market_signals
                WHERE trade_date >= CURRENT_DATE - (%s * INTERVAL '1 day')
            """
            params = [int(days)]
            if signal_type:
                query += " AND signal_type = %s"
                params.append(signal_type)
            query += " ORDER BY trade_date DESC, id DESC"
            cur.execute(query, params)
            cols = [d[0] for d in cur.description]
            rows = []
            for raw in cur.fetchall():
                rec = dict(zip(cols, raw))
                meta = rec.get("meta")
                if isinstance(meta, str):
                    try:
                        rec["meta"] = _json.loads(meta)
                    except (ValueError, TypeError):
                        rec["meta"] = {}
                elif meta is None:
                    rec["meta"] = {}
                rows.append(rec)
            return rows


# ============================================
# JOURNAL OPERATIONS
# ============================================

def save_journal_entry(journal_entry):
    """
    Insert or update a journal entry.

    Args:
        journal_entry: Dictionary with journal entry data
            Required keys: portfolio_id (name), day
            Optional keys: status, market_window, above_21ema, cash_flow,
                          beginning_nlv, ending_nlv, daily_dollar_change,
                          daily_percent_change, percent_invested, spy_close,
                          nasdaq_close, market_notes, market_action, score,
                          highlights, lowlights, mistakes, top_lesson

    Returns:
        int: ID of inserted/updated row
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Get portfolio_id from name
            portfolio_name = journal_entry.get('portfolio_id')
            cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
            result = cur.fetchone()
            if not result:
                raise ValueError(f"Portfolio '{portfolio_name}' not found")
            portfolio_id = result[0]

            # Map the field names from app.py to database column names
            day = journal_entry.get('day')
            status = journal_entry.get('status', 'U')
            market_window = journal_entry.get('market_window', 'Open')
            market_cycle = journal_entry.get('market_cycle', '')
            mct_display_day_num = journal_entry.get('mct_display_day_num')
            # Signed Trend Count (Migration 043). NULL is a first-class
            # value here — 0 is a legit Step-4 arm bar, so we can't use
            # `.get(..., 0)` as a default and later distinguish "not set."
            trend_count = journal_entry.get('trend_count')
            above_21ema = journal_entry.get('above_21ema', 0.0)
            cash_change = journal_entry.get('cash_flow', 0.0)
            beg_nlv = journal_entry.get('beginning_nlv', 0.0)
            end_nlv = journal_entry.get('ending_nlv', 0.0)
            daily_dollar_change = journal_entry.get('daily_dollar_change', 0.0)
            daily_pct_change = journal_entry.get('daily_percent_change', 0.0)
            pct_invested = journal_entry.get('percent_invested', 0.0)
            spy = journal_entry.get('spy_close', 0.0)
            nasdaq = journal_entry.get('nasdaq_close', 0.0)
            market_notes = journal_entry.get('market_notes', '')
            market_action = journal_entry.get('market_action', '')
            portfolio_heat = journal_entry.get('portfolio_heat', 0.0)
            spy_atr = journal_entry.get('spy_atr', 0.0)
            nasdaq_atr = journal_entry.get('nasdaq_atr', 0.0)
            score = journal_entry.get('score', 0)
            highlights = journal_entry.get('highlights', '')
            lowlights = journal_entry.get('lowlights', '')
            mistakes = journal_entry.get('mistakes', '')
            top_lesson = journal_entry.get('top_lesson', '')
            # Phase 7 — rich-text body for the new Daily Thoughts editor.
            # Defaulted to '' to match the migration 031 column default;
            # only the primary INSERT/UPDATE path writes it (fallback paths
            # target pre-migration-031 schemas which lack the column).
            daily_thoughts = journal_entry.get('daily_thoughts', '') or ''
            # Provenance for the End NLV value (migration 013). Defaulted to
            # 'manual' so pre-migration callers keep working; constrained to
            # the three known values (anything else collapses to 'manual').
            nlv_source = journal_entry.get('nlv_source', 'manual')
            if nlv_source not in ('manual', 'ibkr_auto', 'ibkr_override'):
                nlv_source = 'manual'
            # Same shape for Total Holdings provenance (migration 014). Tracked
            # independently — user can accept IBKR's NLV but override Holdings
            # (or vice versa), so the two flags don't have to agree.
            holdings_source = journal_entry.get('holdings_source', 'manual')
            if holdings_source not in ('manual', 'ibkr_auto', 'ibkr_override'):
                holdings_source = 'manual'

            # Check if entry exists for this portfolio and day
            cur.execute(
                "SELECT id FROM trading_journal WHERE portfolio_id = %s AND day = %s",
                (portfolio_id, day)
            )
            existing = cur.fetchone()

            if existing:
                # UPDATE existing entry — try with all columns incl. market_cycle,
                # fall back progressively if newer columns are missing.
                try:
                    # Primary path — includes nlv_source + holdings_source
                    # (migrations 013, 014) and market_cycle (migration 010).
                    # Falls back below if any of those columns is missing.
                    update_query = """
                        UPDATE trading_journal
                        SET status = %s, market_window = %s, market_cycle = %s,
                            mct_display_day_num = %s, trend_count = %s,
                            above_21ema = %s,
                            cash_change = %s, beg_nlv = %s, end_nlv = %s,
                            daily_dollar_change = %s, daily_pct_change = %s,
                            pct_invested = %s, spy = %s, nasdaq = %s,
                            market_notes = %s, market_action = %s,
                            portfolio_heat = %s, spy_atr = %s, nasdaq_atr = %s,
                            score = %s,
                            highlights = %s, lowlights = %s, mistakes = %s,
                            top_lesson = %s,
                            nlv_source = %s, holdings_source = %s,
                            daily_thoughts = %s
                        WHERE id = %s
                        RETURNING id
                    """
                    cur.execute(update_query, (
                        status, market_window, market_cycle,
                        mct_display_day_num, trend_count,
                        above_21ema,
                        cash_change, beg_nlv, end_nlv,
                        daily_dollar_change, daily_pct_change,
                        pct_invested, spy, nasdaq,
                        market_notes, market_action,
                        portfolio_heat, spy_atr, nasdaq_atr,
                        score,
                        highlights, lowlights, mistakes,
                        top_lesson,
                        nlv_source, holdings_source,
                        daily_thoughts,
                        existing[0]
                    ))
                except Exception:
                    conn.rollback()
                    try:
                        update_query = """
                            UPDATE trading_journal
                            SET status = %s, market_window = %s, above_21ema = %s,
                                cash_change = %s, beg_nlv = %s, end_nlv = %s,
                                daily_dollar_change = %s, daily_pct_change = %s,
                                pct_invested = %s, spy = %s, nasdaq = %s,
                                market_notes = %s, market_action = %s,
                                portfolio_heat = %s, spy_atr = %s, nasdaq_atr = %s,
                                score = %s,
                                highlights = %s, lowlights = %s, mistakes = %s,
                                top_lesson = %s
                            WHERE id = %s
                            RETURNING id
                        """
                        cur.execute(update_query, (
                            status, market_window, above_21ema,
                            cash_change, beg_nlv, end_nlv,
                            daily_dollar_change, daily_pct_change,
                            pct_invested, spy, nasdaq,
                            market_notes, market_action,
                            portfolio_heat, spy_atr, nasdaq_atr,
                            score,
                            highlights, lowlights, mistakes,
                            top_lesson,
                            existing[0]
                        ))
                    except Exception:
                        conn.rollback()
                        update_query = """
                            UPDATE trading_journal
                            SET status = %s, market_window = %s, above_21ema = %s,
                                cash_change = %s, beg_nlv = %s, end_nlv = %s,
                                daily_dollar_change = %s, daily_pct_change = %s,
                                pct_invested = %s, spy = %s, nasdaq = %s,
                                market_notes = %s, market_action = %s, score = %s,
                                highlights = %s, lowlights = %s, mistakes = %s,
                                top_lesson = %s
                            WHERE id = %s
                            RETURNING id
                        """
                        cur.execute(update_query, (
                            status, market_window, above_21ema,
                            cash_change, beg_nlv, end_nlv,
                            daily_dollar_change, daily_pct_change,
                            pct_invested, spy, nasdaq,
                            market_notes, market_action, score,
                            highlights, lowlights, mistakes,
                            top_lesson,
                            existing[0]
                        ))
            else:
                # INSERT new entry — try with all columns incl. market_cycle,
                # fall back progressively if newer columns are missing.
                try:
                    # Primary path — includes nlv_source + holdings_source
                    # (migrations 013, 014) and market_cycle (migration 010).
                    # Falls back below if any of those columns is missing.
                    insert_query = """
                        INSERT INTO trading_journal (
                            portfolio_id, day, status, market_window, market_cycle,
                            mct_display_day_num, trend_count,
                            above_21ema,
                            cash_change, beg_nlv, end_nlv, daily_dollar_change,
                            daily_pct_change, pct_invested, spy, nasdaq,
                            market_notes, market_action, portfolio_heat,
                            spy_atr, nasdaq_atr, score,
                            highlights, lowlights, mistakes, top_lesson,
                            nlv_source, holdings_source, daily_thoughts
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        RETURNING id
                    """
                    cur.execute(insert_query, (
                        portfolio_id, day, status, market_window, market_cycle,
                        mct_display_day_num, trend_count,
                        above_21ema,
                        cash_change, beg_nlv, end_nlv, daily_dollar_change,
                        daily_pct_change, pct_invested, spy, nasdaq,
                        market_notes, market_action, portfolio_heat,
                        spy_atr, nasdaq_atr, score,
                        highlights, lowlights, mistakes, top_lesson,
                        nlv_source, holdings_source, daily_thoughts
                    ))
                except Exception:
                    conn.rollback()
                    try:
                        insert_query = """
                            INSERT INTO trading_journal (
                                portfolio_id, day, status, market_window, above_21ema,
                                cash_change, beg_nlv, end_nlv, daily_dollar_change,
                                daily_pct_change, pct_invested, spy, nasdaq,
                                market_notes, market_action, portfolio_heat,
                                spy_atr, nasdaq_atr, score,
                                highlights, lowlights, mistakes, top_lesson
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s
                            )
                            RETURNING id
                        """
                        cur.execute(insert_query, (
                            portfolio_id, day, status, market_window, above_21ema,
                            cash_change, beg_nlv, end_nlv, daily_dollar_change,
                            daily_pct_change, pct_invested, spy, nasdaq,
                            market_notes, market_action, portfolio_heat,
                            spy_atr, nasdaq_atr, score,
                            highlights, lowlights, mistakes, top_lesson
                        ))
                    except Exception:
                        conn.rollback()
                        insert_query = """
                            INSERT INTO trading_journal (
                                portfolio_id, day, status, market_window, above_21ema,
                                cash_change, beg_nlv, end_nlv, daily_dollar_change,
                                daily_pct_change, pct_invested, spy, nasdaq,
                                market_notes, market_action, score,
                                highlights, lowlights, mistakes, top_lesson
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                            )
                            RETURNING id
                        """
                        cur.execute(insert_query, (
                            portfolio_id, day, status, market_window, above_21ema,
                            cash_change, beg_nlv, end_nlv, daily_dollar_change,
                            daily_pct_change, pct_invested, spy, nasdaq,
                            market_notes, market_action, score,
                            highlights, lowlights, mistakes, top_lesson
                        ))

            row_id = cur.fetchone()[0]
            conn.commit()

            # Clear cache so next load gets fresh data
            load_journal.clear()

            return row_id


def update_journal_mct_state(portfolio_name: str, day: str, market_cycle: str | None, mct_display_day_num: int | None) -> int:
    """Targeted UPDATE for just the MCT badge fields on a single journal row.

    save_journal_entry rewrites every column, which clobbers NLV and notes
    when called with a partial dict. The lazy-heal path in
    api/main._heal_recent_mct_stamps only wants to touch the two MCT fields,
    so it goes through here instead. Returns the row id, or 0 if no matching
    row exists. RLS still applies via the standard get_db_connection path.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
            result = cur.fetchone()
            if not result:
                return 0
            portfolio_id = result[0]
            cur.execute(
                "UPDATE trading_journal "
                "   SET market_cycle = %s, mct_display_day_num = %s, updated_at = NOW() "
                " WHERE portfolio_id = %s AND day = %s "
                "   AND deleted_at IS NULL "
                " RETURNING id",
                (market_cycle, mct_display_day_num, portfolio_id, day),
            )
            row = cur.fetchone()
            conn.commit()
            load_journal.clear()
            return int(row[0]) if row else 0


def update_journal_trend_state(portfolio_name: str, day: str, trend_count: int | None) -> int:
    """Targeted UPDATE for just the trend_count field on a single journal row.

    Same rationale as update_journal_mct_state — save_journal_entry rewrites
    every column and would clobber unrelated fields when the lazy-heal path
    only cares about trend_count. Returns the row id or 0 if no matching row.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
            result = cur.fetchone()
            if not result:
                return 0
            portfolio_id = result[0]
            cur.execute(
                "UPDATE trading_journal "
                "   SET trend_count = %s, updated_at = NOW() "
                " WHERE portfolio_id = %s AND day = %s "
                "   AND deleted_at IS NULL "
                " RETURNING id",
                (trend_count, portfolio_id, day),
            )
            row = cur.fetchone()
            conn.commit()
            load_journal.clear()
            return int(row[0]) if row else 0


def delete_journal_entry(portfolio_name, day):
    """
    Delete a journal entry by portfolio name and date.

    Args:
        portfolio_name: Name of the portfolio
        day: Date string (YYYY-MM-DD) of the entry to delete

    Returns:
        bool: True if a row was deleted
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
            result = cur.fetchone()
            if not result:
                raise ValueError(f"Portfolio '{portfolio_name}' not found")
            portfolio_id = result[0]

            cur.execute(
                "UPDATE trading_journal SET deleted_at = NOW() "
                "WHERE portfolio_id = %s AND day = %s "
                "  AND deleted_at IS NULL RETURNING id",
                (portfolio_id, day)
            )
            deleted = cur.fetchone()
            conn.commit()

            # Clear cache so next load gets fresh data
            load_journal.clear()

            return deleted is not None


# ============================================
# WEEKLY RETROS (Migration 025)
# ============================================
# Parent + child pair: weekly_retros (one per Monday) and
# weekly_retro_ticker_grades (delete-then-insert on every parent save, like
# lot_closures). Helpers accept the portfolio NAME (matches the rest of the
# module — every public helper takes a string portfolio name and resolves to
# the integer id internally) and return dicts shaped for direct API
# serialization, including the ticker_grades child rows expanded into a
# {ticker: {grade, behavior, notes}} sub-dict.

_WEEK_GRADE_VOCAB = (
    "A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F",
)


def _serialize_weekly_retro(parent_row: dict, ticker_grades: dict) -> dict:
    """Shared serialization for weekly_retros parent rows + children.

    Used by load_weekly_retro, list_weekly_retros, and upsert_weekly_retro
    so the wire shape is consistent across reads and writes.
    """
    return {
        "id": parent_row["id"],
        "portfolio": parent_row["portfolio"],
        "week_start": (
            parent_row["week_start"].isoformat()
            if hasattr(parent_row["week_start"], "isoformat")
            else parent_row["week_start"]
        ),
        "week_grade": parent_row.get("week_grade"),
        # Phase 4.6: 3-axis grading. Axes nullable on legacy rows; the
        # frontend derives `effectiveOverall` from these when
        # overall_override is False, else falls back to week_grade.
        "execution_grade": parent_row.get("execution_grade"),
        "process_grade": parent_row.get("process_grade"),
        "pnl_grade": parent_row.get("pnl_grade"),
        "overall_override": bool(parent_row.get("overall_override")),
        "reviewed_at": (
            parent_row["reviewed_at"].isoformat()
            if parent_row.get("reviewed_at") and hasattr(parent_row["reviewed_at"], "isoformat")
            else parent_row.get("reviewed_at")
        ),
        "best_decision": parent_row.get("best_decision") or "",
        "worst_decision": parent_row.get("worst_decision") or "",
        "rule_change": bool(parent_row.get("rule_change")),
        "rule_change_text": parent_row.get("rule_change_text") or "",
        # Phase 3: HTML-formatted reflection prose. NOT NULL DEFAULT '' on
        # the column means this is always a string, never None.
        "weekly_thoughts": parent_row.get("weekly_thoughts") or "",
        "ticker_grades": ticker_grades,
        "created_at": (
            parent_row["created_at"].isoformat()
            if parent_row.get("created_at") and hasattr(parent_row["created_at"], "isoformat")
            else parent_row.get("created_at")
        ),
        "updated_at": (
            parent_row["updated_at"].isoformat()
            if parent_row.get("updated_at") and hasattr(parent_row["updated_at"], "isoformat")
            else parent_row.get("updated_at")
        ),
    }


# Phase 4.6: server-side overall-grade derivation. Reuses the same GPA
# mapping that powers avg_grade_from_letters (NotesRail YTD avg). Returns
# None when any axis is null — the frontend handles the "partially graded"
# state by leaving week_grade as-is.
_AXIS_GRADE_VOCAB = (
    "A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F",
)


def _derive_overall_grade(
    execution: str | None,
    process: str | None,
    pnl: str | None,
) -> str | None:
    """Average the 3 axis grades on the 4.3 GPA scale, then bucket back to
    the nearest letter via _NUMERIC_BUCKETS. Returns None if any axis is
    missing or unrecognized.

    Locked Phase 4.6 contract: this is the canonical overall derivation
    when overall_override == False. Frontend ports the same logic for
    real-time UI feedback; backend recomputes here to enforce authority
    (frontend can't poison week_grade by sending a mismatched value).
    """
    if not execution or not process or not pnl:
        return None
    vals: list[float] = []
    for g in (execution, process, pnl):
        key = str(g).strip().upper()
        if key in _GRADE_TO_NUMERIC:
            vals.append(_GRADE_TO_NUMERIC[key])
        else:
            return None
    if len(vals) != 3:
        return None
    mean = sum(vals) / 3.0
    for lower, letter in _NUMERIC_BUCKETS:
        if mean >= lower:
            # The vocab CHECK only allows A+/A/A-/B+/B/B-/C+/C/C-/D/F. The
            # bucket table has D+/D/D- extras (pre-existing inconsistency
            # noted in the audit). Collapse the extras to plain D so the
            # derived value is always vocab-clean.
            if letter in ("D+", "D-"):
                return "D"
            return letter
    return "F"


class WeeklyRetroLockedError(Exception):
    """Raised by upsert_weekly_retro when a write attempts to change graded
    fields on a retro whose reviewed_at is non-null, unless the same payload
    clears reviewed_at. Bubbled up to the API layer as a 409."""


def _fetch_ticker_grades_for_retros(cur, retro_ids: list[int]) -> dict[int, dict]:
    """Bulk-fetch children for a list of retro ids. Returns
    {retro_id: {ticker: {grade, behaviors, behavior, notes}}}. Empty dict
    for retros with no children.

    Migration 045 introduced `behaviors` (JSONB array) as the canonical
    multi-value store; `behavior` (VARCHAR) is retained for one release
    window as a rollback path and set to behaviors[0] on save.
    """
    import json as _json
    out: dict[int, dict] = {rid: {} for rid in retro_ids}
    if not retro_ids:
        return out
    cur.execute(
        "SELECT weekly_retro_id, ticker, grade, behavior, behaviors, notes "
        "FROM weekly_retro_ticker_grades "
        "WHERE weekly_retro_id = ANY(%s)",
        (retro_ids,),
    )
    for row in cur.fetchall():
        behaviors_raw = row.get("behaviors")
        # psycopg2 may return JSONB as parsed list or as string depending
        # on the connection's json handlers; coerce both to a Python list.
        if isinstance(behaviors_raw, list):
            behaviors = [str(b) for b in behaviors_raw if b]
        elif isinstance(behaviors_raw, str) and behaviors_raw:
            try:
                parsed = _json.loads(behaviors_raw)
                behaviors = [str(b) for b in parsed if b] if isinstance(parsed, list) else []
            except (ValueError, TypeError):
                behaviors = []
        else:
            behaviors = []
        # Fallback: if the multi array is empty but the legacy scalar
        # holds a value, project it as a one-element array so the
        # frontend sees consistent shape even before the backfill runs.
        if not behaviors and row.get("behavior"):
            behaviors = [str(row["behavior"])]
        out.setdefault(row["weekly_retro_id"], {})[row["ticker"]] = {
            "grade": row.get("grade") or "",
            "behaviors": behaviors,
            "behavior": row.get("behavior") or "",  # kept for legacy readers
            "notes": row.get("notes") or "",
        }
    return out


def _replace_weekly_retro_ticker_grades(
    cur, weekly_retro_id: int, ticker_grades: dict
) -> None:
    """Delete-then-insert the full child set for a retro. Mirrors the
    lot_closures recompute pattern: the helper deletes every existing child
    row for the parent and re-inserts the current set, so callers don't
    have to diff additions vs removals.

    Skips entries that are fully empty (no grade, no behaviors, no notes) so
    a ticker with all three blanks doesn't persist a useless row.

    Migration 045 dual-write:
      • `behaviors` (JSONB) = the full list, canonical going forward
      • `behavior` (VARCHAR) = behaviors[0] or '', kept populated for one
        release window as a rollback path
    """
    import json as _json
    cur.execute(
        "DELETE FROM weekly_retro_ticker_grades WHERE weekly_retro_id = %s",
        (weekly_retro_id,),
    )
    if not ticker_grades:
        return
    rows: list[tuple] = []
    for ticker, g in ticker_grades.items():
        ticker_clean = str(ticker or "").strip().upper()[:20]
        if not ticker_clean:
            continue
        if not isinstance(g, dict):
            continue
        grade = (g.get("grade") or "").strip()[:20] or None

        # Prefer the array shape; fall back to the legacy scalar so a
        # client that hasn't upgraded yet still saves without loss.
        raw_behaviors = g.get("behaviors")
        if isinstance(raw_behaviors, list):
            behaviors = [str(b).strip()[:40] for b in raw_behaviors if str(b).strip()]
        else:
            legacy = (g.get("behavior") or "").strip()[:40]
            behaviors = [legacy] if legacy else []
        # De-dup while preserving order (a chip that's clicked twice
        # should still land as one tag).
        seen: set = set()
        deduped: list = []
        for b in behaviors:
            if b not in seen:
                seen.add(b)
                deduped.append(b)
        behaviors = deduped

        # Legacy scalar mirrors the first tag so rollback / older readers
        # see a coherent value.
        legacy_behavior = behaviors[0] if behaviors else None
        behaviors_json = _json.dumps(behaviors)

        notes = g.get("notes") or ""
        if not grade and not behaviors and not notes:
            continue
        rows.append((
            weekly_retro_id, ticker_clean, grade,
            legacy_behavior, behaviors_json, notes,
        ))
    if rows:
        execute_values(
            cur,
            "INSERT INTO weekly_retro_ticker_grades "
            "(weekly_retro_id, ticker, grade, behavior, behaviors, notes) VALUES %s",
            rows,
        )


_WEEKLY_RETRO_SELECT_COLS = (
    "r.id, p.name AS portfolio, r.week_start, r.week_grade, "
    "r.execution_grade, r.process_grade, r.pnl_grade, "
    "r.overall_override, r.reviewed_at, "
    "r.best_decision, r.worst_decision, r.rule_change, "
    "r.rule_change_text, r.weekly_thoughts, "
    "r.created_at, r.updated_at"
)


def load_weekly_retro(portfolio_name: str, week_start) -> dict | None:
    """Return the live (non-deleted) retro for the given portfolio + Monday,
    with ticker_grades expanded as a {ticker: {grade, behavior, notes}} dict.
    Returns None if no row exists."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"SELECT {_WEEKLY_RETRO_SELECT_COLS} "
                "FROM weekly_retros r JOIN portfolios p ON p.id = r.portfolio_id "
                "WHERE p.name = %s AND r.week_start = %s AND r.deleted_at IS NULL",
                (portfolio_name, week_start),
            )
            parent = cur.fetchone()
            if not parent:
                return None
            parent = dict(parent)
            tg_map = _fetch_ticker_grades_for_retros(cur, [parent["id"]])
            return _serialize_weekly_retro(parent, tg_map.get(parent["id"], {}))


def list_weekly_retros(portfolio_name: str, limit: int = 200) -> list[dict]:
    """Return all live retros for the portfolio, newest first, each with its
    ticker_grades expanded. Used by the History tab today and the Phase 6
    left rail later."""
    limit = max(1, min(int(limit), 1000))
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"SELECT {_WEEKLY_RETRO_SELECT_COLS} "
                "FROM weekly_retros r JOIN portfolios p ON p.id = r.portfolio_id "
                "WHERE p.name = %s AND r.deleted_at IS NULL "
                "ORDER BY r.week_start DESC LIMIT %s",
                (portfolio_name, limit),
            )
            parents = [dict(r) for r in cur.fetchall()]
            if not parents:
                return []
            tg_map = _fetch_ticker_grades_for_retros(cur, [p["id"] for p in parents])
            return [_serialize_weekly_retro(p, tg_map.get(p["id"], {})) for p in parents]


def upsert_weekly_retro(
    portfolio_name: str,
    week_start,
    *,
    week_grade: str | None = None,
    best_decision: str = "",
    worst_decision: str = "",
    rule_change: bool = False,
    rule_change_text: str = "",
    weekly_thoughts: str = "",
    ticker_grades: dict | None = None,
    # Phase 4.6: 3-axis grading + persisted review state.
    execution_grade: str | None = None,
    process_grade: str | None = None,
    pnl_grade: str | None = None,
    overall_override: bool = False,
    reviewed_at=None,  # datetime | str | None
) -> dict:
    """Upsert by (portfolio_id, week_start). If a soft-deleted row exists
    for the same key, REVIVES it (sets deleted_at = NULL and updates the
    fields) so any tag rows that pointed at the original id survive.
    Children are replaced wholesale via _replace_weekly_retro_ticker_grades.

    Returns the persisted row in the same shape as load_weekly_retro().
    Raises ValueError on unknown portfolio name or invalid letter grade.
    Raises WeeklyRetroLockedError when the target row is already reviewed
    AND the incoming payload would change any graded field without
    simultaneously clearing reviewed_at.
    The DB enforces the Monday CHECK and grade vocabs — IntegrityError
    propagates if the caller bypasses validation.

    Phase 4.6 server authority: when overall_override is False AND all 3
    axes are non-null, week_grade is OVERWRITTEN with _derive_overall_grade
    before the write. The client-supplied value is only trusted when
    override is True (or when axes are incomplete).
    """
    # Validate every letter-grade input up front.
    for name, val in (
        ("week_grade", week_grade),
        ("execution_grade", execution_grade),
        ("process_grade", process_grade),
        ("pnl_grade", pnl_grade),
    ):
        if val is not None and val != "" and val not in _WEEK_GRADE_VOCAB:
            raise ValueError(f"Invalid {name}: {val}")

    # Empty string → NULL on every grade column.
    week_grade_val = week_grade or None
    execution_grade_val = execution_grade or None
    process_grade_val = process_grade or None
    pnl_grade_val = pnl_grade or None
    overall_override = bool(overall_override)

    # Server authority on overall: when not overridden AND every axis is
    # graded, recompute regardless of what the client sent. Prevents the
    # client from setting axes A/A/A but mismatched week_grade=C.
    if (not overall_override
            and execution_grade_val
            and process_grade_val
            and pnl_grade_val):
        week_grade_val = _derive_overall_grade(
            execution_grade_val, process_grade_val, pnl_grade_val,
        )

    best_decision = best_decision or ""
    worst_decision = worst_decision or ""
    rule_change_text = rule_change_text or ""
    weekly_thoughts = weekly_thoughts or ""
    rule_change = bool(rule_change)

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
            prow = cur.fetchone()
            if not prow:
                raise ValueError(f"Portfolio '{portfolio_name}' not found")
            portfolio_id = prow["id"]

            # Find existing row (live OR soft-deleted). The partial unique
            # index guarantees at most one live row; if a deleted row also
            # exists we revive it instead of inserting a duplicate.
            cur.execute(
                "SELECT id, reviewed_at, execution_grade, process_grade, "
                "       pnl_grade, week_grade, overall_override "
                "FROM weekly_retros "
                "WHERE portfolio_id = %s AND week_start = %s "
                "ORDER BY id DESC LIMIT 1",
                (portfolio_id, week_start),
            )
            existing = cur.fetchone()

            # Phase 4.6 lock validation: when the persisted row is reviewed,
            # reject grade changes unless the same payload clears
            # reviewed_at. Frontend disables the selectors; this is the
            # defense-in-depth backstop for direct API hits.
            if existing and existing.get("reviewed_at") and reviewed_at:
                graded_diff = (
                    existing.get("execution_grade") != execution_grade_val
                    or existing.get("process_grade") != process_grade_val
                    or existing.get("pnl_grade") != pnl_grade_val
                    or existing.get("week_grade") != week_grade_val
                    or bool(existing.get("overall_override")) != overall_override
                )
                if graded_diff:
                    raise WeeklyRetroLockedError(
                        "grade locked; un-review to edit"
                    )

            if existing:
                cur.execute(
                    "UPDATE weekly_retros SET "
                    "  week_grade = %s, best_decision = %s, worst_decision = %s, "
                    "  rule_change = %s, rule_change_text = %s, "
                    "  weekly_thoughts = %s, "
                    "  execution_grade = %s, process_grade = %s, pnl_grade = %s, "
                    "  overall_override = %s, reviewed_at = %s, "
                    "  deleted_at = NULL, updated_at = NOW() "
                    "WHERE id = %s "
                    "RETURNING id",
                    (week_grade_val, best_decision, worst_decision,
                     rule_change, rule_change_text, weekly_thoughts,
                     execution_grade_val, process_grade_val, pnl_grade_val,
                     overall_override, reviewed_at,
                     existing["id"]),
                )
                retro_id = cur.fetchone()["id"]
            else:
                cur.execute(
                    "INSERT INTO weekly_retros "
                    "  (portfolio_id, week_start, week_grade, best_decision, "
                    "   worst_decision, rule_change, rule_change_text, "
                    "   weekly_thoughts, "
                    "   execution_grade, process_grade, pnl_grade, "
                    "   overall_override, reviewed_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                    "RETURNING id",
                    (portfolio_id, week_start, week_grade_val, best_decision,
                     worst_decision, rule_change, rule_change_text,
                     weekly_thoughts,
                     execution_grade_val, process_grade_val, pnl_grade_val,
                     overall_override, reviewed_at),
                )
                retro_id = cur.fetchone()["id"]

            _replace_weekly_retro_ticker_grades(cur, retro_id, ticker_grades or {})

            # Re-fetch the persisted row + children in the same transaction
            # so the return shape is authoritative (post-trigger, post-default).
            cur.execute(
                f"SELECT {_WEEKLY_RETRO_SELECT_COLS} "
                "FROM weekly_retros r JOIN portfolios p ON p.id = r.portfolio_id "
                "WHERE r.id = %s",
                (retro_id,),
            )
            parent = dict(cur.fetchone())
            tg_map = _fetch_ticker_grades_for_retros(cur, [retro_id])
            conn.commit()
            return _serialize_weekly_retro(parent, tg_map.get(retro_id, {}))


def soft_delete_weekly_retro(retro_id: int) -> bool:
    """Set deleted_at = NOW() on the parent retro. Children are left in
    place (CASCADE only fires on hard delete) so a future revive via
    upsert restores the full picture intact. Returns True on row hit."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE weekly_retros SET deleted_at = NOW() "
                "WHERE id = %s AND deleted_at IS NULL RETURNING id",
                (retro_id,),
            )
            hit = cur.fetchone()
            conn.commit()
            return hit is not None


# ============================================
# WEEKLY RETRO SNAPSHOTS (Migration 028, Phase 4)
# ============================================
# Image attachments on weekly retros. Metadata lives in
# weekly_retro_snapshots; bytes live in Cloudflare R2 via the upload_blob
# helper in r2_storage.py. Helpers below take portfolio_name as the first
# arg to match the convention of every other public helper in this module,
# even though the retro_id alone is sufficient to identify the row (the
# portfolio is enforced via the parent retro's portfolio_id FK and RLS by
# user_id). Soft-delete via deleted_at — the bytes intentionally remain in
# R2 for v1 (storage is cheap; a future sweep job can reclaim).


def _serialize_snapshot(row: dict, r2_public_url: str) -> dict:
    """Serialize a snapshots row for the wire. Composes view_url from the
    R2_PUBLIC_URL env value + storage_ref. If R2_PUBLIC_URL is unset (local
    dev without R2 configured), view_url falls back to the bare storage_ref
    so the frontend doesn't render a broken absolute URL pointing at
    nothing."""
    storage_ref = row.get("storage_ref") or ""
    if r2_public_url and storage_ref:
        view_url = f"{r2_public_url.rstrip('/')}/{storage_ref}"
    else:
        view_url = storage_ref
    return {
        "id": row["id"],
        "weekly_retro_id": row["weekly_retro_id"],
        "storage_ref": storage_ref,
        "view_url": view_url,
        "file_name": row.get("file_name"),
        "mime_type": row.get("mime_type"),
        "file_size_bytes": row.get("file_size_bytes"),
        "width": row.get("width"),
        "height": row.get("height"),
        "sort_order": row.get("sort_order") or 0,
        "caption": row.get("caption") or "",
        "created_at": (
            row["created_at"].isoformat()
            if row.get("created_at") and hasattr(row["created_at"], "isoformat")
            else row.get("created_at")
        ),
    }


def _resolve_retro_owned_by_portfolio(cur, retro_id: int, portfolio_name: str) -> bool:
    """Verify that retro_id exists, belongs to the given portfolio, and is
    visible to the current app.user_id via RLS. Returns True on hit, False
    on miss (used to surface a 404 instead of leaking parent retro info via
    a constraint error during INSERT)."""
    cur.execute(
        "SELECT 1 FROM weekly_retros r JOIN portfolios p ON p.id = r.portfolio_id "
        "WHERE r.id = %s AND p.name = %s AND r.deleted_at IS NULL",
        (retro_id, portfolio_name),
    )
    return cur.fetchone() is not None


def save_weekly_retro_snapshot(
    portfolio_name: str,
    retro_id: int,
    storage_ref: str,
    *,
    file_name: str | None = None,
    mime_type: str | None = None,
    file_size_bytes: int | None = None,
    width: int | None = None,
    height: int | None = None,
) -> dict | None:
    """INSERT a snapshot row attached to the given retro. Returns the
    serialized row (including view_url) on success, or None if the retro
    doesn't exist / isn't visible to the current tenant."""
    r2_public = (os.environ.get("R2_PUBLIC_URL") or "").rstrip("/")
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if not _resolve_retro_owned_by_portfolio(cur, retro_id, portfolio_name):
                return None
            cur.execute(
                "INSERT INTO weekly_retro_snapshots "
                "  (weekly_retro_id, storage_ref, file_name, mime_type, "
                "   file_size_bytes, width, height) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                "RETURNING id, weekly_retro_id, storage_ref, file_name, "
                "          mime_type, file_size_bytes, width, height, "
                "          sort_order, caption, created_at",
                (retro_id, storage_ref, file_name, mime_type,
                 file_size_bytes, width, height),
            )
            row = dict(cur.fetchone())
            conn.commit()
            return _serialize_snapshot(row, r2_public)


def list_weekly_retro_snapshots(portfolio_name: str, retro_id: int) -> list[dict] | None:
    """Return all live snapshots for the retro, ordered by (sort_order,
    created_at). Returns None if the retro is missing / not visible to the
    current tenant (caller maps to 404). Returns [] if the retro exists
    but has no snapshots."""
    r2_public = (os.environ.get("R2_PUBLIC_URL") or "").rstrip("/")
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if not _resolve_retro_owned_by_portfolio(cur, retro_id, portfolio_name):
                return None
            cur.execute(
                "SELECT id, weekly_retro_id, storage_ref, file_name, "
                "       mime_type, file_size_bytes, width, height, "
                "       sort_order, caption, created_at "
                "FROM weekly_retro_snapshots "
                "WHERE weekly_retro_id = %s AND deleted_at IS NULL "
                "ORDER BY sort_order, created_at",
                (retro_id,),
            )
            return [_serialize_snapshot(dict(r), r2_public) for r in cur.fetchall()]


def verify_retro_ownership(portfolio_name: str, retro_id: int) -> bool:
    """Phase 4.1 — public wrapper around _resolve_retro_owned_by_portfolio
    for endpoints (e.g., POST /api/weekly-retros/{retro_id}/thoughts-images)
    that need an ownership check without an accompanying INSERT/SELECT.
    Opens its own connection. Returns True if the retro exists, is not
    soft-deleted, and is visible to the current tenant via RLS."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            return _resolve_retro_owned_by_portfolio(cur, retro_id, portfolio_name)


def soft_delete_weekly_retro_snapshot(snapshot_id: int) -> bool:
    """Set deleted_at = NOW() on a snapshot row. Returns True on row hit.
    RLS scopes the UPDATE to the current tenant — a cross-tenant snapshot_id
    misses the WHERE clause and returns False, which the caller surfaces as
    404 (NOT a 403 — would leak existence). Bytes remain in R2; future
    sweep job can reclaim."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE weekly_retro_snapshots SET deleted_at = NOW() "
                "WHERE id = %s AND deleted_at IS NULL RETURNING id",
                (snapshot_id,),
            )
            hit = cur.fetchone()
            conn.commit()
            return hit is not None


# ============================================
# TAG SYSTEM (Migration 026, Phase 1)
# ============================================
# Polymorphic, portfolio-scoped, user-created tags. Two tables: `tags` (the
# palette) and `tag_assignments` (the polymorphic many-to-many to entities
# like weekly_retro). Helpers accept the portfolio NAME (matches the rest of
# this module) and return dicts shaped for direct API serialization.
#
# Soft-delete semantics: a soft-deleted tag's assignments stay live in the DB
# but become invisible to the UI because load_tag_assignments LEFT JOINs
# tags filtered by tags.deleted_at IS NULL. Un-deleting a tag (clearing
# deleted_at) reactivates every historical assignment automatically.

# Closed palette (matches the design TAG_PALETTE: rose, amber, emerald, sky,
# violet). Server-side validation rejects anything else with
# {"error": "invalid_color"} so the frontend's TAG_PALETTE fallback never
# silently maps an unknown tone to sky.
_TAG_COLOR_VOCAB = ("rose", "amber", "emerald", "sky", "violet")
_TAG_ENTITY_TYPES = ("weekly_retro", "daily_journal", "trades_summary")


def _serialize_tag(row: dict) -> dict:
    """Shared serialization for tags. Used by load/create/update so reads and
    writes return identical shapes."""
    return {
        "id": row["id"],
        "portfolio": row["portfolio"],
        "name": row["name"],
        "color": row["color"],
        "created_at": (
            row["created_at"].isoformat()
            if row.get("created_at") and hasattr(row["created_at"], "isoformat")
            else row.get("created_at")
        ),
        "updated_at": (
            row["updated_at"].isoformat()
            if row.get("updated_at") and hasattr(row["updated_at"], "isoformat")
            else row.get("updated_at")
        ),
    }


def _serialize_tag_assignment(row: dict) -> dict:
    """Shared serialization for assignment rows joined with their tag's
    display fields (tag_name, tag_color)."""
    return {
        "id": row["id"],
        "tag_id": row["tag_id"],
        "tag_name": row["tag_name"],
        "tag_color": row["tag_color"],
        "entity_type": row["entity_type"],
        "entity_id": row["entity_id"],
        "created_at": (
            row["created_at"].isoformat()
            if row.get("created_at") and hasattr(row["created_at"], "isoformat")
            else row.get("created_at")
        ),
    }


def load_tags(portfolio_name: str) -> list[dict]:
    """Return live tags for the given portfolio, oldest first. RLS scopes by
    user_id so this is safe to call without an explicit user filter."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT t.id, p.name AS portfolio, t.name, t.color, "
                "       t.created_at, t.updated_at "
                "FROM tags t JOIN portfolios p ON p.id = t.portfolio_id "
                "WHERE p.name = %s AND t.deleted_at IS NULL "
                "ORDER BY t.created_at ASC",
                (portfolio_name,),
            )
            return [_serialize_tag(dict(r)) for r in cur.fetchall()]


def create_tag(portfolio_name: str, name: str, color: str) -> dict:
    """Insert a new tag. Validates name (1-60 chars after strip) and color
    (must be in the closed palette). Raises ValueError on bad input. Lets
    psycopg2.errors.UniqueViolation propagate on case-insensitive collision
    so the endpoint can translate to a clean error response."""
    name = (name or "").strip()
    if not name:
        raise ValueError("Tag name cannot be empty")
    if len(name) > 60:
        raise ValueError("Tag name must be 1-60 characters")
    if color not in _TAG_COLOR_VOCAB:
        raise ValueError("invalid_color")
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
            prow = cur.fetchone()
            if not prow:
                raise ValueError(f"Portfolio '{portfolio_name}' not found")
            portfolio_id = prow["id"]
            cur.execute(
                "INSERT INTO tags (portfolio_id, name, color) "
                "VALUES (%s, %s, %s) "
                "RETURNING id, %s AS portfolio, name, color, created_at, updated_at",
                (portfolio_id, name, color, portfolio_name),
            )
            row = dict(cur.fetchone())
            conn.commit()
            return _serialize_tag(row)


def update_tag(
    tag_id: int, *, name: str | None = None, color: str | None = None
) -> dict | None:
    """Patch a tag's name and/or color. Returns the updated row, or None if
    not found / already soft-deleted. Whitelist-only — unknown kwargs are
    ignored. Stamps updated_at = NOW() whenever any field changes."""
    sets: list[str] = []
    params: list = []
    if name is not None:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("Tag name cannot be empty")
        if len(cleaned) > 60:
            raise ValueError("Tag name must be 1-60 characters")
        sets.append("name = %s")
        params.append(cleaned)
    if color is not None:
        if color not in _TAG_COLOR_VOCAB:
            raise ValueError("invalid_color")
        sets.append("color = %s")
        params.append(color)
    if not sets:
        # No-op update — return the current row for caller convenience.
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT t.id, p.name AS portfolio, t.name, t.color, "
                    "       t.created_at, t.updated_at "
                    "FROM tags t JOIN portfolios p ON p.id = t.portfolio_id "
                    "WHERE t.id = %s AND t.deleted_at IS NULL",
                    (tag_id,),
                )
                row = cur.fetchone()
                return _serialize_tag(dict(row)) if row else None
    sets.append("updated_at = NOW()")
    params.append(tag_id)
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"UPDATE tags SET {', '.join(sets)} "
                f"WHERE id = %s AND deleted_at IS NULL RETURNING id",
                tuple(params),
            )
            hit = cur.fetchone()
            if not hit:
                return None
            cur.execute(
                "SELECT t.id, p.name AS portfolio, t.name, t.color, "
                "       t.created_at, t.updated_at "
                "FROM tags t JOIN portfolios p ON p.id = t.portfolio_id "
                "WHERE t.id = %s",
                (tag_id,),
            )
            row = dict(cur.fetchone())
            conn.commit()
            return _serialize_tag(row)


def soft_delete_tag(tag_id: int) -> bool:
    """Soft-delete a tag (sets deleted_at = NOW()). Assignments are LEFT
    untouched in the DB; load_tag_assignments hides them via the JOIN
    filter on tags.deleted_at IS NULL, so un-deleting a tag reactivates
    every historical assignment without a follow-up write."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE tags SET deleted_at = NOW() "
                "WHERE id = %s AND deleted_at IS NULL RETURNING id",
                (tag_id,),
            )
            hit = cur.fetchone()
            conn.commit()
            return hit is not None


def load_tag_assignments(entity_type: str, entity_id: int) -> list[dict]:
    """Return live assignments for one entity, joined with each tag's
    display fields. Soft-deleted tags' assignments are filtered out via the
    JOIN predicate on tags.deleted_at IS NULL — that's how a soft-deleted
    tag visually disappears without us having to mass-update its children."""
    if entity_type not in _TAG_ENTITY_TYPES:
        raise ValueError(f"Unknown entity_type: {entity_type}")
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT a.id, a.tag_id, t.name AS tag_name, t.color AS tag_color, "
                "       a.entity_type, a.entity_id, a.created_at "
                "FROM tag_assignments a "
                "JOIN tags t ON t.id = a.tag_id AND t.deleted_at IS NULL "
                "WHERE a.entity_type = %s AND a.entity_id = %s "
                "  AND a.deleted_at IS NULL "
                "ORDER BY a.created_at ASC",
                (entity_type, entity_id),
            )
            return [_serialize_tag_assignment(dict(r)) for r in cur.fetchall()]


def count_live_tag_assignments(entity_type: str, entity_id: int) -> int:
    """Count of currently-live assignments on an entity. Used by the API to
    enforce the max-10-per-entity cap. Counts only rows whose tag is also
    live (mirrors what load_tag_assignments returns to the client)."""
    if entity_type not in _TAG_ENTITY_TYPES:
        raise ValueError(f"Unknown entity_type: {entity_type}")
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM tag_assignments a "
                "JOIN tags t ON t.id = a.tag_id AND t.deleted_at IS NULL "
                "WHERE a.entity_type = %s AND a.entity_id = %s "
                "  AND a.deleted_at IS NULL",
                (entity_type, entity_id),
            )
            return int(cur.fetchone()[0])


def create_tag_assignment(
    tag_id: int, entity_type: str, entity_id: int
) -> dict:
    """Attach a tag to an entity. Idempotent restore semantics:
      - If a LIVE assignment already exists for the same triple → return it
        (re-attach is a no-op, not an error).
      - If a SOFT-DELETED assignment exists → UPDATE deleted_at = NULL and
        return it. ID stability matters for any future per-assignment
        metadata.
      - Otherwise INSERT a new row.

    portfolio_id is sourced from the parent tag (RLS enforces tenant
    isolation; the tag's portfolio is the assignment's portfolio).

    Raises ValueError if entity_type is unknown or the tag doesn't exist
    (or has been soft-deleted)."""
    if entity_type not in _TAG_ENTITY_TYPES:
        raise ValueError(f"Unknown entity_type: {entity_type}")
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Resolve the tag's portfolio_id. Reject if tag is missing or
            # soft-deleted — attaching a deleted tag would create an
            # invisible assignment.
            cur.execute(
                "SELECT portfolio_id FROM tags "
                "WHERE id = %s AND deleted_at IS NULL",
                (tag_id,),
            )
            trow = cur.fetchone()
            if not trow:
                raise ValueError(f"Tag {tag_id} not found")
            portfolio_id = trow["portfolio_id"]

            # Look for any existing row (live OR soft-deleted) for this
            # triple. The partial unique guarantees at most one live row;
            # if multiple soft-deleted rows somehow exist we revive the
            # newest.
            cur.execute(
                "SELECT id, deleted_at FROM tag_assignments "
                "WHERE tag_id = %s AND entity_type = %s AND entity_id = %s "
                "ORDER BY id DESC LIMIT 1",
                (tag_id, entity_type, entity_id),
            )
            existing = cur.fetchone()

            if existing and existing["deleted_at"] is None:
                assignment_id = existing["id"]
            elif existing and existing["deleted_at"] is not None:
                cur.execute(
                    "UPDATE tag_assignments SET deleted_at = NULL "
                    "WHERE id = %s RETURNING id",
                    (existing["id"],),
                )
                assignment_id = cur.fetchone()["id"]
            else:
                cur.execute(
                    "INSERT INTO tag_assignments "
                    "  (portfolio_id, tag_id, entity_type, entity_id) "
                    "VALUES (%s, %s, %s, %s) RETURNING id",
                    (portfolio_id, tag_id, entity_type, entity_id),
                )
                assignment_id = cur.fetchone()["id"]

            cur.execute(
                "SELECT a.id, a.tag_id, t.name AS tag_name, t.color AS tag_color, "
                "       a.entity_type, a.entity_id, a.created_at "
                "FROM tag_assignments a "
                "JOIN tags t ON t.id = a.tag_id "
                "WHERE a.id = %s",
                (assignment_id,),
            )
            row = dict(cur.fetchone())
            conn.commit()
            return _serialize_tag_assignment(row)


def soft_delete_tag_assignment(assignment_id: int) -> bool:
    """Detach a tag (soft-delete the assignment row). Returns True on hit.
    Soft-delete preserves the row so a subsequent re-attach can revive it
    (idempotent restore in create_tag_assignment), keeping ids stable."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE tag_assignments SET deleted_at = NOW() "
                "WHERE id = %s AND deleted_at IS NULL RETURNING id",
                (assignment_id,),
            )
            hit = cur.fetchone()
            conn.commit()
            return hit is not None


# ============================================
# PRICE REFRESH OPERATIONS
# ============================================

def refresh_open_position_prices(portfolio_name):
    """
    Fetch current market prices and update unrealized_pl for all open positions.

    Args:
        portfolio_name: Portfolio to refresh

    Returns:
        dict: Summary of refresh operation
    """
    import yfinance as yf

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Try to find portfolio - check both exact match and partial match
            cur.execute("SELECT id, name FROM portfolios WHERE name = %s", (portfolio_name,))
            result = cur.fetchone()

            if not result:
                # Try partial match (in case of naming differences)
                # Map common display names to database names
                name_mappings = {
                    'CanSlim (Main)': ['CanSlim', 'CanSlim (Main)', 'Canslim'],
                    'TQQQ Strategy': ['TQQQ', 'TQQQ Strategy'],
                    '457B Plan': ['457B', '457B Plan']
                }

                # Try to find a match
                for display_name, db_names in name_mappings.items():
                    if portfolio_name in db_names or any(name in portfolio_name for name in db_names):
                        for db_name in db_names:
                            cur.execute("SELECT id, name FROM portfolios WHERE name = %s", (db_name,))
                            result = cur.fetchone()
                            if result:
                                break
                    if result:
                        break

            if not result:
                # List available portfolios for debugging
                cur.execute("SELECT name FROM portfolios")
                available = [r[0] for r in cur.fetchall()]
                return {
                    'updated': 0,
                    'error': f"Portfolio '{portfolio_name}' not found. Available: {', '.join(available)}"
                }

            portfolio_id = result[0]
            actual_name = result[1]

            # Get all open positions
            cur.execute("""
                SELECT id, ticker, shares, avg_entry, total_cost
                FROM trades_summary
                WHERE portfolio_id = %s AND status = 'OPEN'
            """, (portfolio_id,))

            open_positions = cur.fetchall()

            if not open_positions:
                return {'updated': 0, 'message': 'No open positions to refresh'}

            # Fetch current prices for all tickers
            tickers = list(set([pos[1] for pos in open_positions]))  # Unique tickers

            try:
                live_data = yf.download(tickers, period="1d", progress=False)['Close']
                if len(tickers) == 1:
                    live_prices = {tickers[0]: float(live_data.iloc[-1])}
                else:
                    live_prices = {ticker: float(live_data[ticker].iloc[-1])
                                   for ticker in tickers if ticker in live_data.columns}
            except Exception as e:
                return {'updated': 0, 'error': f'Price fetch failed: {str(e)}'}

            # Update each position
            updated_count = 0
            for pos_id, ticker, shares, avg_entry, total_cost in open_positions:
                if ticker in live_prices:
                    current_price = live_prices[ticker]
                    # Convert Decimal to float for arithmetic operations
                    shares_float = float(shares)
                    total_cost_float = float(total_cost)

                    market_value = shares_float * current_price
                    unrealized_pl = market_value - total_cost_float
                    return_pct = (unrealized_pl / total_cost_float * 100) if total_cost_float > 0 else 0

                    cur.execute("""
                        UPDATE trades_summary
                        SET unrealized_pl = %s,
                            return_pct = %s,
                            value = %s
                        WHERE id = %s
                    """, (unrealized_pl, return_pct, market_value, pos_id))

                    updated_count += 1

            conn.commit()

            # Clear cache
            load_summary.clear()

            return {
                'updated': updated_count,
                'total_positions': len(open_positions),
                'message': f'Updated {updated_count}/{len(open_positions)} positions'
            }


# ============================================
# TRADE IMAGES MANAGEMENT
# ============================================
def save_trade_image(portfolio_name: str, trade_id: str, ticker: str,
                     image_type: str, image_url: str, file_name: str = None):
    """
    Save trade image metadata to database.

    Args:
        portfolio_name: Portfolio name (e.g., 'CanSlim')
        trade_id: Trade ID
        ticker: Stock ticker
        image_type: 'weekly', 'daily', or 'exit'
        image_url: R2 object key or URL
        file_name: Original file name (optional)

    Returns:
        Database row ID if successful, None if failed
    """
    try:
        # Map display name to database name
        port_map = {
            'CanSlim (Main)': 'CanSlim',
            'TQQQ Strategy': 'TQQQ Strategy',
            '457B Plan': '457B Plan'
        }
        db_portfolio_name = port_map.get(portfolio_name, portfolio_name)

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get portfolio ID
                cur.execute("SELECT id FROM portfolios WHERE name = %s", (db_portfolio_name,))
                result = cur.fetchone()
                if not result:
                    raise ValueError(f"Portfolio '{db_portfolio_name}' not found")

                portfolio_id = result[0]

                # Insert image record (multiple images per type allowed)
                query = """
                    INSERT INTO trade_images
                        (portfolio_id, trade_id, ticker, image_type, image_url, file_name)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """

                cur.execute(query, (portfolio_id, trade_id, ticker, image_type,
                                   image_url, file_name))

                row_id = cur.fetchone()[0]
                conn.commit()
                try:
                    get_trade_images.clear()
                except Exception:
                    pass
                return row_id

    except Exception as e:
        print(f"Failed to save trade image: {e}")
        return None


@ttl_cache(ttl=120, show_spinner=False)
def get_trade_images(portfolio_name: str, trade_id: str):
    """
    Get all images for a specific trade. Cached 2 min; invalidated by
    save/delete_trade_image calls below.

    Args:
        portfolio_name: Portfolio name
        trade_id: Trade ID

    Returns:
        List of dictionaries with image metadata
    """
    try:
        # Map display name to database name
        port_map = {
            'CanSlim (Main)': 'CanSlim',
            'TQQQ Strategy': 'TQQQ Strategy',
            '457B Plan': '457B Plan'
        }
        db_portfolio_name = port_map.get(portfolio_name, portfolio_name)

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get portfolio ID
                cur.execute("SELECT id FROM portfolios WHERE name = %s", (db_portfolio_name,))
                result = cur.fetchone()
                if not result:
                    return []

                portfolio_id = result[0]

                # Get all images for this trade
                query = """
                    SELECT id, ticker, image_type, image_url, file_name, uploaded_at
                    FROM trade_images
                    WHERE portfolio_id = %s AND trade_id = %s
                    ORDER BY image_type, uploaded_at DESC
                """

                cur.execute(query, (portfolio_id, trade_id))
                rows = cur.fetchall()

                images = []
                for row in rows:
                    images.append({
                        'id': row[0],
                        'ticker': row[1],
                        'image_type': row[2],
                        'image_url': row[3],
                        'file_name': row[4],
                        'uploaded_at': row[5]
                    })

                return images

    except Exception as e:
        print(f"Failed to get trade images: {e}")
        return []


def delete_trade_image(portfolio_name: str, trade_id: str, image_type: str):
    """
    Delete a specific trade image from database.

    Args:
        portfolio_name: Portfolio name
        trade_id: Trade ID
        image_type: 'weekly', 'daily', or 'exit'

    Returns:
        True if successful, False if failed
    """
    try:
        # Map display name to database name
        port_map = {
            'CanSlim (Main)': 'CanSlim',
            'TQQQ Strategy': 'TQQQ Strategy',
            '457B Plan': '457B Plan'
        }
        db_portfolio_name = port_map.get(portfolio_name, portfolio_name)

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get portfolio ID
                cur.execute("SELECT id FROM portfolios WHERE name = %s", (db_portfolio_name,))
                result = cur.fetchone()
                if not result:
                    return False

                portfolio_id = result[0]

                # Delete image record
                query = """
                    DELETE FROM trade_images
                    WHERE portfolio_id = %s AND trade_id = %s AND image_type = %s
                """

                cur.execute(query, (portfolio_id, trade_id, image_type))
                conn.commit()
                try:
                    get_trade_images.clear()
                except Exception:
                    pass
                return True

    except Exception as e:
        print(f"Failed to delete trade image: {e}")
        return False


def delete_trade_image_by_id(image_id: int):
    """Delete a single trade image by its database ID."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get image_url before deleting (for R2 cleanup)
                cur.execute("SELECT image_url FROM trade_images WHERE id = %s", (image_id,))
                row = cur.fetchone()
                image_url = row[0] if row else None

                cur.execute("DELETE FROM trade_images WHERE id = %s", (image_id,))
                conn.commit()
                try:
                    get_trade_images.clear()
                except Exception:
                    pass
                return image_url
    except Exception as e:
        print(f"Failed to delete trade image by id: {e}")
        return None


def delete_all_trade_images_db(portfolio_name: str, trade_id: str):
    """
    Delete all image records for a specific trade.

    Args:
        portfolio_name: Portfolio name
        trade_id: Trade ID

    Returns:
        True if successful, False if failed
    """
    try:
        # Map display name to database name
        port_map = {
            'CanSlim (Main)': 'CanSlim',
            'TQQQ Strategy': 'TQQQ Strategy',
            '457B Plan': '457B Plan'
        }
        db_portfolio_name = port_map.get(portfolio_name, portfolio_name)

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get portfolio ID
                cur.execute("SELECT id FROM portfolios WHERE name = %s", (db_portfolio_name,))
                result = cur.fetchone()
                if not result:
                    return False

                portfolio_id = result[0]

                # Delete all image records for this trade
                query = """
                    DELETE FROM trade_images
                    WHERE portfolio_id = %s AND trade_id = %s
                """

                cur.execute(query, (portfolio_id, trade_id))
                conn.commit()
                try:
                    get_trade_images.clear()
                except Exception:
                    pass
                return True

    except Exception as e:
        print(f"Failed to delete all trade images: {e}")
        return False


# ============================================
# TRADE FUNDAMENTALS (Vision API Extraction)
# ============================================
def save_trade_fundamentals(portfolio_name: str, trade_id: str, ticker: str,
                            data: dict, image_id: int = None):
    """
    Save extracted fundamental data from a MarketSurge screenshot.

    Args:
        portfolio_name: Portfolio name
        trade_id: Trade ID
        ticker: Stock ticker
        data: Dictionary of extracted fundamentals
        image_id: Optional trade_images.id this was extracted from

    Returns:
        Row ID if successful, None if failed
    """
    try:
        import json

        port_map = {
            'CanSlim (Main)': 'CanSlim',
            'TQQQ Strategy': 'TQQQ Strategy',
            '457B Plan': '457B Plan'
        }
        db_portfolio_name = port_map.get(portfolio_name, portfolio_name)

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM portfolios WHERE name = %s", (db_portfolio_name,))
                result = cur.fetchone()
                if not result:
                    raise ValueError(f"Portfolio '{db_portfolio_name}' not found")

                portfolio_id = result[0]

                query = """
                    INSERT INTO trade_fundamentals
                        (portfolio_id, trade_id, ticker, image_id,
                         composite_rating, eps_rating, rs_rating, group_rs_rating,
                         smr_rating, acc_dis_rating, timeliness_rating, sponsorship_rating,
                         eps_growth_rate, ud_vol_ratio,
                         mgmt_own_pct, banks_own_pct, funds_own_pct, num_funds,
                         price, market_cap, industry_group, industry_group_rank,
                         raw_json)
                    VALUES (%s, %s, %s, %s,
                            %s, %s, %s, %s,
                            %s, %s, %s, %s,
                            %s, %s,
                            %s, %s, %s, %s,
                            %s, %s, %s, %s,
                            %s)
                    RETURNING id
                """

                cur.execute(query, (
                    portfolio_id, trade_id, ticker, image_id,
                    data.get('composite_rating'), data.get('eps_rating'),
                    data.get('rs_rating'), data.get('group_rs_rating'),
                    data.get('smr_rating'), data.get('acc_dis_rating'),
                    data.get('timeliness_rating'), data.get('sponsorship_rating'),
                    data.get('eps_growth_rate'), data.get('ud_vol_ratio'),
                    data.get('mgmt_own_pct'), data.get('banks_own_pct'),
                    data.get('funds_own_pct'), data.get('num_funds'),
                    data.get('price'), data.get('market_cap'),
                    data.get('industry_group'), data.get('industry_group_rank'),
                    json.dumps(data)
                ))

                row_id = cur.fetchone()[0]
                conn.commit()
                try:
                    get_trade_fundamentals.clear()
                except Exception:
                    pass
                return row_id

    except Exception as e:
        print(f"Failed to save trade fundamentals: {e}")
        return None


@ttl_cache(ttl=300, show_spinner=False)
def get_trade_fundamentals(portfolio_name: str, trade_id: str):
    """
    Get all extracted fundamentals for a trade. Cached 5 min; invalidated
    by save_trade_fundamentals.

    Returns:
        List of dictionaries with fundamental data
    """
    try:
        port_map = {
            'CanSlim (Main)': 'CanSlim',
            'TQQQ Strategy': 'TQQQ Strategy',
            '457B Plan': '457B Plan'
        }
        db_portfolio_name = port_map.get(portfolio_name, portfolio_name)

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM portfolios WHERE name = %s", (db_portfolio_name,))
                result = cur.fetchone()
                if not result:
                    return []

                portfolio_id = result[0]

                # DISTINCT ON (ticker) returns one row per unique ticker —
                # the most recent extraction, since ORDER BY picks it. This
                # collapses accidental duplicate extractions from the same
                # MarketSurge upload (the "duplicate MarketSurge card" bug
                # users saw on options trades) while still allowing legitimate
                # multiple tickers under one trade_id (e.g. an options trade
                # where the user also uploaded the underlying's screenshot).
                query = """
                    SELECT DISTINCT ON (ticker)
                           id, ticker, image_id, extracted_at,
                           composite_rating, eps_rating, rs_rating, group_rs_rating,
                           smr_rating, acc_dis_rating, timeliness_rating, sponsorship_rating,
                           eps_growth_rate, ud_vol_ratio,
                           mgmt_own_pct, banks_own_pct, funds_own_pct, num_funds,
                           price, market_cap, industry_group, industry_group_rank,
                           raw_json
                    FROM trade_fundamentals
                    WHERE portfolio_id = %s AND trade_id = %s
                    ORDER BY ticker, extracted_at DESC
                """

                cur.execute(query, (portfolio_id, trade_id))
                rows = cur.fetchall()

                fundamentals = []
                for row in rows:
                    fundamentals.append({
                        'id': row[0], 'ticker': row[1], 'image_id': row[2],
                        'extracted_at': row[3],
                        'composite_rating': row[4], 'eps_rating': row[5],
                        'rs_rating': row[6], 'group_rs_rating': row[7],
                        'smr_rating': row[8], 'acc_dis_rating': row[9],
                        'timeliness_rating': row[10], 'sponsorship_rating': row[11],
                        'eps_growth_rate': row[12], 'ud_vol_ratio': row[13],
                        'mgmt_own_pct': row[14], 'banks_own_pct': row[15],
                        'funds_own_pct': row[16], 'num_funds': row[17],
                        'price': row[18], 'market_cap': row[19],
                        'industry_group': row[20], 'industry_group_rank': row[21],
                        'raw_json': row[22]
                    })

                return fundamentals

    except Exception as e:
        print(f"Failed to get trade fundamentals: {e}")
        return []


# ============================================
# DRAWDOWN NOTES (Drawdown Discipline tab)
# ============================================
@ttl_cache(ttl=60, show_spinner=False)
def get_drawdown_notes(portfolio_name: str):
    """
    Return {(deck_level, crossing_date_str): note} for a portfolio.
    crossing_date_str is 'YYYY-MM-DD'.
    """
    try:
        port_map = {
            'CanSlim (Main)': 'CanSlim',
            'TQQQ Strategy': 'TQQQ Strategy',
            '457B Plan': '457B Plan',
        }
        db_name = port_map.get(portfolio_name, portfolio_name)
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM portfolios WHERE name = %s", (db_name,))
                r = cur.fetchone()
                if not r:
                    return {}
                pid = r[0]
                cur.execute(
                    "SELECT deck_level, crossing_date, note FROM drawdown_notes WHERE portfolio_id = %s",
                    (pid,),
                )
                return {
                    (lvl, dt.strftime('%Y-%m-%d') if hasattr(dt, 'strftime') else str(dt)): (note or '')
                    for lvl, dt, note in cur.fetchall()
                }
    except Exception as e:
        print(f"Failed to load drawdown notes: {e}")
        return {}


@ttl_cache(ttl=60, show_spinner=False)
def get_trade_lessons(portfolio_name: str):
    """
    Return {trade_id: (note, category)} for a portfolio.
    """
    try:
        port_map = {
            'CanSlim (Main)': 'CanSlim',
            'TQQQ Strategy': 'TQQQ Strategy',
            '457B Plan': '457B Plan',
        }
        db_name = port_map.get(portfolio_name, portfolio_name)
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM portfolios WHERE name = %s", (db_name,))
                r = cur.fetchone()
                if not r:
                    return {}
                pid = r[0]
                cur.execute(
                    "SELECT trade_id, note, category FROM trade_lessons WHERE portfolio_id = %s",
                    (pid,),
                )
                return {row[0]: (row[1] or '', row[2] or '') for row in cur.fetchall()}
    except Exception as e:
        print(f"Failed to load trade lessons: {e}")
        return {}


@ttl_cache(ttl=60, show_spinner=False)
def get_rule_notes(portfolio_name: str, rule_side: str):
    """
    Return {rule_name: (note, status)} for a portfolio and side.
    rule_side: 'buy' or 'sell'.
    """
    try:
        port_map = {
            'CanSlim (Main)': 'CanSlim',
            'TQQQ Strategy': 'TQQQ Strategy',
            '457B Plan': '457B Plan',
        }
        db_name = port_map.get(portfolio_name, portfolio_name)
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM portfolios WHERE name = %s", (db_name,))
                r = cur.fetchone()
                if not r:
                    return {}
                pid = r[0]
                cur.execute(
                    "SELECT rule_name, note, status FROM rule_notes WHERE portfolio_id = %s AND rule_side = %s",
                    (pid, rule_side),
                )
                return {row[0]: (row[1] or '', row[2] or '') for row in cur.fetchall()}
    except Exception as e:
        print(f"Failed to load rule notes: {e}")
        return {}


def save_rule_note(portfolio_name: str, rule_side: str, rule_name: str, note: str, status: str = ''):
    """
    Upsert a rule note. One entry per (portfolio, rule_side, rule_name).
    """
    try:
        port_map = {
            'CanSlim (Main)': 'CanSlim',
            'TQQQ Strategy': 'TQQQ Strategy',
            '457B Plan': '457B Plan',
        }
        db_name = port_map.get(portfolio_name, portfolio_name)
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM portfolios WHERE name = %s", (db_name,))
                r = cur.fetchone()
                if not r:
                    return False
                pid = r[0]
                cur.execute(
                    """
                    INSERT INTO rule_notes (portfolio_id, rule_side, rule_name, note, status, updated_at)
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (portfolio_id, rule_side, rule_name)
                    DO UPDATE SET note = EXCLUDED.note, status = EXCLUDED.status, updated_at = CURRENT_TIMESTAMP
                    """,
                    (pid, rule_side, rule_name, note or '', status or ''),
                )
                conn.commit()
        try:
            get_rule_notes.clear()
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"Failed to save rule note: {e}")
        return False


def save_trade_lesson(portfolio_name: str, trade_id: str, note: str, category: str = ''):
    """
    Upsert a trade lesson note. One entry per (portfolio, trade_id).
    """
    try:
        port_map = {
            'CanSlim (Main)': 'CanSlim',
            'TQQQ Strategy': 'TQQQ Strategy',
            '457B Plan': '457B Plan',
        }
        db_name = port_map.get(portfolio_name, portfolio_name)
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM portfolios WHERE name = %s", (db_name,))
                r = cur.fetchone()
                if not r:
                    return False
                pid = r[0]
                cur.execute(
                    """
                    INSERT INTO trade_lessons (portfolio_id, trade_id, note, category, updated_at)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (portfolio_id, trade_id)
                    DO UPDATE SET note = EXCLUDED.note, category = EXCLUDED.category, updated_at = CURRENT_TIMESTAMP
                    """,
                    (pid, str(trade_id), note or '', category or ''),
                )
                conn.commit()
        try:
            get_trade_lessons.clear()
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"Failed to save trade lesson: {e}")
        return False


def save_drawdown_note(portfolio_name: str, deck_level: str, crossing_date, note: str):
    """
    Upsert a drawdown note. crossing_date can be a date, datetime, or YYYY-MM-DD string.
    """
    try:
        port_map = {
            'CanSlim (Main)': 'CanSlim',
            'TQQQ Strategy': 'TQQQ Strategy',
            '457B Plan': '457B Plan',
        }
        db_name = port_map.get(portfolio_name, portfolio_name)
        # Normalize date to string
        if hasattr(crossing_date, 'strftime'):
            date_str = crossing_date.strftime('%Y-%m-%d')
        else:
            date_str = str(crossing_date)[:10]
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM portfolios WHERE name = %s", (db_name,))
                r = cur.fetchone()
                if not r:
                    return False
                pid = r[0]
                cur.execute(
                    """
                    INSERT INTO drawdown_notes (portfolio_id, deck_level, crossing_date, note, updated_at)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (portfolio_id, deck_level, crossing_date)
                    DO UPDATE SET note = EXCLUDED.note, updated_at = CURRENT_TIMESTAMP
                    """,
                    (pid, deck_level, date_str, note or ''),
                )
                conn.commit()
        try:
            get_drawdown_notes.clear()
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"Failed to save drawdown note: {e}")
        return False


# ============================================
# APP CONFIG (runtime-editable settings)
# ============================================
import json as _json


def get_config(key, default=None):
    """
    Fetch a single config value by key. Returns `default` if missing or DB unavailable.
    Values are JSONB-decoded automatically.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT value FROM app_config WHERE key = %s", (key,))
                row = cur.fetchone()
                if not row:
                    return default
                val = row[0]
                # psycopg2 may already decode JSONB to dict/list/etc, but if it's a string, decode it
                if isinstance(val, str):
                    try:
                        return _json.loads(val)
                    except Exception:
                        return val
                return val
    except Exception as e:
        print(f"get_config({key}) failed: {e}")
        return default


def get_config_by_category(category):
    """
    Return all config rows in a category as list of dicts.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT key, value, value_type, category, description, updated_at, updated_by
                    FROM app_config
                    WHERE category = %s
                    ORDER BY key
                """, (category,))
                rows = cur.fetchall()
                # decode value if string
                for r in rows:
                    v = r['value']
                    if isinstance(v, str):
                        try:
                            r['value'] = _json.loads(v)
                        except Exception:
                            pass
                return rows
    except Exception as e:
        print(f"get_config_by_category({category}) failed: {e}")
        return []


def set_config(key, value, value_type=None, category=None, description=None, user='User'):
    """
    Upsert a config value. If row exists, value/updated_at/updated_by are updated;
    value_type/category/description only update when explicitly passed (non-None).
    Also writes an audit_trail entry.
    """
    try:
        # Auto-detect type if not provided and row doesn't exist
        if value_type is None:
            if isinstance(value, bool):
                value_type = 'json'  # store as bool in JSON
            elif isinstance(value, (int, float)):
                value_type = 'number'
            elif isinstance(value, str):
                # Try date format YYYY-MM-DD
                if len(value) == 10 and value[4] == '-' and value[7] == '-':
                    value_type = 'date'
                else:
                    value_type = 'string'
            else:
                value_type = 'json'

        json_val = _json.dumps(value)

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Check if exists
                cur.execute("SELECT id, value FROM app_config WHERE key = %s", (key,))
                existing = cur.fetchone()

                if existing:
                    old_val = existing[1]
                    # Update only the changed fields
                    set_parts = ["value = %s::jsonb", "updated_at = CURRENT_TIMESTAMP", "updated_by = %s"]
                    params = [json_val, user]
                    if category is not None:
                        set_parts.append("category = %s")
                        params.append(category)
                    if description is not None:
                        set_parts.append("description = %s")
                        params.append(description)
                    if value_type is not None:
                        set_parts.append("value_type = %s")
                        params.append(value_type)
                    params.append(key)
                    cur.execute(
                        f"UPDATE app_config SET {', '.join(set_parts)} WHERE key = %s",
                        params,
                    )
                    audit_action = 'CONFIG_UPDATE'
                    audit_details = f"key={key} old={old_val} new={value}"
                else:
                    cur.execute(
                        """
                        INSERT INTO app_config (key, value, value_type, category, description, updated_by)
                        VALUES (%s, %s::jsonb, %s, %s, %s, %s)
                        """,
                        (key, json_val, value_type, category or 'misc', description or '', user),
                    )
                    audit_action = 'CONFIG_CREATE'
                    audit_details = f"key={key} value={value} category={category}"

                # Audit log — uses the CanSlim portfolio as the home for global config changes
                try:
                    cur.execute("SELECT id FROM portfolios WHERE name = %s", ('CanSlim',))
                    pid_row = cur.fetchone()
                    if pid_row:
                        cur.execute(
                            """
                            INSERT INTO audit_trail (portfolio_id, username, action, trade_id, ticker, details)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            """,
                            (pid_row[0], user, audit_action, key, '', audit_details),
                        )
                except Exception as audit_e:
                    print(f"set_config audit log failed (non-fatal): {audit_e}")

                conn.commit()
        return True
    except Exception as e:
        print(f"set_config({key}) failed: {e}")
        return False


def seed_default_configs():
    """
    Idempotent seeding: insert default config rows if missing.
    Called at app startup to make sure all required keys exist.
    """
    defaults = [
        # key, default value, value_type, category, description
        ('reset_date', '2026-02-24', 'date', 'risk',
         'Date from which drawdown peak is calculated (Risk Manager + Dashboard).'),
        ('hard_decks', {
            'L1': {'pct': 7.5, 'action': 'Remove Margin', 'color': '#eab308'},
            'L2': {'pct': 12.5, 'action': 'Max 30% Invested', 'color': '#f97316'},
            'L3': {'pct': 15.0, 'action': 'Go to Cash', 'color': '#dc2626'},
        }, 'json', 'risk',
         'Hard deck drawdown thresholds (% from peak), action label, and color.'),

        # Portfolio Heat
        ('heat_threshold_pct', 2.5, 'number', 'heat',
         'Total portfolio heat target — values at or above trigger an alert.'),

        # Earnings Planner
        ('earnings_cushion', {
            'pass_pct': 10.0,
            'fail_pct': 0.0,
            'default_max_risk_pct': 0.5,
        }, 'json', 'earnings',
         'Cushion thresholds: PASS at >= pass_pct, FAIL at <= fail_pct, '
         'default max capital risk % for stress test.'),

        # Pyramid Sizer pace rules
        ('pyramid_rules', {
            'trigger_pct': 5.0,    # Last buy must be up at least this much for full add
            'alloc_pct': 20.0,     # Max % of current shares per add
        }, 'json', 'sizing',
         'Pyramid pace: full add at trigger_pct profit on last buy, '
         'capped at alloc_pct of current shares.'),

        # Position size tiers
        ('size_tiers', [
            {'label': 'Shotgun (2.5%)', 'pct': 2.5},
            {'label': 'Half (5%)', 'pct': 5.0},
            {'label': '7.5%', 'pct': 7.5},
            {'label': 'Full (10%)', 'pct': 10.0},
            {'label': '12.5%', 'pct': 12.5},
            {'label': 'Core (15%)', 'pct': 15.0},
            {'label': 'Core+1 (20%)', 'pct': 20.0},
            {'label': 'Max (25%)', 'pct': 25.0},
            {'label': '30%', 'pct': 30.0},
            {'label': '35%', 'pct': 35.0},
            {'label': '40%', 'pct': 40.0},
            {'label': '45%', 'pct': 45.0},
            {'label': '50%', 'pct': 50.0},
        ], 'json', 'sizing',
         'Position size tier dropdown options (label + weight %). '
         'Used by Position Sizer.'),

        # Custom rules (additive, never replace base rules)
        ('custom_buy_rules', [], 'json', 'rules',
         'User-added buy rules merged with the base BUY_RULES list.'),
        ('custom_sell_rules', [], 'json', 'rules',
         'User-added sell rules merged with the base SELL_RULES list.'),

        # Portfolio limits
        ('max_positions', 12, 'number', 'risk',
         'Maximum number of concurrent open positions across the portfolio. '
         'Shown on the Dashboard Live Exposure card (X/N Pos).'),
    ]
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for key, default_val, vtype, category, desc in defaults:
                    cur.execute("SELECT 1 FROM app_config WHERE key = %s", (key,))
                    if cur.fetchone():
                        continue
                    cur.execute(
                        """
                        INSERT INTO app_config (key, value, value_type, category, description, updated_by)
                        VALUES (%s, %s::jsonb, %s, %s, %s, %s)
                        """,
                        (key, _json.dumps(default_val), vtype, category, desc, 'system_seed'),
                    )
                conn.commit()
        return True
    except Exception as e:
        print(f"seed_default_configs failed: {e}")
        return False


# ============================================
# DASHBOARD EVENTS (EC markers)
# ============================================
def load_dashboard_events(scope='CanSlim'):
    """
    Return events for the given portfolio scope as a pandas DataFrame.
    Columns: id, event_date, label, category, notes, color_override,
             portfolio_scope, auto_generated, source_key, created_at, updated_at
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, event_date, label, category, notes, color_override,
                           portfolio_scope, auto_generated, source_key,
                           created_at, updated_at
                    FROM dashboard_events
                    WHERE portfolio_scope = %s OR portfolio_scope = 'All'
                    ORDER BY event_date ASC
                """, (scope,))
                rows = cur.fetchall()
                return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
                    'id', 'event_date', 'label', 'category', 'notes',
                    'color_override', 'portfolio_scope', 'auto_generated',
                    'source_key', 'created_at', 'updated_at'
                ])
    except Exception as e:
        print(f"load_dashboard_events failed: {e}")
        return pd.DataFrame()


def save_dashboard_event(event_date, label, category, notes='',
                         color_override=None, scope='CanSlim',
                         auto_generated=False, source_key=None, user='User'):
    """
    Insert or update a dashboard event. Uniqueness is (event_date, label).
    """
    try:
        # Normalize date
        if hasattr(event_date, 'strftime'):
            date_str = event_date.strftime('%Y-%m-%d')
        else:
            date_str = str(event_date)[:10]

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO dashboard_events
                        (event_date, label, category, notes, color_override,
                         portfolio_scope, auto_generated, source_key, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (event_date, label) DO UPDATE SET
                        category = EXCLUDED.category,
                        notes = EXCLUDED.notes,
                        color_override = EXCLUDED.color_override,
                        portfolio_scope = EXCLUDED.portfolio_scope,
                        auto_generated = EXCLUDED.auto_generated,
                        source_key = EXCLUDED.source_key,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (date_str, label, category, notes or '', color_override,
                     scope, auto_generated, source_key),
                )

                # Audit log
                try:
                    cur.execute("SELECT id FROM portfolios WHERE name = %s", ('CanSlim',))
                    pid_row = cur.fetchone()
                    if pid_row:
                        cur.execute(
                            """
                            INSERT INTO audit_trail (portfolio_id, username, action, trade_id, ticker, details)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            """,
                            (pid_row[0], user, 'EVENT_UPSERT', '', '',
                             f"date={date_str} label={label} cat={category}"),
                        )
                except Exception as audit_e:
                    print(f"save_dashboard_event audit failed (non-fatal): {audit_e}")

                conn.commit()
        return True
    except Exception as e:
        print(f"save_dashboard_event failed: {e}")
        return False


def update_dashboard_event(event_id, event_date=None, label=None, category=None,
                           notes=None, color_override=None, user='User'):
    """
    Edit an existing dashboard event in place. Only non-None fields are updated.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT event_date, label, category, auto_generated FROM dashboard_events WHERE id = %s",
                    (event_id,),
                )
                row = cur.fetchone()
                if not row:
                    print(f"update_dashboard_event: event id={event_id} not found")
                    return False

                old_date, old_label, old_cat = row[0], row[1], row[2]

                # Build dynamic update
                set_parts = ["updated_at = CURRENT_TIMESTAMP"]
                params = []
                if event_date is not None:
                    if hasattr(event_date, 'strftime'):
                        date_str = event_date.strftime('%Y-%m-%d')
                    else:
                        date_str = str(event_date)[:10]
                    set_parts.append("event_date = %s")
                    params.append(date_str)
                if label is not None:
                    set_parts.append("label = %s")
                    params.append(str(label).strip())
                if category is not None:
                    set_parts.append("category = %s")
                    params.append(category)
                if notes is not None:
                    set_parts.append("notes = %s")
                    params.append(notes or '')
                if color_override is not None:
                    set_parts.append("color_override = %s")
                    params.append(color_override or None)

                if len(set_parts) == 1:
                    # Nothing actually changed
                    return True

                params.append(event_id)
                cur.execute(
                    f"UPDATE dashboard_events SET {', '.join(set_parts)} WHERE id = %s",
                    params,
                )

                # Audit
                try:
                    cur.execute("SELECT id FROM portfolios WHERE name = %s", ('CanSlim',))
                    pid_row = cur.fetchone()
                    if pid_row:
                        cur.execute(
                            """
                            INSERT INTO audit_trail (portfolio_id, username, action, trade_id, ticker, details)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            """,
                            (pid_row[0], user, 'EVENT_UPDATE', '', '',
                             f"id={event_id} old=({old_date}, {old_label}, {old_cat}) "
                             f"changes={ {k: v for k, v in [('date', event_date), ('label', label), ('category', category), ('notes', notes), ('color_override', color_override)] if v is not None} }"),
                        )
                except Exception as audit_e:
                    print(f"update_dashboard_event audit failed (non-fatal): {audit_e}")

                conn.commit()
        return True
    except Exception as e:
        print(f"update_dashboard_event failed: {e}")
        return False


def delete_dashboard_event(event_id, user='User'):
    """Delete a dashboard event by ID."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Fetch label/date for audit
                cur.execute("SELECT event_date, label, auto_generated FROM dashboard_events WHERE id = %s", (event_id,))
                row = cur.fetchone()
                if not row:
                    return False
                cur.execute("DELETE FROM dashboard_events WHERE id = %s", (event_id,))

                # Audit log
                try:
                    cur.execute("SELECT id FROM portfolios WHERE name = %s", ('CanSlim',))
                    pid_row = cur.fetchone()
                    if pid_row:
                        cur.execute(
                            """
                            INSERT INTO audit_trail (portfolio_id, username, action, trade_id, ticker, details)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            """,
                            (pid_row[0], user, 'EVENT_DELETE', '', '',
                             f"id={event_id} date={row[0]} label={row[1]}"),
                        )
                except Exception:
                    pass
                conn.commit()
        return True
    except Exception as e:
        print(f"delete_dashboard_event failed: {e}")
        return False


def load_recent_audit_entries(limit=200, action_filter=None):
    """
    Load recent audit_trail rows for the Admin viewer.
    `action_filter` can be a substring (e.g. 'CONFIG' to show only config events).
    Returns a pandas DataFrame.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if action_filter:
                    cur.execute("""
                        SELECT a.timestamp, a.username, a.action, a.trade_id, a.ticker, a.details, p.name AS portfolio
                        FROM audit_trail a
                        LEFT JOIN portfolios p ON a.portfolio_id = p.id
                        WHERE a.action ILIKE %s
                        ORDER BY a.timestamp DESC
                        LIMIT %s
                    """, (f"%{action_filter}%", limit))
                else:
                    cur.execute("""
                        SELECT a.timestamp, a.username, a.action, a.trade_id, a.ticker, a.details, p.name AS portfolio
                        FROM audit_trail a
                        LEFT JOIN portfolios p ON a.portfolio_id = p.id
                        ORDER BY a.timestamp DESC
                        LIMIT %s
                    """, (limit,))
                rows = cur.fetchall()
                return pd.DataFrame(rows) if rows else pd.DataFrame(
                    columns=['timestamp', 'username', 'action', 'trade_id', 'ticker', 'details', 'portfolio']
                )
    except Exception as e:
        print(f"load_recent_audit_entries failed: {e}")
        return pd.DataFrame()


def sync_auto_events_from_config():
    """
    Keep auto-generated events in sync with their underlying config keys.
    Currently syncs RESET_DATE -> a permanent system milestone.
    """
    try:
        reset_date = get_config('reset_date', '2026-02-24')
        if reset_date:
            save_dashboard_event(
                event_date=reset_date,
                label='RESET_DATE',
                category='system',
                notes='Drawdown peak resets from this date.',
                scope='CanSlim',
                auto_generated=True,
                source_key='reset_date',
                user='system_sync',
            )
        return True
    except Exception as e:
        print(f"sync_auto_events_from_config failed: {e}")
        return False


def cleanup_duplicate_marketsurge_images(dry_run=True, user='admin'):
    """
    Find trade_images rows where the same (portfolio_id, trade_id, image_url)
    exists as BOTH 'marketsurge' AND 'entry' types. These are duplicates left
    over from the old Log Buy flow that double-saved MS screenshots.

    For each duplicate:
      1. Re-point any trade_fundamentals.image_id references from the
         marketsurge row to the matching entry row (so fundamentals stay linked
         to a valid image).
      2. Delete the marketsurge row.

    If dry_run=True (default), no changes are made — returns a preview list.
    If dry_run=False, performs the cleanup and returns the actual count.

    Returns: dict with keys {count, rows} where rows is list of
             {id, portfolio, trade_id, ticker, image_url, uploaded_at}.
    """
    result = {'count': 0, 'rows': []}
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Find marketsurge rows that have a matching entry row with same URL
                cur.execute("""
                    SELECT m.id, p.name AS portfolio, m.trade_id, m.ticker,
                           m.image_url, m.uploaded_at,
                           e.id AS entry_id
                    FROM trade_images m
                    JOIN trade_images e
                      ON e.portfolio_id = m.portfolio_id
                     AND e.trade_id = m.trade_id
                     AND e.image_type = 'entry'
                     AND e.image_url = m.image_url
                    LEFT JOIN portfolios p ON m.portfolio_id = p.id
                    WHERE m.image_type = 'marketsurge'
                    ORDER BY m.uploaded_at DESC
                """)
                rows = cur.fetchall()
                result['count'] = len(rows)
                result['rows'] = rows

                if not dry_run and rows:
                    # Step 1: re-point fundamentals.image_id from marketsurge -> entry
                    for r in rows:
                        cur.execute("""
                            UPDATE trade_fundamentals
                            SET image_id = %s
                            WHERE image_id = %s
                        """, (r['entry_id'], r['id']))
                    # Step 2: delete the duplicate marketsurge rows
                    ms_ids = [r['id'] for r in rows]
                    cur.execute("DELETE FROM trade_images WHERE id = ANY(%s)", (ms_ids,))

                    # Audit trail entry
                    try:
                        cur.execute("SELECT id FROM portfolios WHERE name = %s", ('CanSlim',))
                        pid_row = cur.fetchone()
                        if pid_row:
                            cur.execute(
                                """
                                INSERT INTO audit_trail (portfolio_id, username, action, trade_id, ticker, details)
                                VALUES (%s, %s, %s, %s, %s, %s)
                                """,
                                (pid_row['id'], user, 'CLEANUP_MS_DUPES', '', '',
                                 f"Removed {len(rows)} duplicate 'marketsurge' image rows with matching 'entry' rows"),
                            )
                    except Exception as audit_e:
                        print(f"Cleanup audit log failed (non-fatal): {audit_e}")

                    conn.commit()
        return result
    except Exception as e:
        print(f"cleanup_duplicate_marketsurge_images failed: {e}")
        return {'count': 0, 'rows': [], 'error': str(e)}


def migrate_event_categories():
    """
    One-shot migration: rename old 'personal' category.
    - Auto-generated events (e.g. RESET_DATE) -> 'system'
    - User-created events -> 'macro' (closest match to news/political milestones)
    Safe to run repeatedly (no-op if no 'personal' rows remain).
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE dashboard_events
                    SET category = 'system', updated_at = CURRENT_TIMESTAMP
                    WHERE category = 'personal' AND auto_generated = TRUE
                """)
                cur.execute("""
                    UPDATE dashboard_events
                    SET category = 'macro', updated_at = CURRENT_TIMESTAMP
                    WHERE category = 'personal' AND auto_generated = FALSE
                """)
                conn.commit()
        return True
    except Exception as e:
        print(f"migrate_event_categories failed: {e}")
        return False


# ============================================
# TEST CONNECTION
# ============================================
def test_connection():
    """
    Test database connection.
    Returns True if successful, False otherwise.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False


# ============================================
# PORTFOLIO CRUD (multi-tenant)
# ============================================
_PORTFOLIO_COLS = "id, name, starting_capital, reset_date, created_at"


def load_strategies(active_only: bool = True, portfolio_name: str | None = None) -> list[dict]:
    """Return rows from the `strategies` lookup table (Migration 019, scoped
    by Migration 038).

    Sorted by created_at ASC so seeded strategies render in their canonical
    order (CanSlim → StockTalk → 21eStrategy → newer rows). active_only
    filters out soft-disabled strategies — what GET /api/strategies returns
    by default and what log_buy validates new trades against.

    portfolio_name (Migration 038) scopes the result to strategies allowed
    in that portfolio. NULL allowed_portfolio_names = universal (visible
    everywhere). Pass None / omit to skip scoping (admin / cross-portfolio
    views).

    Tiny, uncached read by design: 5 rows today (~10 ever), so the
    simplicity of avoiding cache invalidation when the admin UI mutates
    the table is worth more than the saved sub-millisecond.

    Migration-tolerance: if allowed_portfolio_names doesn't exist yet
    (code deploy raced ahead of migration 038 apply), the SELECT falls
    back to NULL so the scoping filter behaves as "all visible" — the
    same disposition as a NULL allowed value.
    """
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            try:
                cur.execute(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name = 'strategies' "
                    "AND column_name = 'allowed_portfolio_names'"
                )
                has_scoping = cur.fetchone() is not None
            except Exception:
                has_scoping = False

            allowed_select = (
                "allowed_portfolio_names" if has_scoping
                else "NULL::text[] AS allowed_portfolio_names"
            )

            where_clauses = []
            params: list = []
            if active_only:
                where_clauses.append("is_active")
            if portfolio_name and has_scoping:
                where_clauses.append(
                    "(allowed_portfolio_names IS NULL "
                    "OR %s = ANY(allowed_portfolio_names))"
                )
                params.append(portfolio_name)

            where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
            cur.execute(
                f"SELECT name, description, color, is_active, created_at, {allowed_select} "
                f"FROM strategies"
                f"{where_sql} "
                f"ORDER BY created_at ASC",
                params,
            )
            return [dict(r) for r in cur.fetchall()]


def update_trade_strategy(portfolio_name: str, trade_id: str, strategy: str) -> bool:
    """Retag a single campaign's strategy. Returns True if a row was updated.

    Caller is responsible for validating that `strategy` exists in the
    strategies table (the FK from Migration 019 will raise IntegrityError
    on an unknown value, but callers are expected to pre-check for a
    cleaner error response). Soft-deleted rows are skipped — retagging a
    tombstoned campaign is a no-op.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE trades_summary SET strategy = %s "
                "WHERE portfolio_id = (SELECT id FROM portfolios WHERE name = %s) "
                "  AND trade_id = %s "
                "  AND deleted_at IS NULL",
                (strategy, portfolio_name, trade_id),
            )
            updated = cur.rowcount
            conn.commit()
            return updated > 0


def bulk_update_trade_strategy(
    portfolio_name: str, trade_ids: list[str], strategy: str
) -> tuple[int, list[str]]:
    """Retag many campaigns at once in a single UPDATE.

    Returns (updated_count, missing_trade_ids). missing_trade_ids is the
    subset of trade_ids that didn't match an active row — caller surfaces
    them as `failed: [...]` so the user can act on them. Strategy
    validation is the caller's job (matches the single-row helper).

    Atomic: one UPDATE … WHERE trade_id = ANY(%s) inside one transaction.
    Either all matched rows are updated or none are.
    """
    if not trade_ids:
        return 0, []
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE trades_summary SET strategy = %s "
                "WHERE portfolio_id = (SELECT id FROM portfolios WHERE name = %s) "
                "  AND trade_id = ANY(%s) "
                "  AND deleted_at IS NULL "
                "RETURNING trade_id",
                (strategy, portfolio_name, list(trade_ids)),
            )
            matched = {row[0] for row in cur.fetchall()}
            conn.commit()
            missing = [tid for tid in trade_ids if tid not in matched]
            return len(matched), missing


# Hex color regex — used by both create_strategy and update_strategy. Six
# hex digits (no shorthand), case-insensitive. Matches the regex mirrored
# in the frontend's Admin form so client and server agree on what's valid.
_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def create_strategy(
    name: str, color: str, description: str | None = None, is_active: bool = True
) -> dict:
    """Insert a new strategy row. Returns the persisted row.

    Validates name (1–60 chars, non-empty after strip) and color (six-hex
    format) up front so the DB never receives garbage. Raises ValueError
    for input errors and lets psycopg2's UniqueViolation propagate when
    the name is already taken — endpoint catches and surfaces both.
    """
    name = (name or "").strip()
    if not name or len(name) > 60:
        raise ValueError("Strategy name must be 1-60 characters")
    if not _HEX_COLOR_RE.match(color or ""):
        raise ValueError("Color must be a six-digit hex string like '#6366f1'")
    description = (description or "").strip() or None
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "INSERT INTO strategies (name, description, color, is_active) "
                "VALUES (%s, %s, %s, %s) "
                "RETURNING name, description, color, is_active, created_at",
                (name, description, color, is_active),
            )
            row = cur.fetchone()
            conn.commit()
            return dict(row)


def update_strategy(name: str, **fields) -> dict | None:
    """Update opt-in fields on a strategy. Returns the updated row, or None
    if no row exists with that name.

    Recognized fields: description, color, is_active. Unrecognized keys are
    silently ignored so a forward-compatible body (e.g. a future `icon`
    field) doesn't break older clients. Hex color is re-validated; other
    fields are passed through. `name` itself is NOT updatable in v1 — see
    Phase 2 design doc for the rename-deferral rationale.
    """
    set_clauses: list[str] = []
    params: list = []
    if "description" in fields:
        desc = fields["description"]
        set_clauses.append("description = %s")
        params.append((desc or "").strip() or None)
    if "color" in fields:
        color = fields["color"]
        if not _HEX_COLOR_RE.match(color or ""):
            raise ValueError("Color must be a six-digit hex string like '#6366f1'")
        set_clauses.append("color = %s")
        params.append(color)
    if "is_active" in fields:
        set_clauses.append("is_active = %s")
        params.append(bool(fields["is_active"]))
    if not set_clauses:
        # No-op update — return the current row so the endpoint can always
        # respond with a row-shaped body.
        rows = load_strategies(active_only=False)
        return next((r for r in rows if r["name"] == name), None)
    params.append(name)
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"UPDATE strategies SET {', '.join(set_clauses)} "
                f"WHERE name = %s "
                f"RETURNING name, description, color, is_active, created_at",
                tuple(params),
            )
            row = cur.fetchone()
            conn.commit()
            return dict(row) if row else None


def list_portfolios():
    """Return portfolios owned by the current authenticated user plus a
    live cash_balance derived from the cash_transactions ledger.

    RLS filters by app.user_id; caller must have populated current_user_id
    before this runs. Ordered by creation time so the first-created portfolio
    comes first — matches the onboarding UX where a user creates one and
    expects it to be the default.

    cash_balance is a LEFT JOIN subquery, returning 0 for portfolios with no
    activity yet.
    """
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT {", ".join("p." + c for c in _PORTFOLIO_COLS.split(", "))},
                       COALESCE(c.cash_balance, 0) AS cash_balance
                FROM portfolios p
                LEFT JOIN (
                    SELECT portfolio_id, SUM(amount) AS cash_balance
                    FROM cash_transactions
                    GROUP BY portfolio_id
                ) c ON c.portfolio_id = p.id
                ORDER BY p.created_at ASC
                """
            )
            return [dict(r) for r in cur.fetchall()]


def create_portfolio(name, starting_capital=None, reset_date=None):
    """Create a portfolio for the current user and return the new row.

    user_id is populated by the column DEFAULT added in migration 003 (reads
    from app.user_id). The per-user UNIQUE (user_id, name) index from
    migration 007 prevents one user from creating two portfolios with the
    same name. Raises ValueError on empty/oversized name or collision.
    starting_capital and reset_date are optional; both can be set later via
    update_portfolio.
    """
    name = validate_portfolio_name(name)
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            try:
                cur.execute(
                    f"INSERT INTO portfolios (name, starting_capital, reset_date) "
                    f"VALUES (%s, %s, %s) RETURNING {_PORTFOLIO_COLS}",
                    (name, starting_capital, reset_date),
                )
                row = cur.fetchone()
                # Seed the initial-capital deposit row so cash_balance is right
                # from minute one. Same connection keeps both writes atomic.
                _sync_initial_deposit(
                    cur, row["id"], starting_capital,
                    effective_date=reset_date or row["created_at"],
                )
                conn.commit()
                return dict(row)
            except psycopg2.errors.UniqueViolation:
                conn.rollback()
                raise ValueError(f"Portfolio '{name}' already exists")


def update_portfolio(portfolio_id, *, name=None, starting_capital=None, reset_date=None):
    """Update a portfolio the current user owns. Only passed fields change;
    pass `None` (or omit) to leave a column untouched.

    Returns the updated row or None if no portfolio matched (RLS would hide
    rows belonging to other users — callers should treat None as 404).
    """
    updates = []
    params: list = []
    if name is not None:
        updates.append("name = %s")
        params.append(validate_portfolio_name(name))
    if starting_capital is not None:
        updates.append("starting_capital = %s")
        params.append(starting_capital)
    if reset_date is not None:
        updates.append("reset_date = %s")
        params.append(reset_date)

    if not updates:
        # Nothing to do — return current row unchanged so callers get consistent shape.
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"SELECT {_PORTFOLIO_COLS} FROM portfolios WHERE id = %s",
                            (portfolio_id,))
                row = cur.fetchone()
                return dict(row) if row else None

    params.append(portfolio_id)
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            try:
                cur.execute(
                    f"UPDATE portfolios SET {', '.join(updates)} "
                    f"WHERE id = %s RETURNING {_PORTFOLIO_COLS}",
                    params,
                )
                row = cur.fetchone()
                if row is not None and (starting_capital is not None or reset_date is not None):
                    # Keep the initial-deposit cash_tx row in sync whenever the
                    # user edits either setting (both affect the deposit row).
                    _sync_initial_deposit(
                        cur, row["id"], row["starting_capital"],
                        effective_date=row["reset_date"] or row["created_at"],
                    )
                conn.commit()
                return dict(row) if row else None
            except psycopg2.errors.UniqueViolation:
                conn.rollback()
                raise ValueError(f"Portfolio '{name}' already exists")


def delete_portfolio(portfolio_id):
    """Delete a portfolio the current user owns.

    Cascades via trades_summary/trades_details FKs (ON DELETE CASCADE) — all
    trades, journal entries, snapshots, etc. under this portfolio go with it.
    RLS ensures a user can only delete their own portfolios.

    Returns True if a row was deleted, False if not found.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM portfolios WHERE id = %s", (portfolio_id,))
            deleted = cur.rowcount > 0
            conn.commit()
            return deleted


# ============================================
# CASH TRANSACTIONS (ledger for derived NLV)
# ============================================
# A cash_transactions row is appended every time money enters or leaves a
# portfolio's cash balance:
#   - deposit / withdraw — user-initiated transfers (via Settings)
#   - buy / sell — emitted by log_buy / log_sell in api/main.py
#   - reconcile — manual drift correction ("my broker says X, system says Y")
#
# amount is signed: + for money in, - for money out. Cash balance is simply
# SUM(amount) grouped by portfolio_id.
#
# NLV = cash_balance + Σ(open_position.shares × live_price). That derivation
# lives in a separate module (nlv_service.py) once Phase 2 lands.
# ============================================

_INITIAL_DEPOSIT_NOTE = "Initial capital"


def insert_cash_transaction(portfolio_id, amount, source, *, date=None,
                            trade_detail_id=None, note=None):
    """Append a signed cash_transactions row for the current user's portfolio.

    - amount: signed NUMERIC — positive for money in, negative for money out
    - source: one of 'deposit', 'withdraw', 'buy', 'sell', 'reconcile'
    - date: defaults to now() if omitted
    - trade_detail_id: link back to the originating trade (nullable)

    RLS + the column DEFAULT tag the row with the session's user_id, so the
    caller doesn't thread user_id through.
    """
    if date is None:
        date = datetime.now()
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "INSERT INTO cash_transactions "
                "(portfolio_id, date, amount, source, trade_detail_id, note) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "RETURNING id, portfolio_id, date, amount, source, trade_detail_id, note",
                (portfolio_id, date, amount, source, trade_detail_id, note),
            )
            row = cur.fetchone()
            conn.commit()
            return dict(row)


def get_cash_balance(portfolio_id):
    """Current cash balance for a portfolio: signed sum of its cash_tx rows.
    Returns 0.0 when no rows exist (new portfolio, no activity yet)."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(SUM(amount), 0) FROM cash_transactions "
                "WHERE portfolio_id = %s",
                (portfolio_id,),
            )
            row = cur.fetchone()
            return float(row[0])


def get_net_contributions(portfolio_id):
    """Net external contributions for a portfolio — sum of all deposit,
    withdraw, and reconcile rows. BUY/SELL rows are internal cash flows and
    excluded. This is the denominator for LTD return calcs:
        LTD return% = (NLV − net_contributions) / net_contributions × 100
    Starting_capital already lives here as the initial 'Initial capital'
    deposit row, so callers don't have to add it separately.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(SUM(amount), 0) FROM cash_transactions "
                "WHERE portfolio_id = %s "
                "AND source IN ('deposit', 'withdraw', 'reconcile')",
                (portfolio_id,),
            )
            row = cur.fetchone()
            return float(row[0])


def list_cash_transactions(portfolio_id, limit=200, exclude_trade_rows=False):
    """Return the most recent cash_tx rows for a portfolio, newest first.
    Used by an Activity/Cash ledger view in the UI.

    When exclude_trade_rows is True, buy/sell rows (which are auto-emitted by
    save_detail_row and dwarf user-managed cash flows) are filtered at the SQL
    layer so the LIMIT applies to deposit/withdraw/reconcile rows only — fixes
    the 'I added a backdated deposit but I can't see it' bug where a couple
    hundred recent trades pushed the deposit past the cutoff."""
    where = ["portfolio_id = %s"]
    params: list = [portfolio_id]
    if exclude_trade_rows:
        where.append("source NOT IN ('buy', 'sell')")
    params.append(limit)
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, portfolio_id, date, amount, source, trade_detail_id, note, created_at "
                f"FROM cash_transactions WHERE {' AND '.join(where)} "
                "ORDER BY date DESC, id DESC LIMIT %s",
                params,
            )
            return [dict(r) for r in cur.fetchall()]


def get_cash_transaction(tx_id):
    """Fetch a single cash_tx row by id (RLS-scoped to current user).
    Returns None if not found or not owned by the caller."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, portfolio_id, date, amount, source, trade_detail_id, note, created_at "
                "FROM cash_transactions WHERE id = %s",
                (tx_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def update_cash_transaction(tx_id, *, amount=None, date=None, note=None):
    """Patch a cash_tx row in place. Only the fields that aren't None are
    written. Source and signed-amount semantics are preserved by the caller —
    the API layer flips signs for withdraw and reuses the existing source.
    Returns the updated row, or None if the row doesn't exist (or is hidden by
    RLS)."""
    sets, params = [], []
    if amount is not None:
        sets.append("amount = %s")
        params.append(amount)
    if date is not None:
        sets.append("date = %s")
        params.append(date)
    if note is not None:
        sets.append("note = %s")
        params.append(note)
    if not sets:
        return get_cash_transaction(tx_id)
    params.append(tx_id)
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"UPDATE cash_transactions SET {', '.join(sets)} "
                f"WHERE id = %s "
                f"RETURNING id, portfolio_id, date, amount, source, trade_detail_id, note, created_at",
                params,
            )
            row = cur.fetchone()
            conn.commit()
            return dict(row) if row else None


def delete_cash_transaction(tx_id):
    """Delete a cash_tx row by id (RLS-scoped to current user). Returns True
    when a row was deleted, False otherwise."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM cash_transactions WHERE id = %s", (tx_id,))
            deleted = cur.rowcount
            conn.commit()
            return deleted > 0


def _sync_initial_deposit(cur, portfolio_id, starting_capital, effective_date):
    """Keep exactly one 'Initial capital' deposit row in sync with a
    portfolio's starting_capital setting.

    - starting_capital None/<=0 → remove any existing initial deposit
    - starting_capital > 0 and no existing → insert new deposit
    - starting_capital > 0 and existing → update amount + date

    This is the one spot where we mutate an existing cash_tx row (normally
    append-only). The rationale: starting_capital is a user-editable setting,
    not a true historical cash event. If a user typo-ed $20K instead of $200K,
    they should be able to correct the setting without polluting the ledger
    with a reconciliation row. Real adjustments after the fact should use
    deposit/withdraw/reconcile rows.
    """
    cur.execute(
        "SELECT id FROM cash_transactions "
        "WHERE portfolio_id = %s AND source = 'deposit' AND note = %s "
        "ORDER BY id ASC LIMIT 1",
        (portfolio_id, _INITIAL_DEPOSIT_NOTE),
    )
    existing = cur.fetchone()

    if starting_capital is None or float(starting_capital) <= 0:
        if existing:
            cur.execute("DELETE FROM cash_transactions WHERE id = %s", (existing[0],))
        return

    if existing:
        cur.execute(
            "UPDATE cash_transactions SET amount = %s, date = %s "
            "WHERE id = %s",
            (starting_capital, effective_date, existing[0]),
        )
    else:
        cur.execute(
            "INSERT INTO cash_transactions (portfolio_id, date, amount, source, note) "
            "VALUES (%s, %s, %s, 'deposit', %s)",
            (portfolio_id, effective_date, starting_capital, _INITIAL_DEPOSIT_NOTE),
        )


# ============================================================================
# Phase 6 — pinned_entities (Migration 029) and Weekly Retro list helpers
# ============================================================================
# Polymorphic pin store + supporting aggregations for the NotesRail. Pin
# semantics mirror tag_assignments (Phase 1): soft-delete on unpin, idempotent
# revival on re-pin via the partial-unique index. The list_weekly_retros_rail
# helper produces the wrapped {weeks, ytd_stats} envelope the rail consumes,
# including synthetic empty-week rows for Mondays the user hasn't graded yet.

_PIN_ENTITY_TYPES = ("weekly_retro", "daily_journal")


def toggle_pin(entity_type: str, entity_id: int) -> bool:
    """Idempotent pin toggle for the polymorphic pinned_entities table
    (Migration 029).

    Contract:
      - If a LIVE pin exists for (current user, entity_type, entity_id)
        → soft-delete it. Returns False (now unpinned).
      - If a SOFT-DELETED pin exists for the same triple → REVIVE it
        (UPDATE deleted_at = NULL). Returns True (now pinned). Reuses the
        same row id so pinned_at sort stability is preserved across
        unpin/re-pin cycles.
      - Otherwise INSERT a new row. Returns True.

    Tenant isolation is enforced by RLS — the SELECT below only sees the
    caller's rows. The user_id column has a DEFAULT pulling from
    current_setting('app.user_id', true), so the INSERT path doesn't have
    to set it explicitly. Same pattern as create_tag_assignment.

    Raises ValueError on unknown entity_type (defense in depth against the
    CHECK constraint, which would otherwise produce an IntegrityError that
    callers find harder to interpret).
    """
    if entity_type not in _PIN_ENTITY_TYPES:
        raise ValueError(f"Unknown entity_type: {entity_type}")
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, deleted_at FROM pinned_entities "
                "WHERE entity_type = %s AND entity_id = %s "
                "ORDER BY id DESC LIMIT 1",
                (entity_type, entity_id),
            )
            existing = cur.fetchone()

            if existing and existing["deleted_at"] is None:
                cur.execute(
                    "UPDATE pinned_entities SET deleted_at = NOW() WHERE id = %s",
                    (existing["id"],),
                )
                conn.commit()
                return False
            if existing and existing["deleted_at"] is not None:
                cur.execute(
                    "UPDATE pinned_entities SET deleted_at = NULL WHERE id = %s",
                    (existing["id"],),
                )
                conn.commit()
                return True
            cur.execute(
                "INSERT INTO pinned_entities (entity_type, entity_id) VALUES (%s, %s)",
                (entity_type, entity_id),
            )
            conn.commit()
            return True


def list_pinned_entity_ids(entity_type: str) -> set[int]:
    """Returns the set of currently-pinned (live) entity_ids for the given
    type, scoped to the caller via RLS. Used by list_weekly_retros_rail to
    decorate each row with `pinned: bool`."""
    if entity_type not in _PIN_ENTITY_TYPES:
        raise ValueError(f"Unknown entity_type: {entity_type}")
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT entity_id FROM pinned_entities "
                "WHERE entity_type = %s AND deleted_at IS NULL",
                (entity_type,),
            )
            return {row[0] for row in cur.fetchall()}


# ============================================================================
# pinned_routes (Migration 042) — sidebar "Pinned" section
# ============================================================================
# Separate from pinned_entities because route paths are strings, not integer
# FK refs. Same persistence idioms (soft-delete + revive, RLS) without
# polymorphic gymnastics. See migration 042 preamble for the rationale.

# Mirrors the DB CHECK constraint on route_path. Defense-in-depth so callers
# get a ValueError before the DB raises IntegrityError on malformed input —
# matches the _PIN_ENTITY_TYPES whitelist pattern in toggle_pin.
_ROUTE_PATH_RE = re.compile(r'^/[a-z0-9-]+(/[a-z0-9-]+)*$')


def toggle_pin_route(route_path: str) -> bool:
    """Idempotent pin toggle for the pinned_routes table (Migration 042).

    Contract:
      - If a LIVE pin exists for (current user, route_path)
        → soft-delete it. Returns False (now unpinned).
      - If a SOFT-DELETED pin exists for the same pair → REVIVE it
        (UPDATE deleted_at = NULL). Returns True (now pinned). Reuses the
        same row id so pinned_at sort stability is preserved across
        unpin/re-pin cycles. Same idiom as toggle_pin.
      - Otherwise INSERT a new row. Returns True.

    Tenant isolation is enforced by RLS — the SELECT below only sees the
    caller's rows. The user_id column has a DEFAULT pulling from
    current_setting('app.user_id', true), so the INSERT path doesn't have
    to set it explicitly.

    Raises ValueError on a route_path that doesn't match the conservative
    pattern (^/[a-z0-9-]+(/[a-z0-9-]+)*$) — defense in depth vs. the CHECK
    constraint, which would otherwise produce an IntegrityError that
    callers find harder to interpret.
    """
    if not isinstance(route_path, str) or not _ROUTE_PATH_RE.match(route_path):
        raise ValueError(f"Invalid route_path: {route_path!r}")
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, deleted_at FROM pinned_routes "
                "WHERE route_path = %s "
                "ORDER BY id DESC LIMIT 1",
                (route_path,),
            )
            existing = cur.fetchone()

            if existing and existing["deleted_at"] is None:
                cur.execute(
                    "UPDATE pinned_routes SET deleted_at = NOW() WHERE id = %s",
                    (existing["id"],),
                )
                conn.commit()
                return False
            if existing and existing["deleted_at"] is not None:
                cur.execute(
                    "UPDATE pinned_routes SET deleted_at = NULL WHERE id = %s",
                    (existing["id"],),
                )
                conn.commit()
                return True
            cur.execute(
                "INSERT INTO pinned_routes (route_path) VALUES (%s)",
                (route_path,),
            )
            conn.commit()
            return True


def list_pinned_routes() -> list[dict]:
    """Return the caller's live pins ordered by pinned_at ASC (FIFO,
    oldest first), scoped via RLS.

    Each dict: {route_path: str, pinned_at: datetime}. The pinned_at field
    survives unpin/re-pin cycles because toggle_pin_route revives rows
    instead of inserting fresh ones — first-pinned-first ordering is
    stable across user toggle history.
    """
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT route_path, pinned_at FROM pinned_routes "
                "WHERE deleted_at IS NULL "
                "ORDER BY pinned_at ASC, id ASC",
            )
            return [dict(row) for row in cur.fetchall()]


# ----------------------------------------------------------------------------
# Avg grade — 4.3 GPA scale, bucketed back to letter
# ----------------------------------------------------------------------------
# Locked Phase 6 mapping. Frontend + backend SHOULD agree; keeping the
# canonical implementation here. The bucket thresholds use midpoints between
# adjacent letters on the numeric scale so a mean exactly at the midpoint
# rounds to the higher letter.

_GRADE_TO_NUMERIC: dict[str, float] = {
    "A+": 4.3, "A": 4.0, "A-": 3.7,
    "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7,
    "D+": 1.3, "D": 1.0, "D-": 0.7,
    "F":  0.0,
}

# Buckets: (lower_bound_inclusive, letter). Sorted descending so the first
# match on a >= comparison is the right letter. Midpoint between A+ (4.3)
# and A (4.0) is 4.15, etc.
_NUMERIC_BUCKETS: list[tuple[float, str]] = [
    (4.15, "A+"),
    (3.85, "A"),
    (3.50, "A-"),
    (3.15, "B+"),
    (2.85, "B"),
    (2.50, "B-"),
    (2.15, "C+"),
    (1.85, "C"),
    (1.50, "C-"),
    (1.15, "D+"),
    (0.85, "D"),
    (0.35, "D-"),
    (0.00, "F"),
]


def avg_grade_from_letters(grades) -> str | None:
    """Compute the average letter grade from a list of letter strings.

    Locked Phase 6 formula: map each letter to the 4.3 GPA scale, take the
    arithmetic mean, then bucket the mean back to the nearest letter for
    display. Unknown letters and falsy values are ignored. Returns None
    when no valid letters are present.
    """
    if not grades:
        return None
    numeric_values: list[float] = []
    for g in grades:
        if not g:
            continue
        key = str(g).strip().upper()
        if key in _GRADE_TO_NUMERIC:
            numeric_values.append(_GRADE_TO_NUMERIC[key])
    if not numeric_values:
        return None
    mean = sum(numeric_values) / len(numeric_values)
    for lower, letter in _NUMERIC_BUCKETS:
        if mean >= lower:
            return letter
    return "F"


# ----------------------------------------------------------------------------
# Weekly-return series — per-Monday TWR returns over the full journal
# ----------------------------------------------------------------------------
# Single-pass batch helper. Loads the journal once, buckets by ISO-Monday,
# computes chained daily-Dietz returns per bucket (same formula as Phase 5's
# weekly_metrics / _compute_twr_from_journal_df). The list endpoint embeds
# the per-week values into each row's `sparkline_value`, so a 52-week rail
# doesn't have to make 52 separate /api/analytics/weekly-metrics calls.

def weekly_return_series_for_portfolio(portfolio_name: str) -> dict[str, dict]:
    """Returns {week_start_iso: {week_start, week_end, weekly_return_pct,
    weekly_pnl}} for every Monday with at least one journal row in the
    portfolio.

    Reuses Phase 5's `_prepare_journal_for_returns` to produce the per-day
    return series, then groups by the ISO Monday of each row. Each bucket's
    chained product (1+r) gives the week TWR. The weekly_pnl is the same
    NLV-delta formula as Phase 5's weekly_metrics tile —
        end_nlv − (beg_nlv + Σ cash_change)
    over the in-week rows. Computing both in the same batch pass guarantees
    rail values match tile values (cross-page consistency contract is
    enforced by the test_rail_matches_tile_sources test).

    Returned dict keys are ISO date strings (YYYY-MM-DD) so they're directly
    matchable against the synthetic-Monday grid the API list endpoint builds.
    """
    # Local imports — nlv_service imports db_layer at module top, so importing
    # nlv_service here would cycle. Lazy imports break the cycle.
    from nlv_service import _prepare_journal_for_returns, _compute_weekly_nlv_delta
    from datetime import date as _date, timedelta as _td

    journal = load_journal(portfolio_name)
    if journal is None or journal.empty:
        return {}

    # _prepare_journal_for_returns now normalizes Title-Case columns itself
    # (the Phase 6 "0 weeks" regression originated from a missed normalize
    # call here — pushing normalize inside the helper removed the trap).
    work = _prepare_journal_for_returns(journal)
    if work.empty:
        return {}

    # Bucket per-day rows by Monday-of-week. Preserves day-sorted order
    # inside each bucket (work is sorted ascending by _prepare_journal).
    # _compute_weekly_nlv_delta needs a DataFrame, so collect indices per
    # bucket and slice rather than building lists of Series.
    bucket_indices: dict[str, list[int]] = {}
    for i, row in work.iterrows():
        day = row["day"].date() if hasattr(row["day"], "date") else row["day"]
        monday = day - _td(days=day.weekday())
        key = monday.isoformat()
        bucket_indices.setdefault(key, []).append(i)

    out: dict[str, dict] = {}
    for key, idxs in bucket_indices.items():
        monday = _date.fromisoformat(key)
        friday = monday + _td(days=4)
        bucket = work.loc[idxs]
        # Chained TWR for the week.
        product = float((1.0 + bucket["daily_return"]).prod())
        weekly_return_pct = round((product - 1.0) * 100.0, 4)
        # NLV-delta from the canonical helper — guarantees rail and tile
        # agree by construction. The test_rail_matches_tile_sources cross-
        # validation test continues to pin this contract for any future
        # code path that bypasses the helper.
        weekly_pnl = _compute_weekly_nlv_delta(bucket)
        out[key] = {
            "week_start": key,
            "week_end": friday.isoformat(),
            "weekly_return_pct": weekly_return_pct,
            "weekly_pnl": weekly_pnl,
        }

    return out


# ----------------------------------------------------------------------------
# Wrapped list endpoint payload — replaces the bare-array list_weekly_retros
# for the NotesRail consumer. Builds the (id|null) × week grid from the
# earliest-journal Monday to the current week, joins in retros + pins +
# sparkline values, and produces the YTD aggregate.
# ----------------------------------------------------------------------------

def _weekly_transaction_counts_for_portfolio(portfolio_name: str) -> dict[str, int]:
    """Per-week count of trade_details transactions (buys + sells), keyed
    by ISO Monday of the transaction date's week.

    Source matches the Flight Deck "Total Tickets" tile exactly: both
    count rows from `trade_details` (one row per BUY/SELL transaction)
    filtered to the week's Mon-Sun range. A campaign with 1 BUY + 2 SELLs
    contributes 3 to the count, not 1.

    This deliberately differs from "closed campaigns count" (which the
    pre-fix helper returned). Cross-page consistency with the Flight Deck
    tile is enforced by test_rail_trades_count_matches_flight_deck.
    """
    from datetime import timedelta as _td
    import pandas as _pd

    details = load_details(portfolio_name)
    if details is None or details.empty:
        return {}

    # load_details returns Title-Case columns ("Date", "Action", ...).
    # Action filter is intentionally lenient — anything in the table
    # counts as a "transaction" for the rail. trades_details is already
    # the buy/sell ledger; there's no other kind of row.
    date_col = "Date" if "Date" in details.columns else "date"
    if date_col not in details.columns:
        return {}

    work = details.copy()
    work[date_col] = _pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col])
    if work.empty:
        return {}

    out: dict[str, int] = {}
    for _, row in work.iterrows():
        d = row[date_col].date() if hasattr(row[date_col], "date") else row[date_col]
        monday = d - _td(days=d.weekday())
        out[monday.isoformat()] = out.get(monday.isoformat(), 0) + 1
    return out


def _weekly_retro_tags_batch(retro_ids: list[int]) -> dict[int, list[dict]]:
    """Batch-fetch tag_assignments for many weekly_retro entities in one
    query. Returns {entity_id: [{name, color}, ...]}. Soft-deleted tags are
    filtered via the JOIN predicate (same pattern as load_tag_assignments).
    """
    out: dict[int, list[dict]] = {rid: [] for rid in retro_ids}
    if not retro_ids:
        return out
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT a.entity_id, t.name, t.color "
                "FROM tag_assignments a "
                "JOIN tags t ON t.id = a.tag_id AND t.deleted_at IS NULL "
                "WHERE a.entity_type = 'weekly_retro' "
                "  AND a.entity_id = ANY(%s) "
                "  AND a.deleted_at IS NULL "
                "ORDER BY a.created_at ASC",
                (retro_ids,),
            )
            for row in cur.fetchall():
                out.setdefault(row["entity_id"], []).append({
                    "name": row["name"], "color": row["color"],
                })
    return out


def list_weekly_retros_rail(portfolio_name: str) -> dict:
    """Wrapped envelope for the NotesRail. Returns:

        {
          "weeks": [
            { id, key, week_start, week_end, year, month, title,
              has_content, pinned, sparkline_value, week_grade },
            ...
          ],
          "ytd_stats": {
            total_weeks, weeks_graded, avg_grade, weeks_pinned,
          }
        }

    Mondays without a saved retro emit a synthetic row (id: null,
    has_content: False, pinned: False, week_grade: None). Their
    sparkline_value still comes from the journal-derived series, so the
    month sparkline has continuous data points even on ungraded weeks.

    Date range: earliest Monday with a journal row → most recent past
    Monday (or current Monday if today is Mon-Sun). No pagination — at
    ~52 weeks per year, even a 5-year account fits comfortably in one
    payload.

    ytd_stats:
      total_weeks  — count of weeks in current year up to the last
                     returned row's week_start
      weeks_graded — count of rows in current year with a non-null grade
      avg_grade    — letter (via _NUMERIC_BUCKETS) over current-year
                     grades; None when weeks_graded == 0
      weeks_pinned — count of pinned (live) weekly_retro pins, scoped
                     to this portfolio's retros
    """
    from datetime import date as _date, timedelta as _td

    # 1. Saved retros, keyed by week_start ISO. Reuse the existing helper
    # to keep the parent + ticker_grades hydration consistent.
    saved = list_weekly_retros(portfolio_name, limit=1000)
    saved_by_week: dict[str, dict] = {r["week_start"]: r for r in saved}

    # 2. Sparkline values, keyed by week_start ISO.
    series = weekly_return_series_for_portfolio(portfolio_name)

    # 3. Pinned ids for this entity type (RLS scopes to the user).
    pinned_ids = list_pinned_entity_ids("weekly_retro")

    # 4. Per-week transaction counts from trade_details — matches the
    #    Flight Deck "Total Tickets" tile source exactly. Replaces the
    #    pre-fix campaign-count-from-trades_summary path that produced
    #    different numbers from the tile for the same week.
    trade_counts = _weekly_transaction_counts_for_portfolio(portfolio_name)

    # 5. Tags per retro id, batch-fetched in a single SELECT.
    tags_by_retro = _weekly_retro_tags_batch([r["id"] for r in saved])

    # 4. Build the week grid from inception to the current Monday.
    #    Inception = earliest Monday with either a journal row or a saved
    #    retro. If neither exists, the grid is empty and we still return
    #    the envelope with zeros.
    earliest_keys: list[str] = []
    if series:
        earliest_keys.append(min(series.keys()))
    if saved_by_week:
        earliest_keys.append(min(saved_by_week.keys()))
    if not earliest_keys:
        return {
            "weeks": [],
            "ytd_stats": {
                "total_weeks": 0,
                "weeks_graded": 0,
                "avg_grade": None,
                "weeks_pinned": 0,
            },
        }

    start_iso = min(earliest_keys)
    start_monday = _date.fromisoformat(start_iso)
    # Defensive: snap to Monday in case the earliest_key didn't already.
    start_monday = start_monday - _td(days=start_monday.weekday())

    today = _date.today()
    current_monday = today - _td(days=today.weekday())

    weeks: list[dict] = []
    cursor = start_monday
    while cursor <= current_monday:
        friday = cursor + _td(days=4)
        key = cursor.isoformat()
        retro = saved_by_week.get(key)
        series_row = series.get(key, {})
        sparkline = series_row.get("weekly_return_pct")
        # NLV-delta P&L for the week. Same source as the Weekly P&L tile
        # in weekly_metrics → consistent values across rail + tiles.
        weekly_pnl = series_row.get("weekly_pnl")
        trades_count = trade_counts.get(key, 0)
        retro_tags = tags_by_retro.get(retro["id"], []) if retro else []
        # has_content: row exists AND user filled in at least one field.
        # Pure "saved by the auto-save but every field blank" stays
        # has_content=False so the rail draft-dot styling is honest.
        has_content = bool(retro and _retro_has_content(retro))
        weeks.append({
            "id": retro["id"] if retro else None,
            "key": key,
            "week_start": key,
            "week_end": friday.isoformat(),
            "year": cursor.year,
            "month": cursor.month,
            "title": _format_week_title(cursor, friday),
            "has_content": has_content,
            "pinned": bool(retro and retro["id"] in pinned_ids),
            "sparkline_value": sparkline,
            "week_grade": (retro or {}).get("week_grade"),
            # Phase 6 design-fidelity fields. weekly_pnl matches the
            # Weekly P&L tile (NLV-delta). trades_count matches the
            # Flight Deck Total Tickets tile (trade_details rows).
            # win_rate was dropped in Phase 6 stats-format consolidation —
            # the rail's per-row line no longer renders it.
            "weekly_pnl": weekly_pnl,
            "trades_count": trades_count,
            "tags": retro_tags,
            # Phase 4.6 — tri-state dot needs reviewed_at to distinguish
            # "drafted but not closed" (amber) from "reviewed" (green).
            # _serialize_weekly_retro returns an ISO string or None.
            "reviewed_at": (retro or {}).get("reviewed_at"),
        })
        cursor = cursor + _td(days=7)

    # Sort newest first for the rail's primary read order.
    weeks.sort(key=lambda w: w["week_start"], reverse=True)

    # YTD aggregate (current calendar year only).
    current_year = today.year
    ytd_rows = [w for w in weeks if w["year"] == current_year]
    graded_letters = [w["week_grade"] for w in ytd_rows if w["week_grade"]]
    ytd_pinned_count = sum(1 for w in ytd_rows if w["pinned"])

    return {
        "weeks": weeks,
        "ytd_stats": {
            "total_weeks": len(ytd_rows),
            "weeks_graded": len(graded_letters),
            "avg_grade": avg_grade_from_letters(graded_letters),
            "weeks_pinned": ytd_pinned_count,
        },
    }


def _retro_has_content(retro: dict) -> bool:
    """True if the retro has any user-supplied content beyond defaults.
    Drives the rail's draft vs. graded styling. Auto-save can persist a
    row with all-blank fields when the user simply navigates to a week
    — those should still surface as draft (id present so it can be
    pinned, but has_content=False so the dot stays hollow)."""
    return bool(
        retro.get("week_grade")
        or (retro.get("best_decision") or "").strip()
        or (retro.get("worst_decision") or "").strip()
        or (retro.get("rule_change_text") or "").strip()
        or (retro.get("weekly_thoughts") or "").strip()
        or retro.get("rule_change")
        or retro.get("ticker_grades")
    )


_MONTH_NAMES = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)


def _format_week_title(monday, friday) -> str:
    """Human label for a week row. Examples: "May 11 – May 15",
    "Apr 27 – May 1". Server-formats so timezone bugs don't shift the
    visible date in the rail."""
    same_month = monday.month == friday.month
    if same_month:
        return f"{_MONTH_NAMES[monday.month - 1]} {monday.day} – {friday.day}"
    return (
        f"{_MONTH_NAMES[monday.month - 1]} {monday.day} – "
        f"{_MONTH_NAMES[friday.month - 1]} {friday.day}"
    )


# ============================================
# DAILY JOURNAL CAPTURES (Migration 031, Phase 7)
# ============================================
# Image attachments on daily journal entries. Metadata lives in
# daily_journal_captures; bytes live in Cloudflare R2 via the upload_blob
# helper in r2_storage.py. Mirrors the weekly_retro_snapshots helpers
# byte-for-byte except for table + FK names. Soft-delete via deleted_at —
# the bytes intentionally remain in R2 for v1 (storage is cheap; a future
# sweep job can reclaim).


def _serialize_daily_capture(row: dict, r2_public_url: str) -> dict:
    """Serialize a daily_journal_captures row for the wire. Composes
    view_url from R2_PUBLIC_URL + storage_ref. Mirrors _serialize_snapshot
    above; kept as a parallel function rather than a shared helper so a
    future field-set divergence between the two surfaces doesn't require
    a refactor."""
    storage_ref = row.get("storage_ref") or ""
    if r2_public_url and storage_ref:
        view_url = f"{r2_public_url.rstrip('/')}/{storage_ref}"
    else:
        view_url = storage_ref
    return {
        "id": row["id"],
        "daily_journal_id": row["daily_journal_id"],
        "storage_ref": storage_ref,
        "view_url": view_url,
        "file_name": row.get("file_name"),
        "mime_type": row.get("mime_type"),
        "file_size_bytes": row.get("file_size_bytes"),
        "width": row.get("width"),
        "height": row.get("height"),
        "sort_order": row.get("sort_order") or 0,
        "caption": row.get("caption") or "",
        "created_at": (
            row["created_at"].isoformat()
            if row.get("created_at") and hasattr(row["created_at"], "isoformat")
            else row.get("created_at")
        ),
    }


def _resolve_journal_owned_by_portfolio(cur, journal_id: int, portfolio_name: str) -> bool:
    """Verify that the daily journal row exists, belongs to the given
    portfolio, is not soft-deleted, AND is visible to the current
    app.user_id via RLS. Returns True on hit, False on miss (caller maps
    to 404 — never 403, to avoid leaking existence).

    Phase 7 defensive: trading_journal's unique constraint on (portfolio,
    day) is FULL — soft-deleted rows still occupy the slot. The
    `r.deleted_at IS NULL` predicate here makes sure a tombstoned row
    can't accept new captures."""
    cur.execute(
        "SELECT 1 FROM trading_journal j JOIN portfolios p ON p.id = j.portfolio_id "
        "WHERE j.id = %s AND p.name = %s AND j.deleted_at IS NULL",
        (journal_id, portfolio_name),
    )
    return cur.fetchone() is not None


def save_daily_journal_capture(
    portfolio_name: str,
    journal_id: int,
    storage_ref: str,
    *,
    file_name: str | None = None,
    mime_type: str | None = None,
    file_size_bytes: int | None = None,
    width: int | None = None,
    height: int | None = None,
) -> dict | None:
    """INSERT a capture row attached to the given daily journal entry.
    Returns the serialized row (including view_url) on success, or None
    if the journal row doesn't exist / isn't visible to the current
    tenant (caller maps to 404)."""
    r2_public = (os.environ.get("R2_PUBLIC_URL") or "").rstrip("/")
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if not _resolve_journal_owned_by_portfolio(cur, journal_id, portfolio_name):
                return None
            cur.execute(
                "INSERT INTO daily_journal_captures "
                "  (daily_journal_id, storage_ref, file_name, mime_type, "
                "   file_size_bytes, width, height) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                "RETURNING id, daily_journal_id, storage_ref, file_name, "
                "          mime_type, file_size_bytes, width, height, "
                "          sort_order, caption, created_at",
                (journal_id, storage_ref, file_name, mime_type,
                 file_size_bytes, width, height),
            )
            row = dict(cur.fetchone())
            conn.commit()
            return _serialize_daily_capture(row, r2_public)


def list_daily_journal_captures(portfolio_name: str, journal_id: int) -> list[dict] | None:
    """Return all live captures for the journal entry, ordered by
    (sort_order, created_at). Returns None if the journal row is missing
    / not visible (caller maps to 404). Returns [] for a journal with
    no captures."""
    r2_public = (os.environ.get("R2_PUBLIC_URL") or "").rstrip("/")
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if not _resolve_journal_owned_by_portfolio(cur, journal_id, portfolio_name):
                return None
            cur.execute(
                "SELECT id, daily_journal_id, storage_ref, file_name, "
                "       mime_type, file_size_bytes, width, height, "
                "       sort_order, caption, created_at "
                "FROM daily_journal_captures "
                "WHERE daily_journal_id = %s AND deleted_at IS NULL "
                "ORDER BY sort_order, created_at",
                (journal_id,),
            )
            return [_serialize_daily_capture(dict(r), r2_public) for r in cur.fetchall()]


def verify_daily_journal_ownership(portfolio_name: str, journal_id: int) -> bool:
    """Public wrapper around _resolve_journal_owned_by_portfolio for
    endpoints that need an ownership check without an INSERT/SELECT.
    Mirrors verify_retro_ownership."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            return _resolve_journal_owned_by_portfolio(cur, journal_id, portfolio_name)


def soft_delete_daily_journal_capture(capture_id: int) -> bool:
    """Set deleted_at = NOW() on a capture row. RLS scopes the UPDATE
    to the current tenant — a cross-tenant capture_id misses the WHERE
    clause and returns False (caller maps to 404, not 403, to avoid
    leaking existence)."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE daily_journal_captures SET deleted_at = NOW() "
                "WHERE id = %s AND deleted_at IS NULL RETURNING id",
                (capture_id,),
            )
            hit = cur.fetchone()
            conn.commit()
            return hit is not None


# ----------------------------------------------------------------------------
# DAILY NotesRail envelope (Phase 7)
# ----------------------------------------------------------------------------
# Builds the rail item list + YTD aggregates for the Daily Report page's
# left rail. Mirrors list_weekly_retros_rail in shape but:
#   - One row per existing journal entry (NOT one per calendar day —
#     ~252×N is too many for a single payload; the rail is a sparse
#     index of journaled days, not a calendar grid).
#   - sparkline_value = daily_pct_change (recorded directly on the row).
#   - grade = letter mapped from `score` via _daily_score_to_letter
#     (1-5 score scale, same as the Daily Review chips on the page).
#   - reviewed_at = the day itself when score > 0, else None (mirrors
#     the Phase 4.6 tri-state dot: empty = no row; draft = row exists
#     but score == 0; reviewed = score > 0).


# Daily score → letter mapping. Matches daily-report-card.tsx:502 (the
# inline `gradeLabel` ternary chain in <DailyReview/>). Scores are 1-5
# integers; 0 means "no grade yet" (draft).
_DAILY_SCORE_TO_LETTER: dict[int, str] = {
    5: "A+",
    4: "A",
    3: "B",
    2: "C",
    1: "D",
}


def _daily_score_to_letter(score) -> str | None:
    """Returns the letter grade for a daily score (1-5), or None for
    falsy / out-of-range values. Matches the frontend gradeLabel ternary
    used on the daily-report-card so rail + page agree by construction."""
    try:
        n = int(score) if score is not None else 0
    except (TypeError, ValueError):
        return None
    return _DAILY_SCORE_TO_LETTER.get(n)


def _daily_journal_tags_batch(journal_ids: list[int]) -> dict[int, list[dict]]:
    """Batch-fetch tag_assignments for many daily_journal entities in one
    query. Mirrors _weekly_retro_tags_batch."""
    out: dict[int, list[dict]] = {jid: [] for jid in journal_ids}
    if not journal_ids:
        return out
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT a.entity_id, t.name, t.color "
                "FROM tag_assignments a "
                "JOIN tags t ON t.id = a.tag_id AND t.deleted_at IS NULL "
                "WHERE a.entity_type = 'daily_journal' "
                "  AND a.entity_id = ANY(%s) "
                "  AND a.deleted_at IS NULL "
                "ORDER BY a.created_at ASC",
                (journal_ids,),
            )
            for row in cur.fetchall():
                out.setdefault(row["entity_id"], []).append({
                    "name": row["name"], "color": row["color"],
                })
    return out


def _daily_journal_has_content(row: dict) -> bool:
    """True if the daily journal row has user-supplied content beyond
    defaults. Drives the rail's draft vs. graded styling. A row that
    exists because of an auto-save (NLV from IBKR sync) but has no user
    prose yet still surfaces as draft (id present so it can be pinned,
    but has_content=False so the dot stays hollow)."""
    if row.get("score") and int(row["score"]) > 0:
        return True
    for key in ("lowlights", "highlights", "mistakes", "top_lesson",
                "daily_thoughts", "market_notes", "market_action"):
        if (row.get(key) or "").strip():
            return True
    return False


def list_daily_journals_rail(portfolio_name: str) -> dict:
    """Wrapped envelope for the NotesRail on the Daily Report page.
    Returns:

        {
          "days": [
            { id, key (YYYY-MM-DD), date_label, year, month,
              has_content, pinned, sparkline_value (daily_pct_change),
              weekly_pnl (daily_dollar_change — name kept for wire
              compat with NotesRailItem; semantically "period P&L"),
              trades_count, tags, week_grade (letter from score),
              reviewed_at },
            ...
          ],
          "ytd_stats": { total_weeks, weeks_graded, avg_grade,
                         weeks_pinned }
        }

    Wire-shape notes:
      - Top-level key is "days" (parallel to weekly's "weeks"); the
        rail component branches on entityType for the row's date label
        copy.
      - "weekly_pnl" / "week_grade" / "weeks_*" field names are kept
        from the weekly envelope to preserve NotesRailItem compatibility.
        Semantically the daily envelope uses them for daily P&L and
        per-day grade — the rail's existing entityType-branched copy
        handles the rendering distinction.
      - Only days with an existing journal row are returned (skip empty
        days — at ~252 trading days/year a multi-year payload would be
        too large for a single rail load).
    """
    df = load_journal(portfolio_name)
    if df is None or df.empty:
        return {
            "days": [],
            "ytd_stats": {
                "total_weeks": 0,
                "weeks_graded": 0,
                "avg_grade": None,
                "weeks_pinned": 0,
            },
        }

    work = df.copy()
    # load_journal returns Title-Case; normalize so we can address by
    # snake_case from here on.
    from trade_calc import normalize_journal_columns as _norm
    work = _norm(work)
    if "day" not in work.columns:
        return {
            "days": [],
            "ytd_stats": {"total_weeks": 0, "weeks_graded": 0,
                          "avg_grade": None, "weeks_pinned": 0},
        }
    work["day"] = pd.to_datetime(work["day"], errors="coerce")
    work = work.dropna(subset=["day"]).sort_values("day", ascending=False)
    if work.empty:
        return {
            "days": [],
            "ytd_stats": {"total_weeks": 0, "weeks_graded": 0,
                          "avg_grade": None, "weeks_pinned": 0},
        }

    journal_ids = [int(r["id"]) for _, r in work.iterrows()
                   if r.get("id") is not None and not pd.isna(r["id"])]
    pinned_ids = list_pinned_entity_ids("daily_journal")
    tags_by_id = _daily_journal_tags_batch(journal_ids)

    # Per-day transaction counts from trade_details. Mirrors the
    # weekly helper but bucketed by ISO date instead of Monday-of-week.
    details = load_details(portfolio_name)
    trade_counts: dict[str, int] = {}
    if details is not None and not details.empty:
        date_col = "Date" if "Date" in details.columns else "date"
        if date_col in details.columns:
            tmp = details.copy()
            tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
            tmp = tmp.dropna(subset=[date_col])
            for _, row in tmp.iterrows():
                d = row[date_col].date() if hasattr(row[date_col], "date") else row[date_col]
                trade_counts[d.isoformat()] = trade_counts.get(d.isoformat(), 0) + 1

    days: list[dict] = []
    for _, row in work.iterrows():
        try:
            jid = int(row["id"]) if row.get("id") is not None else None
        except (TypeError, ValueError):
            jid = None
        day_dt = row["day"]
        day_iso = day_dt.date().isoformat() if hasattr(day_dt, "date") else str(day_dt)[:10]
        try:
            score = int(row.get("score") or 0)
        except (TypeError, ValueError):
            score = 0
        letter = _daily_score_to_letter(score)
        sparkline = row.get("daily_pct_change")
        if sparkline is not None and not pd.isna(sparkline):
            sparkline_value = float(sparkline)
        else:
            sparkline_value = None
        pnl = row.get("daily_dollar_change")
        weekly_pnl = float(pnl) if pnl is not None and not pd.isna(pnl) else None

        days.append({
            "id": jid,
            "key": day_iso,
            "week_start": day_iso,
            "week_end": day_iso,
            "year": (day_dt.year if hasattr(day_dt, "year") else int(day_iso[:4])),
            "month": (day_dt.month if hasattr(day_dt, "month") else int(day_iso[5:7])),
            "title": _format_daily_title(day_dt),
            "has_content": _daily_journal_has_content(row.to_dict()),
            "pinned": bool(jid is not None and jid in pinned_ids),
            "sparkline_value": sparkline_value,
            "week_grade": letter,
            "weekly_pnl": weekly_pnl,
            "trades_count": trade_counts.get(day_iso, 0),
            "tags": tags_by_id.get(jid, []) if jid is not None else [],
            # Phase 4.6 tri-state dot: reviewed = score > 0 (locked
            # Phase 7 decision — daily entries have no separate
            # `reviewed_at` column, score becomes the proxy).
            "reviewed_at": day_iso if score > 0 else None,
        })

    # YTD aggregate (current calendar year only).
    from datetime import date as _date
    current_year = _date.today().year
    ytd_rows = [d for d in days if d["year"] == current_year]
    graded_letters = [d["week_grade"] for d in ytd_rows if d["week_grade"]]
    ytd_pinned_count = sum(1 for d in ytd_rows if d["pinned"])

    return {
        "days": days,
        "ytd_stats": {
            "total_weeks": len(ytd_rows),
            "weeks_graded": len(graded_letters),
            "avg_grade": avg_grade_from_letters(graded_letters),
            "weeks_pinned": ytd_pinned_count,
        },
    }


def _format_daily_title(day) -> str:
    """Human label for a daily rail row. Example: "May 15".
    Server-formats so timezone bugs don't shift the visible date."""
    try:
        m = day.month
        d = day.day
        return f"{_MONTH_NAMES[m - 1]} {d}"
    except Exception:
        return str(day)[:10]
