# db_layer.py - PostgreSQL abstraction layer for trading journal

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
import pandas as pd
import os
import streamlit as st
from contextlib import contextmanager
from datetime import datetime
import time
from decimal import Decimal

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

    # Check Streamlit secrets (Streamlit Cloud deployment)
    try:
        if hasattr(st, 'secrets') and 'database' in st.secrets:
            return {'dsn': st.secrets['database']['url']}
    except Exception:
        pass

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
@st.cache_data(ttl=30, show_spinner=False)  # Cache for 30 seconds
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
            query = """
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
                    COALESCE(
                        (SELECT d.rule
                         FROM trades_details d
                         WHERE d.trade_id = s.trade_id
                           AND d.portfolio_id = s.portfolio_id
                           AND d.action = 'BUY'
                         ORDER BY d.date ASC
                         LIMIT 1),
                        s.rule
                    ) AS "Buy_Rule"
                FROM trades_summary s
                JOIN portfolios p ON s.portfolio_id = p.id
                WHERE p.name = %s
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


@st.cache_data(ttl=30, show_spinner=False)  # Cache for 30 seconds
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
                    d.retro_notes AS "Retro_Notes"
                FROM trades_details d
                JOIN portfolios p ON d.portfolio_id = p.id
                WHERE p.name = %s
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


@st.cache_data(ttl=30, show_spinner=False)  # Cache for 30 seconds
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

                cycle_select = 'j.market_cycle AS "Market Cycle",\n                        ' if has_market_cycle else ''
                query = f"""
                    SELECT
                        j.day AS "Day",
                        j.status AS "Status",
                        j.market_window AS "Market Window",
                        {cycle_select}j.above_21ema AS "> 21e",
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
                        j.top_lesson AS "Top_Lesson"
                    FROM trading_journal j
                    JOIN portfolios p ON j.portfolio_id = p.id
                    WHERE p.name = %s
                """
                # Build WHERE clause with safe string substitution
                # Note: Using direct substitution instead of params due to psycopg2/DSN compatibility issue
                # Safe because portfolio_name is from controlled selectbox with known values
                where_parts = [f"p.name = '{portfolio_name}'"]

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


# ============================================
# CORE WRITE OPERATIONS (Replace secure_save)
# ============================================
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

            # Check if trade exists
            cur.execute(
                "SELECT id FROM trades_summary WHERE portfolio_id = %s AND trade_id = %s",
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

            if existing:
                # UPDATE existing trade — try with grade, fall back without.
                try:
                    if update_grade:
                        update_query = """
                            UPDATE trades_summary
                            SET ticker = %s, status = %s, open_date = %s, closed_date = %s,
                                shares = %s, avg_entry = %s, avg_exit = %s, total_cost = %s,
                                realized_pl = %s, unrealized_pl = %s, return_pct = %s,
                                sell_rule = %s, notes = %s, stop_loss = %s, rule = %s,
                                buy_notes = %s, sell_notes = %s, risk_budget = %s,
                                grade = %s
                            WHERE id = %s
                            RETURNING id
                        """
                        params = (
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
                            existing[0],
                        )
                    else:
                        update_query = """
                            UPDATE trades_summary
                            SET ticker = %s, status = %s, open_date = %s, closed_date = %s,
                                shares = %s, avg_entry = %s, avg_exit = %s, total_cost = %s,
                                realized_pl = %s, unrealized_pl = %s, return_pct = %s,
                                sell_rule = %s, notes = %s, stop_loss = %s, rule = %s,
                                buy_notes = %s, sell_notes = %s, risk_budget = %s
                            WHERE id = %s
                            RETURNING id
                        """
                        params = (
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
                            existing[0],
                        )
                    cur.execute(update_query, params)
                except Exception:
                    # DB missing grade column — retry without
                    conn.rollback()
                    update_query = """
                        UPDATE trades_summary
                        SET ticker = %s, status = %s, open_date = %s, closed_date = %s,
                            shares = %s, avg_entry = %s, avg_exit = %s, total_cost = %s,
                            realized_pl = %s, unrealized_pl = %s, return_pct = %s,
                            sell_rule = %s, notes = %s, stop_loss = %s, rule = %s,
                            buy_notes = %s, sell_notes = %s, risk_budget = %s
                        WHERE id = %s
                        RETURNING id
                    """
                    cur.execute(update_query, (
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
                        existing[0],
                    ))
            else:
                # INSERT new trade — try with grade, fall back without.
                try:
                    insert_query = """
                        INSERT INTO trades_summary (
                            portfolio_id, trade_id, ticker, status, open_date, closed_date,
                            shares, avg_entry, avg_exit, total_cost, realized_pl, unrealized_pl,
                            return_pct, sell_rule, notes, stop_loss, rule, buy_notes, sell_notes, risk_budget,
                            grade
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
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
                        grade_clean,
                    ))
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


def save_detail_row(portfolio_name, row_dict):
    """
    Insert a transaction detail row.

    Args:
        portfolio_name: Portfolio name
        row_dict: Dictionary with column values

    Returns:
        int: ID of inserted row
    """
    # Sanitize numpy types
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

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Get portfolio_id
            cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
            result = cur.fetchone()
            if not result:
                raise ValueError(f"Portfolio '{portfolio_name}' not found")
            portfolio_id = result[0]

            insert_query = """
                INSERT INTO trades_details (
                    portfolio_id, trade_id, ticker, action, date, shares, amount, value,
                    rule, notes, realized_pl, stop_loss, trx_id, exec_grade, behavior_tag, retro_notes
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                RETURNING id
            """
            cur.execute(insert_query, (
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
                row_dict.get('Retro_Notes')
            ))

            row_id = cur.fetchone()[0]
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

            conn.commit()

            # Clear cache so next load gets fresh data
            load_details.clear()
            load_summary.clear()

            return True


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

            # Delete the row (with portfolio verification)
            cur.execute(
                "DELETE FROM trades_details WHERE id = %s AND portfolio_id = %s",
                (detail_id, portfolio_id)
            )

            if cur.rowcount == 0:
                raise ValueError(f"Detail row {detail_id} not found for portfolio '{portfolio_name}'")

            conn.commit()

            # Clear cache
            load_details.clear()
            load_summary.clear()

            return True


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

            cur.execute("""
                INSERT INTO audit_trail (portfolio_id, username, action, trade_id, ticker, details)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (portfolio_id, username, action, trade_id, ticker, details))

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
    Delete a trade and all its transactions.
    Explicitly deletes from both details and summary tables.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
            result = cur.fetchone()
            if not result:
                raise ValueError(f"Portfolio '{portfolio_name}' not found")
            portfolio_id = result[0]

            # Delete details first (child rows)
            cur.execute("""
                DELETE FROM trades_details
                WHERE portfolio_id = %s AND trade_id = %s
            """, (portfolio_id, trade_id))

            # Delete summary (parent row)
            cur.execute("""
                DELETE FROM trades_summary
                WHERE portfolio_id = %s AND trade_id = %s
            """, (portfolio_id, trade_id))

            conn.commit()

            # Clear cache so next load gets fresh data
            load_summary.clear()
            load_details.clear()


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
            ORDER BY p.name, s.open_date DESC
        """
        return pd.read_sql(query, conn)


# ============================================
# MARKET SIGNALS OPERATIONS
# ============================================

@st.cache_data(ttl=600, show_spinner=False)
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
                    update_query = """
                        UPDATE trading_journal
                        SET status = %s, market_window = %s, market_cycle = %s,
                            above_21ema = %s,
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
                        status, market_window, market_cycle, above_21ema,
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
                    insert_query = """
                        INSERT INTO trading_journal (
                            portfolio_id, day, status, market_window, market_cycle,
                            above_21ema,
                            cash_change, beg_nlv, end_nlv, daily_dollar_change,
                            daily_pct_change, pct_invested, spy, nasdaq,
                            market_notes, market_action, portfolio_heat,
                            spy_atr, nasdaq_atr, score,
                            highlights, lowlights, mistakes, top_lesson
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s
                        )
                        RETURNING id
                    """
                    cur.execute(insert_query, (
                        portfolio_id, day, status, market_window, market_cycle,
                        above_21ema,
                        cash_change, beg_nlv, end_nlv, daily_dollar_change,
                        daily_pct_change, pct_invested, spy, nasdaq,
                        market_notes, market_action, portfolio_heat,
                        spy_atr, nasdaq_atr, score,
                        highlights, lowlights, mistakes, top_lesson
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
                "DELETE FROM trading_journal WHERE portfolio_id = %s AND day = %s RETURNING id",
                (portfolio_id, day)
            )
            deleted = cur.fetchone()
            conn.commit()

            # Clear cache so next load gets fresh data
            load_journal.clear()

            return deleted is not None


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


@st.cache_data(ttl=120, show_spinner=False)
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


@st.cache_data(ttl=300, show_spinner=False)
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

                query = """
                    SELECT id, ticker, image_id, extracted_at,
                           composite_rating, eps_rating, rs_rating, group_rs_rating,
                           smr_rating, acc_dis_rating, timeliness_rating, sponsorship_rating,
                           eps_growth_rate, ud_vol_ratio,
                           mgmt_own_pct, banks_own_pct, funds_own_pct, num_funds,
                           price, market_cap, industry_group, industry_group_rank,
                           raw_json
                    FROM trade_fundamentals
                    WHERE portfolio_id = %s AND trade_id = %s
                    ORDER BY extracted_at DESC
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
@st.cache_data(ttl=60, show_spinner=False)
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


@st.cache_data(ttl=60, show_spinner=False)
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


@st.cache_data(ttl=60, show_spinner=False)
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
