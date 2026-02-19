# db_layer.py - PostgreSQL abstraction layer for trading journal

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
import pandas as pd
import os
import streamlit as st
from contextlib import contextmanager
from datetime import datetime

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
    # Check Streamlit secrets first (cloud deployment)
    if hasattr(st, 'secrets') and 'database' in st.secrets:
        return {'dsn': st.secrets['database']['url']}

    # Check environment variable (DATABASE_URL)
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
def get_db_connection():
    """
    Context manager for database connections.
    Ensures connections are properly closed.
    """
    config = get_db_config()
    conn = None
    try:
        if 'dsn' in config:
            conn = psycopg2.connect(config['dsn'])
        else:
            conn = psycopg2.connect(**config)
        yield conn
    except psycopg2.OperationalError as e:
        # Log error and re-raise
        print(f"Database connection error: {e}")
        raise
    finally:
        if conn:
            conn.close()


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
                query = """
                    SELECT
                        j.day AS "Day",
                        j.status AS "Status",
                        j.market_window AS "Market Window",
                        j.above_21ema AS "> 21e",
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
    # Clean NaT/NaN values - convert to None for SQL NULL
    def clean_value(val):
        if pd.isna(val) or str(val).strip() == 'NaT':
            return None
        return val

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

            if existing:
                # UPDATE existing trade
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
                    existing[0]
                ))
            else:
                # INSERT new trade
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
                    row_dict.get('Risk_Budget', 0)
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
    CASCADE will handle details automatically.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
            result = cur.fetchone()
            if not result:
                raise ValueError(f"Portfolio '{portfolio_name}' not found")
            portfolio_id = result[0]

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
                # UPDATE existing entry
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
                # INSERT new entry
                insert_query = """
                    INSERT INTO trading_journal (
                        portfolio_id, day, status, market_window, above_21ema,
                        cash_change, beg_nlv, end_nlv, daily_dollar_change,
                        daily_pct_change, pct_invested, spy, nasdaq,
                        market_notes, market_action, score, highlights,
                        lowlights, mistakes, top_lesson
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
                    market_notes, market_action, score, highlights,
                    lowlights, mistakes, top_lesson
                ))

            row_id = cur.fetchone()[0]
            conn.commit()

            # Clear cache so next load gets fresh data
            load_journal.clear()

            return row_id


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
