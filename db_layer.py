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
                    s.risk_budget AS "Risk_Budget"
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
                    row_dict.get('Open_Date'),
                    row_dict.get('Closed_Date'),
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
                    row_dict.get('Open_Date'),
                    row_dict.get('Closed_Date'),
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
            return row_id


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
