# migrate_csv_to_postgres.py - Import CSV data to PostgreSQL

import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import os
from datetime import datetime

# Portfolio paths
PORTFOLIOS = {
    'CanSlim': '/Users/momacbookair/Library/Mobile Documents/com~apple~CloudDocs/my_code/portfolios/CanSlim',
    # Uncomment after CanSlim is validated:
    # 'TQQQ Strategy': '/Users/momacbookair/Library/Mobile Documents/com~apple~CloudDocs/my_code/portfolios/TQQQ Strategy',
    # '457B Plan': '/Users/momacbookair/Library/Mobile Documents/com~apple~CloudDocs/my_code/portfolios/457B Plan'
}

# Database connection
# Support both connection string (DATABASE_URL) and individual parameters
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL:
    # Use connection string (for cloud databases)
    DB_CONFIG = {'dsn': DATABASE_URL}
else:
    # Use individual parameters (for local database)
    DB_CONFIG = {
        'host': 'localhost',
        'port': 5432,
        'database': 'trading_journal',
        'user': os.getenv('USER', 'postgres'),
        'password': ''
    }


def clean_numeric(value):
    """Clean numeric values from CSV (handle $, commas, %, etc.)"""
    if pd.isna(value) or value == '':
        return None
    try:
        s = str(value).replace('$', '').replace(',', '').replace('%', '').strip()
        if '(' in s:
            s = s.replace('(', '-').replace(')', '')
        if s == '' or s == 'nan':
            return None
        return float(s)
    except:
        return None


def clean_text(value):
    """Clean text values"""
    if pd.isna(value) or value == '' or value == 'nan':
        return ''
    return str(value).strip()


def clean_date(value):
    """Clean date values - convert NaT to None"""
    if pd.isna(value):
        return None
    try:
        dt = pd.to_datetime(value, errors='coerce')
        if pd.isna(dt):
            return None
        return dt
    except:
        return None


def import_summary(conn, portfolio_name, csv_path):
    """Import Trade_Log_Summary.csv"""
    print(f"  Importing Summary from {csv_path}")

    if not os.path.exists(csv_path):
        print(f"    SKIP: File not found")
        return 0

    df = pd.read_csv(csv_path)
    print(f"    Read {len(df)} rows from CSV")

    # Get portfolio_id
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
        portfolio_id = cur.fetchone()[0]

    # Prepare data
    rows = []
    for _, row in df.iterrows():
        # Clean Trade_ID (remove trailing .0)
        trade_id = str(row.get('Trade_ID', '')).replace('.0', '')

        rows.append((
            portfolio_id,
            trade_id,
            clean_text(row.get('Ticker', '')),
            clean_text(row.get('Status', 'OPEN')),
            clean_date(row.get('Open_Date')),
            clean_date(row.get('Closed_Date')),
            clean_numeric(row.get('Shares', 0)),
            clean_numeric(row.get('Avg_Entry', 0)),
            clean_numeric(row.get('Avg_Exit', 0)),
            clean_numeric(row.get('Total_Cost', 0)),
            clean_numeric(row.get('Realized_PL', 0)),
            clean_numeric(row.get('Unrealized_PL', 0)),
            clean_numeric(row.get('Return_Pct', 0)),
            clean_text(row.get('Sell_Rule', '')),
            clean_text(row.get('Notes', '')),
            clean_numeric(row.get('Stop_Loss')),
            clean_text(row.get('Rule', '')),
            clean_text(row.get('Buy_Notes', '')),
            clean_text(row.get('Sell_Notes', '')),
            clean_numeric(row.get('Risk_Budget', 0))
        ))

    # Bulk insert
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO trades_summary (
                portfolio_id, trade_id, ticker, status, open_date, closed_date,
                shares, avg_entry, avg_exit, total_cost, realized_pl, unrealized_pl,
                return_pct, sell_rule, notes, stop_loss, rule, buy_notes, sell_notes, risk_budget
            ) VALUES %s
            ON CONFLICT (portfolio_id, trade_id) DO NOTHING
        """, rows)

    conn.commit()
    print(f"    ✅ Imported {len(rows)} summary rows")
    return len(rows)


def import_details(conn, portfolio_name, csv_path):
    """Import Trade_Log_Details.csv"""
    print(f"  Importing Details from {csv_path}")

    if not os.path.exists(csv_path):
        print(f"    SKIP: File not found")
        return 0

    df = pd.read_csv(csv_path)
    print(f"    Read {len(df)} rows from CSV")

    # Get portfolio_id
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
        portfolio_id = cur.fetchone()[0]

    # Prepare data
    rows = []
    for _, row in df.iterrows():
        # Clean Trade_ID
        trade_id = str(row.get('Trade_ID', '')).replace('.0', '')

        rows.append((
            portfolio_id,
            trade_id,
            clean_text(row.get('Ticker', '')),
            clean_text(row.get('Action', 'BUY')),
            clean_date(row.get('Date')),
            clean_numeric(row.get('Shares', 0)),
            clean_numeric(row.get('Amount', 0)),
            clean_numeric(row.get('Value', 0)),
            clean_text(row.get('Rule', '')),
            clean_text(row.get('Notes', '')),
            clean_numeric(row.get('Realized_PL', 0)),
            clean_numeric(row.get('Stop_Loss')),
            clean_text(row.get('Trx_ID', '')),
            clean_text(row.get('Exec_Grade', '')),
            clean_text(row.get('Behavior_Tag', '')),
            clean_text(row.get('Retro_Notes', ''))
        ))

    # Bulk insert
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO trades_details (
                portfolio_id, trade_id, ticker, action, date, shares, amount, value,
                rule, notes, realized_pl, stop_loss, trx_id, exec_grade, behavior_tag, retro_notes
            ) VALUES %s
        """, rows)

    conn.commit()
    print(f"    ✅ Imported {len(rows)} detail rows")
    return len(rows)


def import_journal(conn, portfolio_name, csv_path):
    """Import Trading_Journal_Clean.csv"""
    print(f"  Importing Journal from {csv_path}")

    if not os.path.exists(csv_path):
        print(f"    SKIP: File not found")
        return 0

    df = pd.read_csv(csv_path)
    print(f"    Read {len(df)} rows from CSV")

    # Get portfolio_id
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
        portfolio_id = cur.fetchone()[0]

    # Prepare data
    rows = []
    for _, row in df.iterrows():
        rows.append((
            portfolio_id,
            clean_date(row.get('Day')),
            clean_text(row.get('Status', '')),
            clean_text(row.get('Market Window', '')),
            int(clean_numeric(row.get('> 21e', 0)) or 0),
            clean_numeric(row.get('Cash -/+', 0)),
            clean_numeric(row.get('Beg NLV', 0)),
            clean_numeric(row.get('End NLV', 0)),
            clean_numeric(row.get('Daily $ Change', 0)),
            clean_numeric(row.get('Daily % Change', 0)),
            clean_numeric(row.get('% Invested', 0)),
            clean_numeric(row.get('SPY', 0)),
            clean_numeric(row.get('Nasdaq', 0)),
            clean_text(row.get('Market_Notes', '')),
            clean_text(row.get('Market_Action', '')),
            int(clean_numeric(row.get('Score', 0)) or 0),
            clean_text(row.get('Highlights', '')),
            clean_text(row.get('Lowlights', '')),
            clean_text(row.get('Mistakes', '')),
            clean_text(row.get('Top_Lesson', ''))
        ))

    # Bulk insert
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO trading_journal (
                portfolio_id, day, status, market_window, above_21ema, cash_change,
                beg_nlv, end_nlv, daily_dollar_change, daily_pct_change, pct_invested,
                spy, nasdaq, market_notes, market_action, score, highlights, lowlights,
                mistakes, top_lesson
            ) VALUES %s
            ON CONFLICT (portfolio_id, day) DO NOTHING
        """, rows)

    conn.commit()
    print(f"    ✅ Imported {len(rows)} journal rows")
    return len(rows)


def import_audit(conn, portfolio_name, csv_path):
    """Import Audit_Trail.csv"""
    print(f"  Importing Audit from {csv_path}")

    if not os.path.exists(csv_path):
        print(f"    SKIP: File not found (this is OK, new file)")
        return 0

    df = pd.read_csv(csv_path)
    print(f"    Read {len(df)} rows from CSV")

    # Get portfolio_id
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
        portfolio_id = cur.fetchone()[0]

    # Prepare data
    rows = []
    for _, row in df.iterrows():
        rows.append((
            portfolio_id,
            clean_date(row.get('Timestamp')),
            clean_text(row.get('User', 'User')),
            clean_text(row.get('Action', '')),
            clean_text(row.get('Trade_ID', '')),
            clean_text(row.get('Ticker', '')),
            clean_text(row.get('Details', ''))
        ))

    # Bulk insert
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO audit_trail (
                portfolio_id, timestamp, username, action, trade_id, ticker, details
            ) VALUES %s
        """, rows)

    conn.commit()
    print(f"    ✅ Imported {len(rows)} audit rows")
    return len(rows)


def validate_migration(conn):
    """Run validation queries after migration"""
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    with conn.cursor() as cur:
        # Count summary records
        cur.execute("""
            SELECT p.name, COUNT(*) as count
            FROM trades_summary s
            JOIN portfolios p ON s.portfolio_id = p.id
            GROUP BY p.name
            ORDER BY p.name
        """)
        print("\n📊 Summary Rows by Portfolio:")
        for row in cur.fetchall():
            print(f"  {row[0]}: {row[1]} trades")

        # Count detail records
        cur.execute("""
            SELECT p.name, COUNT(*) as count
            FROM trades_details d
            JOIN portfolios p ON d.portfolio_id = p.id
            GROUP BY p.name
            ORDER BY p.name
        """)
        print("\n📝 Detail Rows by Portfolio:")
        for row in cur.fetchall():
            print(f"  {row[0]}: {row[1]} transactions")

        # Count journal records
        cur.execute("""
            SELECT p.name, COUNT(*) as count
            FROM trading_journal j
            JOIN portfolios p ON j.portfolio_id = p.id
            GROUP BY p.name
            ORDER BY p.name
        """)
        print("\n📅 Journal Rows by Portfolio:")
        for row in cur.fetchall():
            print(f"  {row[0]}: {row[1]} days")

        # Check for orphaned details
        cur.execute("""
            SELECT COUNT(*)
            FROM trades_details d
            LEFT JOIN trades_summary s ON d.portfolio_id = s.portfolio_id AND d.trade_id = s.trade_id
            WHERE s.id IS NULL
        """)
        orphans = cur.fetchone()[0]
        if orphans == 0:
            print(f"\n✅ Orphaned Detail Rows: {orphans} (perfect!)")
        else:
            print(f"\n⚠️  Orphaned Detail Rows: {orphans} (should investigate)")

        # Sample some data
        print("\n🔍 Sample Data (First 3 Trades):")
        cur.execute("""
            SELECT trade_id, ticker, status, shares, avg_entry, realized_pl
            FROM trades_summary
            ORDER BY open_date DESC
            LIMIT 3
        """)
        print("  Trade_ID | Ticker | Status | Shares | Avg_Entry | Realized_PL")
        print("  " + "-" * 60)
        for row in cur.fetchall():
            print(f"  {row[0]} | {row[1]} | {row[2]} | {row[3]:.2f} | ${row[4]:.2f} | ${row[5]:.2f}")


def main():
    """Main migration orchestrator"""
    print("=" * 60)
    print("CSV TO POSTGRESQL MIGRATION")
    print("=" * 60)

    # Connect to database
    print("\n1. Connecting to PostgreSQL...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("   ✅ Connected!")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return

    try:
        total_summary = 0
        total_details = 0
        total_journal = 0
        total_audit = 0

        # Import each portfolio
        for portfolio_name, portfolio_path in PORTFOLIOS.items():
            print(f"\n2. Importing Portfolio: {portfolio_name}")
            print(f"   Path: {portfolio_path}")

            # Import in order: Summary first (so foreign keys work)
            total_summary += import_summary(conn, portfolio_name, os.path.join(portfolio_path, 'Trade_Log_Summary.csv'))
            total_details += import_details(conn, portfolio_name, os.path.join(portfolio_path, 'Trade_Log_Details.csv'))
            total_journal += import_journal(conn, portfolio_name, os.path.join(portfolio_path, 'Trading_Journal_Clean.csv'))
            total_audit += import_audit(conn, portfolio_name, os.path.join(portfolio_path, 'Audit_Trail.csv'))

        # Run validation
        validate_migration(conn)

        print("\n" + "=" * 60)
        print("MIGRATION COMPLETE!")
        print("=" * 60)
        print(f"\n📊 Total Imported:")
        print(f"  Summary: {total_summary} rows")
        print(f"  Details: {total_details} rows")
        print(f"  Journal: {total_journal} rows")
        print(f"  Audit: {total_audit} rows")

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
    finally:
        conn.close()
        print("\n✅ Database connection closed.")


if __name__ == '__main__':
    main()
