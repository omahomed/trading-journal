#!/usr/bin/env python3
"""Clear recent market signals data to allow fresh sync"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import database layer
import db_layer as db

print("=" * 80)
print("CLEARING RECENT MARKET SIGNALS DATA")
print("=" * 80)

# Get connection
from db_layer import get_db_connection

with get_db_connection() as conn:
    with conn.cursor() as cur:
        # Check current data
        print("\nCurrent data status:")
        cur.execute("""
            SELECT symbol, COUNT(*) as count, MIN(signal_date) as earliest, MAX(signal_date) as latest
            FROM market_signals
            GROUP BY symbol
            ORDER BY symbol
        """)

        for row in cur.fetchall():
            print(f"  {row[0]}: {row[1]} records from {row[2]} to {row[3]}")

        # Delete all records from 2025-02-24 onward to allow fresh sync
        print("\nDeleting records from 2025-02-24 onward...")
        cur.execute("""
            DELETE FROM market_signals
            WHERE signal_date >= '2025-02-24'
        """)

        deleted = cur.rowcount
        conn.commit()

        print(f"✅ Deleted {deleted} records")

        # Check remaining data
        print("\nRemaining data:")
        cur.execute("""
            SELECT symbol, COUNT(*) as count, MIN(signal_date) as earliest, MAX(signal_date) as latest
            FROM market_signals
            GROUP BY symbol
            ORDER BY symbol
        """)

        for row in cur.fetchall():
            print(f"  {row[0]}: {row[1]} records from {row[2]} to {row[3]}")

print("\n" + "=" * 80)
print("✅ Ready for fresh sync!")
print("Next steps:")
print("1. Go to IBD Market School page in the app")
print("2. Click '💾 Sync to Database'")
print("3. Wait for sync to complete")
print("=" * 80)
