#!/usr/bin/env python3
"""Check raw database values"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from db_layer import get_db_connection

print("=" * 80)
print("RAW DATABASE CHECK")
print("=" * 80)

with get_db_connection() as conn:
    with conn.cursor() as cur:
        # Check recent SPY data
        cur.execute("""
            SELECT signal_date, close_price, daily_change_pct, distribution_count,
                   buy_signals, sell_signals
            FROM market_signals
            WHERE symbol = 'SPY'
              AND signal_date BETWEEN '2026-01-29' AND '2026-02-06'
            ORDER BY signal_date
        """)

        print("\nSPY DATA (Jan 29 - Feb 6, 2026):")
        print(f"{'Date':<12} {'Close':<12} {'Daily %':<12} {'Dist Ct':<8} {'Signals':<30}")
        print("-" * 80)

        for row in cur.fetchall():
            date = row[0]
            close = row[1]
            daily_pct = row[2]
            dist_ct = row[3]
            buy_sigs = row[4] or ''
            sell_sigs = row[5] or ''
            signals = f"Buy:{buy_sigs} Sell:{sell_sigs}" if buy_sigs or sell_sigs else 'None'

            print(f"{date} ${close:<11.2f} {daily_pct:<11.2f}% {dist_ct:<8} {signals:<30}")

        # Check a wider range
        print("\n" + "=" * 80)
        cur.execute("""
            SELECT signal_date, close_price, daily_change_pct
            FROM market_signals
            WHERE symbol = 'SPY'
            ORDER BY signal_date
            LIMIT 10
        """)

        print("\nFIRST 10 SPY RECORDS:")
        print(f"{'Date':<12} {'Close':<12} {'Daily %':<12}")
        print("-" * 40)

        for row in cur.fetchall():
            print(f"{row[0]} ${row[1]:<11.2f} {row[2]:<11.2f}%")

print("\n" + "=" * 80)
