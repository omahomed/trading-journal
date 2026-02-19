#!/usr/bin/env python3
"""Check NASDAQ data for specific dates to verify distribution detection"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from db_layer import get_db_connection

print("=" * 80)
print("NASDAQ DATA FOR SPECIFIC DATES")
print("=" * 80)

with get_db_connection() as conn:
    with conn.cursor() as cur:
        # Check the dates mentioned by user
        cur.execute("""
            SELECT signal_date, close_price, daily_change_pct, distribution_count,
                   buy_signals, sell_signals
            FROM market_signals
            WHERE symbol = '^IXIC'
              AND signal_date IN (
                '2026-01-13', '2026-01-14', '2026-01-20', '2026-01-21',
                '2026-01-22', '2026-01-26', '2026-01-28', '2026-01-29',
                '2026-01-30', '2026-02-03', '2026-02-04', '2026-02-06',
                '2026-02-10'
              )
            ORDER BY signal_date
        """)

        print("\nNASDAQ KEY DATES:")
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

print("\n" + "=" * 80)
