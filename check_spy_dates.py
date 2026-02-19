#!/usr/bin/env python3
"""Check SPY distribution days for Feb 5 and Jan 30, 2026"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import database layer
import db_layer as db

print("=" * 80)
print("SPY DISTRIBUTION DAY VALIDATION")
print("=" * 80)

# Load market signals for SPY in the date range
df = db.load_market_signals(symbol='SPY', days=30)

if df.empty:
    print("❌ No SPY data found in database")
    sys.exit(1)

# Check Feb 5, 2026
feb5 = df[df['signal_date'] == pd.Timestamp('2026-02-05')]
if not feb5.empty:
    row = feb5.iloc[0]
    print("\n📅 FEBRUARY 5, 2026 (SPY):")
    print(f"   Close: ${row['close_price']:.2f}")
    print(f"   Daily Change: {row['daily_change_pct']:.2f}%")
    print(f"   Distribution Count: {row['distribution_count']}")
    print(f"   Buy Signals: {row['buy_signals']}")
    print(f"   Sell Signals: {row['sell_signals']}")
    print(f"   ⚠️ IBD shows: 4 distributions (NO distribution this day)")
    print(f"   ❌ DISCREPANCY: We likely counted a stall day here")
else:
    print("\n❌ No data for Feb 5, 2026")

# Check Jan 30, 2026
jan30 = df[df['signal_date'] == pd.Timestamp('2026-01-30')]
if not jan30.empty:
    row = jan30.iloc[0]
    print("\n📅 JANUARY 30, 2026 (SPY):")
    print(f"   Close: ${row['close_price']:.2f}")
    print(f"   Daily Change: {row['daily_change_pct']:.2f}%")
    print(f"   Distribution Count: {row['distribution_count']}")
    print(f"   Buy Signals: {row['buy_signals']}")
    print(f"   Sell Signals: {row['sell_signals']}")
    print(f"   ⚠️ IBD shows: 3 distributions (NO distribution - volume was LOWER)")
    print(f"   ❌ DISCREPANCY: We likely counted this as distribution")
else:
    print("\n❌ No data for Jan 30, 2026")

# Show context
print("\n" + "=" * 80)
print("CONTEXT: SPY Jan 29 - Feb 6, 2026")
print("=" * 80)

context_df = df[(df['signal_date'] >= '2026-01-29') & (df['signal_date'] <= '2026-02-06')].copy()
context_df = context_df.sort_values('signal_date')

print(f"\n{'Date':<12} {'Close':<10} {'Change %':<10} {'Dist Ct':<8} {'Signals':<30}")
print("-" * 80)

for _, row in context_df.iterrows():
    date_str = row['signal_date'].strftime('%Y-%m-%d')
    signals = []
    if pd.notna(row['buy_signals']):
        signals.append(f"Buy:{row['buy_signals']}")
    if pd.notna(row['sell_signals']):
        signals.append(f"Sell:{row['sell_signals']}")
    signal_str = ", ".join(signals) if signals else "None"

    print(f"{date_str:<12} ${row['close_price']:<9.2f} {row['daily_change_pct']:<9.2f}% {row['distribution_count']:<8} {signal_str:<30}")

print("\n" + "=" * 80)
