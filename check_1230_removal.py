#!/usr/bin/env python3
"""Check when 12/30 should be removed (25 trading days)"""

from market_school_rules import MarketSchoolRules
import pandas as pd

print("=" * 80)
print("VERIFYING 12/30 REMOVAL DATE")
print("=" * 80)

analyzer = MarketSchoolRules("^IXIC")
analyzer.fetch_data(start_date="2025-12-01", end_date="2026-02-18")

# Find 12/30 in the data
start_date = pd.Timestamp('2025-12-30')
data_dates_normalized = [pd.Timestamp(d).tz_localize(None).normalize() for d in analyzer.data.index]

if start_date in data_dates_normalized:
    start_idx = data_dates_normalized.index(start_date)

    print(f"\n12/30/2025 is at index {start_idx}")
    print(f"\nCounting trading days from 12/30:")

    # Count trading days
    trading_days_list = []
    for i in range(start_idx, min(start_idx + 30, len(analyzer.data))):
        date = analyzer.data.index[i]
        days_elapsed = i - start_idx
        trading_days_list.append((date.strftime('%Y-%m-%d %A'), days_elapsed))

    print(f"\n{'Date':<20} {'Trading Days from 12/30':<25}")
    print("-" * 50)
    for date_str, days in trading_days_list:
        marker = " ← 25 TRADING DAYS" if days == 25 else ""
        print(f"{date_str:<20} {days:<25}{marker}")

    # Check if 25th trading day is 2/5 or 2/6
    if len(trading_days_list) > 25:
        day_25 = trading_days_list[25]
        print(f"\n✅ 25th trading day after 12/30 is: {day_25[0]}")
    else:
        print(f"\n❌ Not enough data to count 25 trading days")

    # IBD's rule: Remove on the date that is >= 25 trading days
    print(f"\nIBD Rule: Remove when >= 25 trading days have elapsed")
    print(f"So 12/30 should be removed on the 25th trading day")

print("\n" + "=" * 80)
