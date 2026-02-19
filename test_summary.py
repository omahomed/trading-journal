#!/usr/bin/env python3
"""Test get_daily_summary to see if it returns correct data"""

from market_school_rules import MarketSchoolRules
from datetime import datetime

print("=" * 80)
print("TESTING get_daily_summary")
print("=" * 80)

symbol = "SPY"
analyzer = MarketSchoolRules(symbol)
analyzer.fetch_data(start_date="2024-02-24", end_date="2026-02-18")

print(f"\nFetched {len(analyzer.data)} days of data")
print(f"Date range: {analyzer.data.index[0]} to {analyzer.data.index[-1]}")

analyzer.analyze_market()
print(f"Generated {len(analyzer.signals)} signals")

# Test a few different dates
test_dates = ['2026-01-29', '2026-01-30', '2026-02-03', '2026-02-04', '2026-02-05']

print("\nTesting get_daily_summary for different dates:")
print(f"{'Date':<12} {'Close':<12} {'Daily %':<12} {'Dist Ct':<8}")
print("-" * 50)

for date_str in test_dates:
    summary = analyzer.get_daily_summary(date_str)
    print(f"{date_str:<12} {summary['close']:<12} {summary['daily_change']:<12} {summary['distribution_count']:<8}")

# Also check the raw data
print("\n" + "=" * 80)
print("RAW DATA from analyzer.data:")
print(f"{'Date':<12} {'Close':<12} {'Daily Ret %':<12}")
print("-" * 50)

for date_str in test_dates:
    date = analyzer.data.index[analyzer.data.index == date_str]
    if len(date) > 0:
        row = analyzer.data.loc[date[0]]
        daily_pct = row['daily_gain_pct'] if 'daily_gain_pct' in row else row.get('daily_return', 0) * 100
        print(f"{date_str:<12} ${row['Close']:<11.2f} {daily_pct:<11.2f}%")

print("\n" + "=" * 80)
