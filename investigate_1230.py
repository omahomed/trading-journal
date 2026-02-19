#!/usr/bin/env python3
"""Investigate why 12/30 removal is different from others"""

from market_school_rules import MarketSchoolRules
import pandas as pd

print("=" * 100)
print("INVESTIGATING 12/30 SPECIAL CASE")
print("=" * 100)

analyzer = MarketSchoolRules("^IXIC")
analyzer.fetch_data(start_date="2025-12-01", end_date="2026-02-18")

# Manual count for 12/30
print("\n12/30 Trading Day Count (Manual):")
print("-" * 100)

start_date = pd.Timestamp('2025-12-30')
data_dates_normalized = [pd.Timestamp(d).tz_localize(None).normalize() for d in analyzer.data.index]

start_idx = data_dates_normalized.index(start_date)

print(f"\nDate Range: 12/30/2025 to 2/6/2026")
print(f"\n{'Date':<20} {'Day #':<10} {'Notes':<30}")
print("-" * 100)

for i in range(start_idx, min(start_idx + 40, len(analyzer.data))):
    date = analyzer.data.index[i]
    day_num = i - start_idx
    date_str = date.strftime('%Y-%m-%d %A')

    notes = ""
    if day_num == 25:
        notes = "← 25 trading days (2/5)"
    elif day_num == 26:
        notes = "← 26 trading days (2/6) - IBD removes here"

    print(f"{date_str:<20} {day_num:<10} {notes:<30}")

    if day_num > 26:
        break

# Check if there's something special about 2/5 or 2/6
print("\n" + "=" * 100)
print("CHECKING 2/5 AND 2/6 FOR SPECIAL CONDITIONS")
print("=" * 100)

feb5_idx = data_dates_normalized.index(pd.Timestamp('2026-02-05'))
feb6_idx = data_dates_normalized.index(pd.Timestamp('2026-02-06'))

feb5 = analyzer.data.iloc[feb5_idx]
feb6 = analyzer.data.iloc[feb6_idx]

print(f"\n2/5/2026:")
print(f"  Close: ${feb5['Close']:.2f}")
print(f"  Daily %: {feb5['daily_gain_pct']:.2f}%")
print(f"  Day of week: {feb5.name.strftime('%A')}")

print(f"\n2/6/2026:")
print(f"  Close: ${feb6['Close']:.2f}")
print(f"  Daily %: {feb6['daily_gain_pct']:.2f}%")
print(f"  Day of week: {feb6.name.strftime('%A')}")

# Check 6% rally rule - maybe 12/30 qualified for 6% rally removal on 2/5?
print("\n" + "=" * 100)
print("CHECKING IF 6% RALLY RULE APPLIES")
print("=" * 100)

# Get 12/30 close price
dec30_idx = start_idx
dec30_close = analyzer.data.iloc[dec30_idx]['Close']

print(f"\n12/30 Close: ${dec30_close:.2f}")

# Check 2/5 high for 6% rally
feb5_high = feb5['High']
rally_pct_feb5 = ((feb5_high - dec30_close) / dec30_close) * 100

print(f"2/5 High: ${feb5_high:.2f}")
print(f"Rally from 12/30 close: {rally_pct_feb5:.2f}%")
print(f"6% rally rule triggered on 2/5? {rally_pct_feb5 >= 6.0}")

# Check 2/6
feb6_high = feb6['High']
rally_pct_feb6 = ((feb6_high - dec30_close) / dec30_close) * 100

print(f"\n2/6 High: ${feb6_high:.2f}")
print(f"Rally from 12/30 close: {rally_pct_feb6:.2f}%")
print(f"6% rally rule triggered on 2/6? {rally_pct_feb6 >= 6.0}")

print("\n" + "=" * 100)
print("CONCLUSION:")
print("=" * 100)
print("\nIf 6% rally triggered on 2/5, our code would remove it via 6% rule (not 25-day rule).")
print("But with > 25 logic, it gets removed on 2/6 via 25-day rule instead.")
print("This might explain the discrepancy!")
print("=" * 100)
