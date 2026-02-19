#!/usr/bin/env python3
"""Check what dates are available around missing dates"""

from market_school_rules import MarketSchoolRules
import pandas as pd

analyzer = MarketSchoolRules("^IXIC")
analyzer.fetch_data(start_date="2024-11-01", end_date="2026-02-18")

print("=" * 80)
print("DATES AROUND 2025-12-05")
print("=" * 80)

dec_dates = [d for d in analyzer.data.index if '2025-12' in str(d)]
for d in dec_dates[:10]:
    print(f"  {d.strftime('%Y-%m-%d %A')}")

print("\n" + "=" * 80)
print("DATES AROUND 2026-01-28")
print("=" * 80)

jan_dates = [d for d in analyzer.data.index if '2026-01' in str(d)]
for d in jan_dates[-10:]:
    print(f"  {d.strftime('%Y-%m-%d %A')}")

print("\n" + "=" * 80)
print("CHECK SPECIFIC DATES")
print("=" * 80)

check_dates = ['2025-12-04', '2025-12-05', '2025-12-06',
               '2026-01-27', '2026-01-28', '2026-01-29']

for date_str in check_dates:
    date = pd.Timestamp(date_str)
    if any(pd.Timestamp(d).normalize() == date.normalize() for d in analyzer.data.index):
        print(f"  ✅ {date_str} - Found")
        # Get the actual date
        actual = [d for d in analyzer.data.index if pd.Timestamp(d).normalize() == date.normalize()][0]
        print(f"      {actual.strftime('%Y-%m-%d %A')}")
    else:
        day_of_week = date.strftime('%A')
        print(f"  ❌ {date_str} ({day_of_week}) - NOT Found (Weekend or Holiday)")

print("\n" + "=" * 80)
