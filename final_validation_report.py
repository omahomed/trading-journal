#!/usr/bin/env python3
"""Final validation report comparing our data with IBD expectations"""

from market_school_rules import MarketSchoolRules
import pandas as pd

print("=" * 100)
print("NASDAQ DISTRIBUTION DAY VALIDATION: OUR DATA vs IBD EXPECTATIONS")
print("=" * 100)

analyzer = MarketSchoolRules("^IXIC")
analyzer.fetch_data(start_date="2024-11-01", end_date="2026-02-18")
analyzer.analyze_market()

# User's expected changes
expected_changes = [
    {'date': '2026-01-13', 'add': False, 'remove': ['2025-12-05']},
    {'date': '2026-01-14', 'add': True, 'remove': []},
    {'date': '2026-01-20', 'add': False, 'remove': ['2025-12-11']},
    {'date': '2026-01-21', 'add': False, 'remove': ['2025-12-12']},
    {'date': '2026-01-22', 'add': False, 'remove': []},
    {'date': '2026-01-26', 'add': False, 'remove': ['?']},
    {'date': '2026-01-28', 'add': True, 'remove': []},  # Stall day
    {'date': '2026-01-29', 'add': True, 'remove': []},
    {'date': '2026-01-30', 'add': True, 'remove': []},
    {'date': '2026-02-03', 'add': True, 'remove': []},
    {'date': '2026-02-04', 'add': True, 'remove': ['2025-12-29']},
    {'date': '2026-02-06', 'add': True, 'remove': ['2025-12-30']},
    {'date': '2026-02-10', 'add': True, 'remove': []},
]

print(f"\n{'Date':<12} {'Our Daily%':<12} {'IBD Add?':<10} {'We Add?':<10} {'Status':<30}")
print("-" * 100)

for expected in expected_changes:
    date_str = expected['date']
    date = pd.Timestamp(date_str)

    # Get our data
    data_dates_normalized = [pd.Timestamp(d).tz_localize(None).normalize() for d in analyzer.data.index]
    date_normalized = date.normalize()

    if date_normalized in data_dates_normalized:
        idx = data_dates_normalized.index(date_normalized)
        row = analyzer.data.iloc[idx]
        daily_pct = row['daily_gain_pct']

        # Check if we added a distribution on this date
        we_added = any(
            pd.Timestamp(dd.date).tz_localize(None).normalize() == date_normalized
            for dd in analyzer.distribution_days
        )

        ibd_expects = expected['add']

        if ibd_expects == we_added:
            status = "✅ MATCH"
        else:
            if ibd_expects and not we_added:
                status = "❌ DATA MISMATCH: IBD expects add"
            elif not ibd_expects and we_added:
                status = "❌ DATA MISMATCH: We added, IBD didn't"
            else:
                status = "?"

        print(f"{date_str:<12} {daily_pct:>10.2f}% {'Yes' if ibd_expects else 'No':<10} {'Yes' if we_added else 'No':<10} {status:<30}")

print("\n" + "=" * 100)
print("DATA SOURCE DISCREPANCIES:")
print("=" * 100)
print("\nOur data (yfinance) may differ from IBD's data source (e.g., different close prices).")
print("This can cause distribution day detection differences.\n")
print("Examples where yfinance data conflicts with IBD expectations:")
print("  • 1/20: yfinance shows -2.39% (should add DD), IBD says don't add")
print("  • 2/6:  yfinance shows +2.18% (can't add DD), IBD says add")
print("\n" + "=" * 100)
print("CURRENT ACTIVE DISTRIBUTION DAYS:")
print("=" * 100)

active_dds = [dd for dd in analyzer.distribution_days if dd.removed_date is None]
print(f"\nTotal Active: {len(active_dds)}")
print(f"\n{'Date':<15} {'Type':<15} {'Loss %':<10}")
print("-" * 50)
for dd in sorted(active_dds, key=lambda x: x.date):
    print(f"{dd.date.strftime('%Y-%m-%d'):<15} {dd.type:<15} {dd.loss_percent:>8.2f}%")

print("\n" + "=" * 100)
