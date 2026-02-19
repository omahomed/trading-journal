#!/usr/bin/env python3
"""Check distribution day removals against IBD expectations"""

from market_school_rules import MarketSchoolRules
import pandas as pd

print("=" * 100)
print("DISTRIBUTION DAY REMOVAL VALIDATION")
print("=" * 100)

analyzer = MarketSchoolRules("^IXIC")
analyzer.fetch_data(start_date="2024-11-01", end_date="2026-02-18")
analyzer.analyze_market()

# Expected removals from user
expected_removals = {
    '2026-01-13': ['2025-12-05'],
    '2026-01-20': ['2025-12-11'],
    '2026-01-21': ['2025-12-12'],
    '2026-01-26': ['?'],  # User said "?" - need to figure out which one
    '2026-02-04': ['2025-12-29'],
    '2026-02-06': ['2025-12-30'],
}

print("\nEXPECTED REMOVALS:")
print(f"{'Removal Date':<15} {'Expected DD Removed':<20} {'Actual DD Removed':<20} {'Status':<20}")
print("-" * 100)

for removal_date_str, expected_dd_dates in expected_removals.items():
    removal_date = pd.Timestamp(removal_date_str).normalize()

    # Find distributions that were removed on this date
    removed_dds = [
        dd for dd in analyzer.distribution_days
        if dd.removed_date and pd.Timestamp(dd.removed_date).tz_localize(None).normalize() == removal_date
    ]

    if expected_dd_dates[0] == '?':
        # Just show what was removed
        if removed_dds:
            for dd in removed_dds:
                actual_dd_str = dd.date.strftime('%Y-%m-%d')
                print(f"{removal_date_str:<15} {'?':<20} {actual_dd_str:<20} {'ℹ️  User unsure':<20}")
        else:
            print(f"{removal_date_str:<15} {'?':<20} {'None':<20} {'ℹ️  User unsure':<20}")
    else:
        # Compare expected vs actual
        actual_dd_dates = [dd.date.strftime('%Y-%m-%d') for dd in removed_dds]

        if set(expected_dd_dates) == set(actual_dd_dates):
            status = "✅ MATCH"
        else:
            status = "❌ MISMATCH"

        expected_str = ', '.join(expected_dd_dates)
        actual_str = ', '.join(actual_dd_dates) if actual_dd_dates else 'None'

        print(f"{removal_date_str:<15} {expected_str:<20} {actual_str:<20} {status:<20}")

print("\n" + "=" * 100)
print("ALL REMOVED DISTRIBUTION DAYS (Dec 2025 - Feb 2026):")
print("=" * 100)

removed_dds = [
    dd for dd in analyzer.distribution_days
    if dd.removed_date
    and pd.Timestamp(dd.date).year >= 2025 and pd.Timestamp(dd.date).month >= 12
]

print(f"\n{'Added Date':<15} {'Type':<15} {'Loss %':<10} {'Removed Date':<15} {'Reason':<20}")
print("-" * 100)
for dd in sorted(removed_dds, key=lambda x: x.removed_date):
    print(f"{dd.date.strftime('%Y-%m-%d'):<15} {dd.type:<15} {dd.loss_percent:>8.2f}% {dd.removed_date.strftime('%Y-%m-%d'):<15} {dd.removal_reason:<20}")

print("\n" + "=" * 100)
