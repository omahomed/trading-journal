#!/usr/bin/env python3
"""Verify trading day counts for all removals"""

from market_school_rules import MarketSchoolRules
import pandas as pd

print("=" * 100)
print("VERIFYING ALL REMOVAL TIMING")
print("=" * 100)

analyzer = MarketSchoolRules("^IXIC")
analyzer.fetch_data(start_date="2025-12-01", end_date="2026-02-18")

# Check each distribution day removal
test_cases = [
    ('2025-12-05', '2026-01-13', '2026-01-14'),  # Expected vs Actual
    ('2025-12-11', '2026-01-20', '2026-01-21'),
    ('2025-12-12', '2026-01-21', '2026-01-22'),
    ('2025-12-17', '2026-01-26', '2026-01-27'),
    ('2025-12-29', '2026-02-04', '2026-02-05'),
    ('2025-12-30', '2026-02-06', '2026-02-06'),
]

print(f"\n{'DD Date':<15} {'IBD Removal':<15} {'Our Removal':<15} {'Days to IBD':<15} {'Days to Ours':<15}")
print("-" * 100)

for dd_date_str, ibd_removal_str, our_removal_str in test_cases:
    dd_date = pd.Timestamp(dd_date_str)
    ibd_removal = pd.Timestamp(ibd_removal_str)
    our_removal = pd.Timestamp(our_removal_str)

    # Count trading days to IBD's expected date
    data_dates_normalized = [pd.Timestamp(d).tz_localize(None).normalize() for d in analyzer.data.index]

    dd_idx = data_dates_normalized.index(dd_date) if dd_date in data_dates_normalized else -1
    ibd_idx = data_dates_normalized.index(ibd_removal) if ibd_removal in data_dates_normalized else -1
    our_idx = data_dates_normalized.index(our_removal) if our_removal in data_dates_normalized else -1

    if dd_idx >= 0 and ibd_idx >= 0:
        days_to_ibd = ibd_idx - dd_idx
    else:
        days_to_ibd = -1

    if dd_idx >= 0 and our_idx >= 0:
        days_to_ours = our_idx - dd_idx
    else:
        days_to_ours = -1

    match = "✅" if ibd_removal_str == our_removal_str else "❌"
    print(f"{dd_date_str:<15} {ibd_removal_str:<15} {our_removal_str:<15} {days_to_ibd:<15} {days_to_ours:<15} {match}")

print("\n" + "=" * 100)
print("OBSERVATIONS:")
print("=" * 100)
print("\nIf IBD expects removal consistently at 25 or 26 trading days, all values should be the same.")
print("If values differ, IBD might have inconsistent counting or different calendar.")
print("=" * 100)
