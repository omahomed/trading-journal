#!/usr/bin/env python3
"""Check SPY distribution day removals"""

from market_school_rules import MarketSchoolRules
import pandas as pd

print("=" * 100)
print("SPY DISTRIBUTION DAY REMOVALS")
print("=" * 100)

analyzer = MarketSchoolRules("SPY")
analyzer.fetch_data(start_date="2025-12-01", end_date="2026-02-18")
analyzer.analyze_market()

# User mentioned:
# - 12/26 removed on 1/23
# - Another one removed on 1/26 (not sure which date)
# - 12/29 was a distribution, not sure when it comes off

print("\nALL REMOVED DISTRIBUTION DAYS (Dec 2025 - Jan 2026):")
print("=" * 100)

removed_dds = [
    dd for dd in analyzer.distribution_days
    if dd.removed_date and pd.Timestamp(dd.date).month == 12
]

print(f"\n{'Added Date':<15} {'Type':<15} {'Loss %':<10} {'Removed Date':<15} {'Trading Days':<15} {'Reason':<25}")
print("-" * 100)

for dd in sorted(removed_dds, key=lambda x: x.removed_date):
    # Count trading days
    trading_days = analyzer._count_trading_days(dd.date, dd.removed_date)

    print(f"{dd.date.strftime('%Y-%m-%d'):<15} {dd.type:<15} {dd.loss_percent:>8.2f}% "
          f"{dd.removed_date.strftime('%Y-%m-%d'):<15} {trading_days:<15} {dd.removal_reason:<25}")

print("\n" + "=" * 100)
print("IBD EXPECTED REMOVALS:")
print("=" * 100)

expected_removals = [
    {'added': '12/26', 'removed': '1/23', 'status': '?'},
    {'added': '?', 'removed': '1/26', 'status': '?'},
    {'added': '12/29', 'removed': '?', 'status': 'Unknown when it comes off'},
]

print(f"\n{'IBD Added':<15} {'IBD Removed':<15} {'Notes':<50}")
print("-" * 100)
for exp in expected_removals:
    print(f"{exp['added']:<15} {exp['removed']:<15} {exp['status']:<50}")

print("\n" + "=" * 100)
print("CHECKING SPECIFIC DATES:")
print("=" * 100)

# Check if 12/26 exists in our data
dec26 = [dd for dd in analyzer.distribution_days if pd.Timestamp(dd.date).strftime('%Y-%m-%d') == '2025-12-26']
if dec26:
    dd = dec26[0]
    removed_str = dd.removed_date.strftime('%Y-%m-%d') if dd.removed_date else 'Active'
    print(f"\n12/26/2025: Found")
    print(f"  Type: {dd.type}")
    print(f"  Loss %: {dd.loss_percent:.2f}%")
    print(f"  Removed: {removed_str}")
else:
    print(f"\n12/26/2025: NOT found in our data")

# Check 12/29
dec29 = [dd for dd in analyzer.distribution_days if pd.Timestamp(dd.date).strftime('%Y-%m-%d') == '2025-12-29']
if dec29:
    dd = dec29[0]
    removed_str = dd.removed_date.strftime('%Y-%m-%d') if dd.removed_date else 'Active'
    print(f"\n12/29/2025: Found")
    print(f"  Type: {dd.type}")
    print(f"  Loss %: {dd.loss_percent:.2f}%")
    print(f"  Removed: {removed_str}")
    if dd.removed_date:
        trading_days = analyzer._count_trading_days(dd.date, dd.removed_date)
        print(f"  Trading days to removal: {trading_days}")
else:
    print(f"\n12/29/2025: NOT found in our data")

print("\n" + "=" * 100)
