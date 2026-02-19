#!/usr/bin/env python3
"""Check SPY distribution days"""

from market_school_rules import MarketSchoolRules
import pandas as pd

print("=" * 80)
print("SPY DISTRIBUTION DAY CHECK")
print("=" * 80)

analyzer = MarketSchoolRules("SPY")
analyzer.fetch_data(start_date="2024-11-01", end_date="2026-02-18")
analyzer.analyze_market()

# Get active distribution days
active_dds = [dd for dd in analyzer.distribution_days if dd.removed_date is None]

print(f"\nTotal Active Distribution Days: {len(active_dds)}")
print(f"IBD Expected: 5")
print(f"Difference: {len(active_dds) - 5}")

print("\n" + "=" * 80)
print("ACTIVE DISTRIBUTION DAYS:")
print("=" * 80)

print(f"\n{'Date':<15} {'Type':<15} {'Loss %':<10} {'Close':<12}")
print("-" * 60)
for dd in sorted(active_dds, key=lambda x: x.date):
    print(f"{dd.date.strftime('%Y-%m-%d'):<15} {dd.type:<15} {dd.loss_percent:>8.2f}% ${dd.close:>10.2f}")

print("\n" + "=" * 80)
print("ALL DISTRIBUTION DAYS (Dec 2025 - Feb 2026):")
print("=" * 80)

recent_dds = [
    dd for dd in analyzer.distribution_days
    if pd.Timestamp(dd.date).year >= 2025 and pd.Timestamp(dd.date).month >= 12
]

print(f"\n{'Added Date':<15} {'Type':<15} {'Loss %':<10} {'Removed Date':<15} {'Reason':<20}")
print("-" * 80)
for dd in sorted(recent_dds, key=lambda x: x.date):
    removed = dd.removed_date.strftime('%Y-%m-%d') if dd.removed_date else 'Active'
    reason = dd.removal_reason if dd.removal_reason else '-'
    print(f"{dd.date.strftime('%Y-%m-%d'):<15} {dd.type:<15} {dd.loss_percent:>8.2f}% {removed:<15} {reason:<20}")

print("\n" + "=" * 80)
