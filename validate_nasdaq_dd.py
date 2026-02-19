#!/usr/bin/env python3
"""Validate NASDAQ distribution days against IBD data"""

from market_school_rules import MarketSchoolRules
import pandas as pd
from datetime import datetime

print("=" * 80)
print("NASDAQ DISTRIBUTION DAY VALIDATION")
print("=" * 80)

# Fetch and analyze NASDAQ (need full lookback)
analyzer = MarketSchoolRules("^IXIC")
analyzer.fetch_data(start_date="2024-02-24", end_date="2026-02-18")
analyzer.analyze_market()

print(f"\nFetched {len(analyzer.data)} days of data")
print(f"Generated {len(analyzer.signals)} signals")
print(f"Total distribution days: {len(analyzer.distribution_days)}")

# Expected changes from user
expected_changes = {
    '2026-02-10': {'added': True, 'removed': []},
    '2026-02-06': {'added': True, 'removed': ['2025-12-30']},
    '2026-02-04': {'added': True, 'removed': ['2025-12-29']},
    '2026-02-03': {'added': True, 'removed': []},
    '2026-01-30': {'added': True, 'removed': []},
    '2026-01-29': {'added': True, 'removed': []},
    '2026-01-28': {'added': True, 'removed': []},  # Stall day
    '2026-01-26': {'added': False, 'removed': ['?']},  # Unknown what fell off
    '2026-01-22': {'added': False, 'removed': []},  # Should NOT have removal
    '2026-01-21': {'added': False, 'removed': ['2025-12-12']},
    '2026-01-20': {'added': False, 'removed': ['2025-12-11']},
    '2026-01-14': {'added': True, 'removed': []},
    '2026-01-13': {'added': True, 'removed': ['2025-12-05']},
}

# Check what actually happened
print("\n" + "=" * 80)
print("VALIDATION REPORT")
print("=" * 80)

for date_str, expected in sorted(expected_changes.items()):
    date = pd.Timestamp(date_str).normalize()

    # Find additions
    additions = [dd for dd in analyzer.distribution_days
                 if pd.Timestamp(dd.date).normalize() == date]

    # Find removals
    removals = [dd for dd in analyzer.distribution_days
                if dd.removed_date and pd.Timestamp(dd.removed_date).normalize() == date]

    print(f"\n📅 {date_str}:")
    print(f"   Expected: Add={expected['added']}, Remove={expected['removed']}")

    # Check additions
    if expected['added']:
        if additions:
            add = additions[0]
            status = "✅" if len(additions) == 1 else "⚠️"
            print(f"   {status} Added: {add.type} (-{add.loss_percent:.2f}%)")
        else:
            print(f"   ❌ MISSING: Should have added distribution")
    else:
        if additions:
            print(f"   ❌ FALSE POSITIVE: Should NOT have added distribution")
            for add in additions:
                print(f"      - {add.type} (-{add.loss_percent:.2f}%)")

    # Check removals
    if expected['removed']:
        expected_dates = [d for d in expected['removed'] if d != '?']

        if removals:
            for rem in removals:
                rem_date = pd.Timestamp(rem.date).strftime('%Y-%m-%d')
                if rem_date in expected_dates or '?' in expected['removed']:
                    print(f"   ✅ Removed: {rem_date} ({rem.removal_reason})")
                else:
                    print(f"   ❌ WRONG REMOVAL: {rem_date} (expected {expected['removed']})")
                    print(f"      Reason: {rem.removal_reason}")
        else:
            if '?' not in expected['removed']:
                print(f"   ❌ MISSING REMOVAL: Should have removed {expected['removed']}")
    else:
        if removals:
            print(f"   ❌ FALSE REMOVAL: Should NOT have removed anything")
            for rem in removals:
                rem_date = pd.Timestamp(rem.date).strftime('%Y-%m-%d')
                print(f"      - Removed {rem_date} ({rem.removal_reason})")

# Summary of all distribution days
print("\n" + "=" * 80)
print("ALL DISTRIBUTION DAYS (Dec 2025 - Feb 2026)")
print("=" * 80)

relevant_dds = [dd for dd in analyzer.distribution_days
                if pd.Timestamp(dd.date).tz_localize(None) >= pd.Timestamp('2025-12-01')]

print(f"\n{'Added Date':<12} {'Type':<12} {'Loss %':<8} {'Removed Date':<12} {'Reason':<30}")
print("-" * 80)

for dd in sorted(relevant_dds, key=lambda x: x.date):
    added = pd.Timestamp(dd.date).strftime('%Y-%m-%d')
    removed = pd.Timestamp(dd.removed_date).strftime('%Y-%m-%d') if dd.removed_date else 'Active'
    reason = dd.removal_reason or '-'

    print(f"{added:<12} {dd.type:<12} {dd.loss_percent:<7.2f}% {removed:<12} {reason:<30}")

print("\n" + "=" * 80)
