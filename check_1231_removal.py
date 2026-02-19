#!/usr/bin/env python3
"""Check when 12/31 should be removed"""

from market_school_rules import MarketSchoolRules
import pandas as pd

print("=" * 80)
print("12/31 SPY DISTRIBUTION - REMOVAL CHECK")
print("=" * 80)

analyzer = MarketSchoolRules("SPY")
analyzer.fetch_data(start_date="2025-12-01", end_date="2026-02-18")
analyzer.analyze_market()

# Find 12/31
dec31 = [dd for dd in analyzer.distribution_days if pd.Timestamp(dd.date).strftime('%Y-%m-%d') == '2025-12-31']

if dec31:
    dd = dec31[0]
    removed_str = dd.removed_date.strftime('%Y-%m-%d') if dd.removed_date else 'Active'

    print(f"\n12/31/2025 Distribution:")
    print(f"  Type: {dd.type}")
    print(f"  Loss %: {dd.loss_percent:.2f}%")
    print(f"  Removed: {removed_str}")
    print(f"  Removal reason: {dd.removal_reason if dd.removal_reason else 'N/A'}")

    if dd.removed_date:
        trading_days = analyzer._count_trading_days(dd.date, dd.removed_date)
        print(f"  Trading days to removal: {trading_days}")

        # Check what today is
        print(f"\nToday is: 2026-02-18")
        print(f"Removed on: {removed_str}")
        print(f"Status: {'❌ EXPIRED' if dd.removed_date else '✅ ACTIVE'}")

        if removed_str == '2026-02-06':
            print(f"\nNote: Removed on 2/6, which is 12 days ago from today (2/18)")
            print(f"IBD may still show it as active if they count differently or haven't updated")
    else:
        print(f"\n✅ Currently ACTIVE")
else:
    print(f"\n❌ 12/31/2025 NOT found in our data")

print("\n" + "=" * 80)
