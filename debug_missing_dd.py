#!/usr/bin/env python3
"""Debug why specific distribution days aren't being detected"""

from market_school_rules import MarketSchoolRules
import pandas as pd

print("=" * 80)
print("DEBUGGING MISSING DISTRIBUTION DAYS")
print("=" * 80)

analyzer = MarketSchoolRules("^IXIC")
analyzer.fetch_data(start_date="2024-11-01", end_date="2026-02-18")
analyzer.analyze_market()

# Check specific dates
missing_dates = {
    '2025-12-05': 'Stall day',
    '2026-01-28': 'Stall day',
}

for date_str, expected_type in missing_dates.items():
    print(f"\n{'=' * 80}")
    print(f"CHECKING: {date_str} (Expected: {expected_type})")
    print(f"{'=' * 80}")

    date = pd.Timestamp(date_str)

    # Find the date in data (handle timezone)
    data_dates_normalized = [pd.Timestamp(d).tz_localize(None).normalize() for d in analyzer.data.index]
    date_normalized = pd.Timestamp(date).normalize()

    if date_normalized not in data_dates_normalized:
        print(f"❌ Date not found in data!")
        print(f"   Looking for: {date_normalized}")
        print(f"   Available dates around that time:")
        for d in analyzer.data.index:
            d_norm = pd.Timestamp(d).tz_localize(None)
            if abs((d_norm - date_normalized).days) <= 2:
                print(f"     {d_norm.strftime('%Y-%m-%d %A')}")
        continue

    idx = data_dates_normalized.index(date_normalized)
    date = analyzer.data.index[idx]

    idx = analyzer.data.index.get_loc(date)
    current = analyzer.data.iloc[idx]
    prev = analyzer.data.iloc[idx-1] if idx > 0 else None

    print(f"\nRAW DATA:")
    print(f"  Date: {date}")
    print(f"  Open: ${current['Open']:.2f}")
    print(f"  High: ${current['High']:.2f}")
    print(f"  Low: ${current['Low']:.2f}")
    print(f"  Close: ${current['Close']:.2f}")
    print(f"  Volume: {current['Volume']:,.0f}")
    print(f"  Daily Gain %: {current['daily_gain_pct']:.2f}%")

    if prev is not None:
        print(f"\nPREVIOUS DAY:")
        print(f"  Close: ${prev['Close']:.2f}")
        print(f"  Volume: {prev['Volume']:,.0f}")

    print(f"\nCALCULATED VALUES:")
    print(f"  volume_up: {current['volume_up']}")
    print(f"  close_position: {current['close_position']:.3f}")
    print(f"  give_back: {current['give_back']:.2f}%")
    print(f"  intraday_gain: {current['intraday_gain']:.2f}%")

    print(f"\nDISTRIBUTION DAY CHECK:")
    is_distribution = (current['daily_gain_pct'] <= -0.2 and current['volume_up'])
    print(f"  Daily gain <= -0.2%: {current['daily_gain_pct'] <= -0.2} ({current['daily_gain_pct']:.2f}%)")
    print(f"  Volume up: {current['volume_up']}")
    print(f"  → Is distribution: {is_distribution}")

    print(f"\nSTALL DAY CHECK:")
    print(f"  Volume up: {current['volume_up']}")
    print(f"  Close position < 0.5: {current['close_position'] < 0.5} ({current['close_position']:.3f})")
    print(f"  Give back > 30%: {current['give_back'] > 30} ({current['give_back']:.2f}%)")
    print(f"  Intraday gain > 30%: {current['intraday_gain'] > 30} ({current['intraday_gain']:.2f}%)")

    is_stall = (current['volume_up'] and
               current['close_position'] < 0.5 and
               current['give_back'] > 30 and
               current['intraday_gain'] > 30)
    print(f"  → Is stall: {is_stall}")

    # Check if it was actually added
    added = [dd for dd in analyzer.distribution_days
             if pd.Timestamp(dd.date).normalize() == date.normalize()]

    print(f"\nACTUAL RESULT:")
    if added:
        dd = added[0]
        print(f"  ✅ Found as {dd.type}: {dd.loss_percent:.2f}%")
    else:
        print(f"  ❌ NOT DETECTED")

        # If expected stall but not detected, check what's wrong
        if expected_type == 'Stall day':
            print(f"\n  STALL CRITERIA FAILURES:")
            if not current['volume_up']:
                print(f"    - Volume NOT higher than previous day")
                print(f"      Current: {current['Volume']:,.0f}, Prev: {prev['Volume']:,.0f}")
            if not current['close_position'] < 0.5:
                print(f"    - Close position NOT in lower half ({current['close_position']:.3f})")
            if not current['give_back'] > 30:
                print(f"    - Give back NOT > 30% ({current['give_back']:.2f}%)")
            if not current['intraday_gain'] > 30:
                print(f"    - Intraday gain NOT > 30% ({current['intraday_gain']:.2f}%)")

print("\n" + "=" * 80)
