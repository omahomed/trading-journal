#!/usr/bin/env python3
"""Check SPY volume for the false positive distribution days"""

from market_school_rules import MarketSchoolRules
import pandas as pd

print("=" * 100)
print("SPY VOLUME INVESTIGATION - False Positives")
print("=" * 100)

analyzer = MarketSchoolRules("SPY")
analyzer.fetch_data(start_date="2026-01-01", end_date="2026-02-18")

# Check 1/30 and 2/5
check_dates = ['2026-01-30', '2026-02-05']

for date_str in check_dates:
    print(f"\n{'=' * 100}")
    print(f"CHECKING: {date_str}")
    print(f"{'=' * 100}")

    date = pd.Timestamp(date_str)
    data_dates_normalized = [pd.Timestamp(d).tz_localize(None).normalize() for d in analyzer.data.index]
    date_normalized = date.normalize()

    if date_normalized in data_dates_normalized:
        idx = data_dates_normalized.index(date_normalized)
        current = analyzer.data.iloc[idx]
        prev = analyzer.data.iloc[idx-1] if idx > 0 else None

        print(f"\nRAW DATA:")
        print(f"  Date: {current.name.strftime('%Y-%m-%d %A')}")
        print(f"  Open: ${current['Open']:.2f}")
        print(f"  High: ${current['High']:.2f}")
        print(f"  Low: ${current['Low']:.2f}")
        print(f"  Close: ${current['Close']:.2f}")
        print(f"  Volume: {current['Volume']:,.0f}")
        print(f"  Daily %: {current['daily_gain_pct']:.2f}%")

        if prev is not None:
            print(f"\nPREVIOUS DAY ({prev.name.strftime('%Y-%m-%d %A')}):")
            print(f"  Close: ${prev['Close']:.2f}")
            print(f"  Volume: {prev['Volume']:,.0f}")

            print(f"\nVOLUME COMPARISON:")
            print(f"  Current Volume: {current['Volume']:,.0f}")
            print(f"  Previous Volume: {prev['Volume']:,.0f}")
            print(f"  volume_up flag: {current['volume_up']}")
            print(f"  Actual volume higher?: {current['Volume'] > prev['Volume']}")
            print(f"  Volume Ratio: {(current['Volume'] / prev['Volume']):.4f}")

            # Check if this qualifies as distribution
            is_distribution = (current['daily_gain_pct'] <= -0.2 and current['volume_up'])

            print(f"\nDISTRIBUTION CHECK:")
            print(f"  Daily gain <= -0.2%? {current['daily_gain_pct'] <= -0.2} ({current['daily_gain_pct']:.2f}%)")
            print(f"  Volume up (from data)? {current['volume_up']}")
            print(f"  → Qualifies as distribution? {is_distribution}")

            print(f"\n⚠️  IBD SAYS: Volume was LOWER - NOT a distribution day")

            if is_distribution and current['Volume'] <= prev['Volume']:
                print(f"❌ BUG FOUND: volume_up=True but actual volume is NOT higher!")
            elif is_distribution and current['Volume'] > prev['Volume']:
                print(f"⚠️  DATA MISMATCH: yfinance shows volume UP, IBD shows volume DOWN")
    else:
        print(f"❌ Date not found in data")

print("\n" + "=" * 100)
