#!/usr/bin/env python3
"""Check SPY data for 12/26 and 12/29 to see why they weren't detected"""

from market_school_rules import MarketSchoolRules
import pandas as pd

print("=" * 100)
print("SPY - CHECKING 12/26 AND 12/29")
print("=" * 100)

analyzer = MarketSchoolRules("SPY")
analyzer.fetch_data(start_date="2025-12-20", end_date="2026-01-10")

check_dates = ['2025-12-26', '2025-12-29']

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

            print(f"\nDISTRIBUTION CHECK:")
            print(f"  Daily gain <= -0.2%? {current['daily_gain_pct'] <= -0.2} ({current['daily_gain_pct']:.2f}%)")
            print(f"  Volume up? {current['volume_up']}")
            print(f"  Actual volume higher?: {current['Volume'] > prev['Volume']}")
            print(f"  Volume ratio: {(current['Volume'] / prev['Volume']):.4f}")

            is_distribution = (current['daily_gain_pct'] <= -0.2 and current['volume_up'])
            print(f"  → Qualifies as distribution? {is_distribution}")

            if not is_distribution:
                print(f"\n  ❌ NOT A DISTRIBUTION in our data:")
                if current['daily_gain_pct'] > -0.2:
                    print(f"     - Daily gain too small ({current['daily_gain_pct']:.2f}% > -0.2%)")
                if not current['volume_up']:
                    print(f"     - Volume not higher than previous day")

            print(f"\n  ⚠️  IBD SAYS: This WAS a distribution day")

    else:
        day_of_week = pd.Timestamp(date_str).strftime('%A')
        print(f"❌ {date_str} ({day_of_week}) - NOT in trading data (weekend/holiday)")

print("\n" + "=" * 100)
