#!/usr/bin/env python3
"""Check yfinance data for NASDAQ to compare with IBD"""

from market_school_rules import MarketSchoolRules
import pandas as pd

print("=" * 80)
print("YFINANCE DATA FOR NASDAQ")
print("=" * 80)

analyzer = MarketSchoolRules("^IXIC")
analyzer.fetch_data(start_date="2026-01-01", end_date="2026-02-18")

# Key dates to check
check_dates = [
    '2026-01-13', '2026-01-14', '2026-01-20', '2026-01-21',
    '2026-01-22', '2026-01-26', '2026-01-28', '2026-01-29',
    '2026-01-30', '2026-02-03', '2026-02-04', '2026-02-06',
    '2026-02-10'
]

print("\nRAW OHLCV DATA FROM YFINANCE:")
print(f"{'Date':<12} {'Open':<12} {'High':<12} {'Low':<12} {'Close':<12} {'Volume':<15} {'Daily %':<10}")
print("-" * 100)

for date_str in check_dates:
    date = pd.Timestamp(date_str)

    # Find the date in data
    data_dates_normalized = [pd.Timestamp(d).tz_localize(None).normalize() for d in analyzer.data.index]
    date_normalized = pd.Timestamp(date).normalize()

    if date_normalized in data_dates_normalized:
        idx = data_dates_normalized.index(date_normalized)
        row = analyzer.data.iloc[idx]

        # Calculate daily % manually
        if idx > 0:
            prev_close = analyzer.data.iloc[idx-1]['Close']
            daily_pct = ((row['Close'] - prev_close) / prev_close) * 100
        else:
            daily_pct = 0

        print(f"{date_str:<12} ${row['Open']:<11.2f} ${row['High']:<11.2f} ${row['Low']:<11.2f} ${row['Close']:<11.2f} {row['Volume']:<15,.0f} {daily_pct:>9.2f}%")
    else:
        print(f"{date_str:<12} NOT FOUND (weekend/holiday)")

print("\n" + "=" * 80)
print("NOTES:")
print("- Daily % = (Close - PrevClose) / PrevClose * 100")
print("- Distribution threshold: ≤ -0.2% AND volume up")
print("- yfinance data may differ from IBD's data source")
print("=" * 80)
