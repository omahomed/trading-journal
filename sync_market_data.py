#!/usr/bin/env python3
"""Sync market signals data (same as clicking Sync button in app)"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import required modules
from market_school_rules import MarketSchoolRules
import db_layer as db

print("=" * 80)
print("SYNCING MARKET SIGNALS DATA")
print("=" * 80)

def analyze_symbol(symbol, start_date, end_date):
    """Analyze market signals for a symbol."""
    print(f"\n📊 Analyzing {symbol}...")

    analyzer = MarketSchoolRules(symbol)
    analyzer.fetch_data(start_date=start_date, end_date=end_date)

    if analyzer.data is None or analyzer.data.empty:
        print(f"  ❌ No data fetched")
        return []

    print(f"  ✓ Fetched {len(analyzer.data)} days of data")

    analyzer.analyze_market()
    print(f"  ✓ Generated {len(analyzer.signals)} signals")

    summaries = []
    dates_to_process = analyzer.data.index[260:]  # Skip 260-day lookback
    print(f"  ✓ Processing {len(dates_to_process)} days (after 260-day lookback)")

    for date in dates_to_process:
        date_str = date.strftime('%Y-%m-%d')
        summary = analyzer.get_daily_summary(date_str)

        # Parse signals for this date
        date_normalized = pd.Timestamp(date).normalize()
        day_signals = [s for s in analyzer.signals if pd.Timestamp(s.date).normalize() == date_normalized]
        buy_sigs = [s.signal_type.name for s in day_signals if s.signal_type.name.startswith('B')]
        sell_sigs = [s.signal_type.name for s in day_signals if s.signal_type.name.startswith('S')]

        summary['buy_signals'] = ','.join(buy_sigs) if buy_sigs else None
        summary['sell_signals'] = ','.join(sell_sigs) if sell_sigs else None
        summary['symbol'] = symbol
        summaries.append(summary)

    return summaries

def sync_signals_to_db(symbol, summaries, filter_from_date=None):
    """Store analysis results in database."""
    saved_count = 0
    filtered_summaries = [s for s in summaries
                          if not filter_from_date or str(s['date']) >= str(filter_from_date.date())]

    print(f"  💾 Saving {len(filtered_summaries)} records...")

    for summary in filtered_summaries:
        signal_dict = {
            'symbol': symbol,
            'signal_date': summary['date'],
            'close_price': float(summary['close']),
            'daily_change_pct': float(summary['daily_change'].rstrip('%')),
            'market_exposure': int(summary['market_exposure']),
            'position_allocation': float(summary['position_allocation'].rstrip('%')) / 100,
            'buy_switch': summary['buy_switch'] == 'ON',
            'distribution_count': int(summary['distribution_count']),
            'above_21ema': bool(summary['above_21ema']),
            'above_50ma': bool(summary['above_50ma']),
            'buy_signals': summary.get('buy_signals'),
            'sell_signals': summary.get('sell_signals')
        }

        try:
            db.save_market_signal(signal_dict)
            saved_count += 1
        except Exception as e:
            print(f"    ⚠️ Failed to save {symbol} {summary['date']}: {e}")

    print(f"  ✅ Saved {saved_count} records")
    return saved_count

# Main sync logic
try:
    end_date = datetime.now().strftime('%Y-%m-%d')

    # Check what dates we already have
    nasdaq_latest_date = db.get_latest_signal_date("^IXIC")
    spy_latest_date = db.get_latest_signal_date("SPY")

    # Determine fetch strategy
    if nasdaq_latest_date is None:
        nasdaq_fetch_start = "2024-02-24"
        nasdaq_save_from = pd.Timestamp("2025-02-24")
        print("\n📥 Initial Nasdaq sync from Feb 24, 2025")
    else:
        nasdaq_fetch_start = (nasdaq_latest_date - timedelta(days=30)).strftime('%Y-%m-%d')
        nasdaq_save_from = pd.Timestamp(nasdaq_latest_date) + timedelta(days=1)
        print(f"\n🔄 Updating Nasdaq from {nasdaq_save_from.date()}")

    if spy_latest_date is None:
        spy_fetch_start = "2024-02-24"
        spy_save_from = pd.Timestamp("2025-02-24")
        print("📥 Initial SPY sync from Feb 24, 2025")
    else:
        spy_fetch_start = (spy_latest_date - timedelta(days=30)).strftime('%Y-%m-%d')
        spy_save_from = pd.Timestamp(spy_latest_date) + timedelta(days=1)
        print(f"🔄 Updating SPY from {spy_save_from.date()}")

    # Sync Nasdaq
    nasdaq_summaries = analyze_symbol("^IXIC", nasdaq_fetch_start, end_date)
    nasdaq_saved = sync_signals_to_db("^IXIC", nasdaq_summaries, filter_from_date=nasdaq_save_from)

    # Sync SPY
    spy_summaries = analyze_symbol("SPY", spy_fetch_start, end_date)
    spy_saved = sync_signals_to_db("SPY", spy_summaries, filter_from_date=spy_save_from)

    print("\n" + "=" * 80)
    print(f"🎉 Sync complete! Total: {nasdaq_saved + spy_saved} new records")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ Sync failed: {str(e)}")
    import traceback
    traceback.print_exc()
