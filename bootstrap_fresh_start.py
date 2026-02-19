#!/usr/bin/env python3
"""Bootstrap fresh start from 2/18/2026 with known IBD state"""

import sys
import os
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))

from market_school_rules import MarketSchoolRules
import db_layer as db

print("=" * 80)
print("BOOTSTRAPPING FRESH START - 2/18/2026")
print("=" * 80)

# Known state from IBD as of 2/18/2026
NASDAQ_STATE = {
    'symbol': '^IXIC',
    'distribution_days': [
        '2026-01-14',  # distribution -1.00%
        '2026-01-28',  # stall +0.17%
        '2026-01-29',  # distribution -0.72%
        '2026-01-30',  # distribution -0.94%
        '2026-02-03',  # distribution -1.43%
        '2026-02-04',  # distribution -1.51%
        '2026-02-10',  # distribution -0.59%
    ],
    'distribution_count': 7,
    'buy_switch': False,
    'market_exposure': 0
}

SPY_STATE = {
    'symbol': 'SPY',
    'distribution_days': [
        '2026-01-14',  # -0.49%
        '2026-01-20',  # -2.04%
        '2026-01-30',  # -0.30%
        '2026-02-03',  # -0.85%
        '2026-02-12',  # -1.54%
    ],
    'distribution_count': 5,
    'buy_switch': False,
    'market_exposure': 0
}

def bootstrap_symbol(state):
    """Create initial record for a symbol."""
    symbol = state['symbol']

    print(f"\n📊 Bootstrapping {symbol}...")

    # Fetch just recent data to get today's price
    analyzer = MarketSchoolRules(symbol)
    analyzer.fetch_data(start_date='2026-02-10', end_date='2026-02-18')

    # Get latest available data
    latest_date = analyzer.data.index[-1]
    latest = analyzer.data.iloc[-1]
    prev = analyzer.data.iloc[-2]

    latest_date_str = latest_date.strftime('%Y-%m-%d')
    daily_change_pct = ((latest['Close'] - prev['Close']) / prev['Close']) * 100

    print(f"  Using latest available data: {latest_date_str}")

    # Create the bootstrap record
    signal_dict = {
        'symbol': symbol,
        'signal_date': latest_date_str,
        'close_price': float(latest['Close']),
        'daily_change_pct': float(daily_change_pct),
        'market_exposure': state['market_exposure'],
        'position_allocation': 0.0,
        'buy_switch': state['buy_switch'],
        'distribution_count': state['distribution_count'],
        'above_21ema': bool(latest['Close'] > latest['ema21']),
        'above_50ma': bool(latest['Close'] > latest['sma50']),
        'buy_signals': None,
        'sell_signals': None
    }

    print(f"  Close: ${signal_dict['close_price']:.2f}")
    print(f"  Daily %: {signal_dict['daily_change_pct']:.2f}%")
    print(f"  Distribution Count: {state['distribution_count']}")
    print(f"  Buy Switch: {'ON' if state['buy_switch'] else 'OFF'}")
    print(f"  Exposure: {state['market_exposure']}")

    # Save to database
    db.save_market_signal(signal_dict)
    print(f"  ✅ Saved bootstrap record")

    return signal_dict

# Clear existing data
print("\n🗑️  Clearing all historical data...")
with db.get_db_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM market_signals")
        conn.commit()
        print("  ✅ Database cleared")

# Bootstrap both symbols
bootstrap_symbol(NASDAQ_STATE)
bootstrap_symbol(SPY_STATE)

print("\n" + "=" * 80)
print("✅ BOOTSTRAP COMPLETE!")
print("=" * 80)
print()
print("Starting State (2/18/2026):")
print(f"  NASDAQ: {NASDAQ_STATE['distribution_count']} distributions, Exposure {NASDAQ_STATE['market_exposure']}")
print(f"  SPY:    {SPY_STATE['distribution_count']} distributions, Exposure {SPY_STATE['market_exposure']}")
print()
print("From 2/19 forward, the system will track automatically.")
print("Report any discrepancies with IBD for troubleshooting.")
print("=" * 80)
