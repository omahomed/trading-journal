import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime
import random

# === Setup ===
warnings.simplefilter(action="ignore", category=FutureWarning)
os.makedirs("output", exist_ok=True)

# === User Configuration (SAME AS MAIN SCRIPT) ===
INITIAL_CAPITAL = 150000
MAX_LEVERAGE = 1.8
POSITION_SIZE_PCT = 0.10
DATA_START_DATE = "2023-09-01"
PORTFOLIO_START_DATE = "2024-09-04"
END_DATE = "2025-09-05"

# === DIAGNOSTIC MODE - Track specific ticker ===
WATCH_TICKER = "DOCS"  # Change this to track a different ticker
TRACE_ALL = False  # Set to True to trace everything (verbose!)

print(f"🔍 SIMULATION TRACER - Watching: {WATCH_TICKER}")
print(f"="*60)

# Load tickers
with open("select_ticker.txt", "r") as f:
    tickers = [line.strip().upper() for line in f if line.strip()]

# === Load and prepare all ticker data (same as original) ===
ticker_data = {}
all_dates = set()

for ticker in tickers:
    input_file = f"output/{ticker}_price_data.csv"
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        continue
    
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df = df[(df.index >= DATA_START_DATE) & (df.index <= END_DATE)]
    
    if df.empty:
        continue
    
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    
    # Calculate EMA21
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    
    # Generate signals (same logic as original)
    df["signal"] = 0
    position_open = False
    first_break_low = None
    
    for i in range(2, len(df)):
        if not position_open and i >= 2:
            three_days_above = (df.iloc[i]["Low"] > df.iloc[i]["EMA21"] and 
                               df.iloc[i-1]["Low"] > df.iloc[i-1]["EMA21"] and 
                               df.iloc[i-2]["Low"] > df.iloc[i-2]["EMA21"])
            is_up_day = df.iloc[i]["Close"] > df.iloc[i-1]["Close"]
            
            if three_days_above and is_up_day:
                df.at[df.index[i], "signal"] = 1
                position_open = True
                first_break_low = None
        
        elif position_open:
            if first_break_low is None:
                if df.iloc[i]["Close"] < df.iloc[i]["EMA21"]:
                    if i > 0 and df.iloc[i-1]["Close"] >= df.iloc[i-1]["EMA21"]:
                        first_break_low = df.iloc[i]["Low"]
            
            if first_break_low is not None:
                if df.iloc[i]["Low"] < first_break_low:
                    df.at[df.index[i], "signal"] = -1
                    position_open = False
                    first_break_low = None
    
    ticker_data[ticker] = df
    all_dates.update(df.index)

# Sort dates
all_dates = sorted(list(all_dates))
portfolio_dates = [d for d in all_dates if d >= pd.to_datetime(PORTFOLIO_START_DATE)]

# Check for pre-existing positions
pre_existing_positions = set()
for ticker in ticker_data:
    df = ticker_data[ticker]
    position_open = False
    
    for date in df.index:
        if date >= pd.to_datetime(PORTFOLIO_START_DATE):
            break
        if df.loc[date, "signal"] == 1:
            position_open = True
        elif df.loc[date, "signal"] == -1:
            position_open = False
    
    if position_open:
        pre_existing_positions.add(ticker)

if WATCH_TICKER in pre_existing_positions:
    print(f"⚠️ {WATCH_TICKER} is a pre-existing position")

# === TRACED Portfolio Class ===
class TracedPortfolio:
    def __init__(self, initial_capital, max_leverage, position_size_pct):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_leverage = max_leverage
        self.position_size_pct = position_size_pct
        self.positions = {}
        self.trade_history = []
        self.pre_existing = pre_existing_positions.copy()
        self.trace_log = []
        
    def log(self, message):
        self.trace_log.append(message)
        if TRACE_ALL:
            print(message)
        
    def get_nlv(self, date, ticker_data):
        nlv = self.cash
        for ticker, pos in self.positions.items():
            if ticker in ticker_data and date in ticker_data[ticker].index:
                current_price = ticker_data[ticker].loc[date, "Close"]
                nlv += pos['shares'] * current_price
        return nlv
    
    def get_buying_power(self, nlv):
        max_investment = nlv * self.max_leverage
        current_investment = sum(pos['shares'] * pos['entry_price'] for pos in self.positions.values())
        return max_investment - current_investment
    
    def can_buy(self, nlv):
        position_value = nlv * self.position_size_pct
        buying_power = self.get_buying_power(nlv)
        return buying_power >= position_value
    
    def buy(self, ticker, price, date, nlv):
        if ticker in self.positions:
            return False
        
        if ticker in self.pre_existing:
            return False
        
        position_value = nlv * self.position_size_pct
        buying_power = self.get_buying_power(nlv)
        
        if buying_power < position_value:
            return False
        
        shares = int(position_value / price)
        
        # TRACE WATCH TICKER
        if ticker == WATCH_TICKER:
            print(f"\n🎯 {date.date()} ATTEMPTING BUY {WATCH_TICKER}:")
            print(f"   NLV: ${nlv:,.2f}")
            print(f"   Position size (10% of NLV): ${position_value:,.2f}")
            print(f"   Price: ${price:.2f}")
            print(f"   Calculated shares: {shares}")
            print(f"   Actual cost: ${shares * price:,.2f}")
            print(f"   Cash available: ${self.cash:,.2f}")
            print(f"   Buying power: ${buying_power:,.2f}")
        
        if shares < 1:
            if ticker == WATCH_TICKER:
                print(f"   ❌ Not enough for 1 share")
            return False
        
        actual_cost = shares * price
        
        if actual_cost > self.cash:
            margin_used = actual_cost - self.cash
            self.cash = 0
        else:
            margin_used = 0
            self.cash -= actual_cost
        
        self.positions[ticker] = {
            'shares': shares,
            'entry_price': price,
            'entry_date': date,
            'margin_used': margin_used
        }
        
        self.trade_history.append({
            'date': date,
            'ticker': ticker,
            'action': 'BUY',
            'shares': shares,
            'price': price,
            'value': actual_cost,
            'margin_used': margin_used,
            'nlv': nlv
        })
        
        if ticker == WATCH_TICKER:
            print(f"   ✅ BOUGHT {shares} shares")
            print(f"   Cash after: ${self.cash:,.2f}")
            print(f"   Margin used: ${margin_used:,.2f}")
        
        return True
    
    def sell(self, ticker, price, date, nlv):
        if ticker in self.pre_existing:
            self.pre_existing.remove(ticker)
            if ticker == WATCH_TICKER:
                print(f"\n📉 {date.date()} {WATCH_TICKER} pre-existing position cleared")
            return False
            
        if ticker not in self.positions:
            return False
        
        pos = self.positions[ticker]
        proceeds = pos['shares'] * price
        
        if ticker == WATCH_TICKER:
            print(f"\n🎯 {date.date()} SELLING {WATCH_TICKER}:")
            print(f"   Shares: {pos['shares']}")
            print(f"   Entry price: ${pos['entry_price']:.2f}")
            print(f"   Exit price: ${price:.2f}")
            print(f"   Proceeds: ${proceeds:,.2f}")
            print(f"   P&L: ${proceeds - (pos['shares'] * pos['entry_price']):,.2f}")
        
        if pos['margin_used'] > 0:
            self.cash += proceeds - pos['margin_used']
        else:
            self.cash += proceeds
        
        self.trade_history.append({
            'date': date,
            'ticker': ticker,
            'action': 'SELL',
            'shares': pos['shares'],
            'price': price,
            'value': proceeds,
            'pnl': proceeds - (pos['shares'] * pos['entry_price']),
            'nlv': nlv
        })
        
        del self.positions[ticker]
        
        if ticker == WATCH_TICKER:
            print(f"   ✅ SOLD - Cash after: ${self.cash:,.2f}")
        
        return True

# Initialize traced portfolio
portfolio = TracedPortfolio(INITIAL_CAPITAL, MAX_LEVERAGE, POSITION_SIZE_PCT)

# === Run simulation with tracing ===
print(f"\n🔄 Running traced simulation from {PORTFOLIO_START_DATE} to {END_DATE}")
print(f"📊 Tracking {WATCH_TICKER} specifically")
print("="*60)

trade_count = 0
watch_signals = []
last_nlv = INITIAL_CAPITAL
nlv_history = []

for date in portfolio_dates:
    nlv = portfolio.get_nlv(date, ticker_data)
    
    # TRACK NLV CHANGES
    nlv_change = nlv - last_nlv
    nlv_change_pct = (nlv_change / last_nlv * 100) if last_nlv != 0 else 0
    
    # Alert on huge NLV jumps
    if abs(nlv_change_pct) > 50:  # More than 50% change in one day
        print(f"\n🚨 HUGE NLV JUMP on {date.date()}:")
        print(f"   Previous NLV: ${last_nlv:,.2f}")
        print(f"   Current NLV: ${nlv:,.2f}")
        print(f"   Change: ${nlv_change:,.2f} ({nlv_change_pct:+.1f}%)")
        print(f"   Cash: ${portfolio.cash:,.2f}")
        print(f"   Positions: {len(portfolio.positions)}")
        if portfolio.positions:
            for t, p in list(portfolio.positions.items())[:5]:  # Show first 5
                if t in ticker_data and date in ticker_data[t].index:
                    price = ticker_data[t].loc[date, "Close"]
                    value = p['shares'] * price
                    print(f"     {t}: {p['shares']} shares @ ${price:.2f} = ${value:,.2f}")
    
    nlv_history.append({'date': date, 'nlv': nlv, 'cash': portfolio.cash})
    
    # Track signals for watch ticker
    if WATCH_TICKER in ticker_data and date in ticker_data[WATCH_TICKER].index:
        signal = ticker_data[WATCH_TICKER].loc[date, "signal"]
        if signal != 0:
            watch_signals.append({
                'date': date,
                'signal': 'BUY' if signal == 1 else 'SELL',
                'price': ticker_data[WATCH_TICKER].loc[date, "Close"]
            })
    
    # Check for sell signals first
    for ticker in list(portfolio.positions.keys()):
        if ticker in ticker_data and date in ticker_data[ticker].index:
            if ticker_data[ticker].loc[date, "signal"] == -1:
                price = ticker_data[ticker].loc[date, "Close"]
                
                # LOG SELLS THAT MIGHT CAUSE ISSUES
                if ticker in portfolio.positions:
                    pos = portfolio.positions[ticker]
                    proceeds = pos['shares'] * price
                    if proceeds > INITIAL_CAPITAL * 2:  # Proceeds more than 2x initial capital
                        print(f"\n💰 LARGE SELL on {date.date()}:")
                        print(f"   Ticker: {ticker}")
                        print(f"   Shares: {pos['shares']}")
                        print(f"   Entry: ${pos['entry_price']:.2f}")
                        print(f"   Exit: ${price:.2f}")
                        print(f"   Proceeds: ${proceeds:,.2f}")
                
                if portfolio.sell(ticker, price, date, nlv):
                    trade_count += 1
                    if not (ticker == WATCH_TICKER):  # Already logged
                        if trade_count <= 10 or TRACE_ALL:
                            print(f"📉 {date.date()} SELL {ticker} @ ${price:.2f}")
    
    # Check pre-existing positions for sells
    for ticker in list(portfolio.pre_existing):
        if ticker in ticker_data and date in ticker_data[ticker].index:
            if ticker_data[ticker].loc[date, "signal"] == -1:
                portfolio.pre_existing.remove(ticker)
                if ticker == WATCH_TICKER:
                    print(f"📉 {date.date()} {WATCH_TICKER} pre-existing cleared")
    
    # Update NLV after sells
    nlv = portfolio.get_nlv(date, ticker_data)
    
    # Check for buy signals
    buy_candidates = []
    for ticker in ticker_data:
        if ticker not in portfolio.positions and ticker not in portfolio.pre_existing:
            if date in ticker_data[ticker].index:
                if ticker_data[ticker].loc[date, "signal"] == 1:
                    price = ticker_data[ticker].loc[date, "Close"]
                    buy_candidates.append((ticker, price))
    
    if buy_candidates:
        random.shuffle(buy_candidates)
        
        for ticker, price in buy_candidates:
            if portfolio.can_buy(nlv):
                if portfolio.buy(ticker, price, date, nlv):
                    trade_count += 1
                    if not (ticker == WATCH_TICKER):  # Already logged
                        if trade_count <= 10 or TRACE_ALL:
                            print(f"📈 {date.date()} BUY  {ticker} @ ${price:.2f}")
                    nlv = portfolio.get_nlv(date, ticker_data)
    
    last_nlv = nlv

# === Final Analysis ===
print("\n" + "="*60)
print(f"📊 TRACE RESULTS FOR {WATCH_TICKER}")
print("="*60)

# Show all signals for watch ticker
if watch_signals:
    print(f"\n📡 All {WATCH_TICKER} signals:")
    for sig in watch_signals:
        print(f"   {sig['date'].date()}: {sig['signal']} @ ${sig['price']:.2f}")
else:
    print(f"\n❌ No signals generated for {WATCH_TICKER}")

# Show trades for watch ticker
watch_trades = [t for t in portfolio.trade_history if t['ticker'] == WATCH_TICKER]
if watch_trades:
    print(f"\n💼 Actual {WATCH_TICKER} trades executed:")
    for trade in watch_trades:
        print(f"   {trade['date'].date()}: {trade['action']} {trade['shares']} shares @ ${trade['price']:.2f}")
else:
    print(f"\n❌ No trades executed for {WATCH_TICKER}")

# Final portfolio state
final_nlv = portfolio.get_nlv(portfolio_dates[-1], ticker_data)
print(f"\n📊 Final Portfolio State:")
print(f"   NLV: ${final_nlv:,.2f}")
print(f"   Return: {((final_nlv - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100):.2f}%")
print(f"   Total trades: {len(portfolio.trade_history)}")

if WATCH_TICKER in portfolio.positions:
    pos = portfolio.positions[WATCH_TICKER]
    print(f"\n🎯 {WATCH_TICKER} final position:")
    print(f"   Shares: {pos['shares']}")
    print(f"   Entry: ${pos['entry_price']:.2f}")
    current = ticker_data[WATCH_TICKER].iloc[-1]["Close"]
    print(f"   Current: ${current:.2f}")
    print(f"   P&L: ${(current - pos['entry_price']) * pos['shares']:,.2f}")

# Check for anomalies
print(f"\n🔍 Anomaly Check:")
anomalies = []

for ticker, pos in portfolio.positions.items():
    if pos['shares'] > 10000:
        anomalies.append(f"{ticker}: {pos['shares']} shares (TOO MANY!)")
    
    expected_cost = INITIAL_CAPITAL * MAX_LEVERAGE * 0.15  # Max 15% per position
    actual_cost = pos['shares'] * pos['entry_price']
    if actual_cost > expected_cost:
        anomalies.append(f"{ticker}: Position size ${actual_cost:,.2f} exceeds expected max ${expected_cost:,.2f}")

if anomalies:
    print("⚠️ ANOMALIES DETECTED:")
    for anomaly in anomalies:
        print(f"   - {anomaly}")
else:
    print("✅ No anomalies detected")

print("\n✅ Trace complete!")