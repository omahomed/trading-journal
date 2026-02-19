#!/usr/bin/env python3
"""
Simplified Market School Rules Analyzer
Focuses on core functionality for daily market exposure decisions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SimpleMarketAnalyzer:
    """Simplified market analyzer focusing on essential rules"""
    
    def __init__(self):
        self.data = None
        self.current_exposure = 0  # 0-6 scale
        self.distribution_days = []
        self.ftd_date = None
        self.rally_low = None
        self.buy_switch = False
        
    def load_data(self, filepath):
        """Load and prepare market data"""
        # Read CSV
        self.data = pd.read_csv(filepath)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index('Date', inplace=True)
        
        # Clean numeric columns
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if self.data[col].dtype == 'object':
                self.data[col] = self.data[col].str.replace(',', '').astype(float)
        
        # Calculate indicators
        self._calculate_indicators()
        
        # Analyze recent signals
        self._analyze_recent_market()
        
    def _calculate_indicators(self):
        """Calculate essential technical indicators"""
        # Returns
        self.data['daily_return'] = self.data['Close'].pct_change()
        self.data['daily_pct'] = self.data['daily_return'] * 100
        
        # Moving averages
        self.data['ema21'] = self.data['Close'].ewm(span=21, adjust=False).mean()
        self.data['sma50'] = self.data['Close'].rolling(window=50).mean()
        
        # Volume - ensure numeric comparison
        self.data['Volume'] = pd.to_numeric(self.data['Volume'], errors='coerce')
        self.data['volume_up'] = self.data['Volume'] > self.data['Volume'].shift(1)
        
        # 52-week metrics
        self.data['high_52w'] = self.data['High'].rolling(window=260, min_periods=1).max()
        self.data['low_52w'] = self.data['Low'].rolling(window=260, min_periods=1).min()
        self.data['pct_off_high'] = ((self.data['Close'] - self.data['high_52w']) / self.data['high_52w']) * 100
        
    def _analyze_recent_market(self):
        """Analyze recent market action to determine current state"""
        # Only analyze last 100 days for efficiency
        recent_data = self.data.tail(100).copy()
        
        # Find potential rally start (10% correction)
        for i in range(len(recent_data) - 20):
            if recent_data.iloc[i]['pct_off_high'] <= -10:
                # Found a correction, look for rally
                self.rally_low = recent_data.iloc[i]['Low']
                
                # Check for follow-through day (1.2% up on volume)
                for j in range(i + 4, min(i + 25, len(recent_data))):
                    day = recent_data.iloc[j]
                    if (day['daily_pct'] >= 1.2 and 
                        day['volume_up'] and 
                        day['Low'] > self.rally_low):
                        self.ftd_date = day.name
                        self.buy_switch = True
                        break
                
                if self.buy_switch:
                    break
        
        # Count recent distribution days
        self._count_distribution_days()
        
        # Determine current exposure based on signals
        self._calculate_current_exposure()
        
    def _count_distribution_days(self):
        """Count distribution days with proper removal rules"""
        # Look back 35 trading days to track all potential distribution days
        lookback_data = self.data.tail(35).copy()
        self.distribution_days = []
        
        # Find all distribution days
        for i in range(1, len(lookback_data)):
            day = lookback_data.iloc[i]
            prev_day = lookback_data.iloc[i-1]
            
            # Distribution day: -0.2% or worse on higher volume
            if day['daily_pct'] <= -0.2 and day['volume_up']:
                dist_day = {
                    'date': day.name,
                    'loss': day['daily_pct'],
                    'low': day['Low'],
                    'close': day['Close'],
                    'removed': False,
                    'removal_reason': None
                }
                
                # Check if this distribution day should be removed
                # Rule 1: 25 trading days old (automatically handled by only looking back 35 days)
                days_old = len(self.data.loc[day.name:]) - 1
                if days_old > 25:
                    dist_day['removed'] = True
                    dist_day['removal_reason'] = '25-day rule'
                
                # Rule 2: Market rallies 5% from the distribution day's low
                # or 6% from the distribution day's close
                if not dist_day['removed']:
                    subsequent_data = self.data.loc[day.name:]
                    for j in range(1, len(subsequent_data)):
                        check_day = subsequent_data.iloc[j]
                        
                        # 5% rally from low
                        rally_from_low = ((check_day['High'] - day['Low']) / day['Low']) * 100
                        if rally_from_low >= 5.0:
                            dist_day['removed'] = True
                            dist_day['removal_reason'] = f'5% rally from low on {check_day.name.strftime("%Y-%m-%d")}'
                            break
                        
                        # 6% rally from close
                        rally_from_close = ((check_day['Close'] - day['Close']) / day['Close']) * 100
                        if rally_from_close >= 6.0:
                            dist_day['removed'] = True
                            dist_day['removal_reason'] = f'6% rally from close on {check_day.name.strftime("%Y-%m-%d")}'
                            break
                
                self.distribution_days.append(dist_day)
        
        # Filter to only active (non-removed) distribution days
        self.distribution_days = [d for d in self.distribution_days if not d['removed']]
        
    def _calculate_current_exposure(self):
        """Calculate recommended exposure based on current conditions"""
        current = self.data.iloc[-1]
        
        # Reset if buy switch is off
        if not self.buy_switch:
            self.current_exposure = 0
            return
        
        # Start with base exposure
        exposure = 1
        
        # Add for positive conditions
        if current['Close'] > current['ema21']:
            exposure += 1
        if current['Low'] > current['ema21']:  # Stronger signal
            exposure += 1
        if current['Close'] > current['sma50']:
            exposure += 1
        if current['pct_off_high'] > -5:  # Near highs
            exposure += 1
        
        # Check trending above MAs
        above_21_days = 0
        for i in range(1, min(11, len(self.data))):
            if self.data.iloc[-i]['Low'] > self.data.iloc[-i]['ema21']:
                above_21_days += 1
            else:
                break
        
        if above_21_days >= 10:
            exposure += 1
        
        # Reduce for distribution
        dist_count = len(self.distribution_days)
        if dist_count >= 4:
            exposure -= 1
        if dist_count >= 5:
            exposure -= 2
        if dist_count >= 6:
            exposure = 0
            self.buy_switch = False
        
        # Cap exposure
        self.current_exposure = max(0, min(6, exposure))
        
    def get_position_size(self):
        """Convert exposure to position size percentage"""
        allocation_map = {
            0: 0,
            1: 30,
            2: 55,
            3: 75,
            4: 90,
            5: 100,
            6: 100
        }
        return allocation_map.get(self.current_exposure, 0)
        
    def get_market_health(self):
        """Determine overall market health"""
        current = self.data.iloc[-1]
        dist_count = len(self.distribution_days)
        
        if not self.buy_switch:
            return "🔴 Correction", "Stay in cash - waiting for follow-through day"
        elif dist_count >= 5:
            return "🔴 High Risk", f"{dist_count} distribution days - reduce exposure"
        elif current['Close'] > current['ema21'] and current['Close'] > current['sma50']:
            return "🟢 Healthy", "Market trending well"
        elif current['Close'] > current['ema21']:
            return "🟡 Improving", "Above 21-day but below 50-day MA"
        else:
            return "🟡 Weakening", "Below 21-day MA - caution"
            
    def print_analysis(self):
        """Print comprehensive market analysis"""
        current = self.data.iloc[-1]
        prev = self.data.iloc[-2]
        
        # Calculate metrics
        change_pct = ((current['Close'] - prev['Close']) / prev['Close']) * 100
        dist_count = len(self.distribution_days)
        position_size = self.get_position_size()
        health, health_detail = self.get_market_health()
        
        # Print dashboard
        print("\n" + "="*60)
        print(f"📊 MARKET ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*60)
        
        print(f"\n📈 NASDAQ: {current['Close']:,.2f} ({change_pct:+.2f}%)")
        print(f"   52-week High: {current['high_52w']:,.2f} ({current['pct_off_high']:+.1f}%)")
        print(f"   Market Health: {health}")
        print(f"   Status: {health_detail}")
        
        # Exposure
        print(f"\n💼 RECOMMENDED EXPOSURE: {self.current_exposure}/6 = {position_size}%")
        bar = "█" * self.current_exposure + "░" * (6 - self.current_exposure)
        print(f"   [{bar}]")
        
        # Technical Position
        print(f"\n📊 Technical Position:")
        ema21_status = "✓" if current['Close'] > current['ema21'] else "✗"
        sma50_status = "✓" if current['Close'] > current['sma50'] else "✗"
        ema21_dist = ((current['Close'] - current['ema21']) / current['ema21']) * 100
        sma50_dist = ((current['Close'] - current['sma50']) / current['sma50']) * 100
        
        print(f"   21-day EMA: {ema21_status} ({ema21_dist:+.1f}%)")
        print(f"   50-day MA:  {sma50_status} ({sma50_dist:+.1f}%)")
        
        # Distribution Days
        print(f"\n📉 Distribution Days: {dist_count}")
        if self.distribution_days:
            for d in self.distribution_days[-3:]:  # Show last 3
                print(f"   • {d['date'].strftime('%Y-%m-%d')} ({d['loss']:.2f}%)")
        
        # Market Signals
        print(f"\n🔔 Key Signals:")
        print(f"   Buy Switch: {'ON' if self.buy_switch else 'OFF'}")
        if self.ftd_date:
            print(f"   Last FTD: {self.ftd_date.strftime('%Y-%m-%d')}")
        
        # ACTION
        print(f"\n" + "="*60)
        if self.current_exposure == 0:
            print("⚡ ACTION: STAY IN CASH")
        elif self.current_exposure <= 2:
            print("⚡ ACTION: LIGHT EXPOSURE - Look for quality setups")
        elif self.current_exposure <= 4:
            print("⚡ ACTION: MODERATE EXPOSURE - Build positions gradually")
        else:
            print("⚡ ACTION: FULL EXPOSURE - Let winners run, use stops")
            
        if dist_count >= 4:
            print("   ⚠️  WARNING: High distribution count - be ready to sell")
            
        print("="*60)
        
    def debug_distribution_days(self):
        """Show detailed distribution day analysis for debugging"""
        print("\n" + "="*60)
        print("DISTRIBUTION DAY ANALYSIS")
        print("="*60)
        
        # Look for ALL distribution days in last 35 trading days
        lookback_data = self.data.tail(35).copy()
        all_dist_days = []
        
        for i in range(1, len(lookback_data)):
            day = lookback_data.iloc[i]
            if day['daily_pct'] <= -0.2 and day['volume_up']:
                all_dist_days.append({
                    'date': day.name,
                    'loss': day['daily_pct'],
                    'low': day['Low'],
                    'close': day['Close']
                })
        
        print(f"\nFound {len(all_dist_days)} distribution days in last 35 trading days:")
        
        for dd in all_dist_days:
            print(f"\n📉 {dd['date'].strftime('%Y-%m-%d')} (Loss: {dd['loss']:.2f}%)")
            print(f"   Low: {dd['low']:,.2f}, Close: {dd['close']:,.2f}")
            
            # Check removal conditions
            removed = False
            
            # Check days old
            days_old = len(self.data.loc[dd['date']:]) - 1
            print(f"   Days old: {days_old}")
            
            # Check rally from low/close
            subsequent_data = self.data.loc[dd['date']:]
            max_rally_low = 0
            max_rally_close = 0
            rally_date = None
            
            for j in range(1, len(subsequent_data)):
                check_day = subsequent_data.iloc[j]
                rally_from_low = ((check_day['High'] - dd['low']) / dd['low']) * 100
                rally_from_close = ((check_day['Close'] - dd['close']) / dd['close']) * 100
                
                if rally_from_low > max_rally_low:
                    max_rally_low = rally_from_low
                    if rally_from_low >= 5.0 and not removed:
                        removed = True
                        rally_date = check_day.name
                        print(f"   ✓ REMOVED: 5% rally from low on {rally_date.strftime('%Y-%m-%d')} ({rally_from_low:.2f}%)")
                        
                if rally_from_close > max_rally_close:
                    max_rally_close = rally_from_close
                    if rally_from_close >= 6.0 and not removed:
                        removed = True
                        rally_date = check_day.name
                        print(f"   ✓ REMOVED: 6% rally from close on {rally_date.strftime('%Y-%m-%d')} ({rally_from_close:.2f}%)")
            
            if not removed:
                print(f"   Max rally from low: {max_rally_low:.2f}%")
                print(f"   Max rally from close: {max_rally_close:.2f}%")
                print(f"   ⚠️  ACTIVE distribution day")
            
        print(f"\n📊 Total ACTIVE distribution days: {len(self.distribution_days)}")
        print("="*60)
        """One-line status summary"""
        current = self.data.iloc[-1]
        prev = self.data.iloc[-2]
        change_pct = ((current['Close'] - prev['Close']) / prev['Close']) * 100
        position_size = self.get_position_size()
        dist_count = len(self.distribution_days)
        
        status = "🟢" if self.current_exposure >= 4 else "🟡" if self.current_exposure >= 2 else "🔴"
        
        print(f"{datetime.now().strftime('%Y-%m-%d')} | NASDAQ: {current['Close']:,.0f} ({change_pct:+.2f}%) | Exposure: {self.current_exposure}/6 ({position_size}%) | Dist: {dist_count} | {status}")

# Main execution
if __name__ == "__main__":
    import sys
    
    analyzer = SimpleMarketAnalyzer()
    
    try:
        # Load data
        analyzer.load_data("IXIC_price_data.csv")
        
        # Check command line arguments
        if len(sys.argv) > 1:
            if sys.argv[1] == "quick":
                analyzer.quick_status()
            elif sys.argv[1] == "debug":
                analyzer.print_analysis()
                analyzer.debug_distribution_days()
        else:
            analyzer.print_analysis()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()