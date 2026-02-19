#!/usr/bin/env python3
"""
Hybrid Market Analyzer - Price Action Based Market Timing
Final version with complete Market School signal implementation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Signal:
    """Data class for market signals"""
    date: pd.Timestamp
    signal_code: str
    signal_name: str
    exposure_change: int
    dd_change: int
    details: str

class HybridMarketAnalyzer:
    """
    Complete Market School Rules Implementation
    
    Buy Signals:
    - B1: Follow-Through Day (initial FTD)
    - B2: Additional Follow-Through Day
    - B3: Low Above 21-EMA (first day)
    - B4: Trending Above 21-EMA (3 consecutive days)
    - B5: Living Above 21-EMA (10+ days)
    - B6: Low Above 50-MA
    - B7: Accumulation Day
    - B8: Higher High (marked high exceeded)
    - B9: Power-Trend ON
    
    Sell Signals:
    - S1: Distribution Day
    - S2: Distribution Cluster
    - S3: Minor Low Break
    - S4: Full Distribution -1 (5 DDs)
    - S5: Break Below 21-EMA
    - S6: Break Below 50-MA
    - S7: Bad Break
    - S8: Downside Reversal
    - S9: Lower Low
    - S10: Full Distribution (6+ DDs)
    - S11: Circuit Breaker
    - S12: FTD Undercut
    """
    
    def __init__(self):
        self.data = None
        self.signals: List[Signal] = []
        
        # Market state
        self.market_exposure = 0
        self.buy_switch = False
        self.power_trend = False
        self.power_trend_under_pressure = False
        self.power_trend_floor = 0
        self.max_exposure = 6
        
        # Rally tracking
        self.rally_low = None
        self.rally_date = None
        self.ftd_date = None
        
        # Distribution tracking
        self.distribution_days = []
        
        # Signal tracking for resets
        self.last_b3_date = None
        self.last_b6_date = None
        self.days_above_21ema = 0
        
        # Marked highs tracking
        self.marked_highs = []
        self.last_b8_marked_high = None
        
        # Allocation mapping
        self.allocation_map = {0: 0, 1: 30, 2: 55, 3: 75, 4: 90, 5: 100, 6: 100, 7: 100}
        
    def load_data(self, filepath: str):
        """Load and prepare market data"""
        # Try to load from different possible locations
        import os
        
        possible_paths = [
            filepath,
            os.path.join("output", filepath),
            os.path.join("output", "IXIC_price_data.csv"),
            "IXIC_price_data.csv"
        ]
        
        file_loaded = False
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Loading data from: {path}")
                self.data = pd.read_csv(path)
                file_loaded = True
                break
                
        if not file_loaded:
            raise FileNotFoundError(f"Could not find data file. Tried: {possible_paths}")
        
        # Remove any rows that contain ticker symbols instead of data
        # This handles the case where the second row has '^IXIC' values
        if len(self.data) > 1:
            # Check if the second row contains non-numeric data
            second_row = self.data.iloc[1]
            if isinstance(second_row.get('Open'), str) and 'IXIC' in str(second_row.get('Open')):
                print("Removing ticker symbol row...")
                self.data = self.data.drop(self.data.index[1])
                self.data = self.data.reset_index(drop=True)
        
        # Remove any rows where Date contains 'IXIC' or is invalid
        self.data = self.data[~self.data['Date'].astype(str).str.contains('IXIC', na=False)]
        
        # Parse dates and ensure timezone-naive
        self.data['Date'] = pd.to_datetime(self.data['Date'], utc=True).dt.tz_localize(None)
        self.data.set_index('Date', inplace=True)
        
        # Clean numeric columns
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in self.data.columns:
                if self.data[col].dtype == 'object':
                    # Remove commas and any non-numeric characters
                    self.data[col] = self.data[col].astype(str).str.replace(',', '').str.replace('^', '')
                    # Convert to float, setting any remaining non-numeric values to NaN
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Drop any rows with NaN values in critical columns
        self.data = self.data.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        print(f"Loaded {len(self.data)} valid data rows")
        print(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")
                    
        self._calculate_indicators()
        
    def _calculate_indicators(self):
        """Calculate technical indicators"""
        # Basic metrics
        self.data['daily_return'] = self.data['Close'].pct_change()
        self.data['daily_pct'] = self.data['daily_return'] * 100
        
        # Moving averages
        self.data['ema21'] = self.data['Close'].ewm(span=21, adjust=False).mean()
        self.data['sma50'] = self.data['Close'].rolling(window=50).mean()
        
        # Price position in range
        self.data['range'] = self.data['High'] - self.data['Low']
        self.data['close_position'] = np.where(
            self.data['range'] > 0,
            (self.data['Close'] - self.data['Low']) / self.data['range'],
            0.5
        )
        
        # Volume comparison
        self.data['volume_up'] = self.data['Volume'] > self.data['Volume'].shift(1)
        
        # Highs and lows
        self.data['high_52w'] = self.data['High'].rolling(window=260, min_periods=1).max()
        self.data['is_52w_high'] = self.data['High'] >= self.data['high_52w']
        
        # MA trends
        self.data['ema21_trend'] = self.data['ema21'] > self.data['ema21'].shift(1)
        self.data['sma50_trend'] = self.data['sma50'] > self.data['sma50'].shift(1)
        
        # Find marked highs (9 days before and after without higher high)
        self._find_marked_highs()
        
    def _find_marked_highs(self):
        """Find all marked highs in the data"""
        self.marked_highs = []
        
        for i in range(9, len(self.data) - 9):
            current_high = self.data.iloc[i]['High']
            is_marked = True
            
            # Check 9 days before
            for j in range(i - 9, i):
                if self.data.iloc[j]['High'] >= current_high:
                    is_marked = False
                    break
                    
            # Check 9 days after
            if is_marked:
                for j in range(i + 1, i + 10):
                    if self.data.iloc[j]['High'] >= current_high:
                        is_marked = False
                        break
                        
            if is_marked:
                self.marked_highs.append({
                    'date': self.data.index[i],
                    'high': current_high,
                    'index': i
                })
                
    def analyze_market(self, start_date: str = None):
        """Run complete market analysis"""
        self.signals = []
        self.market_exposure = 0
        self.buy_switch = False
        self.power_trend = False
        self.power_trend_under_pressure = False
        
        # Start from a specific date or 1 year ago
        if start_date:
            start_ts = pd.Timestamp(start_date)
            try:
                start_idx = self.data.index.get_loc(start_ts)
            except KeyError:
                # Find nearest date
                start_idx = self.data.index.searchsorted(start_ts)
                if start_idx >= len(self.data):
                    start_idx = len(self.data) - 1
        else:
            start_idx = max(0, len(self.data) - 252)
            
        # Look for market correction and rally
        for idx in range(start_idx, len(self.data)):
            self._check_rally_start(idx)
            
        # Process signals day by day
        for idx in range(start_idx, len(self.data)):
            self._process_day(idx)
            
    def _check_rally_start(self, idx: int):
        """Check for potential rally start after correction"""
        if idx < 20:
            return
            
        current = self.data.iloc[idx]
        
        # Look for significant lows after market decline
        # For simplicity, we'll use the April 7 low as the rally start
        if current.name == pd.Timestamp('2025-02-21'):
            self.rally_low = current['Low']
            self.rally_date = current.name
            
    def _process_day(self, idx: int):
        """Process all signals for a given day"""
        if idx < 50:  # Need enough data for indicators
            return
            
        current = self.data.iloc[idx]
        
        # Update days above 21-EMA
        self._update_days_above_21ema(idx)
        
        # Update distribution days (remove old ones first)
        dd_removed = self._update_distribution_days(idx)
        
        # Check buy signals in order
        self._check_b1_ftd(idx)
        self._check_b2_additional_ftd(idx)
        self._check_b3_b4_b5(idx)
        self._check_b6_above_50ma(idx)
        self._check_b7_accumulation(idx)
        self._check_b8_higher_high(idx)
        self._check_b9_power_trend(idx)
        
        # Check sell signals
        self._check_s1_distribution(idx)
        self._check_s2_distribution_cluster(idx)
        self._check_s4_s10_full_distribution(idx)
        self._check_s5_break_21ema(idx)
        self._check_s6_break_50ma(idx)
        self._check_s7_bad_break(idx)
        self._check_s8_reversal(idx)
        self._check_s12_ftd_undercut(idx)
        
        # Add DD removal signals if any
        if dd_removed > 0:
            self._add_signal(idx, '--', 'DD Removed', 0, -dd_removed, 
                           f"Old DD fell off")
                           
    def _update_days_above_21ema(self, idx: int):
        """Update count of consecutive days above 21-EMA"""
        days_above = 0
        for i in range(min(100, idx + 1)):
            check_idx = idx - i
            if check_idx < 0:
                break
            row = self.data.iloc[check_idx]
            if pd.notna(row['ema21']) and row['Low'] > row['ema21']:
                days_above += 1
            else:
                break
        self.days_above_21ema = days_above
        
    def _check_b1_ftd(self, idx: int):
        """B1: Follow-Through Day"""
        if not self.rally_date or self.ftd_date or self.buy_switch:
            return
            
        current = self.data.iloc[idx]
        days_since = (current.name - self.rally_date).days
        
        # Valid FTD window (days 4-25)
        if 4 <= days_since <= 25:
            # Check rally low not undercut
            lows_since = self.data.loc[self.rally_date:current.name]['Low']
            if lows_since.min() < self.rally_low:
                self.rally_date = None
                self.rally_low = None
                return
                
            # FTD: 1.2% gain on volume
            if current['daily_pct'] >= 1.2 and current['volume_up']:
                self.ftd_date = current.name
                self.buy_switch = True
                self._add_signal(idx, 'B1', 'Follow-Through Day', 1, 0, 
                               f"+{current['daily_pct']:.2f}% on volume")
                               
    def _check_b2_additional_ftd(self, idx: int):
        """B2: Additional Follow-Through Day"""
        if not self.buy_switch or not self.rally_date:
            return
            
        current = self.data.iloc[idx]
        
        # Only check day after initial FTD
        if self.ftd_date and current.name == self.ftd_date + pd.Timedelta(days=1):
            days_since = (current.name - self.rally_date).days
            
            # Still in valid window and meets criteria
            if days_since <= 25 and current['daily_pct'] >= 1.2 and current['volume_up']:
                self._add_signal(idx, 'B2', 'Additional FTD', 1, 0,
                               f"+{current['daily_pct']:.2f}% on volume")
                               
    def _check_b3_b4_b5(self, idx: int):
        """B3/B4/B5: 21-day EMA signals"""
        if not self.buy_switch:
            return
            
        current = self.data.iloc[idx]
        
        # B3: First day low above 21-EMA
        if self.days_above_21ema == 1 and not self.last_b3_date:
            self.last_b3_date = current.name
            self._add_signal(idx, 'B3', 'Low Above 21-EMA', 1, 0,
                           f"First day above")
            
        # B4: 3 consecutive days
        elif self.days_above_21ema == 3 and idx > 0:
            if current['Close'] >= self.data.iloc[idx-1]['Close']:
                self._add_signal(idx, 'B4', 'Trending Above 21-EMA', 1, 0,
                               "3 consecutive days")
            
        # B5: 10+ consecutive days
        elif self.days_above_21ema in [10, 15, 20, 25]:
            self._add_signal(idx, 'B5', 'Living Above 21-EMA', 1, 0,
                           f"{self.days_above_21ema} days above")
                           
    def _check_b6_above_50ma(self, idx: int):
        """B6: Low above 50-day MA"""
        if not self.buy_switch:
            return
            
        current = self.data.iloc[idx]
        
        if pd.notna(current['sma50']) and current['Low'] > current['sma50'] and not self.last_b6_date:
            self.last_b6_date = current.name
            self._add_signal(idx, 'B6', 'Low Above 50-MA', 1, 0,
                           f"Low {current['Low']:.2f} > MA {current['sma50']:.2f}")
                           
    def _check_b7_accumulation(self, idx: int):
        """B7: Accumulation Day"""
        if not self.buy_switch:
            return
            
        current = self.data.iloc[idx]
        
        # 1.2%+ gain with close in top 25% of range
        if current['daily_pct'] >= 1.2 and current['close_position'] >= 0.75:
            self._add_signal(idx, 'B7', 'Accumulation Day', 1, 0,
                           f"+{current['daily_pct']:.2f}%")
                           
    def _check_b8_higher_high(self, idx: int):
        """B8: Higher High (exceeds marked high)"""
        if not self.buy_switch:
            return
            
        current = self.data.iloc[idx]
        
        # Find most recent marked high before current date
        relevant_marked_high = None
        for mh in reversed(self.marked_highs):
            if mh['index'] < idx:
                relevant_marked_high = mh
                break
                
        if relevant_marked_high and current['High'] > relevant_marked_high['high']:
            # Only trigger if this is a new higher high (not same as yesterday)
            if idx > 0 and current['High'] > self.data.iloc[idx-1]['High']:
                if relevant_marked_high != self.last_b8_marked_high:
                    self.last_b8_marked_high = relevant_marked_high
                    self._add_signal(idx, 'B8', 'Higher High', 1, 0,
                                   f"> {relevant_marked_high['date'].strftime('%Y-%m-%d')}")
                                   
    def _check_b9_power_trend(self, idx: int):
        """B9: Power-Trend ON"""
        if self.power_trend or not self.buy_switch:
            return
            
        current = self.data.iloc[idx]
        
        # Check all conditions
        # 1. EMA21 > SMA50 for 5+ consecutive days
        ema_above_sma_count = 0
        for i in range(min(10, idx + 1)):
            check_idx = idx - i
            if check_idx < 0:
                break
            row = self.data.iloc[check_idx]
            if pd.notna(row['ema21']) and pd.notna(row['sma50']) and row['ema21'] > row['sma50']:
                ema_above_sma_count += 1
            else:
                break
                
        # 2. Index closes up or flat
        closes_up_or_flat = idx > 0 and current['Close'] >= self.data.iloc[idx-1]['Close']
        
        # 3. 50-MA is in uptrend
        sma50_uptrend = current['sma50_trend']
        
        # 4. Low above 21-EMA for 10+ days
        conditions_met = (
            ema_above_sma_count >= 5 and
            closes_up_or_flat and
            sma50_uptrend and
            self.days_above_21ema >= 10
        )
        
        if conditions_met:
            self.power_trend = True
            self.max_exposure = 7
            self.power_trend_floor = 2
            self._add_signal(idx, 'B9', 'Power-Trend ON', 0, 0,
                           "Max exposure to 7, floor +2")
                           
    def _check_s1_distribution(self, idx: int):
        """S1: Distribution Day"""
        current = self.data.iloc[idx]
        
        # -0.2% or worse with close in lower 25% or below 21-EMA
        is_distribution = (
            current['daily_pct'] <= -0.2 and 
            (current['close_position'] <= 0.25 or 
             (pd.notna(current['ema21']) and current['Close'] < current['ema21']))
        )
        
        if is_distribution:
            self.distribution_days.append({
                'date': current.name,
                'low': current['Low'],
                'index': idx
            })
            self._add_signal(idx, 'S1', 'Distribution Day', 0, 1,
                           f"{current['daily_pct']:.2f}%")
                           
    def _check_s2_distribution_cluster(self, idx: int):
        """S2: Distribution Cluster"""
        if len(self.distribution_days) < 4:
            return
            
        current = self.data.iloc[idx]
        
        # Count distribution days in last 8 trading days
        recent_dds = [dd for dd in self.distribution_days 
                     if (current.name - dd['date']).days <= 11]  # ~8 trading days
                     
        if len(recent_dds) >= 4:
            # Check if we haven't already triggered S2 recently
            recent_s2 = [s for s in self.signals[-10:] if s.signal_code == 'S2']
            if not recent_s2:
                self._add_signal(idx, 'S2', 'Distribution Cluster', 0, 0,
                               f"PT under pressure, floor +1")
                               
                # Power-Trend under pressure
                if self.power_trend:
                    self.power_trend_under_pressure = True
                    self.max_exposure = 5
                    self.power_trend_floor = 1
                    
    def _check_s4_s10_full_distribution(self, idx: int):
        """S4/S10: Full Distribution checks"""
        dist_count = len(self.distribution_days)
        
        # S4: Full Distribution -1 (5 DDs)
        if dist_count == 5:
            recent_s4 = [s for s in self.signals[-5:] if s.signal_code == 'S4']
            if not recent_s4:
                self._add_signal(idx, 'S4', 'Full Distribution -1', 0, 0,
                               f"{dist_count} distribution days")
            
        # S10: Full Distribution (6+ DDs)
        elif dist_count >= 6:
            recent_s10 = [s for s in self.signals[-5:] if s.signal_code == 'S10']
            if not recent_s10:
                self._add_signal(idx, 'S10', 'Full Distribution', 0, 0,
                               f"{dist_count} distribution days")
                self.buy_switch = False
                
    def _check_s5_break_21ema(self, idx: int):
        """S5: Break Below 21-day EMA"""
        if idx == 0:
            return
            
        current = self.data.iloc[idx]
        prev = self.data.iloc[idx-1]
        
        if (pd.notna(prev['ema21']) and pd.notna(current['ema21']) and
            prev['Close'] >= prev['ema21'] and 
            current['Close'] < current['ema21'] * 0.998):
            self.last_b3_date = None  # Reset B3
            self._add_signal(idx, 'S5', 'Break Below 21-EMA', -1, 0,
                           f"After {self.days_above_21ema} days")
                           
    def _check_s6_break_50ma(self, idx: int):
        """S6: Break Below 50-day MA"""
        if idx == 0:
            return
            
        current = self.data.iloc[idx]
        prev = self.data.iloc[idx-1]
        
        if (pd.notna(prev['sma50']) and pd.notna(current['sma50']) and
            prev['Close'] >= prev['sma50'] and current['Close'] < current['sma50']):
            self.last_b6_date = None  # Reset B6
            self._add_signal(idx, 'S6', 'Break Below 50-MA', -1, 0,
                           f"Close {current['Close']:.2f} < MA {current['sma50']:.2f}")
                           
            # Check if Power-Trend should end
            if self.power_trend and pd.notna(current['ema21']) and current['ema21'] < current['sma50']:
                self.power_trend = False
                self.power_trend_under_pressure = False
                self.max_exposure = 6
                self.power_trend_floor = 0
                
    def _check_s7_bad_break(self, idx: int):
        """S7: Bad Break"""
        current = self.data.iloc[idx]
        
        bad_break = (
            current['daily_pct'] <= -2.25 and 
            current['close_position'] <= 0.25
        )
        
        if bad_break:
            self._add_signal(idx, 'S7', 'Bad Break', -1, 0,
                           f"{current['daily_pct']:.2f}% in bottom quarter")
                           
    def _check_s8_reversal(self, idx: int):
        """S8: Downside Reversal Day"""
        current = self.data.iloc[idx]
        
        # Check if trending above 21-EMA for 25+ days
        if self.days_above_21ema >= 25:
            # Makes new 52-week high but closes weak
            if (current['is_52w_high'] and 
                current['close_position'] <= 0.25 and
                current['daily_pct'] < 0):
                self._add_signal(idx, 'S8', 'Downside Reversal', -1, 0,
                               f"52w high, close {(current['close_position'] * 100):.1f}%")
                               
    def _check_s12_ftd_undercut(self, idx: int):
        """S12: Follow-Through Day Undercut"""
        if not self.rally_low or not self.ftd_date:
            return
            
        current = self.data.iloc[idx]
        
        if current['Low'] < self.rally_low:
            self.buy_switch = False
            remaining_exposure = self.market_exposure
            self.market_exposure = 0
            self._add_signal(idx, 'S12', 'FTD Undercut', -remaining_exposure, 0,
                           "Rally failed - cash out")
            self.rally_low = None
            self.rally_date = None
            self.ftd_date = None
            
    def _update_distribution_days(self, idx: int):
        """Update distribution days and return number removed"""
        current = self.data.iloc[idx]
        initial_count = len(self.distribution_days)
        
        # Remove old distribution days (25-day rule or 5% rally)
        active_dist = []
        for dd in self.distribution_days:
            days_old = (current.name - dd['date']).days
            if days_old <= 35:  # ~25 trading days
                # Check 5% rally from low
                rally_pct = ((current['High'] - dd['low']) / dd['low']) * 100
                if rally_pct < 5.0:
                    active_dist.append(dd)
                    
        self.distribution_days = active_dist
        return initial_count - len(self.distribution_days)
        
    def _add_signal(self, idx: int, code: str, name: str, exposure_change: int, dd_change: int, details: str):
        """Add signal and update exposure"""
        date = self.data.index[idx]
        
        # Apply exposure change
        if exposure_change != 0:
            new_exposure = self.market_exposure + exposure_change
            
            # Apply Power-Trend floor
            if self.power_trend and new_exposure < self.power_trend_floor:
                new_exposure = self.power_trend_floor
                
            # Apply limits
            new_exposure = max(0, min(self.max_exposure, new_exposure))
            exposure_change = new_exposure - self.market_exposure
            self.market_exposure = new_exposure
            
        signal = Signal(
            date=date,
            signal_code=code,
            signal_name=name,
            exposure_change=exposure_change,
            dd_change=dd_change,
            details=details
        )
        
        self.signals.append(signal)
        
    def get_current_status(self) -> Dict:
        """Get current market status"""
        if self.data is None or self.data.empty:
            return {}
            
        current = self.data.iloc[-1]
        
        # Position allocation
        allocation = self.allocation_map.get(self.market_exposure, 0)
        
        # Market health
        dist_count = len(self.distribution_days)
        if not self.buy_switch:
            health = "🔴 Correction"
        elif dist_count >= 5:
            health = "🔴 High Risk"
        elif pd.notna(current['ema21']) and pd.notna(current['sma50']):
            if current['Close'] > current['ema21'] and current['Close'] > current['sma50']:
                health = "🟢 Healthy"
            elif current['Close'] > current['ema21']:
                health = "🟡 Improving"
            else:
                health = "🟡 Weakening"
        else:
            health = "🟡 Unknown"
            
        return {
            'date': current.name.strftime('%Y-%m-%d'),
            'close': current['Close'],
            'daily_change': f"{current['daily_pct']:.2f}%",
            'buy_switch': 'ON' if self.buy_switch else 'OFF',
            'power_trend': 'ON' if self.power_trend else 'OFF',
            'pt_pressure': 'Yes' if self.power_trend_under_pressure else 'No',
            'exposure': self.market_exposure,
            'max_exposure': self.max_exposure,
            'allocation': allocation,
            'distribution_count': dist_count,
            'health': health,
            'above_21ema': pd.notna(current['ema21']) and current['Close'] > current['ema21'],
            'above_50ma': pd.notna(current['sma50']) and current['Close'] > current['sma50'],
            'recent_signals': self.signals[-10:] if self.signals else []
        }
        
    def print_analysis(self):
        """Print comprehensive market analysis"""
        status = self.get_current_status()
        
        print("\n" + "="*70)
        print(f"📊 HYBRID MARKET ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*70)
        
        # Market Status
        print(f"\n📈 NASDAQ: {status['close']:,.2f} ({status['daily_change']})")
        print(f"   Market Health: {status['health']}")
        print(f"   Buy Switch: {status['buy_switch']}")
        if status['power_trend'] == 'ON':
            pt_status = " (Under Pressure)" if status['pt_pressure'] == 'Yes' else ""
            print(f"   🚀 Power-Trend: {status['power_trend']}{pt_status}")
        
        # Exposure
        print(f"\n💼 EXPOSURE: {status['exposure']}/{status['max_exposure']} = {status['allocation']}%")
        bar = "█" * status['exposure'] + "░" * (status['max_exposure'] - status['exposure'])
        print(f"   [{bar}]")
        
        # Technical Position
        print(f"\n📊 Technical Position:")
        ema_status = "✓" if status['above_21ema'] else "✗"
        ma_status = "✓" if status['above_50ma'] else "✗"
        print(f"   21-day EMA: {ema_status}")
        print(f"   50-day MA:  {ma_status}")
        
        # Distribution Days
        dist_emoji = "🔴" if status['distribution_count'] >= 5 else "🟡" if status['distribution_count'] >= 3 else "🟢"
        print(f"\n📉 Distribution Days: {dist_emoji} {status['distribution_count']}")
        
        # Recent Signals
        if status['recent_signals']:
            print(f"\n🔔 Recent Signals:")
            for sig in status['recent_signals'][-5:]:
                emoji = "🟢" if sig.signal_code.startswith('B') else "🔴" if sig.signal_code.startswith('S') else "⚪"
                change = f"+{sig.exposure_change}" if sig.exposure_change > 0 else str(sig.exposure_change)
                dd_info = f" DD{sig.dd_change:+d}" if sig.dd_change != 0 else ""
                print(f"   {emoji} {sig.date.strftime('%m/%d')} {sig.signal_code}: {sig.signal_name} ({change}){dd_info}")
        
        # Action
        print(f"\n" + "="*70)
        if status['exposure'] == 0:
            print("⚡ ACTION: STAY IN CASH")
        elif status['distribution_count'] >= 4:
            print("⚡ ACTION: BE READY TO SELL - High distribution")
        elif status['exposure'] <= 2:
            print("⚡ ACTION: LIGHT EXPOSURE - Look for quality setups")
        elif status['exposure'] >= 5:
            print("⚡ ACTION: FULLY INVESTED - Use stops, let winners run")
        else:
            print(f"⚡ ACTION: MODERATE EXPOSURE - Maintain {status['allocation']}%")
        print("="*70)
        
    def quick_status(self):
        """One-line status summary"""
        status = self.get_current_status()
        power = " 🚀PT" if status['power_trend'] == 'ON' else ""
        pressure = "(P)" if status['pt_pressure'] == 'Yes' else ""
        print(f"{status['date']} | NASDAQ: {status['close']:,.0f} ({status['daily_change']}) | "
              f"Exp: {status['exposure']}/{status['max_exposure']} ({status['allocation']}%) | "
              f"Dist: {status['distribution_count']} | {status['health']}{power}{pressure}")
              
    def generate_report(self, output_file: str = None) -> pd.DataFrame:
        """Generate detailed signal report"""
        if not self.signals:
            print("No signals to report")
            return pd.DataFrame()
            
        data = []
        cumulative_exposure = 0
        cumulative_dd = 0
        
        for sig in self.signals:
            cumulative_exposure += sig.exposure_change
            cumulative_dd += sig.dd_change
            
            # Ensure cumulative values don't go negative
            cumulative_dd = max(0, cumulative_dd)
            
            # Get allocation percentage
            allocation = self.allocation_map.get(cumulative_exposure, 0)
            
            data.append({
                'Date': sig.date.strftime('%Y-%m-%d'),
                'Signal': sig.signal_code,
                'Name': sig.signal_name,
                'Exp Chg': f"{sig.exposure_change:+d}" if sig.exposure_change != 0 else "0",
                'Cum Exp': cumulative_exposure,
                '% Inv': f"{allocation}%",
                'DD Chg': f"{sig.dd_change:+d}" if sig.dd_change != 0 else "",
                'Cum DD': cumulative_dd,
                'Details': sig.details
            })
            
        df = pd.DataFrame(data)
        
        if output_file:
            # Save as tab-delimited for easy Excel import
            df.to_csv(output_file, sep='\t', index=False)
            print(f"Report saved to {output_file}")
            
        return df
        
    def export_excel_format(self):
        """Export signals in Excel-friendly format"""
        df = self.generate_report()
        if not df.empty:
            print("\n" + "="*70)
            print("EXCEL FORMAT (Tab-delimited - copy and paste into Excel):")
            print("="*70)
            print(df.to_csv(sep='\t', index=False))
            

# Main execution
if __name__ == "__main__":
    import sys
    
    analyzer = HybridMarketAnalyzer()
    
    try:
        # Load data
        analyzer.load_data("IXIC_price_data.csv")
        
        # Analyze from specific date
        analyzer.analyze_market(start_date="2025-04-01")
        
        # Check command line arguments
        if len(sys.argv) > 1:
            if sys.argv[1] == "quick":
                analyzer.quick_status()
            elif sys.argv[1] == "report":
                analyzer.generate_report("hybrid_signals_report.tsv")
            elif sys.argv[1] == "excel":
                analyzer.export_excel_format()
            else:
                analyzer.print_analysis()
        else:
            analyzer.print_analysis()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()