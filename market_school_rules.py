import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import yfinance as yf
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class SignalType(Enum):
    """Enumeration of all signal types"""
    # Buy Signals
    B1 = "Follow-Through Day"
    B2 = "Additional Follow-Through Day"
    B3 = "Low Above 21-day MA"
    B4 = "Trending Above 21-day MA"
    B5 = "Living Above 21-day MA"
    B6 = "Low Above 50-day MA"
    B7 = "Accumulation Day"
    B8 = "Higher High"
    B9 = "Downside Reversal Buyback"
    B10 = "Distribution Day Fall Off"
    
    # Sell Signals
    S1 = "Follow-Through Day Undercut"
    S2 = "Failed Rally Attempt"
    S3 = "Full Distribution minus one"
    S4 = "Full Distribution"
    S5 = "Break Below 21-day MA"
    S6 = "Overdue Break Below 21-day MA"
    S7 = "Trending Below 21-day MA"
    S8 = "Living Below 21-day MA"
    S9 = "Break Below 50-day MA"
    S10 = "Bad Break"
    S11 = "Downside Reversal Day"
    S12 = "Lower Low"
    S13 = "Distribution Cluster"
    S14 = "Break Below Higher High"

@dataclass
class Signal:
    """Data class for market signals"""
    date: pd.Timestamp
    signal_type: SignalType
    price: float
    description: str
    affects_exposure: bool = False
    exposure_change: int = 0

@dataclass
class DistributionDay:
    """Data class for tracking distribution days"""
    date: pd.Timestamp
    type: str  # 'distribution' or 'stall'
    low: float
    close: float
    loss_percent: float
    removed_date: Optional[pd.Timestamp] = None
    removal_reason: Optional[str] = None

class MarketSchoolRules:
    """
    Implementation of IBD Market School Rules for systematic market timing.
    """
    
    def __init__(self, symbol: str = "^IXIC"):
        """
        Initialize the Market School Rules system.
        
        Args:
            symbol: Market index symbol (default: ^IXIC for NASDAQ)
        """
        self.symbol = symbol
        self.data = None
        self.signals: List[Signal] = []
        
        # Market state variables
        self.market_exposure = 0  # 0 to 6 based on position count
        self.buy_switch = False
        self.power_trend = False
        self.power_trend_count = 0
        
        # Rally tracking
        self.rally_start_date = None
        self.rally_low = None
        self.rally_low_idx = None
        self.ftd_date = None
        self.ftd_close = None
        
        # Distribution tracking
        self.distribution_days: List[DistributionDay] = []
        self.active_distribution_count = 0
        
        # Signal state tracking
        self.last_b3_b4_b5_date = None
        self.last_s5_s6_s7_s8_date = None
        self.last_b6_date = None
        self.last_s9_date = None
        
        # Restraint rule
        self.restraint_active = False
        
        # MarketSmith style highs/lows tracking
        self.marked_highs = []
        self.marked_lows = []
        
        # 52-week high tracking for 10% decline requirement
        self.recent_high = None
        self.recent_high_date = None
        self.market_in_correction = False
        
    def load_data(self, filepath: str = None, data: pd.DataFrame = None):
        """
        Load market data from file or DataFrame.
        
        Args:
            filepath: Path to CSV file
            data: Pre-loaded DataFrame
        """
        if filepath:
            self.data = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        elif data is not None:
            self.data = data.copy()
        else:
            raise ValueError("Must provide either filepath or data")
            
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
                
        # Clean data - remove any commas from string numbers
        for col in required_cols:
            if self.data[col].dtype == 'object':
                self.data[col] = self.data[col].str.replace(',', '').astype(float)
                
        # Calculate indicators
        self._calculate_indicators()
        
    def fetch_data(self, start_date: str = None, end_date: str = None):
        """
        Fetch historical market data using yfinance.
        
        Args:
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
        """
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=500)).strftime('%Y-%m-%d')
            
        ticker = yf.Ticker(self.symbol)
        self.data = ticker.history(start=start_date, end=end_date)
        
        # Calculate indicators
        self._calculate_indicators()
        
    def _calculate_indicators(self):
        """Calculate all required technical indicators."""
        # Basic calculations
        self.data['daily_return'] = self.data['Close'].pct_change()
        self.data['daily_gain_pct'] = self.data['daily_return'] * 100
        
        # Moving averages - using EMA for 21-day, SMA for 50-day
        self.data['ema21'] = self.data['Close'].ewm(span=21, adjust=False).mean()
        self.data['sma50'] = self.data['Close'].rolling(window=50).mean()
        
        # 52-week (260 trading days) high
        self.data['high_52w'] = self.data['High'].rolling(window=260, min_periods=1).max()
        
        # Decline from 52-week high
        self.data['decline_from_high'] = (self.data['Close'] - self.data['high_52w']) / self.data['high_52w'] * 100
        
        # Intraday range calculations
        self.data['range'] = self.data['High'] - self.data['Low']
        self.data['close_position'] = np.where(
            self.data['range'] > 0,
            (self.data['Close'] - self.data['Low']) / self.data['range'],
            0.5
        )
        
        # Volume comparison
        self.data['volume_vs_prev'] = self.data['Volume'] / self.data['Volume'].shift(1)
        self.data['volume_up'] = self.data['Volume'] > self.data['Volume'].shift(1)
        
        # Spread calculation for downside reversal
        self.data['spread'] = ((self.data['High'] - self.data['Low']) / self.data['Low']) * 100
        self.data['avg_spread_50d'] = self.data['spread'].rolling(window=50).mean()
        
        # Mark 13-week (65 trading days) highs and lows
        self.data['high_13w'] = self.data['High'].rolling(window=65, min_periods=1).max()
        self.data['low_13w'] = self.data['Low'].rolling(window=65, min_periods=1).min()
        self.data['is_new_high'] = self.data['High'] >= self.data['high_13w']
        self.data['is_new_low'] = self.data['Low'] <= self.data['low_13w']
        
        # Intraday gain for stalling detection
        self.data['intraday_gain'] = ((self.data['High'] - self.data['Open']) / self.data['Open']) * 100
        self.data['give_back'] = ((self.data['High'] - self.data['Close']) / self.data['High']) * 100
        
    def check_market_correction(self, idx: int) -> bool:
        """
        Check if market has declined 10% from 52-week high.
        
        Args:
            idx: Current index in data
            
        Returns:
            True if market is in correction (down 10%+ from high)
        """
        current = self.data.iloc[idx]
        
        # Update 52-week high tracking
        if current['High'] >= current['high_52w']:
            self.recent_high = current['High']
            self.recent_high_date = current.name
            
        # Check if we're down 10% or more
        if current['decline_from_high'] <= -10:
            if not self.market_in_correction:
                self.market_in_correction = True
            return True
        else:
            self.market_in_correction = False
            return False
            
    def detect_rally_start(self, idx: int) -> bool:
        """
        Detect if a new rally attempt is starting.
        
        Args:
            idx: Current index in data
            
        Returns:
            True if rally is starting
        """
        if idx < 5:
            return False
            
        current = self.data.iloc[idx]
        
        # Only look for rally start if we're in correction
        if not self.check_market_correction(idx):
            return False
            
        # Look for a potential low
        recent_lows = self.data.iloc[idx-4:idx+1]['Low']

        if current['Low'] == recent_lows.min():
            # Only set new rally start if:
            # 1. No existing rally, OR
            # 2. This is a LOWER low than existing rally (establishes new low point)
            if self.rally_start_date is None or current['Low'] < self.rally_low:
                self.rally_start_date = current.name
                self.rally_low = current['Low']
                self.rally_low_idx = idx
                return True

        return False
        
    def check_follow_through_day(self, idx: int) -> Optional[Signal]:
        """
        Check if current day qualifies as a Follow-Through Day.
        
        Args:
            idx: Current index in data
            
        Returns:
            FTD signal if detected, None otherwise
        """
        if not self.rally_start_date:
            return None
            
        current = self.data.iloc[idx]
        prev = self.data.iloc[idx-1]

        # Check if rally low has been undercut (do this FIRST, regardless of FTD window)
        lows_since_rally = self.data.iloc[self.rally_low_idx:idx+1]['Low']
        if lows_since_rally.min() < self.rally_low:
            self.rally_start_date = None
            return None

        # Check if we're in valid FTD window (days 4-25 of rally)
        days_since_rally = idx - self.rally_low_idx
        if days_since_rally < 4 or days_since_rally > 25:
            return None

        # Volume requirement
        if not current['volume_up']:
            return None
            
        # Check for 1.2% gain
        if current['daily_gain_pct'] >= 1.2:
            signal_type = SignalType.B1 if not self.ftd_date else SignalType.B2
            
            signal = Signal(
                date=current.name,
                signal_type=signal_type,
                price=current['Close'],
                description=f"{signal_type.value} (+{current['daily_gain_pct']:.2f}%, Vol: {current['volume_vs_prev']:.1f}x)",
                affects_exposure=True,
                exposure_change=1 if signal_type == SignalType.B1 else 0
            )
            
            if signal_type == SignalType.B1:
                self.ftd_date = current.name
                self.ftd_close = current['Close']
                self.buy_switch = True
                
            return signal
            
        return None
        
    def check_moving_average_signals(self, idx: int) -> List[Signal]:
        """Check for all moving average related signals."""
        signals = []
        current = self.data.iloc[idx]
        prev = self.data.iloc[idx-1] if idx > 0 else None
        
        # Skip if moving averages not calculated yet
        if pd.isna(current['ema21']) or pd.isna(current['sma50']):
            return signals
            
        # 21-day EMA Buy Signals (B3, B4, B5)
        if self.buy_switch:
            # B3: Low Above 21-day EMA
            if current['Low'] >= current['ema21']:
                # Check if we need S5 reset
                can_trigger_b3 = True
                if self.last_s5_s6_s7_s8_date and self.last_b3_b4_b5_date:
                    if self.last_s5_s6_s7_s8_date > self.last_b3_b4_b5_date:
                        can_trigger_b3 = False
                elif self.last_s5_s6_s7_s8_date and not self.last_b3_b4_b5_date:
                    can_trigger_b3 = False
                    
                if can_trigger_b3:
                    signals.append(Signal(
                        date=current.name,
                        signal_type=SignalType.B3,
                        price=current['Close'],
                        description="Low Above 21-day EMA",
                        affects_exposure=True,
                        exposure_change=1
                    ))
                    self.last_b3_b4_b5_date = current.name
                    
            # B4: Trending Above 21-day EMA (3 consecutive days)
            if idx >= 2:
                days_above = 0
                for i in range(3):
                    if self.data.iloc[idx-i]['Low'] > self.data.iloc[idx-i]['ema21']:
                        days_above += 1
                        
                if days_above >= 3 and self.last_b3_b4_b5_date:
                    # Check if index closes up/flat on 3rd day
                    if idx > 0 and current['Close'] >= prev['Close']:
                        signals.append(Signal(
                            date=current.name,
                            signal_type=SignalType.B4,
                            price=current['Close'],
                            description="Trending Above 21-day EMA (3 days)",
                            affects_exposure=True,
                            exposure_change=1
                        ))
                        self.last_b3_b4_b5_date = current.name
                        
            # B5: Living Above 21-day EMA (10 days and every 5th day after)
            if idx >= 9:
                days_above = 0
                start_idx = max(0, idx - 60)  # Look back up to 60 days
                
                # Count consecutive days from most recent break
                for i in range(start_idx, idx + 1):
                    if self.data.iloc[i]['Low'] > self.data.iloc[i]['ema21']:
                        days_above += 1
                    else:
                        days_above = 0  # Reset count
                        
                if days_above == 10 or (days_above > 10 and (days_above - 10) % 5 == 0):
                    if self.last_b3_b4_b5_date:
                        signals.append(Signal(
                            date=current.name,
                            signal_type=SignalType.B5,
                            price=current['Close'],
                            description=f"Living Above 21-day EMA ({days_above} days)",
                            affects_exposure=True,
                            exposure_change=1
                        ))
                        self.last_b3_b4_b5_date = current.name
                    
        # 21-day EMA Sell Signals
        # S5: Break Below 21-day EMA
        if prev is not None and prev['Close'] >= prev['ema21'] and current['Close'] < current['ema21'] * 0.998:
            signals.append(Signal(
                date=current.name,
                signal_type=SignalType.S5,
                price=current['Close'],
                description="Break Below 21-day EMA",
                affects_exposure=True,
                exposure_change=-1
            ))
            self.last_s5_s6_s7_s8_date = current.name
            
        # 50-day MA Signals
        # B6: Low Above 50-day MA
        if self.buy_switch and current['Low'] >= current['sma50']:
            # Check if we need S9 reset
            can_trigger_b6 = True
            if self.last_s9_date and self.last_b6_date:
                if self.last_s9_date > self.last_b6_date:
                    can_trigger_b6 = False
            elif self.last_s9_date and not self.last_b6_date:
                can_trigger_b6 = False
                
            if can_trigger_b6:
                signals.append(Signal(
                    date=current.name,
                    signal_type=SignalType.B6,
                    price=current['Close'],
                    description="Low Above 50-day MA",
                    affects_exposure=True,
                    exposure_change=1
                ))
                self.last_b6_date = current.name
                
        # S9: Break Below 50-day MA
        if prev is not None and prev['Close'] >= prev['sma50'] and current['Close'] < current['sma50']:
            # Check shakeout exception
            is_shakeout = (current['close_position'] > 0.5 and 
                         current['Close'] > current['sma50'] * 0.99)
            
            if not is_shakeout:
                signals.append(Signal(
                    date=current.name,
                    signal_type=SignalType.S9,
                    price=current['Close'],
                    description="Break Below 50-day MA",
                    affects_exposure=True,
                    exposure_change=-1
                ))
                self.last_s9_date = current.name
                
        return signals
        
    def check_strength_weakness_signals(self, idx: int) -> List[Signal]:
        """Check for strength and weakness signals."""
        signals = []
        current = self.data.iloc[idx]
        
        # B7: Accumulation Day
        if self.buy_switch and idx > 0:
            # 1.2% threshold for accumulation day
            if (current['daily_gain_pct'] >= 1.2 and
                current['close_position'] > 0.75 and
                not pd.isna(current['ema21']) and
                current['Close'] > current['ema21'] and
                current['volume_up']):
                
                signals.append(Signal(
                    date=current.name,
                    signal_type=SignalType.B7,
                    price=current['Close'],
                    description=f"Accumulation Day (+{current['daily_gain_pct']:.2f}%)",
                    affects_exposure=True,
                    exposure_change=1
                ))
                
        # S10: Bad Break
        if (current['daily_gain_pct'] <= -2.25 or 
            (current['close_position'] < 0.25 and 
             not pd.isna(current['sma50']) and
             (current['Close'] < current['sma50'] or 
              current['High'] < current.get('ema21', float('inf'))))):
            
            signals.append(Signal(
                date=current.name,
                signal_type=SignalType.S10,
                price=current['Close'],
                description=f"Bad Break ({current['daily_gain_pct']:.2f}%)",
                affects_exposure=True,
                exposure_change=-1
            ))
            
        # B8: Higher High
        if self.buy_switch and current['is_new_high']:
            signals.append(Signal(
                date=current.name,
                signal_type=SignalType.B8,
                price=current['Close'],
                description="Higher High (13-week)",
                affects_exposure=True,
                exposure_change=1
            ))
            self.marked_highs.append({
                'date': current.name,
                'high': current['High']
            })
            
        # S12: Lower Low
        if current['is_new_low']:
            signals.append(Signal(
                date=current.name,
                signal_type=SignalType.S12,
                price=current['Close'],
                description="Lower Low (13-week)",
                affects_exposure=True,
                exposure_change=-1
            ))
            self.marked_lows.append({
                'date': current.name,
                'low': current['Low']
            })
            
        return signals
        
    def check_distribution_signals(self, idx: int) -> List[Signal]:
        """Check for distribution-related signals."""
        signals = []
        current = self.data.iloc[idx]
        prev = self.data.iloc[idx-1] if idx > 0 else None

        if prev is None:
            return signals
            
        # Check for distribution day
        is_distribution = (current['daily_gain_pct'] <= -0.2 and current['volume_up'])
        
        # Check for stall day
        is_stall = (current['volume_up'] and
                   current['close_position'] < 0.5 and
                   current['give_back'] > 30 and
                   current['intraday_gain'] > 30)
        
        if is_distribution or is_stall:
            dist_type = 'stall' if is_stall else 'distribution'
            self.distribution_days.append(DistributionDay(
                date=current.name,
                type=dist_type,
                low=current['Low'],
                close=current['Close'],
                loss_percent=current['daily_gain_pct']
            ))
            
        # Remove old distribution days and check 5% rally rule
        self._update_distribution_days(idx)
        
        # Count active distribution days
        active_dist = [d for d in self.distribution_days if d.removed_date is None]
        self.active_distribution_count = len(active_dist)
        
        # Check for distribution signals
        if self.active_distribution_count == 5:
            signals.append(Signal(
                date=current.name,
                signal_type=SignalType.S3,
                price=current['Close'],
                description=f"Full Distribution minus one ({self.active_distribution_count} days)",
                affects_exposure=True,
                exposure_change=-1
            ))
            
        elif self.active_distribution_count >= 6:
            signals.append(Signal(
                date=current.name,
                signal_type=SignalType.S4,
                price=current['Close'],
                description=f"Full Distribution ({self.active_distribution_count} days)",
                affects_exposure=True,
                exposure_change=-1
            ))
            self.buy_switch = False
            
        # B10: Distribution Day Fall Off
        # This happens automatically when distribution days are removed
        
        # S13: Distribution Cluster
        recent_window = current.name - pd.Timedelta(days=11)
        recent_dist = [d for d in active_dist if d.date > recent_window]
        
        if len(recent_dist) >= 4:
            signals.append(Signal(
                date=current.name,
                signal_type=SignalType.S13,
                price=current['Close'],
                description=f"Distribution Cluster ({len(recent_dist)} in 8 days)",
                affects_exposure=True,
                exposure_change=-1
            ))
                
        return signals
        
    def _update_distribution_days(self, idx: int):
        """Update distribution days - remove old ones and check 5% rally rule."""
        current = self.data.iloc[idx]
        current_date = current.name
        
        for dist_day in self.distribution_days:
            if dist_day.removed_date is not None:
                continue
                
            # Check 25-day rule
            days_old = (current_date - dist_day.date).days
            if days_old > 35:  # ~25 trading days
                dist_day.removed_date = current_date
                dist_day.removal_reason = "25-day rule"
                continue
                
            # Check 5% rally from low
            percent_from_low = ((current['High'] - dist_day.low) / dist_day.low) * 100
            if percent_from_low >= 5.0:
                dist_day.removed_date = current_date
                dist_day.removal_reason = f"5% rally from low ({percent_from_low:.2f}%)"
                continue
                
            # Check 6% rally from close
            percent_from_close = ((current['Close'] - dist_day.close) / dist_day.close) * 100
            if percent_from_close >= 6.0:
                dist_day.removed_date = current_date
                dist_day.removal_reason = f"6% rally from close ({percent_from_close:.2f}%)"
                
    def check_reversal_signals(self, idx: int) -> List[Signal]:
        """Check for downside reversal signals."""
        signals = []
        current = self.data.iloc[idx]
        
        # S11: Downside Reversal Day
        if (current['is_new_high'] and 
            current['close_position'] < 0.25 and
            current['daily_gain_pct'] < 0):
            
            # Check spread requirement
            spread_threshold = 1.75
            if not pd.isna(current['avg_spread_50d']) and current['avg_spread_50d'] < 0.75:
                spread_threshold = 1.0
                
            if current['spread'] >= spread_threshold:
                signals.append(Signal(
                    date=current.name,
                    signal_type=SignalType.S11,
                    price=current['Close'],
                    description=f"Downside Reversal Day (spread: {current['spread']:.2f}%)",
                    affects_exposure=True,
                    exposure_change=-1
                ))
                
        # B9: Downside Reversal Buyback
        if len(self.marked_highs) > 0:
            for marked in self.marked_highs[-3:]:
                days_since = (current.name - marked['date']).days
                
                if 0 < days_since <= 2 and current['Close'] > marked['high']:
                    signals.append(Signal(
                        date=current.name,
                        signal_type=SignalType.B9,
                        price=current['Close'],
                        description="Downside Reversal Buyback",
                        affects_exposure=True,
                        exposure_change=1
                    ))
                    break
                    
        return signals
        
    def check_failed_rally(self, idx: int) -> Optional[Signal]:
        """Check for failed rally attempt (S1, S2)."""
        if not self.rally_start_date or not self.ftd_date:
            return None
            
        current = self.data.iloc[idx]
        
        # S1: Follow-Through Day Undercut
        if self.ftd_date and current['Close'] < self.rally_low:
            signal = Signal(
                date=current.name,
                signal_type=SignalType.S1,
                price=current['Close'],
                description="Follow-Through Day Undercut",
                affects_exposure=True,
                exposure_change=-1
            )
            return signal
            
        # S2: Failed Rally Attempt
        if current['Low'] < self.rally_low:
            is_major_low = not self.buy_switch
            
            if is_major_low:
                signal = Signal(
                    date=current.name,
                    signal_type=SignalType.S2,
                    price=current['Close'],
                    description="Failed Rally - Major Low",
                    affects_exposure=True,
                    exposure_change=-self.market_exposure
                )
                self.buy_switch = False
                self.market_exposure = 0
            else:
                signal = Signal(
                    date=current.name,
                    signal_type=SignalType.S2,
                    price=current['Close'],
                    description="Failed Rally - Minor Low",
                    affects_exposure=True,
                    exposure_change=-2
                )
                
            self.rally_start_date = None
            return signal
            
        return None
        
    def apply_restraint_rule(self, signals: List[Signal]) -> List[Signal]:
        """Apply restraint rule to limit exposure early in rally."""
        if not self.restraint_active:
            return signals
            
        # Check if restraint should be lifted
        if self.ftd_date and self.ftd_close:
            current = self.data.iloc[-1]
            progress = ((current['Close'] - self.ftd_close) / self.ftd_close) * 100
            
            # Lift restraint if 1.2% progress or B4 triggered
            has_b4 = any(s.signal_type == SignalType.B4 for s in self.signals)
            
            if progress >= 1.2 or has_b4:
                self.restraint_active = False
                return signals
                
        # Apply restraint - limit exposure to +2
        filtered_signals = []
        for signal in signals:
            if signal.affects_exposure and signal.exposure_change > 0:
                new_exposure = self.market_exposure + signal.exposure_change
                if new_exposure <= 2:
                    filtered_signals.append(signal)
            else:
                filtered_signals.append(signal)
                
        return filtered_signals
        
    def update_market_exposure(self, signal: Signal):
        """Update market exposure based on signal."""
        if signal.affects_exposure:
            self.market_exposure += signal.exposure_change
            self.market_exposure = max(0, min(6, self.market_exposure))
            
    def get_position_allocation(self) -> float:
        """Get recommended position allocation based on market exposure."""
        allocation_map = {
            0: 0.0,
            1: 0.30,
            2: 0.55,
            3: 0.75,
            4: 0.90,
            5: 1.00,
            6: 1.00
        }
        
        return allocation_map.get(self.market_exposure, 0.0)
        
    def analyze_market(self):
        """Run complete market analysis and generate all signals."""
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() or fetch_data() first.")
            
        # Reset state
        self.signals = []
        self.market_exposure = 0
        self.buy_switch = False
        self.distribution_days = []

        # Reset rally tracking
        self.rally_start_date = None
        self.rally_low = None
        self.rally_low_idx = None
        self.ftd_date = None
        self.ftd_close = None
        
        # Process each day
        for idx in range(260, len(self.data)):  # Start after 52-week high can be calculated
            daily_signals = []
            
            # Check for market correction and rally start
            if not self.rally_start_date:
                self.detect_rally_start(idx)

            # Check for Follow-Through Day
            if self.rally_start_date:
                ftd_signal = self.check_follow_through_day(idx)
                if ftd_signal:
                    daily_signals.append(ftd_signal)
                    if self.market_exposure == 0:
                        self.restraint_active = True

                # If rally was reset by check_follow_through_day, check if current day is new rally start
                if not self.rally_start_date:
                    self.detect_rally_start(idx)
                        
            # Check for failed rally
            failed_rally = self.check_failed_rally(idx)
            if failed_rally:
                daily_signals.append(failed_rally)
                
            # Only check other signals if we have valid data
            if not pd.isna(self.data.iloc[idx]['ema21']):
                # Check moving average signals
                ma_signals = self.check_moving_average_signals(idx)
                daily_signals.extend(ma_signals)
                
                # Check strength/weakness signals
                sw_signals = self.check_strength_weakness_signals(idx)
                daily_signals.extend(sw_signals)
                
                # Check distribution signals
                dist_signals = self.check_distribution_signals(idx)
                daily_signals.extend(dist_signals)
                
                # Check reversal signals
                rev_signals = self.check_reversal_signals(idx)
                daily_signals.extend(rev_signals)
                
            # Apply restraint rule if active
            if self.restraint_active:
                daily_signals = self.apply_restraint_rule(daily_signals)
                
            # Update market exposure and save signals
            for signal in daily_signals:
                self.update_market_exposure(signal)
                self.signals.append(signal)
                
    def get_daily_summary(self, date: str = None) -> Dict:
        """
        Get daily summary with exposure level and distribution count.
        
        Args:
            date: Specific date to check (YYYY-MM-DD format). If None, uses latest date.
            
        Returns:
            Dictionary with daily summary information
        """
        if not self.signals:
            self.analyze_market()
            
        # Use latest date if not specified
        if date is None:
            target_date = self.data.index[-1]
        else:
            target_date = pd.Timestamp(date)

        # Normalize to tz-naive for comparison
        target_date = pd.Timestamp(target_date).tz_localize(None)

        # Find state as of target date
        exposure = 0
        buy_switch = False
        signals_to_date = []

        for signal in self.signals:
            signal_date = pd.Timestamp(signal.date).tz_localize(None)
            if signal_date <= target_date:
                signals_to_date.append(signal)
                if signal.affects_exposure:
                    exposure += signal.exposure_change
                    exposure = max(0, min(6, exposure))
                    
                if signal.signal_type == SignalType.B1:
                    buy_switch = True
                elif signal.signal_type == SignalType.S4:
                    buy_switch = False
                    
        # Count active distribution days as of target date
        active_dist = 0
        dist_details = []

        for dist_day in self.distribution_days:
            dist_date = pd.Timestamp(dist_day.date).tz_localize(None)
            if dist_date <= target_date:
                removed_date = pd.Timestamp(dist_day.removed_date).tz_localize(None) if dist_day.removed_date else None
                if removed_date is None or removed_date > target_date:
                    active_dist += 1
                    dist_details.append({
                        'date': dist_day.date.strftime('%Y-%m-%d'),
                        'type': dist_day.type,
                        'loss': f"{dist_day.loss_percent:.2f}%"
                    })
                    
        # Get current price data - normalize index for comparison
        data_index_normalized = self.data.index.tz_localize(None)
        if target_date in data_index_normalized:
            idx_pos = list(data_index_normalized).index(target_date)
            current_data = self.data.iloc[idx_pos]
        else:
            # Find closest date
            closest_idx = data_index_normalized.get_indexer([target_date], method='ffill')[0]
            if closest_idx >= 0:
                current_data = self.data.iloc[closest_idx]
            else:
                current_data = self.data.iloc[-1]
            
        # Calculate position allocation
        allocation = self.get_position_allocation_for_exposure(exposure)
        
        return {
            'date': target_date.strftime('%Y-%m-%d'),
            'close': round(current_data['Close'], 2),
            'daily_change': f"{current_data['daily_gain_pct']:.2f}%",
            'buy_switch': 'ON' if buy_switch else 'OFF',
            'market_exposure': exposure,
            'position_allocation': f"{allocation * 100:.0f}%",
            'distribution_count': active_dist,
            'distribution_details': dist_details,
            'above_21ema': current_data['Close'] > current_data['ema21'],
            'above_50ma': current_data['Close'] > current_data['sma50'],
            'recent_signals': [
                {
                    'date': s.date.strftime('%Y-%m-%d'),
                    'signal': s.signal_type.name,
                    'description': s.description
                }
                for s in signals_to_date[-5:]  # Last 5 signals
            ]
        }
        
    def get_position_allocation_for_exposure(self, exposure: int) -> float:
        """Get position allocation for a given exposure level."""
        allocation_map = {
            0: 0.0,
            1: 0.30,
            2: 0.55,
            3: 0.75,
            4: 0.90,
            5: 1.00,
            6: 1.00
        }
        return allocation_map.get(exposure, 0.0)
        
    def print_daily_summary(self, date: str = None):
        """Print a formatted daily summary."""
        summary = self.get_daily_summary(date)
        
        print("\n" + "="*60)
        print(f"MARKET SCHOOL RULES - DAILY SUMMARY")
        print("="*60)
        print(f"Date: {summary['date']}")
        print(f"NASDAQ Close: {summary['close']} ({summary['daily_change']})")
        print(f"Buy Switch: {summary['buy_switch']}")
        print(f"Market Exposure: {summary['market_exposure']}/6")
        print(f"Position Allocation: {summary['position_allocation']}")
        print(f"Distribution Days: {summary['distribution_count']}")
        
        if summary['distribution_details']:
            print("\nActive Distribution Days:")
            for dist in summary['distribution_details']:
                print(f"  {dist['date']} - {dist['type']} ({dist['loss']})")
                
        print(f"\nMarket Position:")
        print(f"  Above 21-day EMA: {'Yes' if summary['above_21ema'] else 'No'}")
        print(f"  Above 50-day MA: {'Yes' if summary['above_50ma'] else 'No'}")
        
        if summary['recent_signals']:
            print("\nRecent Signals:")
            for sig in summary['recent_signals']:
                print(f"  {sig['date']} - {sig['signal']}: {sig['description']}")
                
        print("="*60)
        
    def get_current_status(self) -> Dict:
        """Get current market status and recommendations."""
        return self.get_daily_summary()
        
    def generate_report(self, output_file: str = None) -> pd.DataFrame:
        """Generate a detailed report of all signals."""
        if not self.signals:
            self.analyze_market()
            
        # Convert signals to DataFrame
        signal_data = []
        running_exposure = 0
        
        for signal in self.signals:
            running_exposure += signal.exposure_change if signal.affects_exposure else 0
            running_exposure = max(0, min(6, running_exposure))
            
            signal_data.append({
                'Date': signal.date,
                'Signal': signal.signal_type.name,
                'Type': 'BUY' if signal.signal_type.name.startswith('B') else 'SELL',
                'Description': signal.description,
                'Price': round(signal.price, 2),
                'Exposure_Change': signal.exposure_change,
                'Market_Exposure': running_exposure,
                'Allocation': f"{self.get_position_allocation_for_exposure(running_exposure) * 100:.0f}%"
            })
            
        df = pd.DataFrame(signal_data)
        
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Report saved to {output_file}")
            
        return df
        
    def plot_signals(self, start_date=None, end_date=None, save_file=None):
        """
        Plot market data with signals.
        
        Args:
            start_date: Start date for plot (YYYY-MM-DD)
            end_date: End date for plot (YYYY-MM-DD)
            save_file: Filename to save plot
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            print("Matplotlib not installed. Skipping plot generation.")
            print("To enable plotting, install matplotlib: pip install matplotlib")
            return
        
        # Filter data by date range
        plot_data = self.data.copy()
        if start_date:
            plot_data = plot_data[plot_data.index >= start_date]
        if end_date:
            plot_data = plot_data[plot_data.index <= end_date]
            
        # Create figure with subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12), 
                                                  gridspec_kw={'height_ratios': [3, 1, 1, 1]})
        
        # Plot 1: Price and moving averages
        ax1.plot(plot_data.index, plot_data['Close'], label='Close', color='black', linewidth=2)
        ax1.plot(plot_data.index, plot_data['ema21'], label='21-day EMA', color='blue', alpha=0.7)
        ax1.plot(plot_data.index, plot_data['sma50'], label='50-day MA', color='red', alpha=0.7)
        
        # Add buy/sell signals
        buy_signals = [s for s in self.signals if s.signal_type.name.startswith('B')]
        sell_signals = [s for s in self.signals if s.signal_type.name.startswith('S')]
        
        # Plot buy signals
        for signal in buy_signals:
            if start_date and signal.date < pd.Timestamp(start_date):
                continue
            if end_date and signal.date > pd.Timestamp(end_date):
                continue
            if signal.date in plot_data.index:
                ax1.scatter(signal.date, signal.price, color='green', marker='^', 
                           s=100, zorder=5)
                if signal.signal_type in [SignalType.B1, SignalType.B3, SignalType.B4, SignalType.B6]:
                    ax1.annotate(signal.signal_type.name, 
                                xy=(signal.date, signal.price),
                                xytext=(0, -20),
                                textcoords='offset points',
                                fontsize=8,
                                color='green',
                                ha='center')
                            
        # Plot sell signals
        for signal in sell_signals:
            if start_date and signal.date < pd.Timestamp(start_date):
                continue
            if end_date and signal.date > pd.Timestamp(end_date):
                continue
            if signal.date in plot_data.index:
                ax1.scatter(signal.date, signal.price, color='red', marker='v', 
                           s=100, zorder=5)
                if signal.signal_type in [SignalType.S3, SignalType.S4, SignalType.S5, SignalType.S9]:
                    ax1.annotate(signal.signal_type.name,
                                xy=(signal.date, signal.price),
                                xytext=(0, 20),
                                textcoords='offset points',
                                fontsize=8,
                                color='red',
                                ha='center')
                            
        # Highlight FTD
        if self.ftd_date and self.ftd_date in plot_data.index:
            ax1.axvline(x=self.ftd_date, color='green', linestyle='--', alpha=0.5, label='FTD')
            
        ax1.set_ylabel('Price')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'{self.symbol} - IBD Market School Rules Analysis')
        
        # Plot 2: Volume
        colors = ['green' if row['daily_return'] > 0 else 'red' for idx, row in plot_data.iterrows()]
        ax2.bar(plot_data.index, plot_data['Volume'], alpha=0.5, color=colors)
        ax2.set_ylabel('Volume')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Distribution days
        dist_data = []
        for idx, row in plot_data.iterrows():
            count = 0
            for dist in self.distribution_days:
                if dist.date <= idx and (dist.removed_date is None or dist.removed_date > idx):
                    count += 1
            dist_data.append(count)
            
        ax3.plot(plot_data.index, dist_data, label='Distribution Days', 
                color='red', linewidth=2)
        ax3.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Warning (S3)')
        ax3.axhline(y=6, color='red', linestyle='--', alpha=0.5, label='Critical (S4)')
        ax3.set_ylabel('Dist. Days')
        ax3.set_ylim(-0.5, 8)
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Market exposure
        exposure_data = []
        current_exposure = 0
        
        for idx, row in plot_data.iterrows():
            # Find signals for this date
            day_signals = [s for s in self.signals if s.date.date() == idx.date()]
            for signal in day_signals:
                if signal.affects_exposure:
                    current_exposure += signal.exposure_change
                    current_exposure = max(0, min(6, current_exposure))
            exposure_data.append(current_exposure)
            
        ax4.plot(plot_data.index, exposure_data, label='Market Exposure', 
                color='purple', linewidth=2)
        ax4.fill_between(plot_data.index, 0, exposure_data, alpha=0.3, color='purple')
        ax4.axhline(y=2, color='orange', linestyle='--', alpha=0.5, label='Restraint Level')
        ax4.set_ylabel('Exposure')
        ax4.set_ylim(-0.5, 6.5)
        ax4.legend(loc='best', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
        plt.tight_layout()
        
        if save_file:
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_file}")
            
        plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize the system
    market_rules = MarketSchoolRules(symbol="^IXIC")
    
    # Load data
    # market_rules.load_data("IXIC_price_data.csv")
    
    # Run analysis
    # market_rules.analyze_market()
    
    # Print daily summary
    # market_rules.print_daily_summary()
    
    # Generate report
    # report = market_rules.generate_report("market_signals_report.csv")
    
    # Plot signals
    # market_rules.plot_signals(start_date="2024-12-01", save_file="market_analysis.png")
    
    print("Market School Rules system ready!")