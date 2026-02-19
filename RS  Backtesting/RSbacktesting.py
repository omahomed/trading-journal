"""
RS (Relative Strength) Backtesting System
==========================================
Buy: RS reclaims both MAs (crosses above both)
Sell: RS closes below MA(s) - configurable

Based on MO RS Pine Script indicator

Supports:
- yfinance for data download
- CSV files from your own scraper
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Optional yfinance import
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Note: yfinance not installed. Use CSV mode or install with: pip install yfinance")


class RSBacktester:
    """
    Backtests a Relative Strength strategy where:
    - RS = Stock Price / Index Price
    - Buy when RS reclaims both MAs
    - Sell when RS closes below MA(s)
    """
    
    def __init__(
        self,
        index_symbol: str = "SPY",
        ma1_period: int = 21,
        ma2_period: int = 50,
        ma_type: str = "SMA",
        sell_mode: str = "either",  # "either" or "both"
        timeframe: str = "daily",
        data_source: str = "yfinance",  # "yfinance" or "csv"
        csv_data_dir: str = "./data"   # Directory containing CSV files
    ):
        """
        Parameters:
        -----------
        index_symbol : str
            Symbol for the index (default: SPY)
        ma1_period : int
            Period for the first (shorter) moving average
        ma2_period : int
            Period for the second (longer) moving average
        ma_type : str
            Type of moving average: SMA, EMA, WMA, RMA
        sell_mode : str
            "either" = sell when RS closes below either MA
            "both" = sell when RS closes below both MAs
        timeframe : str
            "daily", "weekly", or "monthly"
        data_source : str
            "yfinance" = download from Yahoo Finance
            "csv" = load from CSV files in csv_data_dir
        csv_data_dir : str
            Directory containing CSV files (for csv mode)
            Expected format: {symbol}.csv with Date, Open, High, Low, Close, Volume columns
        """
        self.index_symbol = index_symbol
        self.ma1_period = ma1_period
        self.ma2_period = ma2_period
        self.ma_type = ma_type.upper()
        self.sell_mode = sell_mode
        self.timeframe = timeframe
        self.data_source = data_source
        self.csv_data_dir = Path(csv_data_dir)
        
        # Cache for index data (avoid re-fetching)
        self._index_cache = None
        
        # Set default MA periods based on timeframe if not specified
        self._set_default_periods()
        
    def _set_default_periods(self):
        """Set default MA periods based on timeframe (matching Pine Script)"""
        defaults = {
            "daily": (21, 50),
            "weekly": (8, 21),
            "monthly": (10, 24)
        }
        if self.timeframe in defaults:
            if self.ma1_period == 21 and self.ma2_period == 50:
                self.ma1_period, self.ma2_period = defaults[self.timeframe]
    
    def calculate_ma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate moving average based on type"""
        if self.ma_type == "SMA":
            return series.rolling(window=period).mean()
        elif self.ma_type == "EMA":
            return series.ewm(span=period, adjust=False).mean()
        elif self.ma_type == "WMA":
            weights = np.arange(1, period + 1)
            return series.rolling(window=period).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
        elif self.ma_type == "RMA":
            return series.ewm(alpha=1/period, adjust=False).mean()
        else:
            return series.rolling(window=period).mean()
    
    def resample_to_timeframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample daily data to weekly or monthly if needed"""
        if self.timeframe == "daily":
            return df
        elif self.timeframe == "weekly":
            return df.resample('W').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        elif self.timeframe == "monthly":
            return df.resample('ME').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        return df
    
    def fetch_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Fetch stock and index data from yfinance or CSV"""
        
        if self.data_source == "csv":
            return self._fetch_from_csv(symbol, start_date, end_date)
        else:
            return self._fetch_from_yfinance(symbol, start_date, end_date)
    
    def _fetch_from_csv(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load data from CSV files"""
        try:
            # Load stock data
            stock_file = self.csv_data_dir / f"{symbol}.csv"
            if not stock_file.exists():
                # Try lowercase
                stock_file = self.csv_data_dir / f"{symbol.lower()}.csv"
            
            if not stock_file.exists():
                print(f"CSV file not found: {stock_file}")
                return None, None
            
            stock_data = pd.read_csv(stock_file, parse_dates=['Date'], index_col='Date')
            
            # Load index data (use cache if available)
            if self._index_cache is None:
                index_file = self.csv_data_dir / f"{self.index_symbol}.csv"
                if not index_file.exists():
                    index_file = self.csv_data_dir / f"{self.index_symbol.lower()}.csv"
                
                if not index_file.exists():
                    print(f"Index CSV file not found: {index_file}")
                    return None, None
                
                self._index_cache = pd.read_csv(index_file, parse_dates=['Date'], index_col='Date')
            
            index_data = self._index_cache.copy()
            
            # Filter by date range
            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)
            stock_data = stock_data[(stock_data.index >= start) & (stock_data.index <= end)]
            index_data = index_data[(index_data.index >= start) & (index_data.index <= end)]
            
            # Standardize column names
            stock_data = self._standardize_columns(stock_data)
            index_data = self._standardize_columns(index_data)
            
            # Resample if needed
            stock_data = self.resample_to_timeframe(stock_data)
            index_data = self.resample_to_timeframe(index_data)
            
            # Align dates
            common_dates = stock_data.index.intersection(index_data.index)
            stock_data = stock_data.loc[common_dates]
            index_data = index_data.loc[common_dates]
            
            return stock_data, index_data
            
        except Exception as e:
            print(f"Error loading CSV for {symbol}: {e}")
            return None, None
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to Open, High, Low, Close, Volume"""
        col_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'open' in col_lower:
                col_map[col] = 'Open'
            elif 'high' in col_lower:
                col_map[col] = 'High'
            elif 'low' in col_lower:
                col_map[col] = 'Low'
            elif 'close' in col_lower and 'adj' not in col_lower:
                col_map[col] = 'Close'
            elif 'adj' in col_lower and 'close' in col_lower:
                col_map[col] = 'Adj Close'
            elif 'volume' in col_lower:
                col_map[col] = 'Volume'
        
        df = df.rename(columns=col_map)
        return df
    
    def _fetch_from_yfinance(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Fetch data from Yahoo Finance"""
        if not YFINANCE_AVAILABLE:
            print("yfinance not installed. Install with: pip install yfinance")
            return None, None
        
        try:
            stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            # Use cached index data if available
            if self._index_cache is None:
                self._index_cache = yf.download(self.index_symbol, start=start_date, end=end_date, progress=False)
            
            index_data = self._index_cache.copy()
            
            if stock_data.empty or index_data.empty:
                return None, None
            
            # Handle multi-level columns from yfinance
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.get_level_values(0)
            if isinstance(index_data.columns, pd.MultiIndex):
                index_data.columns = index_data.columns.get_level_values(0)
            
            # Resample if needed
            stock_data = self.resample_to_timeframe(stock_data)
            index_data = self.resample_to_timeframe(index_data)
            
            # Align dates
            common_dates = stock_data.index.intersection(index_data.index)
            stock_data = stock_data.loc[common_dates]
            index_data = index_data.loc[common_dates]
            
            return stock_data, index_data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None, None
    
    def calculate_rs_and_signals(
        self, 
        stock_data: pd.DataFrame, 
        index_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate RS and generate buy/sell signals"""
        df = pd.DataFrame(index=stock_data.index)
        df['stock_close'] = stock_data['Close']
        df['index_close'] = index_data['Close']
        
        # Calculate RS
        df['rs'] = df['stock_close'] / df['index_close']
        
        # Calculate MAs on RS
        df['ma1'] = self.calculate_ma(df['rs'], self.ma1_period)
        df['ma2'] = self.calculate_ma(df['rs'], self.ma2_period)
        
        # RS position relative to MAs
        df['rs_above_ma1'] = df['rs'] > df['ma1']
        df['rs_above_ma2'] = df['rs'] > df['ma2']
        df['rs_above_both'] = df['rs_above_ma1'] & df['rs_above_ma2']
        
        # Previous bar conditions
        df['prev_rs_above_ma1'] = df['rs_above_ma1'].shift(1)
        df['prev_rs_above_ma2'] = df['rs_above_ma2'].shift(1)
        df['prev_rs_above_both'] = df['rs_above_both'].shift(1)
        
        # Was below at least one MA in previous bar
        df['was_below_either'] = ~df['prev_rs_above_both']
        
        # Buy Signal: RS crosses above BOTH MAs (reclaims)
        df['buy_signal'] = df['rs_above_both'] & df['was_below_either']
        
        # Sell Signal based on mode
        if self.sell_mode == "either":
            # Sell when RS closes below EITHER MA
            df['rs_below_either'] = ~df['rs_above_ma1'] | ~df['rs_above_ma2']
            df['prev_above_both'] = df['prev_rs_above_both']
            df['sell_signal'] = df['rs_below_either'] & df['prev_above_both'].fillna(False)
        else:
            # Sell when RS closes below BOTH MAs
            df['rs_below_both'] = ~df['rs_above_ma1'] & ~df['rs_above_ma2']
            df['prev_above_either'] = df['prev_rs_above_ma1'] | df['prev_rs_above_ma2']
            df['sell_signal'] = df['rs_below_both'] & df['prev_above_either'].fillna(False)
        
        return df
    
    def run_backtest(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        initial_capital: float = 10000.0
    ) -> Dict:
        """Run backtest for a single symbol"""
        
        # Fetch data
        stock_data, index_data = self.fetch_data(symbol, start_date, end_date)
        if stock_data is None:
            return {"symbol": symbol, "error": "Failed to fetch data"}
        
        # Calculate RS and signals
        df = self.calculate_rs_and_signals(stock_data, index_data)
        
        # Skip if not enough data for MAs
        min_periods = max(self.ma1_period, self.ma2_period)
        if len(df) < min_periods + 10:
            return {"symbol": symbol, "error": "Insufficient data"}
        
        # Initialize tracking variables
        trades = []
        position = 0  # 0 = cash, 1 = in position
        entry_price = 0
        entry_date = None
        shares = 0
        capital = initial_capital
        
        # Track equity curve
        equity_curve = []
        
        # Iterate through data (skip warmup period)
        valid_data = df.iloc[min_periods:].copy()
        
        for date, row in valid_data.iterrows():
            current_price = row['stock_close']
            
            # Calculate current equity
            if position == 1:
                current_equity = shares * current_price
            else:
                current_equity = capital
            equity_curve.append({'date': date, 'equity': current_equity})
            
            # Check for buy signal
            if position == 0 and row['buy_signal']:
                entry_price = current_price
                entry_date = date
                shares = capital / current_price
                position = 1
                
            # Check for sell signal
            elif position == 1 and row['sell_signal']:
                exit_price = current_price
                pnl = (exit_price - entry_price) * shares
                pnl_pct = (exit_price / entry_price - 1) * 100
                capital = shares * exit_price
                
                trades.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': date,
                    'exit_price': exit_price,
                    'shares': shares,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'rs_at_entry': df.loc[entry_date, 'rs'] if entry_date in df.index else None,
                    'rs_at_exit': row['rs']
                })
                
                position = 0
                shares = 0
        
        # Close any open position at end
        if position == 1:
            final_price = valid_data.iloc[-1]['stock_close']
            pnl = (final_price - entry_price) * shares
            pnl_pct = (final_price / entry_price - 1) * 100
            capital = shares * final_price
            
            trades.append({
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': valid_data.index[-1],
                'exit_price': final_price,
                'shares': shares,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'rs_at_entry': df.loc[entry_date, 'rs'] if entry_date in df.index else None,
                'rs_at_exit': valid_data.iloc[-1]['rs'],
                'open_trade': True
            })
        
        # Calculate metrics
        results = self._calculate_metrics(
            symbol, trades, equity_curve, initial_capital, 
            df, valid_data, stock_data
        )
        
        return results
    
    def _calculate_metrics(
        self, 
        symbol: str, 
        trades: List[Dict], 
        equity_curve: List[Dict],
        initial_capital: float,
        df: pd.DataFrame,
        valid_data: pd.DataFrame,
        stock_data: pd.DataFrame
    ) -> Dict:
        """Calculate performance metrics"""
        
        if not trades:
            return {
                "symbol": symbol,
                "total_trades": 0,
                "error": "No trades generated"
            }
        
        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve)
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        # Returns
        final_equity = equity_df.iloc[-1]['equity']
        total_return = (final_equity / initial_capital - 1) * 100
        
        # Buy and hold comparison
        first_price = stock_data['Close'].iloc[max(self.ma1_period, self.ma2_period)]
        last_price = stock_data['Close'].iloc[-1]
        buy_hold_return = (last_price / first_price - 1) * 100
        
        # Average trade metrics
        avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0.01
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
        max_drawdown = equity_df['drawdown'].min()
        
        # Average trade duration
        trades_df['duration'] = (
            pd.to_datetime(trades_df['exit_date']) - 
            pd.to_datetime(trades_df['entry_date'])
        ).dt.days
        avg_duration = trades_df['duration'].mean()
        
        # Expectancy
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
        
        return {
            "symbol": symbol,
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": round(win_rate, 2),
            "total_return_pct": round(total_return, 2),
            "buy_hold_return_pct": round(buy_hold_return, 2),
            "outperformance": round(total_return - buy_hold_return, 2),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "avg_trade_duration_days": round(avg_duration, 1),
            "expectancy": round(expectancy, 2),
            "final_equity": round(final_equity, 2),
            "trades": trades_df.to_dict('records')
        }
    
    def run_multi_symbol_backtest(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 10000.0
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """Run backtest for multiple symbols"""
        
        # Reset index cache for fresh run
        self._index_cache = None
        
        all_results = []
        
        print(f"\nRS Backtest: {self.timeframe.upper()} | MA1={self.ma1_period} | MA2={self.ma2_period} | Sell Mode={self.sell_mode}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Data Source: {self.data_source}")
        print("=" * 70)
        
        for i, symbol in enumerate(symbols):
            print(f"Processing {symbol} ({i+1}/{len(symbols)})...", end=" ")
            result = self.run_backtest(symbol, start_date, end_date, initial_capital)
            all_results.append(result)
            
            if "error" in result:
                print(f"ERROR: {result['error']}")
            else:
                print(f"Trades: {result['total_trades']}, "
                      f"Win Rate: {result['win_rate']}%, "
                      f"Return: {result['total_return_pct']}%")
        
        # Create summary DataFrame
        summary_data = []
        for r in all_results:
            if "error" not in r or r.get("total_trades", 0) > 0:
                summary_data.append({
                    "Symbol": r.get("symbol"),
                    "Trades": r.get("total_trades", 0),
                    "Win Rate %": r.get("win_rate", 0),
                    "Return %": r.get("total_return_pct", 0),
                    "B&H Return %": r.get("buy_hold_return_pct", 0),
                    "Outperform %": r.get("outperformance", 0),
                    "Profit Factor": r.get("profit_factor", 0),
                    "Max DD %": r.get("max_drawdown_pct", 0),
                    "Avg Win %": r.get("avg_win_pct", 0),
                    "Avg Loss %": r.get("avg_loss_pct", 0),
                    "Expectancy": r.get("expectancy", 0)
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        return summary_df, all_results


def print_summary_stats(summary_df: pd.DataFrame):
    """Print aggregate statistics"""
    if summary_df.empty:
        print("\nNo valid results to summarize.")
        return
    
    print("\n" + "=" * 70)
    print("AGGREGATE STATISTICS")
    print("=" * 70)
    
    valid = summary_df[summary_df['Trades'] > 0]
    
    if valid.empty:
        print("No symbols with trades.")
        return
    
    print(f"Symbols Analyzed: {len(valid)}")
    print(f"Total Trades: {valid['Trades'].sum()}")
    print(f"Average Win Rate: {valid['Win Rate %'].mean():.2f}%")
    print(f"Average Return: {valid['Return %'].mean():.2f}%")
    print(f"Average B&H Return: {valid['B&H Return %'].mean():.2f}%")
    print(f"Average Outperformance: {valid['Outperform %'].mean():.2f}%")
    print(f"% of Symbols Outperforming B&H: {(valid['Outperform %'] > 0).sum() / len(valid) * 100:.1f}%")
    print(f"Average Max Drawdown: {valid['Max DD %'].mean():.2f}%")
    print(f"Average Profit Factor: {valid['Profit Factor'].mean():.2f}")


def export_trades_to_excel(all_results: List[Dict], filename: str = "rs_trades_detail.xlsx"):
    """Export all trade details to Excel with formatting"""
    try:
        all_trades = []
        for result in all_results:
            if "trades" in result:
                for trade in result['trades']:
                    trade_copy = trade.copy()
                    trade_copy['symbol'] = result['symbol']
                    all_trades.append(trade_copy)
        
        if not all_trades:
            print("No trades to export.")
            return
        
        trades_df = pd.DataFrame(all_trades)
        
        # Reorder columns
        cols = ['symbol', 'entry_date', 'entry_price', 'exit_date', 'exit_price', 
                'pnl', 'pnl_pct', 'duration', 'rs_at_entry', 'rs_at_exit']
        cols = [c for c in cols if c in trades_df.columns]
        trades_df = trades_df[cols]
        
        trades_df.to_excel(filename, index=False)
        print(f"Trade details exported to {filename}")
        
    except ImportError:
        print("openpyxl not installed. Install with: pip install openpyxl")
        # Fallback to CSV
        trades_df.to_csv(filename.replace('.xlsx', '.csv'), index=False)
        print(f"Trade details exported to {filename.replace('.xlsx', '.csv')} (CSV fallback)")


def run_multi_timeframe_comparison(
    symbols: List[str],
    start_date: str,
    end_date: str,
    index_symbol: str = "SPY",
    data_source: str = "yfinance",
    csv_data_dir: str = "./data"
) -> pd.DataFrame:
    """Run backtests across all timeframes and compare results"""
    
    timeframe_configs = {
        "daily": {"ma1": 21, "ma2": 50},
        "weekly": {"ma1": 8, "ma2": 21},
        "monthly": {"ma1": 10, "ma2": 24}
    }
    
    all_summaries = []
    
    for tf, config in timeframe_configs.items():
        print(f"\n{'='*70}")
        print(f"TESTING {tf.upper()} TIMEFRAME")
        print(f"{'='*70}")
        
        backtester = RSBacktester(
            index_symbol=index_symbol,
            ma1_period=config['ma1'],
            ma2_period=config['ma2'],
            ma_type="SMA",
            sell_mode="either",
            timeframe=tf,
            data_source=data_source,
            csv_data_dir=csv_data_dir
        )
        
        summary_df, _ = backtester.run_multi_symbol_backtest(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        if not summary_df.empty:
            summary_df['Timeframe'] = tf.upper()
            all_summaries.append(summary_df)
    
    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        
        print("\n" + "=" * 70)
        print("MULTI-TIMEFRAME COMPARISON SUMMARY")
        print("=" * 70)
        
        for tf in ["DAILY", "WEEKLY", "MONTHLY"]:
            tf_data = combined[combined['Timeframe'] == tf]
            if not tf_data.empty and tf_data['Trades'].sum() > 0:
                print(f"\n{tf}:")
                print(f"  Avg Win Rate: {tf_data['Win Rate %'].mean():.1f}%")
                print(f"  Avg Return: {tf_data['Return %'].mean():.1f}%")
                print(f"  Avg B&H: {tf_data['B&H Return %'].mean():.1f}%")
                print(f"  Outperformance: {tf_data['Outperform %'].mean():.1f}%")
        
        return combined
    
    return pd.DataFrame()


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    
    # ==========================================
    # CONFIGURATION
    # ==========================================
    
    # Stock symbols to test
    SYMBOLS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "TSLA", "AMD", "NFLX", "CRM"
    ]
    
    # Date range
    START_DATE = "2020-01-01"
    END_DATE = "2024-12-31"
    
    # ==========================================
    # OPTION 1: Using yfinance (downloads data)
    # ==========================================
    
    backtester = RSBacktester(
        index_symbol="SPY",
        ma1_period=21,           # Orange MA in your indicator
        ma2_period=50,           # Blue MA in your indicator
        ma_type="SMA",           # SMA, EMA, WMA, or RMA
        sell_mode="either",      # "either" = sell on first MA break, "both" = sell when below both
        timeframe="daily",       # "daily", "weekly", or "monthly"
        data_source="yfinance",  # Use Yahoo Finance
    )
    
    # ==========================================
    # OPTION 2: Using CSV files from your scraper
    # ==========================================
    
    # Uncomment below to use CSV mode:
    # 
    # backtester = RSBacktester(
    #     index_symbol="SPY",
    #     ma1_period=21,
    #     ma2_period=50,
    #     ma_type="SMA",
    #     sell_mode="either",
    #     timeframe="daily",
    #     data_source="csv",
    #     csv_data_dir="./data"  # Directory with your CSV files
    # )
    #
    # Expected CSV format:
    # Date,Open,High,Low,Close,Volume
    # 2020-01-02,74.06,75.15,73.80,75.09,135480400
    # ...
    
    # ==========================================
    # RUN BACKTEST
    # ==========================================
    
    summary_df, all_results = backtester.run_multi_symbol_backtest(
        symbols=SYMBOLS,
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital=10000.0
    )
    
    # ==========================================
    # DISPLAY RESULTS
    # ==========================================
    
    print("\n" + "=" * 70)
    print("RESULTS BY SYMBOL")
    print("=" * 70)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    
    # Print aggregate stats
    print_summary_stats(summary_df)
    
    # Export to CSV
    summary_df.to_csv("rs_backtest_results.csv", index=False)
    print("\nResults saved to rs_backtest_results.csv")
    
    # ==========================================
    # EXPORT DETAILED TRADES (OPTIONAL)
    # ==========================================
    
    # Uncomment to export all trades to a separate file:
    # 
    # all_trades = []
    # for result in all_results:
    #     if "trades" in result:
    #         for trade in result['trades']:
    #             trade['symbol'] = result['symbol']
    #             all_trades.append(trade)
    # 
    # if all_trades:
    #     trades_df = pd.DataFrame(all_trades)
    #     trades_df.to_csv("rs_backtest_trades.csv", index=False)
    #     print("Trade details saved to rs_backtest_trades.csv")
    
    # ==========================================
    # PRINT TRADES FOR SPECIFIC SYMBOL (OPTIONAL)
    # ==========================================
    
    # Uncomment to see trade details for a specific symbol:
    # 
    # for result in all_results:
    #     if result.get("symbol") == "AAPL" and "trades" in result:
    #         print(f"\n{result['symbol']} Trade Details:")
    #         trades_df = pd.DataFrame(result['trades'])
    #         print(trades_df[['entry_date', 'entry_price', 'exit_date', 'exit_price', 'pnl_pct']].to_string(index=False))