"""
MO Money — FastAPI backend
Wraps existing db_layer.py, market_school_rules.py, and ibkr_flex.py
so the React frontend can fetch real data via REST endpoints.
"""

import sys
import os

# Add parent directory to path so we can import existing modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Query
import io
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, date
import pandas as pd

app = FastAPI(title="MO Money API", version="1.0.0")

# CORS — allow React dev server + Vercel production
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.vercel\.app|https://motrading\.net|http://localhost:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import existing modules
import db_layer as db

# R2 storage — set env vars explicitly if not already present
# (Railway sometimes doesn't inject service vars into the Python runtime)
_R2_DEFAULTS = {
    "R2_ENDPOINT_URL": "https://ecdf0141a44925b262e8334b8850e7b6.r2.cloudflarestorage.com",
    "R2_ACCESS_KEY_ID": "d7e0b5d0ec72ee0936591c63f1ece144",
    "R2_SECRET_ACCESS_KEY": "79a1e621db58ba13ab16839d6eed49c5d2ef6c08e8be8a2f62e1dd9376eace88",
    "R2_BUCKET_NAME": "trading-journal-images",
    "R2_PUBLIC_URL": "https://pub-a55e7ca9f1ed4305a3de0d614ea0ea79.r2.dev",
}
for _k, _v in _R2_DEFAULTS.items():
    if not os.environ.get(_k):
        os.environ[_k] = _v

try:
    import r2_storage as r2
    print(f"[R2] Module loaded, endpoint: {os.environ.get('R2_ENDPOINT_URL', 'NONE')[:40]}")
except Exception as _r2_err:
    print(f"[R2] Import failed: {_r2_err}")
    r2 = None

def _is_r2_available():
    if r2 is None:
        return False
    try:
        cfg = r2._get_r2_config()
        return bool(cfg.get("endpoint_url"))
    except Exception:
        return False


def _df_to_records(df: pd.DataFrame) -> list:
    """Convert DataFrame to list of dicts, handling NaN and Timestamps."""
    if df.empty:
        return []
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df.fillna("").to_dict(orient="records")


# ============================================================
# JOURNAL ENDPOINTS
# ============================================================
def _normalize_journal(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize journal column names to lowercase with underscores."""
    rename = {
        "Day": "day", "Status": "status", "Market Window": "market_window",
        "> 21e": "above_21ema", "Cash -/+": "cash_change",
        "Beg NLV": "beg_nlv", "End NLV": "end_nlv",
        "Daily $ Change": "daily_dollar_change", "Daily % Change": "daily_pct_change",
        "% Invested": "pct_invested", "SPY": "spy", "Nasdaq": "nasdaq",
        "Market_Notes": "market_notes", "Market_Action": "market_action",
        "Portfolio_Heat": "portfolio_heat", "SPY_ATR": "spy_atr", "Nasdaq_ATR": "nasdaq_atr",
        "Score": "score", "Highlights": "highlights", "Lowlights": "lowlights",
        "Mistakes": "mistakes", "Top_Lesson": "top_lesson",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df


@app.get("/api/journal/latest")
def journal_latest(portfolio: str = "CanSlim"):
    """Get the most recent journal entry (NLV, daily change, etc.)."""
    df = db.load_journal(portfolio)
    if df.empty:
        return {"error": "No journal data"}
    df = _normalize_journal(df)
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    df = df.sort_values("day", ascending=False)
    row = df.iloc[0].to_dict()
    for k, v in row.items():
        if isinstance(v, pd.Timestamp):
            row[k] = v.strftime("%Y-%m-%d")
        elif pd.isna(v):
            row[k] = None
        elif hasattr(v, 'item'):  # numpy types
            row[k] = v.item()
    return row


@app.get("/api/journal/history")
def journal_history(portfolio: str = "CanSlim", days: int = 365):
    """Get journal history for equity curve."""
    df = db.load_journal(portfolio)
    if df.empty:
        return []
    df = _normalize_journal(df)
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    df = df.sort_values("day")

    # Clean numeric columns
    for c in ["beg_nlv", "end_nlv", "cash_change", "daily_dollar_change",
              "daily_pct_change", "pct_invested", "spy", "nasdaq",
              "portfolio_heat", "spy_atr", "nasdaq_atr"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Filter to requested days
    if days > 0:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
        df = df[df["day"] >= cutoff]

    # Compute TWR equity curve
    df["adjusted_beg"] = df["beg_nlv"] + df["cash_change"]
    mask = df["adjusted_beg"] != 0
    df["daily_return"] = 0.0
    df.loc[mask, "daily_return"] = (df.loc[mask, "end_nlv"] - df.loc[mask, "adjusted_beg"]) / df.loc[mask, "adjusted_beg"]
    df["twr_curve"] = (1 + df["daily_return"]).cumprod()
    df["portfolio_ltd"] = (df["twr_curve"] - 1) * 100

    # SPY/NDX benchmarks
    if "spy" in df.columns:
        spy_start = df["spy"].iloc[0] if df["spy"].iloc[0] > 0 else 1
        df["spy_ltd"] = (df["spy"] / spy_start - 1) * 100
    if "nasdaq" in df.columns:
        ndx_start = df["nasdaq"].iloc[0] if df["nasdaq"].iloc[0] > 0 else 1
        df["ndx_ltd"] = (df["nasdaq"] / ndx_start - 1) * 100

    cols = ["day", "end_nlv", "beg_nlv", "daily_pct_change", "daily_dollar_change",
            "daily_return", "pct_invested", "portfolio_ltd", "spy_ltd", "ndx_ltd",
            "spy", "nasdaq", "portfolio_heat", "score", "cash_change",
            "market_window", "market_notes", "market_action",
            "spy_atr", "nasdaq_atr",
            "highlights", "lowlights", "mistakes", "top_lesson"]
    available_cols = [c for c in cols if c in df.columns]
    return _df_to_records(df[available_cols])


@app.post("/api/journal/edit")
def journal_edit(entry: dict):
    """Update or insert a journal entry. Preserves existing values for fields not sent."""
    try:
        portfolio = entry.pop("portfolio", "CanSlim")
        day = entry.get("day")

        # Load existing entry to preserve fields not being edited
        existing = {}
        df = db.load_journal(portfolio)
        if not df.empty:
            df = _normalize_journal(df)
            df["day"] = pd.to_datetime(df["day"], errors="coerce").dt.strftime("%Y-%m-%d")
            match = df[df["day"] == str(day).strip()[:10]]
            if not match.empty:
                row = match.iloc[0]
                existing = {
                    "ending_nlv": float(row.get("end_nlv", 0) or 0),
                    "beginning_nlv": float(row.get("beg_nlv", 0) or 0),
                    "cash_flow": float(row.get("cash_change", 0) or 0),
                    "daily_dollar_change": float(row.get("daily_dollar_change", 0) or 0),
                    "daily_percent_change": float(row.get("daily_pct_change", 0) or 0),
                    "percent_invested": float(row.get("pct_invested", 0) or 0),
                    "spy_close": float(row.get("spy", 0) or 0),
                    "nasdaq_close": float(row.get("nasdaq", 0) or 0),
                    "market_window": str(row.get("market_window", "") or ""),
                    "market_notes": str(row.get("market_notes", "") or ""),
                    "market_action": str(row.get("market_action", "") or ""),
                    "portfolio_heat": float(row.get("portfolio_heat", 0) or 0),
                    "spy_atr": float(row.get("spy_atr", 0) or 0),
                    "nasdaq_atr": float(row.get("nasdaq_atr", 0) or 0),
                    "score": int(row.get("score", 0) or 0),
                    "highlights": str(row.get("highlights", "") or ""),
                    "lowlights": str(row.get("lowlights", "") or ""),
                    "mistakes": str(row.get("mistakes", "") or ""),
                    "top_lesson": str(row.get("top_lesson", "") or ""),
                }

        # Merge: use sent values, fall back to existing
        def _f(key, db_key, default=0):
            if key in entry and entry[key] not in (None, ""):
                return float(entry[key])
            return existing.get(db_key, default)

        def _s(key, db_key, default=""):
            if key in entry and entry[key] is not None:
                return str(entry[key])
            return existing.get(db_key, default)

        journal_entry = {
            "portfolio_id": portfolio,
            "day": day,
            "ending_nlv": _f("end_nlv", "ending_nlv"),
            "beginning_nlv": _f("beg_nlv", "beginning_nlv"),
            "cash_flow": _f("cash_change", "cash_flow"),
            "daily_dollar_change": _f("daily_dollar_change", "daily_dollar_change"),
            "daily_percent_change": _f("daily_pct_change", "daily_percent_change"),
            "percent_invested": _f("pct_invested", "percent_invested"),
            "spy_close": _f("spy", "spy_close"),
            "nasdaq_close": _f("nasdaq", "nasdaq_close"),
            "market_window": _s("market_window", "market_window"),
            "market_notes": _s("market_notes", "market_notes"),
            "market_action": _s("market_action", "market_action"),
            "portfolio_heat": _f("portfolio_heat", "portfolio_heat"),
            "spy_atr": _f("spy_atr", "spy_atr"),
            "nasdaq_atr": _f("nasdaq_atr", "nasdaq_atr"),
            "score": int(_f("score", "score")),
            "highlights": _s("highlights", "highlights"),
            "lowlights": _s("lowlights", "lowlights"),
            "mistakes": _s("mistakes", "mistakes"),
            "top_lesson": _s("top_lesson", "top_lesson"),
        }
        row_id = db.save_journal_entry(journal_entry)
        return {"status": "ok", "id": row_id}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ============================================================
# TRADE ENDPOINTS
# ============================================================
def _normalize_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize trade column names to lowercase."""
    rename = {
        "Trade_ID": "trade_id", "Ticker": "ticker", "Status": "status",
        "Open_Date": "open_date", "Closed_Date": "closed_date",
        "Shares": "shares", "Avg_Entry": "avg_entry", "Avg_Exit": "avg_exit",
        "Total_Cost": "total_cost", "Realized_PL": "realized_pl",
        "Unrealized_PL": "unrealized_pl", "Return_Pct": "return_pct",
        "Rule": "rule", "Buy_Notes": "buy_notes", "Sell_Rule": "sell_rule",
        "Sell_Notes": "sell_notes", "Risk_Budget": "risk_budget",
        "Action": "action", "Date": "date", "Amount": "amount",
        "Value": "value", "Notes": "notes", "Stop_Loss": "stop_loss",
        "Trx_ID": "trx_id",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    # Also handle already-lowercase columns (from DB mode)
    return df


@app.get("/api/trades/open")
def trades_open(portfolio: str = "CanSlim"):
    """Get all open positions."""
    df = db.load_summary(portfolio)
    if df.empty:
        return []
    df = _normalize_trades(df)
    status_col = "status" if "status" in df.columns else "Status"
    open_df = df[df[status_col].str.upper() == "OPEN"].copy()
    return _df_to_records(open_df)


@app.get("/api/trades/closed")
def trades_closed(portfolio: str = "CanSlim", limit: int = 50):
    """Get recent closed trades."""
    df = db.load_summary(portfolio)
    if df.empty:
        return []
    df = _normalize_trades(df)
    status_col = "status" if "status" in df.columns else "Status"
    closed = df[df[status_col].str.upper() == "CLOSED"].copy()
    if "closed_date" in closed.columns:
        closed["closed_date"] = pd.to_datetime(closed["closed_date"], errors="coerce")
        closed = closed.sort_values("closed_date", ascending=False)
    return _df_to_records(closed.head(limit))


@app.get("/api/trades/details/{trade_id}")
def trade_details(trade_id: str, portfolio: str = "CanSlim"):
    """Get all transactions for a trade campaign."""
    df = db.load_details(portfolio, trade_id=trade_id)
    if df.empty:
        return []
    df = _normalize_trades(df)
    return _df_to_records(df)


@app.get("/api/trades/open/details")
def trades_open_details(portfolio: str = "CanSlim"):
    """Get all transactions for open trades (for stop loss, pyramid info)."""
    summary_df = db.load_summary(portfolio)
    if summary_df.empty:
        return []
    summary_df = _normalize_trades(summary_df)
    status_col = "status" if "status" in summary_df.columns else "Status"
    open_ids = summary_df[summary_df[status_col].str.upper() == "OPEN"]["trade_id"].tolist()
    if not open_ids:
        return []
    details_df = db.load_details(portfolio)
    if details_df.empty:
        return []
    details_df = _normalize_trades(details_df)
    filtered = details_df[details_df["trade_id"].isin(open_ids)].copy()
    if "date" in filtered.columns:
        filtered["date"] = pd.to_datetime(filtered["date"], errors="coerce")
        filtered = filtered.sort_values(["trade_id", "date"])
    return _df_to_records(filtered)


@app.get("/api/trades/recent")
def trades_recent(portfolio: str = "CanSlim", limit: int = 20):
    """Get most recent trade transactions (buys + sells)."""
    df = db.load_details(portfolio)
    if df.empty:
        return []
    df = _normalize_trades(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date", ascending=False)
    return _df_to_records(df.head(limit))


# ============================================================
# LIVE PRICES
# ============================================================
import re as _re

def _is_option_ticker(ticker):
    return bool(ticker and '$' in ticker and _re.search(r'\d{6}', ticker))

def _to_occ_symbol(readable_ticker):
    try:
        m = _re.match(r'^(\S+)\s+(\d{6})\s+\$([0-9.]+)(C|P)$', readable_ticker.strip())
        if not m:
            return None
        underlying = m.group(1)
        expiry = m.group(2)
        strike = float(m.group(3))
        put_call = m.group(4)
        strike_int = int(strike * 1000)
        return f"{underlying}{expiry}{put_call}{strike_int:08d}"
    except Exception:
        return None


@app.get("/api/prices/lookup")
def price_lookup(ticker: str = ""):
    """Get live price + ATR for a single ticker. Used by Position Sizer and Log Buy."""
    if not ticker.strip():
        return {"error": "No ticker provided"}
    import yfinance as yf

    try:
        t = ticker.strip().upper()
        stock = yf.Ticker(t)
        df = stock.history(period="40d")
        if df.empty:
            return {"error": f"No data for {t}"}

        price = float(df["Close"].iloc[-1])

        # ATR calculation — matches Streamlit: ATR% = SMA(TR,21) / SMA(Low,21) * 100
        atr_pct = 5.0
        atr_21 = 0.0
        if len(df) >= 21:
            tr = pd.concat([
                df["High"] - df["Low"],
                (df["High"] - df["Close"].shift(1)).abs(),
                (df["Low"] - df["Close"].shift(1)).abs(),
            ], axis=1).max(axis=1)
            sma_tr = float(tr.tail(21).mean())
            sma_low = float(df["Low"].tail(21).mean())
            atr_21 = sma_tr
            if sma_low > 0:
                atr_pct = (sma_tr / sma_low) * 100

        return {
            "ticker": t,
            "price": round(price, 2),
            "atr": round(atr_21, 2),
            "atr_pct": round(atr_pct, 2),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/prices/batch")
def batch_prices(tickers: str = ""):
    """Get live prices for a comma-separated list of tickers.
    Supports both stock tickers and readable option format ('LUMN 260717 $8C').
    Returns dict of {readable_ticker: price}.
    """
    if not tickers.strip():
        return {}
    import yfinance as yf

    ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]
    yf_symbols = []
    yf_to_readable = {}
    for t in ticker_list:
        if _is_option_ticker(t):
            occ = _to_occ_symbol(t)
            if occ:
                yf_symbols.append(occ)
                yf_to_readable[occ] = t
        else:
            yf_symbols.append(t)
            yf_to_readable[t] = t

    if not yf_symbols:
        return {}

    try:
        data = yf.download(yf_symbols, period="1d", progress=False)['Close']
        result = {}
        if len(yf_symbols) == 1:
            val = float(data.iloc[-1]) if not data.empty else None
            if val is not None:
                readable = yf_to_readable.get(yf_symbols[0], yf_symbols[0])
                result[readable] = val
        else:
            last = data.iloc[-1]
            for yf_sym in yf_symbols:
                if yf_sym in last.index and not pd.isna(last[yf_sym]):
                    readable = yf_to_readable.get(yf_sym, yf_sym)
                    result[readable] = float(last[yf_sym])
        return result
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# MARKET ENDPOINTS
# ============================================================
@app.get("/api/market/rally-prefix")
def rally_prefix():
    """Get rally day prefix for Market/Global Notes (e.g., 'Day 14 POWERTREND:').
    Uses the same compute_cycle_state() logic as the Streamlit Daily Routine."""
    try:
        # Import the app's compute_cycle_state function
        # It lives in the parent directory's app.py — but it's complex.
        # Instead, replicate the core logic here using the same entry ladder.
        import yfinance as yf
        from market_school_rules import MarketSchoolRules
        from datetime import date as _date

        analyzer = MarketSchoolRules("^IXIC")
        analyzer.fetch_data(start_date="2024-02-24", end_date=_date.today().strftime('%Y-%m-%d'))
        if analyzer.data is None or analyzer.data.empty:
            return {"prefix": ""}
        analyzer.analyze_market()

        df = analyzer.data
        rally_low_idx = analyzer.rally_low_idx
        rally_start_date = analyzer.rally_start_date
        market_in_correction = analyzer.market_in_correction
        ftd_date = analyzer.ftd_date
        buy_switch = analyzer.buy_switch

        # Also check SPY for dual-index FTD
        nasdaq_ftd_date = ftd_date
        try:
            spy_analyzer = MarketSchoolRules("SPY")
            spy_analyzer.fetch_data(start_date="2024-02-24", end_date=_date.today().strftime('%Y-%m-%d'))
            if spy_analyzer.data is not None and not spy_analyzer.data.empty:
                spy_analyzer.analyze_market(
                    external_ftd_date=nasdaq_ftd_date,
                    external_ftd_source='NASDAQ',
                )
                spy_ftd = spy_analyzer.ftd_date
                if spy_ftd is not None:
                    if nasdaq_ftd_date is None or pd.Timestamp(spy_ftd) < pd.Timestamp(nasdaq_ftd_date):
                        analyzer.analyze_market(external_ftd_date=spy_ftd, external_ftd_source='SPY')
                        ftd_date = analyzer.ftd_date
                        buy_switch = analyzer.buy_switch
        except Exception:
            pass

        # Compute MAs
        df['8EMA'] = df['Close'].ewm(span=8, adjust=False).mean()
        df['21EMA'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['50SMA'] = df['Close'].rolling(window=50).mean()
        df['200SMA'] = df['Close'].rolling(window=200).mean()

        curr = df.iloc[-1]
        price = float(curr['Close'])
        ema8 = float(curr['8EMA']) if pd.notna(curr['8EMA']) else 0
        ema21 = float(curr['21EMA']) if pd.notna(curr['21EMA']) else 0
        sma50 = float(curr['50SMA']) if pd.notna(curr['50SMA']) else 0
        sma200 = float(curr['200SMA']) if pd.notna(curr['200SMA']) else 0

        # Rally day determination (same as compute_cycle_state)
        rally_day_idx = rally_low_idx
        if rally_low_idx is not None and rally_low_idx > 0:
            rd_row = df.iloc[rally_low_idx]
            prev_row = df.iloc[rally_low_idx - 1]
            if rd_row['Close'] <= prev_row['Close']:
                day_mid = (rd_row['High'] + rd_row['Low']) / 2
                if rd_row['Close'] < day_mid:
                    for next_i in range(rally_low_idx + 1, len(df)):
                        next_row = df.iloc[next_i]
                        next_prev = df.iloc[next_i - 1]
                        if next_row['Close'] > next_prev['Close']:
                            rally_day_idx = next_i
                            break
                        next_mid = (next_row['High'] + next_row['Low']) / 2
                        if next_row['Close'] >= next_mid:
                            rally_day_idx = next_i
                            break

        days_since_rally = len(df) - rally_day_idx if rally_day_idx is not None else None

        if days_since_rally is None or (market_in_correction and not buy_switch and ftd_date is None):
            return {"prefix": "CORRECTION: ", "day_num": 0, "state": "CORRECTION"}

        # Add 1 if today is after last data bar
        _today = _date.today()
        _last_d = df.index[-1].date() if hasattr(df.index[-1], 'date') else df.index[-1]
        if _today.weekday() < 5 and _today > _last_d:
            days_since_rally += 1

        # Entry ladder steps (same as compute_cycle_state)
        ftd_achieved = ftd_date is not None or (not market_in_correction or buy_switch)
        has_rally = rally_start_date is not None or ftd_achieved

        # Streak checks
        low_above_21_streak = 0
        low_above_50_streak = 0
        for i in range(len(df) - 1, -1, -1):
            row = df.iloc[i]
            if pd.notna(row.get('21EMA')) and row['Low'] > row['21EMA']:
                low_above_21_streak += 1
            else:
                break
        for i in range(len(df) - 1, -1, -1):
            row = df.iloc[i]
            if pd.notna(row.get('50SMA')) and row['Low'] > row['50SMA']:
                low_above_50_streak += 1
            else:
                break

        achieved_steps = set()
        if has_rally:
            if rally_start_date is not None:
                achieved_steps.add(0)
            if ftd_achieved:
                achieved_steps.add(1)
            if price > ema21:
                achieved_steps.add(2)
            if ftd_achieved:
                if curr['Low'] > ema21:
                    achieved_steps.add(3)
                if 3 in achieved_steps and low_above_21_streak >= 3:
                    achieved_steps.add(4)
                if 4 in achieved_steps and low_above_50_streak >= 3:
                    achieved_steps.add(5)
                if 5 in achieved_steps and ema21 > sma50 and ema21 > sma200 and sma50 > sma200:
                    achieved_steps.add(6)
                if 6 in achieved_steps and ema8 > ema21 > sma50 > sma200:
                    achieved_steps.add(7)

        entry_step = max(achieved_steps) if achieved_steps else -1
        if entry_step >= 7:
            state = "POWERTREND"
        elif entry_step >= 4:
            state = "UPTREND"
        elif entry_step >= 0:
            state = "RALLY MODE"
        else:
            state = "CORRECTION"

        # Entry ladder details
        step_labels = [
            "Rally Day", "Follow-Through Day", "Close > 21 EMA",
            "Low > 21 EMA", "Low > 21 EMA (3 days)", "Low > 50 SMA (3 days)",
            "21 EMA > 50 SMA > 200 SMA", "8 EMA > 21 EMA > 50 SMA > 200 SMA",
        ]
        step_exposures = [20, 60, 60, 80, 100, 120, 150, 200]
        entry_ladder = []
        for s in range(8):
            entry_ladder.append({
                "step": s, "label": step_labels[s],
                "achieved": s in achieved_steps,
                "exposure": step_exposures[s],
            })

        # MA stack checks
        stack_8_21 = ema8 > ema21
        stack_21_50 = ema21 > sma50
        stack_50_200 = sma50 > sma200

        # Drawdown + reference high date
        reference_high = analyzer.reference_high or 0
        drawdown_pct = ((price - reference_high) / reference_high * 100) if reference_high > 0 else 0

        # Find reference high date
        ref_high_date = None
        if reference_high > 0:
            for i in range(len(df) - 1, -1, -1):
                if abs(df.iloc[i]['High'] - reference_high) < 0.01:
                    ref_high_date = str(df.index[i])[:10]
                    break

        # Consecutive closes below 21 EMA
        consecutive_below_21 = 0
        for i in range(len(df) - 1, -1, -1):
            row = df.iloc[i]
            if pd.notna(row.get('21EMA')) and row['Close'] < row['21EMA']:
                consecutive_below_21 += 1
            else:
                break

        # Exit alerts
        active_exits = []
        if consecutive_below_21 >= 2:
            active_exits.append({"signal": "21 EMA Confirmed Break", "detail": f"{consecutive_below_21} consecutive closes below 21 EMA", "target": "30%", "severity": "SERIOUS"})
        elif consecutive_below_21 == 1:
            active_exits.append({"signal": "21 EMA Watch", "detail": "1 close below 21 EMA — watching for confirmation", "target": "50%", "severity": "WARNING"})
        if price < sma50:
            active_exits.append({"signal": "50 SMA Violation", "detail": "Price below 50 SMA", "target": "0%", "severity": "CRITICAL"})

        return {
            "prefix": f"Day {days_since_rally} {state}: ",
            "day_num": days_since_rally,
            "state": state,
            "entry_step": entry_step,
            "entry_exposure": step_exposures[entry_step] if entry_step >= 0 else 0,
            "price": round(price, 2),
            "ema8": round(ema8, 2),
            "ema21": round(ema21, 2),
            "sma50": round(sma50, 2),
            "sma200": round(sma200, 2),
            "reference_high": round(reference_high, 2) if reference_high else 0,
            "reference_high_date": ref_high_date,
            "drawdown_pct": round(drawdown_pct, 2),
            "consecutive_below_21": consecutive_below_21,
            "active_exits": active_exits,
            "low_above_21_streak": low_above_21_streak,
            "low_above_50_streak": low_above_50_streak,
            "stack_8_21": stack_8_21,
            "stack_21_50": stack_21_50,
            "stack_50_200": stack_50_200,
            "entry_ladder": entry_ladder,
            "ftd_date": str(ftd_date)[:10] if ftd_date else None,
        }
    except Exception as e:
        return {"prefix": "", "error": str(e)}


@app.get("/api/market/ibd")
def ibd_market_school():
    """Get IBD Market School current status for NASDAQ — exposure, distribution days, signals."""
    try:
        from market_school_rules import MarketSchoolRules
        from datetime import date as _date

        # Run full analysis
        analyzer = MarketSchoolRules("^IXIC")
        analyzer.fetch_data(start_date="2024-02-24", end_date=_date.today().strftime('%Y-%m-%d'))
        if analyzer.data is None or analyzer.data.empty:
            return {"error": "No data"}

        # Dual-index FTD
        ext_ftd = None
        ext_src = None
        try:
            spy_a = MarketSchoolRules("SPY")
            spy_a.fetch_data(start_date="2024-02-24", end_date=_date.today().strftime('%Y-%m-%d'))
            if spy_a.data is not None and not spy_a.data.empty:
                spy_a.analyze_market()
                if spy_a.ftd_date:
                    ext_ftd = spy_a.ftd_date
                    ext_src = "SPY"
        except Exception:
            pass

        analyzer.analyze_market(external_ftd_date=ext_ftd, external_ftd_source=ext_src)

        df = analyzer.data
        curr = df.iloc[-1]
        last_date = df.index[-1]
        price = float(curr['Close'])

        # Latest daily summary
        summary = analyzer.get_daily_summary(last_date.strftime('%Y-%m-%d'))

        # Active distribution days
        active_dd = [dd for dd in analyzer.distribution_days if dd.removed_date is None]

        # Signals on last day
        last_norm = pd.Timestamp(last_date).normalize()
        day_signals = [s for s in analyzer.signals if pd.Timestamp(s.date).normalize() == last_norm]
        buy_sigs = [s.signal_type.name for s in day_signals if s.signal_type.name.startswith('B')]
        sell_sigs = [s.signal_type.name for s in day_signals if s.signal_type.name.startswith('S')]

        # Recent signals (last 30 days)
        cutoff = last_date - pd.Timedelta(days=30)
        recent_signals = []
        for s in analyzer.signals:
            if pd.Timestamp(s.date) >= cutoff:
                recent_signals.append({
                    "date": pd.Timestamp(s.date).strftime('%Y-%m-%d'),
                    "signal": s.signal_type.name,
                    "description": s.description if hasattr(s, 'description') else "",
                })

        # Correction state
        ref_high = analyzer.reference_high or 0
        decline_pct = ((price - ref_high) / ref_high * 100) if ref_high > 0 else 0

        # Exposure pyramid labels
        exp_level = int(summary.get('market_exposure', 0))
        allocation = summary.get('position_allocation', '0%')

        return {
            "as_of": last_date.strftime('%Y-%m-%d'),
            "close": round(price, 2),
            "daily_change": summary.get('daily_change', '0%'),
            "buy_switch": analyzer.buy_switch,
            "market_exposure": exp_level,
            "allocation": allocation,
            "distribution_count": len(active_dd),
            "buy_signals": buy_sigs,
            "sell_signals": sell_sigs,
            "in_correction": analyzer.market_in_correction,
            "ftd_date": analyzer.ftd_date.strftime('%Y-%m-%d') if analyzer.ftd_date else None,
            "nasdaq_ftd": analyzer.ftd_date.strftime('%Y-%m-%d') if analyzer.ftd_date else None,
            "spy_ftd": ext_ftd.strftime('%Y-%m-%d') if ext_ftd and hasattr(ext_ftd, 'strftime') else (str(ext_ftd)[:10] if ext_ftd else None),
            "ftd_source": ext_src if ext_ftd else "NASDAQ",
            "reference_high": round(ref_high, 2),
            "decline_pct": round(decline_pct, 2),
            "distribution_days": [
                {"date": dd.date.strftime('%Y-%m-%d'), "type": dd.type, "loss": round(dd.loss_percent, 2)}
                for dd in sorted(active_dd, key=lambda x: x.date, reverse=True)
            ],
            "recent_signals": recent_signals,
            # Historical exposure (last 30 days)
            "history": [
                {
                    "date": analyzer.data.index[i].strftime('%Y-%m-%d'),
                    "market_exposure": int(analyzer.get_daily_summary(analyzer.data.index[i].strftime('%Y-%m-%d')).get('market_exposure', 0)),
                }
                for i in range(max(0, len(analyzer.data) - 30), len(analyzer.data))
            ],
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/market/rally-data")
def rally_data(ftd_date: str = "", index: str = "^IXIC"):
    """Fetch index closes from FTD through Day 25 for Rally Context chart."""
    if not ftd_date:
        return {"error": "ftd_date required"}
    try:
        import yfinance as yf
        ftd_ts = pd.Timestamp(ftd_date)
        end_ts = ftd_ts + pd.Timedelta(days=40)
        today_ts = pd.Timestamp.now().normalize()
        if end_ts > today_ts:
            end_ts = today_ts + pd.Timedelta(days=1)

        df = yf.Ticker(index).history(start=ftd_ts - pd.Timedelta(days=5), end=end_ts)
        if df.empty:
            return {"error": "No data from yfinance"}

        df.index = df.index.tz_localize(None) if df.index.tz else df.index
        ftd_mask = df.index.date >= ftd_ts.date()
        df_post = df[ftd_mask]
        if df_post.empty:
            return {"error": f"No data on or after {ftd_date}"}

        day0_close = float(df_post.iloc[0]["Close"])
        points = []
        for i in range(1, min(len(df_post), 26)):
            row = df_post.iloc[i]
            pct = ((float(row["Close"]) / day0_close) - 1) * 100
            points.append({
                "day": i,
                "date": df_post.index[i].strftime("%Y-%m-%d"),
                "close": round(float(row["Close"]), 2),
                "low": round(float(row["Low"]), 2),
                "pct": round(pct, 2),
            })

        return {"day0_close": round(day0_close, 2), "points": points}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/market/mfactor")
def market_mfactor():
    """Get current M Factor state for NASDAQ + SPY with full MA stack."""
    try:
        import yfinance as yf

        result = {}
        for ticker, label in [("^IXIC", "nasdaq"), ("SPY", "spy")]:
            df = yf.Ticker(ticker).history(period="1y")
            if df.empty:
                continue
            df["21EMA"] = df["Close"].ewm(span=21, adjust=False).mean()
            df["50SMA"] = df["Close"].rolling(window=50).mean()
            df["200SMA"] = df["Close"].rolling(window=200).mean()
            curr = df.iloc[-1]

            price = float(curr["Close"])
            ema21 = float(curr["21EMA"]) if pd.notna(curr["21EMA"]) else 0
            sma50 = float(curr["50SMA"]) if pd.notna(curr["50SMA"]) else 0
            sma200 = float(curr["200SMA"]) if pd.notna(curr["200SMA"]) else 0

            above_21 = bool(price > ema21) if ema21 > 0 else False
            above_50 = bool(price > sma50) if sma50 > 0 else False
            above_200 = bool(price > sma200) if sma200 > 0 else False

            # Powertrend: Low > 21 EMA for 3+ consecutive days
            low_above_21_streak = 0
            for i in range(len(df) - 1, -1, -1):
                row = df.iloc[i]
                if pd.notna(row.get("21EMA")) and row["Low"] > row["21EMA"]:
                    low_above_21_streak += 1
                else:
                    break
            is_powertrend = low_above_21_streak >= 3 and above_21 and above_50

            # Individual state
            if is_powertrend:
                state = "POWERTREND"
            elif above_21:
                state = "OPEN"
            elif above_50:
                state = "NEUTRAL"
            else:
                state = "CLOSED"

            # % distance from MAs
            d21 = ((price - ema21) / ema21 * 100) if ema21 > 0 else 0
            d50 = ((price - sma50) / sma50 * 100) if sma50 > 0 else 0
            d200 = ((price - sma200) / sma200 * 100) if sma200 > 0 else 0

            result[label] = {
                "price": round(price, 2),
                "ema21": round(ema21, 2),
                "sma50": round(sma50, 2),
                "sma200": round(sma200, 2),
                "above_21ema": above_21,
                "above_50sma": above_50,
                "above_200sma": above_200,
                "d21": round(d21, 2),
                "d50": round(d50, 2),
                "d200": round(d200, 2),
                "state": state,
                "powertrend_streak": low_above_21_streak,
            }

        # Combined state
        ns = result.get("nasdaq", {}).get("state", "CLOSED")
        ss = result.get("spy", {}).get("state", "CLOSED")
        if ns == "POWERTREND" or ss == "POWERTREND":
            combined = "POWERTREND"
        elif ns == "CLOSED" and ss == "CLOSED":
            combined = "CLOSED"
        elif ns in ["NEUTRAL", "CLOSED"] or ss in ["NEUTRAL", "CLOSED"]:
            combined = "NEUTRAL"
        else:
            combined = "OPEN"

        result["combined_state"] = combined
        return result
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# CONFIG ENDPOINTS
# ============================================================
@app.get("/api/config/{key}")
def get_config(key: str):
    """Get a config value from app_config."""
    try:
        val = db.get_config(key)
        return {"key": key, "value": val}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# HEALTH
# ============================================================
@app.get("/api/trades/lessons")
def get_trade_lessons(portfolio: str = "CanSlim"):
    """Get all trade lesson notes for a portfolio."""
    try:
        lessons = db.get_trade_lessons(portfolio)
        return {"lessons": {k: {"note": v[0], "category": v[1]} for k, v in lessons.items()}}
    except Exception as e:
        return {"lessons": {}, "error": str(e)}


@app.post("/api/trades/lessons")
def save_trade_lesson(entry: dict):
    """Save a trade lesson note."""
    try:
        portfolio = entry.get("portfolio", "CanSlim")
        trade_id = entry.get("trade_id", "")
        note = entry.get("note", "")
        category = entry.get("category", "")
        ok = db.save_trade_lesson(portfolio, trade_id, note, category)
        return {"status": "ok" if ok else "error"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/api/health")
def health():
    """Health check."""
    try:
        ok = db.test_connection()
        return {"status": "ok" if ok else "db_error", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ============================================================
# ADMIN — CONFIG WRITE
# ============================================================
@app.post("/api/config/{key}")
def set_config(key: str, body: dict):
    """Set a config value."""
    try:
        value = body.get("value")
        value_type = body.get("value_type")
        category = body.get("category")
        description = body.get("description")
        user = body.get("user", "admin")
        ok = db.set_config(key, value, value_type=value_type, category=category,
                           description=description, user=user)
        if ok:
            # Sync auto-events when reset_date changes
            if key == "reset_date" and hasattr(db, "sync_auto_events_from_config"):
                db.sync_auto_events_from_config()
            return {"status": "ok"}
        return {"status": "error", "detail": "set_config returned False"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ============================================================
# ADMIN — DASHBOARD EVENTS
# ============================================================
@app.get("/api/events")
def list_events(scope: str = "CanSlim"):
    """List dashboard events."""
    try:
        df = db.load_dashboard_events(scope=scope)
        if df.empty:
            return []
        return _df_to_records(df)
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/events")
def add_event(body: dict):
    """Add a dashboard event."""
    try:
        ok = db.save_dashboard_event(
            event_date=body["event_date"],
            label=body["label"],
            category=body.get("category", "market"),
            notes=body.get("notes", ""),
            scope=body.get("scope", "CanSlim"),
            auto_generated=False,
            user=body.get("user", "admin"),
        )
        return {"status": "ok" if ok else "error"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.put("/api/events/{event_id}")
def update_event(event_id: int, body: dict):
    """Update a dashboard event."""
    try:
        ok = db.update_dashboard_event(
            event_id=event_id,
            event_date=body.get("event_date"),
            label=body.get("label"),
            category=body.get("category"),
            notes=body.get("notes"),
            user=body.get("user", "admin"),
        )
        return {"status": "ok" if ok else "error"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.delete("/api/events/{event_id}")
def delete_event(event_id: int, user: str = "admin"):
    """Delete a dashboard event."""
    try:
        ok = db.delete_dashboard_event(event_id, user=user)
        return {"status": "ok" if ok else "error"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ============================================================
# ADMIN — AUDIT TRAIL
# ============================================================
@app.get("/api/audit")
def audit_trail(limit: int = 100, action_filter: str = None):
    """Get recent audit trail entries."""
    try:
        if not hasattr(db, "load_recent_audit_entries"):
            return []
        df = db.load_recent_audit_entries(limit=limit, action_filter=action_filter)
        if df.empty:
            return []
        return _df_to_records(df)
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# ADMIN — DATA CLEANUP
# ============================================================
@app.post("/api/admin/cleanup-marketsurge")
def cleanup_marketsurge(body: dict):
    """Clean up duplicate MarketSurge images."""
    try:
        if not hasattr(db, "cleanup_duplicate_marketsurge_images"):
            return {"error": "Cleanup function not available"}
        dry_run = body.get("dry_run", True)
        result = db.cleanup_duplicate_marketsurge_images(dry_run=dry_run, user="admin")
        return result
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# AI COACH — STREAMING CHAT
# ============================================================
from fastapi.responses import StreamingResponse
import json as _json

def _build_trade_context(portfolio: str, scope: str = "recent", n: int = 10):
    """Build trade data context for AI coach prompts."""
    parts = []
    df_s = db.load_summary(portfolio)
    df_d = db.load_details(portfolio)
    if not df_s.empty:
        df_s = _normalize_trades(df_s)
        closed = df_s[df_s["status"].str.upper() == "CLOSED"].copy()
        if "closed_date" in closed.columns:
            closed["closed_date"] = pd.to_datetime(closed["closed_date"], errors="coerce")
            closed = closed.sort_values("closed_date", ascending=False)
        open_t = df_s[df_s["status"].str.upper() == "OPEN"].copy()
        show = closed.head(n) if scope == "recent" else closed
        if not show.empty:
            cols = ["trade_id", "ticker", "open_date", "closed_date", "shares",
                    "avg_entry", "avg_exit", "realized_pl", "return_pct",
                    "rule", "sell_rule", "stop_loss"]
            use = [c for c in cols if c in show.columns]
            parts.append(f"=== CLOSED TRADES ({len(show)}) ===")
            parts.append(show[use].to_csv(index=False))
        if not open_t.empty:
            cols = ["trade_id", "ticker", "open_date", "shares",
                    "avg_entry", "total_cost", "unrealized_pl", "rule", "stop_loss"]
            use = [c for c in cols if c in open_t.columns]
            parts.append(f"\n=== OPEN TRADES ({len(open_t)}) ===")
            parts.append(open_t[use].to_csv(index=False))
    return "\n".join(parts)


def _build_journal_context(portfolio: str, n: int = 14):
    """Build journal context for AI coach prompts."""
    df = db.load_journal(portfolio)
    if df.empty:
        return "No journal data available."
    df = _normalize_journal(df)
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    df = df.sort_values("day", ascending=False).head(n)
    parts = [f"=== JOURNAL ENTRIES (last {len(df)} days) ==="]
    for _, row in df.iterrows():
        day = row["day"].strftime("%Y-%m-%d") if pd.notna(row.get("day")) else "?"
        entry = f"\n--- {day} ---"
        for field in ["status", "market_window", "market_action", "market_notes",
                      "daily_dollar_change", "daily_pct_change", "end_nlv", "pct_invested",
                      "portfolio_heat", "score", "highlights", "lowlights", "mistakes", "top_lesson"]:
            val = row.get(field)
            if val is not None and str(val).strip() and str(val) != "nan":
                entry += f"\n  {field}: {val}"
        parts.append(entry)
    return "\n".join(parts)


def _build_stats_context(portfolio: str):
    """Build aggregate stats for AI coach."""
    parts = ["=== PORTFOLIO STATS ==="]
    df_s = db.load_summary(portfolio)
    if not df_s.empty:
        df_s = _normalize_trades(df_s)
        closed = df_s[df_s["status"].str.upper() == "CLOSED"]
        open_t = df_s[df_s["status"].str.upper() == "OPEN"]
        if not closed.empty and "realized_pl" in closed.columns:
            pl = pd.to_numeric(closed["realized_pl"], errors="coerce").fillna(0)
            wins = pl[pl > 0]
            losses = pl[pl < 0]
            parts.append(f"Total closed trades: {len(closed)}")
            parts.append(f"Open positions: {len(open_t)}")
            if len(closed) > 0:
                parts.append(f"Win rate: {len(wins)}/{len(closed)} ({len(wins)/len(closed)*100:.1f}%)")
            parts.append(f"Total realized P&L: ${pl.sum():,.2f}")
            if len(wins) > 0:
                parts.append(f"Avg win: ${wins.mean():,.2f}")
            if len(losses) > 0:
                parts.append(f"Avg loss: ${losses.mean():,.2f}")
            parts.append(f"Best trade: ${pl.max():,.2f}")
            parts.append(f"Worst trade: ${pl.min():,.2f}")
    df_j = db.load_journal(portfolio)
    if not df_j.empty:
        df_j = _normalize_journal(df_j)
        for c in ["end_nlv"]:
            if c in df_j.columns:
                df_j[c] = pd.to_numeric(df_j[c], errors="coerce").fillna(0)
        df_j["day"] = pd.to_datetime(df_j["day"], errors="coerce")
        latest = df_j.sort_values("day", ascending=False).iloc[0]
        parts.append(f"\nLatest NLV: ${float(latest.get('end_nlv', 0)):,.0f}")
    return "\n".join(parts)


COACH_SYSTEM_PROMPT = """You are MO's personal AI trading coach. You analyze his CANSLIM trading journal data and provide actionable insights.

TRADING SYSTEM CONTEXT:
- MO trades the CANSLIM / IBD growth stock strategy
- Buy rules: Base breakouts, volume events (HVE/HVSI), moving average reclaims, pullbacks, gap-ups, trendline breaks
- Sell rules: Capital protection (stop loss), selling into strength, portfolio management, change of character, breakout failure
- Position sizing: Risk-based with stop losses and risk budgets
- Transaction IDs: B1/B2 = initial buys, A1/A2 = add-on buys, S1/S2 = sells
- Trade IDs format: YYYYMM-NNN (e.g., 202602-001)

ANALYSIS GUIDELINES:
- Be specific — reference actual tickers, dates, and numbers from the data
- Focus on patterns, not individual trades (unless asked)
- Highlight both strengths and areas for improvement
- Frame feedback constructively — you're a coach, not a critic
- When discussing sell discipline, compare actual exit vs. stop loss and optimal exit
- Keep responses concise and actionable — use bullet points
- Use dollar amounts and percentages to quantify insights
- If the data is insufficient to answer, say so honestly

Current portfolio: CanSlim
"""


@app.post("/api/coach/chat")
def coach_chat(body: dict):
    """AI Coach chat endpoint — streams response via SSE."""
    try:
        import anthropic
    except ImportError:
        return {"error": "anthropic package not installed"}

    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            # Try loading from streamlit secrets
            try:
                import tomllib
                secrets_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                            ".streamlit", "secrets.toml")
                with open(secrets_path, "rb") as f:
                    secrets = tomllib.load(f)
                api_key = secrets.get("anthropic", {}).get("api_key", "")
            except Exception:
                pass

        if not api_key:
            return {"error": "Anthropic API key not configured"}

        client = anthropic.Anthropic(api_key=api_key)
        user_msg = body.get("message", "")
        preset = body.get("preset")
        portfolio = body.get("portfolio", "CanSlim")

        # Build context based on preset or free-form
        if preset == "recent":
            context = _build_trade_context(portfolio, "all")
            if not user_msg:
                user_msg = """Review my recent closed trades. For each trade, analyze:
1. Entry quality — was the buy rule appropriate? Was timing good?
2. Exit execution — did I follow my sell rules? Did I sell too early or too late?
3. Position sizing — was the risk budget reasonable?
Then give me an overall grade (A-F) and your top 3 actionable recommendations."""
        elif preset == "mistakes":
            context = _build_journal_context(portfolio, 30)
            if not user_msg:
                user_msg = """Analyze my journal entries looking for recurring mistake patterns.
Group the mistakes into categories and rank them by frequency and cost.
For each pattern: what the mistake is, how often it occurs, what it costs, and a specific fix.
Also look at my lowlights entries for additional patterns."""
        elif preset == "weekly":
            context = _build_journal_context(portfolio, 14)
            if not user_msg:
                user_msg = """Give me a coaching summary for the current trading week. Cover:
1. P&L performance and equity trend
2. What I did well (from highlights)
3. What needs work (from lowlights/mistakes)
4. Market conditions and how I adapted
5. One key lesson or focus for next week"""
        elif preset == "behavior":
            context = _build_trade_context(portfolio, "all") + "\n\n" + _build_journal_context(portfolio, 30) + "\n\n" + _build_stats_context(portfolio)
            if not user_msg:
                user_msg = """Analyze my trading behavior patterns. Look at:
1. Do I add to winners or average down on losers?
2. Am I cutting losses quickly or holding too long?
3. Am I selling winners too early?
4. Position sizing patterns
5. Time patterns — am I more profitable on certain setups?
Give me a behavioral profile and 3 specific things to work on."""
        else:
            # Free-form: include stats + trades + journal
            lower = user_msg.lower()
            journal_days = 30
            if any(w in lower for w in ["all", "history", "overall", "pattern", "behavior", "trend", "year"]):
                journal_days = 90
            elif any(w in lower for w in ["week", "recent", "last", "today", "yesterday"]):
                journal_days = 14
            context = _build_stats_context(portfolio) + "\n\n" + _build_trade_context(portfolio, "all") + "\n\n" + _build_journal_context(portfolio, journal_days)

        today_str = date.today().strftime("%Y-%m-%d")
        system = COACH_SYSTEM_PROMPT + f"\nToday's date: {today_str}"
        full_msg = f"{context}\n\n{user_msg}"

        def stream_gen():
            try:
                with client.messages.stream(
                    model="claude-sonnet-4-6",
                    max_tokens=2048,
                    system=system,
                    messages=[{"role": "user", "content": full_msg}],
                ) as stream:
                    for text in stream.text_stream:
                        yield f"data: {_json.dumps({'text': text})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {_json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(stream_gen(), media_type="text/event-stream")
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# ============================================================
# TRADE WRITE ENDPOINTS (Log Buy / Log Sell)
# ============================================================
@app.get("/api/trades/next-id")
def next_trade_id(portfolio: str = "CanSlim", date: str = ""):
    """Generate next available trade ID for a given month."""
    try:
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        ym = pd.Timestamp(date).strftime("%Y%m")
        df_s = db.load_summary(portfolio)
        df_s = _normalize_trades(df_s)
        existing = df_s[df_s["trade_id"].str.startswith(ym)]["trade_id"].tolist() if not df_s.empty else []
        seqs = []
        for x in existing:
            try:
                if "-" in x:
                    seqs.append(int(x.split("-")[-1]))
            except:
                pass
        next_seq = (max(seqs) + 1) if seqs else 1
        return {"trade_id": f"{ym}-{next_seq:03d}"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/trades/buy")
def log_buy(body: dict):
    """Log a buy transaction. Creates/updates summary + inserts detail row."""
    try:
        portfolio = body.get("portfolio", "CanSlim")
        action_type = body.get("action_type", "new")  # "new" or "scalein"
        ticker = body.get("ticker", "").upper()
        trade_id = body.get("trade_id", "")
        shares = float(body.get("shares", 0))
        price = float(body.get("price", 0))
        stop_loss = float(body.get("stop_loss", 0))
        rule = body.get("rule", "")
        notes = body.get("notes", "")
        date_str = body.get("date", datetime.now().strftime("%Y-%m-%d"))
        time_str = body.get("time", datetime.now().strftime("%H:%M"))
        trx_id = body.get("trx_id", "")

        if not ticker or not trade_id or shares <= 0 or price <= 0:
            return {"error": "Missing required fields: ticker, trade_id, shares, price"}

        value = shares * price
        date_time = f"{date_str} {time_str}:00"

        # Determine trx_id if not provided
        if not trx_id:
            df_d = db.load_details(portfolio)
            if not df_d.empty:
                df_d = _normalize_trades(df_d)
                existing_txns = df_d[df_d["trade_id"] == trade_id]
                buy_count = len(existing_txns[existing_txns["action"].str.upper() == "BUY"]) if not existing_txns.empty else 0
                trx_id = f"B{buy_count + 1}" if buy_count == 0 else f"A{buy_count}"
            else:
                trx_id = "B1"

        # Build summary row first (FK requires summary before detail)
        if action_type == "new":
            summary_row = {
                "Trade_ID": trade_id, "Ticker": ticker, "Status": "OPEN",
                "Open_Date": date_str, "Shares": shares,
                "Avg_Entry": price, "Total_Cost": value,
                "Stop_Loss": stop_loss, "Rule": rule, "Buy_Notes": notes,
            }
        else:
            # Scale-in: load existing summary and update
            df_s = db.load_summary(portfolio)
            df_s = _normalize_trades(df_s)
            existing = df_s[df_s["trade_id"] == trade_id]
            if not existing.empty:
                row = existing.iloc[0]
                old_shares = float(row.get("shares", 0))
                old_entry = float(row.get("avg_entry", 0))
                old_cost = float(row.get("total_cost", 0))
                new_total_shares = old_shares + shares
                new_total_cost = old_cost + value
                new_avg_entry = new_total_cost / new_total_shares if new_total_shares > 0 else price
                summary_row = {
                    "Trade_ID": trade_id, "Ticker": ticker, "Status": "OPEN",
                    "Open_Date": str(row.get("open_date", date_str))[:10],
                    "Shares": new_total_shares,
                    "Avg_Entry": round(new_avg_entry, 4),
                    "Total_Cost": round(new_total_cost, 2),
                    "Stop_Loss": stop_loss if stop_loss > 0 else row.get("stop_loss", 0),
                    "Rule": row.get("rule") or rule,
                    "Buy_Notes": notes or row.get("buy_notes", ""),
                }
            else:
                summary_row = {
                    "Trade_ID": trade_id, "Ticker": ticker, "Status": "OPEN",
                    "Open_Date": date_str, "Shares": shares,
                    "Avg_Entry": price, "Total_Cost": value,
                    "Stop_Loss": stop_loss, "Rule": rule, "Buy_Notes": notes,
                }

        summary_id = db.save_summary_row(portfolio, summary_row)

        # Save detail row (after summary so FK constraint is satisfied)
        detail_row = {
            "Trade_ID": trade_id, "Ticker": ticker, "Action": "BUY",
            "Date": date_time, "Shares": shares, "Amount": price,
            "Value": value, "Rule": rule, "Notes": notes,
            "Stop_Loss": stop_loss, "Trx_ID": trx_id,
        }
        detail_id = db.save_detail_row(portfolio, detail_row)

        # Audit trail
        try:
            db.log_audit(portfolio, "BUY", trade_id, ticker,
                         f"{trx_id}: {shares} shs @ ${price:.2f}", username="web")
        except Exception:
            pass

        return {"status": "ok", "detail_id": detail_id, "summary_id": summary_id, "trx_id": trx_id}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# FUNDAMENTALS ENDPOINT
# ============================================================
@app.get("/api/fundamentals/{trade_id}")
def get_fundamentals(trade_id: str, portfolio: str = "CanSlim"):
    """Get extracted MarketSurge fundamentals for a trade."""
    try:
        funds = db.get_trade_fundamentals(portfolio, trade_id)
        # Convert Decimal to float for JSON serialization
        for f in funds:
            for k, v in f.items():
                if hasattr(v, 'as_tuple'):  # Decimal
                    f[k] = float(v)
                elif hasattr(v, 'isoformat'):  # datetime
                    f[k] = v.isoformat()
        return funds or []
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# R2 IMAGE ENDPOINTS
# ============================================================
from fastapi import UploadFile, File, Form

@app.get("/api/images/{trade_id}")
def get_trade_images(trade_id: str, portfolio: str = "CanSlim"):
    """Get all image metadata for a trade."""
    try:
        images = db.get_trade_images(portfolio, trade_id)
        # Build viewable URLs — use public R2 CDN
        R2_PUBLIC = (os.environ.get("R2_PUBLIC_URL") or "https://pub-a55e7ca9f1ed4305a3de0d614ea0ea79.r2.dev").rstrip("/")
        if images:
            for img in images:
                key = img.get("image_url", "")
                if key and key.startswith("http"):
                    img["view_url"] = key
                elif key:
                    img["view_url"] = f"{R2_PUBLIC}/{key}"
                else:
                    img["view_url"] = ""
        return images or []
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/images/upload")
async def upload_image(
    file: UploadFile = File(...),
    portfolio: str = Form("CanSlim"),
    trade_id: str = Form(...),
    ticker: str = Form(...),
    image_type: str = Form(...),
):
    """Upload a trade image to R2 and save metadata to DB."""
    if not _is_r2_available():
        return {"error": "R2 storage not configured"}
    try:
        # Read file content
        content = await file.read()
        file_like = io.BytesIO(content)
        file_like.name = file.filename or "upload.png"

        # Upload to R2
        object_key = r2.upload_image(file_like, portfolio, trade_id, ticker, image_type)
        if not object_key:
            return {"error": "Upload to R2 failed"}

        # Save metadata to DB
        image_id = db.save_trade_image(portfolio, trade_id, ticker, image_type, object_key, file.filename)

        return {"status": "ok", "image_id": image_id, "object_key": object_key}
    except Exception as e:
        return {"error": str(e)}


@app.delete("/api/images/{image_id}")
def delete_image(image_id: int):
    """Delete a trade image from R2 and DB."""
    try:
        # Get the image URL from DB before deleting
        if _is_r2_available():
            url = db.delete_trade_image_by_id(image_id)
            if url and not url.startswith("http"):
                r2.delete_image(url)
        else:
            db.delete_trade_image_by_id(image_id)
        return {"status": "ok"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/r2/status")
def r2_status():
    """Check R2 availability."""
    return {"available": _is_r2_available()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
