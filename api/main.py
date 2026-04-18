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
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, date
import pandas as pd

app = FastAPI(title="MO Money API", version="1.0.0")

# CORS — allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3003", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import existing modules
import db_layer as db


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

    cols = ["day", "end_nlv", "daily_pct_change", "daily_dollar_change",
            "pct_invested", "portfolio_ltd", "spy_ltd", "ndx_ltd",
            "spy", "nasdaq", "portfolio_heat", "score"]
    available_cols = [c for c in cols if c in df.columns]
    return _df_to_records(df[available_cols])


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
    closed = df[df["status"].str.upper() == "CLOSED"].copy()
    if "closed_date" in closed.columns:
        closed["closed_date"] = pd.to_datetime(closed["closed_date"], errors="coerce")
        closed = closed.sort_values("closed_date", ascending=False)
    return _df_to_records(closed.head(limit))


@app.get("/api/trades/details/{trade_id}")
def trade_details(trade_id: str, portfolio: str = "CanSlim"):
    """Get all transactions for a trade campaign."""
    df = db.load_details(portfolio)
    if df.empty:
        return []
    filtered = df[df["trade_id"] == trade_id].copy()
    return _df_to_records(filtered)


@app.get("/api/trades/recent")
def trades_recent(portfolio: str = "CanSlim", limit: int = 20):
    """Get most recent trade transactions (buys + sells)."""
    df = db.load_details(portfolio)
    if df.empty:
        return []
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date", ascending=False)
    return _df_to_records(df.head(limit))


# ============================================================
# MARKET ENDPOINTS
# ============================================================
@app.get("/api/market/mfactor")
def market_mfactor():
    """Get current M Factor state for NASDAQ + SPY."""
    try:
        # Import from the app's existing helper (can't call Streamlit functions directly)
        import yfinance as yf

        result = {}
        for ticker, label in [("^IXIC", "nasdaq"), ("SPY", "spy")]:
            df = yf.Ticker(ticker).history(period="1y")
            if df.empty:
                continue
            df["21EMA"] = df["Close"].ewm(span=21, adjust=False).mean()
            df["50SMA"] = df["Close"].rolling(window=50).mean()
            curr = df.iloc[-1]
            result[label] = {
                "price": float(curr["Close"]),
                "ema21": float(curr["21EMA"]) if pd.notna(curr["21EMA"]) else 0,
                "sma50": float(curr["50SMA"]) if pd.notna(curr["50SMA"]) else 0,
                "above_21ema": bool(curr["Close"] > curr["21EMA"]) if pd.notna(curr["21EMA"]) else False,
                "above_50sma": bool(curr["Close"] > curr["50SMA"]) if pd.notna(curr["50SMA"]) else False,
            }
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
@app.get("/api/health")
def health():
    """Health check."""
    try:
        ok = db.test_connection()
        return {"status": "ok" if ok else "db_error", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
