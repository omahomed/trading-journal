# Trading Journal Application

## Overview
Streamlit-based trading journal for tracking CANSLIM and leveraged ETF strategies. Deployed on Streamlit Cloud with PostgreSQL (Supabase) and Cloudflare R2 for image storage. Falls back to CSV locally.

## Tech Stack
- **Frontend**: Streamlit with `streamlit-option-menu`
- **Database**: PostgreSQL via `psycopg2-binary` (auto-detected on Streamlit Cloud)
- **Storage**: Cloudflare R2 via `boto3` for trade chart images
- **Data**: `pandas`, `yfinance`, `matplotlib`, `plotly`

## File Structure
```
app.py              - Main app (~10,600 lines). All pages in one file.
db_layer.py         - Database abstraction layer (load/save/delete for all tables)
r2_storage.py       - Cloudflare R2 image upload/download/delete
schema.sql          - 7 tables: portfolios, trades_summary, trades_details,
                      trading_journal, audit_trail, market_signals, trade_images
requirements.txt    - Python dependencies
.streamlit/secrets.toml - DB + R2 credentials (DO NOT commit)
```

## Architecture
- **Navigation**: `st.session_state.page` with `nav_button()` helper in sidebar (lines ~688-740)
- **Feature flags**: `DB_AVAILABLE`, `R2_AVAILABLE`, `USE_DATABASE` (auto-detect at startup)
- **Data loading**: `load_data(path)` auto-routes to DB or CSV based on `USE_DATABASE`
- **Trade data**: `load_trade_data()` helper returns `(df_d, df_s)` with schema fixes. Used by Log Buy, Log Sell, and Trade Manager.
- **Timezone**: All dates/times use Central Time via `get_current_date_ct()` / `get_current_time_ct()`

## Portfolios / Accounts
- `PORT_CANSLIM = "CanSlim"` - Main CANSLIM strategy account
- `PORT_TQQQ = "TQQQ Strategy"` - Leveraged ETF strategy account
- `PORT_457B = "457B Plan"` - Retirement account
- **Pending**: Merge TQQQ account balance back into CanSlim. Will need RESET_DATE update.

## Key Pages (sidebar nav order)
### Dashboards
- Command Center - Pilot's Panel, Trading Core (Combined), Historical Data
- Dashboard - Main metrics dashboard
- Trading Overview

### Trading Ops
- Active Campaign Summary
- Log Buy - Standalone buy order entry (extracted from Trade Manager)
- Log Sell - Standalone sell order entry (extracted from Trade Manager)
- Position Sizer
- Trade Journal - Visual review of trades with charts. Has cross-links to Log Buy/Log Sell.
- Trade Manager - Update Prices, Edit Transaction, Database Health, Delete Trade, campaign views

### Risk Management
- Earnings Planner
- Portfolio Heat
- Risk Manager - Drawdown tracking with 3 hard deck levels (7.5%, 12.5%, 15%)

### Daily Workflow
- Daily Journal, Daily Report Card, Daily Routine, Weekly Retro

## Risk Manager Details
- `RESET_DATE = pd.Timestamp("2025-12-16")` appears in 3 places (Command Center, Risk Manager, Daily Report Card)
- Drawdown calculated from peak `End NLV` since RESET_DATE
- Hard decks: -7.5% (remove margin), -12.5% (max 30% invested), -15% (go to cash)

## Key Helper Functions (app.py)
- `load_trade_data()` (~line 322) - Shared data loader for trade pages
- `validate_trade_entry()` (~line 471) - Input validation for buy/sell
- `validate_position_size()` (~line 513) - Position size vs equity check
- `generate_trx_id()` (~line 348) - Auto-generate transaction IDs
- `update_campaign_summary()` (~line 378) - LIFO recalculation engine
- `log_audit_trail()` (~line 557) - Audit logging
- `secure_save()` (~line 146) - CSV save with backup

## Conventions
- Buy rules: `BUY_RULES` list (~line 83)
- Sell rules: `SELL_RULES` list (~line 116)
- Trade IDs: Format `YYYYMM-NNN` (e.g., `202602-001`)
- Transaction IDs: `B1`, `B2` for buys, `A1`, `A2` for add-ons, `S1` for sells
- TWR (Time-Weighted Return) used for performance calculation, immune to cash flows
