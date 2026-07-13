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

## Audit triggers and migrations
Any trigger that INSERTs into a tenant-scoped table (one with NOT NULL `user_id`
+ the migration-003 DEFAULT) must source `user_id` explicitly with a founder
fallback:

```sql
COALESCE(
    NULLIF(current_setting('app.user_id', true), '')::uuid,
    'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid
)
```

Migration sessions don't have `app.user_id` set; without the fallback the
trigger NOT NULL-violates and aborts the migration that fired it. See
[migrations/024_audit_trigger_migration_safe.sql](migrations/024_audit_trigger_migration_safe.sql)
for the canonical pattern. `migrations/run.py` also `SET LOCAL`s the founder
UUID per migration as defense in depth.

## Backtest workflow (standing)

When the user says "run a backtest" (or similar phrasing — "backtest X",
"let's backtest", etc.), do NOT assume any values. Ask these three questions,
**one at a time**, waiting for each answer before asking the next:

  1. What's your ticker?
  2. What's your NAV?
  3. What's your buy date?  (the actual entry date — NOT a "search from" date)

The buy date IS the entry — buy 20% NAV on the first trading day at/after that
date, regardless of MO RS signal state. The cascades then govern the hold/exit
from that bar forward. (A late buy after the initial GREEN is the expected use
case.)

**No end-date question.** The default mode is `terminate` — the campaign ends
on the first weekly GD, which is the natural end of the test. Don't pass `--end`.

Then run TWO backtests in parallel from the `mors/` directory — the default
rule set (SR7 ON) and a comparison run with SR7 disabled, so the user sees
both side-by-side every time:

```
cd mors && python3 mors_backtest.py --ticker <TICKER> --start <BUY_DATE> --nav <NAV> --spy data/SPY_daily.csv
cd mors && python3 mors_backtest.py --ticker <TICKER> --start <BUY_DATE> --nav <NAV> --spy data/SPY_daily.csv --no-sr7
```

**Output format:**
1. Show the FULL signal log + summary for the **default (SR7 ON)** run — no truncation
2. Then show a tight one-paragraph comparison: realized P&L for both, the delta,
   and any obvious explanation (e.g., "SR7 fired N times, locked $X early but
   shrank position for the late run").
Do not dump the full no-SR7 log — just the summary line and the SR7-fires line
from its CSV are usually enough for comparison.

Print the **full** stdout — the entire signal log table and the P&L summary,
no truncation, no summarizing. The script lives at
[mors/mors_backtest.py](mors/mors_backtest.py); CSVs are read from
[mors/data/](mors/data/).

**Auto-fetch:** if `mors/data/{TICKER}_price_data.csv` or `mors/data/SPY_daily.csv`
is missing, the script auto-downloads via yfinance (mirroring stock_scraper2).
Pass `--refresh` to force a fresh download for an existing ticker. No need to
pre-run a separate scraper.

## Daily monitor workflow (standing)

When the user says "run my monitor" / "daily monitor" / "check my positions" /
"daily report" or similar, ask these **two** questions, one at a time:

  1. Any position changes since the last run? (new buys, add-ons, partial sells,
     full exits, new positions) — answer "no" or describe them
  2. What's your NLV today?

**If there are changes**, walk through them BEFORE running the monitor and update
[mors/positions.json](mors/positions.json) accordingly:

- **New position**: append `{"ticker", "b1_date", "b1_price", "shares_held", "avg_price"}`.
  For a brand-new buy with no add-ons, `avg_price == b1_price` and `shares_held == initial shares`.
- **Add-on**: keep `b1_date` and `b1_price` unchanged (cushion anchor stays at B1).
  Update `shares_held` to new total and recompute `avg_price` as a weighted average:
  `new_avg = (old_shares * old_avg + add_shares * add_price) / (old_shares + add_shares)`.
  Confirm the new avg with the user before writing.
- **Partial sell**: update `shares_held` only; `avg_price` stays the same (avg cost
  convention; specific-lot accounting is tracked in the user's journal, not here).
- **Full exit**: remove that ticker's entry from positions.json entirely.

Always show the user the computed/updated values for their changed positions before
saving, so they can sanity-check (especially the recomputed avg_price after add-ons).

Then run, from the `mors/` directory:

```
cd mors && python3 monitor.py --nlv <NLV>
```

Print the full report — the action-needed section (if any) and the hold rows.
Don't truncate. The monitor writes a dated copy to [mors/results/](mors/results/)
as `monitor_YYYY-MM-DD.md`.

**Cascade selection (per position, based on current % NLV):**
- `>= 20% NLV` -> 20-cascade: 20% / 15% / 10% / 0% (GREEN/QUICK/QS/GD targets)
- `<  20% NLV` -> 15-cascade: 15% / 11.25% / 7.5% / 0%

**When a signal fires** the report's ACTION line already includes the exact
share count to sell (`TRIM X sh -> Y% NLV target`), computed from the user's
actual `shares_held` and the current price. This is the recommendation — the
user can act on it directly.

Add-ons are intentionally skipped from the cascade record. `shares_held` is the
current total, `b1_price` is the original first-buy fill price (anchors the
cushion / Phase 2 latch), `avg_price` is your cost basis (used for P&L display).

## Option price sync workflow (standing)

Every morning the user refreshes `trades_summary.manual_price` on their open
option positions. The prompts below name distinct broker-scoped sweeps —
each maps to one MCP flow the operator's Claude Code session can reach:

**`sync LTG prices`** — Long-Term Growth is at Robinhood joint account
`••••5578`. On this phrase:

  1. Call `mcp__claude_ai_Robinhood__get_option_positions` with
     `account_number="116841245578", nonzero=true`.
  2. Extract `option_id`s, then call
     `mcp__claude_ai_Robinhood__get_option_quotes` with those IDs.
  3. For each position, normalize to `{underlying, expiration, strike,
     option_type, mark_price}` — pull `expiration_date`, `chain_symbol` from
     the position, `mark_price` (or `adjusted_mark_price`) from the quote.
     Strike + option_type come from the position or quote's derived fields.
  4. Pipe the JSON list to
     `python scripts/mcp_sync_option_prices.py --portfolio "Long-Term Growth" --commit`.
  5. Report the printed match table + total unrealized delta. No confirmation
     needed — this is a small, easily reversible write.

**`sync CanSlim prices`** — CanSlim is at IBKR. On this phrase:

  1. Call `mcp__claude_ai_Interactive_Brokers_IBKR__get_account_positions`
     (no account arg — IBKR MCP is single-account scoped).
  2. Filter to `asset_class == "OPT"`.
  3. Parse each `contract_description` of the form
     `SYM Mon##'YY STRIKE (CALL|PUT) @EXCH` (e.g. `AMZN Oct16'26 240 CALL @AMEX`)
     — the `market_price` field on the position IS the current price;
     no separate quote call needed on IBKR.
  4. Normalize to the same shape as LTG.
  5. Pipe to `python scripts/mcp_sync_option_prices.py --portfolio "CanSlim" --commit`.
  6. Same reporting. No confirmation needed.

**`sync LTG trades`** — pulls new equity + option orders from Robinhood
since the most recent LTG transaction in the app, transforms via
`scripts/mcp_robinhood_to_csv.py`, posts to
`/api/imports/robinhood/preview`. **Stops at preview** — this is the
higher-risk flow (orphan-sell edge case). Wait for the user's "commit"
before running `/commit`. Same-day `--exclude` filter can be added if
they flag a specific dup.

**`sync LTG`** (both) = `sync LTG prices` immediately, then `sync LTG trades`
stopping at preview.

Helper scripts:
- `scripts/mcp_sync_option_prices.py` — dry-run default, `--commit` writes,
  matches by canonical `SYM YYMMDD $STRIKEC/P` ticker; 17 unit tests.
- `scripts/mcp_robinhood_to_csv.py` — transforms MCP JSON to
  Robinhood-format CSV for the existing `/api/imports/robinhood/*` endpoints;
  11 unit tests. Kept the CSV path as the shared write surface so future
  broker MCPs can piggy-back on the same importer + preview.

MCP tools are session-scoped; nothing about these workflows runs from cron.
The operator triggers each sync interactively.

## Backtest default rule set (active on every backtest):
- **Mode = `terminate`**: weekly GD ends the campaign, no re-entry. "Hold winners, exit clean."
- **Sizing = NLV-anchored**: every trim/refill is sized to (NAV + realized P&L) × 20% × cascade_frac.
  Locks in more on early signals as the position grows; refills smaller share counts as price rises.
- **SR7 ON, 2-bar arming**: layered proactive trim — on TWO consecutive daily closes below the
  21 EMA, set a trigger at `arming_bar_low × 0.99`. If the next intraday bar's low breaks the
  trigger, SR7 fires and trims the position back to 20% NLV core. Disarms on any single close
  back above the 21 EMA. Empirically helpful on most tickers in the basket; SNDK was the
  notable exception (parabolic shake-out at $194 cost ~$547k of upside).

## Alternate `--mode` options:
- `revert` — weekly GD reverts to Phase 1, awaits a daily GREEN, then opens a
  compounded sub-entry (20% × current NAV). Useful for A/B comparison.
- `legacy` — Phase 2 stays latched; weekly cascade re-arms on the next weekly
  GREEN and the position re-enters at the original lot size.

**Other flags:**
- `--no-sr7` disables SR7 (for A/B comparison vs MO RS alone)
- `--sr7-bars N` changes the consecutive-closes count to arm SR7 (default 2)
- `--end YYYY-MM-DD` caps the test window
- `--refresh` forces yfinance re-download for an existing ticker
- `--out ""` skips the CSV write (results land in [mors/results/](mors/results/) by default)

Ask the three questions fresh every time — never reuse prior answers.
