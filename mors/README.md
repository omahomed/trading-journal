# MO RS 2.0 Shadow Backtester

Tests the pure **MO RS 2.0** exit framework on one ticker against a stored SPY
benchmark. Drop in a ticker CSV, set a start date, run. SPY lives as a file —
you never re-feed it.

## Folder layout
```
mors/
  mors_backtest.py
  data/
    spy_daily.csv              <- your SPY pull, renamed. The permanent spine.
    PLTR_price_data.csv        <- one file per ticker: {TICKER}_price_data.csv
```
Your `stock_scraper.py` already produces the right format. Just rename the SPY
output to `spy_daily.csv` and drop ticker outputs into `data/`.

## Run
```bash
python3 mors_backtest.py --ticker PLTR --start 2023-01-01 --nav 500000
python3 mors_backtest.py --ticker PLTR --start 2023-01-01 --nav 500000 --end 2025-12-31
```
- `--start` : entry is the **first daily GREEN at or after this date** (= B1).
- `--nav`   : portfolio NAV. Entry deploys 20% of it; targets read in real dollars
              (20% Green / 15% Quick / 10% Quicksand / 0% GD). Default $500K.
- `--end`   : test end. Defaults to the last date both files share.

No subfolders? Keep everything flat and point at the current directory:
```bash
python3 mors_backtest.py --ticker PLTR --start 2023-01-01 --nav 500000 --data . --spy spy_daily.csv
```

## Updating SPY (the "don't re-feed" part)
SPY is read from `data/spy_daily.csv` every run. If you ask for an `--end` past
what SPY covers, the engine **stops and tells you**:
```
[STOP] SPY data only covers through 2026-05-29.
       Requested end date is 2026-08-01.
       Append SPY rows through 2026-08-01 to data/spy_daily.csv, then re-run.
```
Re-run your scraper for SPY, overwrite `data/spy_daily.csv`, done. The file is
the memory — nothing else to maintain.

## Indicator (MO RS 2.0, from MO_RS.pine)
RS = ticker_close / SPY_close. Three EMAs per timeframe:
daily 21/34/50, weekly 8/13/21, monthly 5/8/13 (monthly informational only).
Cascade fires once per cycle, re-arms only on GREEN (RS back above all three):
GREEN -> QUICK (under fast) -> QUICKSAND (under mid) -> GD (under slow).

## Shadow rules encoded
- Entry: first daily GREEN >= start date, 1.0 unit (= 20% NAV = $100K notional).
- Both phases, same target cascade: QUICK -> 75% unit, QUICKSAND -> 50%, GD -> 0,
  GREEN -> rebuild to 100%.
- Phase 1 (cushion < +50% from B1): DAILY MO RS, act on close.
- Phase 2 (cushion >= +50%, latched intra-bar at 1.50 x B1): WEEKLY MO RS, Fri close.
- LIFO accounting on trims.

## Build decisions baked in (flip these in code if you want them different)
- Entry deploys `POSITION_PCT` (20%) of the `--nav` you pass. Single-name
  normalization; % return is invariant to NAV, dollars scale with it. NAV here
  is the at-entry reference, not a dynamically growing portfolio (that's the
  full portfolio sim, later, in Code).
- GD is **reversible** — a later GREEN rebuilds to 100%. (SR8's "GD terminal"
  is a MAIN-trade rule; the shadow rebuilds on green.)
- At the Phase 1->2 switch the weekly cascade is **seeded silently** from the
  current RS-vs-weekly-MA standing, so only NEW weekly crosses after the switch
  trade (no double-counting a cross that already happened in Phase 1).
- `WARMUP_DAYS` = 50 daily RS bars before an entry is allowed.

## Scaling to many tickers later
Loop the `run()` call over a folder of ticker CSVs and collect the summary line
per name. At hundreds of names this stays fast in plain Python. (No need for an
artifact — the file-based SPY spine is the whole point.)
