# ARCHITECTURE — Coupling Map

## Purpose

This is **not** a full architecture doc. It's a map of the **load-bearing joints** —
the DB tables, engine functions, and API endpoints that many things depend on,
so a change here can silently break unrelated pages.

**Consult before**: writing code that touches any listed surface.
**Update when**: adding a new writer/reader to a listed surface, OR when a new
surface earns 3+ dependents.

Standing rules that pair with this doc live in [CLAUDE.md](CLAUDE.md):
"No silent-default guard on money fields," "Migration audit checklist,"
"Engine-behavior overrides audit."

---

## 1. `trading_journal` table

The most fanned-out surface — 5 writers, 6 backend readers, ~10 frontend pages.

**Writers** (all in [db_layer.py](db_layer.py)):

| Function | Behavior | Notes |
|---|---|---|
| `save_journal_entry` | Full-row UPSERT | NLV Entry, journal_edit, journal_batch_edit |
| `save_journal_game_plan` | Hollow INSERT ok | Creates (portfolio_id, day, game_plan) row before NLV lands |
| `update_journal_mct_state` | Targeted UPDATE | market_cycle + mct_display_day_num only; used by heal + save |
| `update_journal_trend_state` | Targeted UPDATE | trend_count only |
| `delete_journal_entry` | Soft delete | Sets deleted_at, doesn't purge |

**Backend readers** ([api/main.py](api/main.py)):

- `journal_latest` — filters to `end_nlv > 0`
- `journal_history` — filters to `end_nlv > 0`; fires `_heal_recent_mct_stamps` (writes back!)
- `journal_edit` — reads existing before update
- `journal_batch_edit`
- `portfolio_heat_preview` — filters to `end_nlv > 0`
- `_heal_recent_mct_stamps` — reads to detect stale stamps, WRITES back via targeted updaters

**Frontend consumers** (via `api.journalLatest()` / `api.journalHistory()`):

Portfolio Heat, Position Sizer, Log Buy, Trade Journal, Dashboard,
NLV Entry, Daily Journal, Journal Log, Weekly Retro, Risk Manager,
Active Campaign, Analytics, Perf Heatmap, Earnings Planner.

**Key invariant**: "logged day" = row where `end_nlv IS NOT NULL AND > 0`.
Applied server-side so no frontend re-implements. Hollow rows written by
`save_journal_game_plan` stay in DB but are invisible to consumers until
NLV lands and promotes them.

**Recent bugs**:
- Migration 054 (LTG reset) wiped rows → frontends fell back to fake $100k
- Game Plan hollow rows leaked into Journal Log + heat-preview
- Save-time MCT stampers ignored override → Journal Log stamp diverged from M Factor

---

## 2. MCT Engine (`api/mct_engine.py` + `api/mct_endpoint_adapter.py`)

**Single entry point**: `run_engine(symbol, as_of, force_correction_at_date)` in
[api/mct_endpoint_adapter.py](api/mct_endpoint_adapter.py).

**Config** (`EngineConfig` dataclass): adding a new field means every `run_engine`
caller must be updated. Python typing doesn't enforce this — grep it manually.

**`run_engine` callers** (must ALL stay in sync when engine behavior changes):

- `rally_prefix` endpoint — M Factor page + tape pill
- `mct_state_by_date_range` endpoint
- `_compute_mct_state_with_day_num` — save-time state stamper
- `_compute_trend_count` — save-time trend stamper
- `_heal_recent_mct_stamps` — heal path on /api/journal/history (writes)
- `_run_engine_and_write_signals` in [api/market_data_updater.py](api/market_data_updater.py)
- [scripts/replay_mct.py](scripts/replay_mct.py)
- [scripts/backfill_mct_state.py](scripts/backfill_mct_state.py)
- [scripts/backfill_trend_count.py](scripts/backfill_trend_count.py)

**Override plumbing**: `_current_override_date()` in [api/main.py](api/main.py)
is a READ-ONLY lookup for the active Force Correction override. Must be
passed to every stamp/heal `run_engine` call. Auto-clear logic lives
ONLY in `rally_prefix` (single writer, no race with save-time paths).

**Key invariant**: given the same date, same override state, and same
market_data, all readers return the same state. Journal Log's stamped
`market_cycle` must equal M Factor's displayed `state`.

**Recent bugs**:
- Force Correction override wired only to `rally_prefix` → stampers stayed systematic
- Heal only re-stamped NULL rows → didn't fix rows stamped before override activated
- Fixed 2026-07-27: heal now re-stamps when stored state != engine state

---

## 3. `/api/journal/latest` endpoint

**Contract**: latest `trading_journal` row where `end_nlv > 0`. `?before=YYYY-MM-DD`
returns the latest such row strictly before that date.

**Frontend consumers** (`api.journalLatest()`):

- portfolio-heat.tsx — equity basis
- position-sizer.tsx — Account Equity prefill
- log-buy.tsx — Account Equity display + submit guard
- trade-journal.tsx — POS SIZE % denominator
- dashboard.tsx — metric tile
- nlv-entry.tsx — `prev_end_nlv` baseline for daily-change diff

**Change impact**: any change to the "what counts as a logged day" rule
affects all 6. This filter is the single source of truth.

---

## 4. `/api/journal/history` endpoint

**Contract**: same NLV filter as `journal_latest`, returned as a list.
**Fires `_heal_recent_mct_stamps` before returning** — so a GET can WRITE
to `trading_journal.market_cycle` / `.mct_display_day_num` / `.trend_count`.

**Frontend consumers**:

- journal-log.tsx (Journal Log historical browse)
- dashboard.tsx (equity curve)
- weekly-retro.tsx (auto-populate week rows)
- daily-journal.tsx (LTD %, historical context)
- period-review.tsx, trend-cycle-review.tsx

**Watch out**: any change to heal logic affects downstream reads
immediately — because the read triggers the write.

---

## 5. `/api/portfolio/heat-preview` endpoint

**Contract**: computes `_compute_portfolio_heat(portfolio, "", equity)`
against the latest NLV-bearing row's `end_nlv`. Returns `{heat, nlv_used}`.

**Frontend consumers**: nlv-entry.tsx (Daily Routine card preview).

**Watch out**: must apply the same NLV filter as `journal_latest`. When
they disagree, NLV Entry's preview tile shows a different heat than the
Portfolio Heat page.

---

## 6. `/api/market/rally-prefix` endpoint

**Contract**: current M Factor state + entry ladder + reference high +
drawdown + FTD (current cycle only). Applies `_current_override_date()`.
Auto-clears the override when systematic state recovers.

**Frontend consumers** (`api.rallyPrefix()`):

- m-factor.tsx — full M Factor page
- tape-status-pill.tsx — desktop tape pill
- mobile-tape-pill.tsx — mobile tape pill
- position-sizer.tsx — auto-mode selection
- log-buy.tsx — sizing mode indicator
- rally-context.tsx — sidebar chip
- risk-manager.tsx — regime badge
- sr8-monitor.tsx — cascade selection
- earnings-planner.tsx — regime aware

**Recent bugs**:
- FTD date leaked between rally cycles → gated to current cycle (2026-07-27)
- Drawdown was close-based, should be peak-to-lowest-low → fixed (2026-07-27)

---

## 7. `portfolios` table (foreign key parent)

Tables with `portfolio_id` FK:

- `trading_journal` → `daily_journal_captures` (CASCADE)
- `trades_summary` → `trades_details` (CASCADE)
- `cash_transactions` (cascade)
- `trade_images`, `trade_fundamentals`, `trade_lessons`, `lot_closures`

**Reset migration canonical shape**: see
[migrations/054_reset_long_term_growth_2026_07_27.sql](migrations/054_reset_long_term_growth_2026_07_27.sql).
DELETE from children before parent. `SET LOCAL app.user_id` for RLS.

---

## 8. `market_data` table (^IXIC bars)

**Writers**: `_upsert_rows` in [api/market_data_updater.py](api/market_data_updater.py)
— idempotent yfinance ingest.

**Readers**: everything that runs the engine (see §2).

**Gates**:
- `_last_business_day()` — post-close (22:00 UTC) settled-bar rule; pre-close
  reads target the prior weekday to avoid stamping an intraday snapshot as
  the settled close.
- `_patch_lagged_close_from_info` — falls back to `yf.Ticker.info` when
  `yf.history` returns NaN close on a settled bar.
- Intraday NaN closes are filtered at write time.

---

## 9. `getActivePortfolio()` — frontend state

**Source**: [frontend/src/lib/api.ts](frontend/src/lib/api.ts), reads from LocalStorage.

**Consumers**: every page that calls a portfolio-scoped endpoint.

**Watch out**: switching active portfolio doesn't invalidate in-flight cached
queries. If a page loads before the switch, stale portfolio data may render
briefly. React Query is not currently used across the app.

---

# Change patterns — "if you touch X, remember Y"

## Migration that DELETEs from a tenant-scoped table
(trading_journal, trades_summary, trades_details, portfolios, cash_transactions)

1. `rg 'journalLatest|tradesOpen|batchPrices|journalHistory' frontend/src/`
2. Verify every consumer handles empty gracefully — no silent `|| <magic-number>` fallbacks
3. See CLAUDE.md "Migration audit checklist"

## Adding a new field to `EngineConfig`

1. `rg 'run_engine\(' api/ scripts/` — every caller needs the field
2. Update `_default_config` — the plumbing point
3. Verify stamp/heal callers know how to source the field's value

## Adding a new writer to `trading_journal`

1. Does it create hollow rows (INSERT with sparse columns, like `save_journal_game_plan`)?
2. If yes, verify `journal_latest`, `journal_history`, and `portfolio_heat_preview`
   filter it out via the `end_nlv > 0` rule

## Adding a new frontend page that needs equity/NLV

1. Read from `api.journalLatest()` — don't create a parallel path
2. Never default to a hardcoded number (see CLAUDE.md "No silent-default guard")
3. Empty state pattern: bail hard for display pages, block submit for trade forms

## Changing what the engine returns for a given date
(new SR7 tuning, new override type, threshold tweak)

1. All `run_engine(` callers must apply the same rules
2. The heal (`_heal_recent_mct_stamps`) re-stamps stale rows automatically once
   the engine change lands, as long as its `run_engine` call also gets the new config

---

# Contract tests

Tests that lock the "these things must agree" invariants. Adding new invariants
here is **cheaper** than debugging class-of-bug regressions.

**Existing**:

- [tests/test_journal_latest_nlv_filter.py](tests/test_journal_latest_nlv_filter.py)
  — journal_latest, journal_history, heat-preview all apply the same NLV filter
- [tests/test_mct_override_stamp_plumbing.py](tests/test_mct_override_stamp_plumbing.py)
  — every engine reader applies the override + heal re-stamps stale rows

**Missing / worth adding** (rough priority):

1. `_compute_mct_state_with_day_num(D)` state == `rally_prefix(as_of=D)` state
   (given same override) — catches "stampers and endpoint drift" the moment it happens
2. `journalLatest().end_nlv == portfolioHeatPreview().nlv_used` for a portfolio
   with a mix of hollow + logged rows
3. Static grep test: every `run_engine(` call site accepts `force_correction_at_date`
4. Every page in the "trading_journal frontend consumers" list reads NLV from
   `journalLatest` — no parallel `load_journal` fetches

---

# Updating this doc

- Add a surface only after it earns 3+ dependents. Before that, it's not load-bearing.
- Add a change pattern when a bug of that class ships in prod (retrospective, not prescriptive).
- Keep under ~350 lines total — must be re-readable in one session.
- When in doubt, prefer deleting a stale section over patching it.
