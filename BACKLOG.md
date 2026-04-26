# Backlog

Tracked follow-up cleanup work that's not blocking but should be picked up
when convenient. Add new items as one-line bullets — no roadmap, no
estimates, just things to do.

## V10 vocabulary cleanup (post-MCT-V11 Phase 4)

- **A. `db_layer.load_market_signals` is dead V10 code.** References columns
  (`market_exposure`, `buy_switch`, `distribution_count`, ...) that no longer
  exist on the `market_signals` table after migration 010 dropped + recreated
  it. Only caller is the utility script `check_spy_dates.py`. Delete the
  function and update or delete `check_spy_dates.py`.

- **B. `schema.sql` still shows the V10 `market_signals` schema.** Migration
  010 dropped the V10 table and recreated it with the V11 vocabulary
  (`trade_date, signal_type, signal_label, exposure_before, exposure_after,
  state_before, state_after, meta`). Update `schema.sql` to match. Comment
  header at line 209 already flags this as out of date.

- **C. Frontend V10 vocabulary leaks outside Phase 4 scope.** Phase 4 cleaned
  up the four spec'd surfaces (market-cycle, daily-report-card, daily-journal,
  tape pill) but left untouched:
  - `frontend/src/components/dashboard.tsx:292` — duplicate "Tape" pill
    rendering V10 `market_window`; redundant with the new global tape pill
  - `frontend/src/components/daily-routine.tsx:159` — sends
    `market_window: ""` in the journalEdit save payload
  - `frontend/src/components/admin.tsx:454` — admin description copy
    references `market_window` as a computed field
  - `frontend/src/lib/api.ts:54` — `market_window: string` field on the
    `JournalHistoryPoint` type (the journal/history endpoint still returns
    the field per Phase 3a "preserve historical data" behavior)
