-- ============================================================================
-- Migration 018: partial UNIQUE index on (portfolio_id, trade_id, trx_id)
--                for active rows in trades_details
-- ============================================================================
-- Pairs with the collision-safe trx_id generator shipped in db_layer.py
-- (commit df6141a). The generator's advisory lock minimizes the race window
-- between SELECT and the caller's INSERT, but doesn't close it — the lock
-- auto-releases at the helper's commit, so two concurrent callers can still
-- receive the same trx_id and both attempt to INSERT it.
--
-- This index is the actual race-safety mechanism. With it in place, a
-- duplicate INSERT raises psycopg2.errors.UniqueViolation, which the caller
-- (api/main.py::_save_detail_with_unique_trx_id) catches and retries by
-- regenerating via the helper. Bounded retries converge on the next free
-- suffix; concurrent inserts can never silently produce duplicates again.
--
-- Why a partial unique INDEX (not a table-level UNIQUE constraint):
--   trades_details uses soft delete (deleted_at TIMESTAMPTZ). A user who
--   deletes a transaction and then re-logs the same campaign-step expects
--   the freed trx_id to be reusable — e.g. delete A1, then add a new lot
--   that gets A1 again. A table-level UNIQUE constraint enforces uniqueness
--   across ALL rows, including soft-deleted ones, which would block this
--   reuse pattern. Staging confirmed the failure mode: an active A1 row
--   collided with a soft-deleted A1 row in trade 202604-044.
--
--   A partial unique index `WHERE deleted_at IS NULL` enforces uniqueness
--   only on active rows, matching the semantics db.generate_unique_trx_id
--   already uses (its existing-trx_id scan filters on deleted_at IS NULL).
--   Soft-deleted rows are invisible to both the helper and the index, so
--   reuse is allowed; concurrent active inserts still fail loudly.
--
-- Production cleanup precedes this migration:
--   1. scripts/dedupe_trx_ids.py renamed 19 historical duplicate rows
--      (commit 5b0915f, applied via --apply against production)
--   2. The verification block below refuses to create the index if any
--      ACTIVE (portfolio_id, trade_id, trx_id) duplicates remain. The
--      verification scope matches the index scope: deleted_at IS NULL
--      and trx_id IS NOT NULL. Empty-string '' trx_ids ARE counted —
--      UNIQUE treats '' as a regular value, so multiple '' rows in one
--      active trade would still collide.
--
-- Operational order: deploy commit df6141a first (so live writes use the
-- collision-safe path), re-run scripts/dedupe_trx_ids.py to catch any
-- duplicates created during the deploy window, then run this migration.
-- If the verification fires, run dedupe again and retry.
-- ============================================================================

-- Verification: refuse to create the index if duplicates exist among ACTIVE
-- rows. Operator runs scripts/dedupe_trx_ids.py --apply if this raises.
DO $$
DECLARE
    dup_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO dup_count
    FROM (
        SELECT portfolio_id, trade_id, trx_id
        FROM trades_details
        WHERE deleted_at IS NULL
          AND trx_id IS NOT NULL  -- NULL doesn't collide under UNIQUE
        GROUP BY portfolio_id, trade_id, trx_id
        HAVING COUNT(*) > 1
    ) AS dups;

    IF dup_count > 0 THEN
        RAISE EXCEPTION
            'Cannot create unique_trx_id_per_trade — % duplicate (portfolio_id, trade_id, trx_id) groups exist among active rows. Run: python scripts/dedupe_trx_ids.py --portfolio <name> --apply',
            dup_count;
    END IF;
END $$;

CREATE UNIQUE INDEX unique_trx_id_per_trade
    ON trades_details (portfolio_id, trade_id, trx_id)
    WHERE deleted_at IS NULL;


-- ============================================================================
-- Verification queries (manual, after COMMIT)
-- ============================================================================
-- Expect: partial unique index present.
--   SELECT indexname, indexdef FROM pg_indexes
--   WHERE tablename = 'trades_details'
--     AND indexname = 'unique_trx_id_per_trade';
--
-- Expect: zero remaining duplicate groups among active rows.
--   SELECT portfolio_id, trade_id, trx_id, COUNT(*) AS n
--   FROM trades_details WHERE deleted_at IS NULL AND trx_id IS NOT NULL
--   GROUP BY portfolio_id, trade_id, trx_id HAVING COUNT(*) > 1;
--
-- Expect: insert of an active duplicate fails.
--   INSERT INTO trades_details (portfolio_id, trade_id, ticker, action, date, shares, amount, value, trx_id)
--   VALUES (1, 'TEST-DUPE', 'TEST', 'SELL', NOW(), 1, 1, 1, 'S1');
--   INSERT INTO trades_details (portfolio_id, trade_id, ticker, action, date, shares, amount, value, trx_id)
--   VALUES (1, 'TEST-DUPE', 'TEST', 'SELL', NOW(), 1, 1, 1, 'S1');  -- should ERROR with unique_trx_id_per_trade
--   DELETE FROM trades_details WHERE trade_id = 'TEST-DUPE';  -- cleanup
--
-- Expect: soft-deleted duplicates are tolerated (the soft-deleted row is
-- outside the index's WHERE clause).
--   INSERT INTO trades_details (portfolio_id, trade_id, ticker, action, date, shares, amount, value, trx_id, deleted_at)
--   VALUES (1, 'TEST-SOFT', 'TEST', 'SELL', NOW(), 1, 1, 1, 'S1', NOW());
--   INSERT INTO trades_details (portfolio_id, trade_id, ticker, action, date, shares, amount, value, trx_id)
--   VALUES (1, 'TEST-SOFT', 'TEST', 'SELL', NOW(), 1, 1, 1, 'S1');  -- should SUCCEED — active vs soft-deleted
--   DELETE FROM trades_details WHERE trade_id = 'TEST-SOFT';  -- cleanup
