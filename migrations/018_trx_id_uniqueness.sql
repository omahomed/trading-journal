-- ============================================================================
-- Migration 018: UNIQUE (portfolio_id, trade_id, trx_id) on trades_details
-- ============================================================================
-- Pairs with the collision-safe trx_id generator shipped in db_layer.py
-- (commit df6141a). The generator's advisory lock minimizes the race window
-- between SELECT and the caller's INSERT, but doesn't close it — the lock
-- auto-releases at the helper's commit, so two concurrent callers can still
-- receive the same trx_id and both attempt to INSERT it.
--
-- This constraint is the actual race-safety mechanism. With it in place, a
-- duplicate INSERT raises psycopg2.errors.UniqueViolation, which the caller
-- (api/main.py::_save_detail_with_unique_trx_id) catches and retries by
-- regenerating via the helper. Bounded retries converge on the next free
-- suffix; concurrent inserts can never silently produce duplicates again.
--
-- Production cleanup precedes this migration:
--   1. scripts/dedupe_trx_ids.py renamed 19 historical duplicate rows
--      (commit 5b0915f, applied via --apply against production)
--   2. The verification block below refuses to add the constraint if any
--      (portfolio_id, trade_id, trx_id) duplicates remain (NULL trx_ids
--      are skipped — UNIQUE treats NULLs as distinct, so multiple NULL
--      trx_ids in a trade don't collide. Empty-string '' trx_ids ARE
--      counted because UNIQUE treats '' as a regular value).
--
-- Operational order: deploy commit df6141a first (so live writes use the
-- collision-safe path), re-run scripts/dedupe_trx_ids.py to catch any
-- duplicates created during the deploy window, then run this migration.
-- If the verification fires, run dedupe again and retry.
-- ============================================================================

-- Verification: refuse to add the constraint if duplicates exist. Operator
-- runs scripts/dedupe_trx_ids.py --apply if this raises.
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
            'Cannot add unique_trx_id_per_trade — % duplicate (portfolio_id, trade_id, trx_id) groups exist. Run: python scripts/dedupe_trx_ids.py --portfolio <name> --apply',
            dup_count;
    END IF;
END $$;

ALTER TABLE trades_details
    ADD CONSTRAINT unique_trx_id_per_trade
    UNIQUE (portfolio_id, trade_id, trx_id);


-- ============================================================================
-- Verification queries (manual, after COMMIT)
-- ============================================================================
-- Expect: constraint present.
--   SELECT conname, contype FROM pg_constraint
--   WHERE conrelid = 'trades_details'::regclass
--     AND conname = 'unique_trx_id_per_trade';
--
-- Expect: zero remaining duplicate groups.
--   SELECT portfolio_id, trade_id, trx_id, COUNT(*) AS n
--   FROM trades_details WHERE deleted_at IS NULL AND trx_id IS NOT NULL
--   GROUP BY portfolio_id, trade_id, trx_id HAVING COUNT(*) > 1;
--
-- Expect: insert of a duplicate fails.
--   INSERT INTO trades_details (portfolio_id, trade_id, ticker, action, date, shares, amount, value, trx_id)
--   VALUES (1, 'TEST-DUPE', 'TEST', 'SELL', NOW(), 1, 1, 1, 'S1');
--   INSERT INTO trades_details (portfolio_id, trade_id, ticker, action, date, shares, amount, value, trx_id)
--   VALUES (1, 'TEST-DUPE', 'TEST', 'SELL', NOW(), 1, 1, 1, 'S1');  -- should ERROR with unique_trx_id_per_trade
--   DELETE FROM trades_details WHERE trade_id = 'TEST-DUPE';  -- cleanup
