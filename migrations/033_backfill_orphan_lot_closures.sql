-- migrations/033_backfill_orphan_lot_closures.sql
--
-- Hard-delete lot_closures rows whose parent trades_summary is
-- soft-deleted. These rows were stranded by delete_trade before the
-- cascade fix in the preceding commit.
--
-- lot_closures has no deleted_at column (per migration 017). The
-- cleanup paradigm is DELETE, mirroring the delete-then-insert
-- recompute path in save_summary_with_closures.
--
-- Verified count at audit time: 2 rows (trade 202605-004, portfolio
-- CanSlim). Production count may differ; the WHERE clause is the same.
--
-- Idempotent: re-running matches zero rows on the second invocation
-- because the orphan-producing condition has been fixed in
-- db.delete_trade.

DO $$
DECLARE
    before_count integer;
    after_count  integer;
BEGIN
    SELECT count(*) INTO before_count
      FROM lot_closures lc
      JOIN trades_summary ts
        ON ts.portfolio_id = lc.portfolio_id
       AND ts.trade_id     = lc.trade_id
     WHERE ts.deleted_at IS NOT NULL;

    RAISE NOTICE 'Orphan lot_closures before delete: %', before_count;

    DELETE FROM lot_closures lc
     USING trades_summary ts
     WHERE ts.portfolio_id = lc.portfolio_id
       AND ts.trade_id     = lc.trade_id
       AND ts.deleted_at IS NOT NULL;

    SELECT count(*) INTO after_count
      FROM lot_closures lc
      JOIN trades_summary ts
        ON ts.portfolio_id = lc.portfolio_id
       AND ts.trade_id     = lc.trade_id
     WHERE ts.deleted_at IS NOT NULL;

    RAISE NOTICE 'Orphan lot_closures after delete:  %', after_count;
END $$;
