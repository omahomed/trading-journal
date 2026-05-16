-- migrations/035_sell_rule_strays_cleanup.sql
--
-- Followup to migration 034. Three non-canonical sell-rule strays
-- in trades_details.rule (action='SELL') were intentionally left
-- outside 034's locked mapping pending user decision. User has now
-- locked the mapping: all three map to sr1 Capital Protection.
--
-- Source values and pre-migration counts (verified against live DB
-- after 034 deployed):
--   'sr10 Scale-Out T1 (-3%)'  5 rows
--   'sr11 Scale-Out T2 (-5%)'  1 row
--   'IBKR'                     1 row  (data anomaly, not a real rule)
--
-- Scope: trades_details.rule only, action='SELL', deleted_at IS NULL.
-- Per 034's verification, none of these strays currently mirror to
-- trades_summary.sell_rule, so no summary update is needed. The
-- action='SELL' filter is mandatory: trades_details.rule is dual-
-- purpose (it holds buy-rule strings on BUY rows too), and we must
-- not touch BUY rows.
--
-- Idempotent: re-running matches zero rows because the source values
-- are no longer present in the data after the first run.

DO $$
DECLARE
    before_count integer;
    after_count  integer;
BEGIN
    SELECT count(*) INTO before_count
      FROM trades_details
     WHERE action = 'SELL'
       AND deleted_at IS NULL
       AND rule IN (
           'sr10 Scale-Out T1 (-3%)',
           'sr11 Scale-Out T2 (-5%)',
           'IBKR'
       );
    RAISE NOTICE 'Stray sell-rule details before cleanup: %', before_count;

    UPDATE trades_details
       SET rule = 'sr1 Capital Protection'
     WHERE action = 'SELL'
       AND deleted_at IS NULL
       AND rule IN (
           'sr10 Scale-Out T1 (-3%)',
           'sr11 Scale-Out T2 (-5%)',
           'IBKR'
       );

    SELECT count(*) INTO after_count
      FROM trades_details
     WHERE action = 'SELL'
       AND deleted_at IS NULL
       AND rule IN (
           'sr10 Scale-Out T1 (-3%)',
           'sr11 Scale-Out T2 (-5%)',
           'IBKR'
       );
    RAISE NOTICE 'Stray sell-rule details after cleanup:  %', after_count;
END $$;
