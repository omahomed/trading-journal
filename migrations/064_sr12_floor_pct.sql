-- ============================================================================
-- Migration 064: sr12_floor_pct — ratcheting profit floor (MCP)
-- ============================================================================
-- Introduces SR12 (Ratcheting Profit Floor) — a "Mental Capital Preservation"
-- disaster backstop that sits underneath the trend exits (SR7/SR8), not in
-- place of them. Named after the LifeCycle Trade MCP concept.
--
-- Doctrine (2026-08-07 handoff):
--   Once b1_max_return_pct crosses +50%, arm a floor at 50% of the peak.
--   The floor ratchets up on every new peak; it never moves down. If price
--   breaks the floor intraday, exit mechanically.
--
--   Continuous rule: floor_pct = peak_pct / 2. The "tier view" in the
--   doctrine handoff (+50→+25, +100→+50, +200→+100) is presentation of the
--   same continuous formula.
--
--   Orthogonal to the tier ladder (SR1/SR11/SR15/SR7/SR8) — a position can
--   be SR7-armed AND SR12-armed simultaneously. The ACS row shows the chip
--   + a nudge banner when the physical broker_stop_price lags the floor.
--
-- Column semantics:
--   sr12_floor_pct — the ratcheted floor as a percent of B1 entry price.
--                    Persisted (not derived) so a ratchet from a peak that
--                    later gets recomputed away can't lower the floor.
--                    NULL = not yet armed (peak never crossed +50%).
--
-- Peak source: reuses b1_max_return_pct (intraday max, bar-derived). Per
-- the user's 2026-08-07 call: "if the price went there, that's the price."
-- The alternative (a closing-highs-only column) was considered and rejected
-- for simplicity — if a rogue spike wick sets an unreachable floor, the
-- operator can manually override via the trade edit form.
--
-- Slot choice: SR12 (freed by migration 063 when TQQQ Strategy Exit was
-- retired). SR14 was considered — the trade-rules.ts comment reserves that
-- slot for "SR11-R / MCP" — but 15 lingering old-SR14 references in the
-- frontend (log-buy, position-sizer, active-campaign) would confuse readers.
-- SR12 is completely clean.
--
-- Backfill: any open campaign whose b1_max_return_pct is already >= 50 gets
-- sr12_floor_pct seeded to peak/2 in this migration. The b1_reconcile loop
-- keeps it ratcheted from there on subsequent runs.
--
-- The migration runner wraps this file in a transaction; no BEGIN/COMMIT.
-- ============================================================================

ALTER TABLE trades_summary
    ADD COLUMN IF NOT EXISTS sr12_floor_pct DECIMAL(10, 4);

COMMENT ON COLUMN trades_summary.sr12_floor_pct IS
    'SR12 Ratcheting Profit Floor (MCP). Ratcheted floor as a percent of B1 '
    'entry. Armed on first b1_max_return_pct >= 50 (floor = peak / 2); '
    'ratchets up on every new peak; never moves down. Fires an exit when '
    'price breaks below b1_entry * (1 + sr12_floor_pct/100) intraday. '
    'Orthogonal to the SR1/SR11/SR15/SR7/SR8 tier ladder — a position can '
    'be SR7-armed and SR12-armed simultaneously. NULL = not yet armed. '
    'See migration 064.';

-- Backfill: seed the floor for any already-cushion-qualified open campaign.
-- Idempotent — the WHERE clause guards against overwriting an existing
-- (higher) floor. Uses the same "ratchet only up" semantic that the runtime
-- update helper will enforce.
UPDATE trades_summary
   SET sr12_floor_pct = ROUND((b1_max_return_pct / 2.0)::numeric, 4)
 WHERE deleted_at IS NULL
   AND b1_max_return_pct IS NOT NULL
   AND b1_max_return_pct >= 50
   AND (sr12_floor_pct IS NULL OR sr12_floor_pct < b1_max_return_pct / 2.0);


-- ============================================================================
-- Verification queries (manual, after COMMIT)
-- ============================================================================
--   SELECT column_name, data_type, is_nullable, column_default
--     FROM information_schema.columns
--    WHERE table_name = 'trades_summary' AND column_name = 'sr12_floor_pct';
--   → numeric(10,4), YES, null
--
--   SELECT p.name, s.trade_id, s.ticker, s.b1_max_return_pct, s.sr12_floor_pct
--     FROM trades_summary s
--     JOIN portfolios p ON p.id = s.portfolio_id
--    WHERE s.deleted_at IS NULL
--      AND s.sr12_floor_pct IS NOT NULL
--   ORDER BY s.b1_max_return_pct DESC;
--   → Every cushion-qualified open campaign should have sr12_floor_pct set
--     to exactly b1_max_return_pct / 2. DELL (peak ~166) should show ~83.
