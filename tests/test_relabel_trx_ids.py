"""Unit tests for scripts/relabel_trx_ids.py.

The script's behavior splits into three layers:

  - classifier (pure function over a single trx_id string)
  - plan_for_trade (pure function over an ordered list of detail dicts)
  - apply / cache-clear / recompute (DB-touching; stubbed here)

These tests cover all three with synthetic in-memory fixtures so we
never touch the real database.
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from unittest import mock

import pytest

# scripts/ isn't a package — load the module directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "relabel_trx_ids", REPO_ROOT / "scripts" / "relabel_trx_ids.py"
)
relabel_trx_ids = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relabel_trx_ids)


# ────────────────────────────────────────────────────────────────────────
# classify(trx_id)
# ────────────────────────────────────────────────────────────────────────


class TestClassify:
    def test_canonical_simple(self):
        cat, prefix, has_extra, is_sub_1 = relabel_trx_ids.classify("B1")
        assert (cat, prefix, has_extra, is_sub_1) == ("CANONICAL", "B", False, False)

    def test_canonical_legacy_sa(self):
        assert relabel_trx_ids.classify("SA3") == ("CANONICAL", "SA", False, False)

    def test_canonical_legacy_sb(self):
        assert relabel_trx_ids.classify("SB7") == ("CANONICAL", "SB", False, False)

    def test_canonical_with_dedupe_suffix(self):
        assert relabel_trx_ids.classify("A4-2") == ("CANONICAL", "A", True, False)

    def test_canonical_with_auto_suffix(self):
        assert relabel_trx_ids.classify("B1-Auto") == ("CANONICAL", "B", True, False)

    def test_b0_pre_literal(self):
        assert relabel_trx_ids.classify("B0-Pre") == ("B0-Pre", None, False, False)

    def test_empty(self):
        assert relabel_trx_ids.classify("") == ("EMPTY", None, False, False)
        assert relabel_trx_ids.classify(None) == ("EMPTY", None, False, False)

    def test_freeform_sub_1_dedupe_double(self):
        """`{base}-Auto-N` is sub-bucket 1 — machine noise from running the
        dedupe script on an -Auto row."""
        cat, prefix, _, is_sub_1 = relabel_trx_ids.classify("S2-Auto-2")
        assert cat == "FREEFORM"
        assert is_sub_1 is True
        assert prefix is None  # sub_1 doesn't expose a prefix via classify()

    def test_freeform_sub_2_user_text(self):
        """Free-form user text is sub-bucket 2."""
        cat, _, _, is_sub_1 = relabel_trx_ids.classify("Market selloff and reducing exposure")
        assert cat == "FREEFORM"
        assert is_sub_1 is False

    def test_leading_zero_rejected_as_freeform(self):
        """The strict regex rejects B01 — same as TRX_ID_PATTERN."""
        cat, _, _, _ = relabel_trx_ids.classify("B01")
        assert cat == "FREEFORM"


# ────────────────────────────────────────────────────────────────────────
# infer_prefix_from_action
# ────────────────────────────────────────────────────────────────────────


class TestInferPrefix:
    def test_sell_always_S(self):
        assert relabel_trx_ids.infer_prefix_from_action({"action": "SELL"}, 0) == "S"
        assert relabel_trx_ids.infer_prefix_from_action({"action": "SELL"}, 5) == "S"

    def test_buy_first_in_trade_is_B(self):
        assert relabel_trx_ids.infer_prefix_from_action({"action": "BUY"}, 0) == "B"

    def test_buy_subsequent_is_A(self):
        assert relabel_trx_ids.infer_prefix_from_action({"action": "BUY"}, 2) == "A"

    def test_unknown_action_empty(self):
        assert relabel_trx_ids.infer_prefix_from_action({"action": "FOO"}, 0) == ""


# ────────────────────────────────────────────────────────────────────────
# _append_notes (NULL-safe " / " join)
# ────────────────────────────────────────────────────────────────────────


class TestAppendNotes:
    def test_prev_null_returns_addition(self):
        assert relabel_trx_ids._append_notes(None, "hello") == "hello"

    def test_prev_empty_returns_addition(self):
        assert relabel_trx_ids._append_notes("", "hello") == "hello"

    def test_prev_present_joins_with_separator(self):
        assert relabel_trx_ids._append_notes("Import", "(was B1-Auto)") == \
            "Import / (was B1-Auto)"

    def test_idempotent_when_addition_already_in_prev(self):
        """Re-running --apply twice mustn't double-append."""
        existing = "Import / (was B1-Auto)"
        assert relabel_trx_ids._append_notes(existing, "(was B1-Auto)") == existing


# ────────────────────────────────────────────────────────────────────────
# plan_for_trade — the core algorithm
# ────────────────────────────────────────────────────────────────────────


def _row(detail_id, trx_id, action="BUY", notes=None,
         created_at=None, portfolio="CanSlim", trade_id="202605-001"):
    """Compact factory for synthetic trades_details rows."""
    return {
        "portfolio": portfolio,
        "portfolio_id": 1,
        "trade_id": trade_id,
        "detail_id": detail_id,
        "trx_id": trx_id,
        "action": action,
        "shares": 100.0,
        "price": 50.0,
        "user_date": datetime(2026, 4, 1),
        "created_at": created_at or datetime(2026, 4, 1, 10, 0, 0),
        "notes": notes,
        "rule": "br1.1",
        "ticker": "TEST",
    }


class TestPlanCanonical:
    def test_all_canonical_no_op(self):
        rows = [
            _row(1, "B1"),
            _row(2, "B2", created_at=datetime(2026, 4, 1, 10, 1)),
            _row(3, "S1", action="SELL", created_at=datetime(2026, 4, 2)),
        ]
        plan = relabel_trx_ids.plan_for_trade(rows)
        assert plan["rename_updates"] == []
        assert plan["notes_updates"] == []

    def test_start_high_renumbers_to_one(self):
        """NBIS-024 pattern: A-rows start at A2."""
        rows = [
            _row(1, "B1"),
            _row(2, "A2", created_at=datetime(2026, 4, 1, 10, 1)),
            _row(3, "A3", created_at=datetime(2026, 4, 1, 10, 2)),
        ]
        plan = relabel_trx_ids.plan_for_trade(rows)
        renames = {(u["prev_trx_id"], u["new_trx_id"]) for u in plan["rename_updates"]}
        assert ("A2", "A1") in renames
        assert ("A3", "A2") in renames
        assert "START_HIGH" in plan["categories"]

    def test_gap_compacts_sequence(self):
        """A1, A2, A4 (A3 soft-deleted earlier) — renumber to A1, A2, A3."""
        rows = [
            _row(1, "A1"),
            _row(2, "A2", created_at=datetime(2026, 4, 1, 10, 1)),
            _row(3, "A4", created_at=datetime(2026, 4, 1, 10, 2)),
        ]
        plan = relabel_trx_ids.plan_for_trade(rows)
        renames = {(u["prev_trx_id"], u["new_trx_id"]) for u in plan["rename_updates"]}
        assert renames == {("A4", "A3")}
        assert "GAP" in plan["categories"]


class TestPlanDedupeAndAuto:
    def test_dedupe_suffix_folds_in(self):
        """A4-2 folds back as a regular A-row at its created_at slot."""
        rows = [
            _row(1, "A1"),
            _row(2, "A4", created_at=datetime(2026, 4, 1, 10, 1)),
            _row(3, "A4-2", created_at=datetime(2026, 4, 1, 10, 2)),
        ]
        plan = relabel_trx_ids.plan_for_trade(rows)
        renames = {(u["prev_trx_id"], u["new_trx_id"]) for u in plan["rename_updates"]}
        assert ("A4", "A2") in renames
        assert ("A4-2", "A3") in renames

    def test_auto_provenance_lifted_to_notes(self):
        """`-Auto` rows get `(was X)` appended to notes before rename."""
        rows = [
            _row(1, "B1-Auto", notes="Import"),
            _row(2, "S1-Auto", action="SELL", notes=None,
                 created_at=datetime(2026, 4, 1, 10, 1)),
        ]
        plan = relabel_trx_ids.plan_for_trade(rows)
        notes_lifts = {
            u["detail_id"]: u["new_notes"] for u in plan["notes_updates"]
        }
        assert notes_lifts[1] == "Import / (was B1-Auto)"
        assert notes_lifts[2] == "(was S1-Auto)"
        # Both also rename to canonical
        renames = {(u["prev_trx_id"], u["new_trx_id"]) for u in plan["rename_updates"]}
        assert ("B1-Auto", "B1") in renames
        assert ("S1-Auto", "S1") in renames

    def test_b0_pre_preserved_alongside_b_sequence(self):
        """B0-Pre is the literal; never renumbered. B-prefix sequence
        still starts at B1."""
        rows = [
            _row(1, "B0-Pre"),
            _row(2, "B2", created_at=datetime(2026, 4, 1, 10, 1)),
        ]
        plan = relabel_trx_ids.plan_for_trade(rows)
        renames = {(u["prev_trx_id"], u["new_trx_id"]) for u in plan["rename_updates"]}
        # B0-Pre is untouched; B2 renumbers to B1
        assert ("B2", "B1") in renames
        assert not any(u["prev_trx_id"] == "B0-Pre" for u in plan["rename_updates"])


class TestPlanFreeform:
    def test_sub_1_dropped_renumbered(self):
        """{base}-Auto-N rows are renumbered as canonical, text discarded.
        No notes-lift fires for these."""
        rows = [
            _row(1, "S1", action="SELL", notes="Import"),
            _row(2, "S2-Auto", action="SELL", notes="Import",
                 created_at=datetime(2026, 4, 1, 10, 1)),
            _row(3, "S2-Auto-2", action="SELL", notes="Import",
                 created_at=datetime(2026, 4, 1, 10, 2)),
        ]
        plan = relabel_trx_ids.plan_for_trade(rows)
        # Sub-1 row 3 should NOT trigger a notes-lift carrying "S2-Auto-2"
        sub_1_notes = [
            u for u in plan["notes_updates"]
            if u["detail_id"] == 3 and "S2-Auto-2" in (u["new_notes"] or "")
        ]
        assert sub_1_notes == [], "Sub-bucket 1 text must not lift to notes"
        # All three should renumber into the S sequence
        renames = {u["detail_id"]: u["new_trx_id"] for u in plan["rename_updates"]}
        assert renames.get(2) == "S2"
        assert renames.get(3) == "S3"

    def test_sub_2_lifts_text_then_renumbers(self):
        """Free-form user text is preserved into notes, row renumbers."""
        rows = [
            _row(1, "S1", action="SELL"),
            _row(2, "Market selloff and reducing exposure", action="SELL",
                 notes=None, created_at=datetime(2026, 4, 1, 10, 1)),
        ]
        plan = relabel_trx_ids.plan_for_trade(rows)
        # Notes-lift for the freeform row
        notes_lifts = {u["detail_id"]: u["new_notes"] for u in plan["notes_updates"]}
        assert notes_lifts[2] == "Market selloff and reducing exposure"
        # Renumber into S sequence (S2 because S1 is taken)
        renames = {u["detail_id"]: u["new_trx_id"] for u in plan["rename_updates"]}
        assert renames[2] == "S2"

    def test_sub_2_lifts_alongside_existing_notes(self):
        rows = [
            _row(1, "S1", action="SELL"),
            _row(2, "mistake entry", action="SELL", notes="trader said review",
                 created_at=datetime(2026, 4, 1, 10, 1)),
        ]
        plan = relabel_trx_ids.plan_for_trade(rows)
        notes_lifts = {u["detail_id"]: u["new_notes"] for u in plan["notes_updates"]}
        assert notes_lifts[2] == "trader said review / mistake entry"


class TestPlanIdempotency:
    def test_idempotent_after_rename(self):
        """Run the planner against the OUTPUT of a previous rename — should
        produce zero updates."""
        rows_pre = [
            _row(1, "A2", created_at=datetime(2026, 4, 1, 10, 1)),
            _row(2, "A4", created_at=datetime(2026, 4, 1, 10, 2)),
            _row(3, "A5", created_at=datetime(2026, 4, 1, 10, 3)),
        ]
        plan1 = relabel_trx_ids.plan_for_trade(rows_pre)
        # Simulate the apply: relabel each row.
        rename_map = {u["detail_id"]: u["new_trx_id"] for u in plan1["rename_updates"]}
        rows_post = [
            {**r, "trx_id": rename_map.get(r["detail_id"], r["trx_id"])}
            for r in rows_pre
        ]
        plan2 = relabel_trx_ids.plan_for_trade(rows_post)
        assert plan2["rename_updates"] == []
        assert plan2["notes_updates"] == []


# ────────────────────────────────────────────────────────────────────────
# cache_clear_and_recompute — hard-gate enforcement
# ────────────────────────────────────────────────────────────────────────


class TestCacheClearHardGate:
    def test_load_details_cache_cleared_before_recompute(self, monkeypatch):
        """Phase C MUST clear load_details + load_summary caches BEFORE
        calling _recompute_summary_lifo. Skipping is the documented
        silent-failure mode. Test asserts the call order."""
        call_log: list[str] = []

        def fake_clear_details():
            call_log.append("clear_details")

        def fake_clear_summary():
            call_log.append("clear_summary")

        def fake_recompute(portfolio, trade_id, ticker):
            call_log.append(f"recompute({portfolio},{trade_id},{ticker})")

        def fake_log_audit(*args, **kwargs):
            call_log.append("log_audit")

        monkeypatch.setattr(relabel_trx_ids.db.load_details, "clear",
                            fake_clear_details)
        monkeypatch.setattr(relabel_trx_ids.db.load_summary, "clear",
                            fake_clear_summary)

        plan = {
            "portfolio": "CanSlim",
            "trade_id": "202605-001",
            "ticker": "AAPL",
            "rename_updates": [{"prev_trx_id": "A2", "new_trx_id": "A1"}],
            "notes_updates": [],
        }
        relabel_trx_ids.cache_clear_and_recompute(plan, fake_recompute, fake_log_audit)

        # Both clears must happen before recompute, recompute before audit.
        assert call_log[0] in ("clear_details", "clear_summary")
        assert call_log[1] in ("clear_details", "clear_summary")
        assert call_log[0] != call_log[1]
        assert call_log[2].startswith("recompute(")
        assert call_log[3] == "log_audit"
