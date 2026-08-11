"""Backend cap enforcement for slice target_pct sums.

Frontend already validates the 100% cap in the Manage Slices modal
(see frontend/src/components/slices.test.tsx). These tests lock the
backend guard so a direct API call can't push a portfolio over 100%.

Coverage:
  * _reject_if_over_cap accepts sums at / under 100 (float tolerance)
  * _reject_if_over_cap raises 422 when the sum exceeds 100
  * slices_create refuses when new slice would push parent over cap
  * slices_update refuses target_pct increases that push over cap
  * slices_update ALLOWS decreases even from an over-cap starting state
    (needed to trim the portfolio back into cap)
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import HTTPException


def test_reject_if_over_cap_accepts_at_or_under_100():
    """Sums up to 100 pass. Float-tolerance allows tiny rounding overshoot
    (e.g. 100.0000000001) since target_pct is stored as a decimal."""
    from api.main import _reject_if_over_cap
    # Under
    _reject_if_over_cap(50.0, "roots")
    _reject_if_over_cap(99.99, "roots")
    # At the boundary
    _reject_if_over_cap(100.0, "roots")
    # Float noise just above 100 — accepted by the epsilon
    _reject_if_over_cap(100.0 + 1e-9, "roots")


def test_reject_if_over_cap_raises_over_100():
    """Any real overage → HTTPException 422 with helpful detail text."""
    from api.main import _reject_if_over_cap
    with pytest.raises(HTTPException) as exc:
        _reject_if_over_cap(115.0, "roots")
    assert exc.value.status_code == 422
    assert "over the 100% cap" in exc.value.detail
    assert "115" in exc.value.detail
    assert "roots" in exc.value.detail


def test_slices_create_refuses_when_would_exceed_cap():
    """Trying to add a 15% root slice when siblings already sum to 90%
    → 422, and db.create_slice is NEVER called."""
    from api import main
    with patch.object(main, "_resolve_portfolio_id", return_value=42), \
         patch.object(main, "_sibling_target_pct_sum", return_value=90.0), \
         patch("db_layer.create_slice") as mock_create:
        with pytest.raises(HTTPException) as exc:
            main.slices_create(
                request=None,
                body={"portfolio": "LTG", "parent_id": None,
                      "name": "TooBig", "target_pct": 15},
            )
        assert exc.value.status_code == 422
        assert "over the 100% cap" in exc.value.detail
        mock_create.assert_not_called()


def test_slices_create_accepts_when_fits_under_cap():
    """Adding 10% when siblings sum to 90% → accepted (100 total)."""
    from api import main
    fake_row = {"id": 999, "portfolio_id": 42, "parent_id": None,
                "name": "OK", "target_pct": 10, "sort_order": 0, "color": None}
    with patch.object(main, "_resolve_portfolio_id", return_value=42), \
         patch.object(main, "_sibling_target_pct_sum", return_value=90.0), \
         patch("db_layer.create_slice", return_value=fake_row) as mock_create:
        result = main.slices_create(
            request=None,
            body={"portfolio": "LTG", "parent_id": None,
                  "name": "OK", "target_pct": 10},
        )
        assert result == {"slice": fake_row}
        mock_create.assert_called_once()


def test_slices_update_refuses_target_pct_increase_that_would_exceed_cap():
    """Existing slice at 5%, siblings sum (excluding self) to 90%. Try
    to raise it to 15% → 105% total → 422. db.update_slice not called."""
    from api import main
    existing = {"id": 1, "portfolio_id": 42, "parent_id": None,
                "name": "A", "target_pct": 5, "sort_order": 0, "color": None}
    with patch("db_layer.get_slice", return_value=existing), \
         patch.object(main, "_sibling_target_pct_sum", return_value=90.0), \
         patch("db_layer.update_slice") as mock_update:
        with pytest.raises(HTTPException) as exc:
            main.slices_update(
                request=None, slice_id=1,
                body={"target_pct": 15},
            )
        assert exc.value.status_code == 422
        mock_update.assert_not_called()


def test_slices_update_allows_decrease_from_over_cap_state():
    """Portfolio currently at 115% (existing over-cap state). User trims
    an existing slice from 20 → 15. Even though siblings still sum to
    95% (making new total = 110% > 100), the DECREASE is allowed —
    trimming toward cap is always OK. This is the load-bearing test:
    without it, a portfolio that got over-cap somehow becomes read-only
    and the user can't trim their way out."""
    from api import main
    existing = {"id": 1, "portfolio_id": 42, "parent_id": None,
                "name": "OverBig", "target_pct": 20, "sort_order": 0,
                "color": None}
    fake_updated = {**existing, "target_pct": 15}
    with patch("db_layer.get_slice", return_value=existing), \
         patch.object(main, "_sibling_target_pct_sum", return_value=95.0), \
         patch("db_layer.update_slice", return_value=fake_updated) as mock_update:
        # DECREASE (20 → 15) — must pass even though siblings sum to 95%
        # (total after would be 110%, still over cap).
        result = main.slices_update(
            request=None, slice_id=1,
            body={"target_pct": 15},
        )
        assert result == {"slice": fake_updated}
        mock_update.assert_called_once()


def test_slices_update_reparent_checks_new_parent_cap():
    """Reparenting a 10% slice from parent A (sum ex-self 80) to parent
    B (sum ex-self 95). New parent B would go to 105% → refuse."""
    from api import main
    existing = {"id": 1, "portfolio_id": 42, "parent_id": 100,
                "name": "Mover", "target_pct": 10, "sort_order": 0,
                "color": None}
    # _sibling_target_pct_sum called with the NEW parent_id (200).
    with patch("db_layer.get_slice", return_value=existing), \
         patch.object(main, "_sibling_target_pct_sum", return_value=95.0), \
         patch("db_layer.update_slice") as mock_update:
        with pytest.raises(HTTPException) as exc:
            main.slices_update(
                request=None, slice_id=1,
                body={"parent_id": 200},
            )
        assert exc.value.status_code == 422
        mock_update.assert_not_called()
