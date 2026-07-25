#!/usr/bin/env python3
"""Per-lot MAE/MFE research export → CSV.

Companion to scripts/backfill_mae_mfe.py — while that script writes
CAMPAIGN-level MAE/MFE to trades_summary (measured from B1's fill
price), this script computes per-LOT excursions for every BUY row
(B1, A1, A2, …) and dumps them to CSV. Each lot is anchored to its
own fill_price + fill_date and measured through the campaign's
close_date (or today for open campaigns).

Purpose: the B1 broker-stop calibration study found that entries
breaching −0.75× ATR21 have ~0% win rate; this export is the raw data
needed to run the same study on add-on lots (do A-series entries have
a different "no recovery" ATR distance?). Not a display feature — the
CSV lands in output/ and is meant to be fed to an offline analysis
session.

Also drives the CR sidecar's per-lot MAE table via the same helper
(api/lot_excursions.py) so the two consumers never disagree on the
numbers.

Usage:
    python scripts/export_lot_excursions.py                        # all portfolios, all campaigns
    python scripts/export_lot_excursions.py --portfolio "LTG"      # one portfolio
    python scripts/export_lot_excursions.py --since 2026-01-01     # skip pre-2026 legacy imports
    python scripts/export_lot_excursions.py --closed-only          # only campaigns already closed
    python scripts/export_lot_excursions.py --open-only            # only campaigns still open
    python scripts/export_lot_excursions.py --out custom_path.csv  # override output path

Output columns (research-friendly, superset of the campaign MAE columns):
    portfolio_name · trade_id · ticker · status · closed_date · trx_id ·
    fill_date · fill_price · shares · window_end_date · days_held ·
    mae_pct · mae_atr_multiple · min_low · min_low_date · days_to_mae ·
    mfe_pct · mfe_atr_multiple · max_high · max_high_date · days_to_mfe ·
    atr21_at_fill_pct · realized_pl · error

Idempotent: multiple runs on the same day overwrite the same dated
file (unless you pass --out). yfinance is called once per campaign
regardless of lot count.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_dotenv_if_present() -> None:
    """Load repo-root .env into os.environ (skip anything already set).

    Standalone parser — the project doesn't depend on python-dotenv and
    the script needs to work from a bare `python3 scripts/…` invocation
    without a wrapping `set -a; source .env` incantation. Without this,
    db_layer.get_db_config() falls through to the localhost default and
    the query hits an old dev DB that predates migration 016 →
    "column s.instrument_type does not exist".

    Skipped when DATABASE_URL is already set (Railway/CI/pre-exported
    shell), or when .env is missing. Not a general-purpose parser —
    just KEY=value lines, ignores # comments + blanks, strips
    surrounding quotes so `DATABASE_URL="postgres://..."` works.
    """
    import os as _os
    if _os.getenv("DATABASE_URL"):
        return
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in _os.environ:
            _os.environ[key] = val


_load_dotenv_if_present()

from api.lot_excursions import compute_all_lot_excursions  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("export_lot_excursions")


# Column order for the CSV. Locked so multiple runs are diffable and
# downstream analysis can hard-code column positions.
#
# P&L note (a real gotcha for downstream analysis): `realized_pl` is
# PER-LOT — sum of lot_closures rows where buy_trx_id = this lot's
# trx_id. NULL when the lot has no closures yet (still fully open in
# an open campaign). `campaign_realized_pl` is the campaign total
# (trades_summary.realized_pl) — same value across every lot of the
# same campaign; kept for cross-check convenience. Pre-fix, "realized_pl"
# carried the campaign total on every row, which read as if per-lot
# but wasn't; that's why the two columns are now explicit.
CSV_COLUMNS = [
    "portfolio_name", "trade_id", "ticker", "status", "closed_date",
    "trx_id", "fill_date", "fill_price", "shares", "shares_closed",
    # Migration 049: §2 Window exempt-reason tag ('sr8_rebuild' /
    # 'fresh_base' / NULL). Filters the 30-add review by declared
    # override reason vs. plain (non-exempt) adds.
    "add_exempt_reason",
    "window_end_date", "days_held",
    "mae_pct", "mae_atr_multiple", "min_low", "min_low_date", "days_to_mae",
    "mfe_pct", "mfe_atr_multiple", "max_high", "max_high_date", "days_to_mfe",
    "atr21_at_fill_pct",
    "realized_pl", "campaign_realized_pl",
    "error",
]


def _default_out_path() -> Path:
    out_dir = REPO_ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    return out_dir / f"lot_excursions_{date.today().isoformat()}.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--portfolio", default=None,
                        help="Limit to one portfolio (default: all).")
    parser.add_argument("--since", default=None,
                        help="Include campaigns still open OR closed on/after "
                             "this date (YYYY-MM-DD). Use to sidestep pre-2026 "
                             "legacy import garbage.")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--closed-only", action="store_true",
                       help="Only include campaigns with status = CLOSED.")
    scope.add_argument("--open-only", action="store_true",
                       help="Only include campaigns with status = OPEN.")
    parser.add_argument("--sleep", type=float, default=0.3,
                        help="Seconds to sleep between per-campaign yfinance "
                             "downloads (default 0.3).")
    parser.add_argument("--out", default=None,
                        help=f"CSV output path. Default: "
                             f"output/lot_excursions_YYYY-MM-DD.csv")
    args = parser.parse_args()

    since_date = None
    if args.since:
        try:
            since_date = datetime.strptime(args.since.strip()[:10], "%Y-%m-%d").date()
        except ValueError:
            log.error("bad --since date: %s", args.since)
            return 2

    include_closed = not args.open_only  # closed-only path handled by client-side filter below

    log.info("Fetching lots (portfolio=%s, since=%s, scope=%s) ...",
             args.portfolio or "all",
             since_date.isoformat() if since_date else "none",
             "closed-only" if args.closed_only else ("open-only" if args.open_only else "all"))

    rows = compute_all_lot_excursions(
        portfolio=args.portfolio,
        include_closed=include_closed,
        since=since_date,
        sleep=args.sleep,
    )
    if not rows:
        log.info("No lots found matching filters.")
        return 0

    # Post-filter for --closed-only: the fetch already limits to
    # include_closed=True by default, so this narrows further to
    # CLOSED status only. --open-only was handled upstream by passing
    # include_closed=False.
    if args.closed_only:
        rows = [r for r in rows if (r.get("status") or "").upper() == "CLOSED"]

    out_path = Path(args.out) if args.out else _default_out_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) if r.get(k) is not None else "" for k in CSV_COLUMNS})

    # Report — counts + a quick breakdown so the operator knows what
    # landed without opening the file.
    ok = sum(1 for r in rows if r.get("error") is None)
    err = len(rows) - ok
    by_trx: dict[str, int] = {}
    for r in rows:
        key = str(r.get("trx_id") or "?")[:2]  # B1 / A1 / A2 / A3 ...
        by_trx[key] = by_trx.get(key, 0) + 1

    log.info("Wrote %d rows → %s", len(rows), out_path)
    log.info("  computed: %d  · errors: %d", ok, err)
    log.info("  by lot type: %s", ", ".join(f"{k}={v}" for k, v in sorted(by_trx.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
