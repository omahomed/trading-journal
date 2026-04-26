#!/usr/bin/env python3
"""Run the MCT V11 engine over a date range and optionally persist signals.

Usage:
  python scripts/replay_mct.py --start 2024-12-16 --end 2026-04-24
  python scripts/replay_mct.py --start 2024-12-16 --write-signals
  python scripts/replay_mct.py --full-history --csv-out /tmp/run.csv
  python scripts/replay_mct.py --start 2024-12-16 --end 2026-04-24 \
      --initial-reference-high 20118.61 --csv-out /tmp/canonical_run.csv

  # After an engine logic change, rebuild market_signals from scratch
  # (DESTRUCTIVE: deletes all existing rows, then re-inserts):
  python scripts/replay_mct.py --full-history --write-signals --rebuild
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

# Make repo root importable when invoked directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from api.mct_engine import MCTEngine, EngineConfig  # noqa: E402
from api.market_data_repo import get_history, get_latest_date  # noqa: E402
from api.mct_signals_writer import write_signals, rebuild_signals  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("replay_mct")


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD (default: latest in market_data)")
    parser.add_argument("--full-history", action="store_true",
                        help="Run engine over full market_data history (overrides --start/--end)")
    parser.add_argument("--write-signals", action="store_true",
                        help="Persist signals to market_signals (idempotent)")
    parser.add_argument("--rebuild", action="store_true",
                        help="DESTRUCTIVE: with --write-signals, DELETE all "
                             "existing market_signals rows then INSERT this "
                             "run's signals (single transaction). Use after "
                             "engine logic changes that obsolete previously "
                             "persisted signals (ON CONFLICT DO NOTHING in the "
                             "default --write-signals path can't notice).")
    parser.add_argument("--symbol", default="^IXIC")
    parser.add_argument("--initial-reference-high", type=float, default=None,
                        help="Override initial reference high")
    parser.add_argument("--initial-state", default="POWERTREND")
    parser.add_argument("--initial-exposure", type=int, default=200)
    parser.add_argument("--no-power-trend", action="store_true",
                        help="Start with power_trend=False instead of True")
    parser.add_argument("--csv-out", help="Write bars DataFrame to CSV")
    args = parser.parse_args()

    # Resolve date range
    if args.full_history:
        start = date(2010, 1, 1)
        end = get_latest_date(args.symbol) or date.today()
    else:
        if not args.start:
            parser.error("--start is required (or use --full-history)")
        start = parse_date(args.start)
        end = parse_date(args.end) if args.end else (
            get_latest_date(args.symbol) or date.today()
        )

    log.info("Loading %s history from %s to %s", args.symbol, start, end)
    history = get_history(args.symbol, start, end)
    if history.empty:
        log.error("No bars found in market_data for that range. Has the backfill run?")
        return 1
    log.info("Loaded %d bars", len(history))

    # For --full-history runs the seed is an ancient bar's high; the ratchet
    # must be armed up front so it can climb to a contemporaneous reference.
    # For stress-test runs with a contemporaneous --initial-reference-high
    # seed (Phase 2 canonical), keep the ratchet disarmed until first
    # nullification (matching the Phase 2 design semantics).
    ratchet_armed = bool(args.full_history)

    config = EngineConfig(
        initial_reference_high=args.initial_reference_high,
        initial_state=args.initial_state,
        initial_exposure=args.initial_exposure,
        initial_power_trend=not args.no_power_trend,
        correction_ever_declared=True,
        initial_ratchet_armed=ratchet_armed,
    )
    log.info("Engine config: ref_high=%s, state=%s, exposure=%d, PT=%s, ratchet_armed=%s",
             config.initial_reference_high, config.initial_state,
             config.initial_exposure, config.initial_power_trend,
             config.initial_ratchet_armed)

    engine = MCTEngine(config)
    result = engine.run(history)
    log.info("Engine emitted %d signals across %d bars",
             len(result.signals), len(result.bars))

    # Print signal log
    if result.signals:
        log.info("Signal log:")
        for ev in result.signals:
            log.info("  %s  %-32s  exp %3d→%-3d  state=%-10s  %s",
                     ev.trade_date, ev.signal_type,
                     ev.exposure_before, ev.exposure_after,
                     ev.state_after, ev.signal_label)

    if args.csv_out:
        result.bars.to_csv(args.csv_out, index=False)
        log.info("Wrote %d bar rows to %s", len(result.bars), args.csv_out)

    if args.write_signals:
        if args.rebuild:
            summary = rebuild_signals(result.signals)
            log.warning(
                "REBUILD: deleted %d stale rows; inserted %d fresh rows from %d events",
                summary["deleted"], summary["inserted"], summary["events_emitted"]
            )
        else:
            inserted = write_signals(result.signals)
            log.info("Persisted %d new signals to market_signals (skipped %d duplicates)",
                     inserted, len(result.signals) - inserted)
    elif args.rebuild:
        parser.error("--rebuild requires --write-signals")

    # Print final state summary
    fs = result.final_state
    log.info("Final state: exposure=%d, state=%s, in_correction=%s, "
             "correction_active=%s, power_trend=%s, cap_at_100=%s, "
             "step4=%s",
             fs.get("exposure"), engine._derive_state(fs),
             fs.get("in_correction"), fs.get("correction_active"),
             fs.get("power_trend"), fs.get("cap_at_100"),
             fs.get("step4_done"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
