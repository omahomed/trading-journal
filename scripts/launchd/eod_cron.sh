#!/bin/bash
# launchd wrapper for the EOD journal save cron job.
#
# Why a wrapper script instead of calling python directly from the plist:
#   1. The repo lives in iCloud Drive ("Library/Mobile Documents/...") which
#      means the path has spaces and weird semantics. launchd hates spaces.
#      Calling a stable wrapper script side-steps the quoting nightmare.
#   2. We need to set DATABASE_URL from a secret file (chmod 600), activate
#      the project venv, and then run the script. Neater here than in plist.
#   3. Easier to re-run manually for debugging — `bash eod_cron.sh` works
#      without launchd in the loop.

set -euo pipefail

# Repo lives in iCloud — be aware launchd may run before iCloud finishes
# sync on boot. The EOD cron fires at 4:05 PM weekdays so this isn't usually
# an issue, but the gateway plist (which runs at boot) lives on local disk
# precisely to dodge this risk.
REPO="/Users/momacbookair/Library/Mobile Documents/com~apple~CloudDocs/my_code"

# Secrets file — chmod 600, contains DATABASE_URL=postgresql://...
# Lives outside the repo so it never accidentally lands in git or iCloud
# sync history. Pattern: one KEY=VALUE per line, no quotes needed.
SECRETS_FILE="$HOME/.config/ibkr-trading-journal.env"
if [[ ! -f "$SECRETS_FILE" ]]; then
    echo "Missing $SECRETS_FILE — create it with DATABASE_URL=postgresql://..." >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$SECRETS_FILE"

if [[ -z "${DATABASE_URL:-}" ]]; then
    echo "$SECRETS_FILE didn't export DATABASE_URL" >&2
    exit 1
fi

cd "$REPO"
# Activate the project venv so `python` resolves to the version with all
# the journal app's pip deps (psycopg2, yfinance, fastapi, etc.).
# shellcheck disable=SC1091
source venv/bin/activate

# Pass through any args ($@) so we can also run with --dry-run from the
# command line for debugging the same way launchd will run it.
exec python scripts/eod_journal_save.py "$@"
