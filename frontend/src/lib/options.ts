// Helpers for parsing the readable option-ticker format standardised in
// Migration 016: "UNDERLYING YYMMDD $STRIKE C|P" — e.g. "NVDA 250620 $150C".
// Same regex shape as api/main.py:_is_option_ticker(); kept in sync here so
// the UI can compute expiration / DTE without round-tripping to the backend.

const OPTION_RE = /^(\S+)\s+(\d{6})\s+\$(\d+(?:\.\d+)?)([CP])$/i;

export interface ParsedOption {
  underlying: string;
  exp: Date;
  strike: number;
  right: "C" | "P";
}

export function parseOptionTicker(ticker: string): ParsedOption | null {
  const m = OPTION_RE.exec((ticker || "").trim());
  if (!m) return null;
  const yy = parseInt(m[2].slice(0, 2), 10);
  const mm = parseInt(m[2].slice(2, 4), 10);
  const dd = parseInt(m[2].slice(4, 6), 10);
  // YY < 50 → 20YY; pre-2050 contracts only ever set the 20xx range.
  const year = yy < 50 ? 2000 + yy : 1900 + yy;
  // Build at UTC midnight so DTE math is timezone-stable.
  const exp = new Date(Date.UTC(year, mm - 1, dd));
  return {
    underlying: m[1].toUpperCase(),
    exp,
    strike: parseFloat(m[3]),
    right: m[4].toUpperCase() as "C" | "P",
  };
}

export function daysUntilExpiration(exp: Date): number {
  const now = new Date();
  // Compare calendar days only — local "today" against the contract's UTC
  // expiration day. Integer-stable across the trading day.
  const today = Date.UTC(now.getFullYear(), now.getMonth(), now.getDate());
  const expDay = Date.UTC(exp.getUTCFullYear(), exp.getUTCMonth(), exp.getUTCDate());
  return Math.round((expDay - today) / 86_400_000);
}
