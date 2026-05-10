/**
 * Single source of truth for USD currency rendering across the frontend.
 *
 * Replaces six ad-hoc local helpers (fmtDol×2, fmtMoney, signedMoney,
 * dollarFmt, fmt$, fmtUsd) and the inline `${"$" + n.toLocaleString(...)}`
 * pattern that produced the Group 4a "trailing zero" drift bug.
 *
 * Always emits explicit minimum + maximum fraction digits — never
 * max-only — so a column rendered through this helper cannot drift to
 * mixed precision (`$76,815` next to `$86,629.96`).
 *
 * Sign is placed before `$` (accounting convention): `-$1,234.56`,
 * never `$-1,234.56`.
 */

/** Options controlling how a numeric value is rendered as USD. */
export interface FormatCurrencyOptions {
  /** Number of fraction digits (min == max). Default 2. In compact mode,
   *  default is 1 for values ≥ 1000 and 0 for values < 1000. */
  decimals?: number;
  /** When true, prepend `+` to strictly-positive values. Zero never gets a
   *  sign. Negative values always get the sign regardless of this flag. */
  showSign?: boolean;
  /** Glyph used for the negative sign. `"ascii"` (default) emits `-`;
   *  `"unicode"` emits the typographic minus `−` (U+2212). */
  signGlyph?: "ascii" | "unicode";
  /** When true, abbreviate values ≥ 1000 with k/M/B suffixes
   *  (e.g. 1500 → `"$1.5k"`). Default false. */
  compact?: boolean;
  /** When true, render `nullDisplay` instead of `"$0.00"` when value === 0. */
  zeroAsDash?: boolean;
  /** Rendering for null / undefined / NaN (and for zero when zeroAsDash is
   *  true). Default `"—"`. */
  nullDisplay?: string;
}

/**
 * Format a numeric USD value for display.
 *
 * @param value Number, null, undefined, or NaN.
 * @param opts See {@link FormatCurrencyOptions}.
 * @returns A locale-formatted dollar string, or `nullDisplay` for nullish/NaN.
 *
 * @example
 *   formatCurrency(1234.56)                          // "$1,234.56"
 *   formatCurrency(-1234.56)                         // "-$1,234.56"
 *   formatCurrency(1234, { showSign: true })         // "+$1,234.00"
 *   formatCurrency(1500, { compact: true })          // "$1.5k"
 *   formatCurrency(0,    { zeroAsDash: true })       // "—"
 *   formatCurrency(null)                             // "—"
 */
export function formatCurrency(
  value: number | null | undefined,
  opts: FormatCurrencyOptions = {},
): string {
  const {
    showSign = false,
    signGlyph = "ascii",
    compact = false,
    zeroAsDash = false,
    nullDisplay = "—",
  } = opts;

  if (value == null || Number.isNaN(value)) return nullDisplay;
  if (zeroAsDash && value === 0) return nullDisplay;

  const minus = signGlyph === "unicode" ? "−" : "-";
  let signStr = "";
  if (value < 0) signStr = minus;
  else if (showSign && value > 0) signStr = "+";

  const abs = Math.abs(value);

  if (compact) {
    let scaled: number;
    let suffix: string;
    let defaultDecimals: number;
    if (abs >= 1e9) {
      scaled = abs / 1e9; suffix = "B"; defaultDecimals = 1;
    } else if (abs >= 1e6) {
      scaled = abs / 1e6; suffix = "M"; defaultDecimals = 1;
    } else if (abs >= 1e3) {
      scaled = abs / 1e3; suffix = "k"; defaultDecimals = 1;
    } else {
      scaled = abs; suffix = ""; defaultDecimals = 0;
    }
    const d = opts.decimals ?? defaultDecimals;
    const formatted = scaled.toLocaleString(undefined, {
      minimumFractionDigits: d,
      maximumFractionDigits: d,
    });
    return `${signStr}$${formatted}${suffix}`;
  }

  const d = opts.decimals ?? 2;
  const formatted = abs.toLocaleString(undefined, {
    minimumFractionDigits: d,
    maximumFractionDigits: d,
  });
  return `${signStr}$${formatted}`;
}
