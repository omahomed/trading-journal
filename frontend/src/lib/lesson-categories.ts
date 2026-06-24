/**
 * Shared catalog of Trade Review lesson categories + their chip colors.
 *
 * Used by:
 *  - Analytics → Trade Review tab (full editor: pick categories, write note)
 *  - Trade Journal trade card (read-only display: category chips + note)
 *
 * Categories are persisted in `trade_lessons.category` as a pipe-separated
 * string, e.g. "Entry timing|Scaled in too fast". Split on "|" before
 * looking up colors. Categories not in CAT_COLORS fall back to the neutral
 * `var(--bg-2)` / `var(--ink-3)` swatch.
 */

export const LESSON_CATEGORIES = [
  "Followed Rules", "Entry timing", "Stop placement", "Undersized", "Oversized",
  "Scaled in too fast", "Exit too early", "Exit too late",
  "Market conditions", "Rule deviation", "Other",
] as const;

export const CAT_COLORS: Record<string, { bg: string; fg: string }> = {
  "Followed Rules": { bg: "color-mix(in oklab, #08a86b 14%, var(--surface))", fg: "#047857" },
  "Entry timing": { bg: "color-mix(in oklab, #f59f00 12%, var(--surface))", fg: "#b45309" },
  "Stop placement": { bg: "#fed7aa", fg: "#c2410c" },
  "Undersized": { bg: "#dbeafe", fg: "#3b82f6" },
  "Oversized": { bg: "#ede9fe", fg: "#6d28d9" },
  "Scaled in too fast": { bg: "color-mix(in oklab, #e5484d 30%, var(--border))", fg: "#b91c1c" },
  "Exit too early": { bg: "#ccfbf1", fg: "#0f766e" },
  "Exit too late": { bg: "#e0e7ff", fg: "#4338ca" },
  "Market conditions": { bg: "var(--border)", fg: "var(--ink-2)" },
  "Rule deviation": { bg: "#ffe4e6", fg: "#be123c" },
  "Other": { bg: "var(--bg-2)", fg: "var(--ink-3)" },
};

/** Neutral fallback swatch for any category not in CAT_COLORS. */
export const CAT_FALLBACK = { bg: "var(--bg-2)", fg: "var(--ink-3)" } as const;
