// Phase 4.6 — letter-grade helpers ported from the Weekly Retro v1 design
// (Design/design_handoff_weekly_retro/design/sections-bottom.jsx:5-54).
//
// Consolidates 2 pre-existing inline copies of gradeColor that lived in
// weekly-retro.tsx and daily-routine.tsx. Backend mirrors GRADE_VAL +
// VAL_GRADE behavior in db_layer._GRADE_TO_NUMERIC + _NUMERIC_BUCKETS;
// both sides must stay in sync.
//
// NOTE: daily-report-card.tsx has a local variable named `gradeColor`
// inside renderDailyReview() that resolves a numeric SCORE (1-5) to a
// color — not a letter grade. Different concept, not consolidated here.

export const GRADE_OPTIONS = [
  "A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F",
] as const;

export type GradeLetter = (typeof GRADE_OPTIONS)[number];

// Verbatim from the design's GRADE_DEFS map.
export const GRADE_DEFS: Record<string, string> = {
  "A+": "Flawless — system + result aligned, would repeat verbatim.",
  "A":  "Excellent — minor nits only.",
  "A-": "Strong — one small slip from textbook.",
  "B+": "Solid execution with one process slip.",
  "B":  "Decent — followed the plan with a couple of misses.",
  "B-": "Acceptable — patchy adherence.",
  "C+": "Mixed — process slipping more than result suggests.",
  "C":  "Forgettable — half right, half wrong.",
  "C-": "Sloppy — broke rules even on winning trades.",
  "D":  "Bad — multiple rule breaks; result followed.",
  "F":  "Reckless — system off-line; rebuild trust before next week.",
};

// 4.3 GPA scale. Matches db_layer._GRADE_TO_NUMERIC.
export const GRADE_VAL: Record<string, number> = {
  "A+": 4.3, "A": 4.0, "A-": 3.7,
  "B+": 3.3, "B": 3.0, "B-": 2.7,
  "C+": 2.3, "C": 2.0, "C-": 1.7,
  "D":  1.0, "F":  0.0,
};

// Bucket a numeric mean back to a letter. Mirrors db_layer._NUMERIC_BUCKETS
// behavior but collapses D+ / D- to plain D (the column vocab CHECK only
// allows D — pre-existing inconsistency noted in the audit).
export function VAL_GRADE(num: number): GradeLetter {
  if (num >= 4.15) return "A+";
  if (num >= 3.85) return "A";
  if (num >= 3.50) return "A-";
  if (num >= 3.15) return "B+";
  if (num >= 2.85) return "B";
  if (num >= 2.50) return "B-";
  if (num >= 2.15) return "C+";
  if (num >= 1.85) return "C";
  if (num >= 1.50) return "C-";
  if (num >= 0.75) return "D";
  return "F";
}

// Returns the canonical Overall Week Grade from the 3 axis values, or
// null when any axis is missing/unknown. Matches the server's
// _derive_overall_grade in db_layer.py — both sides must agree so the
// frontend UI preview equals the persisted value.
export function deriveOverall(
  execution: string | null | undefined,
  process: string | null | undefined,
  pnl: string | null | undefined,
): GradeLetter | null {
  if (!execution || !process || !pnl) return null;
  const vals: number[] = [];
  for (const g of [execution, process, pnl]) {
    const key = g.trim().toUpperCase();
    const v = GRADE_VAL[key];
    if (v == null) return null;
    vals.push(v);
  }
  if (vals.length !== 3) return null;
  return VAL_GRADE(vals.reduce((a, b) => a + b, 0) / 3);
}

// Ink color for a grade letter. Verbatim from sections-bottom.jsx:41-47.
// `null` / empty → the default ink (so unselected dropdowns render neutral).
export function gradeColor(g: string | null | undefined): string {
  if (!g) return "var(--ink)";
  if (g.startsWith("A")) return "#08a86b";
  if (g.startsWith("B")) return "#3b82f6";
  if (g.startsWith("C")) return "#f59f00";
  return "#e5484d";
}

// Tinted surface bg for a grade letter — pairs with gradeColor for the
// axis selector chips. Verbatim from sections-bottom.jsx:48-54.
export function gradeTint(g: string | null | undefined): string {
  if (!g) return "var(--surface)";
  if (g.startsWith("A")) return "color-mix(in oklab, #08a86b 12%, var(--surface))";
  if (g.startsWith("B")) return "color-mix(in oklab, #3b82f6 10%, var(--surface))";
  if (g.startsWith("C")) return "color-mix(in oklab, #f59f00 14%, var(--surface))";
  return "color-mix(in oklab, #e5484d 12%, var(--surface))";
}
