// Shared scorecard helpers — 4 categories (Plan / Stops / Sized /
// FOMO), letter grade derived from the % of max possible score.
//
// Used by NLV Entry (legacy capture point) and the ScorecardMiniForm
// on Daily Routine (new capture point per Phase 2 merger). Sourcing
// both call sites from here prevents drift on the grade thresholds.

export interface ScorecardCategory {
  key: "plan" | "stops" | "sized" | "fomo";
  label: string;
}

export const SCORECARD_CATEGORIES: ScorecardCategory[] = [
  { key: "plan",  label: "Followed plan"     },
  { key: "stops", label: "Respected stops"   },
  { key: "sized", label: "Sized correctly"   },
  { key: "fomo",  label: "No FOMO entries"   },
];

export const SCORECARD_MAX_TOTAL = SCORECARD_CATEGORIES.length * 5;

/** Points 1-5 per category → letter grade based on % of max total. */
export function letterGrade(total: number, max: number): string {
  const pct = (total / max) * 100;
  if (pct >= 100) return "A+";
  if (pct >= 93) return "A";
  if (pct >= 87) return "A-";
  if (pct >= 83) return "B+";
  if (pct >= 77) return "B";
  if (pct >= 70) return "B-";
  if (pct >= 67) return "C+";
  if (pct >= 60) return "C";
  if (pct >= 53) return "C-";
  if (pct >= 47) return "D";
  return "F";
}

/** Coarse mapping used for the `score` numeric column on trading_journal.
 *  1-5, matching the 1-5 sliders the user sees per category. */
export function gradeToScore(g: string): number {
  return g.startsWith("A") ? 5
       : g.startsWith("B") ? 4
       : g.startsWith("C") ? 3
       : g.startsWith("D") ? 2
       : 1;
}

/** Tier color per 1-5 score value. Green ≥4, amber ≥3, red below. */
export function scoreColor(v: number): string {
  return v >= 4 ? "#08a86b" : v >= 3 ? "#f59f00" : "#e5484d";
}

/** Default scores when the user hasn't graded yet — every category at
 *  5, letting them dial DOWN from perfect rather than up from zero.
 *  Matches NLV Entry's initial state. */
export function defaultScores(): Record<ScorecardCategory["key"], number> {
  return { plan: 5, stops: 5, sized: 5, fomo: 5 };
}

/** Parse the `highlights` column (JSON string of per-category scores)
 *  into a scores map. Returns defaults on malformed / missing input. */
export function parseHighlightsScores(highlights: string | null | undefined): Record<ScorecardCategory["key"], number> {
  if (!highlights) return defaultScores();
  try {
    const parsed = JSON.parse(highlights);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      const out = defaultScores();
      for (const cat of SCORECARD_CATEGORIES) {
        const v = parsed[cat.key];
        if (typeof v === "number" && v >= 1 && v <= 5) out[cat.key] = v;
      }
      return out;
    }
  } catch {
    // fall through
  }
  return defaultScores();
}
