import { describe, it, expect } from "vitest";
import {
  SCORECARD_CATEGORIES,
  SCORECARD_MAX_TOTAL,
  defaultScores,
  gradeToScore,
  letterGrade,
  parseHighlightsScores,
  scoreColor,
} from "./scorecard";

describe("SCORECARD_CATEGORIES", () => {
  it("locks the 4-category canon", () => {
    expect(SCORECARD_CATEGORIES.map(c => c.key)).toEqual([
      "plan", "stops", "sized", "fomo",
    ]);
    expect(SCORECARD_MAX_TOTAL).toBe(20);
  });
});

describe("letterGrade", () => {
  it("perfect score → A+", () => {
    expect(letterGrade(20, 20)).toBe("A+");
  });

  it("just below perfect → A", () => {
    expect(letterGrade(19, 20)).toBe("A");
  });

  it("boundary values map to the right letter", () => {
    // Thresholds: A+ 100, A 93, A- 87, B+ 83, B 77, B- 70, C+ 67, C 60, C- 53, D 47.
    expect(letterGrade(83, 100)).toBe("B+");
    expect(letterGrade(82, 100)).toBe("B");   // just below B+ → B
    expect(letterGrade(77, 100)).toBe("B");   // B boundary
    expect(letterGrade(76, 100)).toBe("B-");  // just below B → B-
    expect(letterGrade(70, 100)).toBe("B-");
    expect(letterGrade(60, 100)).toBe("C");
  });

  it("failing → F", () => {
    expect(letterGrade(4, 20)).toBe("F");
  });
});

describe("gradeToScore", () => {
  it("A grades → 5", () => {
    expect(gradeToScore("A+")).toBe(5);
    expect(gradeToScore("A")).toBe(5);
    expect(gradeToScore("A-")).toBe(5);
  });

  it("B → 4, C → 3, D → 2, F → 1", () => {
    expect(gradeToScore("B+")).toBe(4);
    expect(gradeToScore("C")).toBe(3);
    expect(gradeToScore("D")).toBe(2);
    expect(gradeToScore("F")).toBe(1);
  });
});

describe("scoreColor", () => {
  it("≥4 green, ≥3 amber, else red", () => {
    expect(scoreColor(5)).toBe("#08a86b");
    expect(scoreColor(4)).toBe("#08a86b");
    expect(scoreColor(3)).toBe("#f59f00");
    expect(scoreColor(2)).toBe("#e5484d");
    expect(scoreColor(1)).toBe("#e5484d");
  });
});

describe("defaultScores", () => {
  it("every category defaults to 5 (perfect until dialed down)", () => {
    expect(defaultScores()).toEqual({ plan: 5, stops: 5, sized: 5, fomo: 5 });
  });
});

describe("parseHighlightsScores", () => {
  it("valid JSON returns matching scores", () => {
    const s = JSON.stringify({ plan: 4, stops: 3, sized: 5, fomo: 2 });
    expect(parseHighlightsScores(s)).toEqual({ plan: 4, stops: 3, sized: 5, fomo: 2 });
  });

  it("null/empty returns defaults", () => {
    expect(parseHighlightsScores(null)).toEqual(defaultScores());
    expect(parseHighlightsScores(undefined)).toEqual(defaultScores());
    expect(parseHighlightsScores("")).toEqual(defaultScores());
  });

  it("malformed JSON returns defaults", () => {
    expect(parseHighlightsScores("not-json")).toEqual(defaultScores());
    expect(parseHighlightsScores("[1,2,3]")).toEqual(defaultScores());
  });

  it("partial keys fall back to per-category defaults", () => {
    const s = JSON.stringify({ plan: 3, fomo: 1 });
    expect(parseHighlightsScores(s)).toEqual({ plan: 3, stops: 5, sized: 5, fomo: 1 });
  });

  it("out-of-range values ignored (bounds 1-5)", () => {
    const s = JSON.stringify({ plan: 0, stops: 7, sized: 3, fomo: 4 });
    expect(parseHighlightsScores(s)).toEqual({ plan: 5, stops: 5, sized: 3, fomo: 4 });
  });

  it("non-numeric values ignored", () => {
    const s = JSON.stringify({ plan: "high", stops: null, sized: 4 });
    expect(parseHighlightsScores(s)).toEqual({ plan: 5, stops: 5, sized: 4, fomo: 5 });
  });
});
