import { describe, test, expect } from "vitest";
import {
  SIZING_MODES,
  mctStateToSizingMode,
  describeMctSource,
} from "./sizing-mode";


describe("sizing-mode lib", () => {
  describe("SIZING_MODES constant", () => {
    test("indices are stable: 0=defense, 1=normal, 2=offense", () => {
      // Position Sizer + Log Buy index into this array directly. Reordering
      // would silently flip user-visible behavior. This test pins the
      // ordering so refactors can't break it accidentally.
      expect(SIZING_MODES[0].key).toBe("defense");
      expect(SIZING_MODES[1].key).toBe("normal");
      expect(SIZING_MODES[2].key).toBe("offense");
    });

    test("risk percentages match the confirmed values", () => {
      // 0.5 / 0.75 / 1.0 are the user-confirmed risk-per-trade values.
      // Touching these requires explicit greenlight (not in a refactor).
      expect(SIZING_MODES[0].pct).toBe(0.5);
      expect(SIZING_MODES[1].pct).toBe(0.75);
      expect(SIZING_MODES[2].pct).toBe(1.0);
    });
  });

  describe("mctStateToSizingMode", () => {
    test("CORRECTION → Defense (0)", () => {
      expect(mctStateToSizingMode("CORRECTION")).toBe(0);
    });

    test("RALLY MODE → Normal (1)", () => {
      expect(mctStateToSizingMode("RALLY MODE")).toBe(1);
    });

    test("UPTREND → Offense (2)", () => {
      expect(mctStateToSizingMode("UPTREND")).toBe(2);
    });

    test("POWERTREND → Offense (2)", () => {
      // Spec: cap_at_100 doesn't change sizing mode. POWERTREND with the
      // 100% portfolio-exposure cap engaged is still per-trade Offense
      // (1.0% risk). The cap is a portfolio-total constraint, enforced
      // separately by V11's exposure-cap logic.
      expect(mctStateToSizingMode("POWERTREND")).toBe(2);
    });

    test("unknown state defaults to Normal (safe middle)", () => {
      // Three failure modes: null (rally-prefix returned no state),
      // empty string (rally-prefix returned {}), legacy V10 string
      // (e.g. "POWERTREND ON"). All collapse to the safe middle so a
      // transient hiccup doesn't push the user into Offense or
      // restrict them into Defense.
      expect(mctStateToSizingMode(null)).toBe(1);
      expect(mctStateToSizingMode(undefined)).toBe(1);
      expect(mctStateToSizingMode("")).toBe(1);
      expect(mctStateToSizingMode("UNKNOWN")).toBe(1);
      expect(mctStateToSizingMode("POWERTREND ON")).toBe(1);  // V10 legacy
    });
  });

  describe("describeMctSource", () => {
    test("formats the four MCT states verbatim", () => {
      expect(describeMctSource("POWERTREND")).toBe("from MCT POWERTREND");
      expect(describeMctSource("UPTREND")).toBe("from MCT UPTREND");
      expect(describeMctSource("RALLY MODE")).toBe("from MCT RALLY MODE");
      expect(describeMctSource("CORRECTION")).toBe("from MCT CORRECTION");
    });

    test("unknown / null state surfaces 'state unknown' instead of guessing", () => {
      expect(describeMctSource(null)).toBe("MCT state unknown");
      expect(describeMctSource(undefined)).toBe("MCT state unknown");
      expect(describeMctSource("")).toBe("MCT state unknown");
      expect(describeMctSource("POWERTREND ON")).toBe("MCT state unknown");
    });
  });
});
