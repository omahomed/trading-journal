import { describe, test, expect } from "vitest";
import {
  SIZING_MODES,
  SIZING_MODES_DISPLAY,
  mctStateToSizingMode,
  exitLadderFloor,
  deriveAutoSizingMode,
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
      // 0.5 / 0.75 / 1.0 are the user-confirmed risk-per-trade values
      // for the auto-pickable tiers. Touching these requires explicit
      // greenlight (not in a refactor).
      expect(SIZING_MODES[0].pct).toBe(0.5);
      expect(SIZING_MODES[1].pct).toBe(0.75);
      expect(SIZING_MODES[2].pct).toBe(1.0);
    });

    test("Pilot (0.25%, manual-only) lives at index 3", () => {
      // Pilot is appended at index 3 — most conservative tier, but
      // array position is end of array so the auto-pickable indices
      // (0/1/2 → defense/normal/offense) stay pinned. Display order
      // is handled separately by SIZING_MODES_DISPLAY below.
      expect(SIZING_MODES).toHaveLength(4);
      expect(SIZING_MODES[3].key).toBe("pilot");
      expect(SIZING_MODES[3].pct).toBe(0.25);
      expect(SIZING_MODES[3].icon).toBe("✈️");
    });
  });

  describe("SIZING_MODES_DISPLAY", () => {
    test("renders in conservatism order: Pilot · Defense · Normal · Offense", () => {
      // UI radios iterate this array. The order is intentionally NOT
      // the canonical SIZING_MODES order — Pilot leads (left/top),
      // Offense trails (right/bottom), reading left-to-right as a
      // tightening-to-loosening spectrum.
      expect(SIZING_MODES_DISPLAY.map(m => m.key)).toEqual(["pilot", "defense", "normal", "offense"]);
    });

    test("each entry carries its canonical SIZING_MODES.index so click handlers can lookup back", () => {
      // setSizingMode(m.index) in both pages relies on this — display
      // position has nothing to do with state value.
      expect(SIZING_MODES_DISPLAY.map(m => m.index)).toEqual([3, 0, 1, 2]);
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

    test("UPTREND UNDER PRESSURE → Normal (1)", () => {
      // 5th state (Phase 1 frontend, dormant). Post-Step-4 cycle
      // stressed by a 21e break — sized at 0.75% per user spec.
      expect(mctStateToSizingMode("UPTREND UNDER PRESSURE")).toBe(1);
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
    // Function is still NAMED describeMctSource (engine internals are
    // still MCT) but the user-visible output says "M Factor" — that's
    // the entire user-facing rename in this lib.
    test("formats the four states with 'M Factor' as user-facing label", () => {
      expect(describeMctSource("POWERTREND")).toBe("from M Factor POWERTREND");
      expect(describeMctSource("UPTREND")).toBe("from M Factor UPTREND");
      expect(describeMctSource("RALLY MODE")).toBe("from M Factor RALLY MODE");
      expect(describeMctSource("CORRECTION")).toBe("from M Factor CORRECTION");
    });

    test("UPTREND UNDER PRESSURE formats as 'from M Factor UPTREND UNDER PRESSURE'", () => {
      // 5th-state descriptor — machine string is echoed verbatim.
      expect(describeMctSource("UPTREND UNDER PRESSURE"))
        .toBe("from M Factor UPTREND UNDER PRESSURE");
    });

    test("unknown / null state surfaces 'M Factor state unknown' instead of guessing", () => {
      expect(describeMctSource(null)).toBe("M Factor state unknown");
      expect(describeMctSource(undefined)).toBe("M Factor state unknown");
      expect(describeMctSource("")).toBe("M Factor state unknown");
      expect(describeMctSource("POWERTREND ON")).toBe("M Factor state unknown");
    });

    test("appends downshift reason when exit-ladder floor is below state mode", () => {
      // POWERTREND alone → Offense (2). 50 SMA Violation floors to 0
      // (Defense), so the floor is BELOW the state mode → label
      // surfaces the reason.
      expect(describeMctSource("POWERTREND", { idx: 0, reason: "50 SMA Violation" }))
        .toBe("from M Factor POWERTREND, downshifted by 50 SMA Violation");
    });

    test("no downshift suffix when floor doesn't actually constrain", () => {
      // CORRECTION alone → Defense (0). A 21 EMA Confirmed Break
      // floor of 1 (Normal) doesn't bind because state is already
      // more conservative. Don't append a misleading "downshifted
      // by" suffix.
      expect(describeMctSource("CORRECTION", { idx: 1, reason: "21 EMA Confirmed Break" }))
        .toBe("from M Factor CORRECTION");
    });

    test("no downshift suffix when floor reason is null", () => {
      expect(describeMctSource("POWERTREND", { idx: 2, reason: null }))
        .toBe("from M Factor POWERTREND");
    });
  });

  describe("exitLadderFloor", () => {
    test("empty / nullish input → no floor (2)", () => {
      expect(exitLadderFloor([])).toEqual({ idx: 2, reason: null });
      expect(exitLadderFloor(null)).toEqual({ idx: 2, reason: null });
      expect(exitLadderFloor(undefined)).toEqual({ idx: 2, reason: null });
    });

    test("Watch states do not downshift", () => {
      // Watches are informational only — no actual break confirmed.
      // Per user spec, only fired Violations / Confirmed Break
      // downshift the mode.
      expect(exitLadderFloor([{ signal: "21 EMA Watch" }])).toEqual({ idx: 2, reason: null });
      expect(exitLadderFloor([{ signal: "50 SMA Watch" }])).toEqual({ idx: 2, reason: null });
    });

    test("21 EMA Violation → Normal (1)", () => {
      expect(exitLadderFloor([{ signal: "21 EMA Violation" }]))
        .toEqual({ idx: 1, reason: "21 EMA Violation" });
    });

    test("21 EMA Confirmed Break → Normal (1)", () => {
      expect(exitLadderFloor([{ signal: "21 EMA Confirmed Break" }]))
        .toEqual({ idx: 1, reason: "21 EMA Confirmed Break" });
    });

    test("50 SMA Violation → Defense (0)", () => {
      expect(exitLadderFloor([{ signal: "50 SMA Violation" }]))
        .toEqual({ idx: 0, reason: "50 SMA Violation" });
    });

    test("most severe wins when multiple alerts fire", () => {
      // Today's IXIC state: 3 consecutive closes below 21 EMA (=
      // Confirmed Break) AND price below 50 SMA with >1% intraday
      // undercut confirmed (= 50 SMA Violation). Floor should be
      // Defense (0), not Normal (1).
      expect(exitLadderFloor([
        { signal: "21 EMA Confirmed Break" },
        { signal: "50 SMA Violation" },
      ])).toEqual({ idx: 0, reason: "50 SMA Violation" });
    });

    test("unknown signal strings ignored (defensive)", () => {
      // Future signal types we don't know about won't accidentally
      // floor the mode. They'll just be invisible until added here.
      expect(exitLadderFloor([{ signal: "FUTURE_SIGNAL_X" }]))
        .toEqual({ idx: 2, reason: null });
    });
  });

  describe("deriveAutoSizingMode (combined: state + exit ladder)", () => {
    test("POWERTREND with no alerts → Offense (state wins)", () => {
      expect(deriveAutoSizingMode("POWERTREND", []).idx).toBe(2);
    });

    test("POWERTREND + 50 SMA Violation → Defense (floor wins)", () => {
      // The user's motivating example. State says Offense; floor
      // says Defense; minimum wins. New trades sized at 0.50% risk
      // even though the regime hasn't flipped to CORRECTION.
      const result = deriveAutoSizingMode("POWERTREND", [{ signal: "50 SMA Violation" }]);
      expect(result.idx).toBe(0);
      expect(result.source.stateIdx).toBe(2);
      expect(result.source.floor).toEqual({ idx: 0, reason: "50 SMA Violation" });
    });

    test("POWERTREND + 21 EMA Confirmed Break → Normal (floor wins)", () => {
      expect(deriveAutoSizingMode("POWERTREND", [{ signal: "21 EMA Confirmed Break" }]).idx).toBe(1);
    });

    test("POWERTREND + 21 EMA Violation → Normal (floor wins, user spec)", () => {
      // User confirmed: single-close + intraday undercut >1% also
      // downshifts to Normal. Same as Confirmed Break — any fired
      // 21 EMA Violation-tier signal floors at Normal.
      expect(deriveAutoSizingMode("POWERTREND", [{ signal: "21 EMA Violation" }]).idx).toBe(1);
    });

    test("CORRECTION + 50 SMA Violation → Defense (no-op floor; both agree)", () => {
      expect(deriveAutoSizingMode("CORRECTION", [{ signal: "50 SMA Violation" }]).idx).toBe(0);
    });

    test("RALLY MODE + 50 SMA Violation → Defense (floor wins)", () => {
      expect(deriveAutoSizingMode("RALLY MODE", [{ signal: "50 SMA Violation" }]).idx).toBe(0);
    });

    test("UPTREND UNDER PRESSURE + 50 SMA Violation → Defense (floor still binds)", () => {
      // UUP alone is Normal (1). A 50 SMA Violation floors to Defense
      // (0). MIN(1, 0) = 0 — same math as the other states; UUP does
      // not exempt the exit-ladder floor.
      const result = deriveAutoSizingMode(
        "UPTREND UNDER PRESSURE",
        [{ signal: "50 SMA Violation" }],
      );
      expect(result.idx).toBe(0);
      expect(result.source.stateIdx).toBe(1);
      expect(result.source.floor).toEqual({ idx: 0, reason: "50 SMA Violation" });
    });

    test("POWERTREND + Watch only → Offense (no floor binds)", () => {
      expect(deriveAutoSizingMode("POWERTREND", [
        { signal: "21 EMA Watch" },
        { signal: "50 SMA Watch" },
      ]).idx).toBe(2);
    });

    test("auto path never returns Pilot (index 3) regardless of state + alerts", () => {
      // Pilot is manual-only by design. Doesn't matter what M Factor
      // state + exit ladder combination is — the auto-derivation can
      // never land at idx 3. Exhaustive smoke test across the state
      // matrix to pin this invariant.
      const states = ["POWERTREND", "UPTREND", "RALLY MODE", "CORRECTION", null, "UNKNOWN"];
      const alertSets = [
        [],
        [{ signal: "21 EMA Watch" }],
        [{ signal: "21 EMA Violation" }],
        [{ signal: "21 EMA Confirmed Break" }],
        [{ signal: "50 SMA Watch" }],
        [{ signal: "50 SMA Violation" }],
        [{ signal: "50 SMA Violation" }, { signal: "21 EMA Confirmed Break" }],
      ];
      for (const s of states) {
        for (const a of alertSets) {
          const r = deriveAutoSizingMode(s, a);
          expect(r.idx).not.toBe(3);
          expect([0, 1, 2]).toContain(r.idx);
        }
      }
    });
  });
});
