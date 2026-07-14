import { describe, test, expect } from "vitest";
import {
  SIZING_MODES,
  SIZING_MODES_DISPLAY,
  mctStateToSizingMode,
  exitLadderFloor,
  deriveAutoSizingMode,
  describeMctSource,
  clampManualToDownwardOnly,
} from "./sizing-mode";


describe("sizing-mode lib", () => {
  describe("SIZING_MODES constant", () => {
    test("indices are stable: 0=pilot, 1=normal, 2=offense (aggression order)", () => {
      // Position Sizer + Log Buy + New Entry index into this array
      // directly. The retier (New Entry rollout) moved Pilot to index 0
      // and retired Defense; array order now equals aggression order
      // (least → most aggressive). Reordering would silently flip
      // user-visible behavior. This test pins the ordering.
      expect(SIZING_MODES).toHaveLength(3);
      expect(SIZING_MODES[0].key).toBe("pilot");
      expect(SIZING_MODES[1].key).toBe("normal");
      expect(SIZING_MODES[2].key).toBe("offense");
    });

    test("risk percentages match the retiered values", () => {
      // 0.25 / 0.50 / 0.75 are the New Entry–derived risk-per-trade
      // values, adopted globally. Touching these requires explicit
      // greenlight (not in a refactor).
      expect(SIZING_MODES[0].pct).toBe(0.25);
      expect(SIZING_MODES[1].pct).toBe(0.50);
      expect(SIZING_MODES[2].pct).toBe(0.75);
    });

    test("Defense tier is retired (no key === 'defense' in the lineup)", () => {
      // Explicit regression guard: nothing in SIZING_MODES should
      // carry the retired Defense key. If it comes back, this test
      // fires immediately.
      const keys = SIZING_MODES.map(m => m.key);
      expect(keys).not.toContain("defense");
    });
  });

  describe("SIZING_MODES_DISPLAY", () => {
    test("renders in conservatism order: Pilot · Normal · Offense", () => {
      // UI radios iterate this array. Now equal to SIZING_MODES since
      // the canonical array is already in aggression order — kept as
      // a named export so callsites don't have to change.
      expect(SIZING_MODES_DISPLAY.map(m => m.key)).toEqual(["pilot", "normal", "offense"]);
    });

    test("each entry carries its canonical SIZING_MODES.index for lookup", () => {
      // setSizingMode(m.index) in both pages relies on this. Display
      // position now identical to canonical position (0/1/2).
      expect(SIZING_MODES_DISPLAY.map(m => m.index)).toEqual([0, 1, 2]);
    });
  });

  describe("mctStateToSizingMode", () => {
    test("POWERTREND → Offense (2, 0.75%)", () => {
      // The only state that lands on Offense post-retier. UPTREND
      // dropped to Normal; everything else dropped to Pilot.
      expect(mctStateToSizingMode("POWERTREND")).toBe(2);
    });

    test("UPTREND → Normal (1, 0.50%)", () => {
      expect(mctStateToSizingMode("UPTREND")).toBe(1);
    });

    test("UPTREND UNDER PRESSURE → Pilot (0, 0.25%)", () => {
      // UUP slid from Normal (old 0.75%) to Pilot (0.25%). The engine
      // considers UUP a stressed post-Step-4 cycle; the retier treats
      // it accordingly.
      expect(mctStateToSizingMode("UPTREND UNDER PRESSURE")).toBe(0);
    });

    test("RALLY MODE → Pilot (0, 0.25%)", () => {
      // Post-retier alignment: RALLY MODE was Normal (0.75%) under
      // the old ladder; it now shares Pilot with UUP and CORRECTION.
      expect(mctStateToSizingMode("RALLY MODE")).toBe(0);
    });

    test("CORRECTION → Pilot (0, 0.25%)", () => {
      // Was Defense (0.50%). Defense is retired; Pilot inherits the
      // "most conservative auto tier" role.
      expect(mctStateToSizingMode("CORRECTION")).toBe(0);
    });

    test("unknown state defaults to Pilot (safest floor, not a middle tier)", () => {
      // Retiered behavior. Old default was Normal (middle). New
      // default is Pilot (0.25%): the redesign is intentionally
      // conservative and the manual override is downward-only — a
      // middle-tier default on an engine hiccup would violate the
      // "when in doubt, be smaller" invariant.
      expect(mctStateToSizingMode(null)).toBe(0);
      expect(mctStateToSizingMode(undefined)).toBe(0);
      expect(mctStateToSizingMode("")).toBe(0);
      expect(mctStateToSizingMode("UNKNOWN")).toBe(0);
      expect(mctStateToSizingMode("POWERTREND ON")).toBe(0);  // V10 legacy
    });
  });

  describe("describeMctSource", () => {
    // Function is still NAMED describeMctSource (engine internals are
    // still MCT) but the user-visible output says "M Factor".
    test("formats the five states with 'M Factor' as user-facing label", () => {
      expect(describeMctSource("POWERTREND")).toBe("from M Factor POWERTREND");
      expect(describeMctSource("UPTREND")).toBe("from M Factor UPTREND");
      expect(describeMctSource("UPTREND UNDER PRESSURE"))
        .toBe("from M Factor UPTREND UNDER PRESSURE");
      expect(describeMctSource("RALLY MODE")).toBe("from M Factor RALLY MODE");
      expect(describeMctSource("CORRECTION")).toBe("from M Factor CORRECTION");
    });

    test("unknown / null state surfaces 'M Factor state unknown' instead of guessing", () => {
      expect(describeMctSource(null)).toBe("M Factor state unknown");
      expect(describeMctSource(undefined)).toBe("M Factor state unknown");
      expect(describeMctSource("")).toBe("M Factor state unknown");
      expect(describeMctSource("POWERTREND ON")).toBe("M Factor state unknown");
    });

    test("appends downshift reason when exit-ladder floor is below state mode", () => {
      // POWERTREND alone → Offense (2). 50 SMA Violation floors to 0
      // (Pilot post-retier), so the floor is BELOW the state mode →
      // label surfaces the reason.
      expect(describeMctSource("POWERTREND", { idx: 0, reason: "50 SMA Violation" }))
        .toBe("from M Factor POWERTREND, downshifted by 50 SMA Violation");
    });

    test("no downshift suffix when floor doesn't actually constrain", () => {
      // CORRECTION alone → Pilot (0) post-retier. A 21 EMA Confirmed
      // Break floor of 1 (Normal) doesn't bind because state is
      // already more conservative. Don't append a misleading
      // "downshifted by" suffix.
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
      // Only fired Violations / Confirmed Break downshift.
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

    test("50 SMA Violation → Pilot (0) — retiered from Defense", () => {
      // Semantics preserved (most-conservative floor); tier name
      // changed since Defense was retired. Pilot now occupies index 0.
      expect(exitLadderFloor([{ signal: "50 SMA Violation" }]))
        .toEqual({ idx: 0, reason: "50 SMA Violation" });
    });

    test("most severe wins when multiple alerts fire", () => {
      // 3 consecutive closes below 21 EMA (= Confirmed Break) AND price
      // below 50 SMA with >1% intraday undercut (= 50 SMA Violation).
      // Floor should be Pilot (0), not Normal (1).
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
    test("POWERTREND with no alerts → Offense (state wins, 0.75%)", () => {
      expect(deriveAutoSizingMode("POWERTREND", []).idx).toBe(2);
    });

    test("POWERTREND + 50 SMA Violation → Pilot (floor wins)", () => {
      // State says Offense (2); floor says Pilot (0). Minimum wins.
      // New trades sized at 0.25% risk even though the regime hasn't
      // flipped to CORRECTION.
      const result = deriveAutoSizingMode("POWERTREND", [{ signal: "50 SMA Violation" }]);
      expect(result.idx).toBe(0);
      expect(result.source.stateIdx).toBe(2);
      expect(result.source.floor).toEqual({ idx: 0, reason: "50 SMA Violation" });
    });

    test("POWERTREND + 21 EMA Confirmed Break → Normal (floor wins)", () => {
      expect(deriveAutoSizingMode("POWERTREND", [{ signal: "21 EMA Confirmed Break" }]).idx).toBe(1);
    });

    test("POWERTREND + 21 EMA Violation → Normal (floor wins, user spec)", () => {
      // Any fired 21 EMA-tier signal floors at Normal (1). Same for
      // Confirmed Break and single-close Violation.
      expect(deriveAutoSizingMode("POWERTREND", [{ signal: "21 EMA Violation" }]).idx).toBe(1);
    });

    test("UPTREND + 50 SMA Violation → Pilot (state was Normal; floor drops to 0)", () => {
      // UPTREND alone → Normal (1). 50 SMA Violation floor → 0.
      // MIN(1, 0) = 0. Retier didn't change the min-wins semantic,
      // only the tier labels/indices.
      expect(deriveAutoSizingMode("UPTREND", [{ signal: "50 SMA Violation" }]).idx).toBe(0);
    });

    test("CORRECTION + 50 SMA Violation → Pilot (no-op floor; both agree)", () => {
      expect(deriveAutoSizingMode("CORRECTION", [{ signal: "50 SMA Violation" }]).idx).toBe(0);
    });

    test("RALLY MODE + 50 SMA Violation → Pilot (no-op floor; state already at 0)", () => {
      // RALLY MODE dropped from Normal → Pilot in the retier. Floor
      // was already going to land at 0 too, so nothing changes when
      // both agree.
      expect(deriveAutoSizingMode("RALLY MODE", [{ signal: "50 SMA Violation" }]).idx).toBe(0);
    });

    test("UPTREND UNDER PRESSURE + 50 SMA Violation → Pilot (both at 0)", () => {
      // UUP alone is now Pilot (0). A 50 SMA Violation floor of 0
      // doesn't move it further. MIN(0, 0) = 0.
      const result = deriveAutoSizingMode(
        "UPTREND UNDER PRESSURE",
        [{ signal: "50 SMA Violation" }],
      );
      expect(result.idx).toBe(0);
      expect(result.source.stateIdx).toBe(0);
      expect(result.source.floor).toEqual({ idx: 0, reason: "50 SMA Violation" });
    });

    test("POWERTREND + Watch only → Offense (no floor binds)", () => {
      expect(deriveAutoSizingMode("POWERTREND", [
        { signal: "21 EMA Watch" },
        { signal: "50 SMA Watch" },
      ]).idx).toBe(2);
    });

    test("auto-derivation always returns a canonical 0/1/2 across the state matrix", () => {
      // Retier collapsed the tier count from 4 to 3, so the auto
      // path IS the full canonical index range now. Exhaustive smoke
      // test across state × alert-set combinations pins that no
      // out-of-range index can leak.
      const states = ["POWERTREND", "UPTREND", "UPTREND UNDER PRESSURE",
                      "RALLY MODE", "CORRECTION", null, "UNKNOWN"];
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
          expect([0, 1, 2]).toContain(r.idx);
        }
      }
    });
  });

  describe("clampManualToDownwardOnly (New Entry override guard)", () => {
    // New helper introduced with the retier. New Entry's manual mode
    // override is DOWNWARD-ONLY: the user may pick a tier smaller than
    // the auto-selected one, never larger. Position Sizer + Log Buy
    // retain their bidirectional override — this helper is used only
    // by New Entry.
    test("user picks smaller (more conservative) than auto → user wins", () => {
      // Auto = Offense (2); user picks Pilot (0). Downshift honored.
      expect(clampManualToDownwardOnly(2, 0)).toBe(0);
    });

    test("user picks equal to auto → auto index unchanged", () => {
      expect(clampManualToDownwardOnly(1, 1)).toBe(1);
      expect(clampManualToDownwardOnly(2, 2)).toBe(2);
      expect(clampManualToDownwardOnly(0, 0)).toBe(0);
    });

    test("user picks larger (more aggressive) than auto → auto still wins (clamp)", () => {
      // The whole reason the helper exists. If auto is Pilot because
      // exit ladder is firing, the user can't manually re-lift back
      // to Normal / Offense on New Entry. Position Sizer + Log Buy
      // still allow it via their existing bidirectional override —
      // this helper is New Entry-only.
      expect(clampManualToDownwardOnly(0, 2)).toBe(0);  // Pilot auto, Offense user → Pilot
      expect(clampManualToDownwardOnly(1, 2)).toBe(1);  // Normal auto, Offense user → Normal
      expect(clampManualToDownwardOnly(0, 1)).toBe(0);  // Pilot auto, Normal  user → Pilot
    });
  });
});
