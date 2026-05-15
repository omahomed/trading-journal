import { render, screen, fireEvent, act } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach } from "vitest";

import { CloseTheWeek, type CloseTheWeekState } from "./close-the-week";

// Minimal localStorage shim — jsdom usually has one, but some tests run
// in environments without it. (Mirrors the notes-rail.test.tsx idiom.)
if (typeof window !== "undefined" && !(window as any).localStorage?.getItem) {
  const _store = new Map<string, string>();
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: {
      getItem: (k: string) => _store.get(k) ?? null,
      setItem: (k: string, v: string) => { _store.set(k, String(v)); },
      removeItem: (k: string) => { _store.delete(k); },
      clear: () => { _store.clear(); },
      key: (i: number) => Array.from(_store.keys())[i] ?? null,
      get length() { return _store.size; },
    },
    writable: true,
  });
}

function makeState(overrides: Partial<CloseTheWeekState> = {}): CloseTheWeekState {
  return {
    week_grade: "",
    execution_grade: "",
    process_grade: "",
    pnl_grade: "",
    overall_override: false,
    reviewed_at: null,
    best_decision: "",
    worst_decision: "",
    rule_change: false,
    rule_change_text: "",
    ...overrides,
  };
}

interface HarnessProps {
  initial?: Partial<CloseTheWeekState>;
  recentOverall?: string[];
  onSave?: () => void;
}

/** Stateful test harness that mirrors the parent's state-owning pattern.
 *  CloseTheWeek is presentation-only; the harness routes onChange back
 *  to local state so observable behavior (the derived overall syncing,
 *  the override flag flipping, etc.) matches what users see in
 *  WeeklyRetro. */
function Harness({ initial, recentOverall = [], onSave }: HarnessProps) {
  const [state, setState] = (require("react") as typeof import("react"))
    .useState<CloseTheWeekState>(makeState(initial));
  return (
    <CloseTheWeek
      state={state}
      onChange={(p) => setState(prev => ({ ...prev, ...p }))}
      onSave={onSave ?? (() => {})}
      recentOverall={recentOverall}
    />
  );
}

describe("CloseTheWeek — Phase 4.6 redesign", () => {
  beforeEach(() => {
    try { localStorage.clear(); } catch { /* shim */ }
  });

  test("renders 3 axis selectors (Execution / Process / P&L)", () => {
    render(<Harness />);
    expect(screen.getByLabelText("Execution grade")).toBeInTheDocument();
    expect(screen.getByLabelText("Process grade")).toBeInTheDocument();
    expect(screen.getByLabelText("P&L grade")).toBeInTheDocument();
  });

  test("picking a value on an axis selector triggers onChange with that field", () => {
    const onChange = vi.fn();
    render(
      <CloseTheWeek state={makeState()} onChange={onChange}
                    onSave={vi.fn()} recentOverall={[]} />
    );
    const exec = screen.getByLabelText("Execution grade") as HTMLSelectElement;
    fireEvent.change(exec, { target: { value: "A-" } });
    expect(onChange).toHaveBeenCalledWith({ execution_grade: "A-" });
  });

  test("Overall card displays derived value when override is false and all 3 axes are set", async () => {
    // Render in the harness so the derived-overall sync effect can fire
    // and update local state (the parent in production owns the same
    // mirror via handleCtwChange → setWeekGrade).
    render(<Harness initial={{
      execution_grade: "A", process_grade: "A", pnl_grade: "A",
    }} />);
    // Derived A/A/A → A. State line uses italic "Derived from A".
    expect(screen.getByTestId("overall-state-line")).toHaveTextContent(/Derived from A/);
    // The Overall <select> reflects the derived A.
    const sel = screen.getByTestId("overall-select") as HTMLSelectElement;
    expect(sel.value).toBe("A");
  });

  test("Overall card displays client week_grade when overall_override is true", () => {
    render(<Harness initial={{
      execution_grade: "A", process_grade: "A", pnl_grade: "A",
      week_grade: "B-", overall_override: true,
    }} />);
    // Override is true → state line reads "Overridden — average is A".
    expect(screen.getByTestId("overall-state-line")).toHaveTextContent(/Overridden.*A/);
    // The select shows B- (the user's override), not the derived A.
    const sel = screen.getByTestId("overall-select") as HTMLSelectElement;
    expect(sel.value).toBe("B-");
  });

  test("picking a value in the Overall select that differs from derived pins overall_override = true", async () => {
    render(<Harness initial={{
      execution_grade: "A", process_grade: "A", pnl_grade: "A",
    }} />);
    // Initial: derived A, override false. State line shows "Derived from A".
    expect(screen.getByTestId("overall-state-line")).toHaveTextContent(/Derived/);
    const sel = screen.getByTestId("overall-select") as HTMLSelectElement;
    // User picks B-, which doesn't match derived A → override pins.
    await act(async () => {
      fireEvent.change(sel, { target: { value: "B-" } });
    });
    expect(screen.getByTestId("overall-state-line")).toHaveTextContent(/Overridden/);
  });

  test("picking the derived value back clears overall_override", async () => {
    render(<Harness initial={{
      execution_grade: "A", process_grade: "A", pnl_grade: "A",
      week_grade: "B", overall_override: true,
    }} />);
    expect(screen.getByTestId("overall-state-line")).toHaveTextContent(/Overridden/);
    const sel = screen.getByTestId("overall-select") as HTMLSelectElement;
    // Re-pick A (the derived value) → override clears.
    await act(async () => {
      fireEvent.change(sel, { target: { value: "A" } });
    });
    expect(screen.getByTestId("overall-state-line")).toHaveTextContent(/Derived/);
  });

  test("RecentOverallCard renders the last 4 weeks' grades from recentOverall prop", () => {
    render(<Harness recentOverall={["A-", "B+", "B", "A"]} initial={{
      execution_grade: "B", process_grade: "B", pnl_grade: "B",
    }} />);
    const card = screen.getByTestId("recent-overall");
    // Each grade is rendered; the strip uses dot separators.
    expect(card).toHaveTextContent("A-");
    expect(card).toHaveTextContent("B+");
    expect(card).toHaveTextContent("B");
    // The arrow target shows the effective overall (derived B/B/B → B).
    expect(screen.getByTestId("recent-overall-current")).toHaveTextContent("B");
  });

  test("RecentOverallCard renders placeholder dashes when recentOverall is empty", () => {
    render(<Harness recentOverall={[]} />);
    const card = screen.getByTestId("recent-overall");
    // 4 em-dashes for the 4 placeholder slots.
    const dashes = card.textContent?.match(/—/g) ?? [];
    expect(dashes.length).toBeGreaterThanOrEqual(4);
  });

  test("grade interpretation strip renders GRADE_DEFS copy for the effective overall", () => {
    render(<Harness initial={{
      execution_grade: "A-", process_grade: "A-", pnl_grade: "A-",
    }} />);
    // A-/A-/A- averages to A- exactly → "Strong — one small slip from textbook."
    expect(screen.getByTestId("grade-interpretation"))
      .toHaveTextContent(/Strong — one small slip from textbook/);
  });

  test("axis selectors are disabled when reviewed_at is non-null", () => {
    render(<CloseTheWeek
      state={makeState({
        execution_grade: "A", process_grade: "A", pnl_grade: "A",
        reviewed_at: "2026-05-14T10:00:00Z",
      })}
      onChange={vi.fn()} onSave={vi.fn()} recentOverall={[]} />);
    expect(screen.getByLabelText("Execution grade")).toBeDisabled();
    expect(screen.getByLabelText("Process grade")).toBeDisabled();
    expect(screen.getByLabelText("P&L grade")).toBeDisabled();
    expect(screen.getByLabelText("Overall week grade")).toBeDisabled();
  });

  test("reflections textareas remain editable when reviewed (only grades lock)", () => {
    const onChange = vi.fn();
    render(<CloseTheWeek
      state={makeState({ reviewed_at: "2026-05-14T10:00:00Z" })}
      onChange={onChange} onSave={vi.fn()} recentOverall={[]} />);
    const best = screen.getByLabelText("What worked?") as HTMLTextAreaElement;
    expect(best).not.toBeDisabled();
    fireEvent.change(best, { target: { value: "still editable" } });
    expect(onChange).toHaveBeenCalledWith({ best_decision: "still editable" });
  });

  test("Save button label + color swap when reviewed_at is set", () => {
    const { rerender } = render(<CloseTheWeek
      state={makeState()} onChange={vi.fn()} onSave={vi.fn()} recentOverall={[]} />);
    let btn = screen.getByTestId("save-button");
    expect(btn).toHaveTextContent("Save weekly retro");
    // Unreviewed: indigo bg.
    expect(btn.getAttribute("style") || "").toContain("rgb(99, 102, 241)");

    rerender(<CloseTheWeek
      state={makeState({ reviewed_at: "2026-05-14T10:00:00Z" })}
      onChange={vi.fn()} onSave={vi.fn()} recentOverall={[]} />);
    btn = screen.getByTestId("save-button");
    expect(btn).toHaveTextContent("Save & close week");
    // Reviewed: green bg.
    expect(btn.getAttribute("style") || "").toContain("rgb(8, 168, 107)");
  });

  test("Mark-reviewed checkbox toggles reviewed_at on/off", () => {
    const onChange = vi.fn();
    render(<CloseTheWeek
      state={makeState()} onChange={onChange} onSave={vi.fn()} recentOverall={[]} />);
    const cb = screen.getByTestId("mark-reviewed-checkbox") as HTMLInputElement;
    fireEvent.click(cb);
    // First click sets reviewed_at to an ISO string.
    const call = onChange.mock.calls.at(-1)?.[0];
    expect(call?.reviewed_at).toMatch(/^\d{4}-\d{2}-\d{2}T/);
  });

  test("Mark-reviewed click on a reviewed retro clears reviewed_at", () => {
    const onChange = vi.fn();
    render(<CloseTheWeek
      state={makeState({ reviewed_at: "2026-05-14T10:00:00Z" })}
      onChange={onChange} onSave={vi.fn()} recentOverall={[]} />);
    const cb = screen.getByTestId("mark-reviewed-checkbox") as HTMLInputElement;
    fireEvent.click(cb);
    expect(onChange).toHaveBeenLastCalledWith({ reviewed_at: null });
  });

  test("Save button calls onSave when clicked", () => {
    const onSave = vi.fn();
    render(<CloseTheWeek
      state={makeState()} onChange={vi.fn()} onSave={onSave} recentOverall={[]} />);
    fireEvent.click(screen.getByTestId("save-button"));
    expect(onSave).toHaveBeenCalledTimes(1);
  });

  test("rule change checkbox reveals the textarea conditionally", () => {
    const { rerender } = render(<CloseTheWeek
      state={makeState()} onChange={vi.fn()} onSave={vi.fn()} recentOverall={[]} />);
    expect(screen.queryByTestId("rule-change-textarea")).toBeNull();
    rerender(<CloseTheWeek
      state={makeState({ rule_change: true })}
      onChange={vi.fn()} onSave={vi.fn()} recentOverall={[]} />);
    expect(screen.getByTestId("rule-change-textarea")).toBeInTheDocument();
  });

  test("grade-tinted band color reflects the effective overall", () => {
    // Render with axes A/A/A → derived A → green band.
    const { rerender } = render(<CloseTheWeek
      state={makeState({
        execution_grade: "A", process_grade: "A", pnl_grade: "A",
        week_grade: "A",
      })}
      onChange={vi.fn()} onSave={vi.fn()} recentOverall={[]} />);
    let band = screen.getByTestId("grade-band");
    expect(band.getAttribute("style") || "").toContain("rgb(8, 168, 107)");

    // Re-render with F/F/F → derived F → red band. Passing the state
    // directly (not via Harness) bypasses the initialized-once useState
    // so each render uses the fresh state prop.
    rerender(<CloseTheWeek
      state={makeState({
        execution_grade: "F", process_grade: "F", pnl_grade: "F",
        week_grade: "F",
      })}
      onChange={vi.fn()} onSave={vi.fn()} recentOverall={[]} />);
    band = screen.getByTestId("grade-band");
    expect(band.getAttribute("style") || "").toContain("rgb(229, 72, 77)");
  });
});
