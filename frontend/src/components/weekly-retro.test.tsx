import { render, screen, waitFor, fireEvent, act } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach } from "vitest";

// ResizeObserver isn't in jsdom — harmless stub for parity with other tests.
class ResizeObserverStub {
  observe(): void { /* noop */ }
  unobserve(): void { /* noop */ }
  disconnect(): void { /* noop */ }
}
(globalThis as any).ResizeObserver = ResizeObserverStub;

// jsdom's localStorage shim is sometimes absent — install a minimal in-memory
// stand-in (mirrors the active-campaign.test.tsx pattern). Necessary for the
// Phase 0 cleanup assertion that removes the legacy "mo-weekly-retros" key.
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

vi.mock("@/lib/api", () => ({
  api: {
    tradesRecent: vi.fn(),
    weeklyRetroList: vi.fn(),
    weeklyRetroUpsert: vi.fn(),
  },
  getActivePortfolio: () => "CanSlim",
}));

import { api } from "@/lib/api";
import { WeeklyRetro } from "./weekly-retro";

const mTradesRecent = vi.mocked(api.tradesRecent);
const mWeeklyList   = vi.mocked(api.weeklyRetroList);
const mWeeklyUpsert = vi.mocked(api.weeklyRetroUpsert);

function setupDefaults() {
  mTradesRecent.mockResolvedValue({ details: [], lot_closures: [] } as any);
  mWeeklyList.mockResolvedValue([]);
  mWeeklyUpsert.mockResolvedValue({
    id: 1, portfolio: "CanSlim", week_start: "2026-05-11",
    week_grade: "B+", best_decision: "good", worst_decision: "bad",
    rule_change: false, rule_change_text: "",
    ticker_grades: {},
    created_at: "2026-05-13T00:00:00", updated_at: "2026-05-13T00:00:00",
  } as any);
}

// Wait for the component's two-step mount: tradesRecent resolves and flips
// loading=false, then weeklyRetroList populates the retros map. Both must
// settle before any interaction.
async function mountAndSettle() {
  render(<WeeklyRetro navColor="#6366f1" />);
  await waitFor(() => expect(mWeeklyList).toHaveBeenCalled());
  // tradesRecent resolves on mount; once loading flips false the Grade tab
  // renders the Save Weekly Retro button. That button is our render-ready
  // signal.
  await screen.findByRole("button", { name: /save weekly retro/i });
}

// Find the Week Grade <select> via its unique placeholder option text. The
// per-ticker selects use "Select..." without "grade", so this is safe.
async function findWeekGradeSelect(): Promise<HTMLSelectElement> {
  const placeholder = await screen.findByRole("option", { name: "Select grade..." });
  return placeholder.closest("select") as HTMLSelectElement;
}

describe("WeeklyRetro — Phase 0 server persistence swap", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaults();
    try { localStorage.clear(); } catch { /* shim has clear */ }
  });

  test("fetches the retro list via api.weeklyRetroList on mount", async () => {
    await mountAndSettle();
    expect(mWeeklyList).toHaveBeenCalledWith("CanSlim");
  });

  test("clears the legacy localStorage key after a successful list fetch", async () => {
    localStorage.setItem("mo-weekly-retros", JSON.stringify({ "2026-05-04": { weekGrade: "C" } }));
    expect(localStorage.getItem("mo-weekly-retros")).not.toBeNull();

    await mountAndSettle();
    await waitFor(() => {
      expect(localStorage.getItem("mo-weekly-retros")).toBeNull();
    });
  });

  test("hydration alone does not fire an auto-save (dirty flag guards it)", async () => {
    mWeeklyList.mockResolvedValue([{
      id: 7, portfolio: "CanSlim", week_start: "2026-05-04",
      week_grade: "A", best_decision: "x", worst_decision: "y",
      rule_change: false, rule_change_text: "",
      ticker_grades: {},
      created_at: "2026-05-05T00:00:00", updated_at: "2026-05-05T00:00:00",
    }] as any);

    await mountAndSettle();
    // Wait past the 800ms debounce. dirtyRef=false on mount, so no save.
    await new Promise(r => setTimeout(r, 1100));
    expect(mWeeklyUpsert).not.toHaveBeenCalled();
  });

  test("clicking Save Weekly Retro PUTs via api.weeklyRetroUpsert", async () => {
    await mountAndSettle();

    const saveBtn = screen.getByRole("button", { name: /save weekly retro/i });
    await act(async () => { fireEvent.click(saveBtn); });
    await waitFor(() => expect(mWeeklyUpsert).toHaveBeenCalledTimes(1));

    const payload = mWeeklyUpsert.mock.calls[0][0];
    expect(payload.portfolio).toBe("CanSlim");
    expect(payload.week_start).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(payload).toHaveProperty("ticker_grades");
  });

  test("editing the week grade fires a debounced auto-save after 800ms", async () => {
    await mountAndSettle();

    const select = await findWeekGradeSelect();
    await act(async () => { fireEvent.change(select, { target: { value: "B+" } }); });

    // Below the debounce window — no save yet.
    await new Promise(r => setTimeout(r, 400));
    expect(mWeeklyUpsert).not.toHaveBeenCalled();

    // Past the 800ms window — save fires once.
    await new Promise(r => setTimeout(r, 700));
    await waitFor(() => expect(mWeeklyUpsert).toHaveBeenCalledTimes(1));
    expect(mWeeklyUpsert.mock.calls[0][0].week_grade).toBe("B+");
  });

  test("rapid edits collapse into a single debounced save (timer resets)", async () => {
    await mountAndSettle();

    const select = await findWeekGradeSelect();
    // Three changes within the debounce window — only the final value saves.
    await act(async () => { fireEvent.change(select, { target: { value: "A" } }); });
    await new Promise(r => setTimeout(r, 300));
    await act(async () => { fireEvent.change(select, { target: { value: "B" } }); });
    await new Promise(r => setTimeout(r, 300));
    await act(async () => { fireEvent.change(select, { target: { value: "C" } }); });

    expect(mWeeklyUpsert).not.toHaveBeenCalled();
    await new Promise(r => setTimeout(r, 1000));
    await waitFor(() => expect(mWeeklyUpsert).toHaveBeenCalledTimes(1));
    expect(mWeeklyUpsert.mock.calls[0][0].week_grade).toBe("C");
  });

  test("failed save surfaces saveMsg but keeps local state for retry", async () => {
    mWeeklyUpsert.mockResolvedValue({ error: "constraint violation" } as any);
    await mountAndSettle();

    const saveBtn = screen.getByRole("button", { name: /save weekly retro/i });
    await act(async () => { fireEvent.click(saveBtn); });

    expect(await screen.findByText(/save failed/i)).toBeInTheDocument();
    // Save button still present — local state preserved for retry.
    expect(screen.getByRole("button", { name: /save weekly retro/i })).toBeInTheDocument();
  });

  // Regression: a stale weeklyRetroList response (server snapshot taken
  // before the user's save reached the DB) used to overwrite the
  // just-saved row in local state, then the hydration useEffect ran on
  // the retros change and wiped every field. Two fixes applied: the
  // mount fetch now functionally merges with prev (local wins on shared
  // keys), and hydration deps no longer include retros (so post-save
  // setRetros doesn't retrigger the effect at all).
  test("preserves locally-saved row when stale list response arrives after save", async () => {
    let resolveList: (rows: any[]) => void = () => { /* placeholder */ };
    const listPromise = new Promise<any[]>(r => { resolveList = r; });
    mWeeklyList.mockReturnValueOnce(listPromise as any);

    // Echo whatever was saved so the response shape matches the typed input.
    mWeeklyUpsert.mockImplementation(async (p: any) => ({
      id: 42,
      portfolio: p.portfolio,
      week_start: p.week_start,
      week_grade: p.week_grade,
      best_decision: p.best_decision,
      worst_decision: p.worst_decision,
      rule_change: p.rule_change,
      rule_change_text: p.rule_change_text,
      ticker_grades: p.ticker_grades,
      created_at: "2026-05-13T00:00:00Z",
      updated_at: "2026-05-13T00:00:00Z",
    }));

    await mountAndSettle();

    // Type while the list fetch is still pending. Labels aren't formally
    // linked in this component, so query by the unique placeholder text.
    const input = await screen.findByPlaceholderText(/one win to repeat/i);
    await act(async () => {
      fireEvent.change(input, { target: { value: "saved before list" } });
    });

    // Debounced save fires (~800ms idle) — wait for the PUT.
    await waitFor(
      () => expect(mWeeklyUpsert).toHaveBeenCalledTimes(1),
      { timeout: 2000 },
    );
    expect(mWeeklyUpsert.mock.calls[0][0].best_decision).toBe("saved before list");

    // NOW the stale list response arrives — empty (snapshot taken
    // before the save reached the DB). Pre-fix this would clobber the
    // saved row and trigger hydration's else-branch wipe.
    await act(async () => { resolveList([]); });

    // Field must still show the typed value — not be wiped.
    await waitFor(() => {
      expect((input as HTMLInputElement).value).toBe("saved before list");
    });
  });
});
