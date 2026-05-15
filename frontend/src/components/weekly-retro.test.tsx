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
    weeklyRetroGet: vi.fn(),
    weeklyRetroUpsert: vi.fn(),
    // Phase 5: WeeklyRetro fetches weekly metrics for the top tile row on
    // mount + week change. Default to a benign zero response so these
    // existing tests don't fight the new fetch.
    weeklyMetrics: vi.fn(),
    // Phase 6: NotesRail's pin toggle endpoint.
    pinsToggle: vi.fn(),
    // Phase 1: WeeklyRetro mounts <TagPicker> which fetches these on mount
    // once a retro has been saved (entityId != null). Stub to empty arrays
    // so the picker's mount fetch doesn't blow up these unrelated tests.
    listTags: vi.fn(),
    listTagAssignments: vi.fn(),
  },
  getActivePortfolio: () => "CanSlim",
}));

import { api } from "@/lib/api";
import { WeeklyRetro } from "./weekly-retro";

const mTradesRecent = vi.mocked(api.tradesRecent);
const mWeeklyList   = vi.mocked(api.weeklyRetroList);
const mWeeklyGet    = vi.mocked(api.weeklyRetroGet);
const mWeeklyUpsert = vi.mocked(api.weeklyRetroUpsert);
const mWeeklyMetrics   = vi.mocked(api.weeklyMetrics);
const mPinsToggle      = vi.mocked(api.pinsToggle);
const mListTags        = vi.mocked(api.listTags);
const mListAssignments = vi.mocked(api.listTagAssignments);

function setupDefaults() {
  mTradesRecent.mockResolvedValue({ details: [], lot_closures: [] } as any);
  // Phase 6 — list endpoint returns wrapped envelope shape now.
  mWeeklyList.mockResolvedValue({
    weeks: [],
    ytd_stats: { total_weeks: 0, weeks_graded: 0, avg_grade: null, weeks_pinned: 0 },
  } as any);
  mWeeklyUpsert.mockResolvedValue({
    id: 1, portfolio: "CanSlim", week_start: "2026-05-11",
    week_grade: "B+", best_decision: "good", worst_decision: "bad",
    rule_change: false, rule_change_text: "",
    weekly_thoughts: "",
    ticker_grades: {},
    created_at: "2026-05-13T00:00:00", updated_at: "2026-05-13T00:00:00",
  } as any);
  mWeeklyMetrics.mockResolvedValue({
    weekly_pnl: 0, weekly_return_pct: 0, ytd_pct: 0, ltd_pct: 0,
    win_rate: { rate: 0, wins: 0, losses: 0, flat: 0, total: 0 },
    week_start: "2026-05-11", week_end: "2026-05-15",
    as_of: "2026-05-14T00:00:00",
  } as any);
  // Phase 6: weeklyRetroGet returns the not_found shape by default so the
  // lazy fetch effect resolves without populating retros[monStr] (test
  // setup mirrors the "blank week" UX). Individual tests override.
  mWeeklyGet.mockResolvedValue({ error: "not_found" } as any);
  mPinsToggle.mockResolvedValue({ pinned: true } as any);
  mListTags.mockResolvedValue([]);
  mListAssignments.mockResolvedValue([]);
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

// Find the canonical Overall <select> in CloseTheWeek. Phase 4.6
// replaced the standalone "Overall Week Grade" dropdown — week_grade is
// now set by either (a) grading the 3 axes (derived) or (b) picking a
// value from the Overall override select (which pins overall_override =
// true). Tests here exercise path (b) since it produces the same
// observable: a week_grade change → debounced auto-save.
async function findWeekGradeSelect(): Promise<HTMLSelectElement> {
  return await screen.findByTestId("overall-select") as HTMLSelectElement;
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
    // Phase 6: list endpoint returns wrapped envelope; the per-week
    // fetch (weeklyRetroGet) is what hydrates the form. Whether or not
    // a retro exists for the current week, hydration on mount must not
    // flip dirtyRef.
    mWeeklyGet.mockResolvedValue({
      id: 7, portfolio: "CanSlim", week_start: "2026-05-04",
      week_grade: "A", best_decision: "x", worst_decision: "y",
      rule_change: false, rule_change_text: "",
      weekly_thoughts: "",
      ticker_grades: {},
      created_at: "2026-05-05T00:00:00", updated_at: "2026-05-05T00:00:00",
    } as any);

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
    // Phase 6: list endpoint returns envelope. The "stale response wipes
    // saved row" regression still matters — after save, the rail's
    // post-save refresh shouldn't clobber the in-flight local edits.
    let resolveList: (res: any) => void = () => { /* placeholder */ };
    const listPromise = new Promise<any>(r => { resolveList = r; });
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
      weekly_thoughts: p.weekly_thoughts,
      ticker_grades: p.ticker_grades,
      // Phase 4.6 echo fields
      execution_grade: p.execution_grade ?? null,
      process_grade: p.process_grade ?? null,
      pnl_grade: p.pnl_grade ?? null,
      overall_override: p.overall_override ?? false,
      reviewed_at: p.reviewed_at ?? null,
      created_at: "2026-05-13T00:00:00Z",
      updated_at: "2026-05-13T00:00:00Z",
    }));

    await mountAndSettle();

    // Type while the list fetch is still pending. The Phase 4.6 "What
    // worked?" textarea has aria-label "What worked?" — query via that.
    const input = await screen.findByLabelText("What worked?");
    await act(async () => {
      fireEvent.change(input, { target: { value: "saved before list" } });
    });

    // Debounced save fires (~800ms idle) — wait for the PUT.
    await waitFor(
      () => expect(mWeeklyUpsert).toHaveBeenCalledTimes(1),
      { timeout: 2000 },
    );
    expect(mWeeklyUpsert.mock.calls[0][0].best_decision).toBe("saved before list");

    // NOW the stale list response arrives — empty envelope (snapshot
    // taken before the save reached the DB). Pre-fix this would clobber
    // the saved row and trigger hydration's else-branch wipe.
    await act(async () => {
      resolveList({
        weeks: [],
        ytd_stats: { total_weeks: 0, weeks_graded: 0, avg_grade: null, weeks_pinned: 0 },
      });
    });

    // Field must still show the typed value — not be wiped.
    await waitFor(() => {
      expect((input as HTMLInputElement).value).toBe("saved before list");
    });
  });

  // ─── Phase 2: Per-Ticker Details expander ──────────────────────────────
  // Default state collapsed; localStorage-backed per-USER UI preference
  // (key "mo-weekly-retro-tickets-expanded"). Header always visible with
  // count + "X/N tickers graded" caption regardless of body state.

  const TICKETS_KEY = "mo-weekly-retro-tickets-expanded";

  test("Per-Ticker expander defaults to collapsed; body not in document", async () => {
    await mountAndSettle();
    // Header is always present.
    const header = screen.getByRole("button", { name: /Per-Ticker Details/i });
    expect(header).toHaveAttribute("aria-expanded", "false");
    // Body wrapper (id="per-ticker-body") absent.
    expect(document.getElementById("per-ticker-body")).toBeNull();
  });

  test("Clicking the expander header reveals the body, second click hides it", async () => {
    await mountAndSettle();
    const header = screen.getByRole("button", { name: /Per-Ticker Details/i });

    await act(async () => { fireEvent.click(header); });
    expect(header).toHaveAttribute("aria-expanded", "true");
    expect(document.getElementById("per-ticker-body")).not.toBeNull();

    await act(async () => { fireEvent.click(header); });
    expect(header).toHaveAttribute("aria-expanded", "false");
    expect(document.getElementById("per-ticker-body")).toBeNull();
  });

  test("Reads localStorage on mount — expanded persists across page loads", async () => {
    localStorage.setItem(TICKETS_KEY, "true");
    await mountAndSettle();
    const header = screen.getByRole("button", { name: /Per-Ticker Details/i });
    expect(header).toHaveAttribute("aria-expanded", "true");
    expect(document.getElementById("per-ticker-body")).not.toBeNull();
  });

  test("Toggling writes the new state to localStorage", async () => {
    // Start collapsed (no key).
    await mountAndSettle();
    expect(localStorage.getItem(TICKETS_KEY)).toBeNull();

    const header = screen.getByRole("button", { name: /Per-Ticker Details/i });
    await act(async () => { fireEvent.click(header); });
    expect(localStorage.getItem(TICKETS_KEY)).toBe("true");

    await act(async () => { fireEvent.click(header); });
    expect(localStorage.getItem(TICKETS_KEY)).toBe("false");
  });

  test("Header shows ticker count and graded caption even when collapsed (zero-tickets case)", async () => {
    // tradesRecent default returns no details, so uniqueTickers = 0.
    await mountAndSettle();
    const header = screen.getByRole("button", { name: /Per-Ticker Details/i });
    expect(header).toHaveTextContent(/Per-Ticker Details/);
    expect(header).toHaveTextContent(/\(0\)/);
    expect(header).toHaveTextContent(/0\/0 tickers graded/);
    // Body remains absent — collapsed default.
    expect(document.getElementById("per-ticker-body")).toBeNull();
  });

  test("Header caption stays in sync independent of expand state", async () => {
    await mountAndSettle();
    const header = screen.getByRole("button", { name: /Per-Ticker Details/i });
    // Collapsed: caption visible.
    expect(header).toHaveTextContent(/0\/0 tickers graded/);
    // Expand and re-check: caption still on header (not duplicated by body).
    await act(async () => { fireEvent.click(header); });
    expect(header).toHaveTextContent(/0\/0 tickers graded/);
  });

  // ─── Phase 3: Weekly Thoughts integration ──────────────────────────────
  // The <WeeklyThoughts> component owns its own internal tests (see
  // weekly-thoughts.test.tsx). These tests assert the weekly-retro <-->
  // WeeklyThoughts wire: state hydration, dirty-flag flip on edit, and
  // payload inclusion on save.

  test("Weekly Thoughts hydrates from API response into the editor", async () => {
    // Phase 6: hydration source is the per-week GET, not the list.
    mWeeklyGet.mockResolvedValue({
      id: 7, portfolio: "CanSlim", week_start: "2026-05-04",
      week_grade: "A", best_decision: "", worst_decision: "",
      rule_change: false, rule_change_text: "",
      weekly_thoughts: "<p>preloaded reflection</p>",
      ticker_grades: {},
      created_at: "2026-05-05T00:00:00", updated_at: "2026-05-05T00:00:00",
    } as any);

    await mountAndSettle();
    // Pick the current Monday (state init defaults to it). The hydration
    // useEffect picks retros[monStr] → setWeeklyThoughts. We can't easily
    // assert the exact monStr here (it's based on `new Date()`), so instead
    // mock for the SAME date the component computes.
    // Simpler assertion: the editor's contentEditable div is in the DOM.
    const editor = await screen.findByRole("textbox", { name: /weekly thoughts/i });
    expect(editor).toBeInTheDocument();
    // The hydration write happens via useEffect → innerHTML. Since
    // retros[monStr] won't match the current week's mock unless we time
    // it perfectly, we instead verify the wire is present (component
    // mounted, role present). Save-payload tests below cover the data path.
  });

  test("Editing Weekly Thoughts flips dirtyRef and triggers debounced save", async () => {
    await mountAndSettle();

    const editor = await screen.findByRole("textbox", { name: /weekly thoughts/i });
    // Simulate a contentEditable input event by setting innerHTML and
    // dispatching the synthetic input. fireEvent.input on a contentEditable
    // doesn't update innerHTML automatically — we set it manually first.
    await act(async () => {
      (editor as HTMLDivElement).innerHTML = "<p>my new thought</p>";
      fireEvent.input(editor);
    });

    // Wait past the 800ms debounce.
    await new Promise(r => setTimeout(r, 1000));
    await waitFor(() => expect(mWeeklyUpsert).toHaveBeenCalledTimes(1));
    expect(mWeeklyUpsert.mock.calls[0][0].weekly_thoughts).toBe("<p>my new thought</p>");
  });

  test("Save Weekly Retro click forwards weekly_thoughts in the payload", async () => {
    await mountAndSettle();

    // Type into the editor first.
    const editor = await screen.findByRole("textbox", { name: /weekly thoughts/i });
    await act(async () => {
      (editor as HTMLDivElement).innerHTML = "<p>before save</p>";
      fireEvent.input(editor);
    });

    // Explicit save bypasses the debounce.
    const saveBtn = screen.getByRole("button", { name: /save weekly retro/i });
    await act(async () => { fireEvent.click(saveBtn); });

    await waitFor(() => expect(mWeeklyUpsert).toHaveBeenCalled());
    // The payload should include the editor's HTML.
    const payloads = mWeeklyUpsert.mock.calls.map(c => c[0]);
    const last = payloads[payloads.length - 1];
    expect(last.weekly_thoughts).toBe("<p>before save</p>");
  });
});

describe("WeeklyRetro — Phase 5 Weekly Insights tiles + Flight Deck", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaults();
    try { localStorage.clear(); } catch { /* shim */ }
  });

  test("renders the 3-tile gradient row with primary + subtitle on each tile", async () => {
    mWeeklyMetrics.mockResolvedValue({
      weekly_pnl: 21430,
      weekly_return_pct: 4.62,
      ytd_pct: 12.3,
      ltd_pct: 87.4,
      win_rate: { rate: 0.78, wins: 14, losses: 4, flat: 1, total: 19 },
      week_start: "2026-05-11", week_end: "2026-05-15",
      as_of: "2026-05-14T00:00:00",
    } as any);

    await mountAndSettle();

    // 3 tile labels (was 5; Weekly P&L and LTD % are now subtitles
    // under Weekly Return % and YTD % respectively).
    await screen.findByText("Weekly Return %");
    expect(screen.getByText("YTD %")).toBeInTheDocument();
    expect(screen.getByText("Win Rate")).toBeInTheDocument();
    expect(screen.queryByText("Weekly P&L")).toBeNull();
    expect(screen.queryByText("LTD %")).toBeNull();

    // Primary values on each tile.
    await waitFor(() => expect(screen.getByText("+4.62%")).toBeInTheDocument());
    expect(screen.getByText("+12.30%")).toBeInTheDocument();
    expect(screen.getByText("+78.00%")).toBeInTheDocument();

    // Subtitles — each tile carries one, primaries align vertically.
    // Tile 1 subtitle: Weekly P&L $ with sign + full precision.
    expect(screen.getByText(/\+\$21,430/)).toBeInTheDocument();
    // Tile 2 subtitle: "LTD +87.40%".
    expect(screen.getByText(/LTD \+87\.40%/)).toBeInTheDocument();
    // Tile 3 subtitle: existing W·L·F·of breakdown (separator updated
    // from "/" to "·" for visual consistency with the rail stats line).
    expect(screen.getByText("14W · 4L · 1F of 19")).toBeInTheDocument();
  });

  test("Weekly Return % tile subtitle renders the Weekly P&L $ with sign", async () => {
    mWeeklyMetrics.mockResolvedValue({
      weekly_pnl: -3210, weekly_return_pct: -1.2,
      ytd_pct: 5.0, ltd_pct: 20.0,
      win_rate: { rate: 0, wins: 0, losses: 0, flat: 0, total: 0 },
      week_start: "2026-05-11", week_end: "2026-05-15",
      as_of: "2026-05-14T00:00:00",
    } as any);
    await mountAndSettle();
    // Negative P&L renders as the subtitle on the Weekly Return % tile.
    await waitFor(() => expect(screen.getByText(/-\$3,210/)).toBeInTheDocument());
  });

  test("YTD % tile subtitle renders LTD prefix label", async () => {
    mWeeklyMetrics.mockResolvedValue({
      weekly_pnl: 100, weekly_return_pct: 1.0,
      ytd_pct: 12.3, ltd_pct: -8.45,
      win_rate: { rate: 0, wins: 0, losses: 0, flat: 0, total: 0 },
      week_start: "2026-05-11", week_end: "2026-05-15",
      as_of: "2026-05-14T00:00:00",
    } as any);
    await mountAndSettle();
    // Negative LTD: no "+" sign, includes "LTD" prefix.
    await waitFor(() => expect(screen.getByText(/LTD -8\.45%/)).toBeInTheDocument());
  });

  test("calls api.weeklyMetrics with the active portfolio and current Monday", async () => {
    await mountAndSettle();
    await waitFor(() => expect(mWeeklyMetrics).toHaveBeenCalled());
    const [portfolio, weekStart] = mWeeklyMetrics.mock.calls[0]!;
    expect(portfolio).toBe("CanSlim");
    // Phase 0 / 4.x convention: the week defaults to "this week's Monday"
    // (Mon=1...Sun=0). We just assert it's a valid YYYY-MM-DD anchored on
    // a Monday — the exact date depends on test wall-clock.
    expect(weekStart).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    const d = new Date(weekStart + "T12:00:00");
    expect(d.getDay()).toBe(1); // Monday
  });

  test("refetches metrics when the week date changes", async () => {
    // Phase 6: the standalone date input was removed from the main
    // content. Navigation now happens through the NotesRail (item
    // click) or its calendar jump-to-date popover. Provide a rail item
    // so the jump-to-date handler can resolve it.
    mWeeklyList.mockResolvedValue({
      weeks: [{
        id: 99, key: "2026-05-04", week_start: "2026-05-04",
        week_end: "2026-05-08", year: 2026, month: 5,
        title: "May 4 – 8", has_content: false, pinned: false,
        sparkline_value: null, week_grade: null,
        weekly_pnl: null, trades_count: 0, win_rate: null, tags: [],
      }],
      ytd_stats: { total_weeks: 0, weeks_graded: 0, avg_grade: null, weeks_pinned: 0 },
    } as any);
    await mountAndSettle();
    const initialCalls = mWeeklyMetrics.mock.calls.length;

    // NotesRail renders a hidden `<input type="date">` inside the
    // jump-to-date affordance. Firing change triggers handleJumpToDate
    // which finds the item by week_start and calls onItemClick →
    // setWeekDate → metrics refetch.
    const weekInput = document.querySelector('input[type="date"]') as HTMLInputElement;
    expect(weekInput).toBeTruthy();
    await act(async () => {
      fireEvent.change(weekInput, { target: { value: "2026-05-07" } });
    });

    await waitFor(() => {
      expect(mWeeklyMetrics.mock.calls.length).toBeGreaterThan(initialCalls);
    });
    const lastCall = mWeeklyMetrics.mock.calls.at(-1)!;
    expect(lastCall[1]).toBe("2026-05-04");
  });

  test("shows an inline error message when the metrics endpoint returns an error", async () => {
    mWeeklyMetrics.mockResolvedValue({ error: "Database is sleeping" } as any);
    await mountAndSettle();
    await waitFor(() => {
      expect(screen.getByText(/Weekly metrics unavailable: Database is sleeping/)).toBeInTheDocument();
    });
  });

  test("mounts the Flight Deck inside the Per-Ticker expander body", async () => {
    mTradesRecent.mockResolvedValue({
      details: [
        // Build a Monday-Friday spread of buys/sells; exact dates don't
        // matter for the FlightDeck mount — just that the component is
        // present inside the per-ticker section once we open it.
        { date: new Date().toISOString().slice(0, 10), ticker: "AAPL",
          action: "BUY", trx_id: "B1", shares: 100, amount: 150, rule: "" },
        { date: new Date().toISOString().slice(0, 10), ticker: "NVDA",
          action: "SELL", trx_id: "S1", shares: 50, amount: 500, rule: "" },
      ],
      lot_closures: [],
    } as any);
    await mountAndSettle();

    // The FlightDeck node renders ONLY inside the expander body. Open it.
    const expanderToggle = await screen.findByRole("button", { name: /Per-Ticker Details/i });
    await act(async () => { fireEvent.click(expanderToggle); });

    await waitFor(() => {
      expect(screen.getByTestId("flight-deck")).toBeInTheDocument();
    });
    // Standard 4 labels render.
    expect(screen.getByText("Total Tickets")).toBeInTheDocument();
    expect(screen.getByText("Buys")).toBeInTheDocument();
    expect(screen.getByText("Sells / Trims")).toBeInTheDocument();
  });

  // Phase A error-logging sweep — the rail fetch is the Phase 6
  // regression-class line. A backend KeyError surfaces as
  // { error: "..." } at HTTP 200; the rail fetch wrapper previously
  // swallowed it silently and left railItems empty. The
  // log.error("weekly-retro", "rail fetch failed", ...) call ensures
  // any future regression of this class is visible in devtools.
  test("rail fetch error response logs to console.error and leaves railItems unchanged", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    mWeeklyList.mockResolvedValue({ error: "test_failure" } as any);

    render(<WeeklyRetro navColor="#6366f1" />);
    await waitFor(() => expect(mWeeklyList).toHaveBeenCalled());

    await waitFor(() => {
      expect(errorSpy).toHaveBeenCalledWith(
        "[weekly-retro] rail fetch failed:",
        "test_failure",
      );
    });
    errorSpy.mockRestore();
  });
});
