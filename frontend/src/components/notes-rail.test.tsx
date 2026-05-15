import { render, screen, fireEvent, act, waitFor } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach } from "vitest";

import { NotesRail } from "./notes-rail";
import type { NotesRailItem, NotesRailYtdStats } from "@/lib/api";

// Minimal localStorage shim — jsdom usually has one, but some tests run
// in environments without it.
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

// Stable date helpers for the test fixtures. Use ISO Mondays so the
// component's grouping (by year+month) is deterministic.
function item(opts: Partial<NotesRailItem> & {
  key: string; year: number; month: number;
}): NotesRailItem {
  return {
    id: opts.id ?? 1,
    key: opts.key,
    week_start: opts.key,
    week_end: opts.key,
    year: opts.year,
    month: opts.month,
    title: opts.title ?? `Week of ${opts.key}`,
    has_content: opts.has_content ?? true,
    pinned: opts.pinned ?? false,
    sparkline_value: opts.sparkline_value ?? 1.5,
    week_grade: opts.week_grade ?? "B+",
    weekly_pnl: opts.weekly_pnl ?? null,
    trades_count: opts.trades_count ?? 0,
    tags: opts.tags ?? [],
    reviewed_at: opts.reviewed_at ?? null,
  };
}

const EMPTY_STATS: NotesRailYtdStats = {
  total_weeks: 0, weeks_graded: 0, avg_grade: null, weeks_pinned: 0,
};

beforeEach(() => {
  try { localStorage.clear(); } catch { /* shim */ }
});

describe("NotesRail — Phase 6 left-rail navigator", () => {
  test("renders empty-state copy when items is empty", () => {
    render(<NotesRail entityType="weekly_retro" items={[]} ytdStats={EMPTY_STATS}
                     currentEntityKey={null}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    expect(screen.getByTestId("rail-empty-state")).toHaveTextContent(/no weekly retros yet/i);
  });

  test("renders items grouped by month, current month folder default-open", () => {
    const items = [
      item({ id: 10, key: "2026-05-11", year: 2026, month: 5, title: "May 11 – 15" }),
      item({ id: 11, key: "2026-05-04", year: 2026, month: 5, title: "May 4 – 8" }),
      item({ id: 12, key: "2026-04-27", year: 2026, month: 4, title: "Apr 27 – May 1" }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    // Two month folders (May current, April past — both in current year).
    const folders = screen.getAllByTestId("month-folder");
    expect(folders.length).toBe(2);
    // Current month (May=05) default open.
    const may = folders.find(f => f.dataset.monthKey === "2026-05");
    expect(may?.dataset.open).toBe("true");
    // Items in May visible (in the open folder body).
    expect(screen.getByText("May 11 – 15")).toBeInTheDocument();
  });

  test("renders Pinned section above current year when items have pinned: true", () => {
    const items = [
      item({ id: 10, key: "2026-05-11", year: 2026, month: 5, pinned: true,
             title: "May 11 – 15" }),
      item({ id: 11, key: "2026-05-04", year: 2026, month: 5,
             title: "May 4 – 8" }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    expect(screen.getByTestId("pinned-section")).toBeInTheDocument();
    // The pinned week renders BOTH in the pinned section and in its month.
    const allMayCards = screen.getAllByText("May 11 – 15");
    expect(allMayCards.length).toBeGreaterThanOrEqual(2);
  });

  test("does NOT render Pinned section when no items are pinned", () => {
    const items = [item({ id: 10, key: "2026-05-11", year: 2026, month: 5 })];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    expect(screen.queryByTestId("pinned-section")).toBeNull();
  });

  test("active state marks the item matching currentEntityKey via aria-current", () => {
    const items = [
      item({ id: 10, key: "2026-05-11", year: 2026, month: 5 }),
      item({ id: 11, key: "2026-05-04", year: 2026, month: 5 }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey="2026-05-11"
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    const activeRow = screen.getAllByTestId("rail-item")
      .find(el => el.dataset.key === "2026-05-11" && el.dataset.active === "true");
    expect(activeRow).toBeTruthy();
    // ARIA: the active button is aria-current=true.
    expect(activeRow!.querySelector("[aria-current='true']")).toBeTruthy();
  });

  test("clicking an item fires onItemClick with the right item", () => {
    const onItemClick = vi.fn();
    const items = [
      item({ id: 10, key: "2026-05-11", year: 2026, month: 5, title: "May 11 – 15" }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={null}
                     onItemClick={onItemClick} onPinToggle={vi.fn()} />);
    const btn = screen.getByText("May 11 – 15").closest("button")!;
    fireEvent.click(btn);
    expect(onItemClick).toHaveBeenCalledTimes(1);
    expect(onItemClick.mock.calls[0][0].key).toBe("2026-05-11");
  });

  test("clicking the pin star fires onPinToggle with the entity id + current pinned state", async () => {
    const onPinToggle = vi.fn().mockResolvedValue(undefined);
    const items = [
      item({ id: 10, key: "2026-05-11", year: 2026, month: 5, pinned: false }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={onPinToggle} />);
    // Trigger hover to surface the pin button. Pin button is always
    // present for pinned items; for unpinned we simulate hover.
    const row = screen.getAllByTestId("rail-item")[0];
    fireEvent.mouseEnter(row);
    const pinBtn = await screen.findByTestId("rail-pin-btn");
    await act(async () => { fireEvent.click(pinBtn); });
    expect(onPinToggle).toHaveBeenCalledWith(10, false);
  });

  test("month-folder collapse state persists in localStorage", () => {
    const items = [
      item({ id: 10, key: "2026-05-11", year: 2026, month: 5 }),
      item({ id: 11, key: "2026-04-27", year: 2026, month: 4 }),
    ];
    const { unmount } = render(
      <NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                 currentEntityKey={items[0].key}
                 onItemClick={vi.fn()} onPinToggle={vi.fn()} />
    );
    // Open the April folder (which defaults closed since it's not current month).
    const aprFolder = screen.getAllByTestId("month-folder")
      .find(f => f.dataset.monthKey === "2026-04")!;
    expect(aprFolder.dataset.open).toBe("false");
    const aprToggle = aprFolder.querySelector("button")!;
    fireEvent.click(aprToggle);
    expect(aprFolder.dataset.open).toBe("true");

    // Re-mount → April folder should still be open.
    unmount();
    render(
      <NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                 currentEntityKey={items[0].key}
                 onItemClick={vi.fn()} onPinToggle={vi.fn()} />
    );
    const aprFolder2 = screen.getAllByTestId("month-folder")
      .find(f => f.dataset.monthKey === "2026-04")!;
    expect(aprFolder2.dataset.open).toBe("true");
  });

  test("search input filters items by case-insensitive title substring", async () => {
    const items = [
      item({ id: 10, key: "2026-05-11", year: 2026, month: 5, title: "May 11 – 15" }),
      item({ id: 11, key: "2026-05-04", year: 2026, month: 5, title: "May 4 – 8" }),
      item({ id: 12, key: "2026-04-27", year: 2026, month: 4, title: "Apr 27 – May 1" }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    const search = screen.getByTestId("rail-search") as HTMLInputElement;
    await act(async () => { fireEvent.change(search, { target: { value: "apr" } }); });
    // Only the April title should be rendered.
    expect(screen.queryByText("May 11 – 15")).toBeNull();
    expect(screen.queryByText("May 4 – 8")).toBeNull();
    expect(screen.getByText("Apr 27 – May 1")).toBeInTheDocument();
  });

  test("YTD stats footer renders the supplied stats block", () => {
    const stats: NotesRailYtdStats = {
      total_weeks: 17, weeks_graded: 12, avg_grade: "B+", weeks_pinned: 2,
    };
    render(<NotesRail entityType="weekly_retro" items={[]} ytdStats={stats}
                     currentEntityKey={null}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    expect(screen.getByText("B+")).toBeInTheDocument();
    expect(screen.getByText(/12\/17 graded/)).toBeInTheDocument();
    expect(screen.getByText(/2 pinned/)).toBeInTheDocument();
  });

  test("collapse button shrinks the rail to a 56px sliver and persists in localStorage", async () => {
    const items = [item({ id: 10, key: "2026-05-11", year: 2026, month: 5 })];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    const collapseBtn = screen.getByTestId("rail-collapse-btn");
    await act(async () => { fireEvent.click(collapseBtn); });
    expect(screen.getByTestId("notes-rail-collapsed")).toBeInTheDocument();
    expect(localStorage.getItem("mo-notes-rail-weekly_retro-collapsed")).toBe("true");
  });

  test("month sparkline renders one rect per item in the month", () => {
    const items = [
      item({ id: 10, key: "2026-05-11", year: 2026, month: 5, sparkline_value: 4.62 }),
      item({ id: 11, key: "2026-05-04", year: 2026, month: 5, sparkline_value: -2.1 }),
      item({ id: 12, key: "2026-04-27", year: 2026, month: 5, sparkline_value: 1.0 }),
    ];
    const { container } = render(
      <NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                 currentEntityKey={items[0].key}
                 onItemClick={vi.fn()} onPinToggle={vi.fn()} />
    );
    // Scope to the sparkline SVG (56x18) — the calendar icon and other
    // decorative SVGs in the rail header also contain <rect> nodes, so
    // a naive container-level query would over-count.
    const sparklineSvg = Array.from(container.querySelectorAll("svg"))
      .find(s => s.getAttribute("width") === "56" && s.getAttribute("height") === "18");
    expect(sparklineSvg).toBeTruthy();
    const rects = sparklineSvg!.querySelectorAll("rect");
    expect(rects.length).toBe(3);
  });

  test("optimistic pin flips UI immediately, rolls back on rejection", async () => {
    const onPinToggle = vi.fn().mockRejectedValue(new Error("server says no"));
    const items = [
      item({ id: 10, key: "2026-05-11", year: 2026, month: 5, pinned: false }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={onPinToggle} />);
    const row = screen.getAllByTestId("rail-item")[0];
    fireEvent.mouseEnter(row);
    const pinBtn = await screen.findByTestId("rail-pin-btn");
    await act(async () => { fireEvent.click(pinBtn); });
    // Eventually the optimistic flip rolls back since onPinToggle rejected.
    await waitFor(() => {
      const row2 = screen.getAllByTestId("rail-item")[0];
      expect(row2.dataset.pinned).toBe("false");
    });
  });

  test("ArrowDown moves focus to the next item (focus-respecting)", () => {
    const onItemClick = vi.fn();
    const items = [
      item({ id: 10, key: "2026-05-11", year: 2026, month: 5 }),
      item({ id: 11, key: "2026-05-04", year: 2026, month: 5 }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={onItemClick} onPinToggle={vi.fn()} />);
    // Body-level keydown; not inside the search input.
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowDown" }));
    });
    // Press Enter to fire onItemClick with the focused (next) item.
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
    });
    expect(onItemClick).toHaveBeenCalled();
    expect(onItemClick.mock.calls[0][0].key).toBe("2026-05-04");
  });

  test("header subtitle includes years span when items cross multiple years", () => {
    const items = [
      item({ id: 10, key: "2026-05-11", year: 2026, month: 5 }),
      item({ id: 11, key: "2025-12-29", year: 2025, month: 12 }),
      item({ id: 12, key: "2024-09-02", year: 2024, month: 9 }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    // 2026 - 2024 + 1 = 3 years
    expect(screen.getByText(/3 years/)).toBeInTheDocument();
    expect(screen.getByText(/3 weeks/)).toBeInTheDocument();
  });

  test("header subtitle renders '1 year' (singular) when items span one calendar year", () => {
    const items = [
      item({ id: 10, key: "2026-05-11", year: 2026, month: 5 }),
      item({ id: 11, key: "2026-01-06", year: 2026, month: 1 }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    expect(screen.getByText(/1 year(?!s)/)).toBeInTheDocument();
  });

  test("calendar jump-to-date button opens picker; selecting a date fires onItemClick with the right week", async () => {
    const onItemClick = vi.fn();
    const items = [
      item({ id: 10, key: "2026-05-11", year: 2026, month: 5 }),
      item({ id: 11, key: "2026-05-04", year: 2026, month: 5 }),
      item({ id: 12, key: "2026-04-27", year: 2026, month: 4 }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={onItemClick} onPinToggle={vi.fn()} />);
    // Verify the button exists (clicking native picker doesn't work in jsdom).
    expect(screen.getByTestId("rail-jump-date-btn")).toBeInTheDocument();
    // Simulate the date input's change event directly — that's the path
    // a real user hits after picking from the native popover.
    const dateInput = screen.getByTestId("rail-jump-date-input") as HTMLInputElement;
    await act(async () => {
      fireEvent.change(dateInput, { target: { value: "2026-05-07" } }); // Thursday of week 2026-05-04
    });
    expect(onItemClick).toHaveBeenCalledTimes(1);
    expect(onItemClick.mock.calls[0][0].key).toBe("2026-05-04");
  });

  test("calendar jump-to-date picks the nearest week when the exact week isn't loaded", async () => {
    const onItemClick = vi.fn();
    const items = [
      item({ id: 10, key: "2026-05-11", year: 2026, month: 5 }),
      item({ id: 11, key: "2026-05-04", year: 2026, month: 5 }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={onItemClick} onPinToggle={vi.fn()} />);
    const dateInput = screen.getByTestId("rail-jump-date-input") as HTMLInputElement;
    // 2025-01-01 — far before any loaded week. Nearest is 2026-05-04.
    await act(async () => {
      fireEvent.change(dateInput, { target: { value: "2025-01-01" } });
    });
    expect(onItemClick).toHaveBeenCalledTimes(1);
    expect(onItemClick.mock.calls[0][0].key).toBe("2026-05-04");
  });

  test("RailItem renders unified stats line: return% · $P&L · trades (no win_rate)", () => {
    const items = [item({
      id: 10, key: "2026-05-11", year: 2026, month: 5,
      sparkline_value: 4.62,
      weekly_pnl: 16700, trades_count: 14,
      title: "May 11 – 15",
    })];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    // Three segments present. Each renders twice in this single-item
    // setup — once in the MonthFolder aggregate header and once in the
    // RailItem row — so use getAllByText.
    expect(screen.getAllByText(/\+4\.62%/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/\+\$16\.7k/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("14T").length).toBeGreaterThanOrEqual(1);
    // Win-rate suffix is GONE from the row format.
    expect(screen.queryByText("71%W")).toBeNull();
  });

  test("current week (sparkline_value but null pnl + 0 trades) renders the SAME line shape as past weeks — % visible", () => {
    // Regression fence for the bug where current week showed only "+11.99%"
    // while past weeks showed "$P&L · NT". The unified line always leads
    // with %, then appends pnl/trades segments only when present.
    const items = [
      item({
        id: 10, key: "2026-05-11", year: 2026, month: 5,
        sparkline_value: 11.99, weekly_pnl: null, trades_count: 0,
        title: "May 11 – 15",
      }),
      item({
        id: 11, key: "2026-05-04", year: 2026, month: 5,
        sparkline_value: -5.2, weekly_pnl: -11200, trades_count: 12,
        title: "May 4 – 8",
      }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    // Both weeks render their % — current week's was previously hidden
    // behind the "pnl != null || trades > 0" gate.
    expect(screen.getAllByText(/\+11\.99%/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/-5\.20%/).length).toBeGreaterThanOrEqual(1);
    // Past week's pnl + trades still render alongside its %. They may
    // also surface in the MonthFolder aggregate header above the row.
    expect(screen.getAllByText(/-\$11\.2k/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("12T").length).toBeGreaterThanOrEqual(1);
  });

  test("RailItem renders tag chips when tags are present", () => {
    const items = [item({
      id: 10, key: "2026-05-11", year: 2026, month: 5,
      tags: [
        { name: "FTD", color: "emerald" },
        { name: "Breakout", color: "sky" },
      ],
      title: "May 11 – 15",
    })];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    expect(screen.getByText("FTD")).toBeInTheDocument();
    expect(screen.getByText("Breakout")).toBeInTheDocument();
  });

  test("RailItem caps tag chips at 3 and shows '+N' overflow", () => {
    const items = [item({
      id: 10, key: "2026-05-11", year: 2026, month: 5,
      tags: [
        { name: "FTD", color: "emerald" },
        { name: "Breakout", color: "sky" },
        { name: "Choppy", color: "amber" },
        { name: "Drawdown", color: "rose" },
        { name: "FOMC", color: "violet" },
      ],
    })];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    expect(screen.getByText("FTD")).toBeInTheDocument();
    expect(screen.getByText("Breakout")).toBeInTheDocument();
    expect(screen.getByText("Choppy")).toBeInTheDocument();
    expect(screen.queryByText("Drawdown")).toBeNull();
    expect(screen.getByText("+2")).toBeInTheDocument();
  });

  test("MonthFolder header aggregates $P&L + trades across child weeks (unified format)", () => {
    const items = [
      item({
        id: 10, key: "2026-05-11", year: 2026, month: 5,
        sparkline_value: 4.62, weekly_pnl: 16700, trades_count: 14,
      }),
      item({
        id: 11, key: "2026-05-04", year: 2026, month: 5,
        sparkline_value: -2.1, weekly_pnl: -4200, trades_count: 8,
      }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    // Header sum: 16700 + (-4200) = 12500 → +$12.5k. Trades: 14 + 8 = 22T.
    expect(screen.getByText(/\+\$12\.5k/)).toBeInTheDocument();
    expect(screen.getByText("22T")).toBeInTheDocument();
    // Old "Nw · NW" format must be gone from the month header.
    expect(screen.queryByText(/^2w$/)).toBeNull();
    expect(screen.queryByText(/^1W$/)).toBeNull();
  });

  test("MonthFolder skips $P&L when all child weeks have null pnl (no aggregate to display)", () => {
    const items = [
      item({
        id: 10, key: "2026-05-11", year: 2026, month: 5,
        sparkline_value: 4.62, weekly_pnl: null, trades_count: 0,
      }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    // Only the % shows in the header. No $0 stub, no 0T stub.
    expect(screen.queryByText(/^\+\$0/)).toBeNull();
    expect(screen.queryByText(/T$/)).toBeNull();
  });

  test("YearFolder header aggregates $P&L + trades across all child weeks (unified format) and retains MO label", async () => {
    // Past-year months wrap in a YearFolder. The component derives the
    // "current" year from items[0], so the fixture is ordered
    // newest-first (matching the real backend response shape).
    const items = [
      // Current-year row so the YearFolder below wraps the past year.
      item({
        id: 22, key: "2026-05-11", year: 2026, month: 5,
        sparkline_value: 4.62, weekly_pnl: 16700, trades_count: 14,
      }),
      item({
        id: 20, key: "2025-06-02", year: 2025, month: 6,
        sparkline_value: 3.0, weekly_pnl: 50000, trades_count: 100,
      }),
      item({
        id: 21, key: "2025-05-26", year: 2025, month: 5,
        sparkline_value: 1.5, weekly_pnl: 44000, trades_count: 134,
      }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    const yearHeader = screen.getByTestId("year-folder");
    // Aggregate: 50000 + 44000 = 94000 → "+$94.0k" (compact one-decimal).
    // Trades: 100 + 134 = 234T.
    expect(yearHeader.textContent).toContain("+$94.0k");
    expect(yearHeader.textContent).toContain("234T");
    // "MO" label intact ("2 mo" lowercased in the design).
    expect(yearHeader.textContent?.toLowerCase()).toMatch(/2\s*mo/);
    // Old single "Nw" suffix is gone.
    expect(screen.queryByText(/^2w$/)).toBeNull();
  });

  test("YearFolder skips $P&L when all child weeks lack pnl, keeps % and MO label", () => {
    // Newest-first ordering so the 2026 row sets the current year.
    const items = [
      item({
        id: 22, key: "2026-05-11", year: 2026, month: 5,
        sparkline_value: 4.62, weekly_pnl: 16700, trades_count: 14,
      }),
      item({
        id: 20, key: "2025-06-02", year: 2025, month: 6,
        sparkline_value: 3.0, weekly_pnl: null, trades_count: 0,
      }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    const yearHeader = screen.getByTestId("year-folder");
    expect(yearHeader.textContent?.toLowerCase()).toMatch(/1\s*mo/);
    // No $ segment in the year header when no pnl exists in this year.
    // (Open the year folder body is also untouched — only header.)
    expect(yearHeader.firstElementChild?.textContent).not.toMatch(/\$/);
  });

  // ────────────────────────────────────────────────────────────────────
  // Phase 6.5 — tag filter bar
  // ────────────────────────────────────────────────────────────────────

  test("filter bar hidden when no item carries a tag", () => {
    const items = [
      item({ id: 10, key: "2026-05-11", year: 2026, month: 5 }),
      item({ id: 11, key: "2026-05-04", year: 2026, month: 5 }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    expect(screen.queryByTestId("rail-filter-bar")).toBeNull();
  });

  test("filter bar visible once at least one item carries a tag", () => {
    const items = [
      item({ id: 10, key: "2026-05-11", year: 2026, month: 5,
             tags: [{ name: "FTD", color: "emerald" }] }),
      item({ id: 11, key: "2026-05-04", year: 2026, month: 5 }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    expect(screen.getByTestId("rail-filter-bar")).toBeInTheDocument();
    // Resting copy: "All weeks" when no filter active.
    expect(screen.getByText("All weeks")).toBeInTheDocument();
  });

  test("clicking the Filter trigger opens the popover; tags shown sorted by count desc, name asc", () => {
    // Counts: FTD=3, Breakout=2, Choppy=2, Drawdown=1. After count-desc
    // sort, Breakout precedes Choppy because of the name-asc tiebreaker.
    const items = [
      item({ id: 1, key: "2026-05-11", year: 2026, month: 5,
             tags: [{ name: "FTD", color: "emerald" }, { name: "Breakout", color: "sky" }] }),
      item({ id: 2, key: "2026-05-04", year: 2026, month: 5,
             tags: [{ name: "FTD", color: "emerald" }, { name: "Choppy", color: "amber" }] }),
      item({ id: 3, key: "2026-04-27", year: 2026, month: 4,
             tags: [{ name: "FTD", color: "emerald" }, { name: "Breakout", color: "sky" }] }),
      item({ id: 4, key: "2026-04-20", year: 2026, month: 4,
             tags: [{ name: "Choppy", color: "amber" }, { name: "Drawdown", color: "rose" }] }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    fireEvent.click(screen.getByTestId("rail-filter-trigger"));
    expect(screen.getByTestId("rail-filter-popover")).toBeInTheDocument();
    const options = screen.getAllByTestId("rail-filter-option");
    expect(options.map(o => o.dataset.tagName)).toEqual([
      "FTD", "Breakout", "Choppy", "Drawdown",
    ]);
  });

  test("selecting a tag in the popover renders it as an ActiveTagChip", () => {
    const items = [
      item({ id: 1, key: "2026-05-11", year: 2026, month: 5,
             tags: [{ name: "FTD", color: "emerald" }] }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    fireEvent.click(screen.getByTestId("rail-filter-trigger"));
    const ftdOption = screen.getAllByTestId("rail-filter-option")
      .find(el => el.dataset.tagName === "FTD")!;
    fireEvent.click(ftdOption);
    // Chip is the dark-bg pill in the bar (data-testid rail-filter-chip).
    const chip = screen.getByTestId("rail-filter-chip");
    expect(chip.dataset.tagName).toBe("FTD");
    // ActiveTagChip uses the tone-darkened palette dot color as the bg —
    // emerald → #08a86b. White text confirms it's not the row-chip light
    // body. RGB parsing tolerates jsdom's style serialization differences.
    const bg = chip.style.background;
    expect(bg.toLowerCase()).toContain("rgb(8, 168, 107)"); // emerald dot
  });

  test("× on ActiveTagChip removes that filter", () => {
    const items = [
      item({ id: 1, key: "2026-05-11", year: 2026, month: 5,
             tags: [{ name: "FTD", color: "emerald" }] }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    fireEvent.click(screen.getByTestId("rail-filter-trigger"));
    fireEvent.click(screen.getAllByTestId("rail-filter-option")
      .find(el => el.dataset.tagName === "FTD")!);
    expect(screen.getByTestId("rail-filter-chip")).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("Remove FTD filter"));
    expect(screen.queryByTestId("rail-filter-chip")).toBeNull();
    // "All weeks" returns when filters drain.
    expect(screen.getByText("All weeks")).toBeInTheDocument();
  });

  test("multi-select: two selected tags use OR semantics — weeks matching either show", () => {
    const items = [
      item({ id: 1, key: "2026-05-11", year: 2026, month: 5, title: "May 11 – 15",
             tags: [{ name: "FTD", color: "emerald" }] }),
      item({ id: 2, key: "2026-05-04", year: 2026, month: 5, title: "May 4 – 8",
             tags: [{ name: "Breakout", color: "sky" }] }),
      item({ id: 3, key: "2026-04-27", year: 2026, month: 4, title: "Apr 27 – May 1",
             tags: [{ name: "Choppy", color: "amber" }] }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    fireEvent.click(screen.getByTestId("rail-filter-trigger"));
    const opts = screen.getAllByTestId("rail-filter-option");
    fireEvent.click(opts.find(el => el.dataset.tagName === "FTD")!);
    fireEvent.click(opts.find(el => el.dataset.tagName === "Breakout")!);
    // Both FTD-tagged and Breakout-tagged weeks visible; Choppy-only hidden.
    expect(screen.getByText("May 11 – 15")).toBeInTheDocument();
    expect(screen.getByText("May 4 – 8")).toBeInTheDocument();
    expect(screen.queryByText("Apr 27 – May 1")).toBeNull();
  });

  test("search + filter combine with AND between (search narrows, filter narrows further)", async () => {
    const items = [
      item({ id: 1, key: "2026-05-11", year: 2026, month: 5, title: "May 11 – 15",
             tags: [{ name: "FTD", color: "emerald" }] }),
      item({ id: 2, key: "2026-05-04", year: 2026, month: 5, title: "May 4 – 8",
             tags: [{ name: "Breakout", color: "sky" }] }),
      item({ id: 3, key: "2026-04-27", year: 2026, month: 4, title: "Apr 27 – May 1",
             tags: [{ name: "FTD", color: "emerald" }] }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    // Filter on FTD first.
    fireEvent.click(screen.getByTestId("rail-filter-trigger"));
    fireEvent.click(screen.getAllByTestId("rail-filter-option")
      .find(el => el.dataset.tagName === "FTD")!);
    // Now type "Apr" in search — only the April FTD week should pass both.
    const search = screen.getByTestId("rail-search") as HTMLInputElement;
    await act(async () => { fireEvent.change(search, { target: { value: "Apr" } }); });
    expect(screen.getByText("Apr 27 – May 1")).toBeInTheDocument();
    expect(screen.queryByText("May 11 – 15")).toBeNull();
    expect(screen.queryByText("May 4 – 8")).toBeNull();
  });

  test("month folders auto-open when filters are active so matches in collapsed folders surface", () => {
    // April is past-month (not current) → default-closed without filter.
    const items = [
      item({ id: 1, key: "2026-05-11", year: 2026, month: 5, title: "May 11 – 15",
             tags: [{ name: "Choppy", color: "amber" }] }),
      item({ id: 2, key: "2026-04-27", year: 2026, month: 4, title: "Apr 27 – May 1",
             tags: [{ name: "FTD", color: "emerald" }] }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    // Before filter: April folder closed → "Apr 27 – May 1" not in DOM
    // (folder body unmounts when closed via display:none isn't used;
    // body is conditionally rendered).
    const aprBefore = screen.getAllByTestId("month-folder")
      .find(f => f.dataset.monthKey === "2026-04")!;
    expect(aprBefore.dataset.open).toBe("false");
    // Apply FTD filter.
    fireEvent.click(screen.getByTestId("rail-filter-trigger"));
    fireEvent.click(screen.getAllByTestId("rail-filter-option")
      .find(el => el.dataset.tagName === "FTD")!);
    // April folder forced open; FTD-tagged April week now visible.
    const aprAfter = screen.getAllByTestId("month-folder")
      .find(f => f.dataset.monthKey === "2026-04")!;
    expect(aprAfter.dataset.open).toBe("true");
    expect(screen.getByText("Apr 27 – May 1")).toBeInTheDocument();
  });

  test("empty-filter-result state renders copy + Clear filters button; clicking it clears filters", () => {
    // FTD exists in availableTags, but Drawdown doesn't — so selecting
    // a tag that matches no week yields zero. Use a contrived setup:
    // pick FTD then mutate? Easier: tag both items with "Choppy" and
    // filter on "FTD" via the popover ... but FTD wouldn't appear if no
    // item carries it. Reverse approach: two weeks, one tagged "FTD",
    // search "zzz" to force zero, with filter active.
    const items = [
      item({ id: 1, key: "2026-05-11", year: 2026, month: 5, title: "May 11 – 15",
             tags: [{ name: "FTD", color: "emerald" }] }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    fireEvent.click(screen.getByTestId("rail-filter-trigger"));
    fireEvent.click(screen.getAllByTestId("rail-filter-option")
      .find(el => el.dataset.tagName === "FTD")!);
    // Filter applied, FTD week matches. Now force the combined filter
    // to yield zero by adding a no-match search query.
    const search = screen.getByTestId("rail-search") as HTMLInputElement;
    fireEvent.change(search, { target: { value: "zzz" } });
    const empty = screen.getByTestId("rail-no-search-results");
    expect(empty).toHaveTextContent(/No weeks match the current filters/i);
    // Clearing filters via the button removes the filter; the search
    // query is still "zzz" so the row stays hidden, but the empty-state
    // copy reverts to "No matches."
    fireEvent.click(screen.getByTestId("rail-filter-clear-empty"));
    expect(screen.queryByTestId("rail-filter-chip")).toBeNull();
    expect(screen.getByTestId("rail-no-search-results")).toHaveTextContent(/No matches\./i);
  });

  test("Esc closes the filter popover", () => {
    const items = [
      item({ id: 1, key: "2026-05-11", year: 2026, month: 5,
             tags: [{ name: "FTD", color: "emerald" }] }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    fireEvent.click(screen.getByTestId("rail-filter-trigger"));
    expect(screen.getByTestId("rail-filter-popover")).toBeInTheDocument();
    act(() => {
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    });
    expect(screen.queryByTestId("rail-filter-popover")).toBeNull();
  });

  test("click-outside closes the filter popover", () => {
    const items = [
      item({ id: 1, key: "2026-05-11", year: 2026, month: 5,
             tags: [{ name: "FTD", color: "emerald" }] }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    fireEvent.click(screen.getByTestId("rail-filter-trigger"));
    expect(screen.getByTestId("rail-filter-popover")).toBeInTheDocument();
    // mousedown on the document body (outside both anchor and popover).
    act(() => {
      document.body.dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
    });
    expect(screen.queryByTestId("rail-filter-popover")).toBeNull();
  });

  test("popover tag count reflects actual usage in items[]", () => {
    const items = [
      item({ id: 1, key: "2026-05-11", year: 2026, month: 5,
             tags: [{ name: "FTD", color: "emerald" }] }),
      item({ id: 2, key: "2026-05-04", year: 2026, month: 5,
             tags: [{ name: "FTD", color: "emerald" }, { name: "Choppy", color: "amber" }] }),
      item({ id: 3, key: "2026-04-27", year: 2026, month: 4,
             tags: [{ name: "Choppy", color: "amber" }] }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    fireEvent.click(screen.getByTestId("rail-filter-trigger"));
    const opts = screen.getAllByTestId("rail-filter-option");
    const ftdRow = opts.find(o => o.dataset.tagName === "FTD")!;
    const choppyRow = opts.find(o => o.dataset.tagName === "Choppy")!;
    // Last child is the count span.
    expect(ftdRow.textContent).toMatch(/2$/);
    expect(choppyRow.textContent).toMatch(/2$/);
    // Sort order with equal counts → alphabetical: Choppy before FTD.
    expect(opts.map(o => o.dataset.tagName)).toEqual(["Choppy", "FTD"]);
  });

  // ────────────────────────────────────────────────────────────────────
  // Phase 4.6 — tri-state graded dot
  // ────────────────────────────────────────────────────────────────────

  test("dot state: empty when has_content=false (synthetic row)", () => {
    const items = [item({
      id: 10, key: "2026-05-11", year: 2026, month: 5,
      has_content: false, reviewed_at: null,
    })];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    const dot = document.querySelector('[data-dot-state="empty"]');
    expect(dot).toBeTruthy();
  });

  test("dot state: draft when content present but reviewed_at is null", () => {
    const items = [item({
      id: 10, key: "2026-05-11", year: 2026, month: 5,
      has_content: true, reviewed_at: null,
    })];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    const dot = document.querySelector('[data-dot-state="draft"]') as HTMLElement;
    expect(dot).toBeTruthy();
    // Amber fill — matches design's "needs review" accent.
    expect(dot.style.background).toBe("rgb(245, 159, 0)");
  });

  test("dot state: reviewed when reviewed_at is non-null (green, overrides has_content)", () => {
    const items = [item({
      id: 10, key: "2026-05-11", year: 2026, month: 5,
      has_content: true, reviewed_at: "2026-05-14T10:00:00Z",
    })];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={vi.fn()} onPinToggle={vi.fn()} />);
    const dot = document.querySelector('[data-dot-state="reviewed"]') as HTMLElement;
    expect(dot).toBeTruthy();
    // Green fill #08a86b — matches "moves the dot to green" copy.
    expect(dot.style.background).toBe("rgb(8, 168, 107)");
  });

  test("search input doesn't intercept ArrowDown — focus stays in search bar", () => {
    const onItemClick = vi.fn();
    const items = [
      item({ id: 10, key: "2026-05-11", year: 2026, month: 5 }),
      item({ id: 11, key: "2026-05-04", year: 2026, month: 5 }),
    ];
    render(<NotesRail entityType="weekly_retro" items={items} ytdStats={EMPTY_STATS}
                     currentEntityKey={items[0].key}
                     onItemClick={onItemClick} onPinToggle={vi.fn()} />);
    const search = screen.getByTestId("rail-search") as HTMLInputElement;
    search.focus();
    // Dispatch ArrowDown WITH the input as the event target.
    act(() => {
      const ev = new KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true });
      search.dispatchEvent(ev);
    });
    // Enter from the input should NOT trigger onItemClick because the
    // keyboard nav handler bails when tag === input.
    act(() => {
      const ev = new KeyboardEvent("keydown", { key: "Enter", bubbles: true });
      search.dispatchEvent(ev);
    });
    expect(onItemClick).not.toHaveBeenCalled();
  });
});
