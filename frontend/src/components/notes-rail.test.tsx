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
    // SVG is inside the current-month folder header.
    const rects = container.querySelectorAll("svg rect");
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
