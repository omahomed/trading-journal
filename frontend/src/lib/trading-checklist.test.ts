import { describe, it, expect } from "vitest";
import type { RoutineItem } from "@/lib/api";
import {
  computeReorderSwap,
  countTodayDue,
  formatGroupLabel,
  formatShortDate,
  groupRoutineItems,
  itemStatusChip,
  FREQUENCY_ORDER,
  SLOT_ORDER,
} from "./trading-checklist";

// Base fixture: a task item with all optional fields populated. Tests
// spread this and override the fields they care about.
function makeItem(overrides: Partial<RoutineItem> = {}): RoutineItem {
  return {
    id: 1,
    name: "Test item",
    frequency: "daily",
    slot: "after_close",
    item_type: "task",
    link: null,
    is_system: false,
    sort_order: 10,
    last_run: null,
    last_run_date: null,
    overdue_days: null,
    ticked_today: false,
    todays_log_id: null,
    ...overrides,
  };
}

describe("formatGroupLabel", () => {
  it("frequency + slot → 'Daily · after close'", () => {
    expect(formatGroupLabel("daily", "after_close")).toBe("Daily · after close");
  });

  it("null slot → frequency only", () => {
    expect(formatGroupLabel("quarterly", null)).toBe("Quarterly");
    expect(formatGroupLabel("monthly", null)).toBe("Monthly");
  });

  it("all slot variants lowercase in label", () => {
    expect(formatGroupLabel("daily", "premarket")).toBe("Daily · pre-market");
    expect(formatGroupLabel("daily", "intraday")).toBe("Daily · intraday");
    expect(formatGroupLabel("daily", "end_of_shift")).toBe("Daily · end of shift");
    expect(formatGroupLabel("weekly", "weekend")).toBe("Weekly · weekend");
  });
});

describe("groupRoutineItems", () => {
  it("buckets by (frequency, slot) and orders by canon", () => {
    const items = [
      makeItem({ id: 1, frequency: "weekly",    slot: "weekend"     }),
      makeItem({ id: 2, frequency: "daily",     slot: "after_close" }),
      makeItem({ id: 3, frequency: "daily",     slot: "premarket"   }),
      makeItem({ id: 4, frequency: "quarterly", slot: null          }),
      makeItem({ id: 5, frequency: "daily",     slot: "after_close" }),
    ];
    const groups = groupRoutineItems(items);
    expect(groups.map(g => `${g.frequency}|${g.slot ?? ""}`)).toEqual([
      "daily|premarket",
      "daily|after_close",
      "weekly|weekend",
      "quarterly|",
    ]);
    const afterClose = groups.find(g => g.slot === "after_close");
    expect(afterClose?.items.map(i => i.id)).toEqual([2, 5]);
  });

  it("null slot sorts after all named slots within same frequency", () => {
    const items = [
      makeItem({ id: 1, frequency: "monthly", slot: null       }),
      makeItem({ id: 2, frequency: "monthly", slot: "weekend"  }),
    ];
    const groups = groupRoutineItems(items);
    expect(groups.map(g => g.slot)).toEqual(["weekend", null]);
  });

  it("empty input → empty output", () => {
    expect(groupRoutineItems([])).toEqual([]);
  });

  it("canonical orderings exposed for consumers", () => {
    expect(FREQUENCY_ORDER).toEqual(["daily", "weekly", "monthly", "quarterly"]);
    expect(SLOT_ORDER).toEqual(["premarket", "intraday", "end_of_shift", "after_close", "weekend"]);
  });
});

describe("itemStatusChip", () => {
  it("counter → 'log when it happens' regardless of tick state", () => {
    const notTicked = itemStatusChip(makeItem({ item_type: "counter" }));
    const tickedToday = itemStatusChip(makeItem({
      item_type: "counter", ticked_today: true, todays_log_id: 5,
    }));
    expect(notTicked).toEqual({ kind: "counter", text: "log when it happens" });
    expect(tickedToday).toEqual({ kind: "counter", text: "log when it happens" });
  });

  it("overdue_days > 0 → danger chip with days", () => {
    const chip = itemStatusChip(makeItem({ overdue_days: 3, last_run_date: "2026-07-20" }));
    expect(chip.kind).toBe("overdue");
    if (chip.kind === "overdue") {
      expect(chip.days).toBe(3);
      expect(chip.text).toBe("3 days");
    }
  });

  it("overdue_days === 1 → '1 day' singular", () => {
    const chip = itemStatusChip(makeItem({ overdue_days: 1 }));
    expect(chip.kind).toBe("overdue");
    if (chip.kind === "overdue") expect(chip.text).toBe("1 day");
  });

  it("ticked_today wins over last_run_date when not overdue", () => {
    const chip = itemStatusChip(makeItem({
      ticked_today: true, todays_log_id: 5, last_run_date: "2026-07-25",
    }));
    expect(chip.kind).toBe("today");
    expect(chip.text).toBe("ticked today");
  });

  it("last_run_date but not ticked today → 'Completed on MM.DD.YY'", () => {
    const chip = itemStatusChip(makeItem({ last_run_date: "2026-07-19" }));
    expect(chip.kind).toBe("last_run");
    expect(chip.text).toBe("Completed on 07.19.26");
  });

  it("never ticked → em-dash", () => {
    const chip = itemStatusChip(makeItem({}));
    expect(chip.kind).toBe("never");
    expect(chip.text).toBe("—");
  });

  it("overdue beats ticked_today (defensive — shouldn't happen server-side)", () => {
    // If backend somehow returns both overdue and ticked_today, the row
    // is telling us two things; overdue is the more actionable state.
    const chip = itemStatusChip(makeItem({
      overdue_days: 5, ticked_today: true, todays_log_id: 5,
    }));
    expect(chip.kind).toBe("overdue");
  });
});

describe("formatShortDate", () => {
  it("YYYY-MM-DD → 'MM.DD.YY'", () => {
    expect(formatShortDate("2026-07-25")).toBe("07.25.26");
    expect(formatShortDate("2026-01-01")).toBe("01.01.26");
    expect(formatShortDate("2026-12-31")).toBe("12.31.26");
  });

  it("no client-TZ shift at boundaries", () => {
    // A naive Date("2026-01-01") would produce different D depending on
    // client TZ. Splitting the string sidesteps that entirely.
    expect(formatShortDate("2026-01-01")).toBe("01.01.26");
  });

  it("malformed input returned as-is", () => {
    expect(formatShortDate("not-a-date")).toBe("not-a-date");
    expect(formatShortDate("")).toBe("");
  });
});

describe("countTodayDue", () => {
  it("counts task items per slot, excludes counters + ticked", () => {
    const items = [
      makeItem({ id: 1, slot: "premarket",    item_type: "task",    ticked_today: false }),
      makeItem({ id: 2, slot: "premarket",    item_type: "task",    ticked_today: true, todays_log_id: 1 }),
      makeItem({ id: 3, slot: "after_close",  item_type: "task",    ticked_today: false }),
      makeItem({ id: 4, slot: "after_close",  item_type: "counter", ticked_today: false }),
      makeItem({ id: 5, slot: "intraday",     item_type: "task",    ticked_today: false }),
      makeItem({ id: 6, slot: "end_of_shift", item_type: "task",    ticked_today: false }),
    ];
    const counts = countTodayDue(items);
    expect(counts.premarket).toBe(1);   // 2 is ticked
    expect(counts.intraday).toBe(1);
    expect(counts.end_of_shift).toBe(1);
    expect(counts.after_close).toBe(1); // 4 is counter
    expect(counts.total).toBe(4);
  });

  it("weekly items are excluded from today counts", () => {
    const items = [
      makeItem({ id: 1, frequency: "weekly", slot: "weekend" }),
      makeItem({ id: 2, frequency: "daily",  slot: "premarket" }),
    ];
    const counts = countTodayDue(items);
    expect(counts.premarket).toBe(1);
    expect(counts.total).toBe(1);
  });

  it("null slot is skipped (monthly/quarterly with no slot)", () => {
    const items = [
      makeItem({ id: 1, frequency: "quarterly", slot: null, item_type: "task" }),
    ];
    const counts = countTodayDue(items);
    expect(counts.total).toBe(0);
  });

  it("empty list → all zeros", () => {
    const counts = countTodayDue([]);
    expect(counts).toEqual({
      premarket: 0, intraday: 0, end_of_shift: 0, after_close: 0, total: 0,
    });
  });
});

describe("computeReorderSwap", () => {
  // System items always occupy the top of a group (fixed sort_orders 10, 20,
  // 30…); custom items follow (50, 60, 70…). The helper only touches custom
  // rows within the same (frequency, slot) group.
  const items: RoutineItem[] = [
    makeItem({ id: 1, name: "Sys A", is_system: true,  sort_order: 10 }),
    makeItem({ id: 2, name: "Sys B", is_system: true,  sort_order: 20 }),
    makeItem({ id: 3, name: "Cust 1", is_system: false, sort_order: 50 }),
    makeItem({ id: 4, name: "Cust 2", is_system: false, sort_order: 60 }),
    makeItem({ id: 5, name: "Cust 3", is_system: false, sort_order: 70 }),
    // Different group (weekly · weekend) — should never appear in a swap.
    makeItem({ id: 6, name: "Weekly custom", is_system: false, frequency: "weekly", slot: "weekend", sort_order: 10 }),
  ];

  it("moves middle custom up by swapping sort_order with prior custom sibling", () => {
    const swap = computeReorderSwap(items, 4, "up");
    expect(swap).toEqual([
      { id: 4, sort_order: 50 },
      { id: 3, sort_order: 60 },
    ]);
  });

  it("moves middle custom down by swapping sort_order with next custom sibling", () => {
    const swap = computeReorderSwap(items, 4, "down");
    expect(swap).toEqual([
      { id: 4, sort_order: 70 },
      { id: 5, sort_order: 60 },
    ]);
  });

  it("top custom cannot move up → null", () => {
    expect(computeReorderSwap(items, 3, "up")).toBeNull();
  });

  it("bottom custom cannot move down → null", () => {
    expect(computeReorderSwap(items, 5, "down")).toBeNull();
  });

  it("system item → null (never movable)", () => {
    expect(computeReorderSwap(items, 1, "up")).toBeNull();
    expect(computeReorderSwap(items, 2, "down")).toBeNull();
  });

  it("unknown id → null", () => {
    expect(computeReorderSwap(items, 999, "up")).toBeNull();
  });

  it("does not cross group boundaries (weekly custom stays isolated)", () => {
    // id=6 is the only custom item in weekly·weekend → both directions null.
    expect(computeReorderSwap(items, 6, "up")).toBeNull();
    expect(computeReorderSwap(items, 6, "down")).toBeNull();
  });

  it("tie in sort_order → nudges target so display order actually changes", () => {
    const tied: RoutineItem[] = [
      makeItem({ id: 1, is_system: false, sort_order: 50 }),
      makeItem({ id: 2, is_system: false, sort_order: 50 }),
    ];
    // Down move on id=1: target lands ahead of id=2 by nudge +1.
    const down = computeReorderSwap(tied, 1, "down");
    expect(down).toEqual([
      { id: 1, sort_order: 51 },
      { id: 2, sort_order: 50 },
    ]);
    // Up move on id=2: symmetric — target nudged below by -1.
    const up = computeReorderSwap(tied, 2, "up");
    expect(up).toEqual([
      { id: 2, sort_order: 49 },
      { id: 1, sort_order: 50 },
    ]);
  });
});
