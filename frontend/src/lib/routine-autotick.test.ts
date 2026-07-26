import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  __resetAutotickCache,
  autoTickByPrefix,
  SYSTEM_ITEM_PREFIXES,
  todayInChicago,
} from "./routine-autotick";

// api.routineItemsList + api.routineLogTick are mocked to avoid the
// fetchWithAuth network path. Every test resets the module cache first
// so cross-test state doesn't bleed.

vi.mock("./api", () => ({
  api: {
    routineItemsList: vi.fn(),
    routineLogTick: vi.fn(),
  },
}));

import { api } from "./api";
const mockList = api.routineItemsList as ReturnType<typeof vi.fn>;
const mockTick = api.routineLogTick as ReturnType<typeof vi.fn>;

describe("todayInChicago", () => {
  it("returns a YYYY-MM-DD string", () => {
    expect(todayInChicago()).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  });
});

describe("SYSTEM_ITEM_PREFIXES", () => {
  it("prefixes match the DB seed names in db_layer._ROUTINE_SYSTEM_ITEMS", () => {
    // If a prefix here drifts from the seed name, autotick silently
    // stops working. This test locks the coupling.
    expect(SYSTEM_ITEM_PREFIXES.equityRoutine).toBe("Equity routine");
    expect(SYSTEM_ITEM_PREFIXES.journal).toBe("Journal");
  });
});

describe("autoTickByPrefix", () => {
  beforeEach(() => {
    __resetAutotickCache();
    mockList.mockReset();
    mockTick.mockReset();
  });

  it("returns false when items list is empty", async () => {
    mockList.mockResolvedValue({ items: [] });
    const result = await autoTickByPrefix("Equity routine");
    expect(result).toBe(false);
    expect(mockTick).not.toHaveBeenCalled();
  });

  it("returns false when list endpoint errors", async () => {
    mockList.mockResolvedValue({ error: "boom" });
    const result = await autoTickByPrefix("Equity routine");
    expect(result).toBe(false);
  });

  it("returns false when no item matches the prefix", async () => {
    mockList.mockResolvedValue({
      items: [{ id: 5, name: "Something else", is_system: true }],
    });
    const result = await autoTickByPrefix("Equity routine");
    expect(result).toBe(false);
    expect(mockTick).not.toHaveBeenCalled();
  });

  it("ticks the first matching item and returns true", async () => {
    mockList.mockResolvedValue({
      items: [
        { id: 5, name: "Something else", is_system: true },
        { id: 42, name: "Equity routine — log NLV, day P&L", is_system: true },
      ],
    });
    mockTick.mockResolvedValue({ log_id: 1, already_ticked: false });
    const result = await autoTickByPrefix("Equity routine");
    expect(result).toBe(true);
    expect(mockTick).toHaveBeenCalledWith(42);
  });

  it("swallows tick-network errors and returns false", async () => {
    mockList.mockResolvedValue({
      items: [{ id: 42, name: "Journal — chart read", is_system: true }],
    });
    mockTick.mockRejectedValue(new Error("network"));
    const result = await autoTickByPrefix("Journal");
    expect(result).toBe(false);
  });

  it("caches the items list across calls (single list fetch)", async () => {
    mockList.mockResolvedValue({
      items: [{ id: 7, name: "Journal — chart read", is_system: true }],
    });
    mockTick.mockResolvedValue({ log_id: 1, already_ticked: false });
    await autoTickByPrefix("Journal");
    await autoTickByPrefix("Journal");
    await autoTickByPrefix("Journal");
    expect(mockList).toHaveBeenCalledTimes(1);
    expect(mockTick).toHaveBeenCalledTimes(3);
  });

  it("caches even when no match found (still avoids re-list on next call)", async () => {
    mockList.mockResolvedValue({ items: [] });
    await autoTickByPrefix("Journal");
    await autoTickByPrefix("Journal");
    expect(mockList).toHaveBeenCalledTimes(1);
  });
});
