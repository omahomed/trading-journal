import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach } from "vitest";

// localStorage shim — admin doesn't itself read it, but child config
// loads do. Same pattern as other test files in this directory.
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
  });
}

vi.mock("@/lib/api", () => ({
  api: {
    config: vi.fn().mockResolvedValue({ value: null }),
    setConfig: vi.fn(),
    events: vi.fn().mockResolvedValue([]),
    addEvent: vi.fn(),
    updateEvent: vi.fn(),
    deleteEvent: vi.fn(),
    audit: vi.fn().mockResolvedValue([]),
    cleanupMarketsurge: vi.fn(),
    rebuildMctSignals: vi.fn(),
    journalBackfillMetrics: vi.fn(),
    listStrategies: vi.fn(),
    createStrategy: vi.fn(),
    updateStrategy: vi.fn(),
  },
  getActivePortfolio: () => "CanSlim",
}));

import { api } from "@/lib/api";
import { Admin } from "./admin";

const SEED_STRATEGIES = [
  { name: "CanSlim",   description: "primary",   color: "#6366f1", is_active: true,  created_at: "2026-01-01" },
  { name: "StockTalk", description: "small-cap", color: "#d97706", is_active: true,  created_at: "2026-01-02" },
  { name: "Retired",   description: "old",       color: "#888888", is_active: false, created_at: "2026-01-03" },
];

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(api.listStrategies).mockResolvedValue(SEED_STRATEGIES as any);
  vi.mocked(api.createStrategy).mockResolvedValue({
    name: "Momentum", description: "swing", color: "#22c55e",
    is_active: true, created_at: "2026-05-08",
  } as any);
  vi.mocked(api.updateStrategy).mockResolvedValue({
    name: "CanSlim", description: "Updated", color: "#6366f1",
    is_active: true, created_at: "2026-01-01",
  } as any);
});

async function openStrategiesSection() {
  render(<Admin navColor="#6366f1" />);
  // Section header is collapsible — click it to reveal the table.
  const sectionHeader = await screen.findByText("Strategies");
  await act(async () => { fireEvent.click(sectionHeader); });
  await waitFor(() => expect(api.listStrategies).toHaveBeenCalled());
}

describe("Admin — Strategies section", () => {
  test("lists all strategies (including inactive) with their chips", async () => {
    await openStrategiesSection();

    // Each seeded strategy renders.
    expect(await screen.findByText("CanSlim")).toBeInTheDocument();
    expect(screen.getByText("StockTalk")).toBeInTheDocument();
    expect(screen.getByText("Retired")).toBeInTheDocument();
  });

  test("inline is_active toggle calls api.updateStrategy with the flipped value", async () => {
    await openStrategiesSection();

    const toggle = await screen.findByTestId("admin-strategy-active-CanSlim");
    expect(toggle).toBeChecked();

    await act(async () => { fireEvent.click(toggle); });

    await waitFor(() => expect(api.updateStrategy).toHaveBeenCalled());
    const [name, body] = vi.mocked(api.updateStrategy).mock.calls[0];
    expect(name).toBe("CanSlim");
    expect(body.is_active).toBe(false);
  });

  test("Add Strategy modal: submit calls api.createStrategy with form values", async () => {
    await openStrategiesSection();

    const addBtn = await screen.findByTestId("admin-add-strategy");
    await act(async () => { fireEvent.click(addBtn); });

    expect(await screen.findByTestId("admin-strategy-modal")).toBeInTheDocument();

    // Fill the form. Field labels render the input as the next sibling.
    const nameInput = screen.getByPlaceholderText("CanSlim") as HTMLInputElement;
    const colorInput = screen.getByPlaceholderText("#6366f1") as HTMLInputElement;
    const descInput = screen.getByPlaceholderText("Short prose") as HTMLInputElement;

    fireEvent.change(nameInput, { target: { value: "Momentum" } });
    fireEvent.change(colorInput, { target: { value: "#22c55e" } });
    fireEvent.change(descInput, { target: { value: "swing" } });

    await act(async () => {
      fireEvent.click(screen.getByText("Create Strategy"));
    });

    await waitFor(() => expect(api.createStrategy).toHaveBeenCalled());
    const body = vi.mocked(api.createStrategy).mock.calls[0][0];
    expect(body.name).toBe("Momentum");
    expect(body.color).toBe("#22c55e");
    expect(body.description).toBe("swing");
    expect(body.is_active).toBe(true);
  });

  test("Add Strategy modal: client-side hex validation blocks submit on bad color", async () => {
    await openStrategiesSection();

    await act(async () => {
      fireEvent.click(await screen.findByTestId("admin-add-strategy"));
    });

    const nameInput = screen.getByPlaceholderText("CanSlim") as HTMLInputElement;
    const colorInput = screen.getByPlaceholderText("#6366f1") as HTMLInputElement;

    fireEvent.change(nameInput, { target: { value: "Bad" } });
    fireEvent.change(colorInput, { target: { value: "not-hex" } });

    await act(async () => {
      fireEvent.click(screen.getByText("Create Strategy"));
    });

    // Inline error appears; createStrategy is NOT called.
    expect(await screen.findByText(/six-digit hex/i)).toBeInTheDocument();
    expect(api.createStrategy).not.toHaveBeenCalled();
  });

  test("Edit Strategy modal: submit calls api.updateStrategy with patched fields", async () => {
    await openStrategiesSection();

    const editBtn = await screen.findByTestId("admin-strategy-edit-CanSlim");
    await act(async () => { fireEvent.click(editBtn); });

    expect(await screen.findByTestId("admin-strategy-modal")).toBeInTheDocument();

    // Description input shows the existing value; change it.
    const descInput = screen.getByPlaceholderText("Short prose") as HTMLInputElement;
    expect(descInput.value).toBe("primary");
    fireEvent.change(descInput, { target: { value: "Updated description" } });

    await act(async () => {
      fireEvent.click(screen.getByText("Save Changes"));
    });

    await waitFor(() => expect(api.updateStrategy).toHaveBeenCalled());
    const [name, body] = vi.mocked(api.updateStrategy).mock.calls[0];
    expect(name).toBe("CanSlim");
    expect(body.description).toBe("Updated description");
  });
});
