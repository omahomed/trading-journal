import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach } from "vitest";

vi.mock("@/lib/portfolio-context", () => ({
  usePortfolio: () => ({
    activePortfolio: { id: 2, name: "Long-Term Growth" },
    portfolios: [{ id: 2, name: "Long-Term Growth" }],
    loading: false,
    error: null,
    refetch: vi.fn(),
    setActive: vi.fn(),
  }),
}));

vi.mock("@/lib/api", () => ({
  api: {
    slicesList: vi.fn(),
    slicesToggle: vi.fn(),
    slicesCreate: vi.fn(),
    slicesUpdate: vi.fn(),
    slicesDelete: vi.fn(),
    slicesAssignHolding: vi.fn(),
    slicesRemoveHolding: vi.fn(),
  },
  getActivePortfolio: () => "Long-Term Growth",
}));

import { api } from "@/lib/api";
import { Slices } from "./slices";

const enabledEmptyResponse = {
  portfolio: "Long-Term Growth",
  portfolio_id: 2,
  slices_enabled: true,
  total_market_value: 1000,
  slices: [],
  holdings: [],
  unassigned: [
    { ticker: "AMD", shares: 10, avg_entry: 100, current_price: 100,
      multiplier: 1, market_value: 1000, actual_pct_of_portfolio: 100 },
  ],
};

const enabledWithTreeResponse = {
  portfolio: "Long-Term Growth",
  portfolio_id: 2,
  slices_enabled: true,
  total_market_value: 1000,
  slices: [
    { id: 10, portfolio_id: 2, parent_id: null, name: "AI Chips",
      target_pct: 100, sort_order: 0, color: null,
      subtree_value: 1000, subtree_pct: 100 },
  ],
  holdings: [
    { id: 100, portfolio_id: 2, slice_id: 10, ticker: "AMD", target_pct: 100,
      shares: 10, avg_entry: 100, current_price: 100, multiplier: 1,
      market_value: 1000, actual_pct_of_portfolio: 100, held: true },
  ],
  unassigned: [],
};

const disabledResponse = {
  portfolio: "Long-Term Growth",
  portfolio_id: 2,
  slices_enabled: false,
  total_market_value: 0,
  slices: [],
  holdings: [],
  unassigned: [],
};

describe("Slices page", () => {
  beforeEach(() => {
    vi.mocked(api.slicesList).mockReset();
    vi.mocked(api.slicesToggle).mockReset();
    vi.mocked(api.slicesCreate).mockReset();
    vi.mocked(api.slicesUpdate).mockReset();
    vi.mocked(api.slicesDelete).mockReset();
    vi.mocked(api.slicesAssignHolding).mockReset();
    vi.mocked(api.slicesRemoveHolding).mockReset();
  });

  test("shows the DisabledState when the API returns slices_enabled=false", async () => {
    vi.mocked(api.slicesList).mockResolvedValue(disabledResponse);
    render(<Slices navColor="#0891b2" />);
    await waitFor(() =>
      expect(screen.getByText(/Slices is off for Long-Term Growth\./)).toBeTruthy()
    );
    // Enable CTA is the only visible action.
    expect(screen.getByRole("button", { name: /Enable Slices for/i })).toBeTruthy();
    // KPI tiles + tree list are hidden while disabled.
    expect(screen.queryByText(/Open positions market value/)).toBeNull();
    // Manage button is present in the header but disabled.
    const manage = screen.getByRole("button", { name: "Manage Slices" }) as HTMLButtonElement;
    expect(manage.disabled).toBe(true);
  });

  test("clicking Enable calls slicesToggle and refetches", async () => {
    vi.mocked(api.slicesList)
      .mockResolvedValueOnce(disabledResponse)
      .mockResolvedValueOnce(enabledEmptyResponse);
    vi.mocked(api.slicesToggle).mockResolvedValue({
      portfolio: "Long-Term Growth",
      slices_enabled: true,
    });

    render(<Slices navColor="#0891b2" />);
    const enableBtn = await screen.findByRole("button", {
      name: /Enable Slices for/i,
    });
    fireEvent.click(enableBtn);
    await waitFor(() =>
      expect(api.slicesToggle).toHaveBeenCalledWith("Long-Term Growth", true)
    );
    // After the refetch, the empty-state (not the disabled-state) shows.
    await waitFor(() =>
      expect(screen.getByText(/No slices configured for Long-Term Growth/)).toBeTruthy()
    );
  });

  test("enabled + empty tree surfaces the unassigned banner with Assign now", async () => {
    vi.mocked(api.slicesList).mockResolvedValue(enabledEmptyResponse);
    render(<Slices navColor="#0891b2" />);
    await waitFor(() =>
      expect(screen.getByText(/1 unassigned holding/)).toBeTruthy()
    );
    expect(screen.getByRole("button", { name: "Assign now" })).toBeTruthy();
  });

  test("clicking Manage opens the modal with the tree + unassigned", async () => {
    // Mix: one slice + one unassigned so both modal sections render.
    const mixed = {
      ...enabledWithTreeResponse,
      unassigned: [
        { ticker: "NBIS", shares: 5, avg_entry: 200, current_price: 210,
          multiplier: 1, market_value: 1050, actual_pct_of_portfolio: 50 },
      ],
    };
    vi.mocked(api.slicesList).mockResolvedValue(mixed);
    render(<Slices navColor="#0891b2" />);
    await waitFor(() => expect(screen.getByText("AI Chips")).toBeTruthy());

    // Header Manage button (there are TWO manage-clickables — the header
    // button labelled "Manage Slices" and the disabled placeholder — so
    // scope to the button role.
    fireEvent.click(screen.getByRole("button", { name: "Manage Slices" }));

    // Modal renders with the tree ("AI Chips") + unassigned ("NBIS") sections.
    expect(await screen.findByRole("dialog")).toBeTruthy();
    expect(screen.getAllByText("AI Chips").length).toBeGreaterThan(0);
    // NBIS surfaces in both the unassigned list at the bottom of the page
    // AND in the modal's unassigned tickers section, so getByText would
    // multi-match; assert on the count instead.
    expect(screen.getAllByText("NBIS").length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: /Add root slice/i })).toBeTruthy();
  });

  test("shows Roots total pill on the main page (under cap → green)", async () => {
    vi.mocked(api.slicesList).mockResolvedValue({
      ...enabledWithTreeResponse,
      slices: [
        { id: 10, portfolio_id: 2, parent_id: null, name: "A",
          target_pct: 40, sort_order: 0, color: null,
          subtree_value: 400, subtree_pct: 40 },
        { id: 11, portfolio_id: 2, parent_id: null, name: "B",
          target_pct: 35, sort_order: 1, color: null,
          subtree_value: 350, subtree_pct: 35 },
      ],
    });
    render(<Slices navColor="#0891b2" />);
    await waitFor(() => expect(screen.getByText("A")).toBeTruthy());
    // 40 + 35 = 75% total; under cap → no "over by" suffix.
    expect(screen.getByText(/Roots total: 75\.0% \/ 100%/)).toBeTruthy();
    expect(screen.queryByText(/over by/)).toBeNull();
  });

  test("shows Roots total pill over cap when sum > 100%", async () => {
    // 60 + 55 = 115% — matches the DELL-style user report.
    vi.mocked(api.slicesList).mockResolvedValue({
      ...enabledWithTreeResponse,
      slices: [
        { id: 10, portfolio_id: 2, parent_id: null, name: "A",
          target_pct: 60, sort_order: 0, color: null,
          subtree_value: 600, subtree_pct: 60 },
        { id: 11, portfolio_id: 2, parent_id: null, name: "B",
          target_pct: 55, sort_order: 1, color: null,
          subtree_value: 400, subtree_pct: 40 },
      ],
    });
    render(<Slices navColor="#0891b2" />);
    await waitFor(() => expect(screen.getByText("A")).toBeTruthy());
    expect(screen.getByText(/Roots total: 115\.0% \/ 100%/)).toBeTruthy();
    expect(screen.getByText(/over by 15\.0pp/)).toBeTruthy();
  });

  test("blocks adding a root slice that would push roots total over 100%", async () => {
    // Existing roots sum 90%; try adding a 15% slice → 105% (blocked).
    const overCapResponse = {
      ...enabledWithTreeResponse,
      slices: [
        { id: 10, portfolio_id: 2, parent_id: null, name: "A",
          target_pct: 90, sort_order: 0, color: null,
          subtree_value: 900, subtree_pct: 90 },
      ],
      holdings: [],
      unassigned: [],
    };
    vi.mocked(api.slicesList).mockResolvedValue(overCapResponse);
    render(<Slices navColor="#0891b2" />);
    await waitFor(() => expect(screen.getByText("A")).toBeTruthy());
    fireEvent.click(screen.getByRole("button", { name: "Manage Slices" }));
    fireEvent.click(await screen.findByRole("button", { name: /Add root slice/i }));

    const nameInput = await screen.findByPlaceholderText("New slice name");
    fireEvent.change(nameInput, { target: { value: "TooBig" } });
    // Fill target too — the AddSliceRow uses "0" as placeholder for
    // the numeric target% input.
    const targetInput = screen.getByPlaceholderText("0");
    fireEvent.change(targetInput, { target: { value: "15" } });
    fireEvent.click(screen.getByRole("button", { name: "Add" }));

    // Refuses locally — slicesCreate is NOT called, and an error appears.
    await waitFor(() =>
      expect(screen.getByText(/over 100% cap|would push roots total to 105/i)).toBeTruthy()
    );
    expect(api.slicesCreate).not.toHaveBeenCalled();
  });

  test("adding a root slice from the modal calls slicesCreate and refetches", async () => {
    vi.mocked(api.slicesList)
      .mockResolvedValueOnce(enabledEmptyResponse)
      .mockResolvedValueOnce(enabledWithTreeResponse);
    vi.mocked(api.slicesCreate).mockResolvedValue({
      slice: enabledWithTreeResponse.slices[0],
    });
    render(<Slices navColor="#0891b2" />);
    await waitFor(() => expect(screen.getByText(/No slices configured/)).toBeTruthy());
    fireEvent.click(screen.getByRole("button", { name: "Manage Slices" }));
    fireEvent.click(await screen.findByRole("button", { name: /Add root slice/i }));

    const nameInput = await screen.findByPlaceholderText("New slice name");
    fireEvent.change(nameInput, { target: { value: "AI Chips" } });
    fireEvent.click(screen.getByRole("button", { name: "Add" }));

    await waitFor(() =>
      expect(api.slicesCreate).toHaveBeenCalledWith(expect.objectContaining({
        portfolio: "Long-Term Growth",
        parent_id: null,
        name: "AI Chips",
      }))
    );
    // Second slicesList call = the post-mutation refetch.
    await waitFor(() => expect(api.slicesList).toHaveBeenCalledTimes(2));
  });

  test("renders P&L + Return % columns per slice (2026-08-12 perf columns)", async () => {
    // Two roots — one winner, one loser — so the color / sign paths
    // both render. Server pre-rolls subtree_pl / subtree_return_pct.
    vi.mocked(api.slicesList).mockResolvedValue({
      portfolio: "Long-Term Growth",
      portfolio_id: 2,
      slices_enabled: true,
      total_market_value: 10_000,
      slices: [
        { id: 10, portfolio_id: 2, parent_id: null, name: "Winners",
          target_pct: 60, sort_order: 0, color: null,
          subtree_value: 6000, subtree_pct: 60,
          subtree_pl: 1200, subtree_cost: 4800, subtree_return_pct: 25 },
        { id: 11, portfolio_id: 2, parent_id: null, name: "Losers",
          target_pct: 40, sort_order: 1, color: null,
          subtree_value: 4000, subtree_pct: 40,
          subtree_pl: -600, subtree_cost: 4600, subtree_return_pct: -13.04 },
      ],
      holdings: [],
      unassigned: [],
    });
    render(<Slices navColor="#0891b2" />);
    await waitFor(() => expect(screen.getByText("Winners")).toBeTruthy());
    // P&L $ cells (colored green / red respectively).
    expect(screen.getByText("+$1,200")).toBeTruthy();
    expect(screen.getByText("-$600")).toBeTruthy();
    // Return % cells.
    expect(screen.getByText("+25.0%")).toBeTruthy();
    expect(screen.getByText("-13.0%")).toBeTruthy();
    // Column header exists.
    expect(screen.getAllByText("P&L").length).toBeGreaterThan(0);
    expect(screen.getByText("Return %")).toBeTruthy();
  });

  test("renders Total P&L + Best/Worst KPI tiles", async () => {
    vi.mocked(api.slicesList).mockResolvedValue({
      portfolio: "Long-Term Growth",
      portfolio_id: 2,
      slices_enabled: true,
      total_market_value: 10_000,
      slices: [
        { id: 10, portfolio_id: 2, parent_id: null, name: "AI Chips",
          target_pct: 60, sort_order: 0, color: null,
          subtree_value: 6000, subtree_pct: 60,
          subtree_pl: 1200, subtree_cost: 4800, subtree_return_pct: 25 },
        { id: 11, portfolio_id: 2, parent_id: null, name: "Healthcare",
          target_pct: 40, sort_order: 1, color: null,
          subtree_value: 4000, subtree_pct: 40,
          subtree_pl: -600, subtree_cost: 4600, subtree_return_pct: -13.04 },
      ],
      holdings: [],
      unassigned: [],
    });
    render(<Slices navColor="#0891b2" />);
    await waitFor(() => expect(screen.getByText("AI Chips")).toBeTruthy());
    // Total P&L tile: sum of subtree_pl across roots = 1200 + (-600) = 600.
    // Cost basis: 4800 + 4600 = 9400. Return: 600/9400 = 6.38%.
    expect(screen.getByText("Total P&L")).toBeTruthy();
    expect(screen.getByText("+$600")).toBeTruthy();
    expect(screen.getByText("+6.38% vs cost")).toBeTruthy();
    // Best/Worst tile: winner 25%, loser -13.04% → shows both.
    expect(screen.getByText("Best / Worst Slice")).toBeTruthy();
    expect(screen.getByText("+25.0% / -13.0%")).toBeTruthy();
    expect(screen.getByText("AI Chips · Healthcare")).toBeTruthy();
  });

  test("KPI tiles handle empty/no-cost state gracefully", async () => {
    // Slice exists but has no held positions → subtree_cost = 0 →
    // Total P&L shows "—" without a divide-by-zero.
    vi.mocked(api.slicesList).mockResolvedValue({
      portfolio: "Long-Term Growth",
      portfolio_id: 2,
      slices_enabled: true,
      total_market_value: 0,
      slices: [
        { id: 10, portfolio_id: 2, parent_id: null, name: "Empty",
          target_pct: 100, sort_order: 0, color: null,
          subtree_value: 0, subtree_pct: 0,
          subtree_pl: 0, subtree_cost: 0, subtree_return_pct: 0 },
      ],
      holdings: [],
      unassigned: [],
    });
    render(<Slices navColor="#0891b2" />);
    await waitFor(() => expect(screen.getByText("Empty")).toBeTruthy());
    // Total P&L tile: "—" (both value and default sub).
    expect(screen.getByText("No held positions")).toBeTruthy();
    // Best/Worst tile: "—" too.
    expect(screen.getByText("No performance data")).toBeTruthy();
  });
});
