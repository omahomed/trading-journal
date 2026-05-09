import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach } from "vitest";

// jsdom doesn't provide localStorage in some configurations; matches the
// stub used by active-campaign.test.tsx.
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

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(), replace: vi.fn(), refresh: vi.fn(),
    back: vi.fn(), forward: vi.fn(), prefetch: vi.fn(),
  }),
}));

vi.mock("@/lib/api", () => ({
  api: {
    tradesClosed: vi.fn(),
    tradesOpen: vi.fn(),
    journalHistory: vi.fn(),
    tradesRecent: vi.fn(),
    getTradeLessons: vi.fn(),
    batchPrices: vi.fn(),
    listStrategies: vi.fn().mockResolvedValue([
      { name: "CanSlim", description: null, color: "#6366f1", is_active: true, created_at: "2026-01-01" },
      { name: "StockTalk", description: null, color: "#d97706", is_active: true, created_at: "2026-01-02" },
    ]),
    setTradeStrategy: vi.fn().mockResolvedValue({ ok: true }),
    bulkSetStrategy: vi.fn().mockResolvedValue({ ok: true, updated: 0, failed: [] }),
  },
  getActivePortfolio: () => "CanSlim",
}));

import { api } from "@/lib/api";
import { Analytics } from "./analytics";

const mClosed = vi.mocked(api.tradesClosed);
const mOpen = vi.mocked(api.tradesOpen);
const mJournal = vi.mocked(api.journalHistory);
const mRecent = vi.mocked(api.tradesRecent);
const mLessons = vi.mocked(api.getTradeLessons);
const mPrices = vi.mocked(api.batchPrices);

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const closedTrade = (overrides: Partial<any> = {}) => ({
  trade_id: "202601-001",
  ticker: "MSFT",
  status: "CLOSED",
  shares: 0,
  avg_entry: 400,
  avg_exit: 440,
  total_cost: 40000,
  realized_pl: 4000,
  rule: "Breakout",
  open_date: "2026-01-01",
  closed_date: "2026-02-01",
  instrument_type: "STOCK",
  multiplier: 1,
  ...overrides,
}) as any;

const openTrade = (overrides: Partial<any> = {}) => ({
  trade_id: "202603-001",
  ticker: "AAPL",
  status: "OPEN",
  shares: 100,
  avg_entry: 200,
  total_cost: 20000,
  realized_pl: 0,
  rule: "RS leader",
  open_date: "2026-03-01",
  instrument_type: "STOCK",
  multiplier: 1,
  ...overrides,
}) as any;

const txn = (o: { trade_id: string; action: string; date: string; shares: number; amount: number }) => o as any;

beforeEach(() => {
  vi.clearAllMocks();
  mJournal.mockResolvedValue([]);
  mLessons.mockResolvedValue({ lessons: {} });
  mRecent.mockResolvedValue({ details: [], lot_closures: [] });
  mPrices.mockResolvedValue({});
});

describe("Analytics — All Campaigns Flight Deck", () => {
  test("closed-only filter shows realized-trade tiles (Win Rate, Profit Factor, W/L)", async () => {
    mClosed.mockResolvedValue([
      closedTrade({ trade_id: "C1", realized_pl: 4000 }),
      closedTrade({ trade_id: "C2", realized_pl: -1500 }),
      closedTrade({ trade_id: "C3", realized_pl: 2500 }),
    ]);
    mOpen.mockResolvedValue([]);

    render(<Analytics navColor="#08a86b" initialTab="campaigns" />);

    // Closed-mode tile labels
    expect(await screen.findByText("Win Rate")).toBeInTheDocument();
    expect(screen.getByText("Profit Factor")).toBeInTheDocument();
    expect(screen.getByText("W / L")).toBeInTheDocument();
    expect(screen.getByText("Net P&L")).toBeInTheDocument();
    expect(screen.getByText("Avg P&L")).toBeInTheDocument();

    // Open-mode label should NOT be present
    expect(screen.queryByText("Unrealized P&L")).not.toBeInTheDocument();
    expect(screen.queryByText("Avg Days Held")).not.toBeInTheDocument();
  });

  test("open-only filter shows unrealized-trade tiles (Unrealized P&L, Avg Days Held, Total Value)", async () => {
    mClosed.mockResolvedValue([]);
    mOpen.mockResolvedValue([
      openTrade({ trade_id: "O1", ticker: "AAPL" }),
      openTrade({ trade_id: "O2", ticker: "NVDA", avg_entry: 800, total_cost: 80000 }),
    ]);
    mRecent.mockResolvedValue({
      details: [
        txn({ trade_id: "O1", action: "BUY", date: "2026-03-01", shares: 100, amount: 200 }),
        txn({ trade_id: "O2", action: "BUY", date: "2026-03-15", shares: 100, amount: 800 }),
      ],
      lot_closures: [],
    });
    mPrices.mockResolvedValue({ AAPL: 220, NVDA: 850 });

    render(<Analytics navColor="#08a86b" initialTab="campaigns" />);

    // Wait for live prices to flow through
    await waitFor(() => expect(mPrices).toHaveBeenCalled());

    expect(await screen.findByText("Unrealized P&L")).toBeInTheDocument();
    expect(screen.getByText("In Profit / Loss")).toBeInTheDocument();
    expect(screen.getByText("Avg Unrealized P&L")).toBeInTheDocument();
    expect(screen.getByText("Total Value")).toBeInTheDocument();
    expect(screen.getByText("Avg Days Held")).toBeInTheDocument();

    // Closed-mode labels should NOT be present
    expect(screen.queryByText("Win Rate")).not.toBeInTheDocument();
    expect(screen.queryByText("Profit Factor")).not.toBeInTheDocument();
  });

  test("mixed filter shows breakdown tiles (Total P&L, Closed/Open, Net Realized, Net Unrealized)", async () => {
    mClosed.mockResolvedValue([
      closedTrade({ trade_id: "C1", realized_pl: 3000 }),
      closedTrade({ trade_id: "C2", realized_pl: -1000 }),
    ]);
    mOpen.mockResolvedValue([
      openTrade({ trade_id: "O1", ticker: "AAPL" }),
    ]);
    mRecent.mockResolvedValue({
      details: [txn({ trade_id: "O1", action: "BUY", date: "2026-03-01", shares: 100, amount: 200 })],
      lot_closures: [],
    });
    mPrices.mockResolvedValue({ AAPL: 220 });

    render(<Analytics navColor="#08a86b" initialTab="campaigns" />);

    // Default Status filter is "all" → mixed mode (both open + closed visible)
    expect(await screen.findByText("Total P&L")).toBeInTheDocument();
    expect(screen.getByText("Closed / Open")).toBeInTheDocument();
    expect(screen.getByText("Net Realized")).toBeInTheDocument();
    expect(screen.getByText("Net Unrealized")).toBeInTheDocument();
    // Win Rate appears with "closed only" subtitle in mixed mode
    expect(screen.getByText("Win Rate")).toBeInTheDocument();
    expect(screen.getByText("closed only")).toBeInTheDocument();

    // Single-mode-only labels should NOT appear in mixed
    expect(screen.queryByText("Profit Factor")).not.toBeInTheDocument();
    expect(screen.queryByText("Unrealized P&L")).not.toBeInTheDocument();
    expect(screen.queryByText("Avg Days Held")).not.toBeInTheDocument();
  });

  test("totals header shows 'N open · M closed · T total'", async () => {
    mClosed.mockResolvedValue([
      closedTrade({ trade_id: "C1" }),
      closedTrade({ trade_id: "C2" }),
    ]);
    mOpen.mockResolvedValue([
      openTrade({ trade_id: "O1" }),
      openTrade({ trade_id: "O2" }),
      openTrade({ trade_id: "O3" }),
    ]);

    render(<Analytics navColor="#08a86b" />);

    expect(await screen.findByText("3 open · 2 closed · 5 total")).toBeInTheDocument();
  });

  test("filter toggle from All to Closed swaps mixed-mode tiles to closed-mode tiles", async () => {
    mClosed.mockResolvedValue([closedTrade({ trade_id: "C1", realized_pl: 1000 })]);
    mOpen.mockResolvedValue([openTrade({ trade_id: "O1" })]);
    mRecent.mockResolvedValue({
      details: [txn({ trade_id: "O1", action: "BUY", date: "2026-03-01", shares: 100, amount: 200 })],
      lot_closures: [],
    });
    mPrices.mockResolvedValue({ AAPL: 210 });

    render(<Analytics navColor="#08a86b" initialTab="campaigns" />);

    // Initial: mixed mode
    expect(await screen.findByText("Total P&L")).toBeInTheDocument();
    expect(screen.queryByText("Profit Factor")).not.toBeInTheDocument();

    // Click "closed" filter button (it's the third in the filter group, lowercase)
    fireEvent.click(screen.getByRole("button", { name: "closed" }));

    // After filter: closed-only tiles
    await waitFor(() => expect(screen.getByText("Profit Factor")).toBeInTheDocument());
    expect(screen.getByText("W / L")).toBeInTheDocument();
    expect(screen.queryByText("Total P&L")).not.toBeInTheDocument();
  });

  test("winners filter classifies open trade by overall_pl, not realized_pl", async () => {
    // Open trade with NEGATIVE realized_pl (partial sell at a loss) but
    // POSITIVE overall_pl (current price recovered above avg entry).
    // Should appear in "winners" filter and NOT in "losers".
    mClosed.mockResolvedValue([]);
    mOpen.mockResolvedValue([
      openTrade({
        trade_id: "O1",
        ticker: "RECOV",
        shares: 50,
        avg_entry: 100,
        total_cost: 5000,
        realized_pl: -1000, // partial sell at a loss
      }),
    ]);
    mRecent.mockResolvedValue({
      details: [
        txn({ trade_id: "O1", action: "BUY",  date: "2026-01-01", shares: 100, amount: 100 }),
        txn({ trade_id: "O1", action: "SELL", date: "2026-02-01", shares: 50,  amount: 80 }),
      ],
      lot_closures: [],
    });
    // Live price 130 → unrealized = (130-100)*50 = +1500; realized_bank = -1000
    // overall_pl = +500 → Winner
    mPrices.mockResolvedValue({ RECOV: 130 });

    render(<Analytics navColor="#08a86b" initialTab="campaigns" />);

    await waitFor(() => expect(mPrices).toHaveBeenCalled());
    expect(await screen.findByText("RECOV")).toBeInTheDocument();

    // Switch to winners — should still appear (overall_pl > 0)
    fireEvent.click(screen.getByRole("button", { name: "winners" }));
    await waitFor(() => expect(screen.getByText("RECOV")).toBeInTheDocument());

    // Switch to losers — should NOT appear (overall_pl > 0)
    fireEvent.click(screen.getByRole("button", { name: "losers" }));
    await waitFor(() =>
      expect(screen.queryByText("RECOV")).not.toBeInTheDocument()
    );
  });

  test("open trade renders '—' in Close/Exit/Return % even when partial-sell data exists", async () => {
    mClosed.mockResolvedValue([]);
    mOpen.mockResolvedValue([
      openTrade({ trade_id: "O1", ticker: "PSELL", shares: 50, avg_entry: 100, realized_pl: 1000 }),
    ]);
    mRecent.mockResolvedValue({
      details: [
        txn({ trade_id: "O1", action: "BUY",  date: "2026-01-01", shares: 100, amount: 100 }),
        txn({ trade_id: "O1", action: "SELL", date: "2026-02-01", shares: 50,  amount: 120 }),
      ],
      lot_closures: [],
    });
    mPrices.mockResolvedValue({ PSELL: 100 });

    render(<Analytics navColor="#08a86b" initialTab="campaigns" />);

    // Without the fix, partial sell would render: Close="2026-02-01",
    // Exit="$120.00", Return="+20.0%". With the fix all three are "—".
    await screen.findByText("PSELL");
    expect(screen.queryByText("$120.00")).not.toBeInTheDocument();
    expect(screen.queryByText("+20.0%")).not.toBeInTheDocument();
    expect(screen.queryByText("2026-02-01")).not.toBeInTheDocument();
  });
});


// ─────────────────────────────────────────────────────────────────────
// Phase 2 — Bulk-select on All Campaigns.
// ─────────────────────────────────────────────────────────────────────

describe("Analytics — bulk strategy tagging", () => {
  test("selecting rows reveals the toolbar; Tag as → StockTalk fires bulk PATCH", async () => {
    mClosed.mockResolvedValue([
      closedTrade({ trade_id: "C1", ticker: "MSFT" }),
      closedTrade({ trade_id: "C2", ticker: "AAPL" }),
      closedTrade({ trade_id: "C3", ticker: "NVDA" }),
    ]);
    mOpen.mockResolvedValue([]);
    vi.mocked(api.bulkSetStrategy).mockResolvedValue({
      ok: true, updated: 2, failed: [], strategy: "StockTalk",
    } as any);

    render(<Analytics navColor="#08a86b" initialTab="campaigns" />);

    // Wait for strategies + table.
    await waitFor(() => expect(api.listStrategies).toHaveBeenCalled());
    await screen.findByText("MSFT");

    // No toolbar yet.
    expect(screen.queryByTestId("campaigns-bulk-toolbar")).not.toBeInTheDocument();

    // Select two rows by clicking their checkboxes.
    fireEvent.click(screen.getByTestId("campaigns-select-C1"));
    fireEvent.click(screen.getByTestId("campaigns-select-C2"));

    // Toolbar appears with "2 selected".
    const toolbar = await screen.findByTestId("campaigns-bulk-toolbar");
    expect(toolbar).toHaveTextContent("2 selected");

    // Click "Tag as ▾" to open the dropdown.
    fireEvent.click(screen.getByTestId("campaigns-tag-as"));

    // Click StockTalk option.
    fireEvent.click(screen.getByText("StockTalk"));

    await waitFor(() => expect(api.bulkSetStrategy).toHaveBeenCalled());
    const body = vi.mocked(api.bulkSetStrategy).mock.calls[0][0];
    expect(body.strategy).toBe("StockTalk");
    expect(body.trade_ids.sort()).toEqual(["C1", "C2"]);
  });
});
