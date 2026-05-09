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
// All Campaigns filter row — Strategy / Instrument / Buy Rule /
// Sell Rule pills. Replaced the bulk-select toolbar from Phase 2;
// the right-click "Set strategy →" path stays for one-off retags.
// ─────────────────────────────────────────────────────────────────────

describe("Analytics — All Campaigns filters", () => {
  test("listStrategies fires on mount (contract guard for fetch wiring)", async () => {
    // The right-click flyout (active=true) AND the new Strategy filter
    // dropdown (active=false) both depend on listStrategies. Two
    // separate fetches by design — assert the call happens at least
    // once so a regressed useEffect surfaces here, not in a silent
    // empty-state UI.
    mClosed.mockResolvedValue([]);
    mOpen.mockResolvedValue([]);
    render(<Analytics navColor="#08a86b" initialTab="campaigns" />);
    await waitFor(() => expect(api.listStrategies).toHaveBeenCalled());
  });

  test("Strategy filter narrows visible rows and lists inactive strategies with '(inactive)' suffix", async () => {
    // Three trades: two tagged StockTalk, one CanSlim. The Strategy
    // dropdown also includes a Retired (inactive) strategy — it must
    // appear in the dropdown (so existing tagged trades stay
    // filterable post-deactivation) with an '(inactive)' suffix.
    mClosed.mockResolvedValue([
      closedTrade({ trade_id: "C1", ticker: "MSFT", strategy: "StockTalk" }),
      closedTrade({ trade_id: "C2", ticker: "AAPL", strategy: "StockTalk" }),
      closedTrade({ trade_id: "C3", ticker: "NVDA", strategy: "CanSlim" }),
    ]);
    mOpen.mockResolvedValue([]);
    // listStrategies is called twice (active + all). Default mock from
    // the module-level vi.mock returns active-only — override per-call
    // to return all strategies including a Retired inactive row when
    // the filter requests them.
    vi.mocked(api.listStrategies).mockImplementation(({ active = true } = {}) => {
      const rows = [
        { name: "CanSlim",   description: null, color: "#6366f1", is_active: true,  created_at: "2026-01-01" },
        { name: "StockTalk", description: null, color: "#d97706", is_active: true,  created_at: "2026-01-02" },
        { name: "Retired",   description: null, color: "#888888", is_active: false, created_at: "2026-01-03" },
      ];
      return Promise.resolve(active ? rows.filter(r => r.is_active) : rows) as any;
    });

    render(<Analytics navColor="#08a86b" initialTab="campaigns" />);
    await screen.findByText("MSFT");

    // Open the Strategy dropdown.
    const strategyFilter = screen.getByTestId("campaigns-strategy-filter");
    const trigger = strategyFilter.querySelector("button") as HTMLButtonElement;
    await waitFor(() => expect(api.listStrategies).toHaveBeenCalledWith({ active: false }));
    fireEvent.click(trigger);

    // Inactive strategy is listed with the '(inactive)' suffix.
    expect(await screen.findByText("(inactive)")).toBeInTheDocument();
    expect(screen.getByText("Retired")).toBeInTheDocument();

    // Pick StockTalk → table narrows to two rows.
    fireEvent.click(screen.getByText("StockTalk"));
    await waitFor(() => {
      expect(screen.getByText("MSFT")).toBeInTheDocument();
      expect(screen.getByText("AAPL")).toBeInTheDocument();
      expect(screen.queryByText("NVDA")).not.toBeInTheDocument();
    });
  });

  test("Buy Rule filter narrows visible rows", async () => {
    mClosed.mockResolvedValue([
      closedTrade({ trade_id: "C1", ticker: "MSFT", rule: "br1.1 Consolidation" }),
      closedTrade({ trade_id: "C2", ticker: "AAPL", rule: "br3.1 Reclaim 21e" }),
      closedTrade({ trade_id: "C3", ticker: "NVDA", rule: "br1.1 Consolidation" }),
    ]);
    mOpen.mockResolvedValue([]);

    render(<Analytics navColor="#08a86b" initialTab="campaigns" />);
    await screen.findByText("MSFT");

    const select = screen.getByTestId("campaigns-buy-rule-filter") as HTMLSelectElement;
    fireEvent.change(select, { target: { value: "br1.1 Consolidation" } });

    await waitFor(() => {
      expect(screen.getByText("MSFT")).toBeInTheDocument();
      expect(screen.getByText("NVDA")).toBeInTheDocument();
      expect(screen.queryByText("AAPL")).not.toBeInTheDocument();
    });
  });

  test("Sell Rule filter narrows visible rows", async () => {
    mClosed.mockResolvedValue([
      closedTrade({ trade_id: "C1", ticker: "MSFT", sell_rule: "sr1.1 Stop hit" }),
      closedTrade({ trade_id: "C2", ticker: "AAPL", sell_rule: "sr2.1 Target reached" }),
      closedTrade({ trade_id: "C3", ticker: "NVDA", sell_rule: "sr1.1 Stop hit" }),
    ]);
    mOpen.mockResolvedValue([]);

    render(<Analytics navColor="#08a86b" initialTab="campaigns" />);
    await screen.findByText("MSFT");

    const select = screen.getByTestId("campaigns-sell-rule-filter") as HTMLSelectElement;
    fireEvent.change(select, { target: { value: "sr1.1 Stop hit" } });

    await waitFor(() => {
      expect(screen.getByText("MSFT")).toBeInTheDocument();
      expect(screen.getByText("NVDA")).toBeInTheDocument();
      expect(screen.queryByText("AAPL")).not.toBeInTheDocument();
    });
  });

  test("Instrument filter narrows visible rows (STOCK / OPTION)", async () => {
    mClosed.mockResolvedValue([
      closedTrade({ trade_id: "C1", ticker: "MSFT", instrument_type: "STOCK" }),
      closedTrade({ trade_id: "C2", ticker: "AMZN 270115 $270C", instrument_type: "OPTION" }),
      closedTrade({ trade_id: "C3", ticker: "NVDA", instrument_type: "STOCK" }),
    ]);
    mOpen.mockResolvedValue([]);

    render(<Analytics navColor="#08a86b" initialTab="campaigns" />);
    await screen.findByText("MSFT");

    const select = screen.getByTestId("campaigns-instrument-filter") as HTMLSelectElement;
    fireEvent.change(select, { target: { value: "OPTION" } });

    await waitFor(() => {
      expect(screen.getByText("AMZN 270115 $270C")).toBeInTheDocument();
      expect(screen.queryByText("MSFT")).not.toBeInTheDocument();
      expect(screen.queryByText("NVDA")).not.toBeInTheDocument();
    });
  });
});
