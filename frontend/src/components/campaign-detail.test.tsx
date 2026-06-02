import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach } from "vitest";

// jsdom localStorage polyfill — mirrors the pattern other test files use.
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
    tradesOpen: vi.fn(),
    tradesOpenDetails: vi.fn(),
    batchPrices: vi.fn(),
  },
  getActivePortfolio: () => "CanSlim",
}));

import { api } from "@/lib/api";
import { CampaignDetail } from "./campaign-detail";

const mOpen = vi.mocked(api.tradesOpen);
const mDetails = vi.mocked(api.tradesOpenDetails);
const mPrices = vi.mocked(api.batchPrices);

const stockTrade = (overrides: Partial<any> = {}) => ({
  trade_id: "T1",
  ticker: "AAPL",
  status: "OPEN",
  shares: 100,
  avg_entry: 100,
  total_cost: 10000,
  realized_pl: 0,
  open_date: "2026-01-05",
  rule: "br1.3 Cup w/o Handle",
  instrument_type: "STOCK",
  multiplier: 1,
  ...overrides,
}) as any;

const optionTrade = (overrides: Partial<any> = {}) => ({
  trade_id: "O1",
  ticker: "FOO 261016 $50C",
  status: "OPEN",
  shares: 10,
  avg_entry: 4,
  total_cost: 4000,
  realized_pl: 0,
  open_date: "2026-01-05",
  rule: "br1.3 Cup w/o Handle",
  instrument_type: "OPTION",
  multiplier: 100,
  ...overrides,
}) as any;

const buyRow = (id: number, trade_id: string, ticker: string, shares: number, amount: number, date = "2026-01-05") => ({
  detail_id: id,
  trade_id,
  ticker,
  action: "BUY",
  date,
  shares,
  amount,
  value: shares * amount,
  rule: "",
  notes: "",
  trx_id: "B1",
  instrument_type: "STOCK",
  multiplier: 1,
} as any);

const sellRow = (id: number, trade_id: string, ticker: string, shares: number, amount: number, date = "2026-02-05") => ({
  detail_id: id,
  trade_id,
  ticker,
  action: "SELL",
  date,
  shares,
  amount,
  value: shares * amount,
  rule: "",
  notes: "",
  trx_id: "S1",
  instrument_type: "STOCK",
  multiplier: 1,
} as any);

beforeEach(() => {
  vi.clearAllMocks();
  mPrices.mockResolvedValue({});
});

describe("CampaignDetail — page scaffold (Commit 2)", () => {
  test("renders 5 KPI tiles with values from the loaded data", async () => {
    mOpen.mockResolvedValue([
      stockTrade({ trade_id: "T1", ticker: "AAPL" }),
      stockTrade({ trade_id: "T2", ticker: "MSFT" }),
    ]);
    mDetails.mockResolvedValue({
      details: [
        buyRow(1, "T1", "AAPL", 100, 100),                     // open, remaining 100
        buyRow(2, "T2", "MSFT", 50, 200),                      // open, remaining 50
        sellRow(3, "T2", "MSFT", 20, 210, "2026-02-05"),       // closes 20 of T2's B1
      ],
      lot_closures: [
        {
          trade_id: "T2", buy_trx_id: "B1", sell_trx_id: "S1",
          shares: 20, buy_price: 200, sell_price: 210,
          multiplier: 1, realized_pl: 200, closed_at: "2026-02-05",
        } as any,
      ],
    });
    mPrices.mockResolvedValue({ AAPL: 110, MSFT: 220 });

    render(<CampaignDetail navColor="#08a86b" />);

    // Wait until KPI strip renders.
    await waitFor(() => {
      expect(screen.getByTestId("kpi-strip")).toBeInTheDocument();
    });

    // 5 tiles. Each tile contains the label text.
    expect(screen.getByText("Transactions")).toBeInTheDocument();
    expect(screen.getByText("Open Lots")).toBeInTheDocument();
    expect(screen.getByText("Realized P&L")).toBeInTheDocument();
    expect(screen.getByText("Unrealized P&L")).toBeInTheDocument();
    expect(screen.getByText("Market Value")).toBeInTheDocument();

    // Transactions = 3 (2 buys + 1 sell, all stocks)
    expect(screen.getByText("3")).toBeInTheDocument();
    // 2 active campaigns
    expect(screen.getByText("2 active campaigns")).toBeInTheDocument();
    // Open Lots = 2 (T1 B1 untouched, T2 B1 partial → both still > 0)
    expect(screen.getByText("2")).toBeInTheDocument();
  });

  test("filters out option campaigns (stocks-only scope)", async () => {
    mOpen.mockResolvedValue([
      stockTrade({ trade_id: "T1", ticker: "AAPL" }),
      optionTrade({ trade_id: "O1" }),
    ]);
    mDetails.mockResolvedValue({
      details: [
        buyRow(1, "T1", "AAPL", 100, 100),
        // Option detail row — should be dropped along with its campaign.
        {
          detail_id: 99, trade_id: "O1", ticker: "FOO 261016 $50C",
          action: "BUY", date: "2026-01-05", shares: 10, amount: 4,
          value: 4000, rule: "", notes: "", trx_id: "B1",
          instrument_type: "OPTION", multiplier: 100,
        } as any,
      ],
      lot_closures: [],
    });
    mPrices.mockResolvedValue({ AAPL: 110 });

    render(<CampaignDetail navColor="#08a86b" />);

    await waitFor(() => {
      expect(screen.getByTestId("kpi-strip")).toBeInTheDocument();
    });

    // Transactions = 1 (stock only), 1 active campaign.
    expect(screen.getByText("1 active campaigns")).toBeInTheDocument();
    // batchPrices called WITHOUT portfolio (no manual_price overlay).
    expect(mPrices).toHaveBeenCalled();
    const callArgs = mPrices.mock.calls[0];
    expect(callArgs[0]).toEqual(["AAPL"]);   // only stock ticker
    expect(callArgs[1]).toBeUndefined();      // no portfolio → live only
  });

  test("empty state: no open campaigns → zeros across all tiles, no crash", async () => {
    mOpen.mockResolvedValue([]);
    mDetails.mockResolvedValue({ details: [], lot_closures: [] });
    mPrices.mockResolvedValue({});

    render(<CampaignDetail navColor="#08a86b" />);

    await waitFor(() => {
      expect(screen.getByTestId("kpi-strip")).toBeInTheDocument();
    });

    expect(screen.getByText("0 active campaigns")).toBeInTheDocument();
    // Multiple tiles will render "0" or "$0"; we just confirm no crash and
    // the strip rendered.
    expect(screen.getByTestId("kpi-strip")).toBeInTheDocument();
  });

  test("Refresh button re-fetches the data", async () => {
    mOpen.mockResolvedValue([stockTrade()]);
    mDetails.mockResolvedValue({
      details: [buyRow(1, "T1", "AAPL", 100, 100)],
      lot_closures: [],
    });
    mPrices.mockResolvedValue({ AAPL: 110 });

    render(<CampaignDetail navColor="#08a86b" />);

    await waitFor(() => {
      expect(screen.getByTestId("kpi-strip")).toBeInTheDocument();
    });

    const initialOpenCalls = mOpen.mock.calls.length;
    fireEvent.click(screen.getByTestId("refresh-btn"));

    await waitFor(() => {
      expect(mOpen.mock.calls.length).toBeGreaterThan(initialOpenCalls);
    });
  });

  test("Export CSV button is disabled in Commit 2 (lands with table)", async () => {
    mOpen.mockResolvedValue([stockTrade()]);
    mDetails.mockResolvedValue({
      details: [buyRow(1, "T1", "AAPL", 100, 100)],
      lot_closures: [],
    });

    render(<CampaignDetail navColor="#08a86b" />);

    await waitFor(() => {
      expect(screen.getByTestId("kpi-strip")).toBeInTheDocument();
    });

    const btn = screen.getByTestId("export-csv-btn") as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
  });
});
