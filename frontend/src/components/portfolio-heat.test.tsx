import { render, screen, waitFor } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach } from "vitest";

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(), replace: vi.fn(), refresh: vi.fn(),
    back: vi.fn(), forward: vi.fn(), prefetch: vi.fn(),
  }),
}));

vi.mock("@/lib/api", () => ({
  api: {
    tradesOpen: vi.fn(),
    journalLatest: vi.fn(),
    priceLookup: vi.fn(),
    priceLookupBatch: vi.fn(),
  },
  getActivePortfolio: () => "CanSlim",
}));

import { api } from "@/lib/api";
import { PortfolioHeat } from "./portfolio-heat";

const mockedTradesOpen = vi.mocked(api.tradesOpen);
const mockedJournalLatest = vi.mocked(api.journalLatest);
const mockedPriceLookupBatch = vi.mocked(api.priceLookupBatch);


const _STOCK_POSITIONS = [
  { trade_id: "A", ticker: "AAPL", total_cost: 10000, shares: 50, rule: "" },
  { trade_id: "B", ticker: "NVDA", total_cost: 12000, shares: 60, rule: "" },
  { trade_id: "C", ticker: "SNDK", total_cost: 8000, shares: 10, rule: "" },
];


function setupDefaultMocks() {
  mockedTradesOpen.mockResolvedValue(_STOCK_POSITIONS as any);
  mockedJournalLatest.mockResolvedValue({ end_nlv: 100000 } as any);
  mockedPriceLookupBatch.mockResolvedValue({
    results: [
      { ticker: "AAPL", price: 200, atr_pct: 2.5, status: "ok" },
      { ticker: "NVDA", price: 215, atr_pct: 3.9, status: "ok" },
      { ticker: "SNDK", price: 1478, atr_pct: 9.1, status: "ok" },
    ],
  } as any);
}


describe("PortfolioHeat — batch lookup", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaultMocks();
  });

  test("calls priceLookupBatch ONCE with all tickers (not N sequential)", async () => {
    render(<PortfolioHeat navColor="#6366f1" />);
    await waitFor(() => expect(mockedPriceLookupBatch).toHaveBeenCalled());

    // Exactly one batch call, all 3 tickers in the request
    expect(mockedPriceLookupBatch).toHaveBeenCalledTimes(1);
    const tickers = mockedPriceLookupBatch.mock.calls[0][0];
    expect(tickers).toEqual(expect.arrayContaining(["AAPL", "NVDA", "SNDK"]));
    expect(tickers).toHaveLength(3);
  });

  test("per-status badges render: empty, sparse, error", async () => {
    mockedPriceLookupBatch.mockResolvedValue({
      results: [
        { ticker: "AAPL", price: 200, atr_pct: 2.5, status: "ok" },
        { ticker: "NVDA", price: null, atr_pct: null, status: "empty" },
        { ticker: "SNDK", price: 1478, atr_pct: 0, status: "sparse" },
      ],
    } as any);

    render(<PortfolioHeat navColor="#6366f1" />);
    await waitFor(() => expect(mockedPriceLookupBatch).toHaveBeenCalled());

    // AAPL has status="ok" → no status badge
    await waitFor(() => {
      expect(screen.queryByTestId("status-badge-AAPL")).not.toBeInTheDocument();
    });
    expect(screen.getByTestId("status-badge-NVDA")).toHaveTextContent("NO DATA");
    expect(screen.getByTestId("status-badge-SNDK")).toHaveTextContent("SHORT HISTORY");
  });

  test("error status renders FAILED badge with orange accent", async () => {
    mockedPriceLookupBatch.mockResolvedValue({
      results: [
        { ticker: "AAPL", price: 200, atr_pct: 2.5, status: "ok" },
        { ticker: "NVDA", price: null, atr_pct: null, status: "error" },
        { ticker: "SNDK", price: 1478, atr_pct: 9.1, status: "ok" },
      ],
    } as any);

    render(<PortfolioHeat navColor="#6366f1" />);
    await waitFor(() => expect(mockedPriceLookupBatch).toHaveBeenCalled());

    const badge = await screen.findByTestId("status-badge-NVDA");
    expect(badge).toHaveTextContent("FAILED");
  });

  test("429 batch response shows the rate-limit banner", async () => {
    mockedPriceLookupBatch.mockRejectedValue(new Error("API error: 429 Too Many Requests"));
    render(<PortfolioHeat navColor="#6366f1" />);

    const banner = await screen.findByTestId("batch-failure-banner");
    expect(banner).toHaveTextContent(/Rate limit hit/);
  });

  test("non-429 batch failure shows the generic error banner", async () => {
    mockedPriceLookupBatch.mockRejectedValue(new Error("API error: 503 Service Unavailable"));
    render(<PortfolioHeat navColor="#6366f1" />);

    const banner = await screen.findByTestId("batch-failure-banner");
    expect(banner).toHaveTextContent(/Could not load price data/);
  });

  test("renders empty state instead of fake $100k when journal has no NLV", async () => {
    // Historical bug: journalLatest returning {"error": "No journal data"}
    // caused a silent `|| 100000` fallback in the render path, so the page
    // showed EQUITY BASIS = $100,000 and weights summing to 200%+ against
    // that fake basis. Trigger for the regression was migration 054 (LTG
    // reset) which intentionally wipes trading_journal — every downstream
    // consumer of end_nlv is now expected to bail rather than fake.
    // Backend also now walks back to the last row with a real end_nlv so
    // a same-day checklist-only entry doesn't fire this state.
    mockedJournalLatest.mockResolvedValue({ error: "No journal data" } as any);
    render(<PortfolioHeat navColor="#6366f1" />);

    // Empty state renders with the actionable Log-NLV CTA.
    expect(await screen.findByText(/No NLV history for CanSlim/i)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Log NLV/i })).toHaveAttribute("href", "/nlv-entry");

    // Zero fake numbers on the page: no EQUITY BASIS tile, no heat table.
    expect(screen.queryByText(/EQUITY BASIS/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/\$100,000/)).not.toBeInTheDocument();

    // Belt-and-suspenders: the ATR batch fetch must NOT fire either — the
    // page is bailing before it can compute anything against a missing NLV.
    expect(mockedPriceLookupBatch).not.toHaveBeenCalled();
  });

  test("renders empty state when journalLatest rejects (network failure)", async () => {
    mockedJournalLatest.mockRejectedValue(new Error("500 boom"));
    render(<PortfolioHeat navColor="#6366f1" />);
    expect(await screen.findByText(/No NLV history for CanSlim/i)).toBeInTheDocument();
    expect(mockedPriceLookupBatch).not.toHaveBeenCalled();
  });

  test("non-ok summary in header subtitle when statuses are mixed", async () => {
    mockedPriceLookupBatch.mockResolvedValue({
      results: [
        { ticker: "AAPL", price: 200, atr_pct: 2.5, status: "ok" },
        { ticker: "NVDA", price: null, atr_pct: null, status: "empty" },
        { ticker: "SNDK", price: 1478, atr_pct: 0, status: "sparse" },
      ],
    } as any);
    render(<PortfolioHeat navColor="#6366f1" />);

    const summary = await screen.findByTestId("non-ok-summary");
    expect(summary).toHaveTextContent(/1 no data/);
    expect(summary).toHaveTextContent(/1 insufficient history/);
  });
});
