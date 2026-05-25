import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";

vi.mock("@/lib/portfolio-context", () => ({
  usePortfolio: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  api: {
    tradesOpen: vi.fn(),
    tradesOpenDetails: vi.fn(),
    journalLatest: vi.fn(),
    batchPrices: vi.fn(),
    config: vi.fn(),
    tradeImages: vi.fn(),
  },
  getActivePortfolio: vi.fn(() => "CanSlim"),
}));

vi.mock("@/lib/log", () => ({
  log: { error: vi.fn(), warn: vi.fn(), info: vi.fn(), debug: vi.fn() },
}));

import { api } from "@/lib/api";
import { usePortfolio } from "@/lib/portfolio-context";
import { MobileTradeJournal } from "./mobile-trade-journal";
import type { LotClosure, Portfolio, TradeDetail, TradePosition } from "@/lib/api";

const CANSLIM: Portfolio = {
  id: 1,
  name: "CanSlim",
  starting_capital: 100000,
  reset_date: null,
  created_at: "2026-01-01T00:00:00Z",
  cash_balance: 0,
};

function withPortfolio(portfolio: Portfolio | null = CANSLIM) {
  vi.mocked(usePortfolio).mockReturnValue({
    portfolios: portfolio ? [portfolio] : [],
    activePortfolio: portfolio,
    loading: false,
    error: null,
    refetch: vi.fn(),
    setActive: vi.fn(),
  });
}

function tradeFixture(opts: Partial<TradePosition> = {}): TradePosition {
  return {
    trade_id: opts.trade_id ?? "202605-001",
    ticker: opts.ticker ?? "NVDA",
    status: opts.status ?? "OPEN",
    shares: opts.shares ?? 100,
    avg_entry: opts.avg_entry ?? 100,
    total_cost: opts.total_cost ?? 10000,
    realized_pl: opts.realized_pl ?? 0,
    rule: opts.rule ?? "br1.1",
    instrument_type: opts.instrument_type ?? "STOCK",
    multiplier: opts.multiplier ?? 1,
    open_date: opts.open_date ?? "2026-05-01",
    b1_entry_price: opts.b1_entry_price ?? 100,
    ...opts,
  } as TradePosition;
}

function detailFixture(opts: {
  trade_id: string;
  ticker?: string;
  action?: "BUY" | "SELL";
  date?: string;
  shares?: number;
  amount?: number;
  trx_id?: string;
}): TradeDetail {
  return {
    trade_id: opts.trade_id,
    ticker: opts.ticker ?? "NVDA",
    action: opts.action ?? "BUY",
    date: opts.date ?? "2026-05-01",
    shares: opts.shares ?? 100,
    amount: opts.amount ?? 100,
    value: (opts.shares ?? 100) * (opts.amount ?? 100),
    rule: "br1.1",
    trx_id: opts.trx_id,
  } as TradeDetail;
}

type SetMockOpts = {
  trades?: TradePosition[];
  details?: TradeDetail[];
  closures?: LotClosure[];
  endNlv?: number;
  prices?: Record<string, number>;
  pyramidRules?: { trigger_pct: number; alloc_pct: number };
  images?: unknown[];
};

function setApiMocks(opts: SetMockOpts = {}) {
  vi.mocked(api.tradesOpen).mockResolvedValue((opts.trades ?? []) as TradePosition[]);
  vi.mocked(api.tradesOpenDetails).mockResolvedValue({
    details: opts.details ?? [],
    lot_closures: opts.closures ?? [],
  });
  vi.mocked(api.journalLatest).mockResolvedValue({
    end_nlv: opts.endNlv ?? 100_000,
  } as unknown as Awaited<ReturnType<typeof api.journalLatest>>);
  vi.mocked(api.batchPrices).mockResolvedValue(
    (opts.prices ?? {}) as unknown as Awaited<ReturnType<typeof api.batchPrices>>,
  );
  vi.mocked(api.config).mockResolvedValue({
    key: "pyramid_rules",
    value: opts.pyramidRules ?? { trigger_pct: 5, alloc_pct: 20 },
  } as unknown as Awaited<ReturnType<typeof api.config>>);
  vi.mocked(api.tradeImages).mockResolvedValue(opts.images ?? []);
}

function resetApiMocks() {
  vi.mocked(api.tradesOpen).mockReset();
  vi.mocked(api.tradesOpenDetails).mockReset();
  vi.mocked(api.journalLatest).mockReset();
  vi.mocked(api.batchPrices).mockReset();
  vi.mocked(api.config).mockReset();
  vi.mocked(api.tradeImages).mockReset();
}

beforeEach(() => {
  withPortfolio();
  resetApiMocks();
  setApiMocks();
});

afterEach(() => {
  document.body.style.overflow = "";
});

describe("MobileTradeJournal — mount fetch + empty state", () => {
  test("calls all 5 mount endpoints with the active portfolio", async () => {
    setApiMocks();
    render(<MobileTradeJournal />);
    await waitFor(() => expect(api.tradesOpen).toHaveBeenCalledWith("CanSlim"));
    expect(api.tradesOpenDetails).toHaveBeenCalledWith("CanSlim");
    expect(api.journalLatest).toHaveBeenCalledWith("CanSlim");
    expect(api.config).toHaveBeenCalledWith("pyramid_rules");
  });

  test("renders the empty state when no open positions", async () => {
    setApiMocks({ trades: [] });
    render(<MobileTradeJournal />);
    await waitFor(() =>
      expect(screen.getByText(/No open positions in CanSlim/)).toBeInTheDocument(),
    );
  });

  test("renders position cards once data loads", async () => {
    setApiMocks({
      trades: [tradeFixture({ trade_id: "T1", ticker: "NVDA" })],
      details: [detailFixture({ trade_id: "T1", trx_id: "B1" })],
      prices: { NVDA: 110 },
    });
    render(<MobileTradeJournal />);
    expect(await screen.findByText("NVDA")).toBeInTheDocument();
  });
});

describe("MobileTradeJournal — filter chips + ticker search", () => {
  test("Winners chip narrows the list to positive-P&L positions only", async () => {
    setApiMocks({
      trades: [
        tradeFixture({ trade_id: "T1", ticker: "WIN", avg_entry: 100, total_cost: 10_000 }),
        tradeFixture({ trade_id: "T2", ticker: "LOS", avg_entry: 200, total_cost: 20_000 }),
      ],
      details: [
        detailFixture({ trade_id: "T1", ticker: "WIN", amount: 100, trx_id: "B1" }),
        detailFixture({ trade_id: "T2", ticker: "LOS", amount: 200, trx_id: "B1" }),
      ],
      prices: { WIN: 120, LOS: 180 }, // WIN +20%, LOS -10%
    });
    render(<MobileTradeJournal />);
    expect(await screen.findByText("WIN")).toBeInTheDocument();
    expect(screen.getByText("LOS")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Winners · 1/ }));
    expect(screen.getByText("WIN")).toBeInTheDocument();
    expect(screen.queryByText("LOS")).not.toBeInTheDocument();
  });

  test("ticker search filters live as user types", async () => {
    setApiMocks({
      trades: [
        tradeFixture({ trade_id: "T1", ticker: "NVDA" }),
        tradeFixture({ trade_id: "T2", ticker: "AAPL" }),
      ],
      details: [
        detailFixture({ trade_id: "T1", ticker: "NVDA", trx_id: "B1" }),
        detailFixture({ trade_id: "T2", ticker: "AAPL", trx_id: "B1" }),
      ],
    });
    render(<MobileTradeJournal />);
    expect(await screen.findByText("NVDA")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Search ticker"), { target: { value: "nv" } });
    expect(screen.getByText("NVDA")).toBeInTheDocument();
    expect(screen.queryByText("AAPL")).not.toBeInTheDocument();
  });

  test("chip count badges reflect the unfiltered totals", async () => {
    setApiMocks({
      trades: [
        tradeFixture({ trade_id: "T1", ticker: "WIN", instrument_type: "STOCK" }),
        tradeFixture({ trade_id: "T2", ticker: "LOS", instrument_type: "STOCK" }),
        tradeFixture({ trade_id: "T3", ticker: "OPT", instrument_type: "OPTION", multiplier: 100 }),
      ],
      details: [
        detailFixture({ trade_id: "T1", ticker: "WIN", trx_id: "B1" }),
        detailFixture({ trade_id: "T2", ticker: "LOS", amount: 200, trx_id: "B1" }),
        detailFixture({ trade_id: "T3", ticker: "OPT", amount: 1, trx_id: "B1" }),
      ],
      prices: { WIN: 110, LOS: 150, OPT: 1.5 },
    });
    render(<MobileTradeJournal />);
    expect(await screen.findByText("WIN")).toBeInTheDocument();

    expect(screen.getByRole("button", { name: /All · 3/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Options · 1/ })).toBeInTheDocument();
  });
});

describe("MobileTradeJournal — detail sheet + lazy image fetch", () => {
  test("tapping a card opens the detail sheet for that trade", async () => {
    setApiMocks({
      trades: [tradeFixture({ trade_id: "T1", ticker: "NVDA" })],
      details: [detailFixture({ trade_id: "T1", trx_id: "B1" })],
      prices: { NVDA: 110 },
    });
    render(<MobileTradeJournal />);
    const card = await screen.findByText("NVDA");
    fireEvent.click(card);
    const dialog = await screen.findByRole("dialog", { name: /NVDA details/ });
    expect(dialog).toBeInTheDocument();
    expect(within(dialog).getByText("Flight Deck")).toBeInTheDocument();
  });

  test("api.tradeImages fires on first open and reuses cache on reopen", async () => {
    setApiMocks({
      trades: [tradeFixture({ trade_id: "T1", ticker: "NVDA" })],
      details: [detailFixture({ trade_id: "T1", trx_id: "B1" })],
      prices: { NVDA: 110 },
      images: [{ id: 1, view_url: "https://r2/img.png", file_name: "chart.png" }],
    });
    render(<MobileTradeJournal />);
    await screen.findByText("NVDA");

    fireEvent.click(screen.getByText("NVDA"));
    await waitFor(() => expect(api.tradeImages).toHaveBeenCalledWith("T1"));
    expect(api.tradeImages).toHaveBeenCalledTimes(1);

    // Close + reopen — must reuse cache.
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    await waitFor(() =>
      expect(screen.queryByRole("dialog", { name: /NVDA details/ })).not.toBeInTheDocument(),
    );
    fireEvent.click(screen.getByText("NVDA"));
    await screen.findByRole("dialog", { name: /NVDA details/ });
    expect(api.tradeImages).toHaveBeenCalledTimes(1); // still 1 — cache hit
  });

  test("transactions Show-N-more expander reveals hidden rows", async () => {
    // 7 BUYs → top 5 visible, "Show 2 more" reveals the rest.
    const buys = Array.from({ length: 7 }).map((_, i) =>
      detailFixture({
        trade_id: "T1",
        date: `2026-05-${String(i + 1).padStart(2, "0")}`,
        trx_id: `B${i + 1}`,
        shares: 10,
        amount: 100 + i,
      }),
    );
    setApiMocks({
      trades: [tradeFixture({ trade_id: "T1", ticker: "NVDA", shares: 70 })],
      details: buys,
      prices: { NVDA: 110 },
    });
    render(<MobileTradeJournal />);
    await screen.findByText("NVDA");
    fireEvent.click(screen.getByText("NVDA"));

    const dialog = await screen.findByRole("dialog", { name: /NVDA details/ });
    expect(within(dialog).getByText("7 total")).toBeInTheDocument();
    const expander = within(dialog).getByRole("button", { name: /Show 2 more/ });
    fireEvent.click(expander);
    // After expand, the expander disappears.
    expect(
      within(dialog).queryByRole("button", { name: /Show 2 more/ }),
    ).not.toBeInTheDocument();
  });
});

describe("MobileTradeJournal — ?trade_id= deep-link", () => {
  test("opens the detail sheet for the matching trade and clears the prop", async () => {
    const consumed = vi.fn();
    setApiMocks({
      trades: [
        tradeFixture({ trade_id: "T1", ticker: "NVDA" }),
        tradeFixture({ trade_id: "T2", ticker: "AAPL" }),
      ],
      details: [
        detailFixture({ trade_id: "T1", ticker: "NVDA", trx_id: "B1" }),
        detailFixture({ trade_id: "T2", ticker: "AAPL", trx_id: "B1" }),
      ],
      prices: { NVDA: 110, AAPL: 200 },
    });
    render(<MobileTradeJournal initialTradeId="T2" onTradeConsumed={consumed} />);

    await screen.findByText("NVDA"); // wait for load
    const dialog = await screen.findByRole("dialog", { name: /AAPL details/ });
    expect(dialog).toBeInTheDocument();
    await waitFor(() => expect(consumed).toHaveBeenCalled());
  });

  test("silent fall-through when the deep-linked trade isn't in the list", async () => {
    const consumed = vi.fn();
    setApiMocks({
      trades: [tradeFixture({ trade_id: "T1", ticker: "NVDA" })],
      details: [detailFixture({ trade_id: "T1", trx_id: "B1" })],
      prices: { NVDA: 110 },
    });
    render(<MobileTradeJournal initialTradeId="NOTFOUND" onTradeConsumed={consumed} />);

    await screen.findByText("NVDA");
    // No sheet opens, no error UI.
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    await waitFor(() => expect(consumed).toHaveBeenCalled());
  });
});

describe("MobileTradeJournal — state pills + sell rule chip", () => {
  test("single-BUY equity renders without a state pill", async () => {
    setApiMocks({
      trades: [tradeFixture({ trade_id: "T1", ticker: "NVDA" })],
      details: [detailFixture({ trade_id: "T1", trx_id: "B1" })],
      prices: { NVDA: 110 },
    });
    render(<MobileTradeJournal />);
    expect(await screen.findByText("NVDA")).toBeInTheDocument();
    expect(screen.queryByText("READY")).not.toBeInTheDocument();
    expect(screen.queryByText("ADDED")).not.toBeInTheDocument();
    expect(screen.queryByText("CALLS")).not.toBeInTheDocument();
  });

  test("multi-BUY with last-add past pyramid trigger renders the READY pill", async () => {
    // Two BUYs: first at $100, second at $100. Current price $110 →
    // pyramid_pct = 10% > trigger 5% → READY.
    setApiMocks({
      trades: [tradeFixture({ trade_id: "T1", ticker: "NVDA", shares: 200, avg_entry: 100 })],
      details: [
        detailFixture({ trade_id: "T1", trx_id: "B1", date: "2026-05-01", amount: 100 }),
        detailFixture({ trade_id: "T1", trx_id: "A1", date: "2026-05-05", amount: 100 }),
      ],
      prices: { NVDA: 110 },
    });
    render(<MobileTradeJournal />);
    expect(await screen.findByText("READY")).toBeInTheDocument();
  });

  test("equity option renders the CALLS pill", async () => {
    setApiMocks({
      trades: [
        tradeFixture({
          trade_id: "T1",
          ticker: "AAPL 260117C00150000",
          instrument_type: "OPTION",
          multiplier: 100,
          shares: 5,
        }),
      ],
      details: [
        detailFixture({
          trade_id: "T1",
          ticker: "AAPL 260117C00150000",
          trx_id: "B1",
          shares: 5,
          amount: 1,
        }),
      ],
      prices: { "AAPL 260117C00150000": 1.5 },
    });
    render(<MobileTradeJournal />);
    expect(await screen.findByText("CALLS")).toBeInTheDocument();
  });
});
