import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach } from "vitest";

// Recharts (used inside ACS for the option timeline) needs ResizeObserver
// under jsdom. Same stub as dashboard.test.tsx.
class ResizeObserverStub {
  observe(): void { /* noop */ }
  unobserve(): void { /* noop */ }
  disconnect(): void { /* noop */ }
}
(globalThis as any).ResizeObserver = ResizeObserverStub;

// ACS reads window.localStorage directly (multiplier-fix banner dismissal).
// jsdom's localStorage shim is sometimes absent; install a minimal in-memory
// stand-in so the component mounts cleanly.
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

vi.mock("@/lib/portfolio-context", () => ({
  usePortfolio: () => ({
    activePortfolio: { id: 1, name: "CanSlim" },
    portfolios: [{ id: 1, name: "CanSlim" }],
    loading: false,
    error: null,
    refetch: vi.fn(),
    setActive: vi.fn(),
  }),
}));

// Stub session-cache so the component doesn't read/write localStorage
// shapes from previous test runs.
vi.mock("@/lib/session-cache", () => ({
  readCache: () => null,
  writeCache: () => {},
}));

vi.mock("@/lib/api", () => ({
  api: {
    tradesOpen: vi.fn(),
    tradesOpenDetails: vi.fn(),
    portfolioNlv: vi.fn(),
    journalLatest: vi.fn(),
    batchPrices: vi.fn(),
    setManualPrice: vi.fn(),
    exerciseOption: vi.fn(),
    listStrategies: vi.fn().mockResolvedValue([
      { name: "CanSlim", description: null, color: "#6366f1", is_active: true, created_at: "2026-01-01" },
      { name: "StockTalk", description: null, color: "#d97706", is_active: true, created_at: "2026-01-02" },
    ]),
    setTradeStrategy: vi.fn().mockResolvedValue({ ok: true }),
  },
  getActivePortfolio: () => "CanSlim",
}));

vi.mock("./capture-snapshot", () => ({
  CaptureSnapshotButton: () => null,
}));

import { api } from "@/lib/api";
import { ActiveCampaign } from "./active-campaign";

const mOpen = vi.mocked(api.tradesOpen);
const mOpenDetails = vi.mocked(api.tradesOpenDetails);
const mNlv = vi.mocked(api.portfolioNlv);
const mLatest = vi.mocked(api.journalLatest);
const mBatch = vi.mocked(api.batchPrices);
const mExercise = vi.mocked(api.exerciseOption);


// ---------------------------------------------------------------------------
// Fixtures — minimal shape so the component renders without errors. Each
// position needs the fields runLifoEngine + computeEnrichedPositions read.
// ---------------------------------------------------------------------------


function optionPosition(overrides: Partial<any> = {}) {
  return {
    trade_id: "202604-001",
    ticker: "AMZN 270115 $270C",
    shares: 2,
    avg_entry: 35.63,
    total_cost: 7126,
    realized_pl: 0,
    rule: "Breakout",
    buy_notes: "",
    notes: "",
    open_date: "2026-04-01",
    stop_loss: 0,
    instrument_type: "OPTION",
    multiplier: 100,
    risk_budget: 7126,
    ...overrides,
  };
}

function stockPosition(overrides: Partial<any> = {}) {
  return {
    trade_id: "202603-005",
    ticker: "AAPL",
    shares: 100,
    avg_entry: 195.0,
    total_cost: 19500,
    realized_pl: 0,
    rule: "RS leader",
    buy_notes: "",
    notes: "",
    open_date: "2026-03-15",
    stop_loss: 185.0,
    instrument_type: "STOCK",
    multiplier: 1,
    risk_budget: 1000,
    ...overrides,
  };
}

function setupApi(positions: any[]) {
  mOpen.mockResolvedValue(positions as any);
  // tradesOpenDetails returns one BUY detail per position so runLifoEngine
  // can compute. The shape mirrors the real backend bundle.
  const details = positions.map(p => ({
    trade_id: p.trade_id, ticker: p.ticker, action: "BUY",
    date: `${p.open_date} 09:30:00`, shares: p.shares, amount: p.avg_entry,
    value: p.total_cost, trx_id: "B1", stop_loss: p.stop_loss,
    rule: "", notes: "",
  }));
  mOpenDetails.mockResolvedValue({ details, lot_closures: [] } as any);
  mNlv.mockResolvedValue({
    cash: 100000, market_value: 0, nlv: 100000, positions: [], as_of: "2026-05-01",
  } as any);
  mLatest.mockResolvedValue({
    end_nlv: 100000, beg_nlv: 100000, day: "2026-05-01",
    daily_dollar_change: 0, daily_pct_change: 0, pct_invested: 0,
    market_window: "POWERTREND", portfolio_heat: 0,
  } as any);
  mBatch.mockResolvedValue({} as any);
}


// ---------------------------------------------------------------------------
// Test cases — exercise-option context menu + modal
// ---------------------------------------------------------------------------


describe("ActiveCampaign — Exercise option flow", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  test("right-click on OPTION row shows the Exercise option menu item", async () => {
    setupApi([optionPosition()]);
    render(<ActiveCampaign navColor="#6366f1" />);

    // Wait for the option row to render. The component renders option tickers
    // with their full readable string somewhere; we just need to wait for the
    // initial load to settle.
    await waitFor(() => expect(mOpen).toHaveBeenCalled());

    // Locate the option row's contextmenu target via its ticker text.
    const tickerEl = await screen.findByText(/AMZN 270115 \$270C/);
    fireEvent.contextMenu(tickerEl, { clientX: 100, clientY: 100 });

    expect(screen.getByTestId("ctx-exercise-option")).toBeInTheDocument();
    expect(screen.getByText("Exercise option")).toBeInTheDocument();
  });

  test("right-click on STOCK row hides the Exercise option menu item", async () => {
    setupApi([stockPosition()]);
    render(<ActiveCampaign navColor="#6366f1" />);

    await waitFor(() => expect(mOpen).toHaveBeenCalled());

    const tickerEl = await screen.findByText(/^AAPL$/);
    fireEvent.contextMenu(tickerEl, { clientX: 100, clientY: 100 });

    // Original two items still appear; Exercise option is omitted.
    expect(screen.getByText("View in Journal")).toBeInTheDocument();
    expect(screen.queryByTestId("ctx-exercise-option")).not.toBeInTheDocument();
  });

  test("clicking Exercise option opens the modal with computed preview math", async () => {
    setupApi([optionPosition()]);
    render(<ActiveCampaign navColor="#6366f1" />);
    await waitFor(() => expect(mOpen).toHaveBeenCalled());

    const tickerEl = await screen.findByText(/AMZN 270115 \$270C/);
    fireEvent.contextMenu(tickerEl, { clientX: 100, clientY: 100 });
    fireEvent.click(screen.getByTestId("ctx-exercise-option"));

    expect(screen.getByTestId("exercise-modal")).toBeInTheDocument();

    const preview = screen.getByTestId("exercise-preview");
    // 2 contracts → 200 AMZN shares at strike $270 + premium $35.63 = $305.63
    expect(preview).toHaveTextContent("2 contracts");
    expect(preview).toHaveTextContent("200 AMZN");
    expect(preview).toHaveTextContent("$305.63");
    // 200 × 305.63 = 61,126
    expect(preview).toHaveTextContent("$61,126");
  });

  test("date input defaults to today and accepts user changes", async () => {
    setupApi([optionPosition()]);
    render(<ActiveCampaign navColor="#6366f1" />);
    await waitFor(() => expect(mOpen).toHaveBeenCalled());

    const tickerEl = await screen.findByText(/AMZN 270115 \$270C/);
    fireEvent.contextMenu(tickerEl, { clientX: 100, clientY: 100 });
    fireEvent.click(screen.getByTestId("ctx-exercise-option"));

    const dateInput = screen.getByTestId("exercise-date") as HTMLInputElement;
    const today = new Date().toISOString().slice(0, 10);
    expect(dateInput.value).toBe(today);

    fireEvent.change(dateInput, { target: { value: "2026-06-15" } });
    expect(dateInput.value).toBe("2026-06-15");
  });

  test("Confirm posts to api.exerciseOption with the right body", async () => {
    setupApi([optionPosition()]);
    mExercise.mockResolvedValue({
      status: "ok",
      option_trade_id: "202604-001",
      stock_trade_id: "202605-007",
      stock_was_new: true,
      contracts_exercised: 2, shares_acquired: 200, stock_entry_price: 305.63,
    });

    render(<ActiveCampaign navColor="#6366f1" />);
    await waitFor(() => expect(mOpen).toHaveBeenCalled());

    const tickerEl = await screen.findByText(/AMZN 270115 \$270C/);
    fireEvent.contextMenu(tickerEl, { clientX: 100, clientY: 100 });
    fireEvent.click(screen.getByTestId("ctx-exercise-option"));

    fireEvent.change(screen.getByTestId("exercise-date"), {
      target: { value: "2026-05-01" },
    });
    fireEvent.change(screen.getByTestId("exercise-notes"), {
      target: { value: "Exercising before expiry" },
    });
    fireEvent.click(screen.getByTestId("exercise-confirm"));

    await waitFor(() => expect(mExercise).toHaveBeenCalledTimes(1));
    expect(mExercise).toHaveBeenCalledWith({
      trade_id: "202604-001",
      date: "2026-05-01",
      notes: "Exercising before expiry",
    });

    // On success the modal closes.
    await waitFor(() => {
      expect(screen.queryByTestId("exercise-modal")).not.toBeInTheDocument();
    });
  });

  test("backend error shows in the modal without closing it", async () => {
    setupApi([optionPosition()]);
    mExercise.mockResolvedValue({ error: "Trade is not open" });

    render(<ActiveCampaign navColor="#6366f1" />);
    await waitFor(() => expect(mOpen).toHaveBeenCalled());

    const tickerEl = await screen.findByText(/AMZN 270115 \$270C/);
    fireEvent.contextMenu(tickerEl, { clientX: 100, clientY: 100 });
    fireEvent.click(screen.getByTestId("ctx-exercise-option"));
    fireEvent.click(screen.getByTestId("exercise-confirm"));

    const errEl = await screen.findByTestId("exercise-error");
    expect(errEl).toHaveTextContent("Trade is not open");
    // Modal must stay open so user can adjust + retry.
    expect(screen.getByTestId("exercise-modal")).toBeInTheDocument();
  });

  test("Cancel button dismisses the modal", async () => {
    setupApi([optionPosition()]);
    render(<ActiveCampaign navColor="#6366f1" />);
    await waitFor(() => expect(mOpen).toHaveBeenCalled());

    const tickerEl = await screen.findByText(/AMZN 270115 \$270C/);
    fireEvent.contextMenu(tickerEl, { clientX: 100, clientY: 100 });
    fireEvent.click(screen.getByTestId("ctx-exercise-option"));

    expect(screen.getByTestId("exercise-modal")).toBeInTheDocument();
    fireEvent.click(screen.getByTestId("exercise-cancel"));
    expect(screen.queryByTestId("exercise-modal")).not.toBeInTheDocument();
    // No API call was made.
    expect(mExercise).not.toHaveBeenCalled();
  });

  test("ESC dismisses the modal when not saving", async () => {
    setupApi([optionPosition()]);
    render(<ActiveCampaign navColor="#6366f1" />);
    await waitFor(() => expect(mOpen).toHaveBeenCalled());

    const tickerEl = await screen.findByText(/AMZN 270115 \$270C/);
    fireEvent.contextMenu(tickerEl, { clientX: 100, clientY: 100 });
    fireEvent.click(screen.getByTestId("ctx-exercise-option"));

    expect(screen.getByTestId("exercise-modal")).toBeInTheDocument();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByTestId("exercise-modal")).not.toBeInTheDocument();
  });
});


// ─────────────────────────────────────────────────────────────────────
// Phase 2 — Right-click → "Set strategy" flyout submenu.
// ─────────────────────────────────────────────────────────────────────

describe("ActiveCampaign — Set strategy submenu", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // listStrategies + setTradeStrategy fall through to their default
    // resolved values from the module-level mock unless a test overrides.
    vi.mocked(api.listStrategies).mockResolvedValue([
      { name: "CanSlim", description: null, color: "#6366f1", is_active: true, created_at: "2026-01-01" },
      { name: "StockTalk", description: null, color: "#d97706", is_active: true, created_at: "2026-01-02" },
    ] as any);
    vi.mocked(api.setTradeStrategy).mockResolvedValue({ ok: true } as any);
  });

  test("right-click → hover Set strategy → click StockTalk fires PATCH", async () => {
    setupApi([stockPosition()]);
    render(<ActiveCampaign navColor="#6366f1" />);
    await waitFor(() => expect(api.listStrategies).toHaveBeenCalled());

    const tickerEl = await screen.findByText(/AAPL/);
    fireEvent.contextMenu(tickerEl, { clientX: 100, clientY: 100 });

    // Hover the "Set strategy" parent to reveal the flyout.
    const setStrategyBtn = await screen.findByText(/Set strategy/);
    fireEvent.mouseEnter(setStrategyBtn.closest("div")!);
    const flyout = await screen.findByTestId("strategy-flyout");
    expect(flyout).toBeInTheDocument();

    // Regression guard: flyout must use position: fixed so it escapes
    // the parent context menu's overflow-hidden clip. Reverting to
    // position: absolute would silently break the panel in real
    // browsers (jsdom doesn't compute layout, so this className check
    // is the only thing standing between a regression and an invisible
    // bug shipping again). Asserting on className rather than
    // getComputedStyle because jsdom doesn't apply Tailwind classes
    // into computed style — but we DO control the class string.
    expect(flyout.className).toContain("fixed");
    expect(flyout.className).not.toContain("absolute");

    // Click StockTalk option.
    fireEvent.click(screen.getByText("StockTalk"));

    await waitFor(() => expect(api.setTradeStrategy).toHaveBeenCalled());
    const [tradeId, body] = vi.mocked(api.setTradeStrategy).mock.calls[0];
    expect(tradeId).toBe("202603-005");
    expect(body.strategy).toBe("StockTalk");
  });

  test("listStrategies fires on mount (contract guard for fetch wiring)", async () => {
    // Even though the actual flyout-clipping bug wasn't a fetch issue,
    // making the fetch wiring an explicit test contract prevents a
    // regression where someone removes the useEffect and the empty
    // strategies array silently hides the menu item.
    setupApi([stockPosition()]);
    render(<ActiveCampaign navColor="#6366f1" />);
    await waitFor(() => expect(api.listStrategies).toHaveBeenCalled());
  });
});
