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


// ---------------------------------------------------------------------------
// Test cases — Increase / Decrease position context-menu actions.
// Both items show on every row (stock and option) and route to Log Buy
// (scale-in mode) and Log Sell respectively, with the campaign pre-selected
// via the same localStorage handshake those pages already read on mount.
// ---------------------------------------------------------------------------

describe("ActiveCampaign — Increase / Decrease position flow", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  test("right-click on STOCK row shows Increase + Decrease items", async () => {
    setupApi([stockPosition()]);
    render(<ActiveCampaign navColor="#6366f1" />);
    await waitFor(() => expect(mOpen).toHaveBeenCalled());

    const tickerEl = await screen.findByText(/^AAPL$/);
    fireEvent.contextMenu(tickerEl, { clientX: 100, clientY: 100 });

    expect(screen.getByTestId("ctx-increase-position")).toBeInTheDocument();
    expect(screen.getByTestId("ctx-decrease-position")).toBeInTheDocument();
    expect(screen.getByText("Increase position")).toBeInTheDocument();
    expect(screen.getByText("Decrease position")).toBeInTheDocument();
  });

  test("right-click on OPTION row shows Increase + Decrease items", async () => {
    setupApi([optionPosition()]);
    render(<ActiveCampaign navColor="#6366f1" />);
    await waitFor(() => expect(mOpen).toHaveBeenCalled());

    const tickerEl = await screen.findByText(/AMZN 270115 \$270C/);
    fireEvent.contextMenu(tickerEl, { clientX: 100, clientY: 100 });

    expect(screen.getByTestId("ctx-increase-position")).toBeInTheDocument();
    expect(screen.getByTestId("ctx-decrease-position")).toBeInTheDocument();
  });

  test("clicking Increase position writes ps_prefill (scale_in) and navigates to logbuy", async () => {
    const onNavigate = vi.fn();
    setupApi([optionPosition()]);
    render(<ActiveCampaign navColor="#6366f1" onNavigate={onNavigate} />);
    await waitFor(() => expect(mOpen).toHaveBeenCalled());

    const tickerEl = await screen.findByText(/AMZN 270115 \$270C/);
    fireEvent.contextMenu(tickerEl, { clientX: 100, clientY: 100 });
    fireEvent.click(screen.getByTestId("ctx-increase-position"));

    const prefill = JSON.parse(localStorage.getItem("ps_prefill") || "{}");
    expect(prefill).toEqual({ action: "scale_in", trade_id: "202604-001" });
    expect(onNavigate).toHaveBeenCalledWith("logbuy");
  });

  test("clicking Decrease position writes ps_prefill_sell and navigates to logsell", async () => {
    const onNavigate = vi.fn();
    setupApi([stockPosition()]);
    render(<ActiveCampaign navColor="#6366f1" onNavigate={onNavigate} />);
    await waitFor(() => expect(mOpen).toHaveBeenCalled());

    const tickerEl = await screen.findByText(/^AAPL$/);
    fireEvent.contextMenu(tickerEl, { clientX: 100, clientY: 100 });
    fireEvent.click(screen.getByTestId("ctx-decrease-position"));

    const prefill = JSON.parse(localStorage.getItem("ps_prefill_sell") || "{}");
    expect(prefill).toEqual({ trade_id: "202603-005" });
    expect(onNavigate).toHaveBeenCalledWith("logsell");
  });
});

// ---------------------------------------------------------------------------
// Pyramid screener column — spectrum + cheap gates (2026-07-18)
//
// The column no longer shows a binary "Ready" chip; it now surfaces
// classifyPyramidScreener output (full / prorated / blocked). These
// tests pin the four visible states (excluding location-block since
// Rule 1 stays in the sizer). Fixtures dial pyramid_pct via
// livePrices, and pos_size_pct + risk via stop and share count.
// ---------------------------------------------------------------------------

describe("ActiveCampaign — Pyramid screener column", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  test("full: current up 7.7% vs last held buy, risk within Normal budget → 🔺 Full", async () => {
    // 100 sh @ 195, stop 194 → risk = $100 (< $500 = 0.50% × 100K)
    // pos_size = 100 × 210 / 100_000 × 100 = 21% (< 25% ceiling)
    // pyramid_pct = (210 − 195) / 195 × 100 = 7.69% (≥ 5)
    setupApi([stockPosition({ avg_entry: 195, stop_loss: 194 })]);
    mBatch.mockResolvedValue({ AAPL: 210 } as any);
    render(<ActiveCampaign navColor="#6366f1" />);
    await waitFor(() => expect(mOpen).toHaveBeenCalled());
    await waitFor(() => expect(mBatch).toHaveBeenCalled());
    expect(await screen.findByTestId("acs-pyramid-full")).toHaveTextContent(/Full/);
  });

  test("prorated: current up 2.6% (between 0 and 5) → amber chip with pct", async () => {
    // Same fixture, current 200 → pyramid_pct 2.56%.
    setupApi([stockPosition({ avg_entry: 195, stop_loss: 194 })]);
    mBatch.mockResolvedValue({ AAPL: 200 } as any);
    render(<ActiveCampaign navColor="#6366f1" />);
    await waitFor(() => expect(mBatch).toHaveBeenCalled());
    const chip = await screen.findByTestId("acs-pyramid-prorated");
    expect(chip.textContent).toMatch(/\+2\.6%/);
  });

  test("blocked (ceiling): pos_size_pct ≥ 25% → ⛔ Ceiling", async () => {
    // 200 sh @ 195, current 150 → notional 30K / 100K = 30% ≥ 25
    setupApi([stockPosition({ shares: 200, avg_entry: 195, total_cost: 39000, stop_loss: 194 })]);
    mBatch.mockResolvedValue({ AAPL: 150 } as any);
    render(<ActiveCampaign navColor="#6366f1" />);
    await waitFor(() => expect(mBatch).toHaveBeenCalled());
    expect(await screen.findByTestId("acs-pyramid-blocked-ceiling")).toHaveTextContent(/Ceiling/);
  });

  test("blocked (budget): risk > 0.50% × NAV under Normal assumption → ⛔ Budget", async () => {
    // Default stockPosition: shares 100, stop 185, entry 195 → risk 100 × 10 = $1000
    // NAV 100K × 0.50% = $500 budget → 1000 > 500 → block
    // Also set current 210 so progress WOULD be full but budget wins first
    setupApi([stockPosition()]);
    mBatch.mockResolvedValue({ AAPL: 210 } as any);
    render(<ActiveCampaign navColor="#6366f1" />);
    await waitFor(() => expect(mBatch).toHaveBeenCalled());
    expect(await screen.findByTestId("acs-pyramid-blocked-budget")).toHaveTextContent(/Budget/);
  });

  test("blocked (progress): current below last held buy → ⛔ Below", async () => {
    // Tight stop so budget passes; drop current below entry → progress blocks.
    setupApi([stockPosition({ avg_entry: 195, stop_loss: 194 })]);
    mBatch.mockResolvedValue({ AAPL: 190 } as any);
    render(<ActiveCampaign navColor="#6366f1" />);
    await waitFor(() => expect(mBatch).toHaveBeenCalled());
    expect(await screen.findByTestId("acs-pyramid-blocked-progress")).toHaveTextContent(/Below/);
  });

  // Options render in a separate table without a Pyramid column at
  // all (ACS's is_option split at line ~559-560). The classifier's
  // 'n/a' branch is still covered by the pure-lib test suite;
  // there's no equivalent UI to assert on for the ACS renderer.
});
