import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";

vi.mock("@/lib/portfolio-context", () => ({
  usePortfolio: vi.fn(),
}));

// Next 16 useSearchParams — controllable per-test via setSearchParams.
let mockSearchString = "";
vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(mockSearchString),
}));

vi.mock("@/lib/api", () => ({
  api: {
    journalLatest: vi.fn(),
    rallyPrefix: vi.fn(),
    priceLookup: vi.fn(),
    tradesOpen: vi.fn(),
    tradesOpenDetails: vi.fn(),
    config: vi.fn(),
  },
  getActivePortfolio: vi.fn(() => "CanSlim"),
}));

vi.mock("@/lib/log", () => ({
  log: { error: vi.fn(), warn: vi.fn(), info: vi.fn(), debug: vi.fn() },
}));

import { api } from "@/lib/api";
import { usePortfolio } from "@/lib/portfolio-context";
import { MobilePositionSizer } from "./mobile-position-sizer";
import type { Portfolio, TradeDetail, TradePosition } from "@/lib/api";

const CANSLIM: Portfolio = {
  id: 1,
  name: "CanSlim",
  starting_capital: 100000,
  reset_date: null,
  created_at: "2026-01-01T00:00:00Z",
  cash_balance: 0,
};

function withPortfolio() {
  vi.mocked(usePortfolio).mockReturnValue({
    portfolios: [CANSLIM],
    activePortfolio: CANSLIM,
    loading: false,
    error: null,
    refetch: vi.fn(),
    setActive: vi.fn(),
  });
}

function holdingFixture(opts: {
  trade_id?: string;
  ticker?: string;
  shares?: number;
  avg_entry?: number;
  multiplier?: number;
} = {}): TradePosition {
  const shares = opts.shares ?? 100;
  const avg = opts.avg_entry ?? 50;
  return {
    trade_id: opts.trade_id ?? "T1",
    ticker: opts.ticker ?? "AAA",
    status: "OPEN",
    shares,
    avg_entry: avg,
    total_cost: shares * avg,
    realized_pl: 0,
    rule: "",
    multiplier: opts.multiplier ?? 1,
  } as TradePosition;
}

function detailFixture(opts: {
  trade_id: string;
  ticker?: string;
  date?: string;
  action?: "BUY" | "SELL";
  shares: number;
  amount: number;
}): TradeDetail {
  return {
    trade_id: opts.trade_id,
    ticker: opts.ticker ?? "AAA",
    action: opts.action ?? "BUY",
    date: opts.date ?? "2026-05-01",
    shares: opts.shares,
    amount: opts.amount,
    value: opts.shares * opts.amount,
    rule: "",
  } as TradeDetail;
}

function setApiMocks(opts: {
  endNlv?: number | null;
  state?: "POWERTREND" | "UPTREND" | "RALLY MODE" | "CORRECTION" | null;
  price?: number;
  atrPct?: number;
  priceRejects?: boolean;
  holdings?: TradePosition[];
  details?: TradeDetail[];
  pyramidRules?: { trigger_pct: number; alloc_pct: number };
  pyramidRulesRejects?: boolean;
} = {}) {
  vi.mocked(api.journalLatest).mockResolvedValue(
    opts.endNlv === null
      ? (null as any)
      : ({ end_nlv: opts.endNlv ?? 100_000 } as any),
  );
  vi.mocked(api.rallyPrefix).mockResolvedValue(
    opts.state === null ? (null as any) : ({ prefix: "", state: opts.state ?? "POWERTREND" } as any),
  );
  if (opts.priceRejects) {
    vi.mocked(api.priceLookup).mockRejectedValue(new Error("rate limit"));
  } else {
    vi.mocked(api.priceLookup).mockResolvedValue({
      ticker: "NVDA",
      price: opts.price ?? 100,
      atr: 5,
      atr_pct: opts.atrPct ?? 5,
    });
  }
  vi.mocked(api.tradesOpen).mockResolvedValue((opts.holdings ?? []) as any);
  vi.mocked(api.tradesOpenDetails).mockResolvedValue({
    details: opts.details ?? [],
    lot_closures: [],
  } as any);
  if (opts.pyramidRulesRejects) {
    vi.mocked(api.config).mockRejectedValue(new Error("config not found"));
  } else {
    vi.mocked(api.config).mockResolvedValue({
      key: "pyramid_rules",
      value: opts.pyramidRules ?? { trigger_pct: 5, alloc_pct: 20 },
    } as any);
  }
}

function resetApiMocks() {
  vi.mocked(api.journalLatest).mockReset();
  vi.mocked(api.rallyPrefix).mockReset();
  vi.mocked(api.priceLookup).mockReset();
  vi.mocked(api.tradesOpen).mockReset();
  vi.mocked(api.tradesOpenDetails).mockReset();
  vi.mocked(api.config).mockReset();
  mockSearchString = "";
}

describe("MobilePositionSizer — portfolio wiring", () => {
  beforeEach(() => {
    withPortfolio();
    resetApiMocks();
    setApiMocks();
  });

  test("renders 'Sizing for {portfolio name}' when a portfolio is active", async () => {
    render(<MobilePositionSizer />);
    expect(await screen.findByText(/Sizing for/)).toBeInTheDocument();
    expect(screen.getByText("CanSlim")).toBeInTheDocument();
  });

  test("omits the subtitle when no portfolio is active", async () => {
    vi.mocked(usePortfolio).mockReturnValue({
      portfolios: [],
      activePortfolio: null,
      loading: false,
      error: null,
      refetch: vi.fn(),
      setActive: vi.fn(),
    });
    render(<MobilePositionSizer />);
    expect(screen.queryByText(/Sizing for/)).not.toBeInTheDocument();
  });
});

describe("MobilePositionSizer — tab switcher", () => {
  beforeEach(() => {
    withPortfolio();
    resetApiMocks();
    setApiMocks();
  });

  test("renders 5 tabs in desktop order", () => {
    render(<MobilePositionSizer />);
    const tabs = screen.getAllByRole("tab");
    expect(tabs).toHaveLength(5);
    expect(tabs[0]).toHaveTextContent("Sizer");
    expect(tabs[1]).toHaveTextContent("Scale In");
    expect(tabs[2]).toHaveTextContent("Pyramid");
    expect(tabs[3]).toHaveTextContent("Trim");
    expect(tabs[4]).toHaveTextContent("Options");
  });

  test("Volatility is the default active tab", () => {
    render(<MobilePositionSizer />);
    const tab = screen.getByRole("tab", { name: /Sizer/ });
    expect(tab).toHaveAttribute("aria-selected", "true");
  });

  test("switching tabs preserves input state (entry persists across Volatility → Scale-In → Volatility)", async () => {
    setApiMocks({ holdings: [holdingFixture()] });
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.journalLatest).toHaveBeenCalled());

    // Type into entry on Volatility.
    fireEvent.change(screen.getByLabelText("Entry price"), { target: { value: "42.42" } });
    expect((screen.getByLabelText("Entry price") as HTMLInputElement).value).toBe("42.42");

    // Switch to Scale-In — the entry input there reads from same state.
    fireEvent.click(screen.getByRole("tab", { name: /Scale In/ }));
    expect((screen.getByLabelText("Current price") as HTMLInputElement).value).toBe("42.42");

    // Back to Volatility — value still there.
    fireEvent.click(screen.getByRole("tab", { name: /Sizer/ }));
    expect((screen.getByLabelText("Entry price") as HTMLInputElement).value).toBe("42.42");
  });

  test("?tab=scalein URL param activates Scale-In on mount", async () => {
    mockSearchString = "tab=scalein";
    setApiMocks();
    render(<MobilePositionSizer />);
    const tab = await screen.findByRole("tab", { name: /Scale In/ });
    expect(tab).toHaveAttribute("aria-selected", "true");
  });

  test("?tab=garbage falls back to Volatility", async () => {
    mockSearchString = "tab=garbage";
    setApiMocks();
    render(<MobilePositionSizer />);
    const tab = screen.getByRole("tab", { name: /Sizer/ });
    expect(tab).toHaveAttribute("aria-selected", "true");
  });

  test.each([
    ["Trim", "trim"],
    ["Options", "options"],
  ])("renders Coming-Soon placeholder for %s tab", async (label, _key) => {
    render(<MobilePositionSizer />);
    fireEvent.click(screen.getByRole("tab", { name: new RegExp(label) }));
    expect(await screen.findByText("Coming soon")).toBeInTheDocument();
  });
});

describe("MobilePositionSizer — mount fetch", () => {
  beforeEach(() => {
    withPortfolio();
    resetApiMocks();
  });

  test("populates NLV from journalLatest end_nlv after mount", async () => {
    setApiMocks({ endNlv: 487_704 });
    render(<MobilePositionSizer />);
    await waitFor(() => expect(screen.getByText("$487,704")).toBeInTheDocument());
  });

  test("auto-selects Offense when MCT state is POWERTREND", async () => {
    setApiMocks({ endNlv: 100_000, state: "POWERTREND" });
    render(<MobilePositionSizer />);
    const modeTrigger = await screen.findByRole("button", { name: /Mode: Offense/ });
    expect(modeTrigger).toBeInTheDocument();
  });

  test("auto-selects Defense when MCT state is CORRECTION", async () => {
    setApiMocks({ endNlv: 100_000, state: "CORRECTION" });
    render(<MobilePositionSizer />);
    const modeTrigger = await screen.findByRole("button", { name: /Mode: Defense/ });
    expect(modeTrigger).toBeInTheDocument();
  });

  test("falls back to Normal mode when rallyPrefix returns null", async () => {
    setApiMocks({ endNlv: 100_000, state: null });
    render(<MobilePositionSizer />);
    const modeTrigger = await screen.findByRole("button", { name: /Mode: Normal/ });
    expect(modeTrigger).toBeInTheDocument();
  });
});

describe("MobilePositionSizer — debounced priceLookup", () => {
  beforeEach(() => {
    withPortfolio();
    resetApiMocks();
    setApiMocks({ endNlv: 100_000, price: 185.5, atrPct: 5 });
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  test("fires priceLookup 600ms after the ticker changes and auto-fills entry + ATR", async () => {
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.journalLatest).toHaveBeenCalled());

    const tickerInput = screen.getByLabelText("Ticker symbol") as HTMLInputElement;
    fireEvent.change(tickerInput, { target: { value: "nvda" } });
    expect(tickerInput.value).toBe("NVDA");

    expect(api.priceLookup).not.toHaveBeenCalled();

    await act(async () => {
      vi.advanceTimersByTime(700);
    });

    await waitFor(() => expect(api.priceLookup).toHaveBeenCalledWith("NVDA"));

    const entryInput = await screen.findByLabelText("Entry price") as HTMLInputElement;
    expect(entryInput.value).toBe("185.5");
    expect(screen.getByText("5.0%")).toBeInTheDocument();
  });

  test("on priceLookup error, surfaces 'Couldn\\'t fetch price' and keeps inputs editable", async () => {
    setApiMocks({ endNlv: 100_000, priceRejects: true });
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.journalLatest).toHaveBeenCalled());

    const tickerInput = screen.getByLabelText("Ticker symbol") as HTMLInputElement;
    fireEvent.change(tickerInput, { target: { value: "ZZZZ" } });

    await act(async () => {
      vi.advanceTimersByTime(700);
    });

    await waitFor(() => expect(screen.getByText(/Couldn't fetch price/)).toBeInTheDocument());

    const entryInput = screen.getByLabelText("Entry price") as HTMLInputElement;
    expect(entryInput.value).toBe("");
    expect(entryInput).not.toBeDisabled();
  });
});

describe("MobilePositionSizer — Volatility live compute", () => {
  beforeEach(() => {
    withPortfolio();
    resetApiMocks();
    setApiMocks({ endNlv: 100_000, state: "POWERTREND", price: 100, atrPct: 5 });
  });

  test("audit card shows '—' before any inputs are entered", async () => {
    render(<MobilePositionSizer />);
    await waitFor(() => expect(screen.getByText("$100,000")).toBeInTheDocument());
    expect(screen.getByTestId("audit-shares")).toHaveTextContent("—");
  });

  test("computes via shared vol-sizer lib when entry + ATR + MA + NLV are present", async () => {
    render(<MobilePositionSizer />);
    await waitFor(() => expect(screen.getByText("$100,000")).toBeInTheDocument());

    fireEvent.change(screen.getByLabelText("Entry price"), { target: { value: "100" } });
    fireEvent.change(screen.getByLabelText("Key MA level"), { target: { value: "95" } });

    const ticker = screen.getByLabelText("Ticker symbol");
    fireEvent.change(ticker, { target: { value: "X" } });
    await waitFor(() => expect(api.priceLookup).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByText("5.0%")).toBeInTheDocument());

    const sharesEl = await screen.findByTestId("audit-shares");
    await waitFor(() => expect(sharesEl).toHaveTextContent("100"), { timeout: 2000 });

    const verdict = screen.getByTestId("verdict-card");
    expect(verdict).toHaveTextContent(/Sized by tech stop/);
    expect(verdict).toHaveTextContent(/position-size tier/);

    expect(screen.getByTestId("scenario-tech-stop")).toBeInTheDocument();
    expect(screen.getByTestId("scenario-1x-atr")).toBeInTheDocument();
    expect(screen.getByTestId("scenario-1.5x-atr")).toBeInTheDocument();
    expect(screen.getByTestId("scenario-2x-atr")).toBeInTheDocument();

    const pills = screen.getAllByTestId("recommended-pill");
    expect(pills).toHaveLength(1);
    expect(screen.getByTestId("scenario-tech-stop")).toContainElement(pills[0]);
  });

  test("flips recommendation to 1.5× ATR when tech stop sits inside 1 ATR (warning visible)", async () => {
    render(<MobilePositionSizer />);
    await waitFor(() => expect(screen.getByText("$100,000")).toBeInTheDocument());

    fireEvent.change(screen.getByLabelText("Entry price"), { target: { value: "100" } });
    fireEvent.change(screen.getByLabelText("Key MA level"), { target: { value: "99" } });

    const ticker = screen.getByLabelText("Ticker symbol");
    fireEvent.change(ticker, { target: { value: "Y" } });
    await waitFor(() => expect(screen.getByText("5.0%")).toBeInTheDocument());

    const warning = await screen.findByTestId("vol-warning");
    expect(warning).toHaveTextContent(/ATR/);

    const verdict = screen.getByTestId("verdict-card");
    expect(verdict).toHaveTextContent(/Sized by 1.5× ATR cushion/);

    const pills = screen.getAllByTestId("recommended-pill");
    expect(pills).toHaveLength(1);
    expect(screen.getByTestId("scenario-1.5x-atr")).toContainElement(pills[0]);
  });

  test("calculated-stop banner annotates with ATR fraction", async () => {
    render(<MobilePositionSizer />);
    await waitFor(() => expect(screen.getByText("$100,000")).toBeInTheDocument());

    fireEvent.change(screen.getByLabelText("Entry price"), { target: { value: "100" } });
    fireEvent.change(screen.getByLabelText("Key MA level"), { target: { value: "95" } });

    const ticker = screen.getByLabelText("Ticker symbol");
    fireEvent.change(ticker, { target: { value: "Z" } });
    await waitFor(() => expect(screen.getByText("5.0%")).toBeInTheDocument());

    const banner = await screen.findByTestId("calc-stop-banner");
    expect(banner).toHaveTextContent(/1\.19× ATR/);
  });
});

describe("MobilePositionSizer — Mode picker manual override", () => {
  beforeEach(() => {
    withPortfolio();
    resetApiMocks();
    setApiMocks({ endNlv: 100_000, state: "POWERTREND" });
  });

  test("tapping a Mode option closes the sheet and updates the trigger", async () => {
    render(<MobilePositionSizer />);
    const trigger = await screen.findByRole("button", { name: /Mode: Offense/ });
    fireEvent.click(trigger);

    const dialog = await screen.findByRole("dialog", { name: /Sizing mode/ });
    expect(dialog).toBeInTheDocument();

    const defenseOption = within(dialog).getByRole("option", { name: /Defense/ });
    fireEvent.click(defenseOption);

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(await screen.findByRole("button", { name: /Mode: Defense/ })).toBeInTheDocument();
  });
});

describe("MobilePositionSizer — Size picker", () => {
  beforeEach(() => {
    withPortfolio();
    resetApiMocks();
    setApiMocks({ endNlv: 100_000 });
  });

  test("Size picker exposes all 8 desktop SIZE_OPTIONS", async () => {
    render(<MobilePositionSizer />);
    const trigger = await screen.findByRole("button", { name: /Size: Full/ });
    fireEvent.click(trigger);

    const dialog = await screen.findByRole("dialog", { name: /Target size/ });
    for (const label of ["Starter", "Half", "Standard", "Full", "Overweight", "Core", "Core+", "Max"]) {
      expect(within(dialog).getByText(label)).toBeInTheDocument();
    }

    fireEvent.click(within(dialog).getByText("Starter"));
    expect(await screen.findByRole("button", { name: /Size: Starter/ })).toBeInTheDocument();
  });
});

describe("MobileHoldingPicker — sourced from tradesOpen", () => {
  beforeEach(() => {
    withPortfolio();
    resetApiMocks();
  });

  test("renders an empty trigger with 'No open trades' when tradesOpen returns []", async () => {
    setApiMocks({ holdings: [] });
    render(<MobilePositionSizer />);
    fireEvent.click(screen.getByRole("tab", { name: /Scale In/ }));
    expect(await screen.findByText("No open trades")).toBeInTheDocument();
  });

  test("opens a sheet listing each holding and selects on tap", async () => {
    setApiMocks({
      holdings: [
        holdingFixture({ trade_id: "T1", ticker: "AAA", shares: 100, avg_entry: 50 }),
        holdingFixture({ trade_id: "T2", ticker: "BBB", shares: 200, avg_entry: 25 }),
      ],
    });
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.tradesOpen).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("tab", { name: /Scale In/ }));
    const trigger = await screen.findByRole("button", { name: /Holding:/ });
    fireEvent.click(trigger);

    const dialog = await screen.findByRole("dialog", { name: /Select holding/ });
    expect(within(dialog).getByText("AAA")).toBeInTheDocument();
    expect(within(dialog).getByText("BBB")).toBeInTheDocument();

    fireEvent.click(within(dialog).getByText("AAA"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    // priceLookup fires inline on select
    await waitFor(() => expect(api.priceLookup).toHaveBeenCalledWith("AAA"));
    expect(await screen.findByRole("button", { name: /Holding: AAA/ })).toBeInTheDocument();
  });
});

describe("MobilePositionSizer — Scale-In math", () => {
  beforeEach(() => {
    withPortfolio();
    resetApiMocks();
  });

  test("risk-free position renders the risk-free banner and success verdict", async () => {
    setApiMocks({
      endNlv: 100_000,
      state: "POWERTREND",
      holdings: [holdingFixture({ ticker: "AAA", shares: 100, avg_entry: 50 })],
    });
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.tradesOpen).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("tab", { name: /Scale In/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Holding:/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("AAA"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    fireEvent.change(screen.getByLabelText("Current price"), { target: { value: "60" } });
    fireEvent.change(screen.getByLabelText("Key MA level"), { target: { value: "55" } });
    // stop = 55 * 0.99 = 54.45 > avg_entry 50 → risk-free

    expect(await screen.findByTestId("scale-risk-free-banner")).toBeInTheDocument();

    // targetTotalShares = ceil(100000*0.10/60) = 167; targetAdd = 67
    // newAddRiskPerShare = 60 - 54.45 = 5.55
    // maxRiskDol = 1000 (Offense); affordableAdd = floor(1000/5.55) = 180
    // recommendedAdd = min(67, 180) = 67 → success
    expect(screen.getByText("+67")).toBeInTheDocument();
    expect(screen.getByTestId("scale-verdict-success")).toBeInTheDocument();
  });

  test("partial verdict when affordable add caps below the target add", async () => {
    // Setup: avg_entry above stop → existing risk eats most of budget
    setApiMocks({
      endNlv: 100_000,
      state: "POWERTREND", // Offense → maxRisk 1.0 → maxRiskDol 1000
      holdings: [holdingFixture({ ticker: "BBB", shares: 100, avg_entry: 60 })],
    });
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.tradesOpen).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("tab", { name: /Scale In/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Holding:/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("BBB"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    fireEvent.change(screen.getByLabelText("Current price"), { target: { value: "60" } });
    fireEvent.change(screen.getByLabelText("Key MA level"), { target: { value: "55" } });
    // existingRiskPerShare = max(0, 60-54.45) = 5.55; existingRisk = 555
    // remainingBudget = 1000 - 555 = 445; affordable = floor(445/5.55) = 80

    // Bump target to Overweight (12.5%) → targetAdd = ceil(12500/60) - 100 = 109
    // affordable 80 < targetAdd 109 → partial
    fireEvent.click(screen.getByRole("button", { name: /Size: Full/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("Overweight"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    expect(await screen.findByTestId("scale-verdict-partial")).toBeInTheDocument();
    expect(screen.getByText("+80")).toBeInTheDocument();
  });

  test("target-exceeded error when current value is already above target weight", async () => {
    setApiMocks({
      endNlv: 100_000,
      state: "POWERTREND",
      holdings: [holdingFixture({ ticker: "CCC", shares: 100, avg_entry: 50 })],
    });
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.tradesOpen).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("tab", { name: /Scale In/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Holding:/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("CCC"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    fireEvent.change(screen.getByLabelText("Current price"), { target: { value: "60" } });
    fireEvent.change(screen.getByLabelText("Key MA level"), { target: { value: "55" } });

    // Switch to Starter (2.5%) — targetTotalShares = ceil(2500/60) = 42; targetAdd = -58
    fireEvent.click(screen.getByRole("button", { name: /Size: Full/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("Starter"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    expect(await screen.findByTestId("scale-error-banner")).toBeInTheDocument();
    expect(screen.getByTestId("scale-error-banner")).toHaveTextContent(/already at or above the target weight/);
  });

  test("no-budget error when existing risk exceeds the budget", async () => {
    setApiMocks({
      endNlv: 100_000,
      state: "CORRECTION", // Defense → maxRisk 0.5 → maxRiskDol 500
      holdings: [holdingFixture({ ticker: "DDD", shares: 100, avg_entry: 60 })],
    });
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.tradesOpen).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("tab", { name: /Scale In/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Holding:/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("DDD"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    fireEvent.change(screen.getByLabelText("Current price"), { target: { value: "60" } });
    fireEvent.change(screen.getByLabelText("Key MA level"), { target: { value: "55" } });
    // existingRisk = 100 * (60-54.45) = 555 > maxRiskDol 500 → error path

    expect(await screen.findByTestId("scale-error-banner")).toBeInTheDocument();
    expect(screen.getByTestId("scale-error-banner")).toHaveTextContent(/NO ADD/);
  });
});

// ── Pyramid tab (PR2) ─────────────────────────────────────────────

describe("MobilePositionSizer — Pyramid mount fetch", () => {
  beforeEach(() => {
    withPortfolio();
    resetApiMocks();
  });

  test("fetches tradesOpenDetails + pyramid_rules config on mount", async () => {
    setApiMocks();
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.tradesOpenDetails).toHaveBeenCalledWith("CanSlim"));
    expect(api.config).toHaveBeenCalledWith("pyramid_rules");
  });

  test("uses fallback {trigger 5, alloc 20} when config fetch rejects", async () => {
    setApiMocks({ pyramidRulesRejects: true });
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.config).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("tab", { name: /Pyramid/ }));
    fireEvent.click(await screen.findByTestId("pyramid-rules-summary"));
    // Default copy mentions "20%" and "5%"
    const summaryContent = await screen.findByText(/Each add is capped at/i);
    expect(summaryContent.textContent).toMatch(/20%/);
  });
});

describe("MobilePositionSizer — Pyramid Rules expander", () => {
  beforeEach(() => {
    withPortfolio();
    resetApiMocks();
    setApiMocks({ pyramidRules: { trigger_pct: 7, alloc_pct: 25 } });
  });

  test("renders the expander with config-driven copy after mount", async () => {
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.config).toHaveBeenCalled());
    fireEvent.click(screen.getByRole("tab", { name: /Pyramid/ }));

    const summary = await screen.findByTestId("pyramid-rules-summary");
    expect(summary).toHaveTextContent(/View Pyramid Rules/);

    // Expand and confirm config values flow through
    fireEvent.click(summary);
    const firstRule = await screen.findByText(/Each add is capped at/i);
    expect(firstRule.textContent).toMatch(/25%/);
    const secondRule = screen.getByText(/Your last buy must be up/i);
    expect(secondRule.textContent).toMatch(/7%/);
  });
});

describe("MobilePositionSizer — Pyramid math", () => {
  beforeEach(() => {
    withPortfolio();
    resetApiMocks();
  });

  test("success verdict: 100 shs @ avg 50, entry 60 → ADD 20 shares (Pyramid pace)", async () => {
    // Cushion = (60-50)/50*100 = 20% → Tier 1 (tolPct 1.0, atrMult 2.0)
    // baseAddPct = 0.20, threshold = 5; lastBuyProfitPct = 20% ≥ 5 → scaleFactor 1.0
    // pyramidMaxShares = ceil(100 * 0.20 * 1.0) = 20
    // maxSharesCap = floor(100000*0.20/60) = 333; positionCeiling = min(666, 333) = 333
    // roomToAdd = 333 - 100 = 233; pyramidAllowed = min(20, 233) = 20 → success "Pyramid pace"
    setApiMocks({
      endNlv: 100_000,
      atrPct: 5,
      holdings: [holdingFixture({ trade_id: "T1", ticker: "AAA", shares: 100, avg_entry: 50 })],
      details: [detailFixture({ trade_id: "T1", ticker: "AAA", shares: 100, amount: 50 })],
    });
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.tradesOpenDetails).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("tab", { name: /Pyramid/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Holding:/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("AAA"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    // priceLookup auto-filled entry to 100 (default mock). Override + ATR.
    fireEvent.change(screen.getByLabelText("Current price"), { target: { value: "60" } });
    fireEvent.change(screen.getByLabelText("ATR percent"), { target: { value: "5" } });

    const verdict = await screen.findByTestId("pyramid-verdict-success");
    expect(verdict).toHaveTextContent(/ADD 20 shares/);
    expect(verdict).toHaveTextContent(/Pyramid pace/);
  });

  test("error verdict: lastBuy at 65 vs entry 60 → scaleFactor 0 → NO ADD (down)", async () => {
    setApiMocks({
      endNlv: 100_000,
      atrPct: 5,
      holdings: [holdingFixture({ trade_id: "T2", ticker: "BBB", shares: 100, avg_entry: 65 })],
      details: [detailFixture({ trade_id: "T2", ticker: "BBB", shares: 100, amount: 65 })],
    });
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.tradesOpenDetails).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("tab", { name: /Pyramid/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Holding:/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("BBB"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    fireEvent.change(screen.getByLabelText("Current price"), { target: { value: "60" } });
    fireEvent.change(screen.getByLabelText("ATR percent"), { target: { value: "5" } });

    const verdict = await screen.findByTestId("pyramid-verdict-error");
    expect(verdict).toHaveTextContent(/NO ADD/);
    expect(verdict).toHaveTextContent(/down/);
  });

  test("warning verdict: position already at hard-cap ceiling → NO ROOM", async () => {
    // Position 400 shares; equity 100k, entry 60 → maxSharesCap = 333
    // positionCeiling ≤ 333 < 400 → roomToAdd = 0
    // baseAddPct 0.20 × 400 × scaleFactor 1.0 → pyramidMaxShares = 80
    // pyramidAllowed = min(80, 0) = 0 → warning path
    setApiMocks({
      endNlv: 100_000,
      atrPct: 5,
      holdings: [holdingFixture({ trade_id: "T3", ticker: "CCC", shares: 400, avg_entry: 50 })],
      details: [detailFixture({ trade_id: "T3", ticker: "CCC", shares: 400, amount: 50 })],
    });
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.tradesOpenDetails).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("tab", { name: /Pyramid/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Holding:/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("CCC"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    fireEvent.change(screen.getByLabelText("Current price"), { target: { value: "60" } });
    fireEvent.change(screen.getByLabelText("ATR percent"), { target: { value: "5" } });

    const verdict = await screen.findByTestId("pyramid-verdict-warning");
    expect(verdict).toHaveTextContent(/NO ROOM/);
    expect(verdict).toHaveTextContent(/Pyramid says 80 shares/);
  });

  test.each([
    { entry: 60, expected: "Tier 1 (High Cushion)" }, // cushion = 20% → Tier 1
    { entry: 59.99, expected: "Tier 2 (Moderate)" },  // cushion = 19.98% → Tier 2
    { entry: 52.5, expected: "Tier 2 (Moderate)" },   // cushion = 5% → Tier 2
    { entry: 52.4, expected: "Tier 3 (Defense)" },    // cushion = 4.8% → Tier 3
  ])("tier boundary: entry $entry → $expected", async ({ entry, expected }) => {
    setApiMocks({
      endNlv: 100_000,
      atrPct: 5,
      holdings: [holdingFixture({ trade_id: "T4", ticker: "DDD", shares: 100, avg_entry: 50 })],
      details: [detailFixture({ trade_id: "T4", ticker: "DDD", shares: 100, amount: 50 })],
    });
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.tradesOpenDetails).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("tab", { name: /Pyramid/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Holding:/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("DDD"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    fireEvent.change(screen.getByLabelText("Current price"), { target: { value: String(entry) } });
    fireEvent.change(screen.getByLabelText("ATR percent"), { target: { value: "5" } });

    // Position Ceiling card sub-text includes the tier name. Walk to the
    // card root (rounded-m-md class) so we capture both the label row and
    // the sub line below it.
    await waitFor(() => {
      const positionCeiling = screen.getByText("Position Ceiling");
      const card = positionCeiling.closest("div.rounded-m-md") as HTMLElement | null;
      expect(card?.textContent ?? "").toContain(expected);
    });
  });
});
