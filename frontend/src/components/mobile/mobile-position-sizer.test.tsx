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
  /** Per-lot stop from trades_details.stop_loss — feeds Pyramid's
   *  Rule 3 risk accounting. Optional; defaults to 0 for callers that
   *  don't care about pyramid math. */
  stop_loss?: number;
  /** Transaction ID (B1, A1, ...) — shown in the pyramid held-lots
   *  table so the user can identify which lot is contributing risk. */
  trx_id?: string;
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
    stop_loss: opts.stop_loss ?? 0,
    trx_id: opts.trx_id ?? "",
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
      ema_21: null,
      sma_50: null,
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

  test("Coming-Soon placeholder is gone from every tab (PR4 closes the arc)", async () => {
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.journalLatest).toHaveBeenCalled());
    for (const tabName of [/Sizer/, /Scale In/, /Pyramid/, /Trim/, /Options/]) {
      fireEvent.click(screen.getByRole("tab", { name: tabName }));
      expect(screen.queryByText("Coming soon")).not.toBeInTheDocument();
    }
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

  test("auto-selects Pilot when MCT state is CORRECTION (retiered from Defense)", async () => {
    setApiMocks({ endNlv: 100_000, state: "CORRECTION" });
    render(<MobilePositionSizer />);
    const modeTrigger = await screen.findByRole("button", { name: /Mode: Pilot/ });
    expect(modeTrigger).toBeInTheDocument();
  });

  test("falls back to Pilot mode when rallyPrefix returns null (retiered from Normal)", async () => {
    // Post-retier default is Pilot (safest floor). Old default was
    // Normal (safe middle) — that middle no longer exists in the
    // three-tier lineup and a middle-tier default on engine hiccup
    // would violate the "when in doubt, be smaller" invariant.
    setApiMocks({ endNlv: 100_000, state: null });
    render(<MobilePositionSizer />);
    const modeTrigger = await screen.findByRole("button", { name: /Mode: Pilot/ });
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

describe("MobilePositionSizer — Volatility composite-stop model", () => {
  beforeEach(() => {
    withPortfolio();
    resetApiMocks();
    // NLV $400K, UPTREND → Normal (0.50%) mode — matches the DELL/COHR
    // canonical examples in the design conversation.
    setApiMocks({ endNlv: 400_000, state: "UPTREND", price: 100, atrPct: 5 });
  });

  test("answer card shows '—' before any inputs are entered", async () => {
    render(<MobilePositionSizer />);
    await waitFor(() => expect(screen.getByText("$400,000")).toBeInTheDocument());
    expect(screen.getByTestId("audit-shares")).toHaveTextContent("—");
  });

  test("legacy 4-tile UI is gone: no Tech Stop / 1x / 1.5x / 2x ATR cards", async () => {
    render(<MobilePositionSizer />);
    await waitFor(() => expect(screen.getByText("$400,000")).toBeInTheDocument());
    expect(screen.queryByTestId("scenario-tech-stop")).not.toBeInTheDocument();
    expect(screen.queryByTestId("scenario-1x-atr")).not.toBeInTheDocument();
    expect(screen.queryByTestId("scenario-1.5x-atr")).not.toBeInTheDocument();
    expect(screen.queryByTestId("scenario-2x-atr")).not.toBeInTheDocument();
    // No Size trigger on the volatility tab either — ceiling is policy-driven.
    expect(screen.queryByRole("button", { name: /Size: /i })).not.toBeInTheDocument();
  });

  test("DELL canonical case → composite lands on Key Level, risk-bound", async () => {
    // Seed the priceLookup mock with the DELL numbers so the auto-fill
    // populates entry=176.21 / ATR=4.5% for us. The user only needs to
    // type Key Level, which the sizer doesn't fetch from anywhere.
    setApiMocks({ endNlv: 400_000, state: "UPTREND", price: 176.21, atrPct: 4.5 });
    render(<MobilePositionSizer />);
    await waitFor(() => expect(screen.getByText("$400,000")).toBeInTheDocument());

    const ticker = screen.getByLabelText("Ticker symbol");
    fireEvent.change(ticker, { target: { value: "DELL" } });
    await waitFor(() => expect(api.priceLookup).toHaveBeenCalled());
    // Wait for the auto-fill to hydrate the read-only ATR display.
    await waitFor(() => expect(screen.getByText("4.5%")).toBeInTheDocument());
    // The read-only ATR display + entry input picked up the fetched
    // values. Now the user only types Key Level (no auto-source).
    fireEvent.change(screen.getByLabelText("Key level"), { target: { value: "171.365" } });

    // atrPerShare (entry-based, 1-ATR floor) = 176.21 × 4.5% = 7.929
    // atrFloor = 176.21 − 7.929 = 168.28
    // buffer (KL-based) = max(0.5×4.5%×171.365, 171.365×1%) = max(3.856, 1.714) = 3.856
    // keyLevelBuffer = 171.365 − 3.856 = 167.51
    // composite = min(168.28, 167.51) = 167.51 → distance 8.70
    // raw = floor(2000/8.70) = 229; ceiling shares = 340; final = 229.
    const sharesEl = await screen.findByTestId("audit-shares");
    await waitFor(() => expect(sharesEl.textContent).toMatch(/229/), { timeout: 2000 });

    expect(screen.getByTestId("bind-badge").textContent).toBe("Risk-bound");
    // Composite subtitle mentions Key Level winner.
    expect(screen.getByTestId("composite-answer").textContent).toMatch(/Key Level/);
  });

  test("Young-IPO checkbox clamps ceiling to 5% → answer becomes ceiling-bound", async () => {
    setApiMocks({ endNlv: 400_000, state: "UPTREND", price: 176.21, atrPct: 4.5 });
    render(<MobilePositionSizer />);
    await waitFor(() => expect(screen.getByText("$400,000")).toBeInTheDocument());

    const ticker = screen.getByLabelText("Ticker symbol");
    fireEvent.change(ticker, { target: { value: "IPO" } });
    await waitFor(() => expect(api.priceLookup).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByText("4.5%")).toBeInTheDocument());
    fireEvent.change(screen.getByLabelText("Key level"), { target: { value: "171.365" } });
    // Toggle Young IPO clamp.
    fireEvent.click(screen.getByTestId("young-ipo-checkbox"));

    // 5% × 400K / 176.21 = 113 shs (floored). Ceiling now binds.
    const sharesEl = await screen.findByTestId("audit-shares");
    await waitFor(() => expect(sharesEl.textContent).toMatch(/113/), { timeout: 2000 });
    expect(screen.getByTestId("bind-badge").textContent).toBe("Ceiling-bound");
  });

  test("21 EMA / 50 SMA cells render when priceLookup returns non-null values; Use → paste into Key Level", async () => {
    // Override the default mock to include real MA levels for one call.
    setApiMocks({ endNlv: 400_000, state: "UPTREND", price: 176.21, atrPct: 4.5 });
    vi.mocked(api.priceLookup).mockResolvedValue({
      ticker: "DELL",
      price: 176.21,
      atr: 7.93,
      atr_pct: 4.5,
      ema_21: 172.34,
      sma_50: 165.20,
    } as any);
    render(<MobilePositionSizer />);
    await waitFor(() => expect(screen.getByText("$400,000")).toBeInTheDocument());

    const ticker = screen.getByLabelText("Ticker symbol");
    fireEvent.change(ticker, { target: { value: "DELL" } });
    await waitFor(() => expect(api.priceLookup).toHaveBeenCalled());

    const emaCell = await screen.findByTestId("ema21-cell", {}, { timeout: 2000 });
    expect(emaCell.textContent).toMatch(/172\.34/);
    expect(screen.getByTestId("sma50-cell").textContent).toMatch(/165\.20/);

    fireEvent.click(screen.getByTestId("use-sma50-btn"));
    const keyLevelInput = screen.getByLabelText("Key level") as HTMLInputElement;
    expect(keyLevelInput.value).toBe("165.2");
  });

  test("scale-out ladder renders 3 legs at −0.5 / −1.0 / −1.5 ATR (not old locked percentages)", async () => {
    setApiMocks({ endNlv: 400_000, state: "UPTREND", price: 176.21, atrPct: 4.5 });
    render(<MobilePositionSizer />);
    await waitFor(() => expect(screen.getByText("$400,000")).toBeInTheDocument());

    const ticker = screen.getByLabelText("Ticker symbol");
    fireEvent.change(ticker, { target: { value: "DELL" } });
    await waitFor(() => expect(api.priceLookup).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByText("4.5%")).toBeInTheDocument());
    fireEvent.change(screen.getByLabelText("Key level"), { target: { value: "171.365" } });

    const ladder = await screen.findByTestId("scale-out-stops");
    await waitFor(() => expect(ladder.textContent).toMatch(/−0\.50 ATR/));
    expect(ladder.textContent).toMatch(/−1\.00 ATR/);
    expect(ladder.textContent).toMatch(/−1\.50 ATR/);
    expect(ladder.textContent).not.toMatch(/−3%|−5%|−7%/);
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

    // Post-retier: Defense retired. Manual pick target is Pilot (idx 0,
    // most conservative), which is now the same position-in-the-sheet
    // for "downshift to the safest tier" as Defense used to be.
    const pilotOption = within(dialog).getByRole("option", { name: /Pilot/ });
    fireEvent.click(pilotOption);

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(await screen.findByRole("button", { name: /Mode: Pilot/ })).toBeInTheDocument();
  });
});

describe("MobilePositionSizer — Size picker", () => {
  beforeEach(() => {
    withPortfolio();
    resetApiMocks();
    setApiMocks({ endNlv: 100_000 });
  });

  test("Size picker exposes all 8 desktop SIZE_OPTIONS (Scale-In tab)", async () => {
    // Volatility tab no longer has a size picker — the ceiling is
    // policy-driven (15% standard / 5% young-IPO). Scale-In still uses
    // the 8-tier ladder for target-weight sizing.
    setApiMocks({ endNlv: 100_000, holdings: [holdingFixture()] });
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.tradesOpen).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("tab", { name: /Scale In/ }));

    const trigger = await screen.findByRole("button", { name: /Size: Full/ });
    fireEvent.click(trigger);

    const dialog = await screen.findByRole("dialog", { name: /Target size/ });
    for (const label of ["Shotgun", "Half", "Standard", "Full", "Overweight", "Core", "Core+", "Max"]) {
      expect(within(dialog).getByText(label)).toBeInTheDocument();
    }

    fireEvent.click(within(dialog).getByText("Shotgun"));
    expect(await screen.findByRole("button", { name: /Size: Shotgun/ })).toBeInTheDocument();
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
    // maxRiskDol = 750 (Offense post-retier 0.75%); affordableAdd = floor(750/5.55) = 135
    // recommendedAdd = min(67, 135) = 67 → success (target still binds)
    expect(screen.getByText("+67")).toBeInTheDocument();
    expect(screen.getByTestId("scale-verdict-success")).toBeInTheDocument();
  });

  test("partial verdict when affordable add caps below the target add", async () => {
    // Setup: avg_entry above stop → existing risk eats most of budget
    setApiMocks({
      endNlv: 100_000,
      state: "POWERTREND", // Offense → maxRisk 0.75 → maxRiskDol 750 (post-retier)
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
    // remainingBudget = 750 - 555 = 195; affordable = floor(195/5.55) = 35

    // Bump target to Overweight (12.5%) → targetAdd = ceil(12500/60) - 100 = 109
    // affordable 35 < targetAdd 109 → partial
    fireEvent.click(screen.getByRole("button", { name: /Size: Full/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("Overweight"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    expect(await screen.findByTestId("scale-verdict-partial")).toBeInTheDocument();
    expect(screen.getByText("+35")).toBeInTheDocument();
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

    // Switch to Shotgun (2.5%) — targetTotalShares = ceil(2500/60) = 42; targetAdd = -58
    fireEvent.click(screen.getByRole("button", { name: /Size: Full/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("Shotgun"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    expect(await screen.findByTestId("scale-error-banner")).toBeInTheDocument();
    expect(screen.getByTestId("scale-error-banner")).toHaveTextContent(/already at or above the target weight/);
  });

  test("no-budget error when existing risk exceeds the budget", async () => {
    setApiMocks({
      endNlv: 100_000,
      state: "CORRECTION", // Pilot → maxRisk 0.25 → maxRiskDol 250 (post-retier)
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
    // existingRisk = 100 * (60-54.45) = 555 > maxRiskDol 250 → error path

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

  test("renders the six-rule expander even when config fetch rejects (rules no longer config-driven)", async () => {
    // Post-2026-07-18: pyramid rules are constants in @/lib/pyramid-sizer.
    // The config fetch stays for backward-compat with any consumer, but
    // it no longer influences the expander copy.
    setApiMocks({ pyramidRulesRejects: true });
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.config).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("tab", { name: /Pyramid/ }));
    fireEvent.click(await screen.findByTestId("pyramid-rules-summary"));
    expect(await screen.findByText(/Total notional ≤ 25% NAV/)).toBeInTheDocument();
  });
});

describe("MobilePositionSizer — Pyramid Rules expander (six-rule composite model)", () => {
  beforeEach(() => {
    withPortfolio();
    resetApiMocks();
    setApiMocks();
  });

  test("renders the four-gate summary in the expander", async () => {
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.tradesOpenDetails).toHaveBeenCalled());
    fireEvent.click(screen.getByRole("tab", { name: /Pyramid/ }));

    const summary = await screen.findByTestId("pyramid-rules-summary");
    expect(summary).toHaveTextContent(/View Pyramid Rules/);

    fireEvent.click(summary);
    // Four gates: location / progress / budget / ceiling.
    expect(await screen.findByText(/Price ≤ 21 EMA/)).toBeInTheDocument();
    expect(screen.getByText(/Last held buy up ≥ 5%/)).toBeInTheDocument();
    expect(screen.getByText(/Budget = Mode/)).toBeInTheDocument();
    expect(screen.getByText(/Total notional ≤ 25% NAV/)).toBeInTheDocument();
  });
});

describe("MobilePositionSizer — Pyramid math (composite-stop model)", () => {
  beforeEach(() => {
    withPortfolio();
    resetApiMocks();
  });

  test("happy path: profitable name with Key Level anchor → answer card + broker callout", async () => {
    // Fixture: NAV $400K, DELL-ish. 100sh @ 150 stopped at 148 (small
    // per-lot risk), adding at 180. Key Level = 175, ATR 4.5%. Tight
    // stops → positive headroom → sizer produces a real add.
    setApiMocks({
      endNlv: 400_000,
      state: "UPTREND",
      price: 180,
      atrPct: 4.5,
      holdings: [holdingFixture({ trade_id: "T1", ticker: "DELL", shares: 100, avg_entry: 150 })],
      details: [
        detailFixture({ trade_id: "T1", ticker: "DELL", shares: 100, amount: 150, stop_loss: 148, trx_id: "B1" }),
      ],
    });
    vi.mocked(api.priceLookup).mockResolvedValue({
      ticker: "DELL", price: 180, atr: 8.1, atr_pct: 4.5, ema_21: 175, sma_50: 165,
    } as any);
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.tradesOpenDetails).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("tab", { name: /Pyramid/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Holding:/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("DELL"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    await waitFor(() => expect(api.priceLookup).toHaveBeenCalledWith("DELL"));

    // Type Key Level to trigger the sizer.
    fireEvent.change(screen.getByLabelText("Key level"), { target: { value: "175" } });

    // Answer + broker callout + held-lots table all render on success.
    await screen.findByTestId("pyramid-answer");
    expect(await screen.findByTestId("pyramid-broker-callout")).toBeInTheDocument();
    expect(screen.getByTestId("pyramid-held-lots")).toBeInTheDocument();
    expect(screen.getByTestId("pyramid-lot-row-0")).toHaveTextContent(/B1/);
  });

  test("blocked: extended above 21 EMA + 1 ATR → NO ADD with location reason", async () => {
    setApiMocks({
      endNlv: 400_000,
      state: "UPTREND",
      price: 200,
      atrPct: 4.5,
      holdings: [holdingFixture({ trade_id: "T2", ticker: "AAA", shares: 100, avg_entry: 150 })],
      details: [detailFixture({ trade_id: "T2", ticker: "AAA", shares: 100, amount: 150, stop_loss: 145, trx_id: "B1" })],
    });
    vi.mocked(api.priceLookup).mockResolvedValue({
      // Entry 200, ATR = 9 → ceiling = 175 + 9 = 184 → 200 > 184 → BLOCK.
      ticker: "AAA", price: 200, atr: 9, atr_pct: 4.5, ema_21: 175, sma_50: 170,
    } as any);
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.tradesOpenDetails).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("tab", { name: /Pyramid/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Holding:/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("AAA"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    await waitFor(() => expect(api.priceLookup).toHaveBeenCalled());

    fireEvent.change(screen.getByLabelText("Key level"), { target: { value: "175" } });

    const blocked = await screen.findByTestId("pyramid-blocked");
    expect(blocked).toHaveTextContent(/No add/i);
    expect(blocked).toHaveTextContent(/Extended/);
    // Broker callout absent on blocked adds.
    expect(screen.queryByTestId("pyramid-broker-callout")).not.toBeInTheDocument();
  });

  test("blocked: budget exhausted by held-lot risk → NO ADD with headroom reason", async () => {
    // NAV $400K × Normal 0.50% = $2000 budget. Held lot with $3000 risk
    // (100 sh × $30 open-risk) → headroom −$1000.
    setApiMocks({
      endNlv: 400_000,
      state: "UPTREND",
      price: 180,
      atrPct: 4.5,
      holdings: [holdingFixture({ trade_id: "T3", ticker: "BBB", shares: 100, avg_entry: 150 })],
      details: [detailFixture({ trade_id: "T3", ticker: "BBB", shares: 100, amount: 150, stop_loss: 120, trx_id: "B1" })],
    });
    vi.mocked(api.priceLookup).mockResolvedValue({
      ticker: "BBB", price: 180, atr: 8.1, atr_pct: 4.5, ema_21: 175, sma_50: 165,
    } as any);
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.tradesOpenDetails).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("tab", { name: /Pyramid/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Holding:/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("BBB"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    await waitFor(() => expect(api.priceLookup).toHaveBeenCalled());

    fireEvent.change(screen.getByLabelText("Key level"), { target: { value: "175" } });

    const blocked = await screen.findByTestId("pyramid-blocked");
    expect(blocked).toHaveTextContent(/No add/i);
    expect(blocked).toHaveTextContent(/No headroom/);
  });
});

// ── Trim tab (PR3) ───────────────────────────────────────────────

describe("MobilePositionSizer — Trim tab gating", () => {
  beforeEach(() => {
    withPortfolio();
    resetApiMocks();
    setApiMocks();
  });

  test("Trim tab activates via tap", async () => {
    render(<MobilePositionSizer />);
    fireEvent.click(screen.getByRole("tab", { name: /Trim/ }));
    expect(await screen.findByRole("tab", { name: /Trim/ })).toHaveAttribute("aria-selected", "true");
    // Coming-Soon copy must NOT appear on Trim anymore.
    expect(screen.queryByText("Coming soon")).not.toBeInTheDocument();
  });

  test("?tab=trim URL param activates Trim on mount", async () => {
    mockSearchString = "tab=trim";
    setApiMocks();
    render(<MobilePositionSizer />);
    const tab = await screen.findByRole("tab", { name: /Trim/ });
    expect(tab).toHaveAttribute("aria-selected", "true");
  });
});

describe("MobilePositionSizer — Trim math", () => {
  beforeEach(() => {
    withPortfolio();
    resetApiMocks();
  });

  test("no-trim-needed: target weight >= current weight → warning banner", async () => {
    // shares=50 @ entry=50, equity=100k → currWeight 2.5%. Default size=Full
    // (10%) > 2.5% → trigger the early-bail warning.
    setApiMocks({
      endNlv: 100_000,
      holdings: [holdingFixture({ trade_id: "T1", ticker: "AAA", shares: 50, avg_entry: 50 })],
      details: [detailFixture({ trade_id: "T1", ticker: "AAA", shares: 50, amount: 50 })],
    });
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.tradesOpen).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("tab", { name: /Trim/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Holding:/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("AAA"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    fireEvent.change(screen.getByLabelText("Current price"), { target: { value: "50" } });

    const banner = await screen.findByTestId("trim-no-trim-needed");
    expect(banner).toHaveTextContent(/Target \(10%\) is higher than Current \(2\.5%\)/);
    expect(banner).toHaveTextContent(/No trim needed/);
  });

  test("profit case: entry above avg cost → success Profit Lock verdict", async () => {
    // shares=100 @ avg=50, entry=100 → currWeight 10%. Target 5% (Half).
    // valueToSell = 5000 → sharesToSell = 50.
    // LIFO walk over single lot {qty:100, price:50}: take 50 → accumulatedCost = 50*50 = 2500.
    // cashGenerated = 50*100 = 5000; lifoPnl = +2500.
    setApiMocks({
      endNlv: 100_000,
      holdings: [holdingFixture({ trade_id: "T2", ticker: "BBB", shares: 100, avg_entry: 50 })],
      details: [detailFixture({ trade_id: "T2", ticker: "BBB", shares: 100, amount: 50 })],
    });
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.tradesOpen).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("tab", { name: /Trim/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Holding:/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("BBB"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    fireEvent.change(screen.getByLabelText("Current price"), { target: { value: "100" } });

    // Switch target size from Full (10%) to Half (5%).
    fireEvent.click(screen.getByRole("button", { name: /Size: Full/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("Half"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    const verdict = await screen.findByTestId("trim-verdict-success");
    expect(verdict).toHaveTextContent(/Profit Lock/);
    expect(verdict).toHaveTextContent(/\$2,500 profit/);
  });

  test("loss case: entry below avg cost → warning realizes-loss verdict", async () => {
    // shares=100 @ avg=100, entry=50 → currWeight 5%. Target 2.5% (Shotgun).
    // valueToSell = 2500 → sharesToSell = 50.
    // LIFO walk: take 50 from {qty:100, price:100} → accumulatedCost = 5000.
    // cashGenerated = 50*50 = 2500; lifoPnl = -2500.
    setApiMocks({
      endNlv: 100_000,
      holdings: [holdingFixture({ trade_id: "T3", ticker: "CCC", shares: 100, avg_entry: 100 })],
      details: [detailFixture({ trade_id: "T3", ticker: "CCC", shares: 100, amount: 100 })],
    });
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.tradesOpen).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("tab", { name: /Trim/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Holding:/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("CCC"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    fireEvent.change(screen.getByLabelText("Current price"), { target: { value: "50" } });

    fireEvent.click(screen.getByRole("button", { name: /Size: Full/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("Shotgun"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    const verdict = await screen.findByTestId("trim-verdict-loss");
    expect(verdict).toHaveTextContent(/realizes a loss of \$2,500/);
    expect(verdict).toHaveTextContent(/LIFO/);
  });

  test("multi-lot LIFO attribution: 3 lots → trim takes from newest first", async () => {
    // 3 lots ascending date: BUY 30@50, BUY 40@60, BUY 30@70.
    // Total shares 100; LIFO inventory order [{30,50},{40,60},{30,70}].
    // Trim 50 shares: take 30 from {30,70} (2100) + 20 from {40,60} (1200) = 3300.
    // Setup: equity=100k, entry=100, currWeight=10%, target=5% → sharesToSell=50.
    // cashGenerated=5000, lifoPnl = 5000 - 3300 = +1700.
    setApiMocks({
      endNlv: 100_000,
      holdings: [holdingFixture({ trade_id: "T4", ticker: "DDD", shares: 100, avg_entry: 60 })],
      details: [
        detailFixture({ trade_id: "T4", ticker: "DDD", date: "2026-04-01", shares: 30, amount: 50 }),
        detailFixture({ trade_id: "T4", ticker: "DDD", date: "2026-04-15", shares: 40, amount: 60 }),
        detailFixture({ trade_id: "T4", ticker: "DDD", date: "2026-05-01", shares: 30, amount: 70 }),
      ],
    });
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.tradesOpen).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("tab", { name: /Trim/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Holding:/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("DDD"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    fireEvent.change(screen.getByLabelText("Current price"), { target: { value: "100" } });
    fireEvent.click(screen.getByRole("button", { name: /Size: Full/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("Half"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    const verdict = await screen.findByTestId("trim-verdict-success");
    expect(verdict).toHaveTextContent(/\$1,700 profit/);

    // Cost Basis (Sold) card should reflect 3300.
    const costBasisLabel = screen.getByText("Cost Basis (Sold)");
    const card = costBasisLabel.closest("div.rounded-m-md") as HTMLElement | null;
    expect(card?.textContent ?? "").toContain("$3,300");
  });

  test("shortfall fallback: sharesToSell exceeds inventory → falls back to avg_entry", async () => {
    // Holding shows 100 shares @ avg=50 but details only contain a BUY of
    // 60 shares (data inconsistency). LIFO inventory = 60 shares total.
    // Trim 95 shares (cap target small to force big trim):
    //   equity 10k, entry 100, currVal 10000, currWeight 100%, target 5% (Half)
    //   targetVal 500, valueToSell 9500, sharesToSell ceil(9500/100) = 95
    //   LIFO walk takes 60 from inventory @50 → accumulatedCost 3000
    //   Shortfall 35 shares fallback at avg_entry 50 → +35*50 = 1750
    //   Total accumulatedCost = 4750
    //   cashGenerated = 95*100 = 9500; lifoPnl = +4750
    setApiMocks({
      endNlv: 10_000,
      holdings: [holdingFixture({ trade_id: "T5", ticker: "EEE", shares: 100, avg_entry: 50 })],
      details: [detailFixture({ trade_id: "T5", ticker: "EEE", shares: 60, amount: 50 })],
    });
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.tradesOpen).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("tab", { name: /Trim/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Holding:/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("EEE"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    fireEvent.change(screen.getByLabelText("Current price"), { target: { value: "100" } });
    fireEvent.click(screen.getByRole("button", { name: /Size: Full/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("Half"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    const verdict = await screen.findByTestId("trim-verdict-success");
    expect(verdict).toHaveTextContent(/\$4,750 profit/);
  });
});

// ── Options tab (PR4 — closes the arc) ───────────────────────────

describe("MobilePositionSizer — Options tab gating", () => {
  beforeEach(() => {
    withPortfolio();
    resetApiMocks();
    setApiMocks();
  });

  test("Options tab activates via tap (no Coming-soon placeholder)", async () => {
    render(<MobilePositionSizer />);
    fireEvent.click(screen.getByRole("tab", { name: /Options/ }));
    expect(await screen.findByRole("tab", { name: /Options/ })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(screen.queryByText("Coming soon")).not.toBeInTheDocument();
    // Method picker is always present on the Options panel.
    expect(screen.getByRole("button", { name: /Method: Risk/ })).toBeInTheDocument();
  });

  test("?tab=options URL param activates Options on mount", async () => {
    mockSearchString = "tab=options";
    setApiMocks();
    render(<MobilePositionSizer />);
    const tab = await screen.findByRole("tab", { name: /Options/ });
    expect(tab).toHaveAttribute("aria-selected", "true");
  });

  test("Method defaults to Risk; Ticker / Stock Price / Size picker are hidden", async () => {
    render(<MobilePositionSizer />);
    fireEvent.click(screen.getByRole("tab", { name: /Options/ }));
    expect(await screen.findByRole("button", { name: /Method: Risk/ })).toBeInTheDocument();
    // Mode picker IS visible (Risk mode drives the recommendation).
    expect(screen.getByRole("button", { name: /Mode:/ })).toBeInTheDocument();
    // Equivalent-only inputs are hidden.
    expect(screen.queryByLabelText("Stock price")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Ticker symbol")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Size: / })).not.toBeInTheDocument();
  });

  test("Switching method to Equivalent reveals Ticker + Stock Price + Size picker", async () => {
    render(<MobilePositionSizer />);
    fireEvent.click(screen.getByRole("tab", { name: /Options/ }));

    fireEvent.click(await screen.findByRole("button", { name: /Method: Risk/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("Equivalent"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    expect(screen.getByLabelText("Stock price")).toBeInTheDocument();
    expect(screen.getByLabelText("Ticker symbol")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Size:/ })).toBeInTheDocument();
    // Mode picker hidden in Equivalent mode (recommendation is targetSize-driven).
    expect(screen.queryByRole("button", { name: /Mode:/ })).not.toBeInTheDocument();
  });
});

describe("MobilePositionSizer — Options Risk-mode math", () => {
  beforeEach(() => {
    withPortfolio();
    resetApiMocks();
  });

  test("5 tier rows compute deterministically: equity=100k, cpc=$1 → 10 / 20 / 30 / 40 / 50 contracts", async () => {
    setApiMocks({ endNlv: 100_000, state: "POWERTREND" });
    render(<MobilePositionSizer />);
    fireEvent.click(screen.getByRole("tab", { name: /Options/ }));
    await waitFor(() => expect(api.journalLatest).toHaveBeenCalled());

    // Default cpc is "1.00" → $100/contract. Equity defaults to 100k.
    // Conservative 1% = $1,000 budget → 10 contracts
    // Normal       2% = $2,000 budget → 20 contracts
    // Aggressive   3% = $3,000 budget → 30 contracts
    // Heavy        4% = $4,000 budget → 40 contracts
    // Max          5% = $5,000 budget → 50 contracts (equals hard cap)
    const tier1 = await screen.findByTestId("options-risk-tier-1");
    expect(tier1).toHaveTextContent(/Conservative \(1%\)/);
    expect(tier1).toHaveTextContent(/10 contracts/);
    expect(tier1).toHaveTextContent(/Budget \$1,000/);

    const tier2 = screen.getByTestId("options-risk-tier-2");
    expect(tier2).toHaveTextContent(/Normal \(2%\)/);
    expect(tier2).toHaveTextContent(/20 contracts/);

    const tier3 = screen.getByTestId("options-risk-tier-3");
    expect(tier3).toHaveTextContent(/Aggressive \(3%\)/);
    expect(tier3).toHaveTextContent(/30 contracts/);

    const tier4 = screen.getByTestId("options-risk-tier-4");
    expect(tier4).toHaveTextContent(/Heavy \(4%\)/);
    expect(tier4).toHaveTextContent(/40 contracts/);
    expect(tier4).toHaveTextContent(/Budget \$4,000/);

    const tier5 = screen.getByTestId("options-risk-tier-5");
    expect(tier5).toHaveTextContent(/Max \(5%\)/);
    expect(tier5).toHaveTextContent(/50 contracts/);
    expect(tier5).toHaveTextContent(/Budget \$5,000/);

    // Hard-cap footer note must render.
    expect(screen.getByText(/Hard cap: 5% of NLV \(\$5,000\)/)).toBeInTheDocument();
  });

  test("Recommended card reflects sizingMode: Offense (0.75%) → 7 contracts, Pilot (0.25%) → 2 contracts", async () => {
    // Post-retier: Offense dropped to 0.75%; Defense retired; Pilot is
    // now the manual downshift target.
    setApiMocks({ endNlv: 100_000, state: "POWERTREND" }); // → Offense (0.75%)
    render(<MobilePositionSizer />);
    fireEvent.click(screen.getByRole("tab", { name: /Options/ }));
    await waitFor(() => expect(api.journalLatest).toHaveBeenCalled());

    const readRecCard = () => {
      const label = screen.getByText("Recommended");
      return (label.closest("div.rounded-m-md") as HTMLElement | null)?.textContent ?? "";
    };

    // Offense (0.75%) → $750 budget → floor(750/100) = 7 contracts
    await waitFor(() => expect(readRecCard()).toMatch(/7 contracts/));
    expect(readRecCard()).toMatch(/Risk Budget/);

    // Switch to Pilot via Mode picker → 0.25% → $250 budget → 2 contracts
    fireEvent.click(screen.getByRole("button", { name: /Mode: Offense/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("Pilot"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    await waitFor(() => expect(readRecCard()).toMatch(/2 contracts/));
  });

  test("Single contract exceeds budget → warning banner with desktop wording", async () => {
    // Post-retier: CORRECTION → Pilot 0.25% → $250 budget.
    // cpc=$6 → $600/contract. 600 > 250. floor(250/600) = 0 contracts.
    // Warning fires with the new smaller budget.
    setApiMocks({ endNlv: 100_000, state: "CORRECTION" }); // → Pilot (post-retier)
    render(<MobilePositionSizer />);
    fireEvent.click(screen.getByRole("tab", { name: /Options/ }));
    await waitFor(() => expect(api.journalLatest).toHaveBeenCalled());

    fireEvent.change(screen.getByLabelText("Cost per contract"), { target: { value: "6.00" } });

    const banner = await screen.findByTestId("options-risk-warning");
    expect(banner).toHaveTextContent(/A single contract \(\$600\)/);
    expect(banner).toHaveTextContent(/exceeds your risk budget \(\$250\)/);
    expect(banner).toHaveTextContent(/Consider a cheaper strike or spread/);
  });
});

describe("MobilePositionSizer — Options Equivalent-mode math", () => {
  beforeEach(() => {
    withPortfolio();
    resetApiMocks();
  });

  async function switchToEquivalent() {
    fireEvent.click(screen.getByRole("tab", { name: /Options/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Method: Risk/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("Equivalent"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  }

  test("8 tiers compute deterministically: equity=100k, entry=$100 → Full (10%) = 100 shs / 1 contract", async () => {
    setApiMocks({ endNlv: 100_000, state: "POWERTREND" });
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.journalLatest).toHaveBeenCalled());
    await switchToEquivalent();

    fireEvent.change(screen.getByLabelText("Stock price"), { target: { value: "100" } });

    // Wait for at least one tier card to render before iterating.
    await screen.findByTestId("options-equiv-tier-10");

    // All 8 SIZE_OPTIONS (2.5 / 5 / 7.5 / 10 / 12.5 / 15 / 17.5 / 20) render.
    expect(screen.getAllByTestId(/options-equiv-tier-/)).toHaveLength(8);

    // Full (10%) at entry=$100 → positionValue $10k → sharesEquiv 100 → 1 contract.
    const full = screen.getByTestId("options-equiv-tier-10");
    expect(full).toHaveTextContent(/Full/);
    expect(full).toHaveTextContent(/100 shs equiv/);
    expect(full).toHaveTextContent(/1 contract/);

    // Max (20%) at entry=$100 → positionValue $20k → sharesEquiv 200 → 2 contracts.
    const max = screen.getByTestId("options-equiv-tier-20");
    expect(max).toHaveTextContent(/200 shs equiv/);
    expect(max).toHaveTextContent(/2 contracts/);
  });

  test("Success banner reflects active targetSize with deterministic numeric assertion", async () => {
    setApiMocks({ endNlv: 100_000, state: "POWERTREND" });
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.journalLatest).toHaveBeenCalled());
    await switchToEquivalent();

    fireEvent.change(screen.getByLabelText("Stock price"), { target: { value: "100" } });

    // Default size = Full (10%) → 1 contract, 100 share equivalent, totalCost $100, 0.1% NLV.
    const banner = await screen.findByTestId("options-equiv-banner");
    expect(banner).toHaveTextContent(/At 10% target/);
    expect(banner).toHaveTextContent(/Buy 1 contract /); // singular
    expect(banner).toHaveTextContent(/\(100 share equivalent\)/);
    expect(banner).toHaveTextContent(/for \$100/);
    expect(banner).toHaveTextContent(/0\.1% of NLV/);

    // Switch to Max (20%) → 2 contracts, 200 share equivalent.
    fireEvent.click(screen.getByRole("button", { name: /Size: Full/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("Max"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    expect(banner).toHaveTextContent(/At 20% target/);
    expect(banner).toHaveTextContent(/Buy 2 contracts/);
    expect(banner).toHaveTextContent(/200 share equivalent/);
  });

  test("Ticker input uppercase-coerces typed input", async () => {
    setApiMocks({ endNlv: 100_000, state: "POWERTREND" });
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.journalLatest).toHaveBeenCalled());
    await switchToEquivalent();

    const tickerInput = screen.getByLabelText("Ticker symbol") as HTMLInputElement;
    fireEvent.change(tickerInput, { target: { value: "msft" } });
    expect(tickerInput.value).toBe("MSFT");
  });
});

describe("MobilePositionSizer — Options Equivalent priceLookup wiring", () => {
  beforeEach(() => {
    withPortfolio();
    resetApiMocks();
    setApiMocks({ endNlv: 100_000, price: 185.5, atrPct: 5 });
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  async function gotoEquivalent() {
    fireEvent.click(screen.getByRole("tab", { name: /Options/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Method: Risk/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("Equivalent"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  }

  test("fires priceLookup 600ms after Equivalent ticker changes and auto-fills Stock Price", async () => {
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.journalLatest).toHaveBeenCalled());
    await gotoEquivalent();

    const tickerInput = screen.getByLabelText("Ticker symbol") as HTMLInputElement;
    fireEvent.change(tickerInput, { target: { value: "nvda" } });
    expect(tickerInput.value).toBe("NVDA");

    expect(api.priceLookup).not.toHaveBeenCalledWith("NVDA");

    await act(async () => {
      vi.advanceTimersByTime(700);
    });

    await waitFor(() => expect(api.priceLookup).toHaveBeenCalledWith("NVDA"));

    const stockPriceInput = await screen.findByLabelText("Stock price") as HTMLInputElement;
    expect(stockPriceInput.value).toBe("185.5");
  });

  test("does NOT fire priceLookup in Risk mode even when a ticker is set from Equivalent", async () => {
    render(<MobilePositionSizer />);
    await waitFor(() => expect(api.journalLatest).toHaveBeenCalled());

    // Set a ticker in Equivalent first (the only Options mode with a ticker
    // input). Then drain debounce + swap to Risk and confirm a SECOND
    // ticker mutation does NOT re-fire the lookup.
    await gotoEquivalent();
    fireEvent.change(screen.getByLabelText("Ticker symbol"), { target: { value: "AAA" } });
    await act(async () => { vi.advanceTimersByTime(700); });
    await waitFor(() => expect(api.priceLookup).toHaveBeenCalledWith("AAA"));

    vi.mocked(api.priceLookup).mockClear();

    // Switch back to Risk mode. Mode picker reappears; ticker input hidden.
    fireEvent.click(screen.getByRole("button", { name: /Method: Equivalent/ }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByText("Risk"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    // optMode flip itself should NOT re-fire priceLookup (no ticker change).
    await act(async () => { vi.advanceTimersByTime(700); });
    expect(api.priceLookup).not.toHaveBeenCalled();
  });
});
