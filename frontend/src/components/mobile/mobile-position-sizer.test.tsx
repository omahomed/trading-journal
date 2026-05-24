import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";

vi.mock("@/lib/portfolio-context", () => ({
  usePortfolio: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  api: {
    journalLatest: vi.fn(),
    rallyPrefix: vi.fn(),
    priceLookup: vi.fn(),
  },
  getActivePortfolio: vi.fn(() => "CanSlim"),
}));

vi.mock("@/lib/log", () => ({
  log: { error: vi.fn(), warn: vi.fn(), info: vi.fn(), debug: vi.fn() },
}));

import { api } from "@/lib/api";
import { usePortfolio } from "@/lib/portfolio-context";
import { MobilePositionSizer } from "./mobile-position-sizer";
import type { Portfolio } from "@/lib/api";

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

function setApiMocks(opts: {
  endNlv?: number | null;
  state?: "POWERTREND" | "UPTREND" | "RALLY MODE" | "CORRECTION" | null;
  price?: number;
  atrPct?: number;
  priceRejects?: boolean;
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
}

function resetApiMocks() {
  vi.mocked(api.journalLatest).mockReset();
  vi.mocked(api.rallyPrefix).mockReset();
  vi.mocked(api.priceLookup).mockReset();
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
    vi.mocked(api.journalLatest).mockResolvedValue({ end_nlv: 100_000 } as any);
    vi.mocked(api.rallyPrefix).mockResolvedValue(null as any);
    vi.mocked(api.priceLookup).mockResolvedValue({} as any);
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
    expect(tickerInput.value).toBe("NVDA"); // uppercase coercion

    // Debounce hasn't fired yet.
    expect(api.priceLookup).not.toHaveBeenCalled();

    await act(async () => {
      vi.advanceTimersByTime(700);
    });

    await waitFor(() => expect(api.priceLookup).toHaveBeenCalledWith("NVDA"));

    // Entry filled from price; ATR cell shows the percentage.
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

    // Entry input remains empty + editable for manual override.
    const entryInput = screen.getByLabelText("Entry price") as HTMLInputElement;
    expect(entryInput.value).toBe("");
    expect(entryInput).not.toBeDisabled();
  });
});

describe("MobilePositionSizer — live compute", () => {
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

    // Enter values manually (no ticker → no priceLookup needed).
    fireEvent.change(screen.getByLabelText("Entry price"), { target: { value: "100" } });
    fireEvent.change(screen.getByLabelText("Key MA level"), { target: { value: "95" } });

    // Need ATR — easiest path is to spoof a successful priceLookup. Trigger it.
    const ticker = screen.getByLabelText("Ticker symbol");
    fireEvent.change(ticker, { target: { value: "X" } });
    await waitFor(() => expect(api.priceLookup).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByText("5.0%")).toBeInTheDocument());

    // Lib computation (Offense 1.0%, Full 10% target):
    //   riskBudget = 100000 * 1.0 / 100 = 1000
    //   positionCapShares = floor(100000 * 10/100 / 100) = 100
    //   Tech stop: 95 * 0.99 = 94.05; rps = 5.95; candidate = floor(1000/5.95) = 168
    //     → final = min(168, 100) = 100, capBinds=true
    //   ATR scenarios at 1× / 1.5× / 2× all candidate >= 100 (5%/7.5%/10% rps)
    //     → final = 100 for all three, capBinds=true on all
    //   techStop.atrFraction = 5.95/5 = 1.19 ≥ 1.0 → recommended = tech stop
    const sharesEl = await screen.findByTestId("audit-shares");
    await waitFor(() => expect(sharesEl).toHaveTextContent("100"), { timeout: 2000 });

    // Verdict reflects tech-stop recommendation + tier-cap binding
    const verdict = screen.getByTestId("verdict-card");
    expect(verdict).toHaveTextContent(/Sized by tech stop/);
    expect(verdict).toHaveTextContent(/position-size tier/);

    // All four scenarios render with their labels
    expect(screen.getByTestId("scenario-tech-stop")).toBeInTheDocument();
    expect(screen.getByTestId("scenario-1x-atr")).toBeInTheDocument();
    expect(screen.getByTestId("scenario-1.5x-atr")).toBeInTheDocument();
    expect(screen.getByTestId("scenario-2x-atr")).toBeInTheDocument();

    // Recommended pill: exactly one, on the Tech Stop card.
    const pills = screen.getAllByTestId("recommended-pill");
    expect(pills).toHaveLength(1);
    expect(screen.getByTestId("scenario-tech-stop")).toContainElement(pills[0]);
  });

  test("flips recommendation to 1.5× ATR when tech stop sits inside 1 ATR (warning visible)", async () => {
    render(<MobilePositionSizer />);
    await waitFor(() => expect(screen.getByText("$100,000")).toBeInTheDocument());

    fireEvent.change(screen.getByLabelText("Entry price"), { target: { value: "100" } });
    // MA = 99, buffer 1% → stop 98.01, distance ~2% — for atr=5% that's 0.4× ATR
    fireEvent.change(screen.getByLabelText("Key MA level"), { target: { value: "99" } });

    const ticker = screen.getByLabelText("Ticker symbol");
    fireEvent.change(ticker, { target: { value: "Y" } });
    await waitFor(() => expect(screen.getByText("5.0%")).toBeInTheDocument());

    // Warning banner shows + recommendation flips to 1.5× ATR
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
    // stopDistPct = (100-94.05)/100*100 = 5.95%; atrFraction = 5.95/5 = 1.19
    expect(banner).toHaveTextContent(/1\.19× ATR/);
  });
});

describe("MobilePositionSizer — Mode picker manual override", () => {
  beforeEach(() => {
    withPortfolio();
    resetApiMocks();
    setApiMocks({ endNlv: 100_000, state: "POWERTREND" }); // auto-selects Offense
  });

  test("tapping a Mode option closes the sheet and updates the trigger", async () => {
    render(<MobilePositionSizer />);
    const trigger = await screen.findByRole("button", { name: /Mode: Offense/ });
    fireEvent.click(trigger);

    const dialog = await screen.findByRole("dialog", { name: /Sizing mode/ });
    expect(dialog).toBeInTheDocument();

    // Pick Defense.
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
    // Each option's primary label renders as exact-match text — getByText
    // is exact by default, so this also disambiguates "Core" vs "Core+".
    for (const label of ["Starter", "Half", "Standard", "Full", "Overweight", "Core", "Core+", "Max"]) {
      expect(within(dialog).getByText(label)).toBeInTheDocument();
    }

    fireEvent.click(within(dialog).getByText("Starter"));
    expect(await screen.findByRole("button", { name: /Size: Starter/ })).toBeInTheDocument();
  });
});
