import { describe, test, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import type { ReactNode } from "react";

vi.mock("@/lib/portfolio-context", () => ({
  usePortfolio: vi.fn(),
}));

// recharts in jsdom: ResponsiveContainer needs measured layout we don't have.
// Replace the three render primitives we touch with passthroughs so the chart
// branch mounts without errors while preserving the surrounding markup tests
// care about (range pills, legend, "No data" empty state).
vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: { children: ReactNode }) => (
    <div data-testid="rc-container">{children}</div>
  ),
  ComposedChart: ({ children }: { children: ReactNode }) => (
    <div data-testid="rc-chart">{children}</div>
  ),
  Line: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
}));

vi.mock("@/lib/api", () => ({
  api: {
    dashboardMetrics: vi.fn(),
    journalHistory: vi.fn(),
    journalLatest: vi.fn(),
    tradesOpen: vi.fn(),
    tradesClosed: vi.fn(),
  },
  getActivePortfolio: vi.fn(() => "CanSlim"),
}));

vi.mock("@/lib/log", () => ({
  log: { error: vi.fn(), warn: vi.fn(), info: vi.fn(), debug: vi.fn() },
}));

import { api } from "@/lib/api";
import { usePortfolio } from "@/lib/portfolio-context";
import { MobileDashboard } from "./mobile-dashboard";
import type { Portfolio, DashboardMetrics } from "@/lib/api";

const CANSLIM: Portfolio = {
  id: 1,
  name: "CanSlim",
  starting_capital: 100000,
  reset_date: null,
  created_at: "2026-01-01T00:00:00Z",
  cash_balance: 0,
};

function baseMetrics(overrides: Partial<DashboardMetrics> = {}): DashboardMetrics {
  return {
    journal_available: true,
    as_of_date: "2026-05-22",
    nlv: 487704,
    nlv_delta_dollar: 4890,
    nlv_delta_pct: 1.01,
    total_holdings: 425000,
    exposure_pct: 87.4,
    cash: 62704,
    drawdown_current_pct: -2.3,
    drawdown_peak_nlv: 499000,
    drawdown_peak_date: "2026-05-10",
    ltd_pct: 38.5,
    ltd_pl_dollar: 135420,
    ytd_pct: 12.3,
    ytd_pl_dollar: 53000,
    ytd_available: true,
    as_of: "2026-05-22T20:00:00Z",
    ...overrides,
  };
}

// Build a closed-trade record that the Last10 strip will accept. Open
// dates control ordering (newest = rightmost square); pl drives the
// outcome classification (computeLast10Stats's default beDeadzone is 50).
function trade(opts: { id: string; ticker: string; openDate: string; pl: number }) {
  return {
    trade_id: opts.id,
    ticker: opts.ticker,
    status: "CLOSED",
    open_date: opts.openDate,
    realized_pl: opts.pl,
  };
}

function setMocks(opts: {
  metrics?: DashboardMetrics | null;
  history?: any[];
  latest?: any;
  open?: any[];
  closed?: any[];
}) {
  vi.mocked(api.dashboardMetrics).mockResolvedValue(opts.metrics ?? baseMetrics());
  vi.mocked(api.journalHistory).mockResolvedValue(opts.history ?? []);
  vi.mocked(api.journalLatest).mockResolvedValue(opts.latest ?? { portfolio_heat: 2.1 } as any);
  vi.mocked(api.tradesOpen).mockResolvedValue((opts.open ?? []) as any);
  vi.mocked(api.tradesClosed).mockResolvedValue((opts.closed ?? []) as any);
}

function resetApiMocks() {
  vi.mocked(api.dashboardMetrics).mockReset();
  vi.mocked(api.journalHistory).mockReset();
  vi.mocked(api.journalLatest).mockReset();
  vi.mocked(api.tradesOpen).mockReset();
  vi.mocked(api.tradesClosed).mockReset();
}

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

describe("MobileDashboard — sections render", () => {
  beforeEach(() => {
    withPortfolio();
    resetApiMocks();
  });

  test("renders as-of caption, NLV, all 4 KPI labels, EC, Last 10", async () => {
    setMocks({ open: Array.from({ length: 23 }).map((_, i) => ({ ticker: `T${i}` })) });
    render(<MobileDashboard />);
    expect(await screen.findByText("As of 2026-05-22")).toBeInTheDocument();
    expect(screen.getByText("Net Liq Value")).toBeInTheDocument();
    expect(screen.getByText("$487,704")).toBeInTheDocument();
    expect(screen.getByText("LTD Return")).toBeInTheDocument();
    expect(screen.getByText("YTD Return")).toBeInTheDocument();
    expect(screen.getByText("EOD Exposure")).toBeInTheDocument();
    expect(screen.getByText("Drawdown")).toBeInTheDocument();
    expect(screen.getByText("Equity Curve")).toBeInTheDocument();
    expect(screen.getByText("Last 10 Trades")).toBeInTheDocument();
    expect(screen.getByText("OLDEST")).toBeInTheDocument();
    expect(screen.getByText("NEWEST")).toBeInTheDocument();
  });

  test("EOD Exposure surfaces 'N pos · risk N.N%' with open count + heat", async () => {
    setMocks({
      open: Array.from({ length: 23 }).map((_, i) => ({ ticker: `T${i}` })),
      latest: { portfolio_heat: 2.1 } as any,
    });
    render(<MobileDashboard />);
    expect(await screen.findByText("23 pos · risk 2.1%")).toBeInTheDocument();
  });
});

describe("MobileDashboard — conditional styling", () => {
  beforeEach(() => {
    withPortfolio();
    resetApiMocks();
  });

  test("EOD Exposure amber when value > 100% (margin territory)", async () => {
    setMocks({ metrics: baseMetrics({ exposure_pct: 142.3 }) });
    render(<MobileDashboard />);
    const value = await screen.findByText("142.3%");
    expect(value.className).toMatch(/text-m-warn/);
  });

  test("EOD Exposure neutral when value <= 100%", async () => {
    setMocks({ metrics: baseMetrics({ exposure_pct: 87.4 }) });
    render(<MobileDashboard />);
    const value = await screen.findByText("87.4%");
    expect(value.className).not.toMatch(/text-m-warn/);
    expect(value.className).toMatch(/text-m-text/);
  });

  test("Drawdown shows 'Clear' tag when ddPct is effectively zero", async () => {
    setMocks({ metrics: baseMetrics({ drawdown_current_pct: 0 }) });
    render(<MobileDashboard />);
    expect(await screen.findByText("Clear")).toBeInTheDocument();
  });

  test("Drawdown shows peak NLV sub when in drawdown", async () => {
    setMocks({
      metrics: baseMetrics({ drawdown_current_pct: -5.4, drawdown_peak_nlv: 510000 }),
    });
    render(<MobileDashboard />);
    expect(await screen.findByText(/peak \$510,000/)).toBeInTheDocument();
    expect(screen.queryByText("Clear")).not.toBeInTheDocument();
  });
});

describe("MobileDashboard — Last 10 strip", () => {
  beforeEach(() => {
    withPortfolio();
    resetApiMocks();
  });

  test("renders one square per trade, color-coded by outcome, and a win-rate caption", async () => {
    // 7 wins, 3 losses → 70% win rate. open_date ascending so ordering is stable.
    const closed = [
      ...Array.from({ length: 7 }).map((_, i) =>
        trade({ id: `w${i}`, ticker: `WIN${i}`, openDate: `2026-05-0${i + 1}`, pl: 1000 + i }),
      ),
      ...Array.from({ length: 3 }).map((_, i) =>
        trade({ id: `l${i}`, ticker: `LOSS${i}`, openDate: `2026-05-1${i}`, pl: -200 - i }),
      ),
    ];
    setMocks({ closed });
    render(<MobileDashboard />);
    expect(await screen.findByText("70% win rate")).toBeInTheDocument();

    const winners = screen.getAllByRole("button", { name: /winning trade$/ });
    const losers = screen.getAllByRole("button", { name: /losing trade$/ });
    expect(winners.length).toBe(7);
    expect(losers.length).toBe(3);
  });

  test("renders break-even color when |pl| is inside the beDeadzone", async () => {
    // pl=30 falls inside the default 50-dollar BE deadzone in computeLast10Stats.
    setMocks({ closed: [trade({ id: "be0", ticker: "FLAT", openDate: "2026-05-01", pl: 30 })] });
    render(<MobileDashboard />);
    const beSquare = await screen.findByRole("button", { name: /break-even trade$/ });
    expect(beSquare.className).toMatch(/bg-m-text-faint/);
  });

  test("pads with empty (non-tappable) slots when fewer than 10 closed trades", async () => {
    setMocks({
      closed: [
        trade({ id: "w0", ticker: "AAA", openDate: "2026-05-01", pl: 500 }),
        trade({ id: "l0", ticker: "BBB", openDate: "2026-05-02", pl: -100 }),
      ],
    });
    render(<MobileDashboard />);
    await screen.findByText("Last 10 Trades");
    const empty = screen.getAllByRole("button", { name: "Empty slot" });
    expect(empty.length).toBe(8);
    expect((empty[0] as HTMLButtonElement).disabled).toBe(true);
  });
});

describe("MobileDashboard — Last 10 popover", () => {
  beforeEach(() => {
    withPortfolio();
    resetApiMocks();
  });

  test("tapping a square opens a popover with ticker, status, P&L, and a View trade link", async () => {
    setMocks({
      closed: [
        trade({ id: "abc123", ticker: "NVDA", openDate: "2026-05-12", pl: 1500 }),
      ],
    });
    render(<MobileDashboard />);
    const square = await screen.findByRole("button", { name: /NVDA · winning trade$/ });
    fireEvent.click(square);

    const popover = await screen.findByRole("dialog", { name: /NVDA trade detail/ });
    expect(popover).toBeInTheDocument();
    expect(popover.textContent).toMatch(/NVDA/);
    expect(popover.textContent).toMatch(/CLOSED/);
    expect(popover.textContent).toMatch(/\+\$1,500/);
    expect(popover.textContent).toMatch(/2026-05-12/);

    const link = screen.getByRole("link", { name: /View trade/ });
    expect(link).toHaveAttribute("href", "/trade-journal?trade_id=abc123");
  });

  test("tapping the same square again closes the popover", async () => {
    setMocks({
      closed: [trade({ id: "abc", ticker: "AMD", openDate: "2026-05-10", pl: 800 })],
    });
    render(<MobileDashboard />);
    const square = await screen.findByRole("button", { name: /AMD · winning trade$/ });
    fireEvent.click(square);
    expect(await screen.findByRole("dialog")).toBeInTheDocument();
    fireEvent.click(square);
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });

  test("tapping outside the strip dismisses the popover", async () => {
    setMocks({
      closed: [trade({ id: "abc", ticker: "TSLA", openDate: "2026-05-10", pl: 800 })],
    });
    render(
      <div>
        <MobileDashboard />
        <button type="button" data-testid="outside">outside</button>
      </div>,
    );
    const square = await screen.findByRole("button", { name: /TSLA · winning trade$/ });
    fireEvent.click(square);
    expect(await screen.findByRole("dialog")).toBeInTheDocument();
    fireEvent.pointerDown(screen.getByTestId("outside"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });
});

describe("MobileDashboard — Equity Curve range toggle", () => {
  beforeEach(() => {
    withPortfolio();
    resetApiMocks();
  });

  test("default range is 6M (active pill)", async () => {
    setMocks({});
    render(<MobileDashboard />);
    const six = await screen.findByRole("button", { name: "6M" });
    expect(six).toHaveAttribute("aria-pressed", "true");
    const all = screen.getByRole("button", { name: "All" });
    expect(all).toHaveAttribute("aria-pressed", "false");
  });

  test("tapping 'All' makes it active and 6M inactive", async () => {
    setMocks({});
    render(<MobileDashboard />);
    const all = await screen.findByRole("button", { name: "All" });
    fireEvent.click(all);
    await waitFor(() => expect(all).toHaveAttribute("aria-pressed", "true"));
    const six = screen.getByRole("button", { name: "6M" });
    expect(six).toHaveAttribute("aria-pressed", "false");
  });
});
