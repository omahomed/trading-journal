import { describe, test, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, within, fireEvent } from "@testing-library/react";
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

describe("MobileDashboard — sections render", () => {
  beforeEach(() => {
    vi.mocked(usePortfolio).mockReturnValue({
      portfolios: [CANSLIM],
      activePortfolio: CANSLIM,
      loading: false,
      error: null,
      refetch: vi.fn(),
      setActive: vi.fn(),
    });
    vi.mocked(api.dashboardMetrics).mockReset();
    vi.mocked(api.journalHistory).mockReset();
    vi.mocked(api.journalLatest).mockReset();
    vi.mocked(api.tradesOpen).mockReset();
    vi.mocked(api.tradesClosed).mockReset();
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
    vi.mocked(usePortfolio).mockReturnValue({
      portfolios: [CANSLIM],
      activePortfolio: CANSLIM,
      loading: false,
      error: null,
      refetch: vi.fn(),
      setActive: vi.fn(),
    });
    vi.mocked(api.dashboardMetrics).mockReset();
    vi.mocked(api.journalHistory).mockReset();
    vi.mocked(api.journalLatest).mockReset();
    vi.mocked(api.tradesOpen).mockReset();
    vi.mocked(api.tradesClosed).mockReset();
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
    vi.mocked(usePortfolio).mockReturnValue({
      portfolios: [CANSLIM],
      activePortfolio: CANSLIM,
      loading: false,
      error: null,
      refetch: vi.fn(),
      setActive: vi.fn(),
    });
    vi.mocked(api.dashboardMetrics).mockReset();
    vi.mocked(api.journalHistory).mockReset();
    vi.mocked(api.journalLatest).mockReset();
    vi.mocked(api.tradesOpen).mockReset();
    vi.mocked(api.tradesClosed).mockReset();
  });

  test("renders win/loss colors and win-rate percentage", async () => {
    // 7 wins, 3 losses → 70% win rate
    const closed = [
      ...Array.from({ length: 7 }).map((_, i) => ({ realized_pl: 1000 + i })),
      ...Array.from({ length: 3 }).map((_, i) => ({ realized_pl: -200 - i })),
    ];
    setMocks({ closed: closed as any });
    render(<MobileDashboard />);
    expect(await screen.findByText("70% win rate")).toBeInTheDocument();

    const winners = document.querySelectorAll("[aria-label='Winning trade']");
    const losers = document.querySelectorAll("[aria-label='Losing trade']");
    expect(winners.length).toBe(7);
    expect(losers.length).toBe(3);
  });

  test("pads with empty slots when fewer than 10 closed trades", async () => {
    setMocks({ closed: [{ realized_pl: 500 }, { realized_pl: -100 }] as any });
    render(<MobileDashboard />);
    await screen.findByText("Last 10 Trades");
    const empty = document.querySelectorAll("[aria-label='Empty slot']");
    expect(empty.length).toBe(8);
  });
});

describe("MobileDashboard — Equity Curve range toggle", () => {
  beforeEach(() => {
    vi.mocked(usePortfolio).mockReturnValue({
      portfolios: [CANSLIM],
      activePortfolio: CANSLIM,
      loading: false,
      error: null,
      refetch: vi.fn(),
      setActive: vi.fn(),
    });
    vi.mocked(api.dashboardMetrics).mockReset();
    vi.mocked(api.journalHistory).mockReset();
    vi.mocked(api.journalLatest).mockReset();
    vi.mocked(api.tradesOpen).mockReset();
    vi.mocked(api.tradesClosed).mockReset();
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
