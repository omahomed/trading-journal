import { render, screen } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach } from "vitest";
import type { DashboardMetrics } from "@/lib/api";

// Recharts complains under jsdom about ResizeObserver. Stubbing it keeps
// the chart renderer quiet while still rendering everything else.
class ResizeObserverStub {
  observe(): void { /* noop */ }
  unobserve(): void { /* noop */ }
  disconnect(): void { /* noop */ }
}
(globalThis as any).ResizeObserver = ResizeObserverStub;

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

vi.mock("@/lib/api", () => ({
  api: {
    journalLatest: vi.fn(),
    journalHistory: vi.fn(),
    tradesOpen: vi.fn(),
    tradesClosed: vi.fn(),
    events: vi.fn(),
    dashboardMetrics: vi.fn(),
  },
  getActivePortfolio: () => "CanSlim",
}));

// Capture-snapshot calls fetchWithAuth which goes through next-auth.
// Stub the whole component since it's not under test here.
vi.mock("./capture-snapshot", () => ({
  CaptureSnapshotButton: () => null,
}));

import { api } from "@/lib/api";
import { Dashboard } from "./dashboard";

const mLatest = vi.mocked(api.journalLatest);
const mHistory = vi.mocked(api.journalHistory);
const mOpen = vi.mocked(api.tradesOpen);
const mClosed = vi.mocked(api.tradesClosed);
const mEvents = vi.mocked(api.events);
const mDash = vi.mocked(api.dashboardMetrics);


function fullMetrics(overrides: Partial<DashboardMetrics> = {}): DashboardMetrics {
  // Anchored on the user's actual prod data (verified live):
  // NLV $486,630.39, Total Holdings $917,498.79, Exposure 188.5%,
  // LTD 286.51% TWR. The "Live estimate" is intentionally a hair higher
  // than journal NLV — what you'd see if yfinance's intraday quotes
  // moved positions up after the broker's EOD snapshot.
  return {
    journal_available: true,
    as_of_date: "2026-04-24",
    nlv: 486630.39,
    nlv_delta_dollar: 13496.0,
    nlv_delta_pct: 2.85,
    total_holdings: 917498.78,
    exposure_pct: 188.5413,
    cash: -430868.39,
    drawdown_current_pct: 0.0,
    drawdown_peak_nlv: 486630.39,
    drawdown_peak_date: "2026-04-24",
    ltd_pct: 286.51,
    ltd_pl_dollar: 339829.00,
    ytd_pct: 57.16,
    ytd_pl_dollar: 124363.00,
    ytd_available: true,
    live_estimate_nlv: 487291.22,
    live_estimate_diff: 660.83,
    live_estimate_diff_pct: 0.14,
    live_estimate_unavailable: false,
    as_of: "2026-04-27T13:30:00",
    ...overrides,
  };
}


function setupDefaults() {
  mLatest.mockResolvedValue({
    end_nlv: 486630.39, beg_nlv: 471004.89, day: "2026-04-24",
    daily_dollar_change: 13496, daily_pct_change: 2.85,
    pct_invested: 188.5413, market_window: "POWERTREND",
    portfolio_heat: 10.88,
  } as any);
  mHistory.mockResolvedValue([] as any);
  mOpen.mockResolvedValue(Array.from({ length: 24 }, (_, i) => ({
    ticker: `TKR${i}`, shares: 100, avg_entry: 50,
  })) as any);
  mClosed.mockResolvedValue([]);
  mEvents.mockResolvedValue([]);
}


describe("Dashboard — journal-as-source-of-truth refactor", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaults();
  });

  test("renders headline NLV from journal + 'Live estimate' as a subordinate sub-label", async () => {
    mDash.mockResolvedValue(fullMetrics());

    render(<Dashboard navColor="#6366f1" />);

    // The big number on the NLV tile — sourced from journal, not from
    // compute_nlv.
    expect(await screen.findByText("$486,630")).toBeInTheDocument();

    // The live estimate is rendered as the smaller second subtext line on
    // the same tile (not as a peer KPI). data-testid pins it as the
    // "extraSub" slot of the KPITile component.
    const extraSubs = await screen.findAllByTestId("kpi-extra-sub");
    const liveLine = extraSubs.find(el => /Live estimate/.test(el.textContent || ""));
    expect(liveLine).toBeDefined();
    expect(liveLine).toHaveTextContent("Live estimate: $487,291");
    expect(liveLine).toHaveTextContent("(+$661, +0.14%)");
  });

  test("renders 'Live estimate: unavailable' when compute_nlv blew up", async () => {
    mDash.mockResolvedValue(fullMetrics({
      live_estimate_nlv: null,
      live_estimate_diff: null,
      live_estimate_diff_pct: null,
      live_estimate_unavailable: true,
    }));

    render(<Dashboard navColor="#6366f1" />);

    // Headline NLV still renders normally — journal-derived fields aren't
    // affected by a live-estimate failure.
    expect(await screen.findByText("$486,630")).toBeInTheDocument();
    const extraSubs = await screen.findAllByTestId("kpi-extra-sub");
    const liveLine = extraSubs.find(el => /Live estimate/.test(el.textContent || ""));
    expect(liveLine).toHaveTextContent("Live estimate: unavailable");
  });

  test("removes pre-refactor disclaimer + Computed/Diff line", async () => {
    // Two artifacts of the old "live drives the dashboard" world that
    // explicitly need to be gone:
    //   1. "NLV excludes commissions & margin interest. Reconcile in
    //      Settings to match your broker." — the broker-pulled journal
    //      NLV INCLUDES those, so the disclaimer was wrong post-IBKR.
    //   2. "Computed: $X · Diff: ±$Y" — its info now lives in the
    //      "Live estimate" subordinate sub-label.
    mDash.mockResolvedValue(fullMetrics());

    render(<Dashboard navColor="#6366f1" />);
    // wait for the tile to render before asserting absence
    await screen.findByText("$486,630");

    expect(screen.queryByText(/NLV excludes commissions/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Reconcile in Settings/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/^Computed:/)).not.toBeInTheDocument();
    expect(screen.queryByText(/^Diff:/)).not.toBeInTheDocument();
  });

  test("Cash + Total Holdings sub-row is NOT rendered when journal data is available", async () => {
    // The sub-row was useful while journal and live NLV ran in parallel;
    // now that journal is the single source of truth and live is only an
    // informational sub-label, the breakdown is redundant (encoded in
    // Live Exposure) and slightly misleading (the live breakdown wouldn't
    // match the journal-headlined NLV). Numbers still live on the Trade
    // Journal (positions) and Settings → Cash Transactions pages.
    mDash.mockResolvedValue(fullMetrics());

    render(<Dashboard navColor="#6366f1" />);
    // Wait for the dashboard to finish rendering before asserting absence.
    await screen.findByText("$486,630");

    expect(screen.queryByTestId("dashboard-cash-positions-row")).not.toBeInTheDocument();
    // The literal " cash" / " positions" copy from the row should also
    // be gone — guards against someone re-introducing the breakdown
    // without the testid.
    expect(screen.queryByText(/^\$-430,868$/)).not.toBeInTheDocument();
  });

  test("Drawdown tile reads from dashboard-metrics, not from history.max(end_nlv)", async () => {
    // Confirm the drawdown number is sourced from metrics — pass an
    // exaggerated peak via metrics so any fallback to history would be
    // visibly different.
    mDash.mockResolvedValue(fullMetrics({
      drawdown_current_pct: -7.42,
      drawdown_peak_nlv: 525000,
      drawdown_peak_date: "2026-04-15",
    }));
    mHistory.mockResolvedValue([] as any);  // empty — would yield -Infinity peak in old path

    render(<Dashboard navColor="#6366f1" />);

    // Old code would show 0% (since history.max of empty is -Infinity).
    // New code shows the metrics value verbatim.
    expect(await screen.findByText("-7.42%")).toBeInTheDocument();
    expect(screen.getByText(/from peak \$525,000/)).toBeInTheDocument();
  });

  test("Live Exposure tile renders metrics.exposure_pct (no live-prices side fetch)", async () => {
    mDash.mockResolvedValue(fullMetrics());

    render(<Dashboard navColor="#6366f1" />);

    // 188.5413 rounded to 1 decimal place → 188.5%
    expect(await screen.findByText("188.5%")).toBeInTheDocument();
    // The dashboard must NOT call batchPrices any more — that side-fetch
    // was the old live-exposure path. (api mock doesn't include it; if
    // dashboard.tsx still calls it the test would crash.)
    expect((api as any).batchPrices).toBeUndefined();
  });

  test("'As of [date]' badge is shown when journal_available is true", async () => {
    mDash.mockResolvedValue(fullMetrics({ as_of_date: "2026-04-24" }));

    render(<Dashboard navColor="#6366f1" />);

    const badge = await screen.findByTestId("dashboard-as-of-badge");
    expect(badge).toHaveTextContent("As of 2026-04-24");
  });

  test("empty journal renders '—' placeholders + the help banner", async () => {
    // Brand-new portfolio: no journal entries yet. Dashboard must render
    // gracefully — every KPI shows "—" and a clear call-to-action banner
    // sits above the strip pointing at Daily Routine.
    mDash.mockResolvedValue({
      journal_available: false,
      as_of_date: null,
      nlv: null,
      nlv_delta_dollar: null,
      nlv_delta_pct: null,
      total_holdings: null,
      exposure_pct: null,
      cash: null,
      drawdown_current_pct: null,
      drawdown_peak_nlv: null,
      drawdown_peak_date: null,
      ltd_pct: null,
      ltd_pl_dollar: null,
      ytd_pct: null,
      ytd_pl_dollar: null,
      ytd_available: false,
      live_estimate_nlv: null,
      live_estimate_diff: null,
      live_estimate_diff_pct: null,
      live_estimate_unavailable: true,
      as_of: "2026-04-27T13:30:00",
    });
    mLatest.mockResolvedValue(null as any);

    render(<Dashboard navColor="#6366f1" />);

    const banner = await screen.findByTestId("dashboard-empty-state");
    expect(banner).toHaveTextContent(/Save your first daily routine/i);
    // Multiple "—" placeholders — one per tile that has no value
    const dashEntries = await screen.findAllByText("—");
    expect(dashEntries.length).toBeGreaterThanOrEqual(4);
    // No "As of" badge when there's no journal
    expect(screen.queryByTestId("dashboard-as-of-badge")).not.toBeInTheDocument();
    // Cash + positions sub-row is hidden too
    expect(screen.queryByTestId("dashboard-cash-positions-row")).not.toBeInTheDocument();
  });

  test("LTD tile sub renders dollar P&L when available", async () => {
    // Replaces the prior "Time-weighted, since reset" copy. Anchored on
    // the user's expected $339,829 LTD P&L. Format: leading +, comma
    // separators, whole dollars.
    mDash.mockResolvedValue(fullMetrics());

    render(<Dashboard navColor="#6366f1" />);

    expect(await screen.findByText("286.51%")).toBeInTheDocument();
    // The "$+339,829" copy must be present somewhere in the LTD tile area.
    expect(await screen.findByText("$+339,829")).toBeInTheDocument();
    // Old copy is gone
    expect(screen.queryByText(/Time-weighted, since reset/)).not.toBeInTheDocument();
  });

  test("LTD tile sub falls back to descriptive text when ledger lookup fails", async () => {
    // Edge: db.get_net_contributions blew up → backend sets
    // ltd_pl_dollar=null. Tile renders the static fallback rather than
    // an inaccurate "$+0".
    mDash.mockResolvedValue(fullMetrics({ ltd_pl_dollar: null }));

    render(<Dashboard navColor="#6366f1" />);

    expect(await screen.findByText("286.51%")).toBeInTheDocument();
    expect(await screen.findByText(/Time-weighted, since reset/i)).toBeInTheDocument();
    expect(screen.queryByText("$+339,829")).not.toBeInTheDocument();
  });

  test("YTD tile renders two-line sub: dollar P&L primary + SPY/NDX as extraSub", async () => {
    // Spec: "Two-line sub-label: $+124,363 / SPY +4.50% | NDX +6.89%".
    // The benchmarks need history data to compute — provide a minimal
    // pair with prior-year + current-year SPY/NDX values.
    const yr = new Date().getFullYear();
    mHistory.mockResolvedValue([
      { day: `${yr}-01-02`, spy: 600, nasdaq: 21000, end_nlv: 0,
        daily_pct_change: 0, portfolio_ltd: 0, spy_ltd: 0, ndx_ltd: 0,
        pct_invested: 0, portfolio_heat: 0 },
      { day: `${yr}-04-24`, spy: 627, nasdaq: 22446.9, end_nlv: 0,
        daily_pct_change: 0, portfolio_ltd: 0, spy_ltd: 0, ndx_ltd: 0,
        pct_invested: 0, portfolio_heat: 0 },
    ] as any);
    mDash.mockResolvedValue(fullMetrics());

    render(<Dashboard navColor="#6366f1" />);

    // Headline + dollar primary sub
    expect(await screen.findByText("57.16%")).toBeInTheDocument();
    expect(await screen.findByText("$+124,363")).toBeInTheDocument();
    // SPY/NDX line lives in the extraSub slot. It says "SPY +X.XX% | NDX +X.XX%"
    // (without the colon now). Lookup by partial regex so we don't depend on
    // exact percentages.
    const extraSubs = await screen.findAllByTestId("kpi-extra-sub");
    const benchmarkLine = extraSubs.find(el => /SPY .* \| NDX /.test(el.textContent || ""));
    expect(benchmarkLine).toBeDefined();
    expect(benchmarkLine).toHaveTextContent(/SPY \+4\.50%/);
    expect(benchmarkLine).toHaveTextContent(/NDX \+6\.89%/);
  });

  test("YTD tile falls back to SPY/NDX as primary sub when ytd_pl_dollar is null", async () => {
    // No prior-year journal → backend can't anchor YTD baseline → null.
    // SPY/NDX line then takes the primary slot (don't lose the info).
    mDash.mockResolvedValue(fullMetrics({ ytd_pl_dollar: null }));

    render(<Dashboard navColor="#6366f1" />);

    expect(await screen.findByText("57.16%")).toBeInTheDocument();
    expect(screen.queryByText("$+124,363")).not.toBeInTheDocument();
    // No extraSub row on the YTD tile in this case
    const benchmarkText = await screen.findByText(/SPY:.*NDX:/);
    expect(benchmarkText).toBeInTheDocument();
  });
});
