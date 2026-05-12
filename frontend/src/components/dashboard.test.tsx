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
    tradesRecent: vi.fn(),
    batchPrices: vi.fn(),
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
const mRecent = vi.mocked(api.tradesRecent);
const mPrices = vi.mocked(api.batchPrices);


function fullMetrics(overrides: Partial<DashboardMetrics> = {}): DashboardMetrics {
  // Anchored on the user's actual prod data (verified live):
  // NLV $486,630.39, Total Holdings $917,498.79, Exposure 188.5%,
  // LTD 286.51% TWR.
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
  mRecent.mockResolvedValue({ details: [], lot_closures: [] } as any);
  mPrices.mockResolvedValue({});
}


describe("Dashboard — journal-as-source-of-truth refactor", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaults();
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
    expect(screen.queryByText(/^-\$430,868$/)).not.toBeInTheDocument();
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

  test("Live Exposure tile renders metrics.exposure_pct from journal, not from live prices", async () => {
    mDash.mockResolvedValue(fullMetrics());
    // Even with a failed price fetch the exposure tile must still render
    // its metric from the journal — the Live Exposure path is journal-
    // backed; batchPrices is only used by the Last 10 Trades panel.
    mPrices.mockRejectedValue(new Error("simulated price fetch failure"));

    render(<Dashboard navColor="#6366f1" />);

    // 188.5413 rounded to 1 decimal place → 188.5%
    expect(await screen.findByText("188.5%")).toBeInTheDocument();
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
    // The "+$339,829" copy must be present somewhere in the LTD tile area.
    expect(await screen.findByText("+$339,829")).toBeInTheDocument();
    // Old copy is gone
    expect(screen.queryByText(/Time-weighted, since reset/)).not.toBeInTheDocument();
  });

  test("LTD tile sub falls back to descriptive text when ledger lookup fails", async () => {
    // Edge: db.get_net_contributions blew up → backend sets
    // ltd_pl_dollar=null. Tile renders the static fallback rather than
    // an inaccurate "+$0".
    mDash.mockResolvedValue(fullMetrics({ ltd_pl_dollar: null }));

    render(<Dashboard navColor="#6366f1" />);

    expect(await screen.findByText("286.51%")).toBeInTheDocument();
    expect(await screen.findByText(/Time-weighted, since reset/i)).toBeInTheDocument();
    expect(screen.queryByText("+$339,829")).not.toBeInTheDocument();
  });

  test("YTD tile renders two-line sub: dollar P&L primary + SPY/NDX as extraSub", async () => {
    // Spec: "Two-line sub-label: +$124,363 / SPY +4.50% | NDX +6.89%".
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
    expect(await screen.findByText("+$124,363")).toBeInTheDocument();
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
    expect(screen.queryByText("+$124,363")).not.toBeInTheDocument();
    // No extraSub row on the YTD tile in this case
    const benchmarkText = await screen.findByText(/SPY:.*NDX:/);
    expect(benchmarkText).toBeInTheDocument();
  });
});

describe("Dashboard — Last 10 Trades + Discipline Pulse panels", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaults();
    mDash.mockResolvedValue(fullMetrics());
  });

  test("Last 10 panel renders sequence strip with exactly 10 squares when ≥10 trades exist", async () => {
    // 12 closed trades → window of 10 most recent
    const trades = Array.from({ length: 12 }, (_, i) => {
      const day = String(28 - i).padStart(2, "0");
      return {
        trade_id: `T${i}`,
        ticker: `TICK${i}`,
        status: "CLOSED",
        open_date: `2026-04-${day}`,
        closed_date: `2026-04-${day}`,
        realized_pl: i % 2 === 0 ? 1500 : -800,
      };
    });
    mClosed.mockResolvedValue(trades as any);
    mOpen.mockResolvedValue([]);

    render(<Dashboard navColor="#6366f1" />);

    const strip = await screen.findByTestId("last10-sequence");
    expect(strip.children.length).toBe(10);

    // Headline tile labels render
    expect(screen.getByText("Win Rate")).toBeInTheDocument();
    expect(screen.getByText("Net P&L")).toBeInTheDocument();
    expect(screen.getByText("Avg W / L")).toBeInTheDocument();
    // "Profit Factor" appears in BOTH Last 10 panel and Discipline Pulse;
    // assert at least one match rather than uniqueness.
    expect(screen.getAllByText("Profit Factor").length).toBeGreaterThanOrEqual(1);
  });

  test("Last 10 panel handles <10 trades gracefully (no padding)", async () => {
    const trades = Array.from({ length: 4 }, (_, i) => ({
      trade_id: `T${i}`,
      ticker: `TICK${i}`,
      status: "CLOSED",
      open_date: `2026-04-${String(20 - i).padStart(2, "0")}`,
      closed_date: `2026-04-${String(20 - i).padStart(2, "0")}`,
      realized_pl: 1000,
    }));
    mClosed.mockResolvedValue(trades as any);
    mOpen.mockResolvedValue([]);

    render(<Dashboard navColor="#6366f1" />);

    const strip = await screen.findByTestId("last10-sequence");
    expect(strip.children.length).toBe(4);
    // Header reflects actual count
    expect(screen.getByText("Last 4 Trades")).toBeInTheDocument();
  });

  test("Last 10 panel shows empty state when no trades exist", async () => {
    mClosed.mockResolvedValue([]);
    mOpen.mockResolvedValue([]);

    render(<Dashboard navColor="#6366f1" />);

    expect(await screen.findByText("No trades yet")).toBeInTheDocument();
  });

  test("Discipline Pulse renders all three tiles wrapped in /analytics?tab=... links", async () => {
    // Need 2026 closed losses for compliance + ratio computations to be
    // meaningful; the tiles render regardless, but linking is what we
    // assert here.
    mClosed.mockResolvedValue([
      { trade_id: "C1", ticker: "AAA", status: "CLOSED", open_date: "2026-02-01", closed_date: "2026-02-15", realized_pl: 5000 },
      { trade_id: "C2", ticker: "BBB", status: "CLOSED", open_date: "2026-03-01", closed_date: "2026-03-10", realized_pl: -2000 },
    ] as any);
    mHistory.mockResolvedValue([
      { day: "2026-02-01", end_nlv: 500_000 } as any,
      { day: "2026-03-01", end_nlv: 510_000 } as any,
    ]);
    mOpen.mockResolvedValue([]);

    render(<Dashboard navColor="#6366f1" />);

    expect(await screen.findByText("1% Rule Compliance")).toBeInTheDocument();
    expect(screen.getByText("Hold Ratio (W/L)")).toBeInTheDocument();
    // Profit Factor label is now bare (no " · 2026" suffix) — appears in
    // both Last 10 panel and Discipline Pulse, so getAllByText.
    expect(screen.getAllByText("Profit Factor").length).toBeGreaterThanOrEqual(1);

    // All three tiles are wrapped in clickable links to specific tabs
    const drawdownLinks = screen.getAllByRole("link").filter(a => a.getAttribute("href") === "/analytics?tab=drawdown");
    const overviewLinks = screen.getAllByRole("link").filter(a => a.getAttribute("href") === "/analytics?tab=overview");
    expect(drawdownLinks.length).toBeGreaterThanOrEqual(1); // 1% Rule
    expect(overviewLinks.length).toBeGreaterThanOrEqual(2); // Hold Ratio + Profit Factor
  });

  test("1% Rule tile shows 'no losers in window' when no closed losses exist", async () => {
    mClosed.mockResolvedValue([
      { trade_id: "C1", ticker: "AAA", status: "CLOSED", open_date: "2026-02-01", closed_date: "2026-02-15", realized_pl: 5000 },
    ] as any);
    mOpen.mockResolvedValue([]);

    render(<Dashboard navColor="#6366f1" />);

    // The "no losers in window" subtitle should appear (1% Rule, Hold
    // Ratio, and Profit Factor tiles all share this copy when the
    // trailing window has no losers).
    const matches = await screen.findAllByText("no losers in window");
    expect(matches.length).toBeGreaterThanOrEqual(1);
  });

  test("Discipline Pulse header shows trailing-30 subtitle when ≥30 closed trades exist", async () => {
    // 32 closed trades — more than the trailing-30 window
    const trades = Array.from({ length: 32 }, (_, i) => {
      const m = String((i % 12) + 1).padStart(2, "0");
      const d = String((i % 28) + 1).padStart(2, "0");
      return {
        trade_id: `T${i}`, ticker: `TICK${i}`, status: "CLOSED",
        open_date: `2025-${m}-${d}`, closed_date: `2025-${m}-${d}`,
        realized_pl: i % 3 === 0 ? -800 : 1500,
      };
    });
    mClosed.mockResolvedValue(trades as any);
    mOpen.mockResolvedValue([]);
    mHistory.mockResolvedValue([{ day: "2025-01-01", end_nlv: 500_000 } as any]);

    render(<Dashboard navColor="#6366f1" />);

    expect(await screen.findByText("trailing 30 closed")).toBeInTheDocument();
    // LTD baseline subtitle appears (we don't pin the exact ratio — just that
    // the LTD prefix is present somewhere in the panel)
    const ltdMatches = await screen.findAllByText(/^LTD: /);
    expect(ltdMatches.length).toBeGreaterThanOrEqual(1);
  });

  test("Discipline Pulse subtitle shows 'trailing N closed' when fewer than 30 trades exist", async () => {
    const trades = Array.from({ length: 7 }, (_, i) => ({
      trade_id: `T${i}`, ticker: `TICK${i}`, status: "CLOSED",
      open_date: `2026-04-${String(20 - i).padStart(2, "0")}`,
      closed_date: `2026-04-${String(20 - i).padStart(2, "0")}`,
      realized_pl: 1000,
    }));
    mClosed.mockResolvedValue(trades as any);
    mOpen.mockResolvedValue([]);

    render(<Dashboard navColor="#6366f1" />);

    expect(await screen.findByText("trailing 7 closed")).toBeInTheDocument();
  });

  test("sequence strip squares are wrapped in Link to /trade-journal?trade_id=...", async () => {
    const trades = Array.from({ length: 3 }, (_, i) => ({
      trade_id: `T${i + 1}`, ticker: `TKR${i + 1}`, status: "CLOSED",
      open_date: `2026-04-${String(20 - i).padStart(2, "0")}`,
      closed_date: `2026-04-${String(20 - i).padStart(2, "0")}`,
      realized_pl: 1000,
    }));
    mClosed.mockResolvedValue(trades as any);
    mOpen.mockResolvedValue([]);

    render(<Dashboard navColor="#6366f1" />);

    const strip = await screen.findByTestId("last10-sequence");
    // Each child is a Link with the right href
    const hrefs = Array.from(strip.querySelectorAll("a")).map(a => a.getAttribute("href"));
    expect(hrefs).toContain("/trade-journal?trade_id=T1");
    expect(hrefs).toContain("/trade-journal?trade_id=T2");
    expect(hrefs).toContain("/trade-journal?trade_id=T3");
  });

  test("hovering a sequence square shows rich tooltip with ticker, P&L, status, days, rule", async () => {
    const trades = [{
      trade_id: "T1",
      ticker: "RICH",
      status: "CLOSED",
      open_date: "2026-04-01",
      closed_date: "2026-04-15",
      realized_pl: 2500,
      rule: "br1.1 Consolidation",
    }];
    mClosed.mockResolvedValue(trades as any);
    mOpen.mockResolvedValue([]);

    const { fireEvent } = await import("@testing-library/react");
    render(<Dashboard navColor="#6366f1" />);

    const strip = await screen.findByTestId("last10-sequence");
    const square = strip.querySelector("a");
    expect(square).not.toBeNull();

    fireEvent.mouseEnter(square!);

    const tooltip = await screen.findByTestId("last10-tooltip-0");
    expect(tooltip).toHaveTextContent("RICH");
    expect(tooltip).toHaveTextContent("+$2,500");
    expect(tooltip).toHaveTextContent("CLOSED");
    expect(tooltip).toHaveTextContent("2026-04-01");
    expect(tooltip).toHaveTextContent("br1.1 Consolidation");
  });
});
