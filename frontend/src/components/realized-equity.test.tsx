import { render, screen, waitFor } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach } from "vitest";

// Recharts depends on ResizeObserver under jsdom; stub it so the chart
// renderer stays quiet while still mounting the SVG. Same pattern as
// dashboard.test.tsx.
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

vi.mock("@/lib/api", () => ({
  api: {
    journalHistory: vi.fn(),
    realizedCurve: vi.fn(),
    events: vi.fn(),
  },
  getActivePortfolio: () => "CanSlim",
}));

import { api, type RealizedCurveResponse, type JournalHistoryPoint } from "@/lib/api";
import { RealizedEquity } from "./realized-equity";

const mHistory  = vi.mocked(api.journalHistory);
const mRealized = vi.mocked(api.realizedCurve);
const mEvents   = vi.mocked(api.events);


// Build a synthetic 6-day journal series starting 2026-01-01. SPY/Nasdaq
// LTD values are deliberately non-zero so the rebase-to-0 logic has
// something to subtract.
function makeHistory(): JournalHistoryPoint[] {
  const days = [
    "2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06",
    "2026-01-07", "2026-01-08",
  ];
  // SPY/Nasdaq cumulative %. First value becomes the baseline (rebased to 0).
  const spy = [10.0, 10.5, 10.2, 11.0, 11.5, 12.0];
  const ndx = [12.0, 12.4, 12.9, 13.5, 13.8, 14.2];
  return days.map((day, i) => ({
    id: i + 1, day,
    end_nlv: 100000, beg_nlv: 100000,
    daily_pct_change: 0,
    portfolio_ltd: 0, spy_ltd: spy[i], ndx_ltd: ndx[i],
    pct_invested: 0, portfolio_heat: 0,
  } as JournalHistoryPoint));
}

function makeRealized(): RealizedCurveResponse {
  return {
    series: [
      // Two closes — one on 2026-01-05 (mid-series), one on 2026-01-08 (last).
      { day: "2026-01-05", cum_realized_pl: 1000, cum_realized_pct: 1.0 },
      { day: "2026-01-08", cum_realized_pl: 2500, cum_realized_pct: 2.5 },
    ],
    summary: {
      total_realized_pl: 2500,
      realized_pct: 2.5,
      closed_count: 3,
      start_nlv: 100000,
      start_date: "2026-01-01",
      baseline_source: "journal",
    },
  };
}


function setupDefaults() {
  mHistory.mockResolvedValue(makeHistory() as any);
  mRealized.mockResolvedValue(makeRealized() as any);
  mEvents.mockResolvedValue([]);
}


describe("RealizedEquity — chart + summary render with sample data", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaults();
  });

  test("renders page heading + chart panel + benchmark legend rows", async () => {
    render(<RealizedEquity navColor="#6366f1" />);
    // Page mounts; wait for the chart panel which only appears once data
    // loads (no Loading… placeholder).
    await screen.findByTestId("realized-equity-chart-panel");

    // Heading (subtitle is specific enough to be unique).
    expect(screen.getByText(/cumulative realized return since 2026-01-01/i)).toBeInTheDocument();
    // Root mounted means the heading is present.
    expect(screen.getByTestId("realized-equity-root")).toBeInTheDocument();

    // Legend includes all three series with last-value chips.
    // Realized last value = 2.5%, SPY rebased to 0% → +2.0% (12.0 - 10.0),
    // Nasdaq rebased → +2.2% (14.2 - 12.0).
    const legend = screen.getByTestId("realized-equity-chart-panel");
    expect(legend.textContent).toMatch(/Realized \(\+2\.5%\)/);
    expect(legend.textContent).toMatch(/SPY \(\+2\.0%\)/);
    expect(legend.textContent).toMatch(/Nasdaq \(\+2\.2%\)/);
  });

  test("headline summary cards read from the realized summary", async () => {
    render(<RealizedEquity navColor="#6366f1" />);
    const hero = await screen.findByTestId("realized-equity-hero");

    // Banked $ (formatCurrency rounds to whole dollars per the {decimals:0}
    // arg). 2500 → "+$2,500".
    expect(hero.textContent).toContain("+$2,500");
    // Return %
    expect(hero.textContent).toContain("+2.50%");
    // Closed count
    expect(hero.textContent).toContain("3");
    expect(hero.textContent).toContain("lot closures");
    // Baseline source labelled in the % card's sub-line.
    expect(hero.textContent).toContain("journal");
  });
});


describe("RealizedEquity — sparse-series forward-fill semantics", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaults();
  });

  test("realized line steps on close dates and stays anchored to 2026-01-01", async () => {
    // We can't reliably introspect the <Line> data Recharts produced
    // inside the SVG without parsing path coordinates. Instead assert
    // the user-visible last-value chip — which is computed from the
    // final chart row, after the forward-fill. If forward-fill were
    // broken, the legend chip would read "0.0%" (no close on the final
    // day in the synthetic series) or it would re-rebase on range-zoom.
    // Either bug would surface as a wrong number here.
    render(<RealizedEquity navColor="#6366f1" />);
    const panel = await screen.findByTestId("realized-equity-chart-panel");

    // Realized series ends at 2.5% on 2026-01-08 (last fill); the chart's
    // final row (2026-01-08) inherits that value via forward-fill.
    expect(panel.textContent).toMatch(/Realized \(\+2\.5%\)/);
  });
});


describe("RealizedEquity — empty state", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  test("no closes yet → hero cards show 0/—, chart placeholder visible", async () => {
    // Mimic a brand-new portfolio: journal has rows (so we have a date
    // axis) but the realized endpoint returns an empty series with
    // zeroed summary.
    mHistory.mockResolvedValue(makeHistory() as any);
    mRealized.mockResolvedValue({
      series: [],
      summary: {
        total_realized_pl: 0,
        realized_pct: 0,
        closed_count: 0,
        start_nlv: 100000,
        start_date: "2026-01-01",
        baseline_source: "journal",
      },
    } as any);
    mEvents.mockResolvedValue([]);

    render(<RealizedEquity navColor="#6366f1" />);
    const hero = await screen.findByTestId("realized-equity-hero");

    // All three hero values reflect zero state. formatCurrency only
    // prepends "+" when value > 0, so a banked of $0 renders as "$0"
    // (the realized_pct branch builds its own sign explicitly and does
    // render "+0.00%").
    expect(hero.textContent).toContain("$0");       // Banked $
    expect(hero.textContent).toContain("+0.00%");   // Return %
    expect(hero.textContent).toContain("lot closures");
    expect(hero.textContent).toContain("Closed Trades");

    // The chart panel still renders (we have a journal date axis), but
    // the realized line is a flat 0% throughout — legend chip reflects.
    const panel = await screen.findByTestId("realized-equity-chart-panel");
    expect(panel.textContent).toMatch(/Realized \(\+0\.0%\)/);
  });

  test("no journal history at all → chart placeholder shown", async () => {
    mHistory.mockResolvedValue([] as any);
    mRealized.mockResolvedValue({
      series: [],
      summary: {
        total_realized_pl: 0,
        realized_pct: 0,
        closed_count: 0,
        start_nlv: 0,
        start_date: "2026-01-01",
        baseline_source: "none",
      },
    } as any);
    mEvents.mockResolvedValue([]);

    render(<RealizedEquity navColor="#6366f1" />);
    // No journal → no date axis → empty-state placeholder inside the
    // chart panel area.
    await waitFor(() =>
      expect(screen.getByTestId("realized-equity-empty")).toBeInTheDocument(),
    );
  });
});
