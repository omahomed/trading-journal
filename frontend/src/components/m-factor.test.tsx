import { render, screen } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach } from "vitest";

vi.mock("@/lib/api", () => ({
  api: {
    rallyPrefix: vi.fn(),
    marketSignals: vi.fn(),
    // 2026-08-04 hero tile 5 (Actual Exposure). Default mock returns
    // aligned exposure so the tile renders green; individual tests can
    // override to exercise misalignment banding.
    mfactorActualExposure: vi.fn().mockResolvedValue({
      actual_pct: 80, suggested_pct: 80, delta_pct: 0,
      market_value: 800000, nlv: 1000000, portfolios: [],
    }),
  },
}));

import { api } from "@/lib/api";
import { MFactor } from "./m-factor";

const mockedRallyPrefix = vi.mocked(api.rallyPrefix);
const mockedMarketSignals = vi.mocked(api.marketSignals);

const baseRallyPayload = {
  prefix: "Day 18: ",
  state: "POWERTREND" as const,
  day_num: 18,
  entry_step: 8,
  entry_exposure: 200,
  price: 24836.6,
  ema8: 24500.1,
  ema21: 24300.5,
  sma50: 23500.2,
  sma200: 22000.0,
  reference_high: 24854.04,
  reference_high_date: "2026-04-24",
  drawdown_pct: -0.07,
  consecutive_below_21: 0,
  active_exits: [],
  low_above_21_streak: 12,
  low_above_50_streak: 12,
  stack_8_21: true,
  stack_21_50: true,
  stack_50_200: true,
  entry_ladder: [
    { step: 0, label: "Rally Day", achieved: true, exposure: 20 },
    { step: 1, label: "Follow-Through Day", achieved: true, exposure: 40 },
    { step: 2, label: "Close > 21 EMA", achieved: true, exposure: 60 },
    { step: 3, label: "Low > 21 EMA", achieved: true, exposure: 80 },
    { step: 4, label: "Low > 21 EMA (3 days)", achieved: true, exposure: 100 },
    { step: 5, label: "Low > 50 SMA (3 days)", achieved: true, exposure: 120 },
    { step: 6, label: "21 EMA > 50 SMA > 200 SMA", achieved: true, exposure: 140 },
    { step: 7, label: "8 EMA > 21 EMA > 50 SMA > 200 SMA", achieved: true, exposure: 160 },
    { step: 8, label: "Power-Trend ON", achieved: true, exposure: 200 },
  ],
  ftd_date: "2026-04-08",
  data_as_of: "2026-04-24",
  power_trend_on_since: "2026-04-22",
  cap_at_100: false,
  cycle_start_date: "2026-03-31",
};

describe("MFactor — V11 augmented surface", () => {
  beforeEach(() => {
    mockedRallyPrefix.mockReset();
    mockedMarketSignals.mockReset();
    mockedMarketSignals.mockResolvedValue({ signals: [] });
  });

  test("renders POWERTREND state + 200% exposure in the hero tiles", async () => {
    mockedRallyPrefix.mockResolvedValue(baseRallyPayload);
    render(<MFactor navColor="#8b5cf6" />);
    // Post 2026-08-04 redesign: hero is a 5-tile grid, not a single
    // banner. State appears in the Status tile (data-testid=mfactor-tile-status);
    // exposure is a discrete value inside its own tile.
    const statusTile = await screen.findByTestId("mfactor-tile-status");
    expect(statusTile).toHaveTextContent("POWERTREND");
    const exposureTile = screen.getByTestId("mfactor-tile-exposure");
    expect(exposureTile).toHaveTextContent("Suggested Exposure");
    expect(exposureTile).toHaveTextContent("200%");
  });

  test("UPTREND UNDER PRESSURE renders in Status tile with day-count sub", async () => {
    // 5th-state tile. Under the 2026-08-04 tile redesign, the sub-line
    // reads "Day N · cycle started YYYY-MM-DD" when both are present.
    mockedRallyPrefix.mockResolvedValue({
      ...baseRallyPayload,
      state: "UPTREND UNDER PRESSURE",
      day_num: 42,
    });
    render(<MFactor navColor="#8b5cf6" />);
    const statusTile = await screen.findByTestId("mfactor-tile-status");
    expect(statusTile).toHaveTextContent("UPTREND UNDER PRESSURE");
    expect(statusTile).toHaveTextContent(/Day 42/);
  });

  test("renders cap_at_100 indicator when active", async () => {
    mockedRallyPrefix.mockResolvedValue({ ...baseRallyPayload, cap_at_100: true });
    render(<MFactor navColor="#8b5cf6" />);
    expect(await screen.findByText("Capped at 100%")).toBeInTheDocument();
  });

  test("hides cap_at_100 indicator when not active", async () => {
    mockedRallyPrefix.mockResolvedValue({ ...baseRallyPayload, cap_at_100: false });
    render(<MFactor navColor="#8b5cf6" />);
    await screen.findByTestId("mfactor-tile-status");
    expect(screen.queryByText("Capped at 100%")).not.toBeInTheDocument();
  });

  test("Status tile shows cycle_start_date + day_num when both present", async () => {
    mockedRallyPrefix.mockResolvedValue(baseRallyPayload);
    render(<MFactor navColor="#8b5cf6" />);
    const statusTile = await screen.findByTestId("mfactor-tile-status");
    // 2026-08-04 tile-redesign format: "Day 18 · cycle started 2026-03-31"
    expect(statusTile).toHaveTextContent(/Day 18/);
    expect(statusTile).toHaveTextContent(/cycle started 2026-03-31/);
  });

  test("renders Recent Signals section header", async () => {
    mockedRallyPrefix.mockResolvedValue(baseRallyPayload);
    render(<MFactor navColor="#8b5cf6" />);
    expect(await screen.findByText("Recent Signals")).toBeInTheDocument();
  });

  test("signal log shows fetched signals in the table", async () => {
    mockedRallyPrefix.mockResolvedValue(baseRallyPayload);
    mockedMarketSignals.mockResolvedValue({
      signals: [
        {
          trade_date: "2026-04-22",
          signal_type: "STEP_8_POWERTREND_ON",
          signal_label: "Power-Trend ON",
          exposure_before: 160,
          exposure_after: 200,
          state_before: "UPTREND",
          state_after: "POWERTREND",
          meta: {},
        },
      ],
    });
    render(<MFactor navColor="#8b5cf6" />);
    // Wait for the table cell to mount. The signal type also appears in the
    // filter <option>, so findAllByText is used.
    const matches = await screen.findAllByText("STEP_8_POWERTREND_ON");
    expect(matches.length).toBeGreaterThanOrEqual(1);
    // "Power-Trend ON" appears in both the banner ("Power-Trend ON since …")
    // and the signal label cell, so use getAllByText.
    expect(screen.getAllByText("Power-Trend ON").length).toBeGreaterThanOrEqual(1);
    expect(await screen.findByText("160% → 200%")).toBeInTheDocument();
  });
});
