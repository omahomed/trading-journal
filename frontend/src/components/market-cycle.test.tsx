import { render, screen } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach } from "vitest";

vi.mock("@/lib/api", () => ({
  api: {
    rallyPrefix: vi.fn(),
    marketSignals: vi.fn(),
  },
}));

import { api } from "@/lib/api";
import { MarketCycle } from "./market-cycle";

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

describe("MarketCycle — V11 augmented surface", () => {
  beforeEach(() => {
    mockedRallyPrefix.mockReset();
    mockedMarketSignals.mockReset();
    mockedMarketSignals.mockResolvedValue({ signals: [] });
  });

  test("renders POWERTREND state in the banner", async () => {
    mockedRallyPrefix.mockResolvedValue(baseRallyPayload);
    render(<MarketCycle navColor="#8b5cf6" />);
    // POWERTREND appears in both the state banner and the methodology table,
    // so findAllByText is used instead of findByText.
    const matches = await screen.findAllByText("POWERTREND");
    expect(matches.length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/Suggested Exposure: 200%/)).toBeInTheDocument();
  });

  test("renders cap_at_100 indicator when active", async () => {
    mockedRallyPrefix.mockResolvedValue({ ...baseRallyPayload, cap_at_100: true });
    render(<MarketCycle navColor="#8b5cf6" />);
    expect(await screen.findByText("Capped at 100%")).toBeInTheDocument();
  });

  test("hides cap_at_100 indicator when not active", async () => {
    mockedRallyPrefix.mockResolvedValue({ ...baseRallyPayload, cap_at_100: false });
    render(<MarketCycle navColor="#8b5cf6" />);
    await screen.findAllByText("POWERTREND");
    expect(screen.queryByText("Capped at 100%")).not.toBeInTheDocument();
  });

  test("renders cycle_start_date line when present and day_num > 0", async () => {
    mockedRallyPrefix.mockResolvedValue(baseRallyPayload);
    render(<MarketCycle navColor="#8b5cf6" />);
    expect(await screen.findByText(/Cycle started 2026-03-31 \(Day 18\)/)).toBeInTheDocument();
  });

  test("renders Recent Signals section header", async () => {
    mockedRallyPrefix.mockResolvedValue(baseRallyPayload);
    render(<MarketCycle navColor="#8b5cf6" />);
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
    render(<MarketCycle navColor="#8b5cf6" />);
    // Wait for the table cell to mount. The signal type also appears in the
    // filter <option>, so findAllByText is used.
    const matches = await screen.findAllByText("STEP_8_POWERTREND_ON");
    expect(matches.length).toBeGreaterThanOrEqual(1);
    // "Power-Trend ON" appears in both the banner ("Power-Trend ON since …")
    // and the signal label cell, so use getAllByText.
    expect(screen.getAllByText("Power-Trend ON").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("160% → 200%")).toBeInTheDocument();
  });
});
