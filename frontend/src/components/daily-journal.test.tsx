import { render, screen } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach } from "vitest";

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    refresh: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    prefetch: vi.fn(),
  }),
}));

vi.mock("@/lib/api", () => ({
  api: {
    journalHistory: vi.fn(),
    mctStateByDateRange: vi.fn(),
    listEodSnapshots: vi.fn(),
    journalEdit: vi.fn(),
    journalDelete: vi.fn(),
  },
  getActivePortfolio: () => "CanSlim",
}));

import { api } from "@/lib/api";
import { DailyJournal } from "./daily-journal";

const mockedHistory = vi.mocked(api.journalHistory);
const mockedMctStates = vi.mocked(api.mctStateByDateRange);

describe("DailyJournal — V11 MCT State column", () => {
  beforeEach(() => {
    mockedHistory.mockReset();
    mockedMctStates.mockReset();
  });

  test("renders 'MCT State' column header (not 'Window')", async () => {
    mockedHistory.mockResolvedValue([
      {
        day: "2026-04-24",
        end_nlv: 100000,
        daily_pct_change: 0.5,
        portfolio_heat: 5,
        score: 4,
      } as any,
    ]);
    mockedMctStates.mockResolvedValue({ states: [] });

    render(<DailyJournal navColor="#f59f00" />);
    expect(await screen.findByText("MCT State")).toBeInTheDocument();
    expect(screen.queryByText("Window")).not.toBeInTheDocument();
  });

  test("populates MCT State badge from /api/journal/mct-state-by-date-range", async () => {
    mockedHistory.mockResolvedValue([
      {
        day: "2026-04-24",
        end_nlv: 100000,
        daily_pct_change: 0.5,
        portfolio_heat: 5,
        score: 4,
      } as any,
    ]);
    mockedMctStates.mockResolvedValue({
      states: [
        {
          trade_date: "2026-04-24",
          state: "POWERTREND",
          exposure_ceiling: 200,
          cap_at_100: false,
          cycle_day: 18,
          in_correction: false,
          correction_active: false,
          power_trend: true,
        },
      ],
    });

    render(<DailyJournal navColor="#f59f00" />);
    expect(await screen.findByText("POWERTREND")).toBeInTheDocument();
  });

  test("RALLY MODE rows show 'Day N' inline with the badge", async () => {
    // Synthetic recent date so the default "Current Week" filter doesn't
    // hide the row. The badge logic doesn't care about historical accuracy.
    const today = new Date().toISOString().slice(0, 10);
    mockedHistory.mockResolvedValue([
      {
        day: today,
        end_nlv: 90000,
        daily_pct_change: 1.2,
        portfolio_heat: 0,
        score: 3,
      } as any,
    ]);
    mockedMctStates.mockResolvedValue({
      states: [
        {
          trade_date: today,
          state: "RALLY MODE",
          exposure_ceiling: 20,
          cap_at_100: false,
          cycle_day: 1,
          in_correction: true,
          correction_active: true,
          power_trend: false,
        },
      ],
    });

    render(<DailyJournal navColor="#f59f00" />);
    expect(await screen.findByText("RALLY MODE D1")).toBeInTheDocument();
  });

  test("cap_at_100=true renders the lock icon next to the badge", async () => {
    const today = new Date().toISOString().slice(0, 10);
    mockedHistory.mockResolvedValue([
      {
        day: today,
        end_nlv: 95000,
        daily_pct_change: -0.4,
        portfolio_heat: 8,
        score: 3,
      } as any,
    ]);
    mockedMctStates.mockResolvedValue({
      states: [
        {
          trade_date: today,
          state: "RALLY MODE",
          exposure_ceiling: 100,
          cap_at_100: true,
          cycle_day: 5,
          in_correction: true,
          correction_active: true,
          power_trend: false,
        },
      ],
    });

    render(<DailyJournal navColor="#f59f00" />);
    expect(await screen.findByLabelText("Capped at 100%")).toBeInTheDocument();
  });
});
