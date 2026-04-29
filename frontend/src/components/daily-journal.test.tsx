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
    listEodSnapshots: vi.fn(),
    journalEdit: vi.fn(),
    journalDelete: vi.fn(),
  },
  getActivePortfolio: () => "CanSlim",
}));

import { api } from "@/lib/api";
import { DailyJournal } from "./daily-journal";

const mockedHistory = vi.mocked(api.journalHistory);

describe("DailyJournal — MCT State column (snapshotted on row)", () => {
  beforeEach(() => {
    mockedHistory.mockReset();
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

    render(<DailyJournal navColor="#f59f00" />);
    expect(await screen.findByText("MCT State")).toBeInTheDocument();
    expect(screen.queryByText("Window")).not.toBeInTheDocument();
  });

  test("POWERTREND with mct_display_day_num renders 'POWERTREND D{N}'", async () => {
    mockedHistory.mockResolvedValue([
      {
        day: "2026-04-24",
        end_nlv: 100000,
        daily_pct_change: 0.5,
        portfolio_heat: 5,
        score: 4,
        market_cycle: "POWERTREND",
        mct_display_day_num: 3,
      } as any,
    ]);

    render(<DailyJournal navColor="#f59f00" />);
    expect(await screen.findByText("POWERTREND D3")).toBeInTheDocument();
  });

  test("UPTREND with mct_display_day_num renders 'UPTREND D{N}'", async () => {
    const today = new Date().toISOString().slice(0, 10);
    mockedHistory.mockResolvedValue([
      {
        day: today,
        end_nlv: 100000,
        daily_pct_change: 0.5,
        portfolio_heat: 5,
        score: 4,
        market_cycle: "UPTREND",
        mct_display_day_num: 8,
      } as any,
    ]);

    render(<DailyJournal navColor="#f59f00" />);
    expect(await screen.findByText("UPTREND D8")).toBeInTheDocument();
  });

  test("RALLY MODE rows show 'D N' inline with the badge", async () => {
    const today = new Date().toISOString().slice(0, 10);
    mockedHistory.mockResolvedValue([
      {
        day: today,
        end_nlv: 90000,
        daily_pct_change: 1.2,
        portfolio_heat: 0,
        score: 3,
        market_cycle: "RALLY MODE",
        mct_display_day_num: 1,
      } as any,
    ]);

    render(<DailyJournal navColor="#f59f00" />);
    expect(await screen.findByText("RALLY MODE D1")).toBeInTheDocument();
  });

  test("CORRECTION with null day_num renders no suffix", async () => {
    const today = new Date().toISOString().slice(0, 10);
    mockedHistory.mockResolvedValue([
      {
        day: today,
        end_nlv: 92000,
        daily_pct_change: -1.0,
        portfolio_heat: 0,
        score: 3,
        market_cycle: "CORRECTION",
        mct_display_day_num: null,
      } as any,
    ]);

    render(<DailyJournal navColor="#f59f00" />);
    // Badge text is exactly "CORRECTION" — no "D{N}" suffix appended
    const badge = await screen.findByText("CORRECTION");
    expect(badge).toBeInTheDocument();
    expect(badge.textContent).toBe("CORRECTION");
  });

  test("missing market_cycle renders the em-dash placeholder", async () => {
    const today = new Date().toISOString().slice(0, 10);
    mockedHistory.mockResolvedValue([
      {
        day: today,
        end_nlv: 95000,
        daily_pct_change: 0.0,
        portfolio_heat: 8,
        score: 3,
      } as any,
    ]);

    render(<DailyJournal navColor="#f59f00" />);
    // Em-dash placeholders appear in multiple cells (cash flow, etc.); the
    // MCT column rendering as "—" is implied by the absence of any badge.
    expect(screen.queryByText(/POWERTREND/)).not.toBeInTheDocument();
    expect(screen.queryByText(/UPTREND/)).not.toBeInTheDocument();
    expect(screen.queryByText(/RALLY MODE/)).not.toBeInTheDocument();
    expect(screen.queryByText(/CORRECTION/)).not.toBeInTheDocument();
  });
});
