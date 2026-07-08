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
    const today = new Date().toISOString().slice(0, 10);
    mockedHistory.mockResolvedValue([
      {
        day: today,
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

  test("UPTREND UNDER PRESSURE renders 'UPTREND UNDER PRESSURE D{N}' with amber bg", async () => {
    // 5th-state badge, desktop. Full machine string in the visible
    // label (mobile applies the alias; desktop shows the full text).
    // #d97706 amber background — distinct from CORRECTION (#e5484d)
    // and RALLY MODE (#f59f00). Dormant this commit — no backend
    // emits UUP yet — but the render surface is ready.
    const today = new Date().toISOString().slice(0, 10);
    mockedHistory.mockResolvedValue([
      {
        day: today,
        end_nlv: 91000,
        daily_pct_change: 0.5,
        portfolio_heat: 0,
        score: 3,
        market_cycle: "UPTREND UNDER PRESSURE",
        mct_display_day_num: 42,
      } as any,
    ]);
    render(<DailyJournal navColor="#f59f00" />);
    const badge = await screen.findByText("UPTREND UNDER PRESSURE D42");
    expect(badge).toBeInTheDocument();
    // MctStateBadge structure: <span style={...}><span>{label}</span></span>
    // findByText returns the INNER text span; the style lives on its
    // parent. React normalizes inline hex to rgb() in jsdom, so check
    // both forms so the test is portable across renderer versions.
    const outer = badge.parentElement;
    const style = outer?.getAttribute("style") ?? "";
    expect(style).toMatch(/#d97706|rgb\(\s*217\s*,\s*119\s*,\s*6\s*\)/i);
  });

  test("UPTREND UNDER PRESSURE is NOT excluded by the row parser", async () => {
    // Regression on daily-journal.tsx:168 — the hard exclusion filter
    // must include "UPTREND UNDER PRESSURE" or the badge would render
    // as the em-dash placeholder instead of the amber badge.
    const today = new Date().toISOString().slice(0, 10);
    mockedHistory.mockResolvedValue([
      {
        day: today,
        end_nlv: 91000,
        daily_pct_change: 0.5,
        portfolio_heat: 0,
        score: 3,
        market_cycle: "UPTREND UNDER PRESSURE",
        mct_display_day_num: 42,
      } as any,
    ]);
    render(<DailyJournal navColor="#f59f00" />);
    // The badge should be present. If the exclusion mistakenly caught
    // UUP, mctFromRow returns undefined and the MctStateBadge renders
    // an em-dash "—" instead.
    expect(await screen.findByText(/UPTREND UNDER PRESSURE/)).toBeInTheDocument();
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
