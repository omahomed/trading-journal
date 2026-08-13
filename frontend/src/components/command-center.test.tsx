import { render, screen, within, waitFor } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach } from "vitest";
import type { CommandCenterRow } from "@/lib/api";

vi.mock("@/lib/api", () => ({
  api: { commandCenter: vi.fn() },
}));

import { api } from "@/lib/api";
import { CommandCenter } from "./command-center";

const mCC = vi.mocked(api.commandCenter);

function row(overrides: Partial<CommandCenterRow>): CommandCenterRow {
  return {
    portfolio_id: 1,
    portfolio_name: "CanSlim",
    journal_available: true,
    as_of_date: "2026-08-12",
    nlv: 100000,
    nlv_delta_dollar: 0,
    nlv_delta_pct: 0,
    ltd_pct: 10.0,
    ltd_pl_dollar: 10000,
    ytd_pct: 5.0,
    ytd_pl_dollar: 5000,
    ytd_available: true,
    exposure_pct: 50,
    open_position_count: 3,
    cash: 50000,
    drawdown_current_pct: 0,
    drawdown_peak_nlv: 100000,
    drawdown_peak_date: "2026-08-11",
    ...overrides,
  };
}

beforeEach(() => vi.clearAllMocks());

describe("CommandCenter", () => {
  test("renders one row per portfolio and sorts worst-drawdown-first", async () => {
    mCC.mockResolvedValue({
      rows: [
        row({ portfolio_id: 1, portfolio_name: "CanSlim", drawdown_current_pct: -3.0 }),
        row({ portfolio_id: 2, portfolio_name: "LTG",     drawdown_current_pct: -18.5 }),
        row({ portfolio_id: 3, portfolio_name: "Diva",    drawdown_current_pct: -8.2 }),
      ],
    });

    render(<CommandCenter navColor="#334155" />);
    await waitFor(() => expect(screen.getByTestId("command-center-table")).toBeTruthy());

    const rows = screen.getAllByTestId("cc-row");
    // Worst first: LTG (-18.5) → Diva (-8.2) → CanSlim (-3.0)
    expect(rows.map(r => r.getAttribute("data-portfolio"))).toEqual(["LTG", "Diva", "CanSlim"]);
  });

  test("deck badge reflects classifyDeck for each row's drawdown", async () => {
    mCC.mockResolvedValue({
      rows: [
        row({ portfolio_name: "AllClear",  drawdown_current_pct: -3.0 }),   // L0
        row({ portfolio_name: "L1Portfolio", drawdown_current_pct: -8.0 }), // L1
        row({ portfolio_name: "L3Portfolio", drawdown_current_pct: -22.0 }),// L3
      ],
    });
    render(<CommandCenter navColor="#334155" />);
    await waitFor(() => screen.getByTestId("command-center-table"));

    const byPortfolio: Record<string, HTMLElement> = {};
    for (const r of screen.getAllByTestId("cc-row")) {
      byPortfolio[r.getAttribute("data-portfolio")!] = r;
    }
    expect(byPortfolio["AllClear"].getAttribute("data-deck")).toBe("L0");
    expect(byPortfolio["L1Portfolio"].getAttribute("data-deck")).toBe("L1");
    expect(byPortfolio["L3Portfolio"].getAttribute("data-deck")).toBe("L3");
  });

  test("portfolio with no journal renders row with 'no journal' pill", async () => {
    mCC.mockResolvedValue({
      rows: [
        row({
          portfolio_name: "Fresh",
          journal_available: false,
          nlv: null, drawdown_current_pct: null,
          nlv_delta_dollar: null, nlv_delta_pct: null,
          ltd_pct: null, ltd_pl_dollar: null,
          ytd_pct: null, ytd_pl_dollar: null, ytd_available: false,
          exposure_pct: null, cash: null, drawdown_peak_nlv: null,
        }),
      ],
    });
    render(<CommandCenter navColor="#334155" />);
    await waitFor(() => screen.getByTestId("command-center-table"));

    const freshRow = screen.getByTestId("cc-row");
    expect(freshRow.getAttribute("data-portfolio")).toBe("Fresh");
    expect(within(freshRow).getByText("no journal")).toBeTruthy();
    // Null-safe rendering: the "—" dashes cover NLV, day delta, LTD, YTD, drawdown.
    // The row should still classify to L0 (defensive default when drawdown is null).
    expect(freshRow.getAttribute("data-deck")).toBe("L0");
  });

  test("empty rows array shows empty state, not the table", async () => {
    mCC.mockResolvedValue({ rows: [] });
    render(<CommandCenter navColor="#334155" />);
    await waitFor(() => screen.getByTestId("command-center-empty"));
    expect(screen.queryByTestId("command-center-table")).toBeNull();
  });

  test("error response shows error banner and no table", async () => {
    mCC.mockResolvedValue({ error: "database unreachable" });
    render(<CommandCenter navColor="#334155" />);
    await waitFor(() => screen.getByTestId("command-center-error"));
    expect(screen.queryByTestId("command-center-table")).toBeNull();
  });

  test("null-drawdown portfolios sort to the bottom (below actionable rows)", async () => {
    mCC.mockResolvedValue({
      rows: [
        row({ portfolio_name: "Unknown", drawdown_current_pct: null }),
        row({ portfolio_name: "Down",    drawdown_current_pct: -12.0 }),
        row({ portfolio_name: "Flat",    drawdown_current_pct: 0.0 }),
      ],
    });
    render(<CommandCenter navColor="#334155" />);
    await waitFor(() => screen.getByTestId("command-center-table"));

    const rows = screen.getAllByTestId("cc-row");
    expect(rows.map(r => r.getAttribute("data-portfolio"))).toEqual(["Down", "Flat", "Unknown"]);
  });
});
