import { render, screen } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach } from "vitest";

vi.mock("@/lib/api", () => ({
  api: { rallyPrefix: vi.fn() },
}));

import { api } from "@/lib/api";
import { TapeStatusPill } from "./tape-status-pill";

const mockedRallyPrefix = vi.mocked(api.rallyPrefix);

describe("TapeStatusPill — V11 state rendering", () => {
  beforeEach(() => mockedRallyPrefix.mockReset());

  test.each([
    ["POWERTREND", "Power-Trend"],
    ["UPTREND", "Confirmed Uptrend"],
    ["RALLY MODE", "Rally Mode"],
    ["CORRECTION", "Correction"],
  ] as const)("renders %s with label %q", async (state, label) => {
    mockedRallyPrefix.mockResolvedValue({
      prefix: "Day 5: ",
      state,
      day_num: 5,
      drawdown_pct: -2.5,
      power_trend_on_since: "2026-04-22",
      ftd_date: "2026-04-08",
      cap_at_100: false,
    });
    render(<TapeStatusPill />);
    expect(await screen.findByText(new RegExp(label))).toBeInTheDocument();
  });

  test("POWERTREND uses 'since {power_trend_on_since}' formatted as Mon DD", async () => {
    mockedRallyPrefix.mockResolvedValue({
      prefix: "",
      state: "POWERTREND",
      day_num: 18,
      power_trend_on_since: "2026-04-22",
      cap_at_100: false,
    });
    render(<TapeStatusPill />);
    expect(await screen.findByText(/since Apr 22/)).toBeInTheDocument();
  });

  test("UPTREND uses 'since {ftd_date}' formatted as Mon DD", async () => {
    mockedRallyPrefix.mockResolvedValue({
      prefix: "",
      state: "UPTREND",
      day_num: 4,
      ftd_date: "2026-04-08",
      cap_at_100: false,
    });
    render(<TapeStatusPill />);
    expect(await screen.findByText(/since Apr 8/)).toBeInTheDocument();
  });

  test("RALLY MODE uses 'Day N' from cycle_day (kept from staging)", async () => {
    mockedRallyPrefix.mockResolvedValue({
      prefix: "",
      state: "RALLY MODE",
      day_num: 7,
      cap_at_100: false,
    });
    render(<TapeStatusPill />);
    expect(await screen.findByText(/Day 7/)).toBeInTheDocument();
  });

  test("CORRECTION shows '{abs(drawdown_pct)}% off high'", async () => {
    mockedRallyPrefix.mockResolvedValue({
      prefix: "",
      state: "CORRECTION",
      day_num: 0,
      drawdown_pct: -8.42,
    });
    render(<TapeStatusPill />);
    expect(await screen.findByText(/8\.4% off high/)).toBeInTheDocument();
  });

  test("renders lock icon when cap_at_100 is true", async () => {
    mockedRallyPrefix.mockResolvedValue({
      prefix: "",
      state: "POWERTREND",
      day_num: 18,
      power_trend_on_since: "2026-04-22",
      cap_at_100: true,
    });
    render(<TapeStatusPill />);
    expect(await screen.findByLabelText("Capped at 100%")).toBeInTheDocument();
  });

  test("hides lock icon when cap_at_100 is false", async () => {
    mockedRallyPrefix.mockResolvedValue({
      prefix: "",
      state: "POWERTREND",
      day_num: 18,
      power_trend_on_since: "2026-04-22",
      cap_at_100: false,
    });
    render(<TapeStatusPill />);
    await screen.findByText(/Power-Trend/);
    expect(screen.queryByLabelText("Capped at 100%")).not.toBeInTheDocument();
  });

  test("links to /m-factor", async () => {
    mockedRallyPrefix.mockResolvedValue({
      prefix: "",
      state: "UPTREND",
      day_num: 4,
      ftd_date: "2026-04-08",
    });
    render(<TapeStatusPill />);
    const link = await screen.findByRole("link");
    expect(link).toHaveAttribute("href", "/m-factor");
  });
});
