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
    ["POWERTREND", "Power Trend"],
    ["UPTREND", "Uptrend"],
    ["RALLY MODE", "Rally Mode"],
    ["CORRECTION", "Correction"],
  ] as const)("renders %s with label %q", async (state, label) => {
    mockedRallyPrefix.mockResolvedValue({
      prefix: "Day 5: ",
      state,
      day_num: 5,
      drawdown_pct: -2.5,
      cap_at_100: false,
    });
    render(<TapeStatusPill />);
    expect(await screen.findByText(new RegExp(label))).toBeInTheDocument();
  });

  test("appends Day N for POWERTREND/UPTREND/RALLY MODE", async () => {
    mockedRallyPrefix.mockResolvedValue({
      prefix: "",
      state: "POWERTREND",
      day_num: 18,
      cap_at_100: false,
    });
    render(<TapeStatusPill />);
    expect(await screen.findByText(/Day 18/)).toBeInTheDocument();
  });

  test("CORRECTION shows drawdown_pct, not Day N", async () => {
    mockedRallyPrefix.mockResolvedValue({
      prefix: "",
      state: "CORRECTION",
      day_num: 0,
      drawdown_pct: -8.42,
    });
    render(<TapeStatusPill />);
    expect(await screen.findByText(/-8\.4%/)).toBeInTheDocument();
  });

  test("renders lock icon when cap_at_100 is true", async () => {
    mockedRallyPrefix.mockResolvedValue({
      prefix: "",
      state: "POWERTREND",
      day_num: 18,
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
      cap_at_100: false,
    });
    render(<TapeStatusPill />);
    await screen.findByText(/Power Trend/);
    expect(screen.queryByLabelText("Capped at 100%")).not.toBeInTheDocument();
  });

  test("links to /market-cycle", async () => {
    mockedRallyPrefix.mockResolvedValue({
      prefix: "",
      state: "UPTREND",
      day_num: 4,
    });
    render(<TapeStatusPill />);
    const link = await screen.findByRole("link");
    expect(link).toHaveAttribute("href", "/market-cycle");
  });
});
