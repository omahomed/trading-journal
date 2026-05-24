import { describe, test, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";

vi.mock("@/lib/use-rally-state", () => ({
  useRallyState: vi.fn(),
}));

import { useRallyState } from "@/lib/use-rally-state";
import { MobileTapePill } from "./mobile-tape-pill";

describe("MobileTapePill — Phase 2 'Since {MMM D}' format", () => {
  beforeEach(() => {
    vi.mocked(useRallyState).mockReset();
  });

  test("POWERTREND with power_trend_on_since shows 'Since {MMM D}'", () => {
    vi.mocked(useRallyState).mockReturnValue({
      state: "POWERTREND",
      day_num: 38,
      power_trend_on_since: "2026-04-14",
      cap_at_100: false,
    });
    render(<MobileTapePill />);
    expect(screen.getByText("POWERTREND")).toBeInTheDocument();
    expect(screen.getByText("Since Apr 14")).toBeInTheDocument();
    expect(screen.queryByText(/Day 38/)).not.toBeInTheDocument();
  });

  test("falls back to 'Day N' when power_trend_on_since is null", () => {
    vi.mocked(useRallyState).mockReturnValue({
      state: "POWERTREND",
      day_num: 38,
      power_trend_on_since: null,
      cap_at_100: false,
    });
    render(<MobileTapePill />);
    expect(screen.getByText("Day 38")).toBeInTheDocument();
    expect(screen.queryByText(/Since/)).not.toBeInTheDocument();
  });

  test("falls back to 'Day N' when power_trend_on_since is undefined", () => {
    vi.mocked(useRallyState).mockReturnValue({
      state: "UPTREND",
      day_num: 4,
      cap_at_100: false,
    });
    render(<MobileTapePill />);
    expect(screen.getByText("Day 4")).toBeInTheDocument();
  });

  test("CORRECTION still shows '{abs(drawdown_pct)}% off high' (unchanged)", () => {
    vi.mocked(useRallyState).mockReturnValue({
      state: "CORRECTION",
      drawdown_pct: -8.42,
      cap_at_100: false,
    });
    render(<MobileTapePill />);
    expect(screen.getByText(/8\.4% off high/)).toBeInTheDocument();
  });

  test("placeholder shown when useRallyState returns null", () => {
    vi.mocked(useRallyState).mockReturnValue(null);
    render(<MobileTapePill />);
    expect(screen.getByLabelText("Cycle indicator loading")).toBeInTheDocument();
  });
});
