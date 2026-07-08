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

  test("UPTREND UNDER PRESSURE: renders display alias 'Uptrend · Pressure' + deep-amber dot", () => {
    // 5th-state mobile pill. Visible label swaps to the shortened
    // alias (mobile width is tight). Machine string stays in the
    // aria-label so screen readers + downstream matchers get the
    // byte-identical value. Dot backgroundColor overrides to the
    // deep-amber warn token (--m-warn-deep = #d97706). Dormant this
    // commit — no backend emits UUP yet.
    vi.mocked(useRallyState).mockReturnValue({
      state: "UPTREND UNDER PRESSURE",
      day_num: 42,
      ftd_date: "2026-04-08",
      cap_at_100: false,
    });
    render(<MobileTapePill />);
    // Visible label: shortened alias present, full machine string NOT
    // rendered as visible text.
    expect(screen.getByText("Uptrend · Pressure")).toBeInTheDocument();
    expect(screen.queryByText("UPTREND UNDER PRESSURE")).not.toBeInTheDocument();
    // aria-label carries the full machine string.
    expect(
      screen.getByLabelText(/Cycle: UPTREND UNDER PRESSURE/),
    ).toBeInTheDocument();
    // Dot color: inline style is set to var(--m-warn-deep) when
    // state is UUP. Query the first <span> child of the link.
    const link = screen.getByRole("link");
    const dot = link.querySelector("span[aria-hidden='true']");
    expect(dot?.getAttribute("style") || "").toContain("--m-warn-deep");
  });

  test("placeholder shown when useRallyState returns null", () => {
    vi.mocked(useRallyState).mockReturnValue(null);
    render(<MobileTapePill />);
    expect(screen.getByLabelText("Cycle indicator loading")).toBeInTheDocument();
  });
});
