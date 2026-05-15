import { render, screen } from "@testing-library/react";
import { describe, test, expect } from "vitest";

import { WeeklyInsightsTile } from "./weekly-insights-tile";

describe("WeeklyInsightsTile — Phase 5 gradient KPI tile", () => {
  test("renders a currency value with the +sign on positives", () => {
    render(
      <WeeklyInsightsTile
        label="Weekly P&L"
        value={21430}
        formatType="currency"
        gradient="linear-gradient(135deg, #10b981, #34d399)"
      />
    );
    expect(screen.getByText(/Weekly P&L/i)).toBeInTheDocument();
    expect(screen.getByText(/\+\$21,430/)).toBeInTheDocument();
    expect(screen.getByTestId("weekly-insights-tile").dataset.negative).toBe("false");
  });

  test("renders a percent value with two decimals and a + sign", () => {
    render(
      <WeeklyInsightsTile
        label="Weekly Return %"
        value={4.62}
        formatType="percent"
        gradient="linear-gradient(135deg, #0d6efd, #3b82f6)"
      />
    );
    expect(screen.getByText("+4.62%")).toBeInTheDocument();
  });

  test("renders subtitle when provided", () => {
    render(
      <WeeklyInsightsTile
        label="Win Rate"
        value={78}
        formatType="percent"
        subtitle="14W / 4L / 1F of 19"
        gradient="linear-gradient(135deg, #f97316, #fb923c)"
      />
    );
    expect(screen.getByText("14W / 4L / 1F of 19")).toBeInTheDocument();
  });

  test("negative value prefixes a ↓ glyph and marks data-negative=true", () => {
    render(
      <WeeklyInsightsTile
        label="Weekly P&L"
        value={-3210}
        formatType="currency"
        gradient="linear-gradient(135deg, #10b981, #34d399)"
      />
    );
    const tile = screen.getByTestId("weekly-insights-tile");
    expect(tile.dataset.negative).toBe("true");
    // The formatted negative currency itself.
    expect(screen.getByText(/-\$3,210/)).toBeInTheDocument();
    // The down-arrow glyph appears as a sibling node.
    expect(tile.textContent).toContain("↓");
  });

  test("null value renders the em-dash placeholder", () => {
    render(
      <WeeklyInsightsTile
        label="YTD %"
        value={null}
        formatType="percent"
        gradient="linear-gradient(135deg, #8b5cf6, #a78bfa)"
      />
    );
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  test("loading=true shows a skeleton placeholder, not the underlying value", () => {
    render(
      <WeeklyInsightsTile
        label="LTD %"
        value={87.4}
        formatType="percent"
        gradient="linear-gradient(135deg, #ec4899, #f472b6)"
        loading
      />
    );
    // Loading mode short-circuits to the ellipsis — number must not appear.
    expect(screen.queryByText(/87\.40%/)).toBeNull();
    expect(screen.getByText("…")).toBeInTheDocument();
  });
});
