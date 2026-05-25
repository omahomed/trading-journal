import { describe, test, expect, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

import { MobileScoreSelector } from "./mobile-score-selector";

describe("MobileScoreSelector", () => {
  test("renders 5 chips with the active value highlighted via aria-checked", () => {
    render(<MobileScoreSelector label="Plan" value={4} onChange={() => {}} />);
    const chips = screen.getAllByRole("radio");
    expect(chips).toHaveLength(5);
    expect(chips[3]).toHaveAttribute("aria-checked", "true");
    expect(chips[0]).toHaveAttribute("aria-checked", "false");
  });

  test("tapping a chip fires onChange with the new value", () => {
    const onChange = vi.fn();
    render(<MobileScoreSelector label="Plan" value={3} onChange={onChange} />);
    fireEvent.click(screen.getByTestId("score-chip-plan-5"));
    expect(onChange).toHaveBeenCalledWith(5);
  });

  test("default tier mapping: 1-2 low, 3 mid, 4-5 high (data-tier attribute)", () => {
    render(<MobileScoreSelector label="Stops" value={3} onChange={() => {}} />);
    expect(screen.getByTestId("score-chip-stops-1")).toHaveAttribute("data-tier", "low");
    expect(screen.getByTestId("score-chip-stops-2")).toHaveAttribute("data-tier", "low");
    expect(screen.getByTestId("score-chip-stops-3")).toHaveAttribute("data-tier", "mid");
    expect(screen.getByTestId("score-chip-stops-4")).toHaveAttribute("data-tier", "high");
    expect(screen.getByTestId("score-chip-stops-5")).toHaveAttribute("data-tier", "high");
  });

  test("tierFor override applies a custom mapping", () => {
    render(
      <MobileScoreSelector
        label="Custom"
        value={1}
        onChange={() => {}}
        tierFor={() => "high"}
      />,
    );
    // Every chip → high under the override.
    for (let v = 1; v <= 5; v++) {
      expect(screen.getByTestId(`score-chip-custom-${v}`)).toHaveAttribute("data-tier", "high");
    }
  });

  test("aria-label includes the label and the value (e.g. 'Plan: 3 of 5')", () => {
    render(<MobileScoreSelector label="Plan" value={3} onChange={() => {}} />);
    expect(screen.getByRole("radio", { name: "Plan: 3 of 5" })).toBeInTheDocument();
  });
});
