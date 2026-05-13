import { render, screen, fireEvent } from "@testing-library/react";
import { describe, test, expect, vi } from "vitest";

import { TagPill } from "./tag-pill";

describe("TagPill", () => {
  test("renders the label text", () => {
    render(<TagPill label="drawdown" tone="rose" />);
    expect(screen.getByText("drawdown")).toBeInTheDocument();
  });

  test("does not render the remove button when onRemove is absent (read-only mode)", () => {
    render(<TagPill label="drawdown" tone="rose" />);
    expect(screen.queryByRole("button", { name: /remove drawdown/i })).not.toBeInTheDocument();
  });

  test("renders the remove button and calls onRemove when clicked", () => {
    const onRemove = vi.fn();
    render(<TagPill label="drawdown" tone="rose" onRemove={onRemove} />);
    const btn = screen.getByRole("button", { name: /remove drawdown/i });
    fireEvent.click(btn);
    expect(onRemove).toHaveBeenCalledTimes(1);
  });

  test("falls back to sky palette for an unknown tone (defensive)", () => {
    // @ts-expect-error — exercising the defensive fallback path.
    render(<TagPill label="x" tone="bogus" />);
    // Just assert it renders something rather than crashing on the fallback.
    expect(screen.getByText("x")).toBeInTheDocument();
  });
});
