import { render, screen, fireEvent, act } from "@testing-library/react";
import { describe, test, expect, vi, afterEach } from "vitest";

import { ColorPicker } from "./color-picker";

const PALETTE = [
  "#fef3c7",
  "#dbeafe",
  "#dcfce7",
  "#fee2e2",
  "#fde68a",
  "#e0e7ff",
  "#fce7f3",
  "transparent",     // reset sentinel — last entry
];

function setup(onPick = vi.fn(), props: Partial<React.ComponentProps<typeof ColorPicker>> = {}) {
  render(
    <ColorPicker
      ariaLabel="Highlight color"
      triggerGlyph={<span data-testid="trigger-glyph">H</span>}
      triggerSwatch="#fef3c7"
      palette={PALETTE}
      resetLabel="None"
      onPick={onPick}
      {...props}
    />,
  );
}

describe("ColorPicker", () => {
  afterEach(() => { vi.restoreAllMocks(); });

  test("trigger button renders with ariaLabel", () => {
    setup();
    expect(screen.getByRole("button", { name: /Highlight color/i })).toBeInTheDocument();
  });

  test("click trigger opens the popover with palette swatches", () => {
    setup();
    const trigger = screen.getByRole("button", { name: /Highlight color/i });
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    act(() => { fireEvent.click(trigger); });
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("dialog", { name: /Highlight color/i })).toBeInTheDocument();
    // 7 colored swatches + 1 reset swatch = 8 option buttons inside the dialog.
    // (The trigger button is OUTSIDE the dialog, so getAllByRole within the
    // dialog counts only the swatch buttons.)
    const dialog = screen.getByRole("dialog", { name: /Highlight color/i });
    const swatches = dialog.querySelectorAll("button");
    expect(swatches.length).toBe(PALETTE.length);
  });

  test("clicking a swatch calls onPick with that color and closes the popover", () => {
    const onPick = vi.fn();
    setup(onPick);
    const trigger = screen.getByRole("button", { name: /Highlight color/i });
    act(() => { fireEvent.click(trigger); });
    // The third palette entry has aria-label "#dcfce7".
    const swatch = screen.getByRole("button", { name: "#dcfce7" });
    act(() => { fireEvent.click(swatch); });
    expect(onPick).toHaveBeenCalledWith("#dcfce7");
    // Popover closed.
    expect(screen.queryByRole("dialog", { name: /Highlight color/i })).not.toBeInTheDocument();
    expect(trigger).toHaveAttribute("aria-expanded", "false");
  });

  test("clicking the reset swatch calls onPick with the reset sentinel", () => {
    const onPick = vi.fn();
    setup(onPick);
    act(() => { fireEvent.click(screen.getByRole("button", { name: /Highlight color/i })); });
    // Reset swatch is labeled "None" via the resetLabel prop.
    const resetBtn = screen.getByRole("button", { name: /^None$/i });
    act(() => { fireEvent.click(resetBtn); });
    expect(onPick).toHaveBeenCalledWith("transparent");
  });

  test("Esc closes the popover", () => {
    setup();
    const trigger = screen.getByRole("button", { name: /Highlight color/i });
    act(() => { fireEvent.click(trigger); });
    expect(screen.getByRole("dialog", { name: /Highlight color/i })).toBeInTheDocument();
    act(() => { fireEvent.keyDown(window, { key: "Escape" }); });
    expect(screen.queryByRole("dialog", { name: /Highlight color/i })).not.toBeInTheDocument();
  });

  test("click outside closes the popover", () => {
    setup();
    const trigger = screen.getByRole("button", { name: /Highlight color/i });
    act(() => { fireEvent.click(trigger); });
    expect(screen.getByRole("dialog", { name: /Highlight color/i })).toBeInTheDocument();
    // Mousedown on document.body fires the outside-click listener.
    act(() => { fireEvent.mouseDown(document.body); });
    expect(screen.queryByRole("dialog", { name: /Highlight color/i })).not.toBeInTheDocument();
  });

  test("triggerGlyph renders inside the trigger button", () => {
    setup();
    expect(screen.getByTestId("trigger-glyph")).toBeInTheDocument();
  });

  test("works without a triggerSwatch (text-color use case)", () => {
    render(
      <ColorPicker
        ariaLabel="Text color"
        triggerGlyph={<span data-testid="glyph">T</span>}
        palette={PALETTE}
        resetLabel="Default"
        onPick={vi.fn()}
      />,
    );
    expect(screen.getByRole("button", { name: /Text color/i })).toBeInTheDocument();
    expect(screen.getByTestId("glyph")).toBeInTheDocument();
  });
});
