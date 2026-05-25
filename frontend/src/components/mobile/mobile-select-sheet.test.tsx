import { describe, test, expect } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

import { MobileSelectSheet } from "./mobile-select-sheet";

describe("MobileSelectSheet — scrollable overflow (bugfix)", () => {
  test("dialog container has max-height + flex column so it can't push content above the viewport", () => {
    render(
      <MobileSelectSheet
        triggerLabel="Test"
        triggerValue="Pick"
        sheetTitle="Test sheet"
      >
        {() => <div>item</div>}
      </MobileSelectSheet>,
    );
    fireEvent.click(screen.getByRole("button", { name: /Test: Pick/ }));

    const dialog = screen.getByRole("dialog", { name: /Test sheet/ });
    expect(dialog.className).toMatch(/max-h-\[85vh\]/);
    expect(dialog.className).toMatch(/flex/);
    expect(dialog.className).toMatch(/flex-col/);
  });

  test("listbox is independently scrollable when the items list overflows", () => {
    const manyItems = Array.from({ length: 30 }).map((_, i) => `Item ${i + 1}`);
    render(
      <MobileSelectSheet
        triggerLabel="Test"
        triggerValue="Pick"
        sheetTitle="Long list"
      >
        {() => (
          <div>
            {manyItems.map((label) => (
              <button key={label} type="button" role="option">{label}</button>
            ))}
          </div>
        )}
      </MobileSelectSheet>,
    );
    fireEvent.click(screen.getByRole("button", { name: /Test: Pick/ }));

    const listbox = screen.getByRole("listbox", { name: /Long list/ });
    // overflow-y-auto + min-h-0 + flex-1 are the three classes that
    // make the items area shrink-and-scroll inside the flex column.
    expect(listbox.className).toMatch(/overflow-y-auto/);
    expect(listbox.className).toMatch(/min-h-0/);
    expect(listbox.className).toMatch(/flex-1/);
  });

  // Safe-area-inset-bottom is set via inline style on the listbox, but
  // jsdom drops `max()`/`env()` CSS expressions from the style attribute
  // entirely — there's no way to assert on the rendered value from a
  // unit test. Verified by reading the component source; visual
  // verification happens on-device.

  test("header stays fixed at top via shrink-0 so it doesn't scroll with items", () => {
    render(
      <MobileSelectSheet
        triggerLabel="Test"
        triggerValue="Pick"
        sheetTitle="Header test"
      >
        {() => <div>item</div>}
      </MobileSelectSheet>,
    );
    fireEvent.click(screen.getByRole("button", { name: /Test: Pick/ }));

    const heading = screen.getByRole("heading", { level: 2, name: /Header test/ });
    const headerRow = heading.parentElement as HTMLElement;
    expect(headerRow.className).toMatch(/shrink-0/);
  });
});
