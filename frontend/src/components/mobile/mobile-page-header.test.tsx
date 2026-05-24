import { describe, test, expect } from "vitest";
import { render, screen } from "@testing-library/react";

import { MobilePageHeader } from "./mobile-page-header";

describe("MobilePageHeader — wordmark size", () => {
  test("renders the wordmark at the iOS large-title 30px size (Phase 2 Step 2.5 bump)", () => {
    render(<MobilePageHeader title="Position" italicWord="Sizer" />);
    const heading = screen.getByRole("heading", { level: 1 });
    expect(heading.className).toMatch(/text-\[30px\]/);
  });

  test("italic word still uses the green accent and serif display stack", () => {
    render(<MobilePageHeader title="Position" italicWord="Sizer" />);
    const em = screen.getByText("Sizer");
    expect(em.tagName).toBe("EM");
    expect(em.className).toMatch(/text-m-accent/);
    expect(em.className).toMatch(/font-m-display-italic/);
  });

  test("right slot still renders alongside the wordmark", () => {
    render(
      <MobilePageHeader
        title="Position"
        italicWord="Sizer"
        rightSlot={<span data-testid="rs">x</span>}
      />,
    );
    expect(screen.getByTestId("rs")).toBeInTheDocument();
  });
});
