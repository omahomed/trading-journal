import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

// Mock the shared portfolio context — keeps the unit test synchronous
// and lets us assert on `setActive` calls without triggering the
// `window.location.reload()` inside the real provider.
vi.mock("@/lib/portfolio-context", () => ({
  usePortfolio: vi.fn(),
}));

import { usePortfolio } from "@/lib/portfolio-context";
import { MobilePortfolioPicker } from "./mobile-portfolio-picker";
import type { Portfolio } from "@/lib/api";

const mockUsePortfolio = vi.mocked(usePortfolio);

function makePortfolio(id: number, name: string): Portfolio {
  return {
    id,
    name,
    starting_capital: 100000,
    reset_date: null,
    created_at: "2026-01-01T00:00:00Z",
    cash_balance: 0,
  };
}

const CANSLIM = makePortfolio(1, "CanSlim");
const PLAN_457B = makePortfolio(2, "457B Plan");
const LTG = makePortfolio(3, "LTG");

function setMockPortfolio(opts: {
  portfolios?: Portfolio[];
  activePortfolio?: Portfolio | null;
  loading?: boolean;
  setActive?: (p: Portfolio) => void;
}) {
  mockUsePortfolio.mockReturnValue({
    portfolios: opts.portfolios ?? [CANSLIM, PLAN_457B, LTG],
    // Distinguish "key not present" (default to CanSlim) from explicit
    // null (no active portfolio). Plain `?? CANSLIM` collapses both.
    activePortfolio:
      "activePortfolio" in opts ? opts.activePortfolio ?? null : CANSLIM,
    loading: opts.loading ?? false,
    error: null,
    refetch: vi.fn(),
    setActive: opts.setActive ?? vi.fn(),
  });
}

describe("MobilePortfolioPicker", () => {
  beforeEach(() => {
    mockUsePortfolio.mockReset();
  });
  afterEach(() => {
    // The component sets body.style.overflow = "hidden" when open; the
    // cleanup effect restores it on unmount, but defensive reset keeps
    // tests independent.
    document.body.style.overflow = "";
  });

  test("renders the active portfolio name on the closed-state button", () => {
    setMockPortfolio({ activePortfolio: PLAN_457B });
    render(<MobilePortfolioPicker />);
    // Button label includes the active portfolio name; aria-label spells it out.
    expect(
      screen.getByRole("button", { name: /Active portfolio: 457B Plan/i }),
    ).toBeInTheDocument();
  });

  test("tapping the button opens the sheet listing all portfolios", () => {
    setMockPortfolio({});
    render(<MobilePortfolioPicker />);
    fireEvent.click(screen.getByRole("button", { name: /Active portfolio/i }));

    expect(screen.getByRole("dialog", { name: "Switch portfolio" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "CanSlim" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "457B Plan" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "LTG" })).toBeInTheDocument();
  });

  test("the active portfolio row is aria-selected; others are not", () => {
    setMockPortfolio({ activePortfolio: PLAN_457B });
    render(<MobilePortfolioPicker />);
    fireEvent.click(screen.getByRole("button", { name: /Active portfolio/i }));

    expect(screen.getByRole("option", { name: "457B Plan" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(screen.getByRole("option", { name: "CanSlim" })).toHaveAttribute(
      "aria-selected",
      "false",
    );
  });

  test("tapping a non-active portfolio calls setActive with that portfolio and closes the sheet", () => {
    const setActive = vi.fn();
    setMockPortfolio({ activePortfolio: CANSLIM, setActive });
    render(<MobilePortfolioPicker />);
    fireEvent.click(screen.getByRole("button", { name: /Active portfolio/i }));

    fireEvent.click(screen.getByRole("option", { name: "LTG" }));

    expect(setActive).toHaveBeenCalledTimes(1);
    expect(setActive).toHaveBeenCalledWith(LTG);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  test("tapping the close X dismisses without switching", () => {
    const setActive = vi.fn();
    setMockPortfolio({ setActive });
    render(<MobilePortfolioPicker />);
    fireEvent.click(screen.getByRole("button", { name: /Active portfolio/i }));

    fireEvent.click(screen.getByRole("button", { name: "Close" }));

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(setActive).not.toHaveBeenCalled();
  });

  test("tapping the backdrop dismisses without switching", () => {
    const setActive = vi.fn();
    setMockPortfolio({ setActive });
    render(<MobilePortfolioPicker />);
    fireEvent.click(screen.getByRole("button", { name: /Active portfolio/i }));

    fireEvent.click(screen.getByRole("button", { name: "Close portfolio picker" }));

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(setActive).not.toHaveBeenCalled();
  });

  test("pressing Escape closes the sheet", () => {
    setMockPortfolio({});
    render(<MobilePortfolioPicker />);
    fireEvent.click(screen.getByRole("button", { name: /Active portfolio/i }));
    expect(screen.getByRole("dialog")).toBeInTheDocument();

    fireEvent.keyDown(document, { key: "Escape" });

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  test("disables the button and shows '—' when the portfolio list is empty", () => {
    setMockPortfolio({ portfolios: [], activePortfolio: null });
    render(<MobilePortfolioPicker />);
    const btn = screen.getByRole("button", { name: /Active portfolio: —/ });
    expect(btn).toBeDisabled();
  });

  test("shows 'Loading…' while loading with no active portfolio", () => {
    setMockPortfolio({ portfolios: [], activePortfolio: null, loading: true });
    render(<MobilePortfolioPicker />);
    expect(
      screen.getByRole("button", { name: /Active portfolio: Loading…/ }),
    ).toBeInTheDocument();
  });
});

describe("MobilePortfolioPicker — scrollable overflow (bugfix mirror)", () => {
  // Mirrors mobile-select-sheet.test.tsx's structural assertions. The
  // hand-rolled sheet here had the same shape as MobileSelectSheet's
  // pre-fix layout (no max-height, no overflow handling); this fix
  // pulls in the same flex/min-h-0/overflow-y-auto trick verbatim so
  // long portfolio lists shrink-and-scroll instead of pushing content
  // above the viewport.
  //
  // Safe-area-inset-bottom is set via inline style on the listbox, but
  // jsdom drops `max()`/`env()` CSS expressions from the style
  // attribute — there's no way to assert on the rendered value from a
  // unit test. Verified by reading the component source; visual
  // verification happens on-device.
  beforeEach(() => {
    mockUsePortfolio.mockReset();
  });
  afterEach(() => {
    document.body.style.overflow = "";
  });

  test("dialog container has max-height + flex column so it can't push content above the viewport", () => {
    setMockPortfolio({});
    render(<MobilePortfolioPicker />);
    fireEvent.click(screen.getByRole("button", { name: /Active portfolio/i }));

    const dialog = screen.getByRole("dialog", { name: "Switch portfolio" });
    expect(dialog.className).toMatch(/max-h-\[85vh\]/);
    expect(dialog.className).toMatch(/flex/);
    expect(dialog.className).toMatch(/flex-col/);
  });

  test("listbox is independently scrollable when many portfolios overflow", () => {
    const many = Array.from({ length: 25 }).map((_, i) =>
      makePortfolio(i + 1, `Portfolio ${i + 1}`),
    );
    setMockPortfolio({ portfolios: many, activePortfolio: many[0] });
    render(<MobilePortfolioPicker />);
    fireEvent.click(screen.getByRole("button", { name: /Active portfolio/i }));

    const listbox = screen.getByRole("listbox", { name: "Portfolios" });
    // overflow-y-auto + min-h-0 + flex-1 are the three classes that
    // make the items area shrink-and-scroll inside the flex column.
    expect(listbox.className).toMatch(/overflow-y-auto/);
    expect(listbox.className).toMatch(/min-h-0/);
    expect(listbox.className).toMatch(/flex-1/);
  });

  test("header stays fixed at top via shrink-0 so it doesn't scroll with items", () => {
    setMockPortfolio({});
    render(<MobilePortfolioPicker />);
    fireEvent.click(screen.getByRole("button", { name: /Active portfolio/i }));

    const heading = screen.getByRole("heading", { level: 2, name: "Switch portfolio" });
    const headerRow = heading.parentElement as HTMLElement;
    expect(headerRow.className).toMatch(/shrink-0/);
  });
});
