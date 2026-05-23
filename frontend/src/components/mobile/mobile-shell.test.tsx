import { describe, test, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";

// Mock portfolio context — the picker consumes it.
vi.mock("@/lib/portfolio-context", () => ({
  usePortfolio: vi.fn(),
}));
// Mock the tape pill's data source — would otherwise fire a real
// `api.rallyPrefix()` fetch on mount.
vi.mock("@/lib/use-rally-state", () => ({
  useRallyState: vi.fn(),
}));
// Mock next/navigation's usePathname so the bottom nav can decide
// active state without a router.
vi.mock("next/navigation", () => ({
  usePathname: () => "/dashboard",
}));

import { usePortfolio } from "@/lib/portfolio-context";
import { useRallyState } from "@/lib/use-rally-state";
import { MobileShell } from "./mobile-shell";
import type { Portfolio } from "@/lib/api";

const CANSLIM: Portfolio = {
  id: 1,
  name: "CanSlim",
  starting_capital: 100000,
  reset_date: null,
  created_at: "2026-01-01T00:00:00Z",
  cash_balance: 0,
};

describe("MobileShell — header right slot", () => {
  beforeEach(() => {
    vi.mocked(usePortfolio).mockReturnValue({
      portfolios: [CANSLIM],
      activePortfolio: CANSLIM,
      loading: false,
      error: null,
      refetch: vi.fn(),
      setActive: vi.fn(),
    });
    vi.mocked(useRallyState).mockReturnValue(null); // tape pill renders placeholder
  });

  test("mounts the portfolio picker by default", () => {
    render(
      <MobileShell header={{ title: "Position", italicWord: "Sizer" }}>
        <p>page body</p>
      </MobileShell>,
    );
    // The picker's closed-state button advertises itself via aria-label.
    expect(
      screen.getByRole("button", { name: /Active portfolio: CanSlim/ }),
    ).toBeInTheDocument();
  });

  test("a page-supplied rightSlot replaces the default picker", () => {
    render(
      <MobileShell
        header={{
          title: "Position",
          italicWord: "Sizer",
          rightSlot: <span data-testid="custom-right">CUSTOM</span>,
        }}
      >
        <p>page body</p>
      </MobileShell>,
    );
    expect(screen.getByTestId("custom-right")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /Active portfolio/ }),
    ).not.toBeInTheDocument();
  });
});
