import { describe, test, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";

vi.mock("@/lib/portfolio-context", () => ({
  usePortfolio: vi.fn(),
}));

import { usePortfolio } from "@/lib/portfolio-context";
import { MobilePositionSizer } from "./mobile-position-sizer";
import type { Portfolio } from "@/lib/api";

const CANSLIM: Portfolio = {
  id: 1,
  name: "CanSlim",
  starting_capital: 100000,
  reset_date: null,
  created_at: "2026-01-01T00:00:00Z",
  cash_balance: 0,
};

describe("MobilePositionSizer — portfolio wiring", () => {
  beforeEach(() => {
    vi.mocked(usePortfolio).mockReset();
  });

  test("renders 'Sizing for {portfolio name}' when a portfolio is active", () => {
    vi.mocked(usePortfolio).mockReturnValue({
      portfolios: [CANSLIM],
      activePortfolio: CANSLIM,
      loading: false,
      error: null,
      refetch: vi.fn(),
      setActive: vi.fn(),
    });
    render(<MobilePositionSizer />);
    expect(screen.getByText(/Sizing for/)).toBeInTheDocument();
    expect(screen.getByText("CanSlim")).toBeInTheDocument();
  });

  test("omits the subtitle when no portfolio is active", () => {
    vi.mocked(usePortfolio).mockReturnValue({
      portfolios: [],
      activePortfolio: null,
      loading: false,
      error: null,
      refetch: vi.fn(),
      setActive: vi.fn(),
    });
    render(<MobilePositionSizer />);
    expect(screen.queryByText(/Sizing for/)).not.toBeInTheDocument();
  });
});
