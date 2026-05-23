import { describe, test, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";

vi.mock("@/lib/portfolio-context", () => ({
  usePortfolio: vi.fn(),
}));

import { usePortfolio } from "@/lib/portfolio-context";
import { MobileTradeJournal } from "./mobile-trade-journal";
import type { Portfolio } from "@/lib/api";

const CANSLIM: Portfolio = {
  id: 1,
  name: "CanSlim",
  starting_capital: 100000,
  reset_date: null,
  created_at: "2026-01-01T00:00:00Z",
  cash_balance: 0,
};

describe("MobileTradeJournal — portfolio wiring", () => {
  beforeEach(() => {
    vi.mocked(usePortfolio).mockReset();
  });

  test("the Holdings header includes the active portfolio name", () => {
    vi.mocked(usePortfolio).mockReturnValue({
      portfolios: [CANSLIM],
      activePortfolio: CANSLIM,
      loading: false,
      error: null,
      refetch: vi.fn(),
      setActive: vi.fn(),
    });
    render(<MobileTradeJournal />);
    expect(screen.getByText("Holdings")).toBeInTheDocument();
    // The portfolio name renders as a separate span next to "Holdings".
    expect(screen.getByText(/CanSlim/)).toBeInTheDocument();
  });

  test("omits the portfolio name when none is active", () => {
    vi.mocked(usePortfolio).mockReturnValue({
      portfolios: [],
      activePortfolio: null,
      loading: false,
      error: null,
      refetch: vi.fn(),
      setActive: vi.fn(),
    });
    render(<MobileTradeJournal />);
    expect(screen.getByText("Holdings")).toBeInTheDocument();
    expect(screen.queryByText(/CanSlim/)).not.toBeInTheDocument();
  });
});
