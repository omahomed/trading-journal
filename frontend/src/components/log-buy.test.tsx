import { render, screen, waitFor } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach } from "vitest";

class ResizeObserverStub {
  observe(): void { /* noop */ }
  unobserve(): void { /* noop */ }
  disconnect(): void { /* noop */ }
}
(globalThis as any).ResizeObserver = ResizeObserverStub;

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(), replace: vi.fn(), refresh: vi.fn(),
    back: vi.fn(), forward: vi.fn(), prefetch: vi.fn(),
  }),
}));

vi.mock("@/lib/api", () => ({
  api: {
    journalLatest: vi.fn(),
    tradesOpen: vi.fn(),
    tradesOpenDetails: vi.fn(),
    rallyPrefix: vi.fn(),
    nextTradeId: vi.fn(),
    priceLookup: vi.fn(),
  },
  getActivePortfolio: () => "CanSlim",
}));

import { api } from "@/lib/api";
import { LogBuy } from "./log-buy";

const mRally = vi.mocked(api.rallyPrefix);

function setupDefaults() {
  vi.mocked(api.journalLatest).mockResolvedValue({ end_nlv: 100000 } as any);
  vi.mocked(api.tradesOpen).mockResolvedValue([]);
  vi.mocked(api.tradesOpenDetails).mockResolvedValue([]);
  vi.mocked(api.nextTradeId).mockResolvedValue({ trade_id: "202604-001" } as any);
}


describe("LogBuy — read-only MCT-driven sizing mode", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaults();
  });

  test("renders the indicator with the mode + risk + MCT source", async () => {
    mRally.mockResolvedValue({ prefix: "", state: "POWERTREND" } as any);

    render(<LogBuy navColor="#6366f1" />);

    const indicator = await screen.findByTestId("logbuy-sizing-mode-indicator");
    await waitFor(() => {
      expect(indicator.textContent).toMatch(/Sizing:/);
      expect(indicator.textContent).toMatch(/Offense/);
      expect(indicator.textContent).toMatch(/1\.00% risk/);
      expect(indicator.textContent).toMatch(/from MCT POWERTREND/);
    });
  });

  test("no manual override radios — sizing is read-only on this surface", async () => {
    // Spec: "No manual override needed in Log Buy (it's an action, not a
    // calculator)." The Position Sizer is the override surface; Log Buy
    // surfaces the result.
    mRally.mockResolvedValue({ prefix: "", state: "POWERTREND" } as any);

    render(<LogBuy navColor="#6366f1" />);

    await screen.findByTestId("logbuy-sizing-mode-indicator");
    // No "Sizing Mode" form field with radios — pre-refactor it was
    // labelled "Sizing Mode" with three radios for defense/normal/offense.
    expect(screen.queryByText(/^Sizing Mode$/)).not.toBeInTheDocument();
  });

  test("CORRECTION → Defense (0.50% risk)", async () => {
    mRally.mockResolvedValue({ prefix: "", state: "CORRECTION" } as any);

    render(<LogBuy navColor="#6366f1" />);

    const indicator = await screen.findByTestId("logbuy-sizing-mode-indicator");
    await waitFor(() => {
      expect(indicator.textContent).toMatch(/Defense/);
      expect(indicator.textContent).toMatch(/0\.50% risk/);
      expect(indicator.textContent).toMatch(/from MCT CORRECTION/);
    });
  });

  test("rally-prefix returning no state defaults to Normal + 'MCT state unknown'", async () => {
    mRally.mockResolvedValue({ prefix: "" } as any);

    render(<LogBuy navColor="#6366f1" />);

    const indicator = await screen.findByTestId("logbuy-sizing-mode-indicator");
    await waitFor(() => {
      expect(indicator.textContent).toMatch(/Normal/);
      expect(indicator.textContent).toMatch(/0\.75% risk/);
      expect(indicator.textContent).toMatch(/MCT state unknown/);
    });
  });
});
