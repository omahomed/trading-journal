import { render, screen, waitFor, fireEvent, act } from "@testing-library/react";
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
    priceLookup: vi.fn(),
    journalLatest: vi.fn(),
    tradesOpen: vi.fn(),
    tradesOpenDetails: vi.fn(),
    rallyPrefix: vi.fn(),
    config: vi.fn(),
  },
  getActivePortfolio: () => "CanSlim",
}));

import { api } from "@/lib/api";
import { PositionSizer } from "./position-sizer";

const mRally = vi.mocked(api.rallyPrefix);

function setupDefaults() {
  vi.mocked(api.journalLatest).mockResolvedValue({ end_nlv: 100000 } as any);
  vi.mocked(api.tradesOpen).mockResolvedValue([]);
  vi.mocked(api.tradesOpenDetails).mockResolvedValue([]);
  vi.mocked(api.config).mockResolvedValue({ key: "pyramid_rules", value: { trigger_pct: 5, alloc_pct: 20 } } as any);
}


describe("PositionSizer — MCT-driven sizing mode", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaults();
  });

  test("renders 'Auto: Offense (from M Factor POWERTREND)' when MCT state is POWERTREND", async () => {
    mRally.mockResolvedValue({ prefix: "Day 18: ", state: "POWERTREND" } as any);

    render(<PositionSizer navColor="#6366f1" />);

    const indicator = await screen.findByTestId("sizer-mode-indicator");
    await waitFor(() => {
      expect(indicator.textContent).toMatch(/Auto:/);
      expect(indicator.textContent).toMatch(/Offense \(1\.00%\)/);
      expect(indicator.textContent).toMatch(/from M Factor POWERTREND/);
    });
    // No reset button while in auto mode
    expect(screen.queryByTestId("sizer-reset-to-auto")).not.toBeInTheDocument();
  });

  test("CORRECTION → Defense, RALLY MODE → Normal, UPTREND → Offense", async () => {
    // Three remounts because each render calls rallyPrefix once on mount.
    // Clean lifecycle covers all three states without state leakage.
    for (const [state, label] of [
      ["CORRECTION", "Defense (0.50%)"],
      ["RALLY MODE", "Normal (0.75%)"],
      ["UPTREND", "Offense (1.00%)"],
    ] as const) {
      vi.clearAllMocks();
      setupDefaults();
      mRally.mockResolvedValue({ prefix: "", state } as any);

      const { unmount } = render(<PositionSizer navColor="#6366f1" />);
      const indicator = await screen.findByTestId("sizer-mode-indicator");
      await waitFor(() => {
        expect(indicator.textContent).toContain(label);
        expect(indicator.textContent).toContain(`from M Factor ${state}`);
      });
      unmount();
    }
  });

  test("manual override flips indicator to 'Manual: …' and shows a Reset button", async () => {
    mRally.mockResolvedValue({ prefix: "", state: "POWERTREND" } as any);

    render(<PositionSizer navColor="#6366f1" />);

    // Wait for the auto pick to land before overriding.
    const indicator = await screen.findByTestId("sizer-mode-indicator");
    await waitFor(() => expect(indicator.textContent).toMatch(/Auto:/));

    // Click the Defense radio. (Match by accessible label — Position
    // Sizer renders SIZING_MODES with the emoji-prefixed label, so the
    // "Defense" word is enough to disambiguate.)
    const defenseRadio = await screen.findByText(/Defense \(0\.50%\)/);
    await act(async () => {
      fireEvent.click(defenseRadio);
    });

    // Indicator label flips
    await waitFor(() => {
      expect(indicator.textContent).toMatch(/Manual:/);
      expect(indicator.textContent).toMatch(/Defense \(0\.50%\)/);
      // No "from M Factor …" hint while in manual mode (the source is the user)
      expect(indicator.textContent).not.toMatch(/from M Factor/);
    });
    // Reset button appears
    expect(screen.getByTestId("sizer-reset-to-auto")).toBeInTheDocument();
  });

  test("'Reset to auto' restores MCT-driven mode", async () => {
    mRally.mockResolvedValue({ prefix: "", state: "POWERTREND" } as any);

    render(<PositionSizer navColor="#6366f1" />);

    // Override to Defense first
    const defenseRadio = await screen.findByText(/Defense \(0\.50%\)/);
    await act(async () => { fireEvent.click(defenseRadio); });

    const reset = await screen.findByTestId("sizer-reset-to-auto");
    await act(async () => { fireEvent.click(reset); });

    // Back to Auto + Offense (POWERTREND)
    const indicator = screen.getByTestId("sizer-mode-indicator");
    await waitFor(() => {
      expect(indicator.textContent).toMatch(/Auto:/);
      expect(indicator.textContent).toMatch(/Offense \(1\.00%\)/);
      expect(indicator.textContent).toMatch(/from M Factor POWERTREND/);
    });
    expect(screen.queryByTestId("sizer-reset-to-auto")).not.toBeInTheDocument();
  });

  test("rally-prefix returning no state defaults to Normal + 'M Factor state unknown'", async () => {
    // Failure mode: rallyPrefix's catch landed an empty {prefix: ""}
    // (no `state` field). Spec: safe-middle Normal, no fake-source label.
    mRally.mockResolvedValue({ prefix: "" } as any);

    render(<PositionSizer navColor="#6366f1" />);

    const indicator = await screen.findByTestId("sizer-mode-indicator");
    await waitFor(() => {
      expect(indicator.textContent).toMatch(/Auto:/);
      expect(indicator.textContent).toMatch(/Normal \(0\.75%\)/);
      expect(indicator.textContent).toMatch(/M Factor state unknown/);
    });
  });
});
