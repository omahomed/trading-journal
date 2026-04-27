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

  test("manual override radios are present (Override Sizing Mode field)", async () => {
    // The user can toggle defense/normal/offense for THIS Log Buy
    // submission. Override is form-local — refresh / submit / unmount
    // resets back to the MCT-driven auto pick (verified separately by
    // the post-remount-resets-to-MCT test below).
    mRally.mockResolvedValue({ prefix: "", state: "POWERTREND" } as any);

    render(<LogBuy navColor="#6366f1" />);

    await screen.findByTestId("logbuy-sizing-mode-indicator");
    expect(screen.getByText("Override Sizing Mode")).toBeInTheDocument();
  });

  test("clicking a different mode flips indicator to '— manual override' and shows Reset", async () => {
    // Auto: Offense (POWERTREND). User toggles to Defense → indicator
    // copy switches; Reset button appears.
    mRally.mockResolvedValue({ prefix: "", state: "POWERTREND" } as any);

    render(<LogBuy navColor="#6366f1" />);

    const indicator = await screen.findByTestId("logbuy-sizing-mode-indicator");
    await waitFor(() => expect(indicator.textContent).toMatch(/Offense/));
    expect(screen.queryByTestId("logbuy-reset-to-mct")).not.toBeInTheDocument();

    // Click the Defense radio. Mode label includes "Defense (0.50%)".
    const defenseRadio = await screen.findByText(/Defense \(0\.50%\)/);
    await act(async () => { fireEvent.click(defenseRadio); });

    await waitFor(() => {
      expect(indicator.textContent).toMatch(/Defense/);
      expect(indicator.textContent).toMatch(/0\.50% risk/);
      expect(indicator.textContent).toMatch(/manual override/);
      // No "from MCT …" label while in manual mode (the source is the user)
      expect(indicator.textContent).not.toMatch(/from MCT/);
    });
    expect(screen.getByTestId("logbuy-reset-to-mct")).toBeInTheDocument();
  });

  test("'Reset to MCT' restores the auto-derived mode", async () => {
    mRally.mockResolvedValue({ prefix: "", state: "POWERTREND" } as any);

    render(<LogBuy navColor="#6366f1" />);

    // Override to Defense first
    const defenseRadio = await screen.findByText(/Defense \(0\.50%\)/);
    await act(async () => { fireEvent.click(defenseRadio); });

    const reset = await screen.findByTestId("logbuy-reset-to-mct");
    await act(async () => { fireEvent.click(reset); });

    const indicator = screen.getByTestId("logbuy-sizing-mode-indicator");
    await waitFor(() => {
      expect(indicator.textContent).toMatch(/Offense/);
      expect(indicator.textContent).toMatch(/1\.00% risk/);
      expect(indicator.textContent).toMatch(/from MCT POWERTREND/);
      expect(indicator.textContent).not.toMatch(/manual override/);
    });
    expect(screen.queryByTestId("logbuy-reset-to-mct")).not.toBeInTheDocument();
  });

  test("override does not persist across remount — fresh render re-derives from MCT", async () => {
    // Page-reload analogue: unmount + remount the component. State is
    // local to the React tree, so a remount drops sizingModeManual and
    // re-runs the mount effect, which re-applies the MCT auto pick.
    mRally.mockResolvedValue({ prefix: "", state: "POWERTREND" } as any);

    const { unmount } = render(<LogBuy navColor="#6366f1" />);
    const defenseRadio = await screen.findByText(/Defense \(0\.50%\)/);
    await act(async () => { fireEvent.click(defenseRadio); });
    await screen.findByTestId("logbuy-reset-to-mct");
    unmount();

    // Remount with the same MCT mock — should land back on Offense auto.
    render(<LogBuy navColor="#6366f1" />);
    const indicator = await screen.findByTestId("logbuy-sizing-mode-indicator");
    await waitFor(() => {
      expect(indicator.textContent).toMatch(/Offense/);
      expect(indicator.textContent).toMatch(/from MCT POWERTREND/);
    });
    expect(screen.queryByTestId("logbuy-reset-to-mct")).not.toBeInTheDocument();
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
