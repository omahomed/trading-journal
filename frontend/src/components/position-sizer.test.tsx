import { render, screen, waitFor, fireEvent, act } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach } from "vitest";

class ResizeObserverStub {
  observe(): void { /* noop */ }
  unobserve(): void { /* noop */ }
  disconnect(): void { /* noop */ }
}
(globalThis as any).ResizeObserver = ResizeObserverStub;

// JSDOM in this vitest setup ships with a localStorage stub that throws
// on most methods (the "--localstorage-file was provided without a valid
// path" warning is the symptom). Replace it with a minimal in-memory
// implementation so sendToLogBuy can write + read ps_prefill.
const _lsStore = new Map<string, string>();
Object.defineProperty(globalThis, "localStorage", {
  configurable: true,
  value: {
    getItem: (k: string) => _lsStore.get(k) ?? null,
    setItem: (k: string, v: string) => { _lsStore.set(k, v); },
    removeItem: (k: string) => { _lsStore.delete(k); },
    clear: () => { _lsStore.clear(); },
    key: (i: number) => Array.from(_lsStore.keys())[i] ?? null,
    get length() { return _lsStore.size; },
  },
});

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
  vi.mocked(api.tradesOpenDetails).mockResolvedValue({ details: [], lot_closures: [] });
  vi.mocked(api.config).mockResolvedValue({ key: "pyramid_rules", value: { trigger_pct: 5, alloc_pct: 20 } } as any);
  // priceLookup fires on ticker change (debounced) — mock returns the
  // user-entered values so the auto-fill no-ops the user inputs already set.
  vi.mocked(api.priceLookup).mockRejectedValue(new Error("price lookup disabled in test"));
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
      // Post-retier: Offense is 0.75% (was 1.00%).
      expect(indicator.textContent).toMatch(/Offense \(0\.75%\)/);
      expect(indicator.textContent).toMatch(/from M Factor POWERTREND/);
    });
    // No reset button while in auto mode
    expect(screen.queryByTestId("sizer-reset-to-auto")).not.toBeInTheDocument();
  });

  test("state → mode mapping: POWERTREND → Offense, UPTREND → Normal, everything else → Pilot", async () => {
    // Post-retier mapping. Remounts because each render calls
    // rallyPrefix once on mount; clean lifecycle covers each state
    // without state leakage.
    for (const [state, label] of [
      ["POWERTREND", "Offense (0.75%)"],
      ["UPTREND", "Normal (0.50%)"],
      ["UPTREND UNDER PRESSURE", "Pilot (0.25%)"],
      ["RALLY MODE", "Pilot (0.25%)"],
      ["CORRECTION", "Pilot (0.25%)"],
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

    // Override to Pilot (idx 0, most conservative — was Defense pre-retier).
    // Match by the emoji-prefixed radio label.
    const pilotRadio = await screen.findByText(/Pilot \(0\.25%\)/);
    await act(async () => {
      fireEvent.click(pilotRadio);
    });

    // Indicator label flips
    await waitFor(() => {
      expect(indicator.textContent).toMatch(/Manual:/);
      expect(indicator.textContent).toMatch(/Pilot \(0\.25%\)/);
      // No "from M Factor …" hint while in manual mode (the source is the user)
      expect(indicator.textContent).not.toMatch(/from M Factor/);
    });
    // Reset button appears
    expect(screen.getByTestId("sizer-reset-to-auto")).toBeInTheDocument();
  });

  test("'Reset to auto' restores MCT-driven mode", async () => {
    mRally.mockResolvedValue({ prefix: "", state: "POWERTREND" } as any);

    render(<PositionSizer navColor="#6366f1" />);

    // Override to Pilot first (was Defense pre-retier)
    const pilotRadio = await screen.findByText(/Pilot \(0\.25%\)/);
    await act(async () => { fireEvent.click(pilotRadio); });

    const reset = await screen.findByTestId("sizer-reset-to-auto");
    await act(async () => { fireEvent.click(reset); });

    // Back to Auto + Offense (POWERTREND) at the retiered 0.75%
    const indicator = screen.getByTestId("sizer-mode-indicator");
    await waitFor(() => {
      expect(indicator.textContent).toMatch(/Auto:/);
      expect(indicator.textContent).toMatch(/Offense \(0\.75%\)/);
      expect(indicator.textContent).toMatch(/from M Factor POWERTREND/);
    });
    expect(screen.queryByTestId("sizer-reset-to-auto")).not.toBeInTheDocument();
  });

  test("rally-prefix returning no state defaults to Pilot + 'M Factor state unknown'", async () => {
    // Failure mode: rallyPrefix's catch landed an empty {prefix: ""}
    // (no `state` field). Post-retier default is Pilot (most
    // conservative) — the "safe middle" default was retired along
    // with the Defense tier.
    mRally.mockResolvedValue({ prefix: "" } as any);

    render(<PositionSizer navColor="#6366f1" />);

    const indicator = await screen.findByTestId("sizer-mode-indicator");
    await waitFor(() => {
      expect(indicator.textContent).toMatch(/Auto:/);
      expect(indicator.textContent).toMatch(/Pilot \(0\.25%\)/);
      expect(indicator.textContent).toMatch(/M Factor state unknown/);
    });
  });
});

// ─────────────────────────────────────────────────────────────────────
// Volatility Sizer — composite-stop model (2026-07-18)
//
// The tab moved from a 4-tile grid (Tech Stop + 1×/1.5×/2× ATR) with a
// user-picked size ladder to ONE answer computed from a composite stop:
//    Composite = MIN(Entry − 1 ATR, KeyLevel − max(0.5 ATR, 1%))
//    Shares    = RiskBudget ÷ (Entry − Composite), capped at 15% (5% young-IPO).
// Tests below pin the new inputs (Key Level, Young-IPO checkbox), the
// single answer card, the bind indicator, and the ATR-based scale-out.
// ─────────────────────────────────────────────────────────────────────

async function fillVolTabInputs(opts: {
  ticker?: string;
  entry: string;
  keyLevel: string;
  atr: string;
  equity: string;
  youngIpo?: boolean;
}) {
  // Switch to the Volatility tab.
  await act(async () => {
    fireEvent.click(screen.getByRole("button", { name: /New Entry/ }));
  });

  const ticker = screen.getByPlaceholderText("XYZ") as HTMLInputElement;
  await act(async () => {
    fireEvent.change(ticker, { target: { value: opts.ticker ?? "DELL" } });
  });

  // Entry Price (placeholder 0.00, first "0.00" placeholder).
  const numInputs = screen.getAllByPlaceholderText("0.00") as HTMLInputElement[];
  await act(async () => {
    fireEvent.change(numInputs[0], { target: { value: opts.entry } });
  });

  // Equity has no placeholder text — find by step attribute.
  const equityInput = document.querySelector('input[step="1000"]') as HTMLInputElement;
  await act(async () => {
    fireEvent.change(equityInput, { target: { value: opts.equity } });
  });

  // Key Level is testable via a distinct data-testid.
  const keyLevelInput = screen.getByTestId("key-level-input") as HTMLInputElement;
  await act(async () => {
    fireEvent.change(keyLevelInput, { target: { value: opts.keyLevel } });
  });

  // ATR placeholder is 5.0.
  const atrInput = screen.getByPlaceholderText("5.0") as HTMLInputElement;
  await act(async () => {
    fireEvent.change(atrInput, { target: { value: opts.atr } });
  });

  if (opts.youngIpo) {
    const cb = screen.getByTestId("young-ipo-checkbox") as HTMLInputElement;
    await act(async () => {
      fireEvent.click(cb);
    });
  }
}

describe("PositionSizer — Volatility Sizer composite-stop model", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaults();
    // UPTREND → Normal (0.50%) — matches the DELL/COHR canonical
    // examples in the design conversation and vol-sizer.test.ts.
    mRally.mockResolvedValue({ prefix: "", state: "UPTREND" } as any);
    try { localStorage.removeItem("ps_prefill"); } catch { /* JSDOM polyfill quirk */ }
  });

  test("legacy 4-tile UI is gone: no Tech Stop / 1x / 1.5x / 2x ATR scenario cards", async () => {
    render(<PositionSizer navColor="#6366f1" />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /New Entry/ }));
    });

    expect(screen.queryByTestId("scenario-tech-stop")).not.toBeInTheDocument();
    expect(screen.queryByTestId("scenario-1x-atr")).not.toBeInTheDocument();
    expect(screen.queryByTestId("scenario-1.5x-atr")).not.toBeInTheDocument();
    expect(screen.queryByTestId("scenario-2x-atr")).not.toBeInTheDocument();
    // No user-picked target-size ladder on this tab.
    expect(screen.queryByText(/Shotgun \(2\.5%\)/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Overweight \(12\.5%\)/)).not.toBeInTheDocument();
  });

  test("DELL canonical case → composite stop wins on Key Level, 227 shares, risk-bound", async () => {
    render(<PositionSizer navColor="#6366f1" />);
    const indicator = await screen.findByTestId("sizer-mode-indicator");
    await waitFor(() => expect(indicator.textContent).toMatch(/Normal \(0\.50%\)/));

    await fillVolTabInputs({
      ticker: "DELL",
      entry: "176.21",
      keyLevel: "171.365",   // yields composite = 167.40 (Key Level wins)
      atr: "4.5",
      equity: "400000",
    });

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Calculate Size/ }));
    });

    // Final shares 227 from the DELL example in the design conversation.
    const finalShares = await screen.findByTestId("final-shares");
    expect(finalShares.textContent).toMatch(/227/);

    // Bind badge = Risk-bound (Ceiling at 15% would be 340 shs).
    expect(screen.getByTestId("bind-badge").textContent).toBe("Risk-bound");

    // Composite winner subtitle mentions the Key Level candidate.
    const answerCard = screen.getByTestId("composite-answer");
    expect(answerCard.textContent).toMatch(/Key Level/);
    expect(answerCard.textContent).toMatch(/0\.5 ATR/);   // buffer basis under 4.5% ATR
  });

  test("COHR-style hot ATR → composite falls back to 1 ATR floor, 84 shares", async () => {
    render(<PositionSizer navColor="#6366f1" />);
    const indicator = await screen.findByTestId("sizer-mode-indicator");
    await waitFor(() => expect(indicator.textContent).toMatch(/Normal/));

    await fillVolTabInputs({
      ticker: "COHR",
      entry: "246.53",
      keyLevel: "240.00",   // Key Level too close; 1 ATR floor wider → ATR wins
      atr: "9.6",
      equity: "400000",
    });

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Calculate Size/ }));
    });

    const finalShares = await screen.findByTestId("final-shares");
    expect(finalShares.textContent).toMatch(/84/);

    // Composite subtitle names the ATR floor winner.
    const answerCard = screen.getByTestId("composite-answer");
    expect(answerCard.textContent).toMatch(/1 ATR floor/);
  });

  test("Young-IPO checkbox clamps ceiling to 5% → DELL case flips to ceiling-bound at 113 shs", async () => {
    render(<PositionSizer navColor="#6366f1" />);
    const indicator = await screen.findByTestId("sizer-mode-indicator");
    await waitFor(() => expect(indicator.textContent).toMatch(/Normal/));

    await fillVolTabInputs({
      ticker: "IPO",
      entry: "176.21",
      keyLevel: "171.365",
      atr: "4.5",
      equity: "400000",
      youngIpo: true,
    });

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Calculate Size/ }));
    });

    // 5% × 400K / 176.21 = 113 (floored) — vs raw 227 → ceiling wins.
    const finalShares = await screen.findByTestId("final-shares");
    expect(finalShares.textContent).toMatch(/113/);
    expect(screen.getByTestId("bind-badge").textContent).toBe("Ceiling-bound");
  });

  test("Send to Log Buy emits price payload with the resolved composite stop", async () => {
    // Under the new model, the sizer always sends a resolved dollar
    // stop (composite.price) with stopMode='price'. The old ATR-mode
    // handoff is retired — Log Buy no longer needs to recompute a stop
    // from its own ATR lookup because the composite is already fixed.
    render(<PositionSizer navColor="#6366f1" />);
    const indicator = await screen.findByTestId("sizer-mode-indicator");
    await waitFor(() => expect(indicator.textContent).toMatch(/Normal/));

    await fillVolTabInputs({
      ticker: "DELL",
      entry: "176.21",
      keyLevel: "171.365",
      atr: "4.5",
      equity: "400000",
    });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Calculate Size/ }));
    });

    // Match the primary Send button; the Scale-Out card also has a
    // "with ladder" button that would otherwise match.
    const sendBtn = await screen.findByRole("button", { name: /Send to Log Buy —/ });
    await act(async () => { fireEvent.click(sendBtn); });

    const stored = JSON.parse(localStorage.getItem("ps_prefill") || "{}");
    expect(stored.ticker).toBe("DELL");
    expect(stored.shares).toBe(227);
    expect(stored.price).toBe(176.21);
    expect(stored.stopMode).toBe("price");
    expect(stored.stop).toBeCloseTo(167.4, 1);   // composite stop
    expect(stored.atrMultiplier).toBeUndefined();
    expect(stored.action).toBe("new");
  });

  test("21 EMA / 50 SMA cells render when priceLookup returns non-null values; Use → paste into Key Level", async () => {
    // Override the default mock (which rejects) so priceLookup returns
    // real MA levels. The cells + Use buttons should appear after the
    // ticker debounce fires and populate. Clicking Use → on the 21 EMA
    // cell should paste 172.34 into the Key Level input.
    vi.mocked(api.priceLookup).mockResolvedValue({
      ticker: "DELL",
      price: 176.21,
      atr: 7.93,
      atr_pct: 4.5,
      ema_21: 172.34,
      sma_50: 165.20,
    } as any);
    render(<PositionSizer navColor="#6366f1" />);
    const indicator = await screen.findByTestId("sizer-mode-indicator");
    await waitFor(() => expect(indicator.textContent).toMatch(/Normal/));

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /New Entry/ }));
    });
    const ticker = screen.getByPlaceholderText("XYZ") as HTMLInputElement;
    await act(async () => { fireEvent.change(ticker, { target: { value: "DELL" } }); });

    // Cells populate after the priceLookup fires. The 600ms debounce
    // + async resolve is why we waitFor here.
    const emaCell = await screen.findByTestId("ema21-cell", {}, { timeout: 2000 });
    expect(emaCell.textContent).toMatch(/172\.34/);
    expect(screen.getByTestId("sma50-cell").textContent).toMatch(/165\.20/);

    // Click Use → on the 21 EMA cell — Key Level input picks up 172.34.
    await act(async () => { fireEvent.click(screen.getByTestId("use-ema21-btn")); });
    const keyLevelInput = screen.getByTestId("key-level-input") as HTMLInputElement;
    expect(keyLevelInput.value).toBe("172.34");
  });

  test("cells hide when priceLookup returns null MA levels (sparse-history ticker)", async () => {
    vi.mocked(api.priceLookup).mockResolvedValue({
      ticker: "IPO",
      price: 100,
      atr: 5,
      atr_pct: 5,
      ema_21: null,
      sma_50: null,
    } as any);
    render(<PositionSizer navColor="#6366f1" />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /New Entry/ }));
    });
    const ticker = screen.getByPlaceholderText("XYZ") as HTMLInputElement;
    await act(async () => { fireEvent.change(ticker, { target: { value: "IPO" } }); });

    // Wait a beat for the debounced lookup to complete, then confirm
    // both cells are absent.
    await new Promise(r => setTimeout(r, 700));
    expect(screen.queryByTestId("ema21-cell")).not.toBeInTheDocument();
    expect(screen.queryByTestId("sma50-cell")).not.toBeInTheDocument();
  });

  test("Scale-out ladder uses ATR multiples 0.5 / 1.0 / 1.5, not the old −3/−5/−7 percentages", async () => {
    render(<PositionSizer navColor="#6366f1" />);
    const indicator = await screen.findByTestId("sizer-mode-indicator");
    await waitFor(() => expect(indicator.textContent).toMatch(/Normal/));

    await fillVolTabInputs({
      ticker: "DELL",
      entry: "176.21",
      keyLevel: "171.365",
      atr: "4.5",
      equity: "400000",
    });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Calculate Size/ }));
    });

    const ladder = await screen.findByTestId("scale-out-stops");
    expect(ladder.textContent).toMatch(/−0\.50 ATR/);
    expect(ladder.textContent).toMatch(/−1\.00 ATR/);
    expect(ladder.textContent).toMatch(/−1\.50 ATR/);
    // Old locked-percent ladder is gone.
    expect(ladder.textContent).not.toMatch(/−3%|−5%|−7%/);
  });
});
