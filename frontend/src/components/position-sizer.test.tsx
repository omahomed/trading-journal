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

// ─────────────────────────────────────────────────────────────────────
// Volatility Sizer redesign — Commit A
//
// Tests below pin the new layout: three ATR-cushion scenarios + tech
// stop, a referential recommendation, a calculated-stop banner annotated
// with the tech stop's ATR fraction, and a warning banner when the stop
// sits inside one ATR of daily noise.
//
// Inputs use the GOOGL canonical case from frontend/src/lib/vol-sizer
// .test.ts so the rendered output stays in lockstep with the lib's
// unit tests.
// ─────────────────────────────────────────────────────────────────────

async function fillVolTabInputs(opts: {
  ticker?: string;
  entry: string;
  ma: string;
  buffer?: string;
  atr: string;
  equity: string;
  targetPct?: number;
}) {
  // Switch to the Volatility tab
  await act(async () => {
    fireEvent.click(screen.getByRole("button", { name: /New Position Sizer/ }));
  });

  const ticker = screen.getByPlaceholderText("XYZ") as HTMLInputElement;
  await act(async () => {
    fireEvent.change(ticker, { target: { value: opts.ticker ?? "GOOGL" } });
  });

  // Entry / Equity (placeholders 0.00 + step 1000)
  const entryInputs = screen.getAllByPlaceholderText("0.00") as HTMLInputElement[];
  // Order in the inputs section: Entry Price, MA Level (both placeholder 0.00)
  await act(async () => {
    fireEvent.change(entryInputs[0], { target: { value: opts.entry } });
  });

  // Equity input has no placeholder text — find by step
  const equityInput = document.querySelector('input[step="1000"]') as HTMLInputElement;
  await act(async () => {
    fireEvent.change(equityInput, { target: { value: opts.equity } });
  });

  // MA Level + Buffer
  await act(async () => {
    fireEvent.change(entryInputs[1], { target: { value: opts.ma } });
  });
  if (opts.buffer) {
    const bufferInput = screen.getByPlaceholderText("1.00") as HTMLInputElement;
    await act(async () => {
      fireEvent.change(bufferInput, { target: { value: opts.buffer } });
    });
  }

  // ATR
  const atrInput = screen.getByPlaceholderText("5.0") as HTMLInputElement;
  await act(async () => {
    fireEvent.change(atrInput, { target: { value: opts.atr } });
  });

  // Target tier (optional override; default 10% selected on mount)
  if (opts.targetPct !== undefined && opts.targetPct !== 10) {
    const tierLabels: Record<number, string> = {
      2.5: "Starter (2.5%)", 5: "Half (5%)", 7.5: "Standard (7.5%)",
      10: "Full (10%)", 12.5: "Overweight (12.5%)", 15: "Core (15%)",
      17.5: "Core+ (17.5%)", 20: "Max (20%)",
    };
    await act(async () => {
      fireEvent.click(screen.getByText(tierLabels[opts.targetPct!]));
    });
  }
}

describe("PositionSizer — Volatility Sizer redesign (Commit A)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaults();
    // POWERTREND → Offense (1.0%) — matches the GOOGL canonical case in vol-sizer.test.ts
    mRally.mockResolvedValue({ prefix: "", state: "POWERTREND" } as any);
    try { localStorage.removeItem("ps_prefill"); } catch { /* JSDOM polyfill quirk */ }
  });

  test("audit-mode UI is gone: no Sizing Context, no holding picker on vol tab, no Stock Volatility Profile", async () => {
    render(<PositionSizer navColor="#6366f1" />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /New Position Sizer/ }));
    });

    expect(screen.queryByText(/Sizing Context/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Audit Active Position/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Stock Volatility Profile/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Tight \(1\.0x\)/)).not.toBeInTheDocument();
    expect(screen.queryByText(/High-Vol \(1\.5x\)/)).not.toBeInTheDocument();
    // The vol tab no longer renders the Select Position dropdown
    expect(screen.queryByText(/^Select Position$/)).not.toBeInTheDocument();
  });

  test("GOOGL canonical case → recommended = 1.5x ATR, three ATR cards render, warning visible", async () => {
    render(<PositionSizer navColor="#6366f1" />);
    // Wait for MCT auto-pick (POWERTREND → Offense) so tolPct = 1.0
    const indicator = await screen.findByTestId("sizer-mode-indicator");
    await waitFor(() => expect(indicator.textContent).toMatch(/Offense \(1\.00%\)/));

    await fillVolTabInputs({
      ticker: "GOOGL",
      entry: "382.97",
      ma: "379.40",
      buffer: "1.0",
      atr: "2.87",
      equity: "702924",
      targetPct: 10,
    });

    // Trigger calculation
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Calculate Size/ }));
    });

    // Three ATR cards present with their labels
    expect(await screen.findByTestId("scenario-1x-atr")).toBeInTheDocument();
    expect(screen.getByTestId("scenario-1.5x-atr")).toBeInTheDocument();
    expect(screen.getByTestId("scenario-2x-atr")).toBeInTheDocument();
    expect(screen.getByTestId("scenario-tech-stop")).toBeInTheDocument();

    // Recommended pill is on the 1.5x ATR card (tech_stop_inside_noise path)
    const pills = screen.getAllByTestId("recommended-pill");
    expect(pills).toHaveLength(1);
    expect(screen.getByTestId("scenario-1.5x-atr")).toContainElement(pills[0]);

    // Verdict text reflects the 1.5x ATR recommendation + tier-cap binding
    const verdict = screen.getByText(/RECOMMENDED: Buy/);
    expect(verdict.textContent).toMatch(/183/);
    expect(verdict.parentElement?.parentElement?.textContent).toMatch(/Sized by 1\.5× ATR cushion/);
    expect(verdict.parentElement?.parentElement?.textContent).toMatch(/position-size tier \(10% NLV\)/);
  });

  test("Calculated Stop banner annotates with the tech stop's ATR fraction (0.67x for GOOGL)", async () => {
    render(<PositionSizer navColor="#6366f1" />);
    const indicator = await screen.findByTestId("sizer-mode-indicator");
    await waitFor(() => expect(indicator.textContent).toMatch(/Offense/));

    await fillVolTabInputs({
      entry: "382.97",
      ma: "379.40",
      buffer: "1.0",
      atr: "2.87",
      equity: "702924",
    });

    // Banner annotation: tech stop is 7.364/382.97 ≈ 1.92% below entry,
    // divided by 2.87% ATR ≈ 0.67× ATR (matches vol-sizer.test.ts GOOGL).
    await waitFor(() => {
      expect(screen.getByText(/0\.67× ATR/)).toBeInTheDocument();
    });
  });

  test("warning sub-banner shows when stop < 1 ATR (GOOGL) and hides when stop >= 1 ATR", async () => {
    const { unmount } = render(<PositionSizer navColor="#6366f1" />);
    const indicator = await screen.findByTestId("sizer-mode-indicator");
    await waitFor(() => expect(indicator.textContent).toMatch(/Offense/));

    // Sub-1-ATR case (GOOGL). Warning renders in two places — the
    // input-section sub-banner AND embedded in the verdict card — so
    // use getAllByText. We assert at least one match.
    await fillVolTabInputs({ entry: "382.97", ma: "379.40", buffer: "1.0", atr: "2.87", equity: "702924" });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Calculate Size/ }));
    });
    await waitFor(() => {
      const matches = screen.queryAllByText(/Tech stop is 0\.\d{2} ATR — daily noise/);
      expect(matches.length).toBeGreaterThan(0);
    });
    unmount();

    // Tech stop ≥ 1 ATR case → no warning, tech stop is recommended
    setupDefaults();
    mRally.mockResolvedValue({ prefix: "", state: "POWERTREND" } as any);
    render(<PositionSizer navColor="#6366f1" />);
    const ind2 = await screen.findByTestId("sizer-mode-indicator");
    await waitFor(() => expect(ind2.textContent).toMatch(/Offense/));
    // entry=100, ma=100, buf=3, atr=2 → stop=97, stopDist=3, atrFraction=1.5
    await fillVolTabInputs({ entry: "100", ma: "100", buffer: "3.0", atr: "2.0", equity: "100000" });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Calculate Size/ }));
    });
    expect(screen.queryByText(/daily noise will likely stop you out/)).not.toBeInTheDocument();
    // Tech stop is recommended
    const pills = screen.getAllByTestId("recommended-pill");
    expect(screen.getByTestId("scenario-tech-stop")).toContainElement(pills[0]);
    expect(screen.getByText(/Sized by tech stop/)).toBeInTheDocument();
  });

  test("Send to Log Buy emits ATR payload (stopMode + multiplier) when ATR scenario is recommended", async () => {
    // ATR scenarios no longer pre-resolve a dollar stop. The Sizer emits
    // {stopMode:"atr", atrMultiplier:1.5} and Log Buy fetches its own
    // atrPct from /api/prices/lookup to recompute the effective stop —
    // eliminates two-fetch drift between Sizer time and Log Buy time.
    render(<PositionSizer navColor="#6366f1" />);
    const indicator = await screen.findByTestId("sizer-mode-indicator");
    await waitFor(() => expect(indicator.textContent).toMatch(/Offense/));

    await fillVolTabInputs({
      ticker: "GOOGL",
      entry: "382.97",
      ma: "379.40",
      buffer: "1.0",
      atr: "2.87",
      equity: "702924",
    });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Calculate Size/ }));
    });

    // Click "Send to Log Buy"
    // Match the primary Verdict button; the Scale-Out card also has a
// "Send to Log Buy with ladder" button that would otherwise match.
const sendBtn = await screen.findByRole("button", { name: /Send to Log Buy —/ });
    await act(async () => { fireEvent.click(sendBtn); });

    const stored = JSON.parse(localStorage.getItem("ps_prefill") || "{}");
    expect(stored.ticker).toBe("GOOGL");
    expect(stored.shares).toBe(183);
    expect(stored.price).toBe(382.97);
    expect(stored.action).toBe("new");
    // Recommended = 1.5× ATR scenario. ATR-mode handoff:
    expect(stored.stopMode).toBe("atr");
    expect(stored.atrMultiplier).toBe(1.5);
    // No resolved dollar stop on ATR scenarios — Log Buy recomputes
    // from its own atrPct lookup. (Pre-B-3 the field was 366.48.)
    expect(stored.stop).toBeUndefined();
  });

  test("Send to Log Buy emits price payload (stopMode='price' + resolved stop) when tech stop is recommended", async () => {
    // Companion to the ATR-payload test above. Tech-stop scenarios
    // continue to emit a resolved dollar `stop`, tagged stopMode='price'
    // so the receiver flips Log Buy out of its default pct mode.
    // setupDefaults() (and a fresh mRally MOMENTUM mock at the top of
    // this suite) puts us in a fixture where 100/100/3.0/2.0 gives
    // tech_stop_safe (stop=97, stopDist=3, atrFraction=1.5).
    render(<PositionSizer navColor="#6366f1" />);
    const indicator = await screen.findByTestId("sizer-mode-indicator");
    await waitFor(() => expect(indicator.textContent).toMatch(/Offense/));

    await fillVolTabInputs({
      ticker: "AAPL",
      entry: "100",
      ma: "100",
      buffer: "3.0",
      atr: "2.0",
      equity: "100000",
    });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Calculate Size/ }));
    });

    // Match the primary Verdict button; the Scale-Out card also has a
// "Send to Log Buy with ladder" button that would otherwise match.
const sendBtn = await screen.findByRole("button", { name: /Send to Log Buy —/ });
    await act(async () => { fireEvent.click(sendBtn); });

    const stored = JSON.parse(localStorage.getItem("ps_prefill") || "{}");
    expect(stored.ticker).toBe("AAPL");
    expect(stored.price).toBe(100);
    expect(stored.stopMode).toBe("price");
    // Tech-stop math: entry - (entry × stopDistPct/100) = 100 × 0.97 = 97.
    expect(stored.stop).toBeCloseTo(97, 2);
    // No multiplier on tech-stop payload.
    expect(stored.atrMultiplier).toBeUndefined();
    expect(stored.action).toBe("new");
  });
});
