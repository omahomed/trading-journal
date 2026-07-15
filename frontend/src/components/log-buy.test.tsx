import { render, screen, waitFor, fireEvent, act } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach } from "vitest";

class ResizeObserverStub {
  observe(): void { /* noop */ }
  unobserve(): void { /* noop */ }
  disconnect(): void { /* noop */ }
}
(globalThis as any).ResizeObserver = ResizeObserverStub;

// JSDOM ships with a localStorage stub that throws on most methods (the
// "--localstorage-file was provided without a valid path" warning is the
// symptom). Replace with a minimal in-memory implementation so the
// importer-prefill tests can seed ps_prefill before render. Same pattern
// as position-sizer.test.tsx.
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
    journalLatest: vi.fn(),
    tradesOpen: vi.fn(),
    tradesOpenDetails: vi.fn(),
    rallyPrefix: vi.fn(),
    nextTradeId: vi.fn(),
    priceLookup: vi.fn(),
    logBuy: vi.fn(),
    listStrategies: vi.fn(),
  },
  getActivePortfolio: () => "CanSlim",
}));

vi.mock("@/lib/upload-with-timeout", () => ({
  uploadWithTimeout: vi.fn(),
  DEFAULT_UPLOAD_TIMEOUT_MS: 60_000,
}));

const SEED_STRATEGIES = [
  { name: "CanSlim",     description: "primary",   color: "#6366f1", is_active: true, created_at: "2026-01-01T00:00:00" },
  { name: "StockTalk",   description: "small-cap", color: "#d97706", is_active: true, created_at: "2026-01-01T00:00:01" },
  { name: "21eStrategy", description: "21 EMA",    color: "#0d9488", is_active: true, created_at: "2026-01-01T00:00:02" },
];

import { api } from "@/lib/api";
import { uploadWithTimeout } from "@/lib/upload-with-timeout";
import { LogBuy } from "./log-buy";

const mRally = vi.mocked(api.rallyPrefix);
const mUpload = vi.mocked(uploadWithTimeout);

function setupDefaults() {
  vi.mocked(api.journalLatest).mockResolvedValue({ end_nlv: 100000 } as any);
  vi.mocked(api.tradesOpen).mockResolvedValue([]);
  vi.mocked(api.tradesOpenDetails).mockResolvedValue({ details: [], lot_closures: [] });
  vi.mocked(api.nextTradeId).mockResolvedValue({ trade_id: "202604-001" } as any);
  vi.mocked(api.listStrategies).mockResolvedValue(SEED_STRATEGIES as any);
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
      // Post-retier: POWERTREND → Offense at 0.75% (was 1.00%).
      expect(indicator.textContent).toMatch(/Offense/);
      expect(indicator.textContent).toMatch(/0\.75% risk/);
      expect(indicator.textContent).toMatch(/from M Factor POWERTREND/);
    });
  });

  test("manual override radios are present (Override Sizing Mode field)", async () => {
    // The user can toggle pilot/normal/offense for THIS Log Buy
    // submission (Defense tier retired). Override is form-local —
    // refresh / submit / unmount resets back to the MCT-driven auto
    // pick (verified separately by the post-remount test below).
    mRally.mockResolvedValue({ prefix: "", state: "POWERTREND" } as any);

    render(<LogBuy navColor="#6366f1" />);

    await screen.findByTestId("logbuy-sizing-mode-indicator");
    expect(screen.getByText("Override Sizing Mode")).toBeInTheDocument();
  });

  test("clicking a different mode flips indicator to '— manual override' and shows Reset", async () => {
    // Auto: Offense (POWERTREND). User toggles to Pilot → indicator
    // copy switches; Reset button appears. (Defense was retired; Pilot
    // is the manual downshift target now.)
    mRally.mockResolvedValue({ prefix: "", state: "POWERTREND" } as any);

    render(<LogBuy navColor="#6366f1" />);

    const indicator = await screen.findByTestId("logbuy-sizing-mode-indicator");
    await waitFor(() => expect(indicator.textContent).toMatch(/Offense/));
    expect(screen.queryByTestId("logbuy-reset-to-mct")).not.toBeInTheDocument();

    // Click the Pilot radio. Mode label includes "Pilot (0.25%)".
    const pilotRadio = await screen.findByText(/Pilot \(0\.25%\)/);
    await act(async () => { fireEvent.click(pilotRadio); });

    await waitFor(() => {
      expect(indicator.textContent).toMatch(/Pilot/);
      expect(indicator.textContent).toMatch(/0\.25% risk/);
      expect(indicator.textContent).toMatch(/manual override/);
      // No "from M Factor …" label while in manual mode (source is the user)
      expect(indicator.textContent).not.toMatch(/from M Factor/);
    });
    expect(screen.getByTestId("logbuy-reset-to-mct")).toBeInTheDocument();
  });

  test("'Reset to MCT' restores the auto-derived mode", async () => {
    mRally.mockResolvedValue({ prefix: "", state: "POWERTREND" } as any);

    render(<LogBuy navColor="#6366f1" />);

    // Override to Pilot first (was Defense pre-retier)
    const pilotRadio = await screen.findByText(/Pilot \(0\.25%\)/);
    await act(async () => { fireEvent.click(pilotRadio); });

    const reset = await screen.findByTestId("logbuy-reset-to-mct");
    await act(async () => { fireEvent.click(reset); });

    const indicator = screen.getByTestId("logbuy-sizing-mode-indicator");
    await waitFor(() => {
      // Back to Auto + Offense (POWERTREND) at the retiered 0.75%.
      expect(indicator.textContent).toMatch(/Offense/);
      expect(indicator.textContent).toMatch(/0\.75% risk/);
      expect(indicator.textContent).toMatch(/from M Factor POWERTREND/);
      expect(indicator.textContent).not.toMatch(/manual override/);
    });
    expect(screen.queryByTestId("logbuy-reset-to-mct")).not.toBeInTheDocument();
  });

  test("override does not persist across remount — fresh render re-derives from M Factor", async () => {
    // Page-reload analogue: unmount + remount the component. State is
    // local to the React tree, so a remount drops sizingModeManual and
    // re-runs the mount effect, which re-applies the MCT auto pick.
    mRally.mockResolvedValue({ prefix: "", state: "POWERTREND" } as any);

    const { unmount } = render(<LogBuy navColor="#6366f1" />);
    const pilotRadio = await screen.findByText(/Pilot \(0\.25%\)/);
    await act(async () => { fireEvent.click(pilotRadio); });
    await screen.findByTestId("logbuy-reset-to-mct");
    unmount();

    // Remount with the same MCT mock — should land back on Offense auto.
    render(<LogBuy navColor="#6366f1" />);
    const indicator = await screen.findByTestId("logbuy-sizing-mode-indicator");
    await waitFor(() => {
      expect(indicator.textContent).toMatch(/Offense/);
      expect(indicator.textContent).toMatch(/from M Factor POWERTREND/);
    });
    expect(screen.queryByTestId("logbuy-reset-to-mct")).not.toBeInTheDocument();
  });

  test("CORRECTION → Pilot (0.25% risk) — retiered from Defense", async () => {
    mRally.mockResolvedValue({ prefix: "", state: "CORRECTION" } as any);

    render(<LogBuy navColor="#6366f1" />);

    const indicator = await screen.findByTestId("logbuy-sizing-mode-indicator");
    await waitFor(() => {
      expect(indicator.textContent).toMatch(/Pilot/);
      expect(indicator.textContent).toMatch(/0\.25% risk/);
      expect(indicator.textContent).toMatch(/from M Factor CORRECTION/);
    });
  });

  test("rally-prefix returning no state defaults to Pilot + 'M Factor state unknown'", async () => {
    // Post-retier default is Pilot (safest floor) rather than a
    // middle-tier auto-pick. Old default was Normal (safe middle) —
    // that middle-tier default was retired along with Defense to
    // preserve the "when in doubt, be smaller" invariant.
    mRally.mockResolvedValue({ prefix: "" } as any);

    render(<LogBuy navColor="#6366f1" />);

    const indicator = await screen.findByTestId("logbuy-sizing-mode-indicator");
    await waitFor(() => {
      expect(indicator.textContent).toMatch(/Pilot/);
      expect(indicator.textContent).toMatch(/0\.25% risk/);
      expect(indicator.textContent).toMatch(/M Factor state unknown/);
    });
  });
});


// ─────────────────────────────────────────────────────────────────────
// Stop-loss visibility for option trades.
// Stocks render the field as today; options hide it behind a "Show stop
// loss" reveal link, and submit sends stop_loss: null when hidden.
// ─────────────────────────────────────────────────────────────────────

const STOCK_TICKER = "AAPL";
const OPTION_TICKER = "AMZN 260619 $235C";
const FIRST_RULE = "br1.1 Consolidation"; // First entry in BUY_RULES.

function fillByLabel(labelText: string, value: string): void {
  // Field renders <div><label>{label}</label>{children}</div>, so the
  // label's parent is the wrapping div. Grab the input/textarea inside.
  const label = screen.getByText(labelText);
  const input = label.parentElement?.querySelector("input, textarea") as
    | HTMLInputElement
    | HTMLTextAreaElement
    | null;
  if (!input) throw new Error(`No input found in Field "${labelText}"`);
  fireEvent.change(input, { target: { value } });
}

async function selectBuyRule(rule: string): Promise<void> {
  // SearchSelect's trigger is the first <button> inside the Field. Open
  // the dropdown, then click the rule option (rendered as a button by
  // its display text).
  const ruleField = screen.getByText("Primary Buy Rule *");
  const trigger = ruleField.parentElement?.querySelector("button") as HTMLButtonElement;
  await act(async () => { fireEvent.click(trigger); });
  await act(async () => { fireEvent.click(screen.getByText(rule)); });
}

describe("LogBuy — options stop-loss visibility", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaults();
    mRally.mockResolvedValue({ prefix: "", state: "POWERTREND" } as any);
    // Never-resolving promise — keeps the debounced ticker→price lookup
    // from overwriting the price the test types in. The component's
    // .catch() guard handles the unresolved case cleanly.
    vi.mocked(api.priceLookup).mockReturnValue(new Promise(() => {}) as any);
    vi.mocked(api.logBuy).mockResolvedValue({ trx_id: "B1" } as any);
  });

  test("stock ticker shows stop-loss block by default", async () => {
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");

    fillByLabel("Ticker Symbol", STOCK_TICKER);

    await waitFor(() => {
      expect(screen.getByText("Stop Loss Mode")).toBeInTheDocument();
    });
    expect(screen.queryByText("Show stop loss")).not.toBeInTheDocument();
  });

  test("option ticker hides stop-loss block and shows reveal link", async () => {
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");

    fillByLabel("Ticker Symbol", OPTION_TICKER);

    await waitFor(() => {
      expect(screen.queryByText("Stop Loss Mode")).not.toBeInTheDocument();
    });
    expect(screen.getByText("Show stop loss")).toBeInTheDocument();
  });

  test("clicking 'Show stop loss' reveals the field", async () => {
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");

    fillByLabel("Ticker Symbol", OPTION_TICKER);
    const link = await screen.findByText("Show stop loss");
    await act(async () => { fireEvent.click(link); });

    await waitFor(() => {
      expect(screen.getByText("Stop Loss Mode")).toBeInTheDocument();
    });
    expect(screen.queryByText("Show stop loss")).not.toBeInTheDocument();
  });

  test("option submit with stop hidden sends stop_loss: null", async () => {
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");

    fillByLabel("Ticker Symbol", OPTION_TICKER);
    await waitFor(() =>
      expect(screen.queryByText("Stop Loss Mode")).not.toBeInTheDocument()
    );
    await selectBuyRule(FIRST_RULE);
    fillByLabel("Contracts to Add", "5");
    fillByLabel("Premium per Contract ($)", "35.50");

    await act(async () => {
      fireEvent.click(screen.getByText("LOG BUY ORDER"));
    });

    await waitFor(() => expect(api.logBuy).toHaveBeenCalled());
    const body = vi.mocked(api.logBuy).mock.calls[0][0] as Record<string, unknown>;
    // Explicit null in the payload — more forward-compatible and clearer
    // to inspect than an absent key.
    expect(body.stop_loss).toBeNull();
  });

  test("option submit with stop revealed sends entered value", async () => {
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");

    fillByLabel("Ticker Symbol", OPTION_TICKER);
    await act(async () => {
      fireEvent.click(await screen.findByText("Show stop loss"));
    });
    await selectBuyRule(FIRST_RULE);
    fillByLabel("Contracts to Add", "5");
    fillByLabel("Premium per Contract ($)", "10.00");
    // Default for option is pct mode + 50%, so stop_loss = 10 * (1 - 0.5) = 5.0

    await act(async () => {
      fireEvent.click(screen.getByText("LOG BUY ORDER"));
    });

    await waitFor(() => expect(api.logBuy).toHaveBeenCalled());
    const body = vi.mocked(api.logBuy).mock.calls[0][0] as Record<string, unknown>;
    expect(body.stop_loss).toBeCloseTo(5.0, 4);
  });

  test("switching from option back to stock re-shows the field", async () => {
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");

    fillByLabel("Ticker Symbol", OPTION_TICKER);
    await waitFor(() =>
      expect(screen.queryByText("Stop Loss Mode")).not.toBeInTheDocument()
    );

    fillByLabel("Ticker Symbol", STOCK_TICKER);
    await waitFor(() => {
      expect(screen.getByText("Stop Loss Mode")).toBeInTheDocument();
    });
    expect(screen.queryByText("Show stop loss")).not.toBeInTheDocument();
  });

  test("option ticker suppresses < 8% warning even when stop revealed", async () => {
    // 50% stop on a $10 premium = stopPct=50, well above the > 10
    // trigger. For a stock this would render the warning; for an
    // option the !isOption gate suppresses it.
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");

    fillByLabel("Ticker Symbol", OPTION_TICKER);
    await act(async () => {
      fireEvent.click(await screen.findByText("Show stop loss"));
    });
    await selectBuyRule(FIRST_RULE);
    fillByLabel("Contracts to Add", "5");
    fillByLabel("Premium per Contract ($)", "10.00");

    await act(async () => {
      fireEvent.click(screen.getByText("LOG BUY ORDER"));
    });

    // Validate runs on submit click; if the warning had fired, it would
    // be in the DOM by now. Wait for the submit to settle, then assert
    // the warning text is absent.
    await waitFor(() => expect(api.logBuy).toHaveBeenCalled());
    expect(screen.queryByText(/recommend < 8%/)).not.toBeInTheDocument();
  });
});


// ─────────────────────────────────────────────────────────────────────
// Strategy dropdown (Migration 019).
// Defaults to CanSlim, ships in submit body, and on scale-in inherits
// from the parent campaign with the field rendered read-only.
// ─────────────────────────────────────────────────────────────────────

function findStrategyTrigger(): HTMLButtonElement {
  // Field renders <Field label="Strategy *">…SearchSelect…</Field>. The
  // SearchSelect's trigger is the first <button> inside the Field's parent
  // div, same accessor pattern as selectBuyRule above.
  const label = screen.getByText("Strategy *");
  const trigger = label.parentElement?.querySelector("button") as HTMLButtonElement;
  if (!trigger) throw new Error("Strategy trigger not found");
  return trigger;
}

describe("LogBuy — strategy dropdown", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaults();
    mRally.mockResolvedValue({ prefix: "", state: "POWERTREND" } as any);
    vi.mocked(api.priceLookup).mockReturnValue(new Promise(() => {}) as any);
    vi.mocked(api.logBuy).mockResolvedValue({ trx_id: "B1" } as any);
  });

  test("renders Strategy dropdown defaulted to CanSlim", async () => {
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");

    // Wait for listStrategies to resolve and populate the dropdown.
    await waitFor(() => expect(api.listStrategies).toHaveBeenCalled());

    const trigger = findStrategyTrigger();
    expect(trigger.textContent).toContain("CanSlim");
  });

  test("submitting a new buy includes strategy in the request body", async () => {
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");
    await waitFor(() => expect(api.listStrategies).toHaveBeenCalled());

    fillByLabel("Ticker Symbol", STOCK_TICKER);
    await selectBuyRule(FIRST_RULE);
    fillByLabel("Shares to Add", "100");
    fillByLabel("Price ($)", "200.00");

    await act(async () => {
      fireEvent.click(screen.getByText("LOG BUY ORDER"));
    });

    await waitFor(() => expect(api.logBuy).toHaveBeenCalled());
    const body = vi.mocked(api.logBuy).mock.calls[0][0] as Record<string, unknown>;
    expect(body.strategy).toBe("CanSlim");
  });

  test("user can switch strategy and the new value flows into the submit body", async () => {
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");
    await waitFor(() => expect(api.listStrategies).toHaveBeenCalled());

    // Open Strategy dropdown and pick StockTalk.
    const trigger = findStrategyTrigger();
    await act(async () => { fireEvent.click(trigger); });
    await act(async () => { fireEvent.click(screen.getByText("StockTalk")); });

    fillByLabel("Ticker Symbol", STOCK_TICKER);
    await selectBuyRule(FIRST_RULE);
    fillByLabel("Shares to Add", "100");
    fillByLabel("Price ($)", "200.00");

    await act(async () => {
      fireEvent.click(screen.getByText("LOG BUY ORDER"));
    });

    await waitFor(() => expect(api.logBuy).toHaveBeenCalled());
    const body = vi.mocked(api.logBuy).mock.calls[0][0] as Record<string, unknown>;
    expect(body.strategy).toBe("StockTalk");
  });

  test("scale-in disables the Strategy field and prefills from parent", async () => {
    // Parent campaign tagged 21eStrategy. Switching to scale-in mode and
    // selecting it should lock the dropdown to that value.
    vi.mocked(api.tradesOpen).mockResolvedValue([{
      trade_id: "202604-077", ticker: "MSFT", status: "OPEN",
      shares: 50, avg_entry: 350, total_cost: 17500, realized_pl: 0, rule: "br1.1",
      strategy: "21eStrategy",
    }] as any);

    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");
    await waitFor(() => expect(api.listStrategies).toHaveBeenCalled());

    // Switch to scale-in
    await act(async () => {
      fireEvent.click(screen.getByText("Scale In (Add to Existing)"));
    });

    // Pick the parent campaign from the campaign-picker SearchSelect.
    const campaignField = screen.getByText("Select Existing Campaign");
    const campaignTrigger = campaignField.parentElement?.querySelector("button") as HTMLButtonElement;
    await act(async () => { fireEvent.click(campaignTrigger); });
    await act(async () => { fireEvent.click(screen.getByText("MSFT | 202604-077")); });

    // Strategy now reflects the parent and the trigger is disabled.
    await waitFor(() => {
      const trigger = findStrategyTrigger();
      expect(trigger.textContent).toContain("21eStrategy");
      expect(trigger.disabled).toBe(true);
    });
    expect(screen.getByText("Inherited from parent campaign")).toBeInTheDocument();
  });
});


// ─────────────────────────────────────────────────────────────────────
// ATR stop-loss mode — third option alongside Price Level / Percentage.
// Captures atr_pct from /api/prices/lookup, renders multiplier pills
// (1× / 1.5× / 2×, default 1.5×), disables pills when atrPct=0, hides
// the radio entirely on option tickers, and resolves the stop inline
// at submit. Importer prefill without explicit stop fields defaults
// to ATR when atrPct > 0; falls through to pct/5.0 when ATR is
// unavailable (existing behavior, no regression).
// ─────────────────────────────────────────────────────────────────────


describe("LogBuy — ATR stop-loss mode", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaults();
    mRally.mockResolvedValue({ prefix: "", state: "POWERTREND" } as any);
    vi.mocked(api.logBuy).mockResolvedValue({ trx_id: "B1" } as any);
  });

  // Helper: stub priceLookup to resolve with a specific atr_pct on the
  // first call. ATR landing requires settling the debounce + flushing
  // the promise chain, so tests wait for the value to flow into state.
  function mockPriceLookup(price: number, atrPct: number) {
    vi.mocked(api.priceLookup).mockResolvedValue({
      ticker: "AAPL", price, atr: atrPct * price / 100, atr_pct: atrPct,
    } as any);
  }

  test("third radio 'ATR (×)' renders in Stop Loss Mode group on stocks", async () => {
    mockPriceLookup(180, 2.0);
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");
    fillByLabel("Ticker Symbol", STOCK_TICKER);

    await waitFor(() => {
      expect(screen.getByText(/ATR \(×\)/)).toBeInTheDocument();
    });
  });

  test("multiplier pills 1× / 1.5× / 2× render with default selection 1.5×", async () => {
    mockPriceLookup(180, 2.0);
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");
    fillByLabel("Ticker Symbol", STOCK_TICKER);
    // Wait for the priceLookup mock to flush so atrPct lands and the
    // ATR pills are enabled.
    await waitFor(() => expect(api.priceLookup).toHaveBeenCalled());
    await act(async () => { await Promise.resolve(); });

    // Click ATR radio
    await act(async () => {
      fireEvent.click(screen.getByText(/ATR \(×\)/));
    });

    const pills = await screen.findByTestId("logbuy-atr-pills");
    const buttons = pills.querySelectorAll("button");
    expect(buttons).toHaveLength(3);
    expect(buttons[0].textContent).toContain("1×");
    expect(buttons[1].textContent).toContain("1.5×");
    expect(buttons[2].textContent).toContain("2×");
    // Default = 1.5× → aria-pressed=true on middle pill.
    expect(buttons[0].getAttribute("aria-pressed")).toBe("false");
    expect(buttons[1].getAttribute("aria-pressed")).toBe("true");
    expect(buttons[2].getAttribute("aria-pressed")).toBe("false");
  });

  test("ATR pills disabled when atrPct fetched as 0 (insufficient history)", async () => {
    mockPriceLookup(180, 0);  // ATR-unavailable sentinel
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");
    fillByLabel("Ticker Symbol", STOCK_TICKER);
    await waitFor(() => expect(api.priceLookup).toHaveBeenCalled());
    await act(async () => { await Promise.resolve(); });

    await act(async () => {
      fireEvent.click(screen.getByText(/ATR \(×\)/));
    });

    const pills = await screen.findByTestId("logbuy-atr-pills");
    pills.querySelectorAll("button").forEach(btn => {
      expect((btn as HTMLButtonElement).disabled).toBe(true);
    });
  });

  test("ATR radio hidden on option tickers", async () => {
    mockPriceLookup(35.5, 0);
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");
    fillByLabel("Ticker Symbol", OPTION_TICKER);
    // Reveal the stop-loss block so the radio group is visible.
    await act(async () => {
      fireEvent.click(await screen.findByText("Show stop loss"));
    });

    expect(screen.getByText("Stop Loss Mode")).toBeInTheDocument();
    expect(screen.getByText(/Price Level/)).toBeInTheDocument();
    expect(screen.getByText(/Percentage/)).toBeInTheDocument();
    expect(screen.queryByText(/ATR \(×\)/)).not.toBeInTheDocument();
  });

  test("submit math: ATR mode resolves stop_loss to price × (1 − multiplier × atrPct/100)", async () => {
    mockPriceLookup(180, 2.0);
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");

    fillByLabel("Ticker Symbol", STOCK_TICKER);
    await waitFor(() => expect(api.priceLookup).toHaveBeenCalled());
    await act(async () => { await Promise.resolve(); });

    // Override the auto-fetched price so the submit math uses 100.00.
    fillByLabel("Price ($)", "100");
    await selectBuyRule(FIRST_RULE);
    fillByLabel("Shares to Add", "50");

    // Pick ATR mode at default 1.5×. Expected stop_loss = 100 × (1 − 1.5 × 2/100) = 97.
    await act(async () => {
      fireEvent.click(screen.getByText(/ATR \(×\)/));
    });

    await act(async () => {
      fireEvent.click(screen.getByText("LOG BUY ORDER"));
    });

    await waitFor(() => expect(api.logBuy).toHaveBeenCalled());
    const body = vi.mocked(api.logBuy).mock.calls[0][0] as Record<string, unknown>;
    expect(body.stop_loss).toBeCloseTo(97, 4);
  });

  test("selecting ATR radio flips stopMode and replaces the slPct input with the pills", async () => {
    mockPriceLookup(180, 2.0);
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");
    fillByLabel("Ticker Symbol", STOCK_TICKER);
    await waitFor(() => expect(api.priceLookup).toHaveBeenCalled());
    await act(async () => { await Promise.resolve(); });

    // Before ATR click: slPct input visible, no pills.
    expect(screen.getByText("Stop Loss %")).toBeInTheDocument();
    expect(screen.queryByTestId("logbuy-atr-pills")).not.toBeInTheDocument();

    await act(async () => {
      fireEvent.click(screen.getByText(/ATR \(×\)/));
    });

    // After ATR click: pills visible, slPct input gone, label flipped.
    await waitFor(() => {
      expect(screen.getByTestId("logbuy-atr-pills")).toBeInTheDocument();
    });
    expect(screen.queryByText("Stop Loss %")).not.toBeInTheDocument();
    expect(screen.getByText("ATR Multiplier")).toBeInTheDocument();
  });

  test("Sizer handoff with {stopMode:'atr', atrMultiplier:1.5} sets Log Buy to ATR mode + 1.5× pill", async () => {
    mockPriceLookup(180, 2.0);
    localStorage.setItem("ps_prefill", JSON.stringify({
      ticker: STOCK_TICKER, shares: 50, price: 180,
      stopMode: "atr", atrMultiplier: 1.5, action: "new",
    }));
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");
    await waitFor(() => expect(api.priceLookup).toHaveBeenCalled());
    await act(async () => { await Promise.resolve(); });

    // ATR pills should be rendered (mode flipped) with 1.5× selected.
    const pills = await screen.findByTestId("logbuy-atr-pills");
    const buttons = pills.querySelectorAll("button");
    expect(buttons[1].getAttribute("aria-pressed")).toBe("true");
  });

  test("Sizer handoff with stopMode:'price' + resolved stop sets Log Buy to price mode (ride-along bugfix #1)", async () => {
    mockPriceLookup(100, 2.0);
    localStorage.setItem("ps_prefill", JSON.stringify({
      ticker: STOCK_TICKER, shares: 100, price: 100,
      stop: 97, stopMode: "price", action: "new",
    }));
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");
    await waitFor(() => expect(api.priceLookup).toHaveBeenCalled());
    await act(async () => { await Promise.resolve(); });

    // Price mode active → Stop Price ($) label, NOT Stop Loss %.
    await waitFor(() => {
      expect(screen.getByText("Stop Price ($)")).toBeInTheDocument();
    });
    expect(screen.queryByText("Stop Loss %")).not.toBeInTheDocument();
    // The Stop Price input is pre-filled with the resolved dollar value.
    const label = screen.getByText("Stop Price ($)");
    const input = label.parentElement?.querySelector("input") as HTMLInputElement;
    expect(input.value).toBe("97.00");
  });

  test("importer prefill with no stop + atrPct=0 falls back to pct/5.0 default (no regression)", async () => {
    mockPriceLookup(180, 0);  // sparse-history sentinel
    localStorage.setItem("ps_prefill", JSON.stringify({
      ticker: STOCK_TICKER, shares: 50, price: 180, action: "new",
    }));
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");
    await waitFor(() => expect(api.priceLookup).toHaveBeenCalled());
    await act(async () => { await Promise.resolve(); });

    // Settling: the pendingAtrDefault watcher fires on atrPct/fetchingPrice
    // change. Give the effect tick one more flush.
    await act(async () => { await Promise.resolve(); });

    // Default pct/5.0 → Stop Loss % label visible, no ATR pills.
    await waitFor(() => {
      expect(screen.getByText("Stop Loss %")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("logbuy-atr-pills")).not.toBeInTheDocument();
    // Input shows the historical "5.0" default.
    const label = screen.getByText("Stop Loss %");
    const input = label.parentElement?.querySelector("input") as HTMLInputElement;
    expect(input.value).toBe("5.0");
  });

  test("importer prefill with no stop + atrPct > 0 defaults to ATR mode at 1.5×", async () => {
    mockPriceLookup(180, 2.5);
    localStorage.setItem("ps_prefill", JSON.stringify({
      ticker: STOCK_TICKER, shares: 50, price: 180, action: "new",
    }));
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");
    // pendingAtrDefault effect waits on the debounced priceLookup +
    // promise chain. findByTestId polls until the pills render or
    // its timeout (default 1000ms) elapses — longer than the 600ms
    // debounce so the effect chain has time to complete.
    const pills = await screen.findByTestId("logbuy-atr-pills", {}, { timeout: 3000 });
    const buttons = pills.querySelectorAll("button");
    expect(buttons[1].getAttribute("aria-pressed")).toBe("true");
  });

  test("ATR-unavailable explanatory line renders when atrPct = 0", async () => {
    mockPriceLookup(180, 0);
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");
    fillByLabel("Ticker Symbol", STOCK_TICKER);
    await waitFor(() => expect(api.priceLookup).toHaveBeenCalled());
    await act(async () => { await Promise.resolve(); });

    await act(async () => {
      fireEvent.click(screen.getByText(/ATR \(×\)/));
    });

    await waitFor(() => {
      expect(screen.getByText(/ATR unavailable for this ticker/)).toBeInTheDocument();
    });
  });

  test("switching to an option ticker resets stopMode from 'atr' back to 'pct' (ride-along bugfix #2)", async () => {
    mockPriceLookup(180, 2.0);
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");

    // Start on a stock ticker, flip to ATR mode.
    fillByLabel("Ticker Symbol", STOCK_TICKER);
    await waitFor(() => expect(api.priceLookup).toHaveBeenCalled());
    await act(async () => { await Promise.resolve(); });
    await act(async () => {
      fireEvent.click(screen.getByText(/ATR \(×\)/));
    });
    await waitFor(() => {
      expect(screen.getByTestId("logbuy-atr-pills")).toBeInTheDocument();
    });

    // Switch to an option ticker → entire stop-loss block hides (the
    // existing options behavior). Reveal it via "Show stop loss" and
    // confirm the ATR radio is gone AND we're not still in ATR mode
    // (no pills should render).
    fillByLabel("Ticker Symbol", OPTION_TICKER);
    await waitFor(() => {
      expect(screen.queryByText("Stop Loss Mode")).not.toBeInTheDocument();
    });
    await act(async () => {
      fireEvent.click(await screen.findByText("Show stop loss"));
    });
    await waitFor(() => {
      expect(screen.getByText("Stop Loss Mode")).toBeInTheDocument();
    });
    // ATR radio hidden on options; pills not rendered (mode reset to pct).
    expect(screen.queryByText(/ATR \(×\)/)).not.toBeInTheDocument();
    expect(screen.queryByTestId("logbuy-atr-pills")).not.toBeInTheDocument();
    // Percent input visible (pct mode active again).
    expect(screen.getByText("Stop Loss %")).toBeInTheDocument();
  });
});


// ─────────────────────────────────────────────────────────────────────
// ATR UI polish (post-deployment review):
//   - Live Sizer panel now carries an informational ATR row between
//     Account Equity and Risk $; subtle metadata styling.
//   - Multiplier pills under ATR mode render all three (1× / 1.5× / 2×)
//     regardless of which is selected, with the selected pill clearly
//     distinguished by inverted color/border (background = ink-1,
//     color = surface). Inactive pills are outlined (background = bg,
//     border = border). The 1.5× pill is the default selection.
// ─────────────────────────────────────────────────────────────────────


describe("LogBuy — ATR Live Sizer info row + pill rendering", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaults();
    mRally.mockResolvedValue({ prefix: "", state: "POWERTREND" } as any);
    vi.mocked(api.logBuy).mockResolvedValue({ trx_id: "B1" } as any);
  });

  function mockPriceLookup(price: number, atrPct: number) {
    vi.mocked(api.priceLookup).mockResolvedValue({
      ticker: "AAPL", price, atr: atrPct * price / 100, atr_pct: atrPct,
    } as any);
  }

  test("ATR info row hidden when no ticker entered", async () => {
    mockPriceLookup(180, 2.0);
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");
    // No ticker → no priceLookup → no ATR row.
    expect(screen.queryByTestId("logbuy-atr-info")).not.toBeInTheDocument();
  });

  test("ATR info row renders ATR % + $/sh when atrPct > 0", async () => {
    mockPriceLookup(180, 2.87);
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");

    fillByLabel("Ticker Symbol", STOCK_TICKER);
    await waitFor(() => expect(api.priceLookup).toHaveBeenCalled());

    const row = await screen.findByTestId("logbuy-atr-info", {}, { timeout: 3000 });
    expect(row.textContent).toMatch(/ATR \(21d\):/);
    // 180 × 2.87 / 100 = 5.166 → formatted as $5.17
    expect(row.textContent).toMatch(/2\.87%/);
    expect(row.textContent).toMatch(/\$5\.17/);
    expect(row.textContent).toMatch(/\/sh/);
  });

  test("ATR info row renders 'unavailable' when atrPct = 0 (sparse-history sentinel)", async () => {
    mockPriceLookup(180, 0);
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");

    fillByLabel("Ticker Symbol", STOCK_TICKER);
    await waitFor(() => expect(api.priceLookup).toHaveBeenCalled());

    const row = await screen.findByTestId("logbuy-atr-info", {}, { timeout: 3000 });
    expect(row.textContent).toMatch(/ATR \(21d\):/);
    expect(row.textContent).toMatch(/unavailable for this ticker/);
    // No percent or $/sh values when unavailable.
    expect(row.textContent).not.toMatch(/%/);
    expect(row.textContent).not.toMatch(/\/sh/);
  });

  test("ATR pills: all three render with the selected pill visually distinguished", async () => {
    mockPriceLookup(180, 2.0);
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");
    fillByLabel("Ticker Symbol", STOCK_TICKER);
    await waitFor(() => expect(api.priceLookup).toHaveBeenCalled());
    await act(async () => { await Promise.resolve(); });

    // Activate ATR mode so the pills render.
    await act(async () => {
      fireEvent.click(screen.getByText(/ATR \(×\)/));
    });

    const pills = await screen.findByTestId("logbuy-atr-pills");
    const buttons = pills.querySelectorAll("button");

    // All three pills present in the DOM. Earlier production screenshot
    // raised concern that the 1.5× pill was missing visually; this
    // assertion locks in that all three are rendered, with 1.5× as the
    // middle button.
    expect(buttons).toHaveLength(3);
    expect(buttons[0].textContent).toContain("1×");
    expect(buttons[1].textContent).toContain("1.5×");
    expect(buttons[2].textContent).toContain("2×");

    // Visual distinction: selected pill (index 1 = 1.5× default) gets
    // background = var(--ink-1) (filled) and color = var(--surface)
    // (inverted text). Inactive pills get background = var(--bg) with
    // border-only treatment. aria-pressed exposes the selection state
    // for screen readers and tests.
    expect(buttons[0].getAttribute("aria-pressed")).toBe("false");
    expect(buttons[1].getAttribute("aria-pressed")).toBe("true");
    expect(buttons[2].getAttribute("aria-pressed")).toBe("false");

    const selectedStyle = (buttons[1] as HTMLButtonElement).style;
    const inactiveStyle = (buttons[0] as HTMLButtonElement).style;
    // Selected uses ink-1 background; inactive uses bg. These resolve
    // through CSS vars at render time — we just assert the inline-style
    // strings differ, proving the two states render with different
    // declarations and the selected pill can't visually collapse into
    // the inactive pills.
    expect(selectedStyle.background).not.toBe(inactiveStyle.background);
    expect(selectedStyle.color).not.toBe(inactiveStyle.color);
  });

  test("ATR caption ('Default mode for buys with no stop …') is removed", async () => {
    // Regression guard for the post-deployment polish — the caption
    // line was useful in the mockup but adds noise in the live form.
    mockPriceLookup(180, 2.0);
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");
    fillByLabel("Ticker Symbol", STOCK_TICKER);
    await waitFor(() => expect(api.priceLookup).toHaveBeenCalled());
    await act(async () => { await Promise.resolve(); });

    await act(async () => {
      fireEvent.click(screen.getByText(/ATR \(×\)/));
    });

    // Confirmation line still present.
    await waitFor(() => {
      expect(screen.getByText(/→ Stop \$/)).toBeInTheDocument();
    });
    // Removed caption.
    expect(screen.queryByText(/Default mode for buys with no stop/)).not.toBeInTheDocument();
    expect(screen.queryByText(/replaces the old 5% percentage default/)).not.toBeInTheDocument();
  });
});


// ─────────────────────────────────────────────────────────────────────
// Post-submit state refresh.
// After a successful buy the page must refetch tradesOpen + details
// (so openTrades / allDetails reflect the new lot) and pull a fresh
// nextTradeId (the one we just used is consumed). Without this the
// next same-page submission reads pre-submit state.
// ─────────────────────────────────────────────────────────────────────
describe("LogBuy — post-submit state refresh", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaults();
    mRally.mockResolvedValue({ prefix: "", state: "POWERTREND" } as any);
    vi.mocked(api.priceLookup).mockReturnValue(new Promise(() => {}) as any);
    vi.mocked(api.logBuy).mockResolvedValue({ trx_id: "B1", trade_id: "202604-001" } as any);
  });

  test("refetches tradesOpen + tradesOpenDetails + nextTradeId after a successful new buy", async () => {
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");

    // Wait for initial fetches to settle before counting.
    await waitFor(() => expect(api.tradesOpen).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.tradesOpenDetails).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.nextTradeId).toHaveBeenCalledTimes(1));

    fillByLabel("Ticker Symbol", STOCK_TICKER);
    await selectBuyRule(FIRST_RULE);
    fillByLabel("Shares to Add", "10");
    fillByLabel("Price ($)", "150.00");

    await act(async () => {
      fireEvent.click(screen.getByText("LOG BUY ORDER"));
    });

    // The post-submit refresh fires one more call per endpoint.
    await waitFor(() => expect(api.tradesOpen).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(api.tradesOpenDetails).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(api.nextTradeId).toHaveBeenCalledTimes(2));
  });

  test("refetches tradesOpen + tradesOpenDetails after a successful scale-in (no nextTradeId)", async () => {
    // Seed openTrades with one campaign so the scale-in mode has something
    // to pick. Re-running setupDefaults() inside the test would override
    // beforeEach's mocks — use mockResolvedValueOnce instead so the initial
    // mount fetch returns the seeded list.
    vi.mocked(api.tradesOpen).mockResolvedValueOnce([
      { trade_id: "202604-001", ticker: "AAPL", status: "OPEN", shares: 100, avg_entry: 140, total_cost: 14000,
        realized_pl: 0, rule: "br1.1 Consolidation", strategy: "CanSlim" } as any,
    ]);
    vi.mocked(api.tradesOpenDetails).mockResolvedValueOnce({
      details: [{ trade_id: "202604-001", ticker: "AAPL", action: "BUY", date: "2026-04-01",
                  shares: 100, amount: 140, value: 14000, rule: "br1.1 Consolidation" } as any],
      lot_closures: [],
    });

    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");

    await waitFor(() => expect(api.tradesOpen).toHaveBeenCalledTimes(1));
    const nextTradeIdInitialCount = vi.mocked(api.nextTradeId).mock.calls.length;

    // Switch to scale-in mode.
    await act(async () => {
      fireEvent.click(screen.getByText("Scale In (Add to Existing)"));
    });

    // Pick the seeded campaign via the SearchSelect.
    const campField = screen.getByText("Select Existing Campaign");
    const campTrigger = campField.parentElement?.querySelector("button") as HTMLButtonElement;
    await act(async () => { fireEvent.click(campTrigger); });
    await act(async () => { fireEvent.click(screen.getByText(/AAPL/)); });

    await selectBuyRule(FIRST_RULE);
    fillByLabel("Shares to Add", "20");
    fillByLabel("Price ($)", "150.00");

    await act(async () => {
      fireEvent.click(screen.getByText("LOG BUY ORDER"));
    });

    await waitFor(() => expect(api.logBuy).toHaveBeenCalled());
    // Scale-in path refreshes trades data the same way…
    await waitFor(() => expect(api.tradesOpen).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(api.tradesOpenDetails).toHaveBeenCalledTimes(2));
    // …but does NOT fetch a fresh trade_id (no new trade_id is consumed —
    // the lot lands on the existing campaign).
    expect(vi.mocked(api.nextTradeId).mock.calls.length).toBe(nextTradeIdInitialCount);
  });
});


// ─────────────────────────────────────────────────────────────────────
// Background upload tracker.
// After a successful submit, image uploads run in the background via
// uploadWithTimeout. Per-file status appears in <UploadTracker>; failed
// uploads expose a Retry button. The submit chain no longer awaits
// uploads, so a stalled R2 / Vision call can't hang the "Saving…" state.
// ─────────────────────────────────────────────────────────────────────
describe("LogBuy — background upload tracker", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaults();
    mRally.mockResolvedValue({ prefix: "", state: "POWERTREND" } as any);
    vi.mocked(api.priceLookup).mockReturnValue(new Promise(() => {}) as any);
    vi.mocked(api.logBuy).mockResolvedValue({ trx_id: "B1", trade_id: "202604-001" } as any);
  });

  async function attachFileToEntryCharts(file: File) {
    // FileInput renders a hidden <input type="file" multiple>; Entry Charts
    // is the first file input on the page (followed by Position Changes,
    // then the MarketSurge single-file input). Drive the onChange directly.
    const fileInputs = document.querySelectorAll('input[type="file"]') as NodeListOf<HTMLInputElement>;
    expect(fileInputs.length).toBeGreaterThan(0);
    await act(async () => {
      fireEvent.change(fileInputs[0], { target: { files: [file] } });
    });
  }

  async function fillRequiredAndSubmit() {
    fillByLabel("Ticker Symbol", STOCK_TICKER);
    await selectBuyRule(FIRST_RULE);
    fillByLabel("Shares to Add", "10");
    fillByLabel("Price ($)", "150.00");
    await act(async () => {
      fireEvent.click(screen.getByText("LOG BUY ORDER"));
    });
  }

  test("submit with no files: tracker does not render", async () => {
    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");

    await fillRequiredAndSubmit();

    await waitFor(() => expect(api.logBuy).toHaveBeenCalled());
    expect(mUpload).not.toHaveBeenCalled();
    expect(screen.queryByTestId("upload-tracker")).not.toBeInTheDocument();
  });

  test("submit with attached entry chart: tracker renders, uploadWithTimeout fires", async () => {
    // Keep the upload promise pending so we can assert the "Uploading…"
    // state without a race against resolution.
    let resolveUpload: (v: { ok: boolean }) => void = () => {};
    mUpload.mockReturnValue(new Promise(r => { resolveUpload = r; }));

    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");

    const file = new File(["entry-bytes"], "entry-chart.png", { type: "image/png" });
    await attachFileToEntryCharts(file);

    await fillRequiredAndSubmit();
    await waitFor(() => expect(api.logBuy).toHaveBeenCalled());

    // Tracker appeared with the file in "Uploading…" state.
    const tracker = await screen.findByTestId("upload-tracker");
    expect(tracker).toBeInTheDocument();
    expect(tracker.textContent).toContain("entry-chart.png");
    expect(tracker.textContent).toContain("Uploading");

    // uploadWithTimeout was called once with the file + "entry" kind.
    expect(mUpload).toHaveBeenCalledTimes(1);
    const [calledFile, , , , kind] = mUpload.mock.calls[0];
    expect(calledFile).toBe(file);
    expect(kind).toBe("entry");

    // Now resolve and confirm the entry transitions to "Uploaded".
    await act(async () => { resolveUpload({ ok: true }); await Promise.resolve(); });
    await waitFor(() => {
      expect(screen.getByTestId("upload-tracker").textContent).toContain("Uploaded");
    });
  });

  test("failed upload shows Retry button; clicking it re-fires uploadWithTimeout", async () => {
    mUpload.mockResolvedValueOnce({ ok: false, error: "Upload timed out (60s)" });

    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");

    const file = new File(["entry-bytes"], "entry-chart.png", { type: "image/png" });
    await attachFileToEntryCharts(file);

    await fillRequiredAndSubmit();

    // Wait for the failure to propagate to the tracker.
    const tracker = await screen.findByTestId("upload-tracker");
    await waitFor(() => expect(tracker.textContent).toContain("Failed"));
    expect(tracker.textContent).toContain("Upload timed out (60s)");

    // Find the retry button (data-testid is keyed off the entry's id —
    // pick whichever match shows up).
    const retryButton = tracker.querySelector('[data-testid^="upload-retry-"]') as HTMLButtonElement;
    expect(retryButton).toBeTruthy();

    // Second call will resolve OK.
    mUpload.mockResolvedValueOnce({ ok: true });
    await act(async () => { fireEvent.click(retryButton); });

    await waitFor(() => expect(mUpload).toHaveBeenCalledTimes(2));
    await waitFor(() => {
      expect(screen.getByTestId("upload-tracker").textContent).toContain("Uploaded");
    });
  });

  test("submit chain does NOT await uploads (Saving stops even if upload stalls)", async () => {
    // Never-resolving upload — proves handleSubmit doesn't await it.
    mUpload.mockReturnValue(new Promise(() => {}));

    render(<LogBuy navColor="#6366f1" />);
    await screen.findByTestId("logbuy-sizing-mode-indicator");

    const file = new File(["entry-bytes"], "stalled.png", { type: "image/png" });
    await attachFileToEntryCharts(file);

    await fillRequiredAndSubmit();
    await waitFor(() => expect(api.logBuy).toHaveBeenCalled());

    // The post-submit refetch chain still runs to completion, which is
    // the proxy for "submitting flipped back to false". If handleSubmit
    // were still awaiting the upload, nextTradeId would never be called.
    await waitFor(() => expect(api.nextTradeId).toHaveBeenCalledTimes(2));

    // Submit button is no longer disabled — user can start the next entry.
    const submitBtn = screen.getByText(/LOG BUY ORDER/) as HTMLButtonElement;
    expect(submitBtn.disabled).toBe(false);
  });
});
