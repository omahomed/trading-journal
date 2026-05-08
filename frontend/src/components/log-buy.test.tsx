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
    logBuy: vi.fn(),
    listStrategies: vi.fn(),
  },
  getActivePortfolio: () => "CanSlim",
}));

const SEED_STRATEGIES = [
  { name: "CanSlim",     description: "primary",   color: "#6366f1", is_active: true, created_at: "2026-01-01T00:00:00" },
  { name: "StockTalk",   description: "small-cap", color: "#d97706", is_active: true, created_at: "2026-01-01T00:00:01" },
  { name: "21eStrategy", description: "21 EMA",    color: "#0d9488", is_active: true, created_at: "2026-01-01T00:00:02" },
];

import { api } from "@/lib/api";
import { LogBuy } from "./log-buy";

const mRally = vi.mocked(api.rallyPrefix);

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
      expect(indicator.textContent).toMatch(/Offense/);
      expect(indicator.textContent).toMatch(/1\.00% risk/);
      expect(indicator.textContent).toMatch(/from M Factor POWERTREND/);
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
      // No "from M Factor …" label while in manual mode (the source is the user)
      expect(indicator.textContent).not.toMatch(/from M Factor/);
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
    const defenseRadio = await screen.findByText(/Defense \(0\.50%\)/);
    await act(async () => { fireEvent.click(defenseRadio); });
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

  test("CORRECTION → Defense (0.50% risk)", async () => {
    mRally.mockResolvedValue({ prefix: "", state: "CORRECTION" } as any);

    render(<LogBuy navColor="#6366f1" />);

    const indicator = await screen.findByTestId("logbuy-sizing-mode-indicator");
    await waitFor(() => {
      expect(indicator.textContent).toMatch(/Defense/);
      expect(indicator.textContent).toMatch(/0\.50% risk/);
      expect(indicator.textContent).toMatch(/from M Factor CORRECTION/);
    });
  });

  test("rally-prefix returning no state defaults to Normal + 'M Factor state unknown'", async () => {
    mRally.mockResolvedValue({ prefix: "" } as any);

    render(<LogBuy navColor="#6366f1" />);

    const indicator = await screen.findByTestId("logbuy-sizing-mode-indicator");
    await waitFor(() => {
      expect(indicator.textContent).toMatch(/Normal/);
      expect(indicator.textContent).toMatch(/0\.75% risk/);
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
  const ruleField = screen.getByText("Buy Rule *");
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
