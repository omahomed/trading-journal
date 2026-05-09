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
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/",
}));

vi.mock("@/lib/api", () => ({
  api: {
    tradesOpen: vi.fn(),
    tradesClosed: vi.fn(),
    tradesOpenDetails: vi.fn(),
    tradesRecent: vi.fn(),
    journalLatest: vi.fn(),
    batchPrices: vi.fn(),
    editTransaction: vi.fn(),
    deleteTransaction: vi.fn(),
    listStrategies: vi.fn().mockResolvedValue([
      { name: "CanSlim", description: null, color: "#6366f1", is_active: true, created_at: "2026-01-01" },
      { name: "StockTalk", description: null, color: "#d97706", is_active: true, created_at: "2026-01-02" },
    ]),
    setTradeStrategy: vi.fn().mockResolvedValue({ ok: true }),
  },
  getActivePortfolio: () => "CanSlim",
}));

// Stub the InteractiveChart child — it pulls in lightweight-charts which
// jsdom can't render, and it isn't part of the modal flow under test.
vi.mock("./interactive-chart", () => ({
  InteractiveChart: () => null,
}));

import { api } from "@/lib/api";
import { TradeJournal } from "./trade-journal";

// Distinct values per field so getByDisplayValue can identify each input
// without ambiguity.
const TRADE = {
  trade_id: "202604-001", ticker: "AAPL", status: "OPEN",
  open_date: "2026-04-01", closed_date: null,
  shares: 100, avg_entry: 100.0, avg_exit: 0,
  total_cost: 10000, realized_pl: 0, unrealized_pl: 0, return_pct: 0,
  rule: "br1.1 Consolidation",
};

const BUY_DETAIL = {
  detail_id: 101, trade_id: "202604-001", ticker: "AAPL",
  action: "BUY", date: "2026-04-01T09:30:00",
  shares: 100, amount: 99.99, value: 9999,
  rule: "br1.1 Consolidation", trx_id: "B1",
  stop_loss: 75.5, notes: "entry-fingerprint",
};

function setupDefaults() {
  vi.mocked(api.tradesOpen).mockResolvedValue([TRADE] as any);
  vi.mocked(api.tradesOpenDetails).mockResolvedValue({ details: [BUY_DETAIL], lot_closures: [] } as any);
  vi.mocked(api.tradesClosed).mockResolvedValue([]);
  vi.mocked(api.tradesRecent).mockResolvedValue({ details: [], lot_closures: [] } as any);
  vi.mocked(api.journalLatest).mockResolvedValue({ end_nlv: 100000 } as any);
  vi.mocked(api.batchPrices).mockResolvedValue({} as any);
  vi.mocked(api.editTransaction).mockResolvedValue({ status: "ok" } as any);
  vi.mocked(api.deleteTransaction).mockResolvedValue({ status: "ok" } as any);
}

// End-to-end opening the edit modal: render, switch to "open" filter,
// expand the trade card, click Edit on the only row.
async function openEditModal() {
  render(<TradeJournal navColor="#6366f1" />);

  // Filter buttons render as "open (1)" / "all (1)" / "closed (0)" with
  // the count interpolated. Match by capitalized prefix.
  const openFilter = await screen.findByRole("button", { name: /^open \(/i });
  await act(async () => { fireEvent.click(openFilter); });

  // Wait for the trade card to render (the ticker text is a unique signal).
  await screen.findByText("AAPL");

  // Expand the card so Transaction History (and the Edit button) renders.
  const expandBtn = await screen.findByRole("button", { name: /Details/i });
  await act(async () => { fireEvent.click(expandBtn); });

  // Click the Edit button. Only one row in the mock = one Edit button.
  const editBtn = await screen.findByRole("button", { name: /^Edit$/ });
  await act(async () => { fireEvent.click(editBtn); });

  // Modal title — confirms it's open.
  await screen.findByText(/Edit · BUY · AAPL/);
}

describe("TradeJournal — Edit transaction modal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaults();
  });

  test("Edit button opens the modal seeded with the row's values", async () => {
    await openEditModal();

    // Each field shows the BUY's value.
    expect(screen.getByDisplayValue("100")).toBeInTheDocument();        // shares
    expect(screen.getByDisplayValue("99.99")).toBeInTheDocument();      // price
    expect(screen.getByDisplayValue("75.5")).toBeInTheDocument();       // stop_loss
    expect(screen.getByDisplayValue("entry-fingerprint")).toBeInTheDocument();  // notes
    expect(screen.getByDisplayValue("br1.1 Consolidation")).toBeInTheDocument(); // rule select
    expect(screen.getByDisplayValue("B1")).toBeInTheDocument();         // trx_id
  });

  test("backdrop click dismisses the modal", async () => {
    await openEditModal();
    const backdrop = screen.getByTestId("tj-edit-modal-backdrop");
    await act(async () => { fireEvent.click(backdrop); });
    await waitFor(() => {
      expect(screen.queryByText(/Edit · BUY · AAPL/)).not.toBeInTheDocument();
    });
  });

  test("Escape key dismisses the modal", async () => {
    await openEditModal();
    await act(async () => {
      fireEvent.keyDown(window, { key: "Escape" });
    });
    await waitFor(() => {
      expect(screen.queryByText(/Edit · BUY · AAPL/)).not.toBeInTheDocument();
    });
  });

  test("Cancel button dismisses the modal", async () => {
    await openEditModal();
    const cancel = screen.getByRole("button", { name: /^Cancel$/ });
    await act(async () => { fireEvent.click(cancel); });
    await waitFor(() => {
      expect(screen.queryByText(/Edit · BUY · AAPL/)).not.toBeInTheDocument();
    });
  });

  test("Save calls api.editTransaction with the edited payload", async () => {
    await openEditModal();

    // Change shares from 100 to 50.
    const sharesInput = screen.getByDisplayValue("100");
    await act(async () => {
      fireEvent.change(sharesInput, { target: { value: "50" } });
    });

    const save = screen.getByRole("button", { name: /Save Changes/i });
    await act(async () => { fireEvent.click(save); });

    await waitFor(() => {
      expect(vi.mocked(api.editTransaction)).toHaveBeenCalledTimes(1);
    });
    const call = vi.mocked(api.editTransaction).mock.calls[0][0];
    expect(call).toMatchObject({
      detail_id: 101,
      trade_id: "202604-001",
      ticker: "AAPL",
      action: "BUY",
      shares: 50,
      amount: 99.99,
      value: 50 * 99.99,
      rule: "br1.1 Consolidation",
      notes: "entry-fingerprint",
      stop_loss: 75.5,
      trx_id: "B1",
    });
    // Date must round-trip into the payload — exact format varies because
    // jsdom's datetime-local handling isn't deterministic, so we just
    // assert presence and shape.
    expect(call.date).toBeDefined();
    expect(typeof call.date).toBe("string");
  });

  test("Save success refreshes BOTH open and closed cohorts", async () => {
    await openEditModal();

    // Reset call counts so we measure post-save refresh in isolation.
    vi.mocked(api.tradesOpen).mockClear();
    vi.mocked(api.tradesClosed).mockClear();
    vi.mocked(api.tradesOpenDetails).mockClear();
    vi.mocked(api.tradesRecent).mockClear();

    const save = screen.getByRole("button", { name: /Save Changes/i });
    await act(async () => { fireEvent.click(save); });

    // Both cohort loaders fire after save — status may have flipped.
    await waitFor(() => {
      expect(vi.mocked(api.tradesOpen)).toHaveBeenCalled();
      expect(vi.mocked(api.tradesClosed)).toHaveBeenCalled();
    });
  });

  test("Save backend error (e.g. LIFO rejection) displays in-modal without closing", async () => {
    vi.mocked(api.editTransaction).mockResolvedValue({
      error: "This edit would leave 30 sell shares unmatched by buys. Adjust or remove the sells first, or undo the buy deletion.",
    } as any);

    await openEditModal();
    const save = screen.getByRole("button", { name: /Save Changes/i });
    await act(async () => { fireEvent.click(save); });

    await screen.findByText(/sell shares unmatched/i);
    // Modal stays open so the user can adjust + retry.
    expect(screen.getByText(/Edit · BUY · AAPL/)).toBeInTheDocument();
  });

  test("Delete first click arms confirm; api.deleteTransaction NOT yet called", async () => {
    await openEditModal();

    const deleteBtn = screen.getByRole("button", { name: /Delete Transaction/i });
    await act(async () => { fireEvent.click(deleteBtn); });

    // Button text flips to "Confirm Delete" after first click.
    expect(screen.getByRole("button", { name: /Confirm Delete/i })).toBeInTheDocument();
    expect(vi.mocked(api.deleteTransaction)).not.toHaveBeenCalled();
  });

  test("Delete second click calls api.deleteTransaction with row identifiers", async () => {
    await openEditModal();

    // First click → arms confirm.
    const deleteBtn = screen.getByRole("button", { name: /Delete Transaction/i });
    await act(async () => { fireEvent.click(deleteBtn); });

    // Second click → executes.
    const confirmBtn = screen.getByRole("button", { name: /Confirm Delete/i });
    await act(async () => { fireEvent.click(confirmBtn); });

    await waitFor(() => {
      expect(vi.mocked(api.deleteTransaction)).toHaveBeenCalledTimes(1);
    });
    expect(vi.mocked(api.deleteTransaction)).toHaveBeenCalledWith(101, "202604-001", "AAPL");
  });

  test("Delete backend error displays in-modal without closing", async () => {
    vi.mocked(api.deleteTransaction).mockResolvedValue({
      error: "This edit would leave 50 sell shares unmatched by buys.",
    } as any);

    await openEditModal();

    const deleteBtn = screen.getByRole("button", { name: /Delete Transaction/i });
    await act(async () => { fireEvent.click(deleteBtn); });
    const confirmBtn = screen.getByRole("button", { name: /Confirm Delete/i });
    await act(async () => { fireEvent.click(confirmBtn); });

    await screen.findByText(/sell shares unmatched/i);
    expect(screen.getByText(/Edit · BUY · AAPL/)).toBeInTheDocument();
  });

  test("TRX ID input is readOnly", async () => {
    await openEditModal();
    const trxInput = screen.getByDisplayValue("B1") as HTMLInputElement;
    // Direct DOM property check — avoids attribute-name case-sensitivity
    // ambiguity (HTML lowercase vs React camelCase).
    expect(trxInput.readOnly).toBe(true);
  });
});

describe("TradeJournal — URL deep-link", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaults();
  });

  test("?trade_id=XXX in URL pre-expands the matching trade card on mount", async () => {
    // Stage the URL before mount; the component reads window.location.search
    // synchronously inside its useEffect.
    window.history.pushState({}, "", `/trade-journal?trade_id=${TRADE.trade_id}`);

    render(<TradeJournal navColor="#6366f1" />);

    // Trade Journal defaults to "closed" filter; switch to "open" so the
    // mocked OPEN trade is visible. The deep-link only sets expandedCard,
    // not the filter — match the existing test pattern.
    const openFilter = await screen.findByRole("button", { name: /^open \(/i });
    await act(async () => { fireEvent.click(openFilter); });

    // The card pre-expansion is signaled by the Transaction History block,
    // which only renders when expandedCard === trade_id. Look for the
    // BUY transaction's trx_id from the mock — present iff expanded.
    // (The deep-link's pre-expansion is what we're verifying — without it,
    // the user would have to click Details to see B1.)
    await waitFor(() => {
      expect(screen.queryByText("B1")).toBeInTheDocument();
    });

    // Cleanup: restore default URL so it doesn't leak into other tests
    window.history.pushState({}, "", "/");
  });
});


// ─────────────────────────────────────────────────────────────────────
// Phase 2 — Strategy pill on Trade Journal cards.
// Pill renders DB-driven color and continues to render even when the
// strategy has been deactivated since the trade was tagged.
// ─────────────────────────────────────────────────────────────────────

describe("TradeJournal — strategy pill", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaults();
  });

  test("renders the strategy chip with the correct color in the card footer", async () => {
    vi.mocked(api.tradesOpen).mockResolvedValue([
      { ...TRADE, strategy: "StockTalk" } as any,
    ]);

    render(<TradeJournal navColor="#6366f1" />);

    const openFilter = await screen.findByRole("button", { name: /^open \(/i });
    await act(async () => { fireEvent.click(openFilter); });

    // Pill renders with the strategy name.
    await waitFor(() => expect(screen.getByText("StockTalk")).toBeInTheDocument());
    // Swatch picks up the color from the loaded strategies list. The
    // chip's title attribute is the strategy name; its swatch span has
    // the color in inline style. Verify by querying the chip element.
    const chip = screen.getByTitle("StockTalk");
    expect(chip).toBeInTheDocument();
    // The first child span is the swatch (size sm/md/lg → 12 here).
    const swatch = chip.querySelector("span");
    expect(swatch).toHaveStyle({ background: "#d97706" });
  });

  test("pill still renders when the trade's strategy is no longer active", async () => {
    // Per Phase 2 design: a trade tagged when StockTalk was active
    // continues to display the StockTalk chip even after is_active=false.
    // Server returns only active strategies, so this case becomes the
    // "strategy not in loaded list" branch — chip falls back to a grey
    // swatch but still renders the recorded name.
    vi.mocked(api.tradesOpen).mockResolvedValue([
      { ...TRADE, strategy: "Retired" } as any,
    ]);
    // listStrategies only returns the two active ones (no "Retired").

    render(<TradeJournal navColor="#6366f1" />);

    const openFilter = await screen.findByRole("button", { name: /^open \(/i });
    await act(async () => { fireEvent.click(openFilter); });

    // The recorded strategy name is shown even though it's missing from
    // the active list.
    await waitFor(() => expect(screen.getByText("Retired")).toBeInTheDocument());
  });
});
