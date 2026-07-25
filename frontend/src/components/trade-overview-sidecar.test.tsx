import { render, screen, waitFor, within } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach } from "vitest";

// Localstorage shim for jsdom environments that don't ship one.
if (typeof window !== "undefined" && !(window as any).localStorage?.getItem) {
  const _store = new Map<string, string>();
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: {
      getItem: (k: string) => _store.get(k) ?? null,
      setItem: (k: string, v: string) => { _store.set(k, String(v)); },
      removeItem: (k: string) => { _store.delete(k); },
      clear: () => { _store.clear(); },
      key: (i: number) => Array.from(_store.keys())[i] ?? null,
      get length() { return _store.size; },
    },
  });
}

vi.mock("@/lib/api", () => ({
  api: {
    lotExcursions: vi.fn(),
  },
  getActivePortfolio: () => "CanSlim",
}));

import { api } from "@/lib/api";
import { TradeOverviewSidecar } from "./trade-overview-sidecar";

const mLotEx = vi.mocked(api.lotExcursions);


const stockTrade = (overrides: Partial<any> = {}) => ({
  trade_id: "202601-001",
  ticker: "DELL",
  status: "OPEN",
  shares: 100,
  avg_entry: 100,
  total_cost: 10000,
  realized_pl: 0,
  open_date: "2026-01-10",
  rule: "br1.3 Reclaim 50",
  instrument_type: "STOCK",
  multiplier: 1,
  ...overrides,
}) as any;

const buyRow = (trx_id: string, date: string, price: number, shares = 25) => ({
  detail_id: Math.floor(Math.random() * 1e9),
  trade_id: "202601-001",
  ticker: "DELL",
  action: "BUY",
  date,
  shares,
  amount: price,
  value: shares * price,
  rule: "br1.3 Reclaim 50",
  trx_id,
});

const lotExcursion = (overrides: any) => ({
  trade_id: "202601-001",
  portfolio_name: "CanSlim",
  portfolio_id: 1,
  ticker: "DELL",
  status: "OPEN",
  closed_date: null,
  trx_id: "B1",
  fill_date: "2026-01-10",
  fill_price: 100,
  shares: 25,
  window_end_date: "2026-01-25",
  days_held: 15,
  mae_pct: -3.5,
  mfe_pct: 8.0,
  days_to_mae: 3,
  days_to_mfe: 10,
  atr21_at_fill_pct: 2.5,
  mae_atr_multiple: 1.4,
  mfe_atr_multiple: 3.2,
  min_low: 96.5,
  min_low_date: "2026-01-13",
  max_high: 108.0,
  max_high_date: "2026-01-20",
  realized_pl: null,
  error: null,
  ...overrides,
});


describe("TradeOverviewSidecar — per-lot excursion table", () => {
  beforeEach(() => {
    // mockReset (not clearAllMocks) — the latter clears CALL HISTORY
    // but preserves any queued `mockResolvedValueOnce` / `mockRejectedValueOnce`
    // from prior tests, which then leaks into the next mount's useEffect
    // and produces bafflingly wrong state (e.g. test N showing test N-1's
    // error message). Full reset here keeps each test isolated.
    mLotEx.mockReset();
  });

  test("fetches + renders per-lot MAE table with B1 + A1 + A2 rows", async () => {
    mLotEx.mockResolvedValueOnce({
      lots: [
        lotExcursion({ trx_id: "B1", fill_date: "2026-01-10", fill_price: 100, mae_pct: -4.0 }),
        lotExcursion({ trx_id: "A1", fill_date: "2026-01-15", fill_price: 105, mae_pct: -2.5 }),
        lotExcursion({ trx_id: "A2", fill_date: "2026-01-20", fill_price: 110, mae_pct: -1.0 }),
      ],
    });

    render(<TradeOverviewSidecar
      trade={stockTrade()}
      details={[
        buyRow("B1", "2026-01-10", 100),
        buyRow("A1", "2026-01-15", 105),
        buyRow("A2", "2026-01-20", 110),
      ]}
      portfolio="CanSlim"
      onClose={() => {}}
    />);

    await waitFor(() => expect(mLotEx).toHaveBeenCalledWith("202601-001", "CanSlim"));
    // Table appears after the fetch resolves.
    const table = await screen.findByTestId("lot-excursion-table");
    // One row per lot, sorted A1 < A2 < B1 (localeCompare puts A before B).
    expect(within(table).getByTestId("lot-excursion-row-B1")).toBeInTheDocument();
    expect(within(table).getByTestId("lot-excursion-row-A1")).toBeInTheDocument();
    expect(within(table).getByTestId("lot-excursion-row-A2")).toBeInTheDocument();
    // MAE % rendered as signed 2-decimal.
    expect(table.textContent).toContain("-4.00%");
    expect(table.textContent).toContain("-2.50%");
    expect(table.textContent).toContain("-1.00%");
    // MFE with + sign.
    expect(table.textContent).toContain("+8.00%");
  });

  test("worst MAE lot renders in red-emphasized style", async () => {
    // B1 = -4% (worst), A1 = -2.5%, A2 = -1%
    mLotEx.mockResolvedValueOnce({
      lots: [
        lotExcursion({ trx_id: "B1", mae_pct: -4.0 }),
        lotExcursion({ trx_id: "A1", mae_pct: -2.5 }),
        lotExcursion({ trx_id: "A2", mae_pct: -1.0 }),
      ],
    });
    render(<TradeOverviewSidecar
      trade={stockTrade()}
      details={[buyRow("B1", "2026-01-10", 100)]}
      portfolio="CanSlim"
      onClose={() => {}}
    />);

    await screen.findByTestId("lot-excursion-table");
    const b1Row = screen.getByTestId("lot-excursion-row-B1");
    // Worst-MAE cell is bold (fontWeight: 700). Locate the MAE % cell
    // (4th td) and confirm the bold treatment.
    const maeCell = b1Row.querySelectorAll("td")[3] as HTMLElement;
    expect(maeCell.textContent).toContain("-4.00%");
    expect(maeCell.style.fontWeight).toBe("700");

    // A1's MAE cell is red (loss) but NOT bold — only the worst lot
    // gets the emphasis so the eye lands on it.
    const a1Row = screen.getByTestId("lot-excursion-row-A1");
    const a1MaeCell = a1Row.querySelectorAll("td")[3] as HTMLElement;
    expect(a1MaeCell.style.fontWeight).not.toBe("700");
  });

  test("options campaign hides the per-lot section entirely", async () => {
    // Server returns empty (upstream filter drops options); the
    // component's own isOption check also hides the section — belt +
    // suspenders. We assert the section header isn't rendered.
    mLotEx.mockResolvedValueOnce({ lots: [] });

    const optionTrade = stockTrade({
      ticker: "DELL 261016 $100C",
      instrument_type: "OPTION",
      multiplier: 100,
    });
    render(<TradeOverviewSidecar
      trade={optionTrade}
      details={[]}
      portfolio="CanSlim"
      onClose={() => {}}
    />);

    // No fetch was fired (component early-returns on isOption).
    expect(mLotEx).not.toHaveBeenCalled();
    // Regression: the section header shouldn't appear.
    expect(screen.queryByText(/Per-Lot Excursion/i)).not.toBeInTheDocument();
    expect(screen.queryByTestId("lot-excursion-table")).not.toBeInTheDocument();
  });

  test("empty lots response shows a friendly no-data message, not a broken table", async () => {
    mLotEx.mockResolvedValueOnce({ lots: [] });
    render(<TradeOverviewSidecar
      trade={stockTrade()}
      details={[buyRow("B1", "2026-01-10", 100)]}
      portfolio="CanSlim"
      onClose={() => {}}
    />);
    await waitFor(() => expect(mLotEx).toHaveBeenCalled());
    await screen.findByText(/No per-lot excursion data/i);
    expect(screen.queryByTestId("lot-excursion-table")).not.toBeInTheDocument();
  });

  test("API failure surfaces an error line instead of hanging on 'Loading'", async () => {
    mLotEx.mockRejectedValueOnce(new Error("upstream 500"));
    render(<TradeOverviewSidecar
      trade={stockTrade()}
      details={[buyRow("B1", "2026-01-10", 100)]}
      portfolio="CanSlim"
      onClose={() => {}}
    />);
    await screen.findByText(/upstream 500/i);
  });

  test("error rows collapse to a placeholder row (bad lot doesn't hide healthy siblings)", async () => {
    mLotEx.mockResolvedValueOnce({
      lots: [
        lotExcursion({ trx_id: "B1" }),  // healthy
        lotExcursion({ trx_id: "A1", error: "no_bars_in_window", mae_pct: null, mfe_pct: null }),
      ],
    });
    render(<TradeOverviewSidecar
      trade={stockTrade()}
      details={[buyRow("B1", "2026-01-10", 100)]}
      portfolio="CanSlim"
      onClose={() => {}}
    />);
    const table = await screen.findByTestId("lot-excursion-table");
    // Healthy row still there.
    expect(within(table).getByTestId("lot-excursion-row-B1")).toBeInTheDocument();
    // Error row placeholder present + carries the error slug.
    const a1Row = within(table).getByTestId("lot-excursion-row-A1");
    expect(a1Row.textContent).toContain("no_bars_in_window");
  });
});


