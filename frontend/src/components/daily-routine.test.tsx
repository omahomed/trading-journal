import { render, screen, waitFor, fireEvent, act } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach } from "vitest";

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    refresh: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    prefetch: vi.fn(),
  }),
}));

// usePortfolio is read at the top of the component; provide a stable
// CanSlim active portfolio so the same path renders in every test.
vi.mock("@/lib/portfolio-context", () => ({
  usePortfolio: () => ({
    activePortfolio: { id: 1, name: "CanSlim" },
    portfolios: [{ id: 1, name: "CanSlim" }],
    loading: false,
    error: null,
    refetch: vi.fn(),
    setActive: vi.fn(),
  }),
}));

vi.mock("@/lib/api", () => ({
  api: {
    journalLatest: vi.fn(),
    batchPrices: vi.fn(),
    rallyPrefix: vi.fn(),
    tradesRecent: vi.fn(),
    ibkrNavForDate: vi.fn(),
    journalEdit: vi.fn(),
  },
  getActivePortfolio: () => "CanSlim",
}));

import { api } from "@/lib/api";
import { DailyRoutine } from "./daily-routine";

const mockedJournalLatest = vi.mocked(api.journalLatest);
const mockedBatchPrices = vi.mocked(api.batchPrices);
const mockedRallyPrefix = vi.mocked(api.rallyPrefix);
const mockedTradesRecent = vi.mocked(api.tradesRecent);
const mockedIbkrNav = vi.mocked(api.ibkrNavForDate);
const mockedJournalEdit = vi.mocked(api.journalEdit);


function setupDefaultMocks() {
  // Defaults: clean slate for the journal/market/lookups so each test only
  // has to override what it cares about.
  mockedJournalLatest.mockResolvedValue({ end_nlv: 100000 } as any);
  mockedBatchPrices.mockResolvedValue({ SPY: 500.0, "^IXIC": 18000.0 } as any);
  mockedRallyPrefix.mockResolvedValue({ prefix: "" } as any);
  mockedTradesRecent.mockResolvedValue([]);
  mockedJournalEdit.mockResolvedValue({ status: "ok", id: 1 });
}


describe("DailyRoutine — IBKR auto-fill", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaultMocks();
  });

  test("auto-fills the End NLV field on a successful IBKR pull", async () => {
    mockedIbkrNav.mockResolvedValue({
      success: true,
      nav: 487264.5,
      cash_balance: -431003.85,
      position_value: 918268.35,
      report_date: "2026-04-27",
      currency: "USD",
      account: "U1234567",
      source: "ibkr_flex_query",
    });

    render(<DailyRoutine navColor="#f59f00" />);

    // Field gets the auto-filled value once the pull resolves
    const nlvInput = await screen.findByLabelText("Closing NLV");
    await waitFor(() => {
      expect((nlvInput as HTMLInputElement).value).toBe("487264.5");
    });

    // Auto badge appears
    expect(await screen.findByTestId("nlv-auto-badge")).toBeInTheDocument();
    // No warning banner
    expect(screen.queryByTestId("ibkr-warning-banner")).not.toBeInTheDocument();
    // Field is editable (the user can still override)
    expect(nlvInput).not.toBeDisabled();
  });

  test("renders warning banner when IBKR pull fails, leaves field empty", async () => {
    mockedIbkrNav.mockResolvedValue({
      success: false,
      error: "no_data_for_date",
      message: "No NAV data for 2026-04-27 — possibly market not yet closed.",
    });

    render(<DailyRoutine navColor="#f59f00" />);

    const banner = await screen.findByTestId("ibkr-warning-banner");
    expect(banner).toHaveTextContent(/Could not auto-fill NLV from IBKR/);
    expect(banner).toHaveTextContent(/No NAV data for 2026-04-27/);

    const nlvInput = await screen.findByLabelText("Closing NLV") as HTMLInputElement;
    expect(nlvInput.value).toBe("");
    expect(nlvInput).not.toBeDisabled();
    expect(screen.queryByTestId("nlv-auto-badge")).not.toBeInTheDocument();
  });

  test("renders warning banner when ibkrNavForDate throws (network error)", async () => {
    // Belt-and-suspenders: even if the wrapper's catch fires (true network
    // failure rather than a 200-OK-with-success-false), the user must still
    // see a banner and be able to type a value.
    mockedIbkrNav.mockRejectedValue(new Error("Network timeout"));

    render(<DailyRoutine navColor="#f59f00" />);

    const banner = await screen.findByTestId("ibkr-warning-banner");
    expect(banner).toHaveTextContent(/Network timeout/);
  });

  test("flips nlv_source to 'ibkr_override' when the user edits the auto-filled value, and sends it on save", async () => {
    mockedIbkrNav.mockResolvedValue({
      success: true,
      nav: 487264.5,
      cash_balance: -431003.85,
      position_value: 918268.35,
      report_date: "2026-04-27",
      currency: "USD",
      account: "U1234567",
      source: "ibkr_flex_query",
    });

    render(<DailyRoutine navColor="#f59f00" />);

    const nlvInput = await screen.findByLabelText("Closing NLV") as HTMLInputElement;
    await waitFor(() => expect(nlvInput.value).toBe("487264.5"));

    // User types over the auto-filled value
    await act(async () => {
      fireEvent.change(nlvInput, { target: { value: "490000" } });
    });

    // Override badge replaces the auto badge
    expect(await screen.findByTestId("nlv-override-badge")).toBeInTheDocument();
    expect(screen.queryByTestId("nlv-auto-badge")).not.toBeInTheDocument();

    // Submit and verify the save payload tags the row as ibkr_override
    const saveBtn = screen.getByRole("button", { name: /Save Daily Routine/i });
    await act(async () => {
      fireEvent.click(saveBtn);
    });

    await waitFor(() => expect(mockedJournalEdit).toHaveBeenCalled());
    const payload = mockedJournalEdit.mock.calls[0][0];
    expect(payload.nlv_source).toBe("ibkr_override");
    expect(payload.end_nlv).toBe(490000);
  });

  test("save payload tags as 'ibkr_auto' when user accepts the IBKR value unchanged", async () => {
    mockedIbkrNav.mockResolvedValue({
      success: true,
      nav: 487264.5,
      cash_balance: 0,
      position_value: 487264.5,
      report_date: "2026-04-27",
      currency: "USD",
      account: "U1234567",
      source: "ibkr_flex_query",
    });

    render(<DailyRoutine navColor="#f59f00" />);

    const nlvInput = await screen.findByLabelText("Closing NLV") as HTMLInputElement;
    await waitFor(() => expect(nlvInput.value).toBe("487264.5"));

    const saveBtn = screen.getByRole("button", { name: /Save Daily Routine/i });
    await act(async () => {
      fireEvent.click(saveBtn);
    });

    await waitFor(() => expect(mockedJournalEdit).toHaveBeenCalled());
    expect(mockedJournalEdit.mock.calls[0][0].nlv_source).toBe("ibkr_auto");
  });

  test("save payload tags as 'manual' when IBKR pull failed and user typed a value", async () => {
    mockedIbkrNav.mockResolvedValue({
      success: false,
      error: "ibkr_not_configured",
      message: "IBKR NAV puller not configured.",
    });

    render(<DailyRoutine navColor="#f59f00" />);

    const nlvInput = await screen.findByLabelText("Closing NLV") as HTMLInputElement;
    // Wait for the failure banner to confirm the load is settled
    await screen.findByTestId("ibkr-warning-banner");

    await act(async () => {
      fireEvent.change(nlvInput, { target: { value: "100000" } });
    });

    const saveBtn = screen.getByRole("button", { name: /Save Daily Routine/i });
    await act(async () => {
      fireEvent.click(saveBtn);
    });

    await waitFor(() => expect(mockedJournalEdit).toHaveBeenCalled());
    expect(mockedJournalEdit.mock.calls[0][0].nlv_source).toBe("manual");
  });
});
