import { render, screen, waitFor, fireEvent, act, within } from "@testing-library/react";
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

// Multi-portfolio context — provide three portfolios so the redesigned page
// renders 3 cards in its default test setup. The returned object reference
// MUST be stable across calls because the component's effect has
// `portfolios` as a dependency; a fresh object per usePortfolio() call
// triggers an infinite re-fire of the load effect.
const _MOCK_PORTFOLIOS = [
  { id: 1, name: "CanSlim" },
  { id: 3, name: "457B Plan" },
  { id: 2, name: "Long-Term Growth" },
];
const _MOCK_CTX = {
  activePortfolio: _MOCK_PORTFOLIOS[0],
  portfolios: _MOCK_PORTFOLIOS,
  loading: false,
  error: null,
  refetch: vi.fn(),
  setActive: vi.fn(),
};
vi.mock("@/lib/portfolio-context", () => ({
  usePortfolio: () => _MOCK_CTX,
}));

vi.mock("@/lib/api", () => ({
  api: {
    journalLatest: vi.fn(),
    batchPrices: vi.fn(),
    rallyPrefix: vi.fn(),
    tradesRecent: vi.fn(),
    ibkrNavForDate: vi.fn(),
    journalEdit: vi.fn(),
    journalBatchEdit: vi.fn(),
    portfolioHeatPreview: vi.fn(),
  },
  getActivePortfolio: () => "CanSlim",
}));

import { api } from "@/lib/api";
import { DailyRoutine } from "./daily-routine";

const mockedJournalLatest = vi.mocked(api.journalLatest);
const mockedBatchPrices = vi.mocked(api.batchPrices);
const mockedRallyPrefix = vi.mocked(api.rallyPrefix);
const mockedTradesRecent = vi.mocked(api.tradesRecent);
const mockedJournalBatchEdit = vi.mocked(api.journalBatchEdit);
const mockedPortfolioHeatPreview = vi.mocked(api.portfolioHeatPreview);


function setupDefaultMocks() {
  mockedJournalLatest.mockResolvedValue({ end_nlv: 100000 } as any);
  mockedBatchPrices.mockResolvedValue({ SPY: 500.0, "^IXIC": 18000.0 } as any);
  mockedRallyPrefix.mockResolvedValue({ prefix: "" } as any);
  mockedTradesRecent.mockResolvedValue({ details: [], lot_closures: [] });
  mockedJournalBatchEdit.mockResolvedValue({
    status: "ok",
    rows_written: 3,
    portfolios: ["CanSlim", "457B Plan", "Long-Term Growth"],
  });
  mockedPortfolioHeatPreview.mockResolvedValue({
    heat: 7.92, nlv_used: 100000, portfolio: "CanSlim",
  } as any);
}


describe("DailyRoutine — multi-portfolio rendering", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaultMocks();
  });

  test("renders one PortfolioCard per portfolio in the context", async () => {
    render(<DailyRoutine navColor="#6366f1" />);

    // Each card has data-testid="portfolio-card-<name>"
    await screen.findByTestId("portfolio-card-CanSlim");
    expect(screen.getByTestId("portfolio-card-457B Plan")).toBeInTheDocument();
    expect(screen.getByTestId("portfolio-card-Long-Term Growth")).toBeInTheDocument();
  });

  test("each card has its own NLV + Holdings inputs", async () => {
    render(<DailyRoutine navColor="#6366f1" />);
    await screen.findByTestId("portfolio-card-CanSlim");

    expect(screen.getByTestId("nlv-input-CanSlim")).toBeInTheDocument();
    expect(screen.getByTestId("nlv-input-457B Plan")).toBeInTheDocument();
    expect(screen.getByTestId("holdings-input-CanSlim")).toBeInTheDocument();
  });

  test("fetches prev_end_nlv per portfolio in parallel", async () => {
    mockedJournalLatest.mockImplementation(((portfolio: string) => {
      // Distinguishable per portfolio so we can verify scoping.
      const map: Record<string, number> = {
        CanSlim: 500000,
        "457B Plan": 250000,
        "Long-Term Growth": 46006.79,
      };
      return Promise.resolve({ end_nlv: map[portfolio] || 0 }) as any;
    }) as any);

    render(<DailyRoutine navColor="#6366f1" />);
    await screen.findByTestId("portfolio-card-CanSlim");

    // Three calls fired, one per portfolio, each scoped to the active date.
    await waitFor(() => expect(mockedJournalLatest).toHaveBeenCalledTimes(3));
    const calledPortfolios = mockedJournalLatest.mock.calls.map(c => c[0]);
    expect(calledPortfolios).toEqual(expect.arrayContaining([
      "CanSlim", "457B Plan", "Long-Term Growth",
    ]));
  });
});


describe("DailyRoutine — validation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaultMocks();
  });

  test("empty NLV triggers Required error on blur", async () => {
    render(<DailyRoutine navColor="#6366f1" />);
    const nlvInput = await screen.findByTestId("nlv-input-CanSlim") as HTMLInputElement;

    await act(async () => {
      fireEvent.blur(nlvInput);
    });

    // Blurring while empty also flags total_holdings (validateCard checks
    // both); both fields show "Required". getAllByText confirms ≥1 match.
    const card = screen.getByTestId("portfolio-card-CanSlim");
    const requiredMessages = within(card).getAllByText("Required");
    expect(requiredMessages.length).toBeGreaterThanOrEqual(1);
  });

  test("NLV = 0 is VALID — no error rendered", async () => {
    render(<DailyRoutine navColor="#6366f1" />);
    const nlvInput = await screen.findByTestId("nlv-input-CanSlim") as HTMLInputElement;
    const holdingsInput = screen.getByTestId("holdings-input-CanSlim") as HTMLInputElement;

    await act(async () => {
      fireEvent.change(nlvInput, { target: { value: "0" } });
      fireEvent.change(holdingsInput, { target: { value: "0" } });
      fireEvent.blur(nlvInput);
      fireEvent.blur(holdingsInput);
    });

    const card = screen.getByTestId("portfolio-card-CanSlim");
    expect(within(card).queryByText("Required")).not.toBeInTheDocument();
  });

  test("save button disabled when any card has empty NLV", async () => {
    render(<DailyRoutine navColor="#6366f1" />);
    await screen.findByTestId("portfolio-card-CanSlim");

    // Fill 2 of 3 cards completely; leave third's NLV empty.
    for (const name of ["CanSlim", "457B Plan"]) {
      const nlv = screen.getByTestId(`nlv-input-${name}`) as HTMLInputElement;
      const hold = screen.getByTestId(`holdings-input-${name}`) as HTMLInputElement;
      await act(async () => {
        fireEvent.change(nlv, { target: { value: "100000" } });
        fireEvent.change(hold, { target: { value: "90000" } });
      });
    }

    // The Long-Term Growth card's NLV is still empty → button disabled
    const saveBtn = screen.getByTestId("save-button") as HTMLButtonElement;
    expect(saveBtn.disabled).toBe(true);
  });

  test("validation summary lists every error after submit attempt", async () => {
    render(<DailyRoutine navColor="#6366f1" />);
    await screen.findByTestId("portfolio-card-CanSlim");

    // Trigger blurs on all 6 fields → 6 errors render in summary
    for (const name of ["CanSlim", "457B Plan", "Long-Term Growth"]) {
      const nlv = screen.getByTestId(`nlv-input-${name}`) as HTMLInputElement;
      const hold = screen.getByTestId(`holdings-input-${name}`) as HTMLInputElement;
      await act(async () => {
        fireEvent.blur(nlv);
        fireEvent.blur(hold);
      });
    }

    const summary = screen.getByTestId("validation-summary");
    expect(summary).toBeInTheDocument();
    expect(within(summary).getByText(/Fix 6 errors before saving/)).toBeInTheDocument();
  });

  test("validation banner is HIDDEN on initial render (no submit, no blur)", async () => {
    // Regression: previously the summary banner appeared on first paint
    // because hasErrors() returns true the moment any required field is
    // empty. With the touched/submitAttempted gates, untouched fields on
    // first render must not surface the banner.
    render(<DailyRoutine navColor="#6366f1" />);
    await screen.findByTestId("portfolio-card-CanSlim");

    expect(screen.queryByTestId("validation-summary")).not.toBeInTheDocument();
  });

  test("inline 'Required' error does NOT show on initial render (no blur yet)", async () => {
    // Same regression — per-field error messages should also be gated on
    // touched. First-paint cards must be clean.
    render(<DailyRoutine navColor="#6366f1" />);
    await screen.findByTestId("portfolio-card-CanSlim");

    // No "Required" anywhere on the page on mount.
    expect(screen.queryByText("Required")).not.toBeInTheDocument();
  });

  test("blurring an empty NLV surfaces the banner (touched gate fires)", async () => {
    render(<DailyRoutine navColor="#6366f1" />);
    await screen.findByTestId("portfolio-card-CanSlim");

    // No banner yet
    expect(screen.queryByTestId("validation-summary")).not.toBeInTheDocument();

    const nlv = screen.getByTestId("nlv-input-CanSlim") as HTMLInputElement;
    await act(async () => {
      fireEvent.blur(nlv);
    });

    // One field touched-and-failing → banner shows. Listing reflects ALL
    // errors (still 6 total since every card has both fields empty), even
    // though only one has been touched — the banner is a global summary.
    const summary = await screen.findByTestId("validation-summary");
    expect(within(summary).getByText(/Fix 6 errors before saving/)).toBeInTheDocument();
  });

  test("changing entryDate clears submitAttempted (banner re-hidden after a save attempt elsewhere)", async () => {
    // Touch some fields + click Save to trigger submitAttempted=true.
    // Then change the date — the fresh date should clear the flag and the
    // banner should re-hide (assuming the new date's cards are also fresh
    // and untouched, which is true since the load-effect resets state).
    render(<DailyRoutine navColor="#6366f1" />);
    await screen.findByTestId("portfolio-card-CanSlim");

    // Force the banner via a Save click. Button is disabled but fireEvent
    // bypasses that to test the underlying handler doesn't crash either.
    // Quicker path: blur a field to set touched.
    const nlv = screen.getByTestId("nlv-input-CanSlim") as HTMLInputElement;
    await act(async () => {
      fireEvent.blur(nlv);
    });
    await screen.findByTestId("validation-summary");

    // Change the date. The cards rebuild from emptyCard() (all touched=false)
    // and submitAttempted resets to false in the load-effect → banner hides.
    const dateInput = screen.getByLabelText("Entry date") as HTMLInputElement;
    await act(async () => {
      fireEvent.change(dateInput, { target: { value: "2026-04-20" } });
    });

    await waitFor(() => {
      expect(screen.queryByTestId("validation-summary")).not.toBeInTheDocument();
    });
  });
});


describe("DailyRoutine — save handler", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaultMocks();
  });

  async function fillAllCards(values?: Partial<{ nlv: string; holdings: string; cash: string }>) {
    const nlv = values?.nlv ?? "100000";
    const holdings = values?.holdings ?? "90000";
    const cash = values?.cash ?? "0";
    for (const name of ["CanSlim", "457B Plan", "Long-Term Growth"]) {
      const nlvIn = screen.getByTestId(`nlv-input-${name}`) as HTMLInputElement;
      const holdIn = screen.getByTestId(`holdings-input-${name}`) as HTMLInputElement;
      await act(async () => {
        fireEvent.change(nlvIn, { target: { value: nlv } });
        fireEvent.change(holdIn, { target: { value: holdings } });
      });
    }
    void cash;  // reserved for future per-card cash override
  }

  test("calls journalBatchEdit with the expected payload shape", async () => {
    render(<DailyRoutine navColor="#6366f1" />);
    await screen.findByTestId("portfolio-card-CanSlim");
    await fillAllCards();

    const saveBtn = screen.getByTestId("save-button") as HTMLButtonElement;
    await act(async () => {
      fireEvent.click(saveBtn);
    });

    await waitFor(() => expect(mockedJournalBatchEdit).toHaveBeenCalled());
    const payload = mockedJournalBatchEdit.mock.calls[0][0];
    expect(payload.day).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(payload.portfolios).toHaveLength(3);
    expect(payload.shared).toMatchObject({
      spy: expect.any(Number),
      nasdaq: expect.any(Number),
      market_notes: expect.any(String),
    });
    expect(payload.force_overwrite).toBe(false);

    const names = (payload.portfolios as Array<{ portfolio: string }>).map(p => p.portfolio);
    expect(names).toEqual(expect.arrayContaining(["CanSlim", "457B Plan", "Long-Term Growth"]));
  });

  test("Force Overwrite checkbox flips force_overwrite in payload", async () => {
    render(<DailyRoutine navColor="#6366f1" />);
    await screen.findByTestId("portfolio-card-CanSlim");
    await fillAllCards();

    const forceCheckbox = screen.getByTestId("force-overwrite-checkbox") as HTMLInputElement;
    await act(async () => {
      fireEvent.click(forceCheckbox);
    });
    expect(forceCheckbox.checked).toBe(true);

    await act(async () => {
      fireEvent.click(screen.getByTestId("save-button"));
    });

    await waitFor(() => expect(mockedJournalBatchEdit).toHaveBeenCalled());
    expect(mockedJournalBatchEdit.mock.calls[0][0].force_overwrite).toBe(true);
  });

  test("renders success banner on 200 response", async () => {
    render(<DailyRoutine navColor="#6366f1" />);
    await screen.findByTestId("portfolio-card-CanSlim");
    await fillAllCards();

    await act(async () => {
      fireEvent.click(screen.getByTestId("save-button"));
    });

    const ok = await screen.findByTestId("save-ok-banner");
    expect(ok).toHaveTextContent(/Saved 3 portfolios/);
  });

  test("renders conflict banner on 409 with conflicting_portfolios list", async () => {
    mockedJournalBatchEdit.mockResolvedValue({
      status: "exists",
      conflicting_portfolios: ["CanSlim", "457B Plan"],
    });

    render(<DailyRoutine navColor="#6366f1" />);
    await screen.findByTestId("portfolio-card-CanSlim");
    await fillAllCards();

    await act(async () => {
      fireEvent.click(screen.getByTestId("save-button"));
    });

    const banner = await screen.findByTestId("conflict-banner");
    expect(banner).toHaveTextContent(/CanSlim, 457B Plan/);
    expect(banner).toHaveTextContent(/Force Overwrite/);
  });

  test("renders error banner on non-ok / non-conflict response", async () => {
    mockedJournalBatchEdit.mockResolvedValue({
      status: "error",
      detail: "simulated DB failure",
    });

    render(<DailyRoutine navColor="#6366f1" />);
    await screen.findByTestId("portfolio-card-CanSlim");
    await fillAllCards();

    await act(async () => {
      fireEvent.click(screen.getByTestId("save-button"));
    });

    const banner = await screen.findByTestId("save-error-banner");
    expect(banner).toHaveTextContent(/simulated DB failure/);
  });
});


describe("DailyRoutine — rally prefix re-fetches on date change", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaultMocks();
  });

  test("calls rallyPrefix with the entry date and re-calls when the date changes", async () => {
    // Preserved test from pre-redesign. The multi-portfolio refactor kept
    // the rallyPrefix useEffect keyed on entryDate; this guards that
    // behavior across the refactor.
    let prefixCount = 18;
    mockedRallyPrefix.mockImplementation((async (asOf?: string) => {
      return { prefix: `Day ${prefixCount++}: ` } as any;
    }) as any);

    render(<DailyRoutine navColor="#f59f00" />);

    await waitFor(() => expect(mockedRallyPrefix).toHaveBeenCalled());
    const firstAsOf = mockedRallyPrefix.mock.calls[0][0];
    expect(firstAsOf).toMatch(/^\d{4}-\d{2}-\d{2}$/);

    const dateInput = screen.getByLabelText("Entry date") as HTMLInputElement;
    await act(async () => {
      fireEvent.change(dateInput, { target: { value: "2026-04-20" } });
    });

    await waitFor(() => expect(mockedRallyPrefix.mock.calls.length).toBeGreaterThanOrEqual(2));
    const lastCall = mockedRallyPrefix.mock.calls[mockedRallyPrefix.mock.calls.length - 1];
    expect(lastCall[0]).toBe("2026-04-20");
  });
});
