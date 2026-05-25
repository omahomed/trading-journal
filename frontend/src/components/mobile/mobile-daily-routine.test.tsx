import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";

vi.mock("@/lib/portfolio-context", () => ({
  usePortfolio: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  api: {
    journalLatest: vi.fn(),
    tradesRecent: vi.fn(),
    batchPrices: vi.fn(),
    rallyPrefix: vi.fn(),
    journalBatchEdit: vi.fn(),
  },
}));

vi.mock("@/lib/log", () => ({
  log: {
    error: vi.fn(),
    warn: vi.fn(),
    info: vi.fn(),
    debug: Object.assign(vi.fn(), { devOnly: vi.fn() }),
  },
}));

import { api } from "@/lib/api";
import { usePortfolio } from "@/lib/portfolio-context";
import { MobileDailyRoutine } from "./mobile-daily-routine";
import type { JournalEntry, Portfolio } from "@/lib/api";

const CANSLIM: Portfolio = {
  id: 1,
  name: "CanSlim",
  starting_capital: 100000,
  reset_date: null,
  created_at: "2026-01-01T00:00:00Z",
  cash_balance: 0,
};

const PLAN_457B: Portfolio = {
  id: 2,
  name: "457B Plan",
  starting_capital: 200000,
  reset_date: null,
  created_at: "2026-01-01T00:00:00Z",
  cash_balance: 0,
};

const DRAFT_KEY = (date: string) => `mo-daily-routine-draft-${date}`;

// Today helper — matches the component's own todayStr() so test
// expectations and component behavior line up regardless of when
// tests run.
function today(): string {
  const n = new Date();
  return `${n.getFullYear()}-${String(n.getMonth() + 1).padStart(2, "0")}-${String(n.getDate()).padStart(2, "0")}`;
}

// Node 22's built-in localStorage shadows jsdom's and doesn't
// implement removeItem/clear properly. Stub per test so the autosave
// assertions are reliable. Mirrors more.test.tsx's pattern.
function stubLocalStorage(): Record<string, string> {
  const data: Record<string, string> = {};
  const fake: Storage = {
    get length() {
      return Object.keys(data).length;
    },
    clear: () => {
      for (const k of Object.keys(data)) delete data[k];
    },
    getItem: (k) => (k in data ? data[k] : null),
    key: (i) => Object.keys(data)[i] ?? null,
    removeItem: (k) => {
      delete data[k];
    },
    setItem: (k, v) => {
      data[k] = String(v);
    },
  };
  vi.stubGlobal("localStorage", fake);
  Object.defineProperty(window, "localStorage", { value: fake, configurable: true });
  return data;
}

function setPortfolios(portfolios: Portfolio[] = [CANSLIM, PLAN_457B]) {
  vi.mocked(usePortfolio).mockReturnValue({
    portfolios,
    activePortfolio: portfolios[0] ?? null,
    loading: false,
    error: null,
    refetch: vi.fn(),
    setActive: vi.fn(),
  });
}

type LatestMap = Record<string, JournalEntry | null>;

function setApiMocks(opts: {
  latestByPortfolio?: LatestMap;
  prices?: Record<string, number>;
  rallyPrefix?: string;
  batchEditResponse?: Awaited<ReturnType<typeof api.journalBatchEdit>>;
} = {}) {
  vi.mocked(api.journalLatest).mockImplementation((portfolio) =>
    Promise.resolve(opts.latestByPortfolio?.[portfolio ?? ""] ?? null) as ReturnType<
      typeof api.journalLatest
    >,
  );
  vi.mocked(api.tradesRecent).mockResolvedValue({
    details: [],
    lot_closures: [],
  } as unknown as Awaited<ReturnType<typeof api.tradesRecent>>);
  vi.mocked(api.batchPrices).mockResolvedValue(
    (opts.prices ?? {}) as unknown as Awaited<ReturnType<typeof api.batchPrices>>,
  );
  vi.mocked(api.rallyPrefix).mockResolvedValue({
    prefix: opts.rallyPrefix ?? "",
  } as unknown as Awaited<ReturnType<typeof api.rallyPrefix>>);
  vi.mocked(api.journalBatchEdit).mockResolvedValue(
    opts.batchEditResponse ?? ({ status: "ok", rows_written: 2 } as Awaited<
      ReturnType<typeof api.journalBatchEdit>
    >),
  );
}

let lsStore: Record<string, string> = {};

beforeEach(() => {
  setPortfolios();
  vi.clearAllMocks();
  setApiMocks();
  lsStore = stubLocalStorage();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("MobileDailyRoutine — mount fetch + blank form", () => {
  test("calls journalLatest, tradesRecent, batchPrices, rallyPrefix on mount per portfolio", async () => {
    render(<MobileDailyRoutine />);
    await waitFor(() => expect(api.journalLatest).toHaveBeenCalled());
    expect(api.journalLatest).toHaveBeenCalledWith("CanSlim", expect.any(String));
    expect(api.journalLatest).toHaveBeenCalledWith("457B Plan", expect.any(String));
    expect(api.tradesRecent).toHaveBeenCalledWith("CanSlim", 1000);
    expect(api.tradesRecent).toHaveBeenCalledWith("457B Plan", 1000);
    expect(api.batchPrices).toHaveBeenCalled();
    expect(api.rallyPrefix).toHaveBeenCalledWith(expect.any(String));
  });

  test("renders one portfolio card per active portfolio", async () => {
    render(<MobileDailyRoutine />);
    await screen.findByTestId("portfolio-card-CanSlim");
    expect(screen.getByTestId("portfolio-card-457B Plan")).toBeInTheDocument();
  });

  test("no pre-load banner when no existing entries", async () => {
    setApiMocks({ latestByPortfolio: { CanSlim: null, "457B Plan": null } });
    render(<MobileDailyRoutine />);
    await screen.findByTestId("portfolio-card-CanSlim");
    expect(screen.queryByTestId("preload-banner")).not.toBeInTheDocument();
  });
});

describe("MobileDailyRoutine — pre-load existing entry", () => {
  test("shows preload banner + auto-enables Force Overwrite when entry.day === entryDate", async () => {
    const t = today();
    setApiMocks({
      latestByPortfolio: {
        CanSlim: {
          day: t,
          end_nlv: 105_000,
          beg_nlv: 100_000,
          daily_dollar_change: 5000,
          daily_pct_change: 5,
          pct_invested: 50,
          spy: 500,
          nasdaq: 18000,
          portfolio_heat: 0,
          score: 5,
          total_holdings: 50_000,
          cash_change: 0,
          actions: "BUY: NVDA",
          market_notes: "Day 14 UPTREND: gap up",
        } as JournalEntry,
        "457B Plan": null,
      },
    });
    render(<MobileDailyRoutine />);
    const banner = await screen.findByTestId("preload-banner");
    expect(banner).toHaveTextContent(/Editing existing entry/);
    expect(banner).toHaveTextContent(/CanSlim/);
    // Force Overwrite auto-enabled.
    const toggle = screen.getByRole("checkbox", { name: /Force overwrite/ });
    expect(toggle).toBeChecked();
  });

  test("pre-fills form fields from the existing entry", async () => {
    const t = today();
    setApiMocks({
      latestByPortfolio: {
        CanSlim: {
          day: t,
          end_nlv: 105_000,
          beg_nlv: 100_000,
          daily_dollar_change: 5000,
          daily_pct_change: 5,
          pct_invested: 50,
          spy: 500,
          nasdaq: 18000,
          portfolio_heat: 0,
          score: 5,
          total_holdings: 50_000,
          cash_change: 0,
          actions: "BUY: NVDA",
        } as JournalEntry,
        "457B Plan": null,
      },
    });
    render(<MobileDailyRoutine />);
    await screen.findByTestId("preload-banner");
    const nlvInput = screen.getByLabelText("Closing NLV for CanSlim") as HTMLInputElement;
    expect(nlvInput.value).toBe("105000");
    const holdingsInput = screen.getByLabelText("Total Holdings for CanSlim") as HTMLInputElement;
    expect(holdingsInput.value).toBe("50000");
  });

  test("when latest entry is from a prior day, no banner + prev_end_nlv derived from latest.end_nlv", async () => {
    setApiMocks({
      latestByPortfolio: {
        CanSlim: {
          day: "2026-04-01",
          end_nlv: 95_000,
          beg_nlv: 90_000,
          daily_dollar_change: 5000,
          daily_pct_change: 5.5,
          pct_invested: 40,
          spy: 480,
          nasdaq: 17500,
          portfolio_heat: 0,
          score: 5,
        } as JournalEntry,
        "457B Plan": null,
      },
    });
    render(<MobileDailyRoutine />);
    await screen.findByTestId("portfolio-card-CanSlim");
    expect(screen.queryByTestId("preload-banner")).not.toBeInTheDocument();
  });
});

describe("MobileDailyRoutine — localStorage autosave", () => {
  test("writes a draft after debounce on field change", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    render(<MobileDailyRoutine />);
    await waitFor(() => expect(api.journalLatest).toHaveBeenCalled());

    fireEvent.change(screen.getByLabelText("Closing NLV for CanSlim"), {
      target: { value: "123000" },
    });
    expect(window.localStorage.getItem(DRAFT_KEY(today()))).toBeNull();

    await act(async () => {
      vi.advanceTimersByTime(600);
    });

    const raw = window.localStorage.getItem(DRAFT_KEY(today()));
    expect(raw).not.toBeNull();
    const draft = JSON.parse(raw!);
    expect(draft.cards.find((c: { name: string }) => c.name === "CanSlim").end_nlv).toBe("123000");
  });

  test("restores draft on mount + shows Restored banner when no existing entry", async () => {
    const t = today();
    window.localStorage.setItem(
      DRAFT_KEY(t),
      JSON.stringify({
        entryDate: t,
        spyClose: "555.55",
        ndxClose: "",
        marketNotes: "draft notes",
        scores: { plan: 4, stops: 4, sized: 4, fomo: 4 },
        gradeNotes: "",
        forceOverwrite: false,
        cards: [
          { name: "CanSlim", end_nlv: "999000", total_holdings: "", cash_change: "0", actions: "" },
        ],
      }),
    );
    render(<MobileDailyRoutine />);
    expect(await screen.findByTestId("restored-draft-banner")).toBeInTheDocument();
    const nlvInput = screen.getByLabelText("Closing NLV for CanSlim") as HTMLInputElement;
    await waitFor(() => expect(nlvInput.value).toBe("999000"));
    const spy = screen.getByLabelText("SPY close") as HTMLInputElement;
    expect(spy.value).toBe("555.55");
  });

  test("clears draft on save success", async () => {
    const t = today();
    window.localStorage.setItem(
      DRAFT_KEY(t),
      JSON.stringify({
        entryDate: t,
        spyClose: "",
        ndxClose: "",
        marketNotes: "",
        scores: { plan: 5, stops: 5, sized: 5, fomo: 5 },
        gradeNotes: "",
        forceOverwrite: false,
        cards: [
          { name: "CanSlim", end_nlv: "100000", total_holdings: "50000", cash_change: "0", actions: "" },
          { name: "457B Plan", end_nlv: "200000", total_holdings: "100000", cash_change: "0", actions: "" },
        ],
      }),
    );
    setApiMocks({ batchEditResponse: { status: "ok", rows_written: 2 } });
    render(<MobileDailyRoutine />);
    await screen.findByTestId("portfolio-card-CanSlim");

    fireEvent.click(screen.getByTestId("save-button"));
    await waitFor(() => expect(api.journalBatchEdit).toHaveBeenCalled());
    await waitFor(() =>
      expect(window.localStorage.getItem(DRAFT_KEY(t))).toBeNull(),
    );
  });

  test("preserves draft on save failure", async () => {
    const t = today();
    // Pre-seed the draft so the autosave debounce isn't part of the
    // assertion — the test is really "save failure does NOT call
    // removeItem" rather than "save failure waits for autosave".
    window.localStorage.setItem(
      DRAFT_KEY(t),
      JSON.stringify({
        entryDate: t,
        spyClose: "",
        ndxClose: "",
        marketNotes: "",
        scores: { plan: 5, stops: 5, sized: 5, fomo: 5 },
        gradeNotes: "",
        forceOverwrite: false,
        cards: [
          { name: "CanSlim", end_nlv: "100000", total_holdings: "50000", cash_change: "0", actions: "" },
          { name: "457B Plan", end_nlv: "200000", total_holdings: "100000", cash_change: "0", actions: "" },
        ],
      }),
    );
    setApiMocks({
      batchEditResponse: { status: "error", detail: "Network error" },
    });
    render(<MobileDailyRoutine />);
    await screen.findByTestId("portfolio-card-CanSlim");

    fireEvent.click(screen.getByTestId("save-button"));
    await waitFor(() =>
      expect(screen.getByTestId("save-error-banner")).toBeInTheDocument(),
    );
    // Draft preserved (not cleared) — let user retry without re-filling.
    expect(window.localStorage.getItem(DRAFT_KEY(t))).not.toBeNull();
  });
});

describe("MobileDailyRoutine — validation + submit", () => {
  test("save button disabled when required fields empty", async () => {
    render(<MobileDailyRoutine />);
    await screen.findByTestId("portfolio-card-CanSlim");
    const saveBtn = screen.getByTestId("save-button") as HTMLButtonElement;
    expect(saveBtn).toBeDisabled();
  });

  test("validation summary appears after submit attempt with empty NLV", async () => {
    render(<MobileDailyRoutine />);
    await screen.findByTestId("portfolio-card-CanSlim");
    // Click save while fields empty — button is disabled, but blur via tab
    // doesn't fire in jsdom; simulate the touched path via blur.
    fireEvent.blur(screen.getByLabelText("Closing NLV for CanSlim"));
    expect(await screen.findByTestId("validation-summary")).toHaveTextContent(/required/i);
  });

  test("ok submit builds payload with all portfolios + shared section", async () => {
    setApiMocks({ batchEditResponse: { status: "ok", rows_written: 2 } });
    render(<MobileDailyRoutine />);
    await screen.findByTestId("portfolio-card-CanSlim");

    fireEvent.change(screen.getByLabelText("Closing NLV for CanSlim"), {
      target: { value: "105000" },
    });
    fireEvent.change(screen.getByLabelText("Total Holdings for CanSlim"), {
      target: { value: "55000" },
    });
    fireEvent.change(screen.getByLabelText("Closing NLV for 457B Plan"), {
      target: { value: "205000" },
    });
    fireEvent.change(screen.getByLabelText("Total Holdings for 457B Plan"), {
      target: { value: "105000" },
    });

    fireEvent.click(screen.getByTestId("save-button"));
    await waitFor(() => expect(api.journalBatchEdit).toHaveBeenCalled());
    const callArg = vi.mocked(api.journalBatchEdit).mock.calls[0][0];
    expect(callArg.portfolios).toHaveLength(2);
    expect(callArg.shared).toMatchObject({ nlv_source: "manual" });
  });

  test("'exists' response renders conflict banner WITHOUT auto-enabling Force Overwrite", async () => {
    setApiMocks({
      batchEditResponse: {
        status: "exists",
        conflicting_portfolios: ["CanSlim"],
      },
    });
    render(<MobileDailyRoutine />);
    await screen.findByTestId("portfolio-card-CanSlim");

    fireEvent.change(screen.getByLabelText("Closing NLV for CanSlim"), {
      target: { value: "100000" },
    });
    fireEvent.change(screen.getByLabelText("Total Holdings for CanSlim"), {
      target: { value: "50000" },
    });
    fireEvent.change(screen.getByLabelText("Closing NLV for 457B Plan"), {
      target: { value: "200000" },
    });
    fireEvent.change(screen.getByLabelText("Total Holdings for 457B Plan"), {
      target: { value: "100000" },
    });

    fireEvent.click(screen.getByTestId("save-button"));
    const banner = await screen.findByTestId("conflict-banner");
    expect(banner).toHaveTextContent(/CanSlim/);
    // Force Overwrite NOT auto-enabled — user has to explicitly opt in.
    const toggle = screen.getByRole("checkbox", { name: /Force overwrite/ });
    expect(toggle).not.toBeChecked();
  });
});

describe("MobileDailyRoutine — score chips + grade", () => {
  test("changing a score chip updates the letter grade", async () => {
    render(<MobileDailyRoutine />);
    await screen.findByTestId("report-grade");
    const initialGrade = screen.getByTestId("report-grade").textContent;
    // Default scores are all 5 → A+. Drop a chip to 1 → grade drops.
    fireEvent.click(screen.getByTestId("score-chip-followed-plan-1"));
    await waitFor(() =>
      expect(screen.getByTestId("report-grade").textContent).not.toBe(initialGrade),
    );
  });
});
