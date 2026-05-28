import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import { fireEvent, render, screen, waitFor, within, act } from "@testing-library/react";

vi.mock("@/lib/portfolio-context", () => ({
  usePortfolio: vi.fn(),
}));

const pushMock = vi.fn();
const backMock = vi.fn();
const replaceMock = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: pushMock, back: backMock, replace: replaceMock }),
}));

vi.mock("@/lib/api", () => ({
  api: {
    journalHistory: vi.fn(),
    listEodSnapshots: vi.fn(),
    listDailyJournalCaptures: vi.fn(),
    listTagAssignments: vi.fn(),
    tradesRecent: vi.fn(),
    tradesClosed: vi.fn(),
    listTags: vi.fn(),
    uploadDailyJournalCapture: vi.fn(),
    deleteDailyJournalCapture: vi.fn(),
    createTagAssignment: vi.fn(),
    journalEdit: vi.fn(),
  },
  getActivePortfolio: vi.fn(() => "CanSlim"),
}));

vi.mock("@/lib/log", () => ({
  log: {
    error: vi.fn(),
    warn: vi.fn(),
    info: vi.fn(),
    debug: Object.assign(vi.fn(), { devOnly: vi.fn() }),
  },
}));

// Stub the textarea editor so the daily-report tests stay focused on
// consumer wiring (Edit pill → sheet → Save → API) without dragging
// ReactMarkdown + DOMPurify through every test. The stub renders a
// test-only button that fires onChange + onDirtyChange the same way
// the real editor would on user edits.
vi.mock("./mobile-textarea-editor", () => ({
  __esModule: true,
  MobileTextareaEditor: ({
    initialValue,
    onChange,
    onDirtyChange,
  }: {
    initialValue: string;
    onChange: (html: string) => void;
    onDirtyChange?: (dirty: boolean) => void;
  }) => (
    <div data-testid="mobile-rich-text-editor-body">
      <span data-testid="rte-stub-initial">{initialValue}</span>
      <button
        type="button"
        data-testid="rte-stub-fire-change"
        onClick={() => {
          onChange("<p>Updated recap</p>");
          onDirtyChange?.(true);
        }}
      >
        fire change
      </button>
    </div>
  ),
}));

const setFocusModeActiveMock = vi.fn();
vi.mock("@/lib/format", async () => {
  const actual = await vi.importActual<typeof import("@/lib/format")>("@/lib/format");
  return {
    ...actual,
    setFocusModeActive: (...args: Parameters<typeof actual.setFocusModeActive>) => {
      setFocusModeActiveMock(...args);
      return actual.setFocusModeActive(...args);
    },
  };
});

import { api } from "@/lib/api";
import { usePortfolio } from "@/lib/portfolio-context";
import { MobileDailyReport } from "./mobile-daily-report";
import type {
  DailyJournalCaptureRow,
  JournalHistoryPoint,
  Portfolio,
  Tag,
  TagAssignment,
  TradeDetail,
  TradePosition,
} from "@/lib/api";

const CANSLIM: Portfolio = {
  id: 1,
  name: "CanSlim",
  starting_capital: 100000,
  reset_date: null,
  created_at: "2026-01-01T00:00:00Z",
  cash_balance: 0,
};

function setPortfolio(p: Portfolio | null = CANSLIM) {
  vi.mocked(usePortfolio).mockReturnValue({
    portfolios: p ? [p] : [],
    activePortfolio: p,
    loading: false,
    error: null,
    refetch: vi.fn(),
    setActive: vi.fn(),
  });
}

function journalFixture(opts: Partial<JournalHistoryPoint> & { day: string }): JournalHistoryPoint {
  return {
    id: 100,
    end_nlv: 500_000,
    daily_pct_change: 0,
    portfolio_ltd: 0,
    spy_ltd: 0,
    ndx_ltd: 0,
    pct_invested: 50,
    portfolio_heat: 0,
    daily_thoughts: "",
    ...opts,
  } as JournalHistoryPoint;
}

function captureFixture(opts: Partial<DailyJournalCaptureRow> & { id: number }): DailyJournalCaptureRow {
  return {
    id: opts.id,
    daily_journal_id: opts.daily_journal_id ?? 100,
    storage_ref: opts.storage_ref ?? `r2/key-${opts.id}.png`,
    view_url: opts.view_url ?? `https://cdn.example.com/cap-${opts.id}.png`,
    file_name: opts.file_name ?? `capture-${opts.id}.png`,
    mime_type: opts.mime_type ?? "image/png",
    file_size_bytes: opts.file_size_bytes ?? 1024,
    width: opts.width ?? 800,
    height: opts.height ?? 600,
    sort_order: opts.sort_order ?? 0,
    caption: opts.caption ?? "",
    created_at: opts.created_at ?? "2026-05-25T10:00:00Z",
  };
}

// Node 22's built-in localStorage shadows jsdom's and doesn't
// implement removeItem/clear properly. Stub per test for reliability.
// Mirrors more.test.tsx + mobile-daily-routine.test.tsx pattern.
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

beforeEach(() => {
  setPortfolio();
  vi.clearAllMocks();
  stubLocalStorage();
  vi.mocked(api.journalHistory).mockResolvedValue([]);
  vi.mocked(api.listEodSnapshots).mockResolvedValue([]);
  vi.mocked(api.listDailyJournalCaptures).mockResolvedValue([]);
  vi.mocked(api.listTagAssignments).mockResolvedValue([]);
  vi.mocked(api.tradesRecent).mockResolvedValue({ details: [], lot_closures: [] });
  vi.mocked(api.tradesClosed).mockResolvedValue([]);
  vi.mocked(api.listTags).mockResolvedValue([]);
  document.body.classList.remove("privacy");
});

afterEach(() => {
  try {
    document.body.classList.remove("privacy");
  } catch {
    // best effort
  }
});

// ── Mount fetch ────────────────────────────────────────────────────

describe("MobileDailyReport — mount fetch", () => {
  test("fires all parallel fetches with active portfolio + date", async () => {
    render(<MobileDailyReport initialDate="2026-05-25" />);
    await waitFor(() => {
      expect(api.journalHistory).toHaveBeenCalledWith("CanSlim", 0);
      expect(api.listEodSnapshots).toHaveBeenCalledWith("2026-05-25", "CanSlim");
      expect(api.tradesRecent).toHaveBeenCalledWith("CanSlim", 500);
      expect(api.tradesClosed).toHaveBeenCalledWith("CanSlim", 500);
      expect(api.listTags).toHaveBeenCalledWith("CanSlim");
    });
  });

  test("fetches captures + tag assignments only when journalRow.id exists", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: 42 }),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    await waitFor(() => {
      expect(api.listDailyJournalCaptures).toHaveBeenCalledWith(42, "CanSlim");
      expect(api.listTagAssignments).toHaveBeenCalledWith({
        entity_type: "daily_journal",
        entity_id: 42,
      });
    });
  });
});

// ── Disabled state ─────────────────────────────────────────────────

describe("MobileDailyReport — no journal entry", () => {
  test("redirects to /daily-journal after fetch resolves with no matching row", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-04-01", id: 1 }),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    await waitFor(() =>
      expect(replaceMock).toHaveBeenCalledWith("/daily-journal"),
    );
  });

  test("disabled-state pill button routes to /daily-journal unconditionally", async () => {
    // Defensive-fallback path: while the redirect effect fires, the
    // no-entry banner still renders briefly. Clicking its pill must
    // honor the label promise even if the redirect somehow stalled.
    vi.mocked(api.journalHistory).mockResolvedValue([]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    const back = await screen.findByTestId("no-entry-back-button");
    fireEvent.click(back);
    expect(pushMock).toHaveBeenCalledWith("/daily-journal");
    expect(backMock).not.toHaveBeenCalled();
  });
});

describe("MobileDailyReport — invalid date redirect", () => {
  test("undefined initialDate fires router.replace('/daily-journal')", async () => {
    render(<MobileDailyReport />);
    await waitFor(() =>
      expect(replaceMock).toHaveBeenCalledWith("/daily-journal"),
    );
  });

  test("malformed initialDate fires router.replace('/daily-journal')", async () => {
    render(<MobileDailyReport initialDate="not-a-date" />);
    await waitFor(() =>
      expect(replaceMock).toHaveBeenCalledWith("/daily-journal"),
    );
  });
});

// ── Header + back nav ──────────────────────────────────────────────

describe("MobileDailyReport — header back nav", () => {
  test("back chevron routes to /daily-journal unconditionally", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: 1 }),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    await screen.findByTestId("drawdown-tile");
    const back = screen.getByTestId("report-back-button");
    fireEvent.click(back);
    expect(pushMock).toHaveBeenCalledWith("/daily-journal");
    expect(backMock).not.toHaveBeenCalled();
  });
});

// ── Drawdown calc ──────────────────────────────────────────────────

describe("MobileDailyReport — drawdown", () => {
  test("computes (curr - peak) / peak * 100 with upTo-selected-date semantic", async () => {
    // Peak at 500k on 2026-05-20; current at 475k on 2026-05-25 → -5%.
    // A LATER day at 600k should NOT affect peak (upTo filter).
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-20", id: 1, end_nlv: 500_000 }),
      journalFixture({ day: "2026-05-25", id: 2, end_nlv: 475_000 }),
      journalFixture({ day: "2026-05-30", id: 3, end_nlv: 600_000 }), // future — must be ignored
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    const pct = await screen.findByTestId("drawdown-pct");
    expect(pct).toHaveTextContent(/-5\.00%/);
  });

  test.each([
    [-5, "GREEN LIGHT", "var(--m-accent)"],
    [-10, "CAUTION", "var(--m-warn)"],
    [-13, "MAX 30% INVESTED", "var(--m-down)"],
    [-20, "GO TO CASH", "var(--m-down)"],
  ])("drawdown %s%% → %s tier", async (pct, message) => {
    // Peak 100, current = peak * (1 + pct/100).
    const peak = 500_000;
    const curr = peak * (1 + pct / 100);
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-20", id: 1, end_nlv: peak }),
      journalFixture({ day: "2026-05-25", id: 2, end_nlv: curr }),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    const msg = await screen.findByTestId("drawdown-message");
    expect(msg).toHaveTextContent(message);
  });
});

// ── Daily $ delta ──────────────────────────────────────────────────

describe("MobileDailyReport — daily $ delta in metrics tile", () => {
  test("computes end_nlv - prevDayRow.end_nlv", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-24", id: 1, end_nlv: 487_704 }),
      journalFixture({ day: "2026-05-25", id: 2, end_nlv: 494_776, daily_pct_change: 1.45 }),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    // Daily $ = 494,776 - 487,704 = $7,072 (positive). The dollar string
    // is unique to the DAILY tile sub-line.
    await waitFor(() => {
      expect(screen.getByText(/\+\$7,072/)).toBeInTheDocument();
    });
  });

  test("omits dollar sub-line when no prevDayRow", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: 1, end_nlv: 100_000, daily_pct_change: 0 }),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    // Wait for the drawdown tile to render (indicates the loaded state).
    await screen.findByTestId("drawdown-tile");
    // No signed dollar delta line should appear under DAILY. NLV renders
    // as plain $100,000 (no sign); the daily-dollar sub-line carries the
    // explicit +/− glyph and would match this pattern.
    expect(screen.queryByText(/[+−]\$[\d,]+/)).toBeNull();
  });
});

// ── Focus mode ─────────────────────────────────────────────────────

describe("MobileDailyReport — focus mode hydration", () => {
  // The per-page focus-mode toggle UI was removed in T2-4 follow-up.
  // The app's global driver lives elsewhere; this page only honors
  // masking via the lib/format singleton + .privacy-mask class.

  test("hydrates from localStorage on mount and calls setFocusModeActive", async () => {
    window.localStorage.setItem("mo-focus-mode", "on");
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: 1 }),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    await screen.findByTestId("drawdown-tile");
    expect(setFocusModeActiveMock).toHaveBeenCalledWith(true);
    expect(document.body.classList.contains("privacy")).toBe(true);
  });

  test("no toggle button is rendered in the header", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: 1 }),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    await screen.findByTestId("drawdown-tile");
    expect(screen.queryByTestId("focus-mode-toggle")).not.toBeInTheDocument();
  });

  test("$ elements carry privacy-mask className", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: 1, end_nlv: 100_000 }),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    await screen.findByTestId("drawdown-tile");
    expect(document.querySelectorAll(".privacy-mask").length).toBeGreaterThan(0);
  });
});

// ── Daily Recap preview + edit pill (T2-4b) ───────────────────────

describe("MobileDailyReport — Daily Recap preview", () => {
  test("renders markdown preview when expanded and lowlights present", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({
        day: "2026-05-25",
        id: 1,
        lowlights: "**Bold** recap with [link](https://x.example)",
      } as never),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    // Section is collapsed by default — expand it first.
    fireEvent.click(await screen.findByTestId("recap-section-toggle"));
    const preview = await screen.findByTestId("recap-preview");
    expect(preview.querySelector("strong")?.textContent).toBe("Bold");
  });

  test("renders HTML output (from MobileRichTextEditor) via rehypeRaw", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({
        day: "2026-05-25",
        id: 1,
        lowlights: "<h2>Heading</h2><p>HTML <strong>recap</strong></p>",
      } as never),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    fireEvent.click(await screen.findByTestId("recap-section-toggle"));
    const preview = await screen.findByTestId("recap-preview");
    expect(preview.querySelector("h2")?.textContent).toBe("Heading");
    expect(preview.querySelector("strong")?.textContent).toBe("recap");
  });

  test("empty state when expanded and lowlights empty", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: 1, lowlights: "" } as never),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    fireEvent.click(await screen.findByTestId("recap-section-toggle"));
    await screen.findByTestId("recap-empty");
    expect(screen.queryByTestId("recap-preview")).not.toBeInTheDocument();
  });

  test("section is collapsed by default — preview hidden until toggle", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({
        day: "2026-05-25",
        id: 1,
        lowlights: "**Bold** recap",
      } as never),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    await screen.findByTestId("recap-section-toggle");
    expect(screen.queryByTestId("recap-preview")).not.toBeInTheDocument();
    expect(screen.queryByTestId("recap-empty")).not.toBeInTheDocument();
  });

  test("Edit pill is reachable from collapsed state without expanding", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: 1 } as never),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    await screen.findByTestId("recap-edit-pill");
    fireEvent.click(screen.getByTestId("recap-edit-pill"));
    // Sheet opens; section remains collapsed.
    await screen.findByTestId("mobile-edit-sheet");
    expect(screen.queryByTestId("recap-preview")).not.toBeInTheDocument();
  });

  test("toggle button rotates chevron via aria-expanded", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: 1, lowlights: "x" } as never),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    const toggle = await screen.findByTestId("recap-section-toggle");
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
  });
});

// ── Daily Thoughts preview + edit pill (T2-4b) ────────────────────

describe("MobileDailyReport — Daily Thoughts preview", () => {
  test("renders HTML preview when expanded", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({
        day: "2026-05-25",
        id: 1,
        daily_thoughts: "<p>Hello <strong>world</strong></p>",
      }),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    fireEvent.click(await screen.findByTestId("thoughts-section-toggle"));
    const preview = await screen.findByTestId("thoughts-preview");
    expect(preview.querySelector("strong")?.textContent).toBe("world");
  });

  test("renders class-based text color (.text-color-emerald)", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({
        day: "2026-05-25",
        id: 1,
        daily_thoughts:
          '<p>Mixed <span class="text-color-emerald">colored</span> text</p>',
      }),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    fireEvent.click(await screen.findByTestId("thoughts-section-toggle"));
    const preview = await screen.findByTestId("thoughts-preview");
    expect(preview.querySelector("span.text-color-emerald")).not.toBeNull();
  });

  test("empty state when expanded and no daily_thoughts", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: 1, daily_thoughts: "" }),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    fireEvent.click(await screen.findByTestId("thoughts-section-toggle"));
    await screen.findByTestId("thoughts-empty");
    expect(screen.queryByTestId("thoughts-preview")).not.toBeInTheDocument();
  });

  test("section is collapsed by default", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: 1, daily_thoughts: "<p>x</p>" }),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    await screen.findByTestId("thoughts-section-toggle");
    expect(screen.queryByTestId("thoughts-preview")).not.toBeInTheDocument();
  });

  test("no 'Edit in T2-4b →' indigo placeholder", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: 1, daily_thoughts: "<p>x</p>" }),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    const section = await screen.findByTestId("thoughts-section");
    expect(within(section).queryByText(/T2-4b/)).not.toBeInTheDocument();
  });

  test("Edit pill opens the sheet from collapsed state", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: 1, daily_thoughts: "<p>x</p>" }),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    const pill = await screen.findByTestId("thoughts-edit-pill");
    fireEvent.click(pill);
    await screen.findByTestId("mobile-edit-sheet");
    expect(screen.getByTestId("mobile-edit-sheet-title")).toHaveTextContent(
      "Daily Thoughts",
    );
    // Preview stays collapsed even after opening the sheet.
    expect(screen.queryByTestId("thoughts-preview")).not.toBeInTheDocument();
  });
});

// ── Edit-sheet save flow (Recap) ──────────────────────────────────

describe("MobileDailyReport — Recap edit sheet save flow", () => {
  test("opens sheet on Edit pill tap, seeded with lowlights value", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({
        day: "2026-05-25",
        id: 1,
        lowlights: "Seeded value",
      } as never),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    const pill = await screen.findByTestId("recap-edit-pill");
    fireEvent.click(pill);
    await screen.findByTestId("mobile-edit-sheet");
    expect(screen.getByTestId("mobile-edit-sheet-title")).toHaveTextContent(
      "Daily Recap",
    );
    // Editor body content is set imperatively (innerHTML); assert presence.
    expect(screen.getByTestId("mobile-rich-text-editor-body")).toBeInTheDocument();
  });

  test("Save fires journalEdit with { portfolio, day, lowlights } when dirty", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: 1, lowlights: "x" } as never),
    ]);
    vi.mocked(api.journalEdit).mockResolvedValue({ status: "ok", id: 1 });
    render(<MobileDailyReport initialDate="2026-05-25" />);
    fireEvent.click(await screen.findByTestId("recap-edit-pill"));
    await screen.findByTestId("mobile-edit-sheet");
    // Use the editor stub's test-only button to fire onChange with new
    // HTML — Lexical's controlled DOM means we can't manipulate the
    // editor body directly in jsdom; the consumer wiring is what we're
    // testing here, not the editor itself.
    fireEvent.click(await screen.findByTestId("rte-stub-fire-change"));
    const save = screen.getByTestId("mobile-edit-sheet-save") as HTMLButtonElement;
    await waitFor(() => expect(save.disabled).toBe(false));
    fireEvent.click(save);
    await waitFor(() =>
      expect(api.journalEdit).toHaveBeenCalledWith({
        portfolio: "CanSlim",
        day: "2026-05-25",
        lowlights: expect.stringContaining("Updated recap"),
      }),
    );
  });
});

// ── Performance Comparison ─────────────────────────────────────────

describe("MobileDailyReport — Performance Comparison", () => {
  test("renders 3 rows (Portfolio / SPY / NASDAQ), each with Daily + YTD", async () => {
    // Two rows in 2026 to enable YTD chain.
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({
        day: "2026-01-02",
        id: 1,
        end_nlv: 100_000,
        daily_pct_change: 0,
        spy: 500,
        nasdaq: 17000,
      } as never),
      journalFixture({
        day: "2026-05-25",
        id: 2,
        end_nlv: 110_000,
        daily_pct_change: 1.0,
        spy: 525,
        nasdaq: 17850,
      } as never),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    await screen.findByTestId("perf-row-portfolio");
    expect(screen.getByTestId("perf-row-spy")).toBeInTheDocument();
    expect(screen.getByTestId("perf-row-nasdaq")).toBeInTheDocument();
    // SPY YTD = (525 - 500) / 500 * 100 = 5%
    expect(screen.getByTestId("perf-spy-ytd")).toHaveTextContent(/\+5\.00%/);
    // NASDAQ YTD = (17850 - 17000) / 17000 * 100 = 5%
    expect(screen.getByTestId("perf-nasdaq-ytd")).toHaveTextContent(/\+5\.00%/);
  });
});

// ── Positions Opened / Closed ──────────────────────────────────────

describe("MobileDailyReport — Positions Opened", () => {
  test("filters tradesRecent.details by date AND action=BUY", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: 1 }),
    ]);
    const opened: TradeDetail = {
      trade_id: "202605-001",
      ticker: "NVDA",
      action: "BUY",
      date: "2026-05-25",
      shares: 100,
      amount: 478.2,
      value: 47820,
      rule: "Cup breakout",
    };
    const otherDay: TradeDetail = { ...opened, trade_id: "202605-002", date: "2026-05-20" };
    const sellSameDay: TradeDetail = { ...opened, trade_id: "202605-003", action: "SELL" };
    vi.mocked(api.tradesRecent).mockResolvedValue({
      details: [opened, otherDay, sellSameDay],
      lot_closures: [],
    });
    render(<MobileDailyReport initialDate="2026-05-25" />);
    await screen.findByTestId("opened-row-202605-001");
    expect(screen.queryByTestId("opened-row-202605-002")).not.toBeInTheDocument();
    expect(screen.queryByTestId("opened-row-202605-003")).not.toBeInTheDocument();
  });

  test("empty state when no positions opened on the date", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: 1 }),
    ]);
    vi.mocked(api.tradesRecent).mockResolvedValue({ details: [], lot_closures: [] });
    render(<MobileDailyReport initialDate="2026-05-25" />);
    await screen.findByTestId("opened-section");
    // Section starts collapsed when count is 0; expand it.
    const sectionHeader = within(screen.getByTestId("opened-section")).getByRole("button");
    fireEvent.click(sectionHeader);
    await screen.findByTestId("opened-empty");
  });
});

describe("MobileDailyReport — Positions Closed", () => {
  test("filters tradesClosed by closed_date", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: 1 }),
    ]);
    const closed: TradePosition = {
      trade_id: "202605-007",
      ticker: "PLTR",
      status: "closed",
      shares: 100,
      avg_entry: 30,
      total_cost: 3000,
      realized_pl: 4210,
      rule: "Hit target",
      closed_date: "2026-05-25",
      return_pct: 18.3,
    } as TradePosition;
    const otherDay = { ...closed, trade_id: "202605-008", closed_date: "2026-05-20" };
    vi.mocked(api.tradesClosed).mockResolvedValue([closed, otherDay] as TradePosition[]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    await screen.findByTestId("closed-row-202605-007");
    expect(screen.queryByTestId("closed-row-202605-008")).not.toBeInTheDocument();
  });
});

// ── Captures + EOD merged gallery ──────────────────────────────────

describe("MobileDailyReport — Captures + EOD gallery", () => {
  test("merges chronologically newest-first across both sources", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: 100 }),
    ]);
    vi.mocked(api.listDailyJournalCaptures).mockResolvedValue([
      captureFixture({ id: 1, created_at: "2026-05-25T08:00:00Z" }),
    ]);
    vi.mocked(api.listEodSnapshots).mockResolvedValue([
      {
        id: 50,
        view_url: "https://cdn.example.com/eod-50.png",
        image_type: "eod_dashboard",
        uploaded_at: "2026-05-25T16:00:00Z",
        file_name: "dashboard.png",
      },
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    await screen.findByTestId("captures-section");
    // EOD has later timestamp → appears first.
    await waitFor(() => {
      const strip = screen.getByTestId("image-upload-strip");
      const thumbs = within(strip).getAllByRole("button", { name: /View/i });
      // First thumb is EOD (uploaded later).
      expect(thumbs[0]).toHaveAttribute("aria-label", expect.stringContaining("dashboard.png"));
      expect(thumbs[1]).toHaveAttribute("aria-label", expect.stringContaining("capture-1.png"));
    });
  });

  test("EOD items show 'EOD' badge and have no remove (X) button", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: 100 }),
    ]);
    vi.mocked(api.listEodSnapshots).mockResolvedValue([
      {
        id: 50,
        view_url: "https://cdn.example.com/eod-50.png",
        image_type: "eod_dashboard",
        uploaded_at: "2026-05-25T16:00:00Z",
        file_name: "dashboard.png",
      },
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    await screen.findByTestId("captures-section");
    // EOD id is namespaced to -1000 - 50 = -1050.
    await screen.findByTestId("image-upload-badge--1050");
    expect(screen.queryByTestId("image-upload-remove--1050")).not.toBeInTheDocument();
  });

  test("capture items have remove (X) button", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: 100 }),
    ]);
    vi.mocked(api.listDailyJournalCaptures).mockResolvedValue([
      captureFixture({ id: 7 }),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    await screen.findByTestId("image-upload-remove-7");
    // Captures do not get a badge.
    expect(screen.queryByTestId("image-upload-badge-7")).not.toBeInTheDocument();
  });

  test("disabled state when journalRow.id is null", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: null }),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    await screen.findByTestId("image-upload-disabled");
  });
});

// ── Tags ───────────────────────────────────────────────────────────

describe("MobileDailyReport — Tags", () => {
  test("renders assigned tag chips with stored color", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: 100 }),
    ]);
    const assigns: TagAssignment[] = [
      {
        id: 1,
        tag_id: 10,
        tag_name: "breakout",
        tag_color: "emerald",
        entity_type: "daily_journal",
        entity_id: 100,
        created_at: "2026-05-25T10:00:00Z",
      },
    ];
    vi.mocked(api.listTagAssignments).mockResolvedValue(assigns);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    await screen.findByTestId("tag-chip-1");
    expect(screen.getByTestId("tag-chip-1")).toHaveTextContent("breakout");
  });

  test("Add button opens bottom sheet listing available tags (assigned excluded)", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: 100 }),
    ]);
    const tags: Tag[] = [
      {
        id: 10,
        portfolio: "CanSlim",
        name: "breakout",
        color: "emerald",
        created_at: "",
        updated_at: "",
      },
      {
        id: 11,
        portfolio: "CanSlim",
        name: "conviction",
        color: "violet",
        created_at: "",
        updated_at: "",
      },
    ];
    vi.mocked(api.listTags).mockResolvedValue(tags);
    vi.mocked(api.listTagAssignments).mockResolvedValue([
      {
        id: 1,
        tag_id: 10,
        tag_name: "breakout",
        tag_color: "emerald",
        entity_type: "daily_journal",
        entity_id: 100,
        created_at: "2026-05-25T10:00:00Z",
      },
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    const addBtn = await screen.findByTestId("tag-add-button");
    fireEvent.click(addBtn);
    const sheet = await screen.findByTestId("tag-add-sheet");
    // breakout already assigned → excluded; conviction visible.
    expect(within(sheet).queryByTestId("tag-option-10")).not.toBeInTheDocument();
    expect(within(sheet).getByTestId("tag-option-11")).toBeInTheDocument();
  });

  test("Add disabled when journalId null", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: null }),
    ]);
    vi.mocked(api.listTags).mockResolvedValue([
      {
        id: 10,
        portfolio: "CanSlim",
        name: "breakout",
        color: "emerald",
        created_at: "",
        updated_at: "",
      },
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    const addBtn = (await screen.findByTestId("tag-add-button")) as HTMLButtonElement;
    expect(addBtn.disabled).toBe(true);
  });

  test("Add disabled when at 10/10 cap", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: 100 }),
    ]);
    const assigns: TagAssignment[] = Array.from({ length: 10 }, (_, i) => ({
      id: i + 1,
      tag_id: i + 100,
      tag_name: `tag-${i}`,
      tag_color: "emerald",
      entity_type: "daily_journal" as const,
      entity_id: 100,
      created_at: "2026-05-25T10:00:00Z",
    }));
    vi.mocked(api.listTagAssignments).mockResolvedValue(assigns);
    vi.mocked(api.listTags).mockResolvedValue(
      Array.from({ length: 12 }, (_, i) => ({
        id: i + 100,
        portfolio: "CanSlim",
        name: `tag-${i}`,
        color: "emerald",
        created_at: "",
        updated_at: "",
      })),
    );
    render(<MobileDailyReport initialDate="2026-05-25" />);
    // Wait for the assignments fetch to populate so the cap predicate
    // evaluates after data has landed; `findByTestId` resolves before
    // the secondary fetch chain completes.
    await waitFor(() => {
      const addBtn = screen.getByTestId("tag-add-button") as HTMLButtonElement;
      expect(addBtn.disabled).toBe(true);
    });
  });
});

// ── Grade tile ─────────────────────────────────────────────────────

describe("MobileDailyReport — grade tile", () => {
  test.each([
    [5, "A+"],
    [4, "A"],
    [3, "B"],
    [2, "C"],
    [1, "D"],
  ])("score=%s → grade %s", async (score, label) => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: 1, score } as never),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    const grade = await screen.findByTestId("metric-grade-value");
    expect(grade).toHaveTextContent(label);
  });
});

// ── DAILY P&L label (T2-4b rename) ────────────────────────────────

describe("MobileDailyReport — DAILY P&L label", () => {
  test("metrics tile uses 'DAILY P&L' label (T2-4 was 'DAILY')", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: 1 } as never),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    await screen.findByTestId("drawdown-tile");
    expect(screen.getByText("DAILY P&L")).toBeInTheDocument();
    expect(screen.queryByText(/^DAILY$/)).not.toBeInTheDocument();
  });
});

// ── Market Notes header + edit (T2-4b new consumer) ───────────────

describe("MobileDailyReport — Market Notes header line", () => {
  test("renders market_notes text in header subtitle area when present", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({
        day: "2026-05-25",
        id: 1,
        market_notes: "QQQ at 21EMA, strong open",
      } as never),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    const text = await screen.findByTestId("header-market-notes-text");
    expect(text).toHaveTextContent("QQQ at 21EMA, strong open");
  });

  test("renders 'Add market notes' prompt when market_notes empty", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: 1, market_notes: "" } as never),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    const header = await screen.findByTestId("header-market-notes");
    expect(header).toHaveTextContent(/Add market notes/);
    expect(screen.queryByTestId("header-market-notes-text")).not.toBeInTheDocument();
  });

  test("pencil button opens Market Notes edit sheet with textarea (no rich editor)", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({
        day: "2026-05-25",
        id: 1,
        market_notes: "Seeded",
      } as never),
    ]);
    render(<MobileDailyReport initialDate="2026-05-25" />);
    const pencil = await screen.findByTestId("header-market-notes-edit");
    fireEvent.click(pencil);
    await screen.findByTestId("mobile-edit-sheet");
    expect(screen.getByTestId("mobile-edit-sheet-title")).toHaveTextContent(
      "Market Notes",
    );
    const textarea = screen.getByTestId("market-notes-textarea") as HTMLTextAreaElement;
    expect(textarea.value).toBe("Seeded");
    // Confirm NO rich-text editor in this sheet.
    expect(screen.queryByTestId("mobile-rich-text-editor")).not.toBeInTheDocument();
  });

  test("Save fires journalEdit with { portfolio, day, market_notes }", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      journalFixture({ day: "2026-05-25", id: 1, market_notes: "old" } as never),
    ]);
    vi.mocked(api.journalEdit).mockResolvedValue({ status: "ok", id: 1 });
    render(<MobileDailyReport initialDate="2026-05-25" />);
    fireEvent.click(await screen.findByTestId("header-market-notes-edit"));
    const textarea = await screen.findByTestId("market-notes-textarea");
    fireEvent.change(textarea, { target: { value: "new notes" } });
    const save = screen.getByTestId("mobile-edit-sheet-save");
    fireEvent.click(save);
    await waitFor(() =>
      expect(api.journalEdit).toHaveBeenCalledWith({
        portfolio: "CanSlim",
        day: "2026-05-25",
        market_notes: "new notes",
      }),
    );
  });
});
