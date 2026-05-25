import { describe, test, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";

vi.mock("@/lib/portfolio-context", () => ({
  usePortfolio: vi.fn(),
}));

const pushMock = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: pushMock }),
}));

vi.mock("@/lib/api", () => ({
  api: {
    journalHistory: vi.fn(),
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

import { api } from "@/lib/api";
import { usePortfolio } from "@/lib/portfolio-context";
import { MobileDailyJournal } from "./mobile-daily-journal";
import type { JournalHistoryPoint, Portfolio } from "@/lib/api";

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

function todayStr(): string {
  const n = new Date();
  return `${n.getFullYear()}-${String(n.getMonth() + 1).padStart(2, "0")}-${String(n.getDate()).padStart(2, "0")}`;
}

function daysAgo(n: number): string {
  const d = new Date(Date.now() - n * 86400000);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

function entryFixture(opts: {
  day: string;
  end_nlv?: number;
  daily_pct_change?: number;
  portfolio_ltd?: number;
  pct_invested?: number;
  ndx_daily_pct?: number;
  score?: number;
  market_cycle?: string;
  mct_display_day_num?: number | null;
}): JournalHistoryPoint {
  return {
    id: 1,
    day: opts.day,
    end_nlv: opts.end_nlv ?? 100_000,
    daily_pct_change: opts.daily_pct_change ?? 0,
    portfolio_ltd: opts.portfolio_ltd ?? 0,
    spy_ltd: 0,
    ndx_ltd: 0,
    pct_invested: opts.pct_invested ?? 50,
    portfolio_heat: 0,
    ndx_daily_pct: opts.ndx_daily_pct ?? 0,
    score: opts.score ?? 5,
    market_cycle: opts.market_cycle ?? "",
    mct_display_day_num: opts.mct_display_day_num ?? null,
  } as JournalHistoryPoint;
}

beforeEach(() => {
  setPortfolio();
  vi.clearAllMocks();
  vi.mocked(api.journalHistory).mockResolvedValue([]);
});

describe("MobileDailyJournal — mount fetch + sort", () => {
  test("calls journalHistory with active portfolio + days=0 (full history)", async () => {
    render(<MobileDailyJournal />);
    await waitFor(() =>
      expect(api.journalHistory).toHaveBeenCalledWith("CanSlim", 0),
    );
  });

  test("renders entries newest-first regardless of fetch order", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      entryFixture({ day: daysAgo(2) }), // older arrives first
      entryFixture({ day: todayStr() }),
      entryFixture({ day: daysAgo(5) }),
    ]);
    render(<MobileDailyJournal />);
    // findAllByLabelText avoids matching the sub-row testid (which
    // shares the `day-card-sub-…` prefix). Only the <button> cards
    // carry the "Open daily report for …" aria-label.
    const cards = await screen.findAllByLabelText(/Open daily report for/);
    expect(cards[0]).toHaveAttribute("data-testid", `day-card-${todayStr()}`);
    expect(cards[1]).toHaveAttribute("data-testid", `day-card-${daysAgo(2)}`);
    expect(cards[2]).toHaveAttribute("data-testid", `day-card-${daysAgo(5)}`);
  });
});

describe("MobileDailyJournal — filter chips", () => {
  test("Week chip is default-selected (aria-checked)", async () => {
    render(<MobileDailyJournal />);
    const weekChip = await screen.findByRole("radio", { name: "Week" });
    expect(weekChip).toHaveAttribute("aria-checked", "true");
  });

  test("tapping All chip widens to full history", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      entryFixture({ day: todayStr() }),
      entryFixture({ day: daysAgo(45) }), // outside Week window
    ]);
    render(<MobileDailyJournal />);
    // Default Week → only today's entry visible.
    await screen.findByTestId(`day-card-${todayStr()}`);
    expect(screen.queryByTestId(`day-card-${daysAgo(45)}`)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("radio", { name: "All" }));
    await screen.findByTestId(`day-card-${daysAgo(45)}`);
  });

  test("Week filter excludes entries older than 7 days", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      entryFixture({ day: todayStr() }),
      entryFixture({ day: daysAgo(3) }), // inside Week
      entryFixture({ day: daysAgo(10) }), // outside Week
    ]);
    render(<MobileDailyJournal />);
    await screen.findByTestId(`day-card-${todayStr()}`);
    expect(screen.getByTestId(`day-card-${daysAgo(3)}`)).toBeInTheDocument();
    expect(screen.queryByTestId(`day-card-${daysAgo(10)}`)).not.toBeInTheDocument();
  });
});

describe("MobileDailyJournal — sticky month headers", () => {
  test("renders one month header per distinct month present in the filtered set", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      entryFixture({ day: "2026-05-20" }),
      entryFixture({ day: "2026-05-10" }),
      entryFixture({ day: "2026-04-28" }),
    ]);
    render(<MobileDailyJournal />);
    fireEvent.click(await screen.findByRole("radio", { name: "All" }));
    await screen.findByTestId("month-header-2026-05");
    expect(screen.getByTestId("month-header-2026-04")).toBeInTheDocument();
    // Exactly 2 month headers — not 3.
    expect(screen.getAllByTestId(/^month-header-/)).toHaveLength(2);
  });
});

describe("MobileDailyJournal — card tap navigation", () => {
  test("tapping a card pushes to /daily-report?date=YYYY-MM-DD", async () => {
    const day = todayStr();
    vi.mocked(api.journalHistory).mockResolvedValue([entryFixture({ day })]);
    render(<MobileDailyJournal />);
    const card = await screen.findByTestId(`day-card-${day}`);
    fireEvent.click(card);
    expect(pushMock).toHaveBeenCalledWith(`/daily-report?date=${day}`);
  });
});

describe("MobileDailyJournal — grade pill tier mapping", () => {
  test("A+/A → high tier (green)", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      entryFixture({ day: todayStr(), score: 5 }),
    ]);
    render(<MobileDailyJournal />);
    const pill = await screen.findByTestId("grade-pill-high");
    expect(pill).toHaveTextContent("A+");
  });

  test("B → mid tier (amber)", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      entryFixture({ day: todayStr(), score: 3 }),
    ]);
    render(<MobileDailyJournal />);
    const pill = await screen.findByTestId("grade-pill-mid");
    expect(pill).toHaveTextContent("B");
  });

  test("C/D → low tier (red)", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      entryFixture({ day: todayStr(), score: 2 }),
    ]);
    render(<MobileDailyJournal />);
    const pill = await screen.findByTestId("grade-pill-low");
    expect(pill).toHaveTextContent("C");
  });

  test("score=0 → no grade pill rendered (em-dash placeholder)", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      entryFixture({ day: todayStr(), score: 0 }),
    ]);
    render(<MobileDailyJournal />);
    await screen.findByTestId(`day-card-${todayStr()}`);
    expect(screen.queryByTestId("grade-pill-high")).not.toBeInTheDocument();
    expect(screen.queryByTestId("grade-pill-mid")).not.toBeInTheDocument();
    expect(screen.queryByTestId("grade-pill-low")).not.toBeInTheDocument();
  });
});

describe("MobileDailyJournal — MCT badge", () => {
  test("renders state name + D{N} day suffix when present", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      entryFixture({
        day: todayStr(),
        market_cycle: "POWERTREND",
        mct_display_day_num: 3,
      }),
    ]);
    render(<MobileDailyJournal />);
    const card = await screen.findByTestId(`day-card-${todayStr()}`);
    expect(within(card).getByText(/POWERTREND D3/)).toBeInTheDocument();
  });

  test("hides D{N} suffix when mct_display_day_num is null", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      entryFixture({
        day: todayStr(),
        market_cycle: "CORRECTION",
        mct_display_day_num: null,
      }),
    ]);
    render(<MobileDailyJournal />);
    const card = await screen.findByTestId(`day-card-${todayStr()}`);
    expect(within(card).getByText(/CORRECTION/)).toBeInTheDocument();
    expect(within(card).queryByText(/D\d+/)).not.toBeInTheDocument();
  });
});

describe("MobileDailyJournal — sub-row layout (3 items, no Heat)", () => {
  test("renders LTD · % inv · NDX, no Heat", async () => {
    const day = todayStr();
    vi.mocked(api.journalHistory).mockResolvedValue([
      entryFixture({
        day,
        portfolio_ltd: 12.34,
        pct_invested: 56.7,
        ndx_daily_pct: -1.23,
      }),
    ]);
    render(<MobileDailyJournal />);
    const sub = await screen.findByTestId(`day-card-sub-${day}`);
    expect(within(sub).getByTestId(`sub-ltd-${day}`)).toHaveTextContent(/\+12\.34% LTD/);
    expect(within(sub).getByTestId(`sub-inv-${day}`)).toHaveTextContent("56.7% inv");
    expect(within(sub).getByTestId(`sub-ndx-${day}`)).toHaveTextContent(/-1\.23% NDX/);
    expect(sub.textContent).not.toMatch(/heat/i);
  });

  test("LTD% colored by sign (positive green, negative red), % inv neutral", async () => {
    const day = todayStr();
    vi.mocked(api.journalHistory).mockResolvedValue([
      entryFixture({
        day,
        portfolio_ltd: -5.0,
        ndx_daily_pct: 2.5,
      }),
    ]);
    render(<MobileDailyJournal />);
    await screen.findByTestId(`day-card-${day}`);
    const ltdSpan = screen.getByTestId(`sub-ltd-${day}`).querySelector("span");
    expect(ltdSpan?.className).toMatch(/text-m-down/);
    const ndxSpan = screen.getByTestId(`sub-ndx-${day}`).querySelector("span");
    expect(ndxSpan?.className).toMatch(/text-m-accent/);
  });
});

describe("MobileDailyJournal — empty states", () => {
  test("filter excludes all entries → placeholder + CTA link to /daily-routine", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      entryFixture({ day: daysAgo(45) }), // outside Week
    ]);
    render(<MobileDailyJournal />);
    const empty = await screen.findByTestId("empty-state");
    expect(empty).toHaveTextContent(/No entries for this week/);
    const cta = within(empty).getByRole("button", { name: /Save daily routine/ });
    fireEvent.click(cta);
    expect(pushMock).toHaveBeenCalledWith("/daily-routine");
  });

  test("portfolio has 0 entries → 'No entries yet' empty state", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([]);
    render(<MobileDailyJournal />);
    const empty = await screen.findByTestId("empty-state");
    expect(empty).toHaveTextContent(/No entries yet/);
  });
});

describe("MobileDailyJournal — end-of-list footer", () => {
  test("Week filter shows footer suggesting Month", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      entryFixture({ day: todayStr() }),
    ]);
    render(<MobileDailyJournal />);
    const footer = await screen.findByTestId("end-of-list-footer");
    expect(footer).toHaveTextContent(/End of week · Tap Month/);
  });

  test("All filter hides the footer", async () => {
    vi.mocked(api.journalHistory).mockResolvedValue([
      entryFixture({ day: todayStr() }),
    ]);
    render(<MobileDailyJournal />);
    await screen.findByTestId(`day-card-${todayStr()}`);
    fireEvent.click(screen.getByRole("radio", { name: "All" }));
    await waitFor(() =>
      expect(screen.queryByTestId("end-of-list-footer")).not.toBeInTheDocument(),
    );
  });
});
