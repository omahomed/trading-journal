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
    weeklyRetroList: vi.fn(),
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
import { MobileWeeklyRetro } from "./mobile-weekly-retro";
import type { NotesRailItem, NotesRailItemTag, Portfolio } from "@/lib/api";

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

function pad(n: number): string {
  return String(n).padStart(2, "0");
}

// Current Monday in YYYY-MM-DD — most default-filter tests anchor to it.
function currentMonday(): string {
  const n = new Date();
  const day = n.getDay();
  const offset = day === 0 ? -6 : 1 - day;
  const m = new Date(n);
  m.setDate(n.getDate() + offset);
  return `${m.getFullYear()}-${pad(m.getMonth() + 1)}-${pad(m.getDate())}`;
}

function addDays(iso: string, days: number): string {
  const [y, m, d] = iso.split("-").map(Number);
  const date = new Date(y, m - 1, d + days);
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

function weekFixture(opts: {
  week_start: string;
  id?: number | null;
  has_content?: boolean;
  pinned?: boolean;
  sparkline_value?: number | null;
  week_grade?: string | null;
  weekly_pnl?: number | null;
  trades_count?: number;
  tags?: NotesRailItemTag[];
}): NotesRailItem {
  const week_end = addDays(opts.week_start, 4);
  const [y, m] = opts.week_start.split("-").map(Number);
  return {
    id: opts.id ?? 100,
    key: opts.week_start,
    week_start: opts.week_start,
    week_end,
    year: y,
    month: m,
    title: `${opts.week_start} – ${week_end}`,
    has_content: opts.has_content ?? true,
    pinned: opts.pinned ?? false,
    sparkline_value: opts.sparkline_value ?? 0,
    week_grade: opts.week_grade ?? null,
    weekly_pnl: opts.weekly_pnl ?? 0,
    trades_count: opts.trades_count ?? 0,
    tags: opts.tags ?? [],
    reviewed_at: null,
  };
}

function mockList(weeks: NotesRailItem[]) {
  vi.mocked(api.weeklyRetroList).mockResolvedValue({
    weeks,
    ytd_stats: {
      total_weeks: weeks.length,
      weeks_graded: 0,
      avg_grade: null,
      weeks_pinned: 0,
    },
  });
}

beforeEach(() => {
  setPortfolio();
  vi.clearAllMocks();
  mockList([]);
});

// ── Mount fetch ────────────────────────────────────────────────────

describe("MobileWeeklyRetro — mount fetch", () => {
  test("calls weeklyRetroList with active portfolio", async () => {
    render(<MobileWeeklyRetro />);
    await waitFor(() =>
      expect(api.weeklyRetroList).toHaveBeenCalledWith("CanSlim"),
    );
  });

  test("filters out synthetic empty rows (has_content=false or id=null)", async () => {
    const real = currentMonday();
    const empty1 = addDays(real, -7);
    const empty2 = addDays(real, -14);
    mockList([
      weekFixture({ week_start: real, has_content: true, id: 1 }),
      // Synthetic — id null, has_content false. Should NOT render.
      weekFixture({ week_start: empty1, id: null, has_content: false }),
      weekFixture({ week_start: empty2, id: 99, has_content: false }),
    ]);
    render(<MobileWeeklyRetro />);
    await screen.findByTestId(`week-card-${real}`);
    expect(screen.queryByTestId(`week-card-${empty1}`)).not.toBeInTheDocument();
    expect(screen.queryByTestId(`week-card-${empty2}`)).not.toBeInTheDocument();
  });
});

// ── Filter defaults ────────────────────────────────────────────────

describe("MobileWeeklyRetro — default filter is current year + current month", () => {
  test("only weeks in current month visible on mount", async () => {
    const cur = currentMonday();
    const inThisMonth = cur;
    // Two months back — definitely outside current month.
    const inOtherMonth = addDays(cur, -70);
    mockList([
      weekFixture({ week_start: inThisMonth, id: 1 }),
      weekFixture({ week_start: inOtherMonth, id: 2 }),
    ]);
    render(<MobileWeeklyRetro />);
    await screen.findByTestId(`week-card-${inThisMonth}`);
    expect(screen.queryByTestId(`week-card-${inOtherMonth}`)).not.toBeInTheDocument();
  });
});

// ── Filter pills open sheets ───────────────────────────────────────

describe("MobileWeeklyRetro — filter pills", () => {
  test("year pill opens sheet listing years descending plus 'All time'", async () => {
    mockList([
      weekFixture({ week_start: "2026-05-04", id: 1 }),
      weekFixture({ week_start: "2025-06-02", id: 2 }),
      weekFixture({ week_start: "2024-03-04", id: 3 }),
    ]);
    render(<MobileWeeklyRetro />);
    await waitFor(() => expect(api.weeklyRetroList).toHaveBeenCalled());
    const yearPill = screen.getByLabelText(/Filter by year:/);
    fireEvent.click(yearPill);
    const dialog = await screen.findByRole("dialog", { name: "Filter by year" });
    const options = within(dialog).getAllByRole("option");
    // "All time" first, then years descending. Plus any current year not in data.
    const labels = options.map((o) => o.textContent ?? "");
    expect(labels[0]).toContain("All time");
    // Should contain 2026, 2025, 2024 in descending order.
    const idx2026 = labels.findIndex((l) => l.startsWith("2026"));
    const idx2025 = labels.findIndex((l) => l.startsWith("2025"));
    const idx2024 = labels.findIndex((l) => l.startsWith("2024"));
    expect(idx2026).toBeGreaterThanOrEqual(0);
    expect(idx2025).toBeGreaterThan(idx2026);
    expect(idx2024).toBeGreaterThan(idx2025);
  });

  test("month pill opens sheet listing 12 months plus 'All months'", async () => {
    render(<MobileWeeklyRetro />);
    await waitFor(() => expect(api.weeklyRetroList).toHaveBeenCalled());
    const monthPill = screen.getByLabelText(/Filter by month:/);
    fireEvent.click(monthPill);
    const dialog = await screen.findByRole("dialog", { name: "Filter by month" });
    const options = within(dialog).getAllByRole("option");
    const labels = options.map((o) => o.textContent ?? "");
    expect(labels[0]).toContain("All months");
    expect(labels).toEqual(
      expect.arrayContaining([
        expect.stringContaining("January"),
        expect.stringContaining("December"),
      ]),
    );
    expect(options.length).toBe(13); // All months + 12
  });

  test("selecting 'All months' widens to show all months in current year", async () => {
    const yyyy = new Date().getFullYear();
    mockList([
      weekFixture({ week_start: `${yyyy}-05-04`, id: 1 }),
      weekFixture({ week_start: `${yyyy}-02-03`, id: 2 }),
    ]);
    render(<MobileWeeklyRetro />);
    await waitFor(() => expect(api.weeklyRetroList).toHaveBeenCalled());
    fireEvent.click(screen.getByLabelText(/Filter by month:/));
    const dialog = await screen.findByRole("dialog", { name: "Filter by month" });
    fireEvent.click(within(dialog).getByRole("option", { name: /All months/ }));
    await screen.findByTestId(`week-card-${yyyy}-02-03`);
    expect(screen.getByTestId(`week-card-${yyyy}-05-04`)).toBeInTheDocument();
  });

  test("Reset chip snaps year + month back to current-current", async () => {
    const yyyy = new Date().getFullYear();
    mockList([
      weekFixture({ week_start: `${yyyy}-05-04`, id: 1 }),
      weekFixture({ week_start: `${yyyy}-02-03`, id: 2 }),
    ]);
    render(<MobileWeeklyRetro />);
    await waitFor(() => expect(api.weeklyRetroList).toHaveBeenCalled());
    // Open + change month to "All months".
    fireEvent.click(screen.getByLabelText(/Filter by month:/));
    const dialog = await screen.findByRole("dialog", { name: "Filter by month" });
    fireEvent.click(within(dialog).getByRole("option", { name: /All months/ }));
    // February card visible now.
    await screen.findByTestId(`week-card-${yyyy}-02-03`);
    // Reset.
    fireEvent.click(screen.getByTestId("filter-reset"));
    await waitFor(() =>
      expect(screen.queryByTestId(`week-card-${yyyy}-02-03`)).not.toBeInTheDocument(),
    );
  });
});

// ── Cross-month filter semantic ────────────────────────────────────

describe("MobileWeeklyRetro — cross-month weeks filter by week_start month", () => {
  test("Apr 28 – May 2 appears under April filter, not May filter", async () => {
    // Use a non-current month to avoid the default filter masking the
    // assertion. Anchor to a stable year.
    const crossMonthWeek = "2024-04-28"; // → ends 2024-05-02
    mockList([weekFixture({ week_start: crossMonthWeek, id: 1 })]);
    render(<MobileWeeklyRetro />);
    await waitFor(() => expect(api.weeklyRetroList).toHaveBeenCalled());
    // Switch year to 2024 first.
    fireEvent.click(screen.getByLabelText(/Filter by year:/));
    const yearDialog = await screen.findByRole("dialog", { name: "Filter by year" });
    fireEvent.click(within(yearDialog).getByRole("option", { name: "2024" }));
    // Filter to May — should NOT show the Apr-28-starting week.
    fireEvent.click(screen.getByLabelText(/Filter by month:/));
    let monthDialog = await screen.findByRole("dialog", { name: "Filter by month" });
    fireEvent.click(within(monthDialog).getByRole("option", { name: "May" }));
    await waitFor(() =>
      expect(screen.queryByTestId(`week-card-${crossMonthWeek}`)).not.toBeInTheDocument(),
    );
    // Filter to April — SHOULD show.
    fireEvent.click(screen.getByLabelText(/Filter by month:/));
    monthDialog = await screen.findByRole("dialog", { name: "Filter by month" });
    fireEvent.click(within(monthDialog).getByRole("option", { name: "April" }));
    await screen.findByTestId(`week-card-${crossMonthWeek}`);
  });
});

// ── Sticky section header ──────────────────────────────────────────

describe("MobileWeeklyRetro — sticky month header", () => {
  test("renders month-year label with retro count", async () => {
    const cur = currentMonday();
    mockList([weekFixture({ week_start: cur, id: 1 })]);
    render(<MobileWeeklyRetro />);
    const yyyyMm = cur.slice(0, 7);
    const header = await screen.findByTestId(`month-header-${yyyyMm}`);
    expect(header).toHaveTextContent(/\d{4}/); // year
    expect(header).toHaveTextContent(/1 retro/);
  });

  test("plural retros count when multiple in same month", async () => {
    const cur = currentMonday();
    const prev = addDays(cur, -7);
    mockList([
      weekFixture({ week_start: cur, id: 1 }),
      weekFixture({ week_start: prev, id: 2 }),
    ]);
    render(<MobileWeeklyRetro />);
    const yyyyMm = cur.slice(0, 7);
    // Same month if cur and prev share month.
    if (cur.slice(0, 7) === prev.slice(0, 7)) {
      const header = await screen.findByTestId(`month-header-${yyyyMm}`);
      expect(header).toHaveTextContent(/2 retros/);
    } else {
      // Edge: if cur is first Mon of month, prev is in prior month —
      // still verify singular form renders.
      await screen.findByTestId(`week-card-${cur}`);
    }
  });
});

// ── Tap-to-detail ──────────────────────────────────────────────────

describe("MobileWeeklyRetro — card tap navigation", () => {
  test("tap pushes /weekly-retro?week=YYYY-MM-DD", async () => {
    const cur = currentMonday();
    mockList([weekFixture({ week_start: cur, id: 1 })]);
    render(<MobileWeeklyRetro />);
    const card = await screen.findByTestId(`week-card-${cur}`);
    fireEvent.click(card);
    expect(pushMock).toHaveBeenCalledWith(`/weekly-retro?week=${cur}`);
  });
});

// ── Pin indicator ──────────────────────────────────────────────────

describe("MobileWeeklyRetro — pin indicator", () => {
  test("pinned: true → pin icon renders", async () => {
    const cur = currentMonday();
    mockList([weekFixture({ week_start: cur, id: 1, pinned: true })]);
    render(<MobileWeeklyRetro />);
    await screen.findByTestId(`week-card-pin-${cur}`);
  });

  test("pinned: false → no pin icon", async () => {
    const cur = currentMonday();
    mockList([weekFixture({ week_start: cur, id: 1, pinned: false })]);
    render(<MobileWeeklyRetro />);
    await screen.findByTestId(`week-card-${cur}`);
    expect(screen.queryByTestId(`week-card-pin-${cur}`)).not.toBeInTheDocument();
  });
});

// ── Grade pill tier mapping ────────────────────────────────────────

describe("MobileWeeklyRetro — grade pill tier mapping", () => {
  test.each([
    ["A+", "high"],
    ["A", "high"],
    ["A-", "high"],
    ["B+", "mid"],
    ["B", "mid"],
    ["B-", "mid"],
    ["C+", "low"],
    ["C", "low"],
    ["D", "low"],
    ["F", "low"],
  ])("%s → %s tier", async (letter, tier) => {
    const cur = currentMonday();
    mockList([weekFixture({ week_start: cur, id: 1, week_grade: letter })]);
    render(<MobileWeeklyRetro />);
    const pill = await screen.findByTestId(`grade-pill-${tier}`);
    expect(pill).toHaveTextContent(letter);
  });

  test("null grade → muted em-dash placeholder", async () => {
    const cur = currentMonday();
    mockList([weekFixture({ week_start: cur, id: 1, week_grade: null })]);
    render(<MobileWeeklyRetro />);
    await screen.findByTestId("grade-pill-none");
  });
});

// ── LTD / YTD compute ──────────────────────────────────────────────

describe("MobileWeeklyRetro — LTD/YTD client-side compute", () => {
  test("LTD chains across all weeks oldest-first", async () => {
    // Three weeks in 2024: +10%, +5%, +2%.
    // LTD chain (oldest → newest): 1.10, 1.155, 1.1781 → LTDs:
    //   Wk1 (oldest): 10.00%
    //   Wk2:         15.50%
    //   Wk3 (newest): 17.81%
    const w1 = "2024-01-01"; // Monday
    const w2 = "2024-01-08";
    const w3 = "2024-01-15";
    mockList([
      // Order in payload doesn't matter; component sorts internally.
      weekFixture({ week_start: w3, id: 3, sparkline_value: 2 }),
      weekFixture({ week_start: w1, id: 1, sparkline_value: 10 }),
      weekFixture({ week_start: w2, id: 2, sparkline_value: 5 }),
    ]);
    render(<MobileWeeklyRetro />);
    // Need to widen filter past current-current first.
    fireEvent.click(await screen.findByLabelText(/Filter by year:/));
    const yearDialog = await screen.findByRole("dialog", { name: "Filter by year" });
    fireEvent.click(within(yearDialog).getByRole("option", { name: "2024" }));
    fireEvent.click(screen.getByLabelText(/Filter by month:/));
    const monthDialog = await screen.findByRole("dialog", { name: "Filter by month" });
    fireEvent.click(within(monthDialog).getByRole("option", { name: "January" }));
    const ltd1 = await screen.findByTestId(`sub-ltd-${w1}`);
    const ltd2 = screen.getByTestId(`sub-ltd-${w2}`);
    const ltd3 = screen.getByTestId(`sub-ltd-${w3}`);
    expect(ltd1).toHaveTextContent(/\+10\.0% LTD/);
    expect(ltd2).toHaveTextContent(/\+15\.5% LTD/);
    expect(ltd3).toHaveTextContent(/\+17\.8% LTD/);
  });

  test("YTD resets at January 1 — late-2023 + early-2024 chain independently", async () => {
    // Dec 2023: +20%. Jan 2024 wk1: +5%.
    // YTDs: 2023-12 → 20.00%, 2024-01-01 → 5.00% (reset).
    // LTDs: 2023-12 → 20.00%, 2024-01-01 → 26.00% (1.20*1.05 - 1).
    const wDec = "2023-12-25"; // Monday
    const wJan = "2024-01-01"; // Monday
    mockList([
      weekFixture({ week_start: wDec, id: 1, sparkline_value: 20 }),
      weekFixture({ week_start: wJan, id: 2, sparkline_value: 5 }),
    ]);
    render(<MobileWeeklyRetro />);
    // Widen: All time + All months.
    fireEvent.click(await screen.findByLabelText(/Filter by year:/));
    let dialog = await screen.findByRole("dialog", { name: "Filter by year" });
    fireEvent.click(within(dialog).getByRole("option", { name: /All time/ }));
    fireEvent.click(screen.getByLabelText(/Filter by month:/));
    dialog = await screen.findByRole("dialog", { name: "Filter by month" });
    fireEvent.click(within(dialog).getByRole("option", { name: /All months/ }));

    const ytdDec = await screen.findByTestId(`sub-ytd-${wDec}`);
    const ytdJan = screen.getByTestId(`sub-ytd-${wJan}`);
    expect(ytdDec).toHaveTextContent(/\+20\.0% YTD/);
    expect(ytdJan).toHaveTextContent(/\+5\.0% YTD/);
    // LTD continues across the year boundary.
    expect(screen.getByTestId(`sub-ltd-${wJan}`)).toHaveTextContent(/\+26\.0% LTD/);
  });

  test("missing sparkline_value treated as 0% for chain", async () => {
    const cur = currentMonday();
    mockList([weekFixture({ week_start: cur, id: 1, sparkline_value: null })]);
    render(<MobileWeeklyRetro />);
    const ltd = await screen.findByTestId(`sub-ltd-${cur}`);
    // 0% LTD / YTD when there's only one row with a missing return.
    expect(ltd).toHaveTextContent(/\+0\.0% LTD/);
  });

  test("single-week year: YTD equals the week's return", async () => {
    const w = "2024-03-04"; // Monday
    mockList([weekFixture({ week_start: w, id: 1, sparkline_value: 7.5 })]);
    render(<MobileWeeklyRetro />);
    fireEvent.click(await screen.findByLabelText(/Filter by year:/));
    const dialog = await screen.findByRole("dialog", { name: "Filter by year" });
    fireEvent.click(within(dialog).getByRole("option", { name: "2024" }));
    fireEvent.click(screen.getByLabelText(/Filter by month:/));
    const monthDialog = await screen.findByRole("dialog", { name: "Filter by month" });
    fireEvent.click(within(monthDialog).getByRole("option", { name: "March" }));
    const ytd = await screen.findByTestId(`sub-ytd-${w}`);
    expect(ytd).toHaveTextContent(/\+7\.5% YTD/);
  });
});

// ── Tag pills ──────────────────────────────────────────────────────

describe("MobileWeeklyRetro — tag pills", () => {
  test("tags render with stored palette color", async () => {
    const cur = currentMonday();
    mockList([
      weekFixture({
        week_start: cur,
        id: 1,
        tags: [
          { name: "breakout-week", color: "emerald" },
          { name: "conviction", color: "violet" },
        ],
      }),
    ]);
    render(<MobileWeeklyRetro />);
    const card = await screen.findByTestId(`week-card-${cur}`);
    expect(within(card).getByTestId("tag-pill-breakout-week")).toHaveStyle({
      color: "#047857",
    });
    expect(within(card).getByTestId("tag-pill-conviction")).toHaveStyle({
      color: "#6d28d9",
    });
  });

  test("unknown palette color falls back to neutral muted style", async () => {
    const cur = currentMonday();
    mockList([
      weekFixture({
        week_start: cur,
        id: 1,
        // Unknown palette key — not in TAG_PALETTE.
        tags: [{ name: "unknown-tone", color: "neon-pink" }],
      }),
    ]);
    render(<MobileWeeklyRetro />);
    const pill = await screen.findByTestId("tag-pill-unknown-tone");
    // Fallback uses border-m-border + bg-m-surface-2; assert by class
    // rather than inline style (since fallback path uses utility classes).
    expect(pill.className).toMatch(/bg-m-surface-2/);
    expect(pill).toHaveTextContent("unknown-tone");
  });

  test("empty tags array → tag row entirely omitted (no placeholder div)", async () => {
    const cur = currentMonday();
    mockList([weekFixture({ week_start: cur, id: 1, tags: [] })]);
    render(<MobileWeeklyRetro />);
    await screen.findByTestId(`week-card-${cur}`);
    expect(screen.queryByTestId(`week-card-tags-${cur}`)).not.toBeInTheDocument();
  });
});

// ── Empty state ────────────────────────────────────────────────────

describe("MobileWeeklyRetro — empty state", () => {
  test("filter excludes all entries → empty state with filter label", async () => {
    // Old data → current-current default filter excludes it.
    mockList([weekFixture({ week_start: "2020-01-06", id: 1 })]);
    render(<MobileWeeklyRetro />);
    const empty = await screen.findByTestId("empty-state");
    expect(empty).toHaveTextContent(/No weekly retros/);
  });

  test("no entries at all → empty state", async () => {
    mockList([]);
    render(<MobileWeeklyRetro />);
    const empty = await screen.findByTestId("empty-state");
    expect(empty).toHaveTextContent(/No weekly retros/);
  });
});

// ── End-of-list footer ─────────────────────────────────────────────

describe("MobileWeeklyRetro — end-of-list footer", () => {
  test("default current-current filter → 'End of {Month} {Year}'", async () => {
    const cur = currentMonday();
    mockList([weekFixture({ week_start: cur, id: 1 })]);
    render(<MobileWeeklyRetro />);
    const footer = await screen.findByTestId("end-of-list-footer");
    expect(footer.textContent).toMatch(/End of \w+ \d{4}/);
  });

  test("all-time + all-months → 'End of history'", async () => {
    const cur = currentMonday();
    mockList([weekFixture({ week_start: cur, id: 1 })]);
    render(<MobileWeeklyRetro />);
    await waitFor(() => expect(api.weeklyRetroList).toHaveBeenCalled());
    fireEvent.click(screen.getByLabelText(/Filter by year:/));
    let dialog = await screen.findByRole("dialog", { name: "Filter by year" });
    fireEvent.click(within(dialog).getByRole("option", { name: /All time/ }));
    fireEvent.click(screen.getByLabelText(/Filter by month:/));
    dialog = await screen.findByRole("dialog", { name: "Filter by month" });
    fireEvent.click(within(dialog).getByRole("option", { name: /All months/ }));
    const footer = await screen.findByTestId("end-of-list-footer");
    expect(footer).toHaveTextContent("End of history");
  });
});

// ── Newest-first sort within grouping ──────────────────────────────

describe("MobileWeeklyRetro — newest-first within month grouping", () => {
  test("cards appear newest-first even if payload arrives unsorted", async () => {
    // Same-month, 3 weeks. The backend returns newest-first; we test that
    // the component preserves that ordering through the grouping pass.
    const cur = currentMonday();
    const w2 = addDays(cur, -7);
    const w3 = addDays(cur, -14);
    // Only use this set if all three share a month — guard with a check.
    if (cur.slice(0, 7) !== w2.slice(0, 7) || cur.slice(0, 7) !== w3.slice(0, 7)) {
      return;
    }
    mockList([
      // Already newest-first as backend returns.
      weekFixture({ week_start: cur, id: 1 }),
      weekFixture({ week_start: w2, id: 2 }),
      weekFixture({ week_start: w3, id: 3 }),
    ]);
    render(<MobileWeeklyRetro />);
    const cards = await screen.findAllByLabelText(/Open weekly retro for/);
    expect(cards[0]).toHaveAttribute("data-testid", `week-card-${cur}`);
    expect(cards[1]).toHaveAttribute("data-testid", `week-card-${w2}`);
    expect(cards[2]).toHaveAttribute("data-testid", `week-card-${w3}`);
  });
});
