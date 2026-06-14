import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach } from "vitest";

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
    sr8Monitor: vi.fn(),
    sr8Refresh: vi.fn(),
    journalLatest: vi.fn(),
  },
  getActivePortfolio: () => "CanSlim",
}));

import { api } from "@/lib/api";
import { Sr8Monitor } from "./sr8-monitor";

const mMonitor = vi.mocked(api.sr8Monitor);
const mRefresh = vi.mocked(api.sr8Refresh);
const mJournal = vi.mocked(api.journalLatest);

function makeResponse(overrides: Partial<any> = {}) {
  return {
    summary: {
      total_positions: 10,
      flagged_count: 3,
      at_risk_pct: 85.7,
      to_trim_dollars: 161000,
      tier_breakdown: { green: 6, quick: 2, quicksand: 1, gd: 0 },
    },
    positions: [],
    meta: {
      fetched_at: "2026-04-13T16:00:00",
      nlv: 448382,
    },
    ...overrides,
  };
}

// Helper — build an SR8AnalyzedPosition with sensible defaults so each
// test only specifies what's load-bearing.
function makePosition(overrides: Partial<any> = {}) {
  return {
    ticker: "AAA",
    b1_date: "2026-04-01",
    b1_price: 100,
    shares_held: 100,
    avg_price: 100,
    current_price: 120,
    current_dollars: 12000,
    current_pct_nlv: 12.0,
    current_tier: "GREEN",
    tier_pct_nlv: 15.0,
    target_dollars: 60000,
    delta_dollars: 0,
    delta_shares: 0,
    unreal_dollars: 2000,
    unreal_pct: 20,
    last_signal: "GREEN",
    last_signal_date: "2026-04-13",
    last_bar_date: "2026-04-18",
    signal_today: false,
    terminated: false,
    phase: 2,
    is_action: false,
    early_warn: false,
    fetch_failed: false,
    fetch_error: "",
    ...overrides,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  mJournal.mockResolvedValue({ end_nlv: 500000 } as any);
});

describe("Sr8Monitor — page scaffold (Commit 2)", () => {
  test("seeds NLV from journalLatest, then renders summary chip values from the endpoint", async () => {
    mJournal.mockResolvedValue({ end_nlv: 448382 } as any);
    mMonitor.mockResolvedValue(makeResponse());

    render(<Sr8Monitor navColor="#e5484d" />);

    // Wait until summary chips render.
    await waitFor(() => expect(screen.getByTestId("sr8-summary")).toBeInTheDocument());

    // sr8Monitor was called with the seeded NLV from journalLatest, plus
    // the active portfolio (so the backend filter matches what Active
    // Campaign shows).
    await waitFor(() => expect(mMonitor).toHaveBeenCalled());
    expect(mMonitor.mock.calls[0][0]).toBeCloseTo(448382, 0);
    expect(mMonitor.mock.calls[0][1]).toBe("CanSlim");

    // Summary chips render the response values.
    expect(screen.getByTestId("sr8-chip-positions").textContent).toContain("10");
    expect(screen.getByTestId("sr8-chip-at-risk").textContent).toContain("85.7%");
    expect(screen.getByTestId("sr8-chip-to-trim").textContent).toContain("$161K");
    // Tiers chip replaces the obsolete "2 20-cas / 7 15-cas" cascade chip.
    expect(screen.getByTestId("sr8-chip-tiers").textContent).toContain("6G · 2Q · 1QS · 0GD");
  });

  test("NLV input edit on blur re-fetches with the new value", async () => {
    mJournal.mockResolvedValue({ end_nlv: 500000 } as any);
    mMonitor.mockResolvedValue(makeResponse());

    render(<Sr8Monitor navColor="#e5484d" />);
    await waitFor(() => expect(screen.getByTestId("sr8-summary")).toBeInTheDocument());

    const initialCallCount = mMonitor.mock.calls.length;
    const input = screen.getByTestId("sr8-nlv-input") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "750000" } });
    fireEvent.blur(input);

    await waitFor(() => {
      expect(mMonitor.mock.calls.length).toBeGreaterThan(initialCallCount);
    });
    // Most recent call uses the new NLV.
    const lastCall = mMonitor.mock.calls[mMonitor.mock.calls.length - 1];
    expect(lastCall[0]).toBeCloseTo(750000, 0);
  });

  test("Refresh button calls api.sr8Refresh with the current NLV", async () => {
    mJournal.mockResolvedValue({ end_nlv: 448382 } as any);
    mMonitor.mockResolvedValue(makeResponse());
    mRefresh.mockResolvedValue(makeResponse());

    render(<Sr8Monitor navColor="#e5484d" />);
    await waitFor(() => expect(screen.getByTestId("sr8-summary")).toBeInTheDocument());

    fireEvent.click(screen.getByTestId("sr8-refresh-btn"));

    // Wait for sr8Refresh to fire — the explicit assertion is on the
    // wiring (button → endpoint), not on the rendered post-refresh
    // payload (which has its own race with the initial fetch resolving
    // and is covered by the existing setData path in the happy-path test).
    await waitFor(() => expect(mRefresh).toHaveBeenCalled());
    expect(mRefresh.mock.calls[0][0]).toBeCloseTo(448382, 0);
    expect(mRefresh.mock.calls[0][1]).toBe("CanSlim");
  });

  test("empty positions render the placeholder copy without crashing", async () => {
    mJournal.mockResolvedValue({ end_nlv: 500000 } as any);
    mMonitor.mockResolvedValue(makeResponse({
      summary: {
        total_positions: 0,
        flagged_count: 0,
        at_risk_pct: 0,
        to_trim_dollars: 0,
        tier_breakdown: { green: 0, quick: 0, quicksand: 0, gd: 0 },
      },
      positions: [],
    }));

    render(<Sr8Monitor navColor="#e5484d" />);
    await waitFor(() => expect(screen.getByTestId("sr8-empty-state")).toBeInTheDocument());
    expect(screen.getByTestId("sr8-empty-state").textContent).toMatch(/No positions tagged sr8/);
  });

  test("error from endpoint renders an inline error message", async () => {
    mJournal.mockResolvedValue({ end_nlv: 500000 } as any);
    mMonitor.mockResolvedValue({ error: "engine import failed" } as any);

    render(<Sr8Monitor navColor="#e5484d" />);

    await waitFor(() => expect(screen.getByTestId("sr8-error")).toBeInTheDocument());
    expect(screen.getByTestId("sr8-error").textContent).toContain("engine import failed");
  });

  test("loading state renders skeleton placeholders before first response", async () => {
    mJournal.mockResolvedValue({ end_nlv: 500000 } as any);
    let resolveFetch: (v: any) => void = () => {};
    mMonitor.mockReturnValueOnce(new Promise(r => { resolveFetch = r; }));

    render(<Sr8Monitor navColor="#e5484d" />);
    // Wait for the seed effect to complete + skeleton to appear.
    await waitFor(() => expect(screen.getByTestId("sr8-loading")).toBeInTheDocument());

    // Resolve the fetch — skeleton should give way to the body placeholder.
    resolveFetch(makeResponse());
    await waitFor(() => expect(screen.queryByTestId("sr8-loading")).not.toBeInTheDocument());
  });
});

// ─── Action / Hold sections + Mark-done (Commit 3) ──────────────────

describe("Sr8Monitor — Action / Hold sections (Commit 3)", () => {
  beforeEach(() => {
    // Ensure localStorage starts clean for the mark-done tests.
    window.localStorage.clear();
  });

  test("flagged positions render as Action rows; non-flagged in Hold table", async () => {
    mJournal.mockResolvedValue({ end_nlv: 500000 } as any);
    mMonitor.mockResolvedValue(makeResponse({
      positions: [
        makePosition({ ticker: "FOO", is_action: true, delta_shares: 50, tier_pct_nlv: 15, last_signal: "GREEN", current_pct_nlv: 22 }),
        makePosition({ ticker: "BAR", is_action: false, last_signal: "ENTRY", current_pct_nlv: 5.2 }),
      ],
    }));

    render(<Sr8Monitor navColor="#e5484d" />);

    await waitFor(() => expect(screen.getByTestId("sr8-action-section")).toBeInTheDocument());

    // FOO renders as an action row; BAR as a hold row.
    expect(screen.getByTestId("sr8-action-FOO")).toBeInTheDocument();
    expect(screen.getByTestId("sr8-hold-row-BAR")).toBeInTheDocument();
    expect(screen.queryByTestId("sr8-action-BAR")).not.toBeInTheDocument();
    expect(screen.queryByTestId("sr8-hold-row-FOO")).not.toBeInTheDocument();
  });

  test("EXIT action (terminated) renders 'EXIT all N sh' verb", async () => {
    mJournal.mockResolvedValue({ end_nlv: 500000 } as any);
    mMonitor.mockResolvedValue(makeResponse({
      positions: [
        makePosition({ ticker: "DEAD", is_action: true, terminated: true, last_signal: "GD", shares_held: 80, current_pct_nlv: 8.7 }),
      ],
    }));

    render(<Sr8Monitor navColor="#e5484d" />);
    await waitFor(() => expect(screen.getByTestId("sr8-action-DEAD")).toBeInTheDocument());

    const row = screen.getByTestId("sr8-action-DEAD");
    expect(row.textContent).toContain("EXIT");
    expect(row.textContent).toContain("80 sh");
    expect(row.textContent).toContain("full exit ends campaign");
  });

  test("TRIM action renders 'TRIM N sh → tier% NLV target'", async () => {
    mJournal.mockResolvedValue({ end_nlv: 500000 } as any);
    mMonitor.mockResolvedValue(makeResponse({
      positions: [
        // QUICK tier with the new single 15/10/5/0 schedule → floor 10%.
        // 168 sh trim to bring position down to 10% NLV.
        makePosition({ ticker: "FOO", is_action: true, current_tier: "QUICK", last_signal: "QUICK",
                       delta_shares: 168, tier_pct_nlv: 10, current_pct_nlv: 16.3 }),
      ],
    }));

    render(<Sr8Monitor navColor="#e5484d" />);
    await waitFor(() => expect(screen.getByTestId("sr8-action-FOO")).toBeInTheDocument());

    const row = screen.getByTestId("sr8-action-FOO");
    expect(row.textContent).toContain("TRIM");
    expect(row.textContent).toContain("168 sh");
    expect(row.textContent).toContain("10% NLV target");
    // The hint line now omits the cascade-core prefix (single schedule).
    expect(row.textContent).toContain("16.3% → 10%");
    expect(row.textContent).not.toContain("-cascade");
  });

  test("Signal badge binds to current_tier (live ratchet), not last_signal", async () => {
    // Newly-entered SR8 position: log emission is "ENTRY" (the engine's
    // initial deploy bar) but the live cascade tier is GREEN. Pre-fix,
    // the badge would read ENTRY; post-fix it must read GREEN so SNDK-
    // style positions show their true tier.
    mJournal.mockResolvedValue({ end_nlv: 500000 } as any);
    mMonitor.mockResolvedValue(makeResponse({
      positions: [
        makePosition({
          ticker: "SNDK",
          // Below 15% threshold so it lands in the Hold section (not Action).
          current_pct_nlv: 14.2,
          is_action: false,
          last_signal: "ENTRY",      // log says ENTRY
          current_tier: "GREEN",     // ratchet says GREEN
        }),
      ],
    }));

    render(<Sr8Monitor navColor="#e5484d" />);
    await waitFor(() => expect(screen.getByTestId("sr8-hold-row-SNDK")).toBeInTheDocument());

    // SignalBadge renders data-testid=`sr8-signal-${label}`. The badge
    // must reflect the live tier, not the last log emission.
    expect(screen.getByTestId("sr8-signal-GREEN")).toBeInTheDocument();
    expect(screen.queryByTestId("sr8-signal-ENTRY")).not.toBeInTheDocument();
  });

  test("Mark done removes the row from the Action section after the 320ms animation", async () => {
    mJournal.mockResolvedValue({ end_nlv: 500000 } as any);
    mMonitor.mockResolvedValue(makeResponse({
      positions: [
        makePosition({ ticker: "FOO", is_action: true, delta_shares: 50, tier_pct_nlv: 15, last_signal: "GREEN", current_pct_nlv: 22 }),
      ],
    }));

    render(<Sr8Monitor navColor="#e5484d" />);
    await waitFor(() => expect(screen.getByTestId("sr8-action-FOO")).toBeInTheDocument());

    fireEvent.click(screen.getByTestId("sr8-mark-done-FOO"));

    // Wait past the 320ms collapse animation — waitFor polls until the
    // assertion passes or times out (default 1000ms, plenty of headroom).
    await waitFor(() => {
      expect(screen.queryByTestId("sr8-action-FOO")).not.toBeInTheDocument();
    });
  });

  test("Mark done writes the ticker to localStorage, namespaced by fetched_at", async () => {
    mJournal.mockResolvedValue({ end_nlv: 500000 } as any);
    mMonitor.mockResolvedValue(makeResponse({
      positions: [
        makePosition({ ticker: "FOO", is_action: true, delta_shares: 50, tier_pct_nlv: 15, last_signal: "GREEN", current_pct_nlv: 22 }),
      ],
      meta: { fetched_at: "2026-04-13T16:00:00", nlv: 500000 },
    }));

    render(<Sr8Monitor navColor="#e5484d" />);
    await waitFor(() => expect(screen.getByTestId("sr8-action-FOO")).toBeInTheDocument());

    fireEvent.click(screen.getByTestId("sr8-mark-done-FOO"));

    // Wait until the localStorage write happens (post-animation).
    await waitFor(() => {
      const raw = window.localStorage.getItem("sr8_monitor_done_v1");
      expect(raw).toBeTruthy();
    });
    const parsed = JSON.parse(window.localStorage.getItem("sr8_monitor_done_v1")!);
    expect(parsed.fetched_at).toBe("2026-04-13T16:00:00");
    expect(parsed.tickers).toContain("FOO");
  });

  test("done list resets when fetched_at changes (new snapshot week)", async () => {
    // Pre-seed localStorage with a done entry under an OLD fetched_at.
    window.localStorage.setItem("sr8_monitor_done_v1", JSON.stringify({
      fetched_at: "2026-04-06T16:00:00",  // last week
      tickers: ["FOO"],
    }));

    mJournal.mockResolvedValue({ end_nlv: 500000 } as any);
    mMonitor.mockResolvedValue(makeResponse({
      positions: [
        makePosition({ ticker: "FOO", is_action: true, delta_shares: 50, tier_pct_nlv: 15, last_signal: "GREEN", current_pct_nlv: 22 }),
      ],
      meta: { fetched_at: "2026-04-13T16:00:00", nlv: 500000 },  // new week
    }));

    render(<Sr8Monitor navColor="#e5484d" />);

    // FOO should appear in the Action list — the previous-week's done
    // entry is stale (fetched_at mismatch) and doesn't carry over.
    await waitFor(() => expect(screen.getByTestId("sr8-action-FOO")).toBeInTheDocument());
  });

  test("Hold table renders sortable headers with default sort indicator", async () => {
    mJournal.mockResolvedValue({ end_nlv: 500000 } as any);
    mMonitor.mockResolvedValue(makeResponse({
      positions: [
        makePosition({ ticker: "A", is_action: false, current_pct_nlv: 5.0 }),
        makePosition({ ticker: "B", is_action: false, current_pct_nlv: 12.5 }),
        makePosition({ ticker: "C", is_action: false, current_pct_nlv: 8.0 }),
      ],
    }));

    render(<Sr8Monitor navColor="#e5484d" />);
    await waitFor(() => expect(screen.getByTestId("sr8-hold-table")).toBeInTheDocument());

    // Default sort: % NLV desc → B (12.5) first, C (8.0), A (5.0).
    const rows = screen.getByTestId("sr8-hold-table").querySelectorAll("tbody tr");
    expect(rows[0].textContent).toContain("B");
    expect(rows[1].textContent).toContain("C");
    expect(rows[2].textContent).toContain("A");

    // The % NLV header carries the active sort caret.
    expect(screen.getByTestId("sr8-hold-th-current_pct_nlv").textContent).toContain("▼");
  });

  test("Clicking the Ticker header toggles to ascending alphabetical sort", async () => {
    mJournal.mockResolvedValue({ end_nlv: 500000 } as any);
    mMonitor.mockResolvedValue(makeResponse({
      positions: [
        makePosition({ ticker: "MU", is_action: false, current_pct_nlv: 5 }),
        makePosition({ ticker: "ALAB", is_action: false, current_pct_nlv: 8 }),
        makePosition({ ticker: "ZZZ", is_action: false, current_pct_nlv: 3 }),
      ],
    }));

    render(<Sr8Monitor navColor="#e5484d" />);
    await waitFor(() => expect(screen.getByTestId("sr8-hold-table")).toBeInTheDocument());

    fireEvent.click(screen.getByTestId("sr8-hold-th-ticker"));

    await waitFor(() => {
      const rows = screen.getByTestId("sr8-hold-table").querySelectorAll("tbody tr");
      // After click → asc → ALAB, MU, ZZZ.
      expect(rows[0].textContent).toContain("ALAB");
      expect(rows[1].textContent).toContain("MU");
      expect(rows[2].textContent).toContain("ZZZ");
    });
  });

  test("Early-warning row carries the NEAR pill", async () => {
    mJournal.mockResolvedValue({ end_nlv: 500000 } as any);
    mMonitor.mockResolvedValue(makeResponse({
      positions: [
        makePosition({ ticker: "TER", is_action: false, early_warn: true, last_signal: "QUICK", current_pct_nlv: 11.1, tier_pct_nlv: 11.25 }),
      ],
    }));

    render(<Sr8Monitor navColor="#e5484d" />);
    await waitFor(() => expect(screen.getByTestId("sr8-hold-row-TER")).toBeInTheDocument());

    expect(screen.getByTestId("sr8-near-TER")).toBeInTheDocument();
    expect(screen.getByTestId("sr8-near-TER").textContent).toContain("NEAR");
  });

  test("Fetch-failed row renders muted with 'price unavailable' + a clickable Retry button", async () => {
    mJournal.mockResolvedValue({ end_nlv: 500000 } as any);
    mMonitor.mockResolvedValue(makeResponse({
      positions: [
        makePosition({ ticker: "DELL", fetch_failed: true, fetch_error: "yfinance 429", current_price: null }),
      ],
    }));

    render(<Sr8Monitor navColor="#e5484d" />);
    await waitFor(() => expect(screen.getByTestId("sr8-hold-row-DELL")).toBeInTheDocument());

    const row = screen.getByTestId("sr8-hold-row-DELL");
    expect(row.textContent).toContain("fetch failed");
    expect(row.textContent).toContain("price unavailable");

    // Retry is now wired (Commit 4): clickable, with the live label.
    const retryBtn = screen.getByTestId("sr8-retry-DELL") as HTMLButtonElement;
    expect(retryBtn.disabled).toBe(false);
    expect(retryBtn.textContent).toContain("Retry");
  });

  test("ENTRY-tier row (defensive fallback) shows '/ building' in the % NLV column", async () => {
    // Under the SR8-weekly conform, current_tier is always one of
    // GREEN/QUICK/QUICKSAND/GD — the engine never surfaces "ENTRY" as
    // a live tier (it's a log-emission label only). The "/ building"
    // UX is kept as a defensive fallback in HoldRow; this test pins
    // that fallback by forcing current_tier="ENTRY" explicitly.
    mJournal.mockResolvedValue({ end_nlv: 500000 } as any);
    mMonitor.mockResolvedValue(makeResponse({
      positions: [
        makePosition({ ticker: "SNDK", is_action: false,
                       current_tier: "ENTRY", last_signal: "ENTRY",
                       current_pct_nlv: 5.2 }),
      ],
    }));

    render(<Sr8Monitor navColor="#e5484d" />);
    await waitFor(() => expect(screen.getByTestId("sr8-hold-row-SNDK")).toBeInTheDocument());
    expect(screen.getByTestId("sr8-hold-row-SNDK").textContent).toContain("/ building");
  });

  test("All-clear panel appears when no actions but positions exist", async () => {
    mJournal.mockResolvedValue({ end_nlv: 500000 } as any);
    mMonitor.mockResolvedValue(makeResponse({
      positions: [
        makePosition({ ticker: "AAA", is_action: false }),
        makePosition({ ticker: "BBB", is_action: false }),
      ],
    }));

    render(<Sr8Monitor navColor="#e5484d" />);
    await waitFor(() => expect(screen.getByTestId("sr8-action-section")).toBeInTheDocument());

    expect(screen.getByTestId("sr8-all-clear")).toBeInTheDocument();
    expect(screen.queryByTestId("sr8-action-rows")).not.toBeInTheDocument();
  });
});

// ─── Polished states (Commit 4) ─────────────────────────────────────

describe("Sr8Monitor — polished states (Commit 4)", () => {
  test("All-clear panel renders the heading + active-positions count", async () => {
    mJournal.mockResolvedValue({ end_nlv: 500000 } as any);
    mMonitor.mockResolvedValue(makeResponse({
      positions: [
        makePosition({ ticker: "AAA", is_action: false }),
        makePosition({ ticker: "BBB", is_action: false }),
        makePosition({ ticker: "CCC", is_action: false }),
      ],
    }));

    render(<Sr8Monitor navColor="#e5484d" />);
    await waitFor(() => expect(screen.getByTestId("sr8-all-clear")).toBeInTheDocument());

    const panel = screen.getByTestId("sr8-all-clear");
    expect(panel.textContent).toContain("No actions today");
    expect(panel.textContent).toContain("All");
    // 3 priced holds (no failed rows in this fixture).
    expect(panel.textContent).toContain("3");
    expect(panel.textContent).toContain("holding to plan");
  });

  test("Empty state renders the calm panel + sr8 chip + tagging copy", async () => {
    mJournal.mockResolvedValue({ end_nlv: 500000 } as any);
    mMonitor.mockResolvedValue(makeResponse({
      positions: [],
      summary: {
        total_positions: 0,
        flagged_count: 0,
        at_risk_pct: 0,
        to_trim_dollars: 0,
        tier_breakdown: { green: 0, quick: 0, quicksand: 0, gd: 0 },
      },
    }));

    render(<Sr8Monitor navColor="#e5484d" />);
    await waitFor(() => expect(screen.getByTestId("sr8-empty-state")).toBeInTheDocument());

    const panel = screen.getByTestId("sr8-empty-state");
    expect(panel.textContent).toContain("No positions tagged sr8");
    expect(panel.textContent).toContain("hold / trim / exit");
    // sr8 rendered as a chip (mono <code>).
    const chip = panel.querySelector("code");
    expect(chip?.textContent).toBe("sr8");
  });

  test("Loading state renders pulsing dot + 3 skeleton rows before first response", async () => {
    mJournal.mockResolvedValue({ end_nlv: 500000 } as any);
    let resolveFetch: (v: any) => void = () => {};
    mMonitor.mockReturnValueOnce(new Promise(r => { resolveFetch = r; }));

    render(<Sr8Monitor navColor="#e5484d" />);

    await waitFor(() => expect(screen.getByTestId("sr8-loading")).toBeInTheDocument());
    // 3 shimmer skeleton rows.
    expect(screen.getByTestId("sr8-skeleton-0")).toBeInTheDocument();
    expect(screen.getByTestId("sr8-skeleton-1")).toBeInTheDocument();
    expect(screen.getByTestId("sr8-skeleton-2")).toBeInTheDocument();
    // "Fetching latest prices…" copy.
    expect(screen.getByTestId("sr8-loading").textContent).toContain("Fetching latest prices");

    // Resolve fetch → loading state disappears.
    resolveFetch(makeResponse());
    await waitFor(() => expect(screen.queryByTestId("sr8-loading")).not.toBeInTheDocument());
  });

  test("Retry button on a failed row calls sr8Refresh", async () => {
    mJournal.mockResolvedValue({ end_nlv: 500000 } as any);
    mMonitor.mockResolvedValue(makeResponse({
      positions: [
        makePosition({ ticker: "DELL", fetch_failed: true, current_price: null }),
      ],
    }));
    mRefresh.mockResolvedValue(makeResponse({
      positions: [
        makePosition({ ticker: "DELL", fetch_failed: false, current_price: 200, current_pct_nlv: 8.5 }),
      ],
    }));

    render(<Sr8Monitor navColor="#e5484d" />);
    await waitFor(() => expect(screen.getByTestId("sr8-retry-DELL")).toBeInTheDocument());

    const btn = screen.getByTestId("sr8-retry-DELL") as HTMLButtonElement;
    expect(btn.disabled).toBe(false);  // no longer the disabled stub
    expect(btn.textContent).toContain("Retry");

    fireEvent.click(btn);

    // sr8Refresh fired with the seeded NLV and the active portfolio.
    await waitFor(() => expect(mRefresh).toHaveBeenCalled());
    expect(mRefresh.mock.calls[0][0]).toBe(500000);
    expect(mRefresh.mock.calls[0][1]).toBe("CanSlim");
  });

  test("Retry while in flight shows the 'Retrying…' label", async () => {
    mJournal.mockResolvedValue({ end_nlv: 500000 } as any);
    mMonitor.mockResolvedValue(makeResponse({
      positions: [
        makePosition({ ticker: "DELL", fetch_failed: true, current_price: null }),
      ],
    }));
    // Hold the refresh promise open so we can observe the in-flight state.
    let resolveRefresh: (v: any) => void = () => {};
    mRefresh.mockReturnValueOnce(new Promise(r => { resolveRefresh = r; }));

    render(<Sr8Monitor navColor="#e5484d" />);
    await waitFor(() => expect(screen.getByTestId("sr8-retry-DELL")).toBeInTheDocument());

    fireEvent.click(screen.getByTestId("sr8-retry-DELL"));

    await waitFor(() => {
      expect((screen.getByTestId("sr8-retry-DELL") as HTMLButtonElement).textContent).toContain("Retrying");
    });

    // Resolve refresh → button label flips back (and the row resolves
    // to a normal hold row if the refresh succeeded — fixture re-uses
    // the same failed payload to keep the test focused on the label
    // transition, not the row's post-retry rendering).
    resolveRefresh(makeResponse({
      positions: [
        makePosition({ ticker: "DELL", fetch_failed: true, current_price: null }),
      ],
    }));
    await waitFor(() => {
      expect((screen.getByTestId("sr8-retry-DELL") as HTMLButtonElement).textContent).not.toContain("Retrying");
    });
  });
});
