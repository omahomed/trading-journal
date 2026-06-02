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
      cascade_breakdown: { cascade_20: 2, cascade_15: 7 },
    },
    positions: [],
    meta: {
      fetched_at: "2026-04-13T16:00:00",
      nlv: 448382,
    },
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

    // sr8Monitor was called with the seeded NLV from journalLatest.
    await waitFor(() => expect(mMonitor).toHaveBeenCalled());
    expect(mMonitor.mock.calls[0][0]).toBeCloseTo(448382, 0);

    // Summary chips render the response values.
    expect(screen.getByTestId("sr8-chip-positions").textContent).toContain("10");
    expect(screen.getByTestId("sr8-chip-at-risk").textContent).toContain("85.7%");
    expect(screen.getByTestId("sr8-chip-to-trim").textContent).toContain("$161K");
    expect(screen.getByTestId("sr8-chip-cascades").textContent).toContain("2 20-cas / 7 15-cas");
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
  });

  test("empty positions render the placeholder copy without crashing", async () => {
    mJournal.mockResolvedValue({ end_nlv: 500000 } as any);
    mMonitor.mockResolvedValue(makeResponse({
      summary: {
        total_positions: 0,
        flagged_count: 0,
        at_risk_pct: 0,
        to_trim_dollars: 0,
        cascade_breakdown: { cascade_20: 0, cascade_15: 0 },
      },
      positions: [],
    }));

    render(<Sr8Monitor navColor="#e5484d" />);
    await waitFor(() => expect(screen.getByTestId("sr8-body-placeholder")).toBeInTheDocument());
    expect(screen.getByTestId("sr8-body-placeholder").textContent).toMatch(/No positions tagged sr8/);
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
