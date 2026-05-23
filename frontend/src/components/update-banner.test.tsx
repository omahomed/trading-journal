import { render, act } from "@testing-library/react";
import { describe, test, expect, beforeEach, afterEach, vi } from "vitest";

// The component captures process.env.NEXT_PUBLIC_BUILD_ID into a module-
// level const at import time. ES module imports are hoisted by vitest's
// transformer, so a plain top-level vi.stubEnv runs AFTER the import and
// arrives too late. vi.hoisted() pushes the env mutation before all
// imports — that's the documented escape hatch for exactly this case.
vi.hoisted(() => {
  process.env.NEXT_PUBLIC_BUILD_ID = "test-loaded-build";
});

import { UpdateBanner } from "./update-banner";

function mockVersionFetch(buildId: string) {
  globalThis.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ buildId }),
  }) as any;
}

function mockVersionFetchReject() {
  globalThis.fetch = vi.fn().mockRejectedValue(new Error("net down")) as any;
}

let reloadMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  reloadMock = vi.fn();
  Object.defineProperty(window, "location", {
    configurable: true,
    value: { ...window.location, reload: reloadMock },
  });
  vi.useFakeTimers({ shouldAdvanceTime: true });
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("UpdateBanner", () => {
  test("renders no DOM output", async () => {
    mockVersionFetch("test-loaded-build");
    const { container } = render(<UpdateBanner />);
    await act(async () => { await Promise.resolve(); });
    expect(container.innerHTML).toBe("");
  });

  test("matching server build does not trigger reload", async () => {
    mockVersionFetch("test-loaded-build");
    render(<UpdateBanner />);
    // Flush the mount-time check + its promise chain.
    await act(async () => { await Promise.resolve(); });
    await act(async () => { await Promise.resolve(); });
    expect(reloadMock).not.toHaveBeenCalled();
  });

  test("differing server build triggers reload immediately on first poll", async () => {
    mockVersionFetch("server-new-build");
    render(<UpdateBanner />);
    await act(async () => { await Promise.resolve(); });
    await act(async () => { await Promise.resolve(); });
    expect(reloadMock).toHaveBeenCalledOnce();
  });

  test("fetch rejection does not trigger reload (caller will retry next interval)", async () => {
    mockVersionFetchReject();
    render(<UpdateBanner />);
    await act(async () => { await Promise.resolve(); });
    await act(async () => { await Promise.resolve(); });
    expect(reloadMock).not.toHaveBeenCalled();
  });

  test("polling interval also triggers reload on mismatch", async () => {
    // Mount-time check returns matching ID — no reload yet.
    mockVersionFetch("test-loaded-build");
    render(<UpdateBanner />);
    await act(async () => { await Promise.resolve(); });
    await act(async () => { await Promise.resolve(); });
    expect(reloadMock).not.toHaveBeenCalled();

    // Server flips to a new build; advance past the poll interval.
    mockVersionFetch("server-new-build");
    await act(async () => {
      vi.advanceTimersByTime(60_001);
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(reloadMock).toHaveBeenCalledOnce();
  });

  test("reload fires at most once even if multiple poll triggers fire", async () => {
    mockVersionFetch("server-new-build");
    render(<UpdateBanner />);
    // First mount-time check triggers reload.
    await act(async () => { await Promise.resolve(); });
    await act(async () => { await Promise.resolve(); });
    expect(reloadMock).toHaveBeenCalledOnce();

    // Subsequent poll ticks should NOT call reload again — the
    // component sets the cancelled flag on first mismatch to short-
    // circuit any in-flight or future checks.
    await act(async () => {
      vi.advanceTimersByTime(60_001);
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(reloadMock).toHaveBeenCalledOnce();
  });
});
