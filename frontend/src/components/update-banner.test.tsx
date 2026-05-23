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

// usePathname is the only piece of next/navigation we use. A hoisted ref
// lets each test swap the value the hook returns, which is the
// navigation event the new banner reacts to.
const pathnameRef = vi.hoisted(() => ({ current: "/dashboard" }));
vi.mock("next/navigation", () => ({
  usePathname: () => pathnameRef.current,
}));

import { UpdateBanner } from "./update-banner";

function mockVersionFetch(buildId: string) {
  globalThis.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ buildId }),
  }) as any;
}

let reloadMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  pathnameRef.current = "/dashboard";
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

  test("matching server build does not trigger reload on later navigation", async () => {
    mockVersionFetch("test-loaded-build");
    const { rerender } = render(<UpdateBanner />);
    // Let the mount-time check resolve so updateAvailable would have
    // flipped if it were going to.
    await act(async () => { await Promise.resolve(); });
    await act(async () => { await Promise.resolve(); });

    // Navigate. Since no update was detected, no reload.
    pathnameRef.current = "/trade-journal";
    rerender(<UpdateBanner />);
    await act(async () => { await Promise.resolve(); });

    expect(reloadMock).not.toHaveBeenCalled();
  });

  test("differing server build flips state but does not reload until navigation", async () => {
    mockVersionFetch("server-new-build-1");
    render(<UpdateBanner />);
    // Let the version check resolve and the state update flush.
    await act(async () => { await Promise.resolve(); });
    await act(async () => { await Promise.resolve(); });

    // updateAvailable is now true internally, but pathname hasn't
    // changed → reload must not fire yet.
    expect(reloadMock).not.toHaveBeenCalled();
  });

  test("updateAvailable + navigation triggers reload", async () => {
    mockVersionFetch("server-new-build-1");
    const { rerender } = render(<UpdateBanner />);
    await act(async () => { await Promise.resolve(); });
    await act(async () => { await Promise.resolve(); });

    expect(reloadMock).not.toHaveBeenCalled();

    // Simulate navigation.
    pathnameRef.current = "/trade-journal";
    rerender(<UpdateBanner />);
    await act(async () => { await Promise.resolve(); });

    expect(reloadMock).toHaveBeenCalledOnce();
  });

  test("updateAvailable without pathname change does not reload on re-render", async () => {
    mockVersionFetch("server-new-build-1");
    const { rerender } = render(<UpdateBanner />);
    await act(async () => { await Promise.resolve(); });
    await act(async () => { await Promise.resolve(); });

    // Re-render with the same pathname — no navigation event, no reload.
    rerender(<UpdateBanner />);
    await act(async () => { await Promise.resolve(); });

    expect(reloadMock).not.toHaveBeenCalled();
  });
});
