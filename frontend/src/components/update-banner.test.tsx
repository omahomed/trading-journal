import { render, screen, fireEvent, act } from "@testing-library/react";
import { describe, test, expect, beforeEach, afterEach, vi } from "vitest";

// The component captures process.env.NEXT_PUBLIC_BUILD_ID into a module-
// level const at import time. ES module imports are hoisted by vitest's
// transformer, so a plain top-level vi.stubEnv runs AFTER the import and
// arrives too late. vi.hoisted() pushes the env mutation before all
// imports — that's the documented escape hatch for exactly this case.
vi.hoisted(() => {
  process.env.NEXT_PUBLIC_BUILD_ID = "test-loaded-build";
});

// jsdom minimal sessionStorage shim — mirrors active-campaign.test.tsx.
if (typeof window !== "undefined" && !(window as any).sessionStorage?.getItem) {
  const _store = new Map<string, string>();
  Object.defineProperty(window, "sessionStorage", {
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

import { UpdateBanner } from "./update-banner";

function mockVersionFetch(buildId: string) {
  globalThis.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ buildId }),
  }) as any;
}

beforeEach(() => {
  sessionStorage.clear();
  vi.useFakeTimers({ shouldAdvanceTime: true });
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("UpdateBanner", () => {
  test("does not render when server build matches loaded build", async () => {
    mockVersionFetch("test-loaded-build");
    const { container } = render(<UpdateBanner />);
    // Let the mount-time check resolve.
    await act(async () => { await Promise.resolve(); });
    expect(container.textContent).not.toContain("New version available");
  });

  test("renders when server build differs from loaded build", async () => {
    mockVersionFetch("server-new-build-1");
    render(<UpdateBanner />);
    expect(await screen.findByText("New version available")).toBeDefined();
  });

  test("dismiss writes the SERVER build ID to sessionStorage (not loaded)", async () => {
    mockVersionFetch("server-new-build-1");
    render(<UpdateBanner />);
    await screen.findByText("New version available");
    fireEvent.click(screen.getByTitle(/Dismiss/));
    expect(
      sessionStorage.getItem("mo-update-banner-dismissed:server-new-build-1"),
    ).toBeTruthy();
    // The loaded build's key should NOT be set — dismiss binds to the
    // server's new build so future polls of THIS build stay quiet.
    expect(
      sessionStorage.getItem("mo-update-banner-dismissed:test-loaded-build"),
    ).toBeNull();
  });

  test("dismiss persists across remount (banner stays hidden)", async () => {
    mockVersionFetch("server-new-build-1");
    const { unmount } = render(<UpdateBanner />);
    await screen.findByText("New version available");
    fireEvent.click(screen.getByTitle(/Dismiss/));
    unmount();

    // Same server build on re-mount — banner must stay hidden because
    // the sessionStorage key for this build is set.
    mockVersionFetch("server-new-build-1");
    const { container } = render(<UpdateBanner />);
    await act(async () => { await Promise.resolve(); });
    // Allow the async fetch to resolve.
    await act(async () => { await Promise.resolve(); });
    expect(container.textContent).not.toContain("New version available");
  });

  test("new server build after dismiss surfaces the banner again", async () => {
    mockVersionFetch("server-new-build-1");
    const { unmount } = render(<UpdateBanner />);
    await screen.findByText("New version available");
    fireEvent.click(screen.getByTitle(/Dismiss/));
    unmount();

    // A DIFFERENT server build ID — dismiss was for build-1, build-2
    // is new and not yet dismissed.
    mockVersionFetch("server-new-build-2");
    render(<UpdateBanner />);
    expect(await screen.findByText("New version available")).toBeDefined();
  });

  test("Refresh button confirms before reloading; cancel leaves banner intact", async () => {
    mockVersionFetch("server-new-build-1");
    const reloadMock = vi.fn();
    Object.defineProperty(window, "location", {
      configurable: true,
      value: { ...window.location, reload: reloadMock },
    });
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false);

    render(<UpdateBanner />);
    await screen.findByText("New version available");
    fireEvent.click(screen.getByText("Refresh"));

    expect(confirmSpy).toHaveBeenCalledOnce();
    expect(reloadMock).not.toHaveBeenCalled();
    // Banner still present — user cancelled.
    expect(screen.getByText("New version available")).toBeDefined();
  });

  test("Refresh button confirms and reloads on accept", async () => {
    mockVersionFetch("server-new-build-1");
    const reloadMock = vi.fn();
    Object.defineProperty(window, "location", {
      configurable: true,
      value: { ...window.location, reload: reloadMock },
    });
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(<UpdateBanner />);
    await screen.findByText("New version available");
    fireEvent.click(screen.getByText("Refresh"));

    expect(reloadMock).toHaveBeenCalledOnce();
  });
});
