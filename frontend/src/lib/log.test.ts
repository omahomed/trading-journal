import { describe, test, expect, vi, afterEach } from "vitest";
import { log } from "./log";

describe("log", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  test("log.error emits '[area] what:' prefix with the err as second arg", () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    log.error("weekly-retro", "rail fetch failed", "boom");
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy).toHaveBeenCalledWith("[weekly-retro] rail fetch failed:", "boom");
  });

  test("log.warn routes to console.warn with the same bracket format", () => {
    const spy = vi.spyOn(console, "warn").mockImplementation(() => {});
    log.warn("pwa", "service worker registration failed", new Error("denied"));
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy).toHaveBeenCalledWith(
      "[pwa] service worker registration failed:",
      expect.any(Error),
    );
  });

  test("log.debug routes to console.debug for expected non-error cases", () => {
    const spy = vi.spyOn(console, "debug").mockImplementation(() => {});
    log.debug("weekly-retro", "retro missing (expected)", { status: 404 });
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy).toHaveBeenCalledWith(
      "[weekly-retro] retro missing (expected):",
      { status: 404 },
    );
  });
});
