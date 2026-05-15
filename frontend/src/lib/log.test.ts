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

  // ============================================================
  // Phase B (v2) — devOnly variants
  // ============================================================
  // Gate: process.env.NODE_ENV !== "production". The tests mutate
  // NODE_ENV per case and restore it in cleanup. Vitest defaults to
  // NODE_ENV=test which is "not production" → fires by default; the
  // production case is what we explicitly verify.

  test("log.warn.devOnly fires in non-production (NODE_ENV=test default)", () => {
    const spy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const orig = process.env.NODE_ENV;
    (process.env as any).NODE_ENV = "development";
    try {
      log.warn.devOnly("pwa", "service worker registration failed",
        new Error("denied"));
      expect(spy).toHaveBeenCalledTimes(1);
      expect(spy).toHaveBeenCalledWith(
        "[pwa] service worker registration failed:",
        expect.any(Error),
      );
    } finally {
      (process.env as any).NODE_ENV = orig;
    }
  });

  test("log.warn.devOnly is a no-op in production", () => {
    const spy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const orig = process.env.NODE_ENV;
    (process.env as any).NODE_ENV = "production";
    try {
      log.warn.devOnly("pwa", "service worker registration failed",
        new Error("denied"));
      expect(spy).not.toHaveBeenCalled();
    } finally {
      (process.env as any).NODE_ENV = orig;
    }
  });

  test("log.debug.devOnly fires in non-production", () => {
    const spy = vi.spyOn(console, "debug").mockImplementation(() => {});
    const orig = process.env.NODE_ENV;
    (process.env as any).NODE_ENV = "development";
    try {
      log.debug.devOnly("daily-routine", "pre-fill missing (expected)",
        { status: 404 });
      expect(spy).toHaveBeenCalledTimes(1);
      expect(spy).toHaveBeenCalledWith(
        "[daily-routine] pre-fill missing (expected):",
        { status: 404 },
      );
    } finally {
      (process.env as any).NODE_ENV = orig;
    }
  });

  test("log.debug.devOnly is a no-op in production", () => {
    const spy = vi.spyOn(console, "debug").mockImplementation(() => {});
    const orig = process.env.NODE_ENV;
    (process.env as any).NODE_ENV = "production";
    try {
      log.debug.devOnly("daily-routine", "pre-fill missing (expected)",
        { status: 404 });
      expect(spy).not.toHaveBeenCalled();
    } finally {
      (process.env as any).NODE_ENV = orig;
    }
  });
});
