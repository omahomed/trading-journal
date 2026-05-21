import { describe, test, expect, beforeEach, vi } from "vitest";
import { isChunkLoadError, tryAutoReload, handleChunkLoadError } from "./chunk-reload";

describe("isChunkLoadError", () => {
  test("detects ChunkLoadError by name", () => {
    const err = new Error("whatever");
    (err as any).name = "ChunkLoadError";
    expect(isChunkLoadError(err)).toBe(true);
  });

  test.each([
    "Loading chunk 5 failed",
    "Loading chunk vendors~main.js failed",
    "Failed to fetch dynamically imported module: /chunks/foo.js",
    "Importing a module script failed",
    "error loading dynamically imported module",
  ])("detects message pattern: %s", (msg) => {
    expect(isChunkLoadError(new Error(msg))).toBe(true);
  });

  test("returns false for unrelated errors", () => {
    expect(isChunkLoadError(new Error("Something else"))).toBe(false);
    expect(isChunkLoadError(new TypeError("undefined is not a function"))).toBe(false);
  });

  test("returns false for null / undefined / non-error inputs", () => {
    expect(isChunkLoadError(null)).toBe(false);
    expect(isChunkLoadError(undefined)).toBe(false);
    expect(isChunkLoadError("string error")).toBe(false);
    expect(isChunkLoadError({})).toBe(false);
  });
});

describe("tryAutoReload", () => {
  let reloadMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    // jsdom's window.location.reload throws "Not implemented" by
    // default. Replace just the reload fn so the rest of location
    // (href, etc.) stays intact.
    sessionStorage.clear();
    reloadMock = vi.fn();
    Object.defineProperty(window, "location", {
      configurable: true,
      value: { ...window.location, reload: reloadMock },
    });
  });

  test("reloads on first call", () => {
    const result = tryAutoReload();
    expect(result).toBe(true);
    expect(reloadMock).toHaveBeenCalledOnce();
    expect(sessionStorage.getItem("mo-chunk-reload-attempted")).toBeTruthy();
  });

  test("does not reload on second call within same session", () => {
    tryAutoReload();
    reloadMock.mockClear();
    const result = tryAutoReload();
    expect(result).toBe(false);
    expect(reloadMock).not.toHaveBeenCalled();
  });

  test("clearing sessionStorage allows a fresh attempt", () => {
    tryAutoReload();
    reloadMock.mockClear();
    sessionStorage.clear();
    const result = tryAutoReload();
    expect(result).toBe(true);
    expect(reloadMock).toHaveBeenCalledOnce();
  });
});

describe("handleChunkLoadError", () => {
  let reloadMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    sessionStorage.clear();
    reloadMock = vi.fn();
    Object.defineProperty(window, "location", {
      configurable: true,
      value: { ...window.location, reload: reloadMock },
    });
  });

  test("triggers reload for chunk errors", () => {
    expect(handleChunkLoadError(new Error("Loading chunk 1 failed"))).toBe(true);
    expect(reloadMock).toHaveBeenCalledOnce();
  });

  test("returns false for non-chunk errors and does not reload", () => {
    expect(handleChunkLoadError(new Error("Cannot read property"))).toBe(false);
    expect(reloadMock).not.toHaveBeenCalled();
  });

  test("second chunk error in same session falls through (no reload)", () => {
    handleChunkLoadError(new Error("Loading chunk 1 failed"));
    reloadMock.mockClear();
    const result = handleChunkLoadError(new Error("Loading chunk 2 failed"));
    expect(result).toBe(false);
    expect(reloadMock).not.toHaveBeenCalled();
  });
});
