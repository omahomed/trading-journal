import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
  usePathname: () => "/more",
}));

vi.mock("@/lib/use-viewport", () => ({
  useIsMobile: () => true,
}));

import { setFocusModeActive, getFocusModeSnapshot } from "@/lib/format";
import MorePage from "./page";

const FOCUS_KEY = "mo-focus-mode";

// Node 22's built-in localStorage shadows jsdom's in this harness and
// doesn't implement removeItem/clear; stub a fresh in-memory store per
// test so the persistence assertion is reliable without touching the
// real backing store.
function stubLocalStorage(): Record<string, string> {
  const data: Record<string, string> = {};
  const fake: Storage = {
    get length() {
      return Object.keys(data).length;
    },
    clear: () => {
      for (const k of Object.keys(data)) delete data[k];
    },
    getItem: (k) => (k in data ? data[k] : null),
    key: (i) => Object.keys(data)[i] ?? null,
    removeItem: (k) => {
      delete data[k];
    },
    setItem: (k, v) => {
      data[k] = String(v);
    },
  };
  vi.stubGlobal("localStorage", fake);
  // Some code paths read window.localStorage explicitly.
  Object.defineProperty(window, "localStorage", { value: fake, configurable: true });
  return data;
}

describe("MorePage — Focus Mode toggle", () => {
  let store: Record<string, string>;

  beforeEach(() => {
    store = stubLocalStorage();
    setFocusModeActive(false);
  });

  afterEach(() => {
    setFocusModeActive(false);
    vi.unstubAllGlobals();
  });

  test("renders a toggle row labeled 'Focus Mode' with the description copy", () => {
    render(<MorePage />);
    expect(screen.getByText("Focus Mode")).toBeInTheDocument();
    expect(screen.getByText(/Hide dollar amounts/)).toBeInTheDocument();
    expect(screen.getByRole("checkbox")).toBeInTheDocument();
  });

  test("reflects the current Focus Mode state on mount", () => {
    setFocusModeActive(true);
    render(<MorePage />);
    expect(screen.getByRole("checkbox")).toBeChecked();
  });

  test("toggling persists to localStorage and flips the module mirror", () => {
    render(<MorePage />);
    const toggle = screen.getByRole("checkbox") as HTMLInputElement;
    expect(toggle).not.toBeChecked();

    fireEvent.click(toggle);
    expect(getFocusModeSnapshot()).toBe(true);
    expect(store[FOCUS_KEY]).toBe("on");

    fireEvent.click(toggle);
    expect(getFocusModeSnapshot()).toBe(false);
    expect(store[FOCUS_KEY]).toBe("off");
  });
});
