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
import MoreClient from "./more-client";

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

describe("MoreClient — Focus Mode toggle", () => {
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
    render(<MoreClient />);
    expect(screen.getByText("Focus Mode")).toBeInTheDocument();
    expect(screen.getByText(/Hide dollar amounts/)).toBeInTheDocument();
    expect(screen.getByRole("checkbox")).toBeInTheDocument();
  });

  test("reflects the current Focus Mode state on mount", () => {
    setFocusModeActive(true);
    render(<MoreClient />);
    expect(screen.getByRole("checkbox")).toBeChecked();
  });

  test("toggling persists to localStorage and flips the module mirror", () => {
    render(<MoreClient />);
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

describe("MoreClient — T2-6 mobile Daily section removed", () => {
  beforeEach(() => {
    stubLocalStorage();
    setFocusModeActive(false);
  });

  afterEach(() => {
    setFocusModeActive(false);
    vi.unstubAllGlobals();
  });

  test("Daily section header no longer renders", () => {
    render(<MoreClient />);
    // The mobile More previously rendered a section titled "Daily"
    // containing Daily Routine / Daily Journal / Daily Report / Weekly
    // Retro. T2-6 removes that section entirely.
    expect(screen.queryByText("Daily")).not.toBeInTheDocument();
  });

  test("Daily Workflow destinations are not reachable from mobile More menu", () => {
    render(<MoreClient />);
    // The bottom-nav Daily tab + the quick-link card row on
    // /daily-journal own these destinations now. Verify by link role +
    // accessible name so we don't false-positive on other components
    // that happen to mention these words.
    expect(
      screen.queryByRole("link", { name: "Daily Routine" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("link", { name: "Daily Journal" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("link", { name: "Daily Report" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("link", { name: "Weekly Retro" }),
    ).not.toBeInTheDocument();
  });

  test("other sections still render after Daily removal", () => {
    render(<MoreClient />);
    // Spot-check one item from each remaining section to confirm the
    // cleanup didn't accidentally drop neighbors.
    expect(screen.getByText("Display")).toBeInTheDocument();
    expect(screen.getByText("Dashboards")).toBeInTheDocument();
    expect(screen.getByText("Trading Ops")).toBeInTheDocument();
    expect(screen.getByText("Risk")).toBeInTheDocument();
    expect(screen.getByText("Market Intel")).toBeInTheDocument();
    expect(screen.getByText("Deep Dive")).toBeInTheDocument();
    expect(screen.getByText("AI")).toBeInTheDocument();
    expect(screen.getByText("Account")).toBeInTheDocument();
  });

  test("remaining sections render in the documented order", () => {
    render(<MoreClient />);
    const expected = [
      "Display",
      "Dashboards",
      "Trading Ops",
      "Risk",
      "Market Intel",
      "Deep Dive",
      "AI",
      "Account",
    ];
    const headers = expected
      .map((label) => screen.getByText(label))
      .map((el) => el.compareDocumentPosition.bind(el));
    // Confirm strict pairwise document order.
    for (let i = 0; i < expected.length - 1; i++) {
      const a = screen.getByText(expected[i]);
      const b = screen.getByText(expected[i + 1]);
      expect(
        a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING,
      ).toBeTruthy();
    }
    // Silence unused-var: the bind chain above is just to ensure the
    // type-checked accessor is exercised on each header.
    void headers;
  });
});
