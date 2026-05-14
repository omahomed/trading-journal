import { render, screen, fireEvent, act } from "@testing-library/react";
import { describe, test, expect, beforeEach } from "vitest";

// jsdom localStorage shim — matches the weekly-retro / weekly-thoughts
// pattern so the lazy initializer doesn't throw on first call.
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
    writable: true,
  });
}

import { SectionExpander } from "./section-expander";

const KEY = "test-section-expander-key";

describe("SectionExpander", () => {
  beforeEach(() => {
    try { localStorage.clear(); } catch { /* shim */ }
  });

  test("renders the title", () => {
    render(
      <SectionExpander title="My Section" defaultExpanded localStorageKey={KEY}>
        <div>body content</div>
      </SectionExpander>,
    );
    expect(screen.getByRole("button", { name: /My Section/i })).toBeInTheDocument();
  });

  // Initial expanded state — three localStorage cases.

  test("localStorage key present + 'true' → expanded regardless of defaultExpanded", () => {
    localStorage.setItem(KEY, "true");
    render(
      <SectionExpander title="X" defaultExpanded={false} localStorageKey={KEY}>
        <div data-testid="body">body</div>
      </SectionExpander>,
    );
    expect(screen.getByRole("button", { name: /X/i })).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByTestId("body")).toBeInTheDocument();
  });

  test("localStorage key present + 'false' → collapsed regardless of defaultExpanded", () => {
    localStorage.setItem(KEY, "false");
    render(
      <SectionExpander title="X" defaultExpanded={true} localStorageKey={KEY}>
        <div data-testid="body">body</div>
      </SectionExpander>,
    );
    expect(screen.getByRole("button", { name: /X/i })).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByTestId("body")).not.toBeInTheDocument();
  });

  test("localStorage key absent → falls back to defaultExpanded", () => {
    render(
      <SectionExpander title="X" defaultExpanded={true} localStorageKey={KEY}>
        <div data-testid="body">body</div>
      </SectionExpander>,
    );
    expect(screen.getByRole("button", { name: /X/i })).toHaveAttribute("aria-expanded", "true");
  });

  test("click toggles state and writes to localStorage", () => {
    render(
      <SectionExpander title="X" defaultExpanded={false} localStorageKey={KEY}>
        <div>body</div>
      </SectionExpander>,
    );
    const btn = screen.getByRole("button", { name: /X/i });
    expect(btn).toHaveAttribute("aria-expanded", "false");
    expect(localStorage.getItem(KEY)).toBeNull();

    act(() => { fireEvent.click(btn); });
    expect(btn).toHaveAttribute("aria-expanded", "true");
    expect(localStorage.getItem(KEY)).toBe("true");

    act(() => { fireEvent.click(btn); });
    expect(btn).toHaveAttribute("aria-expanded", "false");
    expect(localStorage.getItem(KEY)).toBe("false");
  });

  test("body is UNMOUNTED when collapsed (not just hidden)", () => {
    render(
      <SectionExpander
        title="X"
        defaultExpanded={true}
        localStorageKey={KEY}
        bodyId="my-body"
      >
        <div>body content</div>
      </SectionExpander>,
    );
    // Expanded by default — body in document.
    expect(document.getElementById("my-body")).not.toBeNull();
    // Collapse.
    act(() => { fireEvent.click(screen.getByRole("button", { name: /X/i })); });
    expect(document.getElementById("my-body")).toBeNull();
    // Re-expand → body re-mounted (fresh DOM node).
    act(() => { fireEvent.click(screen.getByRole("button", { name: /X/i })); });
    expect(document.getElementById("my-body")).not.toBeNull();
  });

  test("aria-expanded and aria-controls reflect state and id", () => {
    render(
      <SectionExpander
        title="X"
        defaultExpanded={false}
        localStorageKey={KEY}
        bodyId="body-foo"
      >
        <div>body</div>
      </SectionExpander>,
    );
    const btn = screen.getByRole("button", { name: /X/i });
    expect(btn).toHaveAttribute("aria-controls", "body-foo");
    expect(btn).toHaveAttribute("aria-expanded", "false");
    act(() => { fireEvent.click(btn); });
    expect(btn).toHaveAttribute("aria-expanded", "true");
  });

  test("bodyId auto-derives from title when omitted", () => {
    render(
      <SectionExpander
        title="My Cool Section"
        defaultExpanded={true}
        localStorageKey={KEY}
      >
        <div>body</div>
      </SectionExpander>,
    );
    expect(document.getElementById("section-my-cool-section")).not.toBeNull();
    expect(screen.getByRole("button", { name: /My Cool Section/i }))
      .toHaveAttribute("aria-controls", "section-my-cool-section");
  });

  test("showDot=true renders the amber dot", () => {
    render(
      <SectionExpander
        title="X"
        defaultExpanded={false}
        localStorageKey={KEY}
        showDot
      >
        <div>body</div>
      </SectionExpander>,
    );
    expect(screen.getByTestId("section-expander-dot")).toBeInTheDocument();
  });

  test("showDot omitted or false → no dot", () => {
    const { rerender } = render(
      <SectionExpander
        title="X"
        defaultExpanded={false}
        localStorageKey={KEY}
      >
        <div>body</div>
      </SectionExpander>,
    );
    expect(screen.queryByTestId("section-expander-dot")).not.toBeInTheDocument();

    rerender(
      <SectionExpander
        title="X"
        defaultExpanded={false}
        localStorageKey={KEY}
        showDot={false}
      >
        <div>body</div>
      </SectionExpander>,
    );
    expect(screen.queryByTestId("section-expander-dot")).not.toBeInTheDocument();
  });

  test("headerCaption is invoked with expanded state; return value rendered", () => {
    const { rerender } = render(
      <SectionExpander
        title="X"
        defaultExpanded={false}
        localStorageKey={KEY}
        headerCaption={(open) => open ? "is open" : "is closed"}
      >
        <div>body</div>
      </SectionExpander>,
    );
    const btn = screen.getByRole("button", { name: /X/i });
    expect(btn).toHaveTextContent(/is closed/);

    act(() => { fireEvent.click(btn); });
    expect(btn).toHaveTextContent(/is open/);

    // Returning null from the caption hides it (no caption text).
    rerender(
      <SectionExpander
        title="X"
        defaultExpanded={true}
        localStorageKey={KEY}
        headerCaption={(open) => open ? null : "is closed"}
      >
        <div>body</div>
      </SectionExpander>,
    );
    // After rerender, expanded state from localStorage may be "true"; caption returns null.
    expect(screen.getByRole("button", { name: /X/i })).not.toHaveTextContent(/is closed/);
    expect(screen.getByRole("button", { name: /X/i })).not.toHaveTextContent(/is open/);
  });

  test("no headerCaption prop → no caption rendered", () => {
    render(
      <SectionExpander title="X" defaultExpanded={true} localStorageKey={KEY}>
        <div>body</div>
      </SectionExpander>,
    );
    // The only text in the button should be the title.
    const btn = screen.getByRole("button", { name: /X/i });
    expect(btn.textContent?.trim()).toBe("X");
  });
});
