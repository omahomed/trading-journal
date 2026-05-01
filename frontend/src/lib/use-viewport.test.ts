import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import * as React from "react";
import { renderToString } from "react-dom/server";
import { act, render } from "@testing-library/react";
import { MOBILE_BREAKPOINT_PX, useIsMobile } from "./use-viewport";

// jsdom does not implement window.matchMedia. Fake one whose `matches`
// can be flipped at runtime so we can drive both initial state and
// `change` events.
function installMatchMedia(initialMatches: boolean) {
  const listeners = new Set<(e: MediaQueryListEvent) => void>();
  const mql = {
    matches: initialMatches,
    media: "",
    onchange: null,
    addEventListener: (type: string, l: (e: MediaQueryListEvent) => void) => {
      if (type === "change") listeners.add(l);
    },
    removeEventListener: (type: string, l: (e: MediaQueryListEvent) => void) => {
      if (type === "change") listeners.delete(l);
    },
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
  };
  const fire = (matches: boolean) => {
    mql.matches = matches;
    listeners.forEach((l) => l({ matches } as MediaQueryListEvent));
  };
  (window as unknown as { matchMedia: (q: string) => unknown }).matchMedia =
    vi.fn().mockReturnValue(mql);
  return { mql, fire };
}

function Probe({ onValue }: { onValue: (v: boolean) => void }) {
  const v = useIsMobile();
  onValue(v);
  return null;
}

describe("useIsMobile / MOBILE_BREAKPOINT_PX", () => {
  let consoleErrorSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
  });
  afterEach(() => {
    consoleErrorSpy.mockRestore();
    vi.restoreAllMocks();
  });

  test("breakpoint constant is 1024", () => {
    expect(MOBILE_BREAKPOINT_PX).toBe(1024);
  });

  test("server snapshot is always false (SSR-safe — never mismatches client first render)", () => {
    // renderToString uses getServerSnapshot. Even if the matchMedia mock
    // would say `true`, the hook must return `false` on the server so the
    // initial client render matches and React produces no hydration warning.
    installMatchMedia(true);
    const html = renderToString(
      React.createElement(Probe, {
        onValue: (v) => expect(v).toBe(false),
      }),
    );
    expect(html).toBe("");
  });

  test("client snapshot reflects matchMedia.matches", () => {
    installMatchMedia(true);
    let observed: boolean | null = null;
    render(
      React.createElement(Probe, {
        onValue: (v) => {
          observed = v;
        },
      }),
    );
    expect(observed).toBe(true);
  });

  test("re-renders when the media query 'change' event fires", () => {
    const { fire } = installMatchMedia(false);
    const values: boolean[] = [];
    render(
      React.createElement(Probe, {
        onValue: (v) => values.push(v),
      }),
    );
    expect(values.at(-1)).toBe(false);
    act(() => fire(true));
    expect(values.at(-1)).toBe(true);
    act(() => fire(false));
    expect(values.at(-1)).toBe(false);
  });

  test("emits no console.error or React warnings during mount/update/unmount", () => {
    const { fire } = installMatchMedia(false);
    const { unmount } = render(
      React.createElement(Probe, { onValue: () => {} }),
    );
    act(() => fire(true));
    unmount();
    expect(consoleErrorSpy).not.toHaveBeenCalled();
  });
});
