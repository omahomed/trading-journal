"use client";

import { useSyncExternalStore } from "react";

/** Pixel width below which we consider the viewport "mobile". */
export const MOBILE_BREAKPOINT_PX = 1024;

const QUERY = `(max-width: ${MOBILE_BREAKPOINT_PX - 1}px)`;

function subscribe(callback: () => void): () => void {
  const mql = window.matchMedia(QUERY);
  mql.addEventListener("change", callback);
  return () => mql.removeEventListener("change", callback);
}

function getSnapshot(): boolean {
  return window.matchMedia(QUERY).matches;
}

function getServerSnapshot(): boolean {
  return false;
}

/**
 * Returns `true` when the viewport is narrower than {@link MOBILE_BREAKPOINT_PX}.
 *
 * **SSR safety.** Built on `useSyncExternalStore` with an explicit server
 * snapshot of `false`. During server rendering and on the very first
 * client render the hook returns `false`, matching the server tree —
 * so React never produces a hydration mismatch warning. Once the
 * component mounts on the client, React runs the live `getSnapshot`
 * and re-renders if the real viewport disagrees. The trade-off is one
 * frame of desktop-shaped output on a phone before the re-render; this
 * is intentional and avoids a layout flash on the dominant desktop case.
 *
 * The hook updates automatically when the viewport crosses the breakpoint
 * (orientation change, window resize, devtools docking).
 *
 * For shell-level visibility — wrapping desktop vs. mobile chrome under
 * one route — prefer the CSS-only `.m-only` / `.d-only` classes from
 * `mobile-tokens.css`. Reach for `useIsMobile()` only when component-
 * internal branching is meaningfully cheaper than rendering both trees.
 */
export function useIsMobile(): boolean {
  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}
