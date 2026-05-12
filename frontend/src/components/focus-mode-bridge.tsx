"use client";

import { useFocusMode } from "@/lib/use-focus-mode";

/**
 * Subscribes to the Focus Mode store and re-renders {children} whenever
 * the store flips. Lets the children subtree of DesktopShell pick up
 * masking changes immediately even though Next.js's layout pass-through
 * pattern would otherwise keep the subtree from re-rendering on parent
 * state changes.
 *
 * The component intentionally doesn't read the return value of
 * useFocusMode — subscribing is enough. formatCurrency reads the
 * module-level mirror directly during each child's render.
 */
export function FocusModeBridge({ children }: { children: React.ReactNode }) {
  useFocusMode();
  return <>{children}</>;
}
