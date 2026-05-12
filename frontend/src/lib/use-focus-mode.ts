import { useSyncExternalStore } from "react";
import { subscribeFocusMode, getFocusModeSnapshot } from "./format";

/**
 * Subscribe a React component to the Focus Mode store. Component
 * re-renders whenever setFocusModeActive flips the underlying state.
 * Used by FocusModeBridge to make the children subtree of DesktopShell
 * reactive without relying on parent-state propagation through
 * Next.js's layout boundaries.
 *
 * SSR snapshot is `false` (Focus Mode is off on the server — the
 * helper only activates after hydration once DesktopShell's load
 * effect runs).
 */
export function useFocusMode(): boolean {
  return useSyncExternalStore(
    subscribeFocusMode,
    getFocusModeSnapshot,
    () => false,
  );
}
