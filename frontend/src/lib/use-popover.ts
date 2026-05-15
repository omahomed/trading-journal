"use client";

// Shared popover machinery — extracted from four call sites that hand-
// rolled the same outside-click + Escape close idiom (color-picker,
// url-popover, toolbar-dropdown, FilterPopover in notes-rail). Rule of
// three exceeded; the helper bug class that motivated the extraction
// was the Phase 6 "consumer must remember to call X" pattern.
//
// Listener target = window (NOT document). The existing color-picker
// test fires `fireEvent.keyDown(window, ...)`; events dispatched
// directly on window do NOT bubble down to document, so a document-
// level listener would miss them. Other tests (notes-rail filter
// popover) fire on `document` with bubbles:true — those bubble UP to
// window and are caught fine. window is the only target that
// preserves the extraction-oracle invariant (existing site tests pass
// unchanged).
//
// onClose semantics — KEY: `opts.onClose` fires ONLY when the popover
// closes via outside-click or Escape, NOT when the caller invokes
// `setIsOpen(false)` / `close()` / `toggle()` directly. This lets
// parents distinguish "user dismissed" (cleanup, save partial state)
// from "we decided to close" (selection complete, form submitted).
// Conflating them double-fires cleanup in flows where the parent is
// already handling the close locally.

import { useCallback, useEffect, useRef, useState } from "react";

export interface UsePopoverOptions {
  /** Called when the popover closes via outside-click or Escape.
   *  NOT called when the caller explicitly invokes setIsOpen(false),
   *  close(), or toggle() while open. The distinction lets parents
   *  manage "user dismissed" (onClose) separately from "we decided
   *  to close" (selection complete, form submitted, etc.). */
  onClose?: () => void;

  /** Starting open state. Default false. Pass `true` for components
   *  that are conditionally mounted by their parent (UrlPopover,
   *  FilterPopover) — the hook should already be in the open state
   *  from first render. */
  initialOpen?: boolean;

  /** External anchor element to exclude from the outside-click check.
   *  Required ONLY when the anchor lives outside the popover surface
   *  (e.g., FilterPopover's anchor is owned by its FilterBar parent
   *  in a separate component).
   *
   *  For "single-wrapper" usages (ColorPicker, ToolbarDropdown),
   *  attach `surfaceRef` to a wrapper containing both anchor and
   *  surface — the single contains check finds both. Omit
   *  `anchorRef` in that case. */
  anchorRef?: React.RefObject<HTMLElement | null>;
}

export interface UsePopoverReturn<T extends HTMLElement = HTMLElement> {
  isOpen: boolean;
  setIsOpen: (next: boolean) => void;
  toggle: () => void;
  open: () => void;
  close: () => void;

  /** Attach to the popover surface OR to a wrapper that contains
   *  both the trigger and the surface (single-wrapper pattern).
   *  The generic parameter lets consumers narrow the element type
   *  (e.g., `usePopover<HTMLDivElement>(...)`) so React's strict
   *  ref-type invariance accepts the assignment to a div / span /
   *  button without a cast. */
  surfaceRef: React.RefObject<T | null>;
}

export function usePopover<T extends HTMLElement = HTMLElement>(
  opts: UsePopoverOptions = {},
): UsePopoverReturn<T> {
  const { onClose, initialOpen = false, anchorRef } = opts;
  const [isOpen, setIsOpenState] = useState<boolean>(initialOpen);
  const surfaceRef = useRef<T | null>(null);

  // Stable callback refs — onClose may identity-change across renders
  // (e.g., parent passes an inline arrow). Capture via ref so the
  // listener-attach useEffect doesn't tear down + reattach on every
  // parent render, which would race with rapid mousedown events.
  const onCloseRef = useRef<typeof onClose>(onClose);
  useEffect(() => { onCloseRef.current = onClose; }, [onClose]);

  const setIsOpen = useCallback((next: boolean) => {
    setIsOpenState(next);
  }, []);
  const toggle = useCallback(() => {
    setIsOpenState(prev => !prev);
  }, []);
  const open = useCallback(() => {
    setIsOpenState(true);
  }, []);
  const close = useCallback(() => {
    setIsOpenState(false);
  }, []);

  useEffect(() => {
    if (!isOpen) return;

    const handleMouseDown = (e: MouseEvent) => {
      const target = e.target as Node | null;
      if (!target) return;
      if (surfaceRef.current && surfaceRef.current.contains(target)) return;
      if (anchorRef?.current && anchorRef.current.contains(target)) return;
      setIsOpenState(false);
      onCloseRef.current?.();
    };
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        setIsOpenState(false);
        onCloseRef.current?.();
      }
    };

    window.addEventListener("mousedown", handleMouseDown);
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("mousedown", handleMouseDown);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isOpen, anchorRef]);

  return { isOpen, setIsOpen, toggle, open, close, surfaceRef };
}
