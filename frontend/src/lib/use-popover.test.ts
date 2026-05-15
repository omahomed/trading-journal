// @vitest-environment jsdom
import { describe, test, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useEffect, useRef } from "react";

import { usePopover, type UsePopoverOptions } from "./use-popover";

// Test harness for cases that need a surfaceRef bound to a real DOM
// node. renderHook gives us hook state, but it doesn't attach the
// returned surfaceRef to anything — for click-outside semantics we
// need an actual element the listener can run `.contains()` against.
function mountWithRefs(opts?: UsePopoverOptions & {
  surfaceEl?: HTMLElement | null;
  anchorEl?: HTMLElement | null;
}) {
  const surfaceEl = opts?.surfaceEl ?? document.createElement("div");
  document.body.appendChild(surfaceEl);

  let anchorEl: HTMLElement | null = null;
  if (opts?.anchorEl !== undefined) {
    anchorEl = opts.anchorEl;
    if (anchorEl) document.body.appendChild(anchorEl);
  }

  // Wrap the hook so we can inject a passed-in anchorRef when needed.
  const anchorRefHolder: { current: HTMLElement | null } = { current: anchorEl };

  const result = renderHook(() => {
    const hook = usePopover({
      onClose: opts?.onClose,
      initialOpen: opts?.initialOpen,
      anchorRef: opts?.anchorEl !== undefined ? anchorRefHolder : undefined,
    });
    // Wire surfaceRef → the actual DOM node so contains() works.
    useEffect(() => {
      (hook.surfaceRef as { current: HTMLElement | null }).current = surfaceEl;
    });
    return hook;
  });

  const cleanup = () => {
    if (surfaceEl.parentNode) surfaceEl.parentNode.removeChild(surfaceEl);
    if (anchorEl?.parentNode) anchorEl.parentNode.removeChild(anchorEl);
    result.unmount();
  };

  return { ...result, surfaceEl, anchorEl, cleanup };
}

describe("usePopover", () => {
  beforeEach(() => {
    // Defensive — strip any stray nodes between tests.
    while (document.body.firstChild) {
      document.body.removeChild(document.body.firstChild);
    }
  });

  test("isOpen starts at the default (false) when initialOpen is omitted", () => {
    const { result, cleanup } = mountWithRefs();
    expect(result.current.isOpen).toBe(false);
    cleanup();
  });

  test("isOpen starts true when initialOpen: true (conditional-mount idiom)", () => {
    const { result, cleanup } = mountWithRefs({ initialOpen: true });
    expect(result.current.isOpen).toBe(true);
    cleanup();
  });

  test("toggle() flips state in both directions", () => {
    const { result, cleanup } = mountWithRefs();
    expect(result.current.isOpen).toBe(false);
    act(() => { result.current.toggle(); });
    expect(result.current.isOpen).toBe(true);
    act(() => { result.current.toggle(); });
    expect(result.current.isOpen).toBe(false);
    cleanup();
  });

  test("open() sets true; close() sets false", () => {
    const { result, cleanup } = mountWithRefs();
    act(() => { result.current.open(); });
    expect(result.current.isOpen).toBe(true);
    act(() => { result.current.close(); });
    expect(result.current.isOpen).toBe(false);
    cleanup();
  });

  test("mousedown outside surfaceRef closes the popover", () => {
    const { result, cleanup } = mountWithRefs({ initialOpen: true });
    expect(result.current.isOpen).toBe(true);
    act(() => {
      const evt = new MouseEvent("mousedown", { bubbles: true });
      document.body.dispatchEvent(evt);
    });
    expect(result.current.isOpen).toBe(false);
    cleanup();
  });

  test("mousedown INSIDE surfaceRef does NOT close", () => {
    const surfaceEl = document.createElement("div");
    const inner = document.createElement("span");
    surfaceEl.appendChild(inner);
    const { result, cleanup } = mountWithRefs({ initialOpen: true, surfaceEl });
    expect(result.current.isOpen).toBe(true);
    act(() => {
      const evt = new MouseEvent("mousedown", { bubbles: true });
      inner.dispatchEvent(evt);
    });
    expect(result.current.isOpen).toBe(true);
    cleanup();
  });

  test("mousedown INSIDE anchorRef does NOT close (external-anchor case)", () => {
    // FilterPopover-style: the anchor lives outside the surface; the
    // hook must treat clicks on it as "inside" so the parent's anchor-
    // click toggle isn't fought by the hook closing the popover first.
    const anchorEl = document.createElement("button");
    const { result, cleanup } = mountWithRefs({ initialOpen: true, anchorEl });
    expect(result.current.isOpen).toBe(true);
    act(() => {
      const evt = new MouseEvent("mousedown", { bubbles: true });
      anchorEl.dispatchEvent(evt);
    });
    expect(result.current.isOpen).toBe(true);
    cleanup();
  });

  test("Escape keydown closes + e.preventDefault() is called", () => {
    const { result, cleanup } = mountWithRefs({ initialOpen: true });
    const preventDefault = vi.fn();
    act(() => {
      // Construct + dispatch a real KeyboardEvent so the hook's
      // e.preventDefault() can be observed. Patch preventDefault
      // before dispatch — bubbles to window which the hook listens on.
      const evt = new KeyboardEvent("keydown", { key: "Escape", bubbles: true });
      evt.preventDefault = preventDefault;
      window.dispatchEvent(evt);
    });
    expect(result.current.isOpen).toBe(false);
    expect(preventDefault).toHaveBeenCalledTimes(1);
    cleanup();
  });

  test("onClose fires on outside-click", () => {
    const onClose = vi.fn();
    const { cleanup } = mountWithRefs({ initialOpen: true, onClose });
    act(() => {
      const evt = new MouseEvent("mousedown", { bubbles: true });
      document.body.dispatchEvent(evt);
    });
    expect(onClose).toHaveBeenCalledTimes(1);
    cleanup();
  });

  test("onClose fires on Escape", () => {
    const onClose = vi.fn();
    const { cleanup } = mountWithRefs({ initialOpen: true, onClose });
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    });
    expect(onClose).toHaveBeenCalledTimes(1);
    cleanup();
  });

  test("onClose does NOT fire when caller invokes close() or setIsOpen(false) directly", () => {
    // The hook distinguishes "user dismissed" (fires onClose) from
    // "caller-decided close" (does NOT fire onClose). Without this,
    // parents would receive a spurious onClose every time they
    // programmatically close, double-firing any cleanup they wired.
    const onClose = vi.fn();
    const { result, cleanup } = mountWithRefs({ initialOpen: true, onClose });
    act(() => { result.current.close(); });
    expect(result.current.isOpen).toBe(false);
    expect(onClose).not.toHaveBeenCalled();

    act(() => { result.current.setIsOpen(true); });
    act(() => { result.current.setIsOpen(false); });
    expect(onClose).not.toHaveBeenCalled();

    act(() => { result.current.setIsOpen(true); });
    act(() => { result.current.toggle(); });   // open → close via toggle
    expect(onClose).not.toHaveBeenCalled();
    cleanup();
  });

  test("unmount removes listeners (no leaked mousedown/keydown)", () => {
    const onClose = vi.fn();
    const { unmount, surfaceEl } = mountWithRefs({ initialOpen: true, onClose });
    // Sanity: a pre-unmount mousedown closes (listener is live).
    // We don't assert state here — just unmount and verify nothing
    // fires post-unmount.
    surfaceEl.parentNode?.removeChild(surfaceEl);
    unmount();

    // After unmount, dispatching mousedown anywhere must NOT fire
    // onClose. If the cleanup didn't remove the listener, this would
    // fire (and the listener would also dereference a stale ref).
    const evt = new MouseEvent("mousedown", { bubbles: true });
    document.body.dispatchEvent(evt);
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    expect(onClose).not.toHaveBeenCalled();
  });

  test("listeners don't fire when isOpen is false (gating)", () => {
    const onClose = vi.fn();
    const { result, cleanup } = mountWithRefs({ initialOpen: false, onClose });
    expect(result.current.isOpen).toBe(false);
    act(() => {
      document.body.dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    });
    expect(onClose).not.toHaveBeenCalled();
    cleanup();
  });
});
