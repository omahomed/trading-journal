"use client";

import { log } from "@/lib/log";

/** Escape a value for CSV — wraps in quotes and doubles inner quotes
 *  when the value contains a delimiter, quote, or newline. Callers join
 *  with "," and rows with "\n". */
export function csvEscape(v: unknown): string {
  if (v == null) return "";
  const s = String(v);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/** Trigger a browser download of `text` as a file at `filename`. Used
 *  by CSV exporters — writing to a Blob then clicking a synthesized
 *  anchor is the standard client-only download recipe. */
export function downloadTextFile(text: string, filename: string, mime = "text/csv;charset=utf-8"): void {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/** Snapshot a DOM subtree to a PNG and download it.
 *
 *  Uses html-to-image (already in node_modules; the Weekly Retro export
 *  and Trend Cycle Review use the same lib). Dynamic-imports so pages
 *  that never call it don't pay for the module in their initial bundle.
 *  Background defaults to the current --bg CSS var so light/dark themes
 *  both render legibly (no unstyled white behind dark cards).
 *
 *  Callers own the filename; a convention like
 *  `${page}-${portfolio}-${YYYY-MM-DD}.png` keeps downloads sortable.
 */
export async function exportPng(node: HTMLElement | null, filename: string): Promise<void> {
  if (!node) return;

  // Unclip everything scrollable in the subtree — plus the capture root
  // itself — so the snapshot captures the FULL natural content instead
  // of the viewport-clipped slice. The Tailwind `w-full` pattern used
  // on ACS's Equities table means the table is 100% of the
  // overflow-x-auto parent, which is 100% of the ACS root. Setting the
  // root to `max-content` (width + height) lets the whole chain size to
  // natural content. `min-width: max-content` on every scrollable
  // descendant forces the wide table's parent to grow instead of
  // clipping. Every mutation is reverted in the finally block.
  const restores: Array<() => void> = [];
  const forceExpand = (el: HTMLElement, alsoRoot = false): void => {
    const orig = {
      overflow: el.style.overflow,
      overflowY: el.style.overflowY,
      overflowX: el.style.overflowX,
      maxHeight: el.style.maxHeight,
      height: el.style.height,
      maxWidth: el.style.maxWidth,
      width: el.style.width,
      minWidth: el.style.minWidth,
    };
    restores.push(() => {
      el.style.overflow = orig.overflow;
      el.style.overflowY = orig.overflowY;
      el.style.overflowX = orig.overflowX;
      el.style.maxHeight = orig.maxHeight;
      el.style.height = orig.height;
      el.style.maxWidth = orig.maxWidth;
      el.style.width = orig.width;
      el.style.minWidth = orig.minWidth;
    });
    el.style.overflow = "visible";
    el.style.overflowY = "visible";
    el.style.overflowX = "visible";
    el.style.maxHeight = "none";
    el.style.maxWidth = "none";
    el.style.minWidth = "max-content";
    if (alsoRoot) {
      // Root: also force width/height auto so the whole capture region
      // sizes to its natural content, not the viewport it's mounted in.
      el.style.width = "max-content";
      el.style.height = "max-content";
    } else {
      el.style.height = "auto";
    }
  };
  const isScrollable = (el: Element): boolean => {
    const cs = getComputedStyle(el);
    return (
      /(auto|scroll)/.test(cs.overflow) ||
      /(auto|scroll)/.test(cs.overflowY) ||
      /(auto|scroll)/.test(cs.overflowX)
    );
  };
  // Root always gets expanded (even if it's not scrollable itself, its
  // ancestor <main> is; forcing max-content decouples the snapshot from
  // the viewport).
  forceExpand(node, true);
  node.querySelectorAll<HTMLElement>("*").forEach((el) => {
    if (isScrollable(el)) forceExpand(el);
  });

  try {
    const { toPng } = await import("html-to-image");
    // Force reflow BEFORE reading scrollWidth/Height so the values
    // reflect the post-unclip layout, not the pre-mutation size.
    // getBoundingClientRect is the standard synchronous-layout trigger.
    void node.getBoundingClientRect();
    const w = Math.max(node.scrollWidth, node.offsetWidth);
    const h = Math.max(node.scrollHeight, node.offsetHeight);
    const dataUrl = await toPng(node, {
      cacheBust: true,
      pixelRatio: 2,
      width: w,
      height: h,
      canvasWidth: w,
      canvasHeight: h,
      backgroundColor:
        getComputedStyle(document.body).getPropertyValue("--bg").trim() || "#f6f7fb",
    });
    const a = document.createElement("a");
    a.href = dataUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  } catch (e) {
    log.error("page-export", "PNG export failed", e);
    // Alert is loud, but PNG export is user-triggered — a silent failure
    // would leave them wondering why nothing downloaded. Fine for now.
    alert("PNG export failed — see console.");
  } finally {
    // Always restore the styles even if toPng threw or the download
    // failed — otherwise the user is stuck with a permanently
    // unscrollable table after a broken export attempt.
    for (const restore of restores) restore();
  }
}

/** Convenience: today's date as YYYY-MM-DD, for naming downloads. */
export function todayStamp(): string {
  return new Date().toISOString().slice(0, 10);
}
