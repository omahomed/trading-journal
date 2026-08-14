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

  // Walk the subtree and unclip any scrollable descendants so the
  // snapshot captures the full content instead of only what's visible
  // in the viewport. ACS's Equities table (overflow-auto + inline
  // width) was truncating the last 2 rows on export — same class of
  // bug bites any page with a scrolling data table. Every mutation is
  // reverted in the finally block, even on error.
  const restores: Array<() => void> = [];
  const isScrollable = (el: Element): boolean => {
    const cs = getComputedStyle(el);
    return (
      /(auto|scroll)/.test(cs.overflow) ||
      /(auto|scroll)/.test(cs.overflowY) ||
      /(auto|scroll)/.test(cs.overflowX)
    );
  };
  const nodesToUnclip: HTMLElement[] = [];
  if (isScrollable(node)) nodesToUnclip.push(node);
  node.querySelectorAll<HTMLElement>("*").forEach((el) => {
    if (isScrollable(el)) nodesToUnclip.push(el);
  });
  for (const el of nodesToUnclip) {
    const orig = {
      overflow: el.style.overflow,
      overflowY: el.style.overflowY,
      overflowX: el.style.overflowX,
      maxHeight: el.style.maxHeight,
      height: el.style.height,
    };
    restores.push(() => {
      el.style.overflow = orig.overflow;
      el.style.overflowY = orig.overflowY;
      el.style.overflowX = orig.overflowX;
      el.style.maxHeight = orig.maxHeight;
      el.style.height = orig.height;
    });
    el.style.overflow = "visible";
    el.style.overflowY = "visible";
    el.style.overflowX = "visible";
    el.style.maxHeight = "none";
    el.style.height = "auto";
  }

  try {
    const { toPng } = await import("html-to-image");
    // Force canvas dimensions to the fully-expanded scrollHeight/Width
    // so html-to-image doesn't guess based on the pre-unclip layout.
    const dataUrl = await toPng(node, {
      cacheBust: true,
      pixelRatio: 2,
      width: node.scrollWidth,
      height: node.scrollHeight,
      canvasWidth: node.scrollWidth,
      canvasHeight: node.scrollHeight,
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
