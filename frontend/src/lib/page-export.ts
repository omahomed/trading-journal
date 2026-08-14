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

  // Unclip vertically-scrollable descendants (frees rows below the
  // fold) + push horizontally-scrollable descendants + their inner
  // <table> elements to `min-width: max-content` so the table can lay
  // out at its natural column widths instead of shrinking to
  // viewport. Every mutation is reverted in the finally block.
  //
  // NOTE: we deliberately DON'T touch the capture root itself. An
  // earlier version set `width/height: max-content` on the root, but
  // that collapsed the entire snapshot to a blank rectangle because
  // the root sits inside a flex-parent (`<main class="flex-1">`) and
  // `max-content` on a flex child interacts badly with the parent's
  // width resolution. Descendant-only mutations are the surgical fix.
  const restores: Array<() => void> = [];
  const saveAndSet = (
    el: HTMLElement,
    updates: Partial<CSSStyleDeclaration>,
  ): void => {
    const keys = Object.keys(updates) as Array<keyof CSSStyleDeclaration>;
    const orig: Record<string, string> = {};
    for (const k of keys) orig[k as string] = el.style.getPropertyValue(k as string);
    restores.push(() => {
      for (const k of keys) {
        const val = orig[k as string];
        if (val) el.style.setProperty(k as string, val);
        else el.style.removeProperty(k as string);
      }
    });
    Object.assign(el.style, updates);
  };
  const isScrollable = (el: Element): boolean => {
    const cs = getComputedStyle(el);
    return (
      /(auto|scroll)/.test(cs.overflow) ||
      /(auto|scroll)/.test(cs.overflowY) ||
      /(auto|scroll)/.test(cs.overflowX)
    );
  };
  node.querySelectorAll<HTMLElement>("*").forEach((el) => {
    if (isScrollable(el)) {
      saveAndSet(el, {
        overflow: "visible",
        overflowY: "visible",
        overflowX: "visible",
        maxHeight: "none",
        maxWidth: "none",
        // Force horizontal expansion so a wide table (e.g. ACS's
        // 14-column Equities table) can grow past viewport width
        // instead of getting clipped by `w-full` inside an
        // overflow-x-auto wrapper.
        minWidth: "max-content",
      });
    }
  });
  // Direct min-width on <table> elements too — `w-full` on a table
  // means 100% of its parent, so lifting the wrapper alone isn't
  // enough; the table itself needs permission to grow.
  node.querySelectorAll<HTMLElement>("table").forEach((tbl) => {
    saveAndSet(tbl, { minWidth: "max-content", width: "max-content" });
  });

  try {
    const { toPng } = await import("html-to-image");
    // Force synchronous reflow so the fresh scrollWidth/Height reflect
    // the post-unclip layout, not the pre-mutation size.
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
