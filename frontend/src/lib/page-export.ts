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
  try {
    const { toPng } = await import("html-to-image");
    const dataUrl = await toPng(node, {
      cacheBust: true,
      pixelRatio: 2,
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
  }
}

/** Convenience: today's date as YYYY-MM-DD, for naming downloads. */
export function todayStamp(): string {
  return new Date().toISOString().slice(0, 10);
}
