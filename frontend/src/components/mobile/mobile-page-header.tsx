"use client";

import type { ReactNode } from "react";

export type MobilePageHeaderProps = {
  /** First (non-italic) part of the wordmark, e.g. `"Position"`. */
  title: string;
  /** Italic-green word that closes the wordmark, e.g. `"Sizer"`. */
  italicWord: string;
  /** Optional left-aligned accessory — Phase 2 T2-4 introduced this for
   *  detail-page back chevrons. List pages omit it (`undefined`) and the
   *  bottom nav handles navigation. */
  leftSlot?: ReactNode;
  /** Optional right-aligned accessory — Phase 2 mounts the portfolio
   *  picker (list pages) or the focus-mode toggle (detail pages) here. */
  rightSlot?: ReactNode;
};

/**
 * Mobile page header. Left-aligned wordmark with italic-green emphasis
 * on `italicWord`. Optional left + right accessory slots.
 *
 * The italic word uses the locked anchor design's serif italic stack
 * (`font-m-display-italic` → Iowan Old Style / Palatino / Georgia)
 * tinted with `text-m-accent` (#4ADE80).
 *
 * Slot conventions established across phases:
 *   - List pages (Daily Journal, Weekly Retro list, Trade Journal, etc.)
 *     mount the global portfolio picker in `rightSlot` and leave
 *     `leftSlot` undefined — back navigation lives in the bottom nav.
 *   - Detail pages (Daily Report, Weekly Retro detail, etc.) mount a
 *     back chevron in `leftSlot` and a context-specific control
 *     (focus-mode toggle, kebab menu, etc.) in `rightSlot`. The
 *     portfolio picker is omitted because the page is scoped to the
 *     entity's portfolio.
 *
 * Phase 1 had a three-slot centered layout with a left back-chevron
 * stub and per-page right icon stubs. Phase 2 step 1 reframed the
 * header around a single right slot (portfolio picker). Phase 2 T2-4
 * reintroduces `leftSlot` for detail pages — the original three-slot
 * intent was correct; the Phase 1 stubs were just empty.
 */
export function MobilePageHeader({
  title,
  italicWord,
  leftSlot,
  rightSlot,
}: MobilePageHeaderProps) {
  return (
    <header className="flex items-center justify-between gap-3 px-5 pt-3 pb-2">
      {leftSlot ? <div className="shrink-0">{leftSlot}</div> : null}
      <h1 className="min-w-0 flex-1 truncate text-[30px] font-medium leading-tight tracking-[-0.01em] text-m-text">
        {title}
        {title ? " " : ""}
        <em className="font-m-display-italic font-normal italic text-m-accent">
          {italicWord}
        </em>
      </h1>
      {rightSlot ? <div className="shrink-0">{rightSlot}</div> : null}
    </header>
  );
}
