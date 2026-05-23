"use client";

import type { ReactNode } from "react";

export type MobilePageHeaderProps = {
  /** First (non-italic) part of the wordmark, e.g. `"Position"`. */
  title: string;
  /** Italic-green word that closes the wordmark, e.g. `"Sizer"`. */
  italicWord: string;
  /** Optional right-aligned accessory — Phase 2 mounts the portfolio picker here. */
  rightSlot?: ReactNode;
};

/**
 * Mobile page header. Left-aligned wordmark with italic-green emphasis
 * on `italicWord`, optional right-aligned accessory slot.
 *
 * The italic word uses the locked anchor design's serif italic stack
 * (`font-m-display-italic` → Iowan Old Style / Palatino / Georgia)
 * tinted with `text-m-accent` (#4ADE80).
 *
 * Phase 1 had a three-slot centered layout with a left back-chevron
 * stub and per-page right icon stubs. Phase 2 step 1 (portfolio
 * context plumbing) reframes the header around the global portfolio
 * picker on the right; back navigation lives in the bottom nav.
 */
export function MobilePageHeader({
  title,
  italicWord,
  rightSlot,
}: MobilePageHeaderProps) {
  return (
    <header className="flex items-center justify-between gap-3 px-5 pt-3.5 pb-2.5">
      <h1 className="min-w-0 truncate text-base font-medium tracking-tight text-m-text">
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
