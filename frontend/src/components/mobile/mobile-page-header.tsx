"use client";

import type { ReactNode } from "react";

export type MobilePageHeaderProps = {
  /** First (non-italic) part of the wordmark, e.g. `"Position"`. */
  title: string;
  /** Italic-green word that closes the wordmark, e.g. `"Sizer"`. */
  italicWord: string;
  /** Optional content for the left action slot (typically a back button). */
  leftSlot?: ReactNode;
  /** Optional content for the right action slot (settings, search, more). */
  rightSlot?: ReactNode;
};

/**
 * Mobile page header. Renders three slots: optional left action, the
 * centered wordmark with italic-green emphasis on `italicWord`, and an
 * optional right action.
 *
 * The italic word uses the locked anchor design's serif italic stack
 * (`font-m-display-italic` → Iowan Old Style / Palatino / Georgia)
 * tinted with `text-m-accent` (#4ADE80). Both empty slots render an
 * invisible spacer so the title stays centered regardless of which
 * slots are present.
 */
export function MobilePageHeader({
  title,
  italicWord,
  leftSlot,
  rightSlot,
}: MobilePageHeaderProps) {
  return (
    <header className="flex items-center justify-between px-5 pt-3.5 pb-2.5">
      <div className="flex h-5 w-5 items-center justify-center">{leftSlot}</div>
      <h1 className="text-base font-medium tracking-tight text-m-text">
        {title}{" "}
        <em className="font-m-display-italic font-normal italic text-m-accent">
          {italicWord}
        </em>
      </h1>
      <div className="flex h-5 w-5 items-center justify-center">{rightSlot}</div>
    </header>
  );
}
