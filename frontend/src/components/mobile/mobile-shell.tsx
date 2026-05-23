"use client";

import type { ReactNode } from "react";
import { MobileBottomNav } from "./mobile-bottom-nav";
import { MobilePageHeader, type MobilePageHeaderProps } from "./mobile-page-header";
import { MobilePortfolioPicker } from "./mobile-portfolio-picker";
import { MobileTapePill } from "./mobile-tape-pill";

type Props = {
  /** Header configuration: wordmark + optional right accessory. The
   *  portfolio picker is injected as the default `rightSlot`; pages
   *  that need a different right accessory can override. */
  header: Omit<MobilePageHeaderProps, "rightSlot"> & Pick<MobilePageHeaderProps, "rightSlot">;
  /** Page content for the scrollable middle region. */
  children: ReactNode;
};

/**
 * Top-level mobile chrome.
 *
 * Layout pattern is iOS-standard: the outer container is exactly the
 * dynamic viewport height (`100dvh`). Top zone (tape pill + page
 * header) and bottom nav are flex children that don't shrink; the
 * middle `<main>` takes the remaining space and scrolls internally.
 * As a result the bottom nav and header stay pinned without needing
 * `position: sticky`, and the tape pill sits above the page content.
 *
 * Phase 2 step 1: the portfolio picker is mounted globally as the
 * header's right-slot accessory. Per-route `rightSlot` overrides are
 * still permitted (callers pass their own in `header.rightSlot`), but
 * the default is the picker.
 *
 * All colors and spacing come from the `m-*` token utilities defined
 * in `mobile-tokens.css`. No inline `style={{}}` is used in this tree.
 */
export function MobileShell({ header, children }: Props) {
  const resolvedHeader: MobilePageHeaderProps = {
    ...header,
    rightSlot: header.rightSlot ?? <MobilePortfolioPicker />,
  };
  return (
    <div className="flex h-[100dvh] flex-col bg-m-bg font-m-ui text-m-text">
      <div className="shrink-0 px-4 pt-2">
        <MobileTapePill />
      </div>
      <div className="shrink-0">
        <MobilePageHeader {...resolvedHeader} />
      </div>
      <main className="flex-1 overflow-y-auto px-5 pb-4">{children}</main>
      <div className="shrink-0">
        <MobileBottomNav />
      </div>
    </div>
  );
}
