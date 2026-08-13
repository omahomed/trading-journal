"use client";

import { useIsMobile } from "@/lib/use-viewport";

interface Props {
  /**
   * Optional page-specific reason line rendered below the headline.
   * Default is a generic "designed for a wider viewport" message.
   */
  reason?: string;
}

/**
 * Advisory banner for pages that are meaningfully hard to use on a
 * phone (dense tables, wide charts, long forms with lots of controls).
 * Renders ONLY on mobile viewports — no-op on desktop.
 *
 * Design intent: don't hide the page (mobile users may still need to
 * *read* a campaign or *see* a chart), but tell them upfront that the
 * page is a desktop workflow so they know to switch devices before
 * trying to actually operate on it.
 *
 * Follows the mobile-token palette (`--m-*`) so it fits inside
 * MobileShell's warm-dark chrome without leaking desktop tokens.
 * Uses the amber-warn family — same visual weight the M Factor tape
 * pill uses for "UPTREND UNDER PRESSURE" so the "not ideal" reading
 * lands without shouting.
 */
export function MobileDesktopOnlyBanner({ reason }: Props) {
  const isMobile = useIsMobile();
  if (!isMobile) return null;

  return (
    <div className="mx-1 mb-4 px-4 py-3 rounded-[10px]"
         data-testid="mobile-desktop-only-banner"
         style={{
           background: "color-mix(in oklab, var(--m-warn) 10%, var(--m-surface))",
           border: "1px solid var(--m-warn-border-soft)",
         }}>
      <div className="text-[13px] font-semibold" style={{ color: "var(--m-warn)" }}>
        Best on desktop
      </div>
      <div className="text-[12px] mt-1 leading-snug" style={{ color: "var(--m-text-muted)" }}>
        {reason ?? "This page is designed for a wider viewport. Open motrading.net on a computer for the full experience."}
      </div>
    </div>
  );
}
