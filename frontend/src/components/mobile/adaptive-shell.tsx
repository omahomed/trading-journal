"use client";

import type { ReactNode } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ChevronLeft, EllipsisVertical, Settings } from "lucide-react";
import { DesktopShell } from "@/components/desktop-shell";
import { MobileShell } from "./mobile-shell";
import type { MobilePageHeaderProps } from "./mobile-page-header";
import { getNavItemForHref } from "@/lib/nav";

/**
 * Splits a nav label like `"Position Sizer"` into `{ title: "Position",
 * italicWord: "Sizer" }`. The last whitespace-separated word becomes
 * the italic-green wordmark; everything before becomes the upright
 * prefix. Single-word labels return `{ title: "", italicWord: label }`.
 */
function splitWordmark(label: string): { title: string; italicWord: string } {
  const trimmed = label.trim();
  if (!trimmed) return { title: "", italicWord: "" };
  const idx = trimmed.lastIndexOf(" ");
  if (idx === -1) return { title: "", italicWord: trimmed };
  return {
    title: trimmed.slice(0, idx),
    italicWord: trimmed.slice(idx + 1),
  };
}

/**
 * Last-segment fallback for routes that have no `nav.ts` entry — kebab
 * cased to Title Case. Examples: "/more" → "More", "/dev/foo" → "Foo",
 * "/mobile-shell-preview" → "Mobile Shell Preview".
 */
function fallbackLabel(pathname: string): string {
  const last = pathname.split("/").filter(Boolean).pop();
  if (!last) return "";
  return last
    .split("-")
    .map((w) => (w ? w[0].toUpperCase() + w.slice(1) : w))
    .join(" ");
}

/**
 * Per-route mobile header overrides. The `MobileShell` chrome is
 * provided by this AdaptiveShell on every mobile page, so per-page
 * customization happens here rather than each page rendering its own
 * shell. Keeps pages focused on content and avoids the SSR-hydration
 * flash that page-owned shells would produce when `useIsMobile()`
 * flips after the first paint.
 *
 * Pattern is intentionally bespoke for Phase 1's two custom-mobile
 * routes. If Phase 2+ accretes enough entries to warrant a richer
 * mechanism (sub-pages, dynamic right slots, header actions) this
 * table is the natural seed for that abstraction.
 */
const ROUTE_HEADER_OVERRIDES: Record<string, Partial<MobilePageHeaderProps>> = {
  "/position-sizer": {
    leftSlot: (
      <Link
        href="/dashboard"
        aria-label="Back"
        className="flex h-5 w-5 items-center justify-center text-m-text-muted"
      >
        <ChevronLeft size={20} strokeWidth={1.5} aria-hidden="true" />
      </Link>
    ),
    rightSlot: (
      <Link
        href="/settings"
        aria-label="Settings"
        className="flex h-5 w-5 items-center justify-center text-m-text-muted"
      >
        <Settings size={20} strokeWidth={1.5} aria-hidden="true" />
      </Link>
    ),
  },
  "/trade-journal": {
    leftSlot: (
      <Link
        href="/dashboard"
        aria-label="Back"
        className="flex h-5 w-5 items-center justify-center text-m-text-muted"
      >
        <ChevronLeft size={20} strokeWidth={1.5} aria-hidden="true" />
      </Link>
    ),
    // TODO Phase 4: open the trade-journal "more" sheet.
    rightSlot: (
      <button
        type="button"
        aria-label="More options"
        onClick={() => {
          /* TODO Phase 4 */
        }}
        className="flex h-5 w-5 items-center justify-center text-m-text-muted"
      >
        <EllipsisVertical size={20} strokeWidth={1.5} aria-hidden="true" />
      </button>
    ),
  },
};

/**
 * Mounts both desktop and mobile chromes simultaneously and gates
 * visibility with the CSS-only `.d-only` / `.m-only` classes. No JS
 * branching for the shell — at any viewport width exactly one chrome
 * is `display: block`, the other is `display: none`. Mobile chrome
 * (tape pill, header, bottom nav) is always provided here; pages
 * supply only their content as `children`.
 */
export function AdaptiveShell({ children }: { children: ReactNode }) {
  const pathname = usePathname() ?? "";
  const navLabel = getNavItemForHref(pathname)?.label ?? fallbackLabel(pathname);
  const header: MobilePageHeaderProps = {
    ...splitWordmark(navLabel),
    ...ROUTE_HEADER_OVERRIDES[pathname],
  };

  return (
    <>
      <div className="d-only">
        <DesktopShell>{children}</DesktopShell>
      </div>
      <div className="m-only">
        <MobileShell header={header}>{children}</MobileShell>
      </div>
    </>
  );
}
