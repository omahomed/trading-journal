"use client";

import type { ReactNode } from "react";
import { usePathname } from "next/navigation";
import { DesktopShell } from "@/components/desktop-shell";
import { MobileShell } from "./mobile-shell";
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
 * Mounts both desktop and mobile chromes simultaneously and gates
 * visibility with the CSS-only `.d-only` / `.m-only` classes from
 * `mobile-tokens.css`. No JS branching for the shell — at any viewport
 * width exactly one chrome is `display: block`, the other is
 * `display: none`. The wrapping divs add a DOM node above each shell
 * but contribute no layout (no width/height/margin/padding); the inner
 * shell's own `flex h-screen` (desktop) and `h-[100dvh] flex flex-col`
 * (mobile) drive the page geometry exactly as before.
 *
 * Mobile header config is derived from `getNavItemForHref(pathname)`
 * — the last word of the nav label becomes the italic-green wordmark.
 * Routes without a nav-table entry (e.g. dynamic dev routes) render
 * with an empty wordmark; pages that have a custom mobile design will
 * carry their own chrome from Step 6 onward (this AdaptiveShell only
 * provides the default wrapper).
 */
export function AdaptiveShell({ children }: { children: ReactNode }) {
  const pathname = usePathname() ?? "";
  const label = getNavItemForHref(pathname)?.label ?? "";
  const header = splitWordmark(label);

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
