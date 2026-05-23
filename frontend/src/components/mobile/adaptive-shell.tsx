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
 * Mounts both desktop and mobile chromes simultaneously and gates
 * visibility with the CSS-only `.d-only` / `.m-only` classes. No JS
 * branching for the shell — at any viewport width exactly one chrome
 * is `display: block`, the other is `display: none`. Mobile chrome
 * (tape pill, header with portfolio picker, bottom nav) is always
 * provided here; pages supply only their content as `children`.
 *
 * Phase 2 step 1 removed the per-route header override table (back
 * chevron + per-page right icons were Phase 1 visual stubs with no
 * behavior). The portfolio picker now occupies the right slot
 * globally via `MobileShell`'s default.
 */
export function AdaptiveShell({ children }: { children: ReactNode }) {
  const pathname = usePathname() ?? "";
  const navLabel = getNavItemForHref(pathname)?.label ?? fallbackLabel(pathname);
  const header = splitWordmark(navLabel);

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
