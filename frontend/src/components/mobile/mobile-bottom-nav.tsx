"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BookOpen,
  CalendarDays,
  Calculator,
  ChartLine,
  Ellipsis,
  type LucideIcon,
} from "lucide-react";

type NavItem = {
  href: string;
  label: string;
  Icon: LucideIcon;
  /** Optional extra pathname prefixes that should also light this tab
   *  as active. The default `pathname === href || startsWith(href + "/")`
   *  predicate covers self + descendants; this adds sibling routes
   *  for tabs whose conceptual surface spans multiple top-level paths. */
  extraActivePrefixes?: readonly string[];
};

const ITEMS: readonly NavItem[] = [
  { href: "/dashboard", label: "Dashboard", Icon: ChartLine },
  { href: "/position-sizer", label: "Sizer", Icon: Calculator },
  { href: "/trade-journal", label: "Journal", Icon: BookOpen },
  {
    href: "/daily-journal",
    label: "Daily",
    Icon: CalendarDays,
    // The Daily tab is the conceptual home of the daily-workflow
    // surface, so it stays active when the user navigates to the
    // quick-link destinations (Weekly Retro, Daily Routine) and to
    // the Daily Report detail view (queried as /daily-report?date=...).
    extraActivePrefixes: ["/daily-routine", "/weekly-retro", "/daily-report"],
  },
  { href: "/more", label: "More", Icon: Ellipsis },
] as const;

/**
 * Sticky bottom navigation for mobile. Five destinations, lucide icons,
 * 44×44 minimum touch targets, no hover effects (active state on tap
 * only). Active item is the one whose `href` matches the current
 * pathname or is a strict prefix — plus any `extraActivePrefixes`
 * declared on the item (used by the Daily tab to span the daily-
 * workflow surface across /daily-journal, /daily-routine, /weekly-
 * retro, and /daily-report).
 *
 * Cycle was dropped in Phase 2 Step 2 — the M Factor state now rides on
 * the global tape pill (visible on every mobile route), so a dedicated
 * destination duplicated the indicator without adding unique content.
 *
 * Visible only on mobile viewports — wrapped in `.m-only` by the shell
 * is unnecessary because `MobileBottomNav` is only mounted from inside
 * `MobileShell`, and `MobileShell` itself is on the mobile branch of
 * the AdaptiveShell.
 */
export function MobileBottomNav() {
  const pathname = usePathname() ?? "";

  return (
    <nav
      className="flex justify-around border-t border-m-border bg-m-bg px-4 pt-1.5 pb-5"
      aria-label="Primary"
    >
      {ITEMS.map(({ href, label, Icon, extraActivePrefixes }) => {
        const baseActive =
          pathname === href || pathname.startsWith(href + "/");
        const extraActive = extraActivePrefixes?.some(
          (p) => pathname === p || pathname.startsWith(p + "/"),
        );
        const active = baseActive || !!extraActive;
        const tone = active ? "text-m-accent" : "text-m-text-faint";
        return (
          <Link
            key={href}
            href={href}
            aria-label={label}
            aria-current={active ? "page" : undefined}
            className={`flex min-h-[44px] min-w-[44px] flex-col items-center justify-center gap-0.5 px-1 ${tone}`}
          >
            <Icon size={22} strokeWidth={1.5} aria-hidden="true" />
            <span className={`text-[11px] ${active ? "font-medium" : ""}`}>
              {label}
            </span>
          </Link>
        );
      })}
    </nav>
  );
}
