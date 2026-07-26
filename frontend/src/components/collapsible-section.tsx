"use client";

// Small collapsible card used by the merged Daily Routine to keep the
// page scannable when Positions Opened/Closed or Daily Recap get long.
// Header shows the section label + a count preview + a chevron; the
// body only mounts when open. Auto-open state seeded once from
// `defaultOpen`; the user can toggle from there.
//
// Persistence intentionally omitted — we tried it in daily-thoughts and
// hit "state didn't apply on first paint" issues. The auto-open rule
// (e.g. "open when count ≤ 3") is a reasonable default; user toggles
// live for the session only, which is fine for a page you visit once
// a day.

import { useState, type ReactNode } from "react";

export interface CollapsibleSectionProps {
  title: string;
  /** Small right-aligned meta text — e.g. "3 positions" or "1267 words". */
  meta?: string;
  /** Whether the section is open on first mount. Callers compute this
   *  from data (e.g. `positions.length <= 3`). */
  defaultOpen: boolean;
  children: ReactNode;
  /** Data-testid on the outer wrapper so tests can target it. */
  testId?: string;
}

export function CollapsibleSection({
  title,
  meta,
  defaultOpen,
  children,
  testId,
}: CollapsibleSectionProps) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="rounded-[14px] overflow-hidden"
         style={{ background: "var(--surface)", border: "1px solid var(--border)" }}
         data-testid={testId}>
      <button type="button"
              onClick={() => setOpen(o => !o)}
              className="w-full px-4 py-3 flex items-center justify-between text-left transition-colors hover:brightness-95"
              style={{ background: "transparent", color: "var(--ink-1)" }}
              aria-expanded={open}>
        <span className="text-[13px] font-semibold">{title}</span>
        <span className="flex items-center gap-2 text-[12px]" style={{ color: "var(--ink-4)" }}>
          {meta && <span>{meta}</span>}
          <span style={{ transform: open ? "rotate(90deg)" : "none", transition: "transform 120ms" }}>›</span>
        </span>
      </button>
      {open && (
        <div style={{ borderTop: "1px solid var(--border)" }}>
          {children}
        </div>
      )}
    </div>
  );
}
