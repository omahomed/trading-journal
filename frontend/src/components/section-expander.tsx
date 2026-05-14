"use client";

// Shared collapsible-section chrome — the card pattern adopted by Weekly
// Thoughts (Phase 3) and now Per-Ticker Details (Phase 2, migrated as
// part of this refactor) and Weekly Snapshot (Phase 4, upcoming).
//
// Canonical visual: a 14px-radius card with `var(--card-shadow)`, an
// integrated header button that toggles a body sitting flush inside the
// card. Header layout: chevron + (optional 6×6 orange dot) + title +
// right-aligned caption.
//
// What this component owns:
//   - Card chrome (border, radius, shadow, marginBottom)
//   - Header button (chevron rotation, dot, title, caption)
//   - localStorage-persisted expand/collapse state, with a defaultExpanded
//     fallback and SSR-safe lazy initializer
//   - aria-expanded / aria-controls wiring (bodyId auto-derived from a
//     sanitized title when not provided)
//   - Body unmounting when collapsed (NOT just hidden) — matches both
//     migrated call sites' existing behavior
//
// What this component does NOT own:
//   - Internal body padding. Each call site owns whatever padding /
//     internal layout it needs inside the children block. Keeps the
//     component chrome-only and avoids prescribing layout for the
//     children's content.

import { useCallback, useState } from "react";
import { Icons } from "./icons";

interface SectionExpanderProps {
  title: string;
  defaultExpanded: boolean;
  localStorageKey: string;
  /** Render the 6×6 amber (#f59f00) dot before the title. Default false. */
  showDot?: boolean;
  /** Override the body element's id; used for aria-controls. When omitted,
   *  a stable id is derived from the title. */
  bodyId?: string;
  /** Render the right-aligned header caption. Called with the current
   *  expanded state so callers can vary copy by state (or return null to
   *  suppress conditionally). */
  headerCaption?: (expanded: boolean) => React.ReactNode;
  children: React.ReactNode;
}

// Sanitize a title into a stable, valid HTML id.
function deriveBodyId(title: string): string {
  return (
    "section-" +
    title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
  );
}

export function SectionExpander({
  title,
  defaultExpanded,
  localStorageKey,
  showDot = false,
  bodyId,
  headerCaption,
  children,
}: SectionExpanderProps) {
  // SSR-safe lazy initializer matching the pattern Weekly Thoughts and
  // Per-Ticker both used pre-refactor: try localStorage, fall back to
  // defaultExpanded on read error (incognito, missing key, etc.).
  const [expanded, setExpanded] = useState<boolean>(() => {
    try {
      const stored = localStorage.getItem(localStorageKey);
      return stored == null ? defaultExpanded : stored === "true";
    } catch {
      return defaultExpanded;
    }
  });

  const toggle = useCallback(() => {
    setExpanded(prev => {
      const next = !prev;
      try { localStorage.setItem(localStorageKey, String(next)); }
      catch { /* incognito quota — UI still works, just not persisted */ }
      return next;
    });
  }, [localStorageKey]);

  const resolvedBodyId = bodyId || deriveBodyId(title);

  return (
    <div
      style={{
        borderRadius: 14,
        overflow: "hidden",
        marginBottom: 24,
        background: "var(--surface)",
        border: "1px solid var(--border)",
        boxShadow: "var(--card-shadow)",
      }}
    >
      <button
        type="button"
        onClick={toggle}
        aria-expanded={expanded}
        aria-controls={resolvedBodyId}
        className="w-full flex items-center text-left"
        style={{
          gap: 10,
          padding: "12px 18px",
          background: "transparent",
          border: "none",
          borderBottom: expanded ? "1px solid var(--border)" : "none",
          cursor: "pointer",
        }}
      >
        <span
          aria-hidden
          style={{
            display: "inline-flex",
            transition: "transform 150ms",
            transform: expanded ? "rotate(90deg)" : "none",
            color: "var(--ink-4)",
          }}
        >
          <Icons.chevronRight />
        </span>
        {showDot && (
          <span
            aria-hidden
            data-testid="section-expander-dot"
            style={{
              width: 6,
              height: 6,
              borderRadius: 999,
              background: "#f59f00",
            }}
          />
        )}
        <span style={{ fontSize: 13, fontWeight: 600 }}>{title}</span>
        <span style={{ flex: 1 }} />
        {headerCaption && (
          <span style={{ fontSize: 11, color: "var(--ink-4)" }}>
            {headerCaption(expanded)}
          </span>
        )}
      </button>

      {expanded && (
        <div id={resolvedBodyId}>
          {children}
        </div>
      )}
    </div>
  );
}
