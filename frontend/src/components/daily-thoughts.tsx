"use client";

// Phase 7 — Daily Thoughts editor for the Daily Report page. Thin
// wrapper around the shared <ThoughtsEditor> from thoughts-editor.tsx.
// Identical behavior to Weekly Thoughts (toolbar, paste sanitization,
// inline image embed, etc.) — only the title, expanded caption,
// storage key, and upload endpoint (via entityType) differ.

import { ThoughtsEditor } from "./thoughts-editor";

export interface DailyThoughtsProps {
  value: string;
  onChange: (html: string) => void;
  /** The daily journal row's PK from /api/journal/history. Required for
   *  inline image paste; null shows a "save the journal entry first"
   *  inline error matching the weekly disabled-state idiom. */
  journalId?: number | null;
  /** Portfolio name forwarded to the upload endpoint. */
  portfolio?: string;
}

export function DailyThoughts({
  value,
  onChange,
  journalId = null,
  portfolio = "",
}: DailyThoughtsProps) {
  return (
    <ThoughtsEditor
      value={value}
      onChange={onChange}
      entityType="daily_journal"
      entityId={journalId}
      portfolio={portfolio}
      title="Daily Thoughts"
      localStorageKey="mo-daily-report-thoughts-expanded"
      bodyId="daily-thoughts-body"
      expandedCaption="What did you observe today?"
      placeholderText="What did you observe today? Trades, market behavior, decisions made…"
      ariaLabel="Daily Thoughts"
      toolbarAriaLabel="Daily Thoughts formatting"
      noEntityErrorMessage="Save the journal entry first to embed images."
    />
  );
}
