"use client";

// Thin wrapper around the shared <ThoughtsEditor> — the weekend Watch List
// section of Weekly Retro. Mirrors WeeklyThoughts (see weekly-thoughts.tsx)
// so the toolbar, contentEditable, paste sanitization, image upload, table
// tools, etc. all behave identically. Only the title, storage key, ARIA
// labels, and placeholder text differ.
//
// Images pasted here upload to the SAME R2 prefix as Weekly Thoughts
// (weekly_retros/{retro_id}/thoughts/…) — both editors are attachments of
// the same weekly_retros row, so entityType stays "weekly_retro". Cleanup
// scans the concatenation of weekly_thoughts + watch_list HTML.

import { ThoughtsEditor } from "./thoughts-editor";

export interface WatchListProps {
  value: string;
  onChange: (html: string) => void;
  /** Required for inline image paste. When null, image-paste attempts
   *  surface a friendly "save the retro first" inline error. */
  retroId?: number | null;
  /** Portfolio name forwarded to the upload endpoint. */
  portfolio?: string;
}

export function WatchList({
  value,
  onChange,
  retroId = null,
  portfolio = "",
}: WatchListProps) {
  return (
    <ThoughtsEditor
      value={value}
      onChange={onChange}
      entityType="weekly_retro"
      entityId={retroId}
      portfolio={portfolio}
      title="Watch List"
      localStorageKey="mo-weekly-retro-watch-list-expanded"
      bodyId="weekly-watch-list-body"
      expandedCaption="IBD 50 · Growth 250 · Big Cap 20 — screens, tickers, charts"
      placeholderText="Paste screens, notes, or charts…"
      ariaLabel="Watch List"
      toolbarAriaLabel="Watch List formatting"
      noEntityErrorMessage="Save the retro first to embed images."
    />
  );
}
