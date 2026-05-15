"use client";

// Phase 7 — thin wrapper around the shared <ThoughtsEditor>. The body of
// the editor (toolbar, contentEditable, paste sanitization, image
// upload, table tools, etc.) lives in thoughts-editor.tsx; this file
// binds the weekly-retro presentation (title, storage key, captions,
// upload endpoint via entityType="weekly_retro") so existing callsites
// and tests continue to work without modification.
//
// The component's public API is unchanged from Phase 4.3:
//   <WeeklyThoughts
//     value={...}
//     onChange={...}
//     retroId={...}
//     portfolio={...} />
// — matched here byte-for-byte by name so weekly-retro.tsx and
// weekly-thoughts.test.tsx don't need any code edits.

import { ThoughtsEditor } from "./thoughts-editor";

// Re-export the named utility helpers that weekly-thoughts.test.tsx
// imports alongside the component. Same functions, just sourced from
// the renamed module — no behavior change.
export { normalizeIndentation, parseVideoUrl } from "./thoughts-editor";

export interface WeeklyThoughtsProps {
  value: string;
  onChange: (html: string) => void;
  /** Required for inline image paste. When null, image-paste attempts
   *  surface a friendly "save the retro first" inline error. */
  retroId?: number | null;
  /** Portfolio name forwarded to the upload endpoint. */
  portfolio?: string;
}

export function WeeklyThoughts({
  value,
  onChange,
  retroId = null,
  portfolio = "",
}: WeeklyThoughtsProps) {
  return (
    <ThoughtsEditor
      value={value}
      onChange={onChange}
      entityType="weekly_retro"
      entityId={retroId}
      portfolio={portfolio}
      title="Weekly Thoughts"
      localStorageKey="mo-weekly-retro-thoughts-expanded"
      bodyId="weekly-thoughts-body"
      expandedCaption="Synthesize what you learned this week"
      placeholderText="What did you learn this week? Patterns, mistakes, rule refinements…"
      ariaLabel="Weekly Thoughts"
      toolbarAriaLabel="Weekly Thoughts formatting"
      noEntityErrorMessage="Save the retro first to embed images."
    />
  );
}
