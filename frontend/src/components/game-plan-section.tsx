"use client";

// Game Plan section for the Daily Journal shell (migration 052). Two
// visual modes gated by `editable`:
//
//   editable=true  → renders the shared <ThoughtsEditor> (same rich-text
//                    surface as Daily Thoughts / Weekly Thoughts). Auto-
//                    saves via the parent's debounced effect; this
//                    component owns none of that logic.
//   editable=false → read-only display. The SectionExpander chrome mirrors
//                    the editor's shell so collapse/expand + storage key
//                    stay coherent across day-to-day navigation; the body
//                    renders sanitized HTML via dangerouslySetInnerHTML.
//
// The HTML stored in game_plan comes from ThoughtsEditor's sanitize-on-
// paste pass (see thoughts-editor.tsx PASTE_ALLOWED_TAGS / ATTR), so we
// trust the value on render — same trust model as Daily Thoughts / Weekly
// Thoughts read-back paths (which also render user-authored HTML through
// the editor's contentEditable region).

import { SectionExpander } from "./section-expander";
import { ThoughtsEditor } from "./thoughts-editor";

const STORAGE_KEY = "mo-daily-journal-game-plan-expanded";
const BODY_ID = "game-plan-body";

export interface GamePlanSectionProps {
  value: string;
  onChange: (html: string) => void;
  /** True while today (America/Chicago) < the lock date for this row's
   *  day. False otherwise. Same rule mirrored on the server; see
   *  api/main.py:_game_plan_lock_date and lib/game-plan.ts. */
  editable: boolean;
  /** Daily-journal row PK for inline image uploads. Optional — null shows
   *  the shared "save first" inline error inside the editor. */
  journalId?: number | null;
  /** Portfolio name forwarded to the image upload endpoint. */
  portfolio?: string;
}

/** Word count from an HTML string. Strips tags, collapses whitespace,
 *  splits on any run of whitespace. Same shape as the caption on Daily
 *  Recap so the two sections read consistently. */
function countWords(html: string): number {
  const stripped = html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
  return stripped ? stripped.split(/\s+/).length : 0;
}

export function GamePlanSection({
  value,
  onChange,
  editable,
  journalId = null,
  portfolio = "",
}: GamePlanSectionProps) {
  if (editable) {
    return (
      <ThoughtsEditor
        value={value}
        onChange={onChange}
        entityType="daily_journal"
        entityId={journalId}
        portfolio={portfolio}
        title="Game Plan"
        localStorageKey={STORAGE_KEY}
        bodyId={BODY_ID}
        expandedCaption="What am I doing tomorrow?"
        placeholderText="Positions to watch, planned buys/trims — or 'no action, monitoring only.'"
        ariaLabel="Game Plan"
        toolbarAriaLabel="Game Plan formatting"
        noEntityErrorMessage="Save the game plan first to embed images."
      />
    );
  }
  // Locked read-only mode. Mirror the editor's SectionExpander chrome so
  // the header state (title, dot, caption shape) stays consistent when
  // the same page toggles from editable to locked across days.
  const trimmed = (value || "").trim();
  const words = countWords(value || "");
  return (
    <SectionExpander
      title="Game Plan"
      showDot
      defaultExpanded={!!trimmed}
      localStorageKey={STORAGE_KEY}
      bodyId={BODY_ID}
      headerCaption={(open) =>
        open
          ? "🔒 locked"
          : trimmed
            ? `${words} words · locked`
            : "no plan · locked"
      }
    >
      <div className="p-4">
        {trimmed ? (
          <div
            data-testid="game-plan-locked-body"
            className="px-4 py-3 rounded-[10px] text-[13px] prose-custom"
            style={{
              background: "var(--bg)",
              border: "1px solid var(--border)",
              color: "var(--ink)",
              lineHeight: 1.6,
            }}
            dangerouslySetInnerHTML={{ __html: value }}
          />
        ) : (
          <div
            data-testid="game-plan-locked-empty"
            className="px-4 py-3 rounded-[10px] text-[13px]"
            style={{
              background: "var(--bg)",
              border: "1px solid var(--border)",
              color: "var(--ink-4)",
              fontStyle: "italic",
              lineHeight: 1.6,
            }}
          >
            No plan was set for this day.
          </div>
        )}
      </div>
    </SectionExpander>
  );
}
