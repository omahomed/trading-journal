// Visual-only tag pill. Matches the design at
// Design/design_handoff_weekly_retro/design/sections-top.jsx (TagPill).
//
// Phase 1: TagPicker uses this in interactive mode (with onRemove). The
// component is exported separately so a future rail filter (Phase 6) can
// render read-only pills by omitting onRemove.

import { TAG_PALETTE, type TagTone } from "@/lib/tag-palette";
import { Icons } from "./icons";

interface TagPillProps {
  label: string;
  tone: TagTone;
  onRemove?: () => void;
}

export function TagPill({ label, tone, onRemove }: TagPillProps) {
  // Defensive fallback. Backend validates color against the closed palette,
  // so an unknown tone here is always a frontend bug — but if it ever ships,
  // sky is the least visually surprising default.
  const t = TAG_PALETTE[tone] ?? TAG_PALETTE.sky;
  return (
    <span
      className="inline-flex items-center"
      style={{
        gap: 6,
        height: 24,
        padding: "0 8px",
        borderRadius: 999,
        background: t.body,
        border: `1px solid ${t.ring}`,
        color: t.text,
        fontSize: 11,
        fontWeight: 600,
      }}
    >
      <span
        aria-hidden
        style={{ width: 6, height: 6, borderRadius: 999, background: t.dot }}
      />
      {label}
      {onRemove && (
        <button
          type="button"
          onClick={onRemove}
          aria-label={`Remove ${label}`}
          className="grid place-items-center"
          style={{
            width: 14,
            height: 14,
            borderRadius: 999,
            color: t.text,
            opacity: 0.7,
            background: "transparent",
            border: "none",
            cursor: "pointer",
          }}
        >
          <Icons.x />
        </button>
      )}
    </span>
  );
}
