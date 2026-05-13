// Tag color palette — Phase 1 (Weekly Retro tags).
//
// Five tones lifted verbatim from
// Design/design_handoff_weekly_retro/design/sections-top.jsx. Each pill
// body uses color-mix at 14%/30% against --surface/--border, mirroring
// the existing ticker B/S badge pattern in src/components/weekly-retro.tsx.
//
// Backend stores the palette KEY as the tag color (rose|amber|emerald|sky|
// violet) and validates against this same closed set. Adding a tone here
// requires also extending the backend's _TAG_COLOR_VOCAB in db_layer.py and
// _TAG_VALID_COLORS in api/main.py. Keep them in lockstep.

export const TAG_PALETTE = {
  rose:    { dot: "#f43f5e", body: "color-mix(in oklab, #f43f5e 14%, var(--surface))", text: "#be123c", ring: "color-mix(in oklab, #f43f5e 30%, var(--border))" },
  amber:   { dot: "#f59f00", body: "color-mix(in oklab, #f59f00 14%, var(--surface))", text: "#b45309", ring: "color-mix(in oklab, #f59f00 30%, var(--border))" },
  emerald: { dot: "#08a86b", body: "color-mix(in oklab, #08a86b 14%, var(--surface))", text: "#047857", ring: "color-mix(in oklab, #08a86b 30%, var(--border))" },
  sky:     { dot: "#0d6efd", body: "color-mix(in oklab, #0d6efd 14%, var(--surface))", text: "#1d4ed8", ring: "color-mix(in oklab, #0d6efd 30%, var(--border))" },
  violet:  { dot: "#8b5cf6", body: "color-mix(in oklab, #8b5cf6 14%, var(--surface))", text: "#6d28d9", ring: "color-mix(in oklab, #8b5cf6 30%, var(--border))" },
} as const;

export type TagTone = keyof typeof TAG_PALETTE;

export const TAG_TONES: TagTone[] = ["rose", "amber", "emerald", "sky", "violet"];
