"use client";

// Phase 1 entity-agnostic tag picker.
//
// Mounted on Weekly Retro this phase. Daily journal (Phase 7) and trade
// summaries (Phase 8) will mount this same component with different
// entity_type / entity_id values; no per-mount-site logic lives here.
//
// Behavior:
//   - When entityId is null → disabled state with hover-tooltip ("Save the
//     retro first to add tags"). The Weekly Retro mount passes null until
//     the parent retro has been saved at least once (no SERIAL id yet).
//   - When entityId is set → fetches tags + assignments on mount, renders
//     pills for the intersection, plus a "+ Add tag" affordance that opens
//     a dropdown with autocomplete + filtered list + "+ Create '{q}'" row.
//   - All mutations are optimistic with revert-on-error and a saveMsg-style
//     error toast (auto-dismissed after 2s). Matches the inline-toast
//     pattern from weekly-retro.tsx (no shared <Toast> primitive yet).
//   - Hard cap of 10 tags per entity, mirroring the API enforcement so we
//     can disable the affordance with a tooltip before the round-trip.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type Tag, type TagAssignment } from "@/lib/api";
import { TAG_TONES, type TagTone } from "@/lib/tag-palette";
import { TagPill } from "./tag-pill";
import { Icons } from "./icons";
import { log } from "@/lib/log";

const MAX_TAGS_PER_ENTITY = 10;

type EntityType = "weekly_retro" | "daily_journal" | "trades_summary";

interface TagPickerProps {
  entityType: EntityType;
  entityId: number | null;
  portfolio: string;
  /** Phase 8 — fires after the server confirms a tag mutation
   *  (assign / create+assign / detach / palette delete). NOT fired on
   *  optimistic update, NOT fired on error paths. Optional — consumers
   *  that don't care (e.g., the trades_summary mount that has no rail)
   *  omit it and behavior is unchanged. */
  onTagsChanged?: () => void;
}

export function TagPicker({ entityType, entityId, portfolio, onTagsChanged }: TagPickerProps) {
  const [tags, setTags] = useState<Tag[]>([]);
  const [assignments, setAssignments] = useState<TagAssignment[]>([]);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [selectedTone, setSelectedTone] = useState<TagTone>("sky");
  const [error, setError] = useState("");
  const [creating, setCreating] = useState(false);

  // Tag-palette delete affordance. armedDeleteTagId tracks which row (if
  // any) is in its second-click "Confirm" state — single-value so only one
  // row can be armed at a time. hoveredTagId drives trash-icon visibility
  // (React-tracked rather than CSS-only to match this file's onMouseEnter/
  // Leave style; see follow-up audit for rationale).
  const [armedDeleteTagId, setArmedDeleteTagId] = useState<number | null>(null);
  const [hoveredTagId, setHoveredTagId] = useState<number | null>(null);

  const wrapperRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const showError = useCallback((msg: string) => {
    setError(msg);
    window.setTimeout(() => setError(""), 2000);
  }, []);

  // Initial fetch — tags (per portfolio) + assignments (per entity). Both
  // independent; Promise.all lets them race. Failures log to console.error
  // via the log wrapper; no UI surface for tag errors today.
  useEffect(() => {
    if (entityId == null || !portfolio) return;
    let cancelled = false;
    Promise.all([
      api.listTags(portfolio).catch((err) => {
        log.error("tag-picker", "listTags fetch failed", err);
        return [] as Tag[];
      }),
      api.listTagAssignments({ entity_type: entityType, entity_id: entityId })
        .catch((err) => {
          log.error("tag-picker", "listTagAssignments fetch failed", err);
          return [] as TagAssignment[];
        }),
    ]).then(([t, a]) => {
      if (cancelled) return;
      setTags(t);
      setAssignments(a);
    });
    return () => { cancelled = true; };
  }, [entityType, entityId, portfolio]);

  // Click-outside-to-close. Pattern copied from log-buy.tsx SearchSelect
  // (mousedown listener + ref containment check).
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (!wrapperRef.current) return;
      if (!wrapperRef.current.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("mousedown", handler);
    return () => window.removeEventListener("mousedown", handler);
  }, [open]);

  // Autofocus the autocomplete input when the dropdown opens.
  useEffect(() => {
    if (open) inputRef.current?.focus();
    else { setQuery(""); setSelectedTone("sky"); }
  }, [open]);

  const assignedTagIds = useMemo(
    () => new Set(assignments.map(a => a.tag_id)),
    [assignments],
  );

  const unassignedTags = useMemo(
    () => tags.filter(t => !assignedTagIds.has(t.id)),
    [tags, assignedTagIds],
  );

  const filteredUnassigned = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return unassignedTags;
    return unassignedTags.filter(t => t.name.toLowerCase().includes(q));
  }, [unassignedTags, query]);

  const exactMatch = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return null;
    // Exact (case-insensitive) match across ALL tags — assigned or not — so
    // typing a name that's already attached doesn't show "+ Create".
    return tags.find(t => t.name.toLowerCase() === q) ?? null;
  }, [tags, query]);

  const atCap = assignments.length >= MAX_TAGS_PER_ENTITY;

  const handleAssign = useCallback(async (tag: Tag) => {
    if (atCap) { showError("Maximum 10 tags per entry"); return; }
    if (assignedTagIds.has(tag.id)) return;  // no-op if already attached

    // Optimistic add via temp negative id; replace with server's row on success.
    const tempId = -Date.now();
    const optimistic: TagAssignment = {
      id: tempId,
      tag_id: tag.id,
      tag_name: tag.name,
      tag_color: tag.color,
      entity_type: entityType,
      entity_id: entityId as number,
      created_at: new Date().toISOString(),
    };
    setAssignments(prev => [...prev, optimistic]);
    setOpen(false);

    try {
      const result = await api.createTagAssignment({
        tag_id: tag.id,
        entity_type: entityType,
        entity_id: entityId as number,
      });
      if ("error" in result) throw new Error(result.error);
      setAssignments(prev => prev.map(a => a.id === tempId ? result : a));
      onTagsChanged?.();
    } catch (e) {
      setAssignments(prev => prev.filter(a => a.id !== tempId));
      const msg = e instanceof Error ? e.message : "Failed to add tag";
      showError(msg === "tag_limit_reached" ? "Maximum 10 tags per entry" : msg);
    }
  }, [atCap, assignedTagIds, entityType, entityId, showError, onTagsChanged]);

  const handleCreateAndAssign = useCallback(async (name: string, tone: TagTone) => {
    if (atCap) { showError("Maximum 10 tags per entry"); return; }
    const cleaned = name.trim();
    if (!cleaned) return;
    setCreating(true);

    // Optimistic add of both the tag and its assignment.
    const tempTagId = -Date.now();
    const tempAssignmentId = tempTagId - 1;
    const optimisticTag: Tag = {
      id: tempTagId,
      portfolio,
      name: cleaned,
      color: tone,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
    const optimisticAssignment: TagAssignment = {
      id: tempAssignmentId,
      tag_id: tempTagId,
      tag_name: cleaned,
      tag_color: tone,
      entity_type: entityType,
      entity_id: entityId as number,
      created_at: new Date().toISOString(),
    };
    setTags(prev => [...prev, optimisticTag]);
    setAssignments(prev => [...prev, optimisticAssignment]);
    setOpen(false);

    try {
      const tagResult = await api.createTag({ portfolio, name: cleaned, color: tone });
      if ("error" in tagResult) {
        const msg = tagResult.error === "tag_name_exists"
          ? "A tag with that name already exists"
          : tagResult.error;
        throw new Error(msg);
      }
      // Replace the optimistic temp tag with the real server row.
      setTags(prev => prev.map(t => t.id === tempTagId ? tagResult : t));

      const assignResult = await api.createTagAssignment({
        tag_id: tagResult.id,
        entity_type: entityType,
        entity_id: entityId as number,
      });
      if ("error" in assignResult) throw new Error(assignResult.error);
      setAssignments(prev => prev.map(
        a => a.id === tempAssignmentId ? assignResult : a,
      ));
      onTagsChanged?.();
    } catch (e) {
      // Roll back both optimistic rows.
      setTags(prev => prev.filter(t => t.id !== tempTagId));
      setAssignments(prev => prev.filter(a => a.id !== tempAssignmentId));
      const msg = e instanceof Error ? e.message : "Failed to create tag";
      showError(msg === "tag_limit_reached" ? "Maximum 10 tags per entry" : msg);
    } finally {
      setCreating(false);
    }
  }, [atCap, portfolio, entityType, entityId, showError, onTagsChanged]);

  const handleDetach = useCallback(async (assignment: TagAssignment) => {
    const removed = assignment;
    setAssignments(prev => prev.filter(a => a.id !== removed.id));
    try {
      const result = await api.deleteTagAssignment(removed.id);
      if ("error" in result) throw new Error(result.error);
      onTagsChanged?.();
    } catch (e) {
      // Restore on failure.
      setAssignments(prev => [...prev, removed]);
      const msg = e instanceof Error ? e.message : "Failed to remove tag";
      showError(msg);
    }
  }, [showError, onTagsChanged]);

  // Permanently delete a tag from the user's palette. Different from
  // detach: this removes the tag itself (soft-deletes server-side via
  // DELETE /api/tags/{id}), which also hides every assignment that points
  // at it from every entity. The server-side LEFT JOIN by tags.deleted_at
  // IS NULL handles the visibility; locally we mirror by filtering both
  // tags and assignments slots.
  //
  // Snapshot-restore covers BOTH state slots: if the API call fails, the
  // tag and any matching assignments come back. assignments must be in the
  // deps array so the snapshot captures the current set, not stale.
  const handleDeleteTag = useCallback(async (tag: Tag) => {
    const tagSnapshot = tag;
    const assignmentSnapshot = assignments.filter(a => a.tag_id === tag.id);
    setTags(prev => prev.filter(t => t.id !== tag.id));
    setAssignments(prev => prev.filter(a => a.tag_id !== tag.id));
    setArmedDeleteTagId(null);
    try {
      const result = await api.deleteTag(tag.id);
      if ("error" in result) throw new Error(result.error);
      // Palette delete removes the tag from EVERY entity it was attached
      // to. Rail filter bar should reflect the change too.
      onTagsChanged?.();
    } catch (e) {
      setTags(prev => [...prev, tagSnapshot]);
      setAssignments(prev => [...prev, ...assignmentSnapshot]);
      const msg = e instanceof Error ? e.message : "Failed to delete tag";
      showError(`Couldn't delete tag: ${msg}`);
    }
  }, [assignments, showError, onTagsChanged]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Escape") {
      e.preventDefault();
      setOpen(false);
      return;
    }
    if (e.key === "Enter") {
      e.preventDefault();
      const trimmed = query.trim();
      if (!trimmed) return;
      if (exactMatch) {
        if (assignedTagIds.has(exactMatch.id)) {
          // Already attached — close without re-assigning.
          setOpen(false);
        } else {
          void handleAssign(exactMatch);
        }
      } else {
        void handleCreateAndAssign(trimmed, selectedTone);
      }
    }
  }, [query, exactMatch, assignedTagIds, selectedTone, handleAssign, handleCreateAndAssign]);

  // Disabled-empty state (entity not yet saved). The container is still
  // present so the layout doesn't shift when the entity gets saved.
  if (entityId == null) {
    return (
      <div
        className="flex items-center flex-wrap"
        style={{ gap: 6, marginTop: 12 }}
      >
        <span
          title="Save the retro first to add tags"
          className="inline-flex items-center"
          style={{
            gap: 4,
            height: 24,
            padding: "0 10px 0 8px",
            borderRadius: 999,
            fontSize: 11,
            fontWeight: 600,
            background: "var(--surface)",
            border: "1px dashed var(--border-2)",
            color: "var(--ink-4)",
            opacity: 0.6,
            cursor: "not-allowed",
          }}
        >
          <Icons.plus />
          Add tag
        </span>
      </div>
    );
  }

  return (
    <div ref={wrapperRef} className="relative" style={{ marginTop: 12 }}>
      <div className="flex items-center flex-wrap" style={{ gap: 6 }}>
        {assignments.map(a => (
          <TagPill
            key={a.id}
            label={a.tag_name}
            tone={a.tag_color as TagTone}
            onRemove={() => void handleDetach(a)}
          />
        ))}

        <button
          type="button"
          onClick={() => { if (!atCap) setOpen(prev => !prev); else showError("Maximum 10 tags per entry"); }}
          aria-label="Add tag"
          title={atCap ? "Maximum 10 tags per entry" : "Add a tag"}
          className="inline-flex items-center"
          style={{
            gap: 4,
            height: 24,
            padding: "0 10px 0 8px",
            borderRadius: 999,
            fontSize: 11,
            fontWeight: 600,
            background: "var(--surface)",
            border: "1px dashed var(--border-2)",
            color: atCap ? "var(--ink-4)" : "var(--ink-3)",
            opacity: atCap ? 0.6 : 1,
            cursor: atCap ? "not-allowed" : "pointer",
          }}
        >
          <Icons.plus />
          Add tag
        </button>
      </div>

      {open && (
        <div
          role="listbox"
          className="absolute z-20 mt-1.5"
          style={{
            top: "100%",
            left: 0,
            minWidth: 240,
            maxWidth: 320,
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderRadius: 10,
            boxShadow: "var(--card-shadow)",
            padding: 6,
          }}
        >
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search or create tag…"
            className="w-full"
            style={{
              height: 30,
              padding: "0 8px",
              fontSize: 12,
              background: "var(--bg)",
              border: "1px solid var(--border)",
              borderRadius: 6,
              color: "var(--ink)",
              outline: "none",
            }}
          />

          <div
            className="mt-1"
            style={{ maxHeight: 220, overflowY: "auto" }}
          >
            {filteredUnassigned.length === 0 && !query.trim() && (
              <div
                className="text-center"
                style={{ padding: "8px 4px", fontSize: 11, color: "var(--ink-4)" }}
              >
                No more tags. Type to create.
              </div>
            )}

            {filteredUnassigned.map(t => {
              const armed = armedDeleteTagId === t.id;
              const showTrash = hoveredTagId === t.id || armed;
              const rowBg =
                hoveredTagId === t.id ? "var(--surface-2)" : "transparent";
              return (
                <div
                  key={t.id}
                  className="flex items-center justify-between"
                  style={{
                    minHeight: 28,
                    padding: "0 6px",
                    borderRadius: 6,
                    background: rowBg,
                  }}
                  onMouseEnter={() => setHoveredTagId(t.id)}
                  onMouseLeave={() => {
                    setHoveredTagId(prev => (prev === t.id ? null : prev));
                    // Hover-out un-arms — prevents a stale armed row from
                    // sitting in the dropdown if the user moves on.
                    setArmedDeleteTagId(prev => (prev === t.id ? null : prev));
                  }}
                >
                  {/* Left: assign click area */}
                  <button
                    type="button"
                    onClick={() => {
                      setArmedDeleteTagId(null);
                      void handleAssign(t);
                    }}
                    className="flex items-center text-left"
                    style={{
                      flex: 1,
                      gap: 6,
                      minWidth: 0,
                      height: 28,
                      background: "transparent",
                      border: "none",
                      borderRadius: 6,
                      color: "var(--ink)",
                      cursor: "pointer",
                      padding: 0,
                    }}
                  >
                    <TagPill label={t.name} tone={t.color as TagTone} />
                  </button>

                  {/* Right: trash icon (hover-revealed) OR armed confirm strip */}
                  {armed ? (
                    <div className="flex items-center" style={{ gap: 6, paddingLeft: 8 }}>
                      <span
                        style={{
                          fontSize: 9,
                          color: "var(--ink-4)",
                          fontStyle: "italic",
                          whiteSpace: "nowrap",
                        }}
                      >
                        Hides everywhere it&rsquo;s used
                      </span>
                      <button
                        type="button"
                        onClick={() => setArmedDeleteTagId(null)}
                        aria-label={`Cancel delete ${t.name}`}
                        style={{
                          fontSize: 10,
                          fontWeight: 600,
                          padding: "2px 6px",
                          borderRadius: 4,
                          background: "transparent",
                          border: "1px solid var(--border)",
                          color: "var(--ink-3)",
                          cursor: "pointer",
                        }}
                      >
                        Cancel
                      </button>
                      <button
                        type="button"
                        onClick={() => void handleDeleteTag(t)}
                        aria-label={`Confirm delete ${t.name}`}
                        style={{
                          fontSize: 10,
                          fontWeight: 600,
                          padding: "2px 6px",
                          borderRadius: 4,
                          background: "#e5484d",
                          border: "1px solid #e5484d",
                          color: "white",
                          cursor: "pointer",
                        }}
                      >
                        Confirm
                      </button>
                    </div>
                  ) : (
                    <button
                      type="button"
                      onClick={() => setArmedDeleteTagId(t.id)}
                      aria-label={`Delete tag ${t.name}`}
                      style={{
                        opacity: showTrash ? 0.7 : 0,
                        transition: "opacity 0.12s",
                        background: "transparent",
                        border: "none",
                        color: "var(--ink-4)",
                        cursor: "pointer",
                        padding: "0 4px",
                        height: 24,
                        display: "grid",
                        placeItems: "center",
                      }}
                    >
                      <Icons.trash />
                    </button>
                  )}
                </div>
              );
            })}

            {query.trim() && !exactMatch && (
              <div
                style={{
                  padding: 6,
                  borderTop: filteredUnassigned.length > 0
                    ? "1px solid var(--border)"
                    : "none",
                  marginTop: filteredUnassigned.length > 0 ? 4 : 0,
                }}
              >
                <div
                  className="flex items-center"
                  style={{ gap: 6, marginBottom: 6, fontSize: 10, color: "var(--ink-4)", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 600 }}
                >
                  Create new
                </div>
                <div className="flex items-center" style={{ gap: 6, marginBottom: 6 }}>
                  {TAG_TONES.map(tone => (
                    <button
                      key={tone}
                      type="button"
                      aria-label={`Pick ${tone} color`}
                      onClick={() => setSelectedTone(tone)}
                      style={{
                        width: 18,
                        height: 18,
                        borderRadius: 999,
                        border: selectedTone === tone
                          ? "2px solid var(--ink)"
                          : "2px solid transparent",
                        padding: 0,
                        cursor: "pointer",
                        background: "transparent",
                      }}
                    >
                      <span
                        style={{
                          display: "block",
                          width: 12,
                          height: 12,
                          borderRadius: 999,
                          background: ({
                            rose: "#f43f5e", amber: "#f59f00", emerald: "#08a86b",
                            sky: "#0d6efd", violet: "#8b5cf6",
                          } as const)[tone],
                        }}
                      />
                    </button>
                  ))}
                </div>
                <button
                  type="button"
                  disabled={creating}
                  onClick={() => void handleCreateAndAssign(query, selectedTone)}
                  className="w-full text-left flex items-center"
                  style={{
                    gap: 6,
                    height: 28,
                    padding: "0 6px",
                    fontSize: 12,
                    background: "var(--surface-2)",
                    border: "none",
                    borderRadius: 6,
                    color: "var(--ink)",
                    cursor: creating ? "wait" : "pointer",
                  }}
                >
                  <Icons.plus />
                  <span>Create &ldquo;{query.trim()}&rdquo;</span>
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {error && (
        <div
          role="status"
          className="mt-1.5"
          style={{
            fontSize: 11,
            padding: "4px 8px",
            borderRadius: 6,
            background: "color-mix(in oklab, #e5484d 10%, var(--surface))",
            color: "#dc2626",
            border: "1px solid color-mix(in oklab, #e5484d 30%, var(--border))",
            display: "inline-block",
          }}
        >
          {error}
        </div>
      )}
    </div>
  );
}
