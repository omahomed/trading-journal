"use client";

// Trading Checklist — desktop view. Reads /api/routine/items, renders
// grouped by (frequency, slot), lets the user tick / untick with same-day-
// undo enforcement, and provides a small custom-item editor at the top.
//
// Theme matches Campaign Review (var(--ink-*), var(--surface), var(--bg-2),
// rounded-[10px], small type). Nav color is threaded in from the client
// wrapper for accent tinting on the italic title word + active states.
//
// State model: single `items` list is the source of truth; each mutation
// (tick, untick, create, edit, delete) does a targeted refetch afterwards
// so derived fields (last_run, overdue_days, ticked_today) stay honest.
// Optimistic UI is intentionally avoided in v1 — a checkbox that lies
// after a network error would defeat the point of an evidence log.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type RoutineFrequency, type RoutineItem, type RoutineSlot } from "@/lib/api";
import {
  FREQUENCY_LABELS,
  SLOT_LABELS,
  FREQUENCY_ORDER,
  SLOT_ORDER,
  computeReorderSwap,
  groupRoutineItems,
  itemStatusChip,
} from "@/lib/trading-checklist";
import { log } from "@/lib/log";

type AddFormState = {
  name: string;
  frequency: RoutineFrequency;
  slot: RoutineSlot | "";
  link: string;
};

const EMPTY_FORM: AddFormState = { name: "", frequency: "daily", slot: "after_close", link: "" };

const DELETE_ARM_TIMEOUT_MS = 3000;

export function TradingChecklist({ navColor }: { navColor: string }) {
  const [items, setItems] = useState<RoutineItem[] | null>(null);
  const [loadError, setLoadError] = useState<string>("");
  const [refreshing, setRefreshing] = useState(false);
  const [busyId, setBusyId] = useState<number | null>(null);
  const [rowError, setRowError] = useState<{ id: number; msg: string } | null>(null);
  const [addForm, setAddForm] = useState<AddFormState>(EMPTY_FORM);
  const [addSubmitting, setAddSubmitting] = useState(false);
  const [addError, setAddError] = useState<string>("");
  const [editingId, setEditingId] = useState<number | null>(null);
  // Two-click inline delete confirm (mirrors image-gallery.tsx). One item
  // armed at a time; a click on a different item swaps the arming. Timer
  // auto-clears the armed state so a stray click doesn't leave the row in
  // a confirm state indefinitely.
  const [pendingDeleteId, setPendingDeleteId] = useState<number | null>(null);
  const pendingDeleteTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => () => {
    if (pendingDeleteTimer.current) clearTimeout(pendingDeleteTimer.current);
  }, []);

  const load = useCallback(async (silent = false) => {
    if (!silent) setRefreshing(true);
    setLoadError("");
    try {
      const res = await api.routineItemsList();
      if ("error" in res) {
        setLoadError(res.error);
      } else {
        setItems(res.items);
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setLoadError(msg);
      log.error("trading-checklist", "load failed", e);
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  const groups = useMemo(() => (items ? groupRoutineItems(items) : []), [items]);

  const onTick = useCallback(async (item: RoutineItem) => {
    setBusyId(item.id);
    setRowError(null);
    try {
      const res = await api.routineLogTick(item.id);
      if ("error" in res) {
        setRowError({ id: item.id, msg: res.error });
      } else {
        await load(true);
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setRowError({ id: item.id, msg });
    } finally {
      setBusyId(null);
    }
  }, [load]);

  const onUntick = useCallback(async (item: RoutineItem) => {
    if (!item.todays_log_id) return;
    setBusyId(item.id);
    setRowError(null);
    try {
      const res = await api.routineLogUntick(item.todays_log_id);
      if (res.status === 409) {
        setRowError({ id: item.id, msg: "Cannot undo — log is from a prior day (America/Chicago)." });
      } else if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setRowError({ id: item.id, msg: body.error || `Untick failed: ${res.status}` });
      } else {
        await load(true);
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setRowError({ id: item.id, msg });
    } finally {
      setBusyId(null);
    }
  }, [load]);

  const onAdd = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    setAddSubmitting(true);
    setAddError("");
    try {
      const res = await api.routineItemsCreate({
        name: addForm.name.trim(),
        frequency: addForm.frequency,
        slot: addForm.slot ? addForm.slot : null,
        link: addForm.link.trim() ? addForm.link.trim() : null,
      });
      if ("error" in res) {
        setAddError(res.error);
      } else {
        setAddForm(EMPTY_FORM);
        await load(true);
      }
    } catch (err) {
      setAddError(err instanceof Error ? err.message : String(err));
    } finally {
      setAddSubmitting(false);
    }
  }, [addForm, load]);

  const commitDelete = useCallback(async (item: RoutineItem) => {
    setBusyId(item.id);
    setRowError(null);
    try {
      const res = await api.routineItemsDelete(item.id);
      if ("error" in res) {
        setRowError({ id: item.id, msg: res.error });
      } else {
        await load(true);
      }
    } catch (err) {
      setRowError({ id: item.id, msg: err instanceof Error ? err.message : String(err) });
    } finally {
      setBusyId(null);
    }
  }, [load]);

  const armDelete = useCallback((id: number) => {
    setPendingDeleteId(id);
    if (pendingDeleteTimer.current) clearTimeout(pendingDeleteTimer.current);
    pendingDeleteTimer.current = setTimeout(() => {
      setPendingDeleteId(null);
      pendingDeleteTimer.current = null;
    }, DELETE_ARM_TIMEOUT_MS);
  }, []);

  const onDeleteClick = useCallback((item: RoutineItem) => {
    if (pendingDeleteId === item.id) {
      // Second click — commit.
      if (pendingDeleteTimer.current) {
        clearTimeout(pendingDeleteTimer.current);
        pendingDeleteTimer.current = null;
      }
      setPendingDeleteId(null);
      void commitDelete(item);
    } else {
      // First click, or arm swap from another row.
      armDelete(item.id);
    }
  }, [pendingDeleteId, armDelete, commitDelete]);

  const onReorder = useCallback(async (item: RoutineItem, direction: "up" | "down") => {
    if (!items) return;
    const swap = computeReorderSwap(items, item.id, direction);
    if (!swap) return;
    setBusyId(item.id);
    setRowError(null);
    try {
      const res = await api.routineItemsReorder(swap);
      if ("error" in res) {
        setRowError({ id: item.id, msg: res.error });
      } else {
        await load(true);
      }
    } catch (err) {
      setRowError({ id: item.id, msg: err instanceof Error ? err.message : String(err) });
    } finally {
      setBusyId(null);
    }
  }, [items, load]);

  // No internal header — the merged Daily Routine wraps this component
  // in a SectionExpander that provides the collapse chrome, title, and
  // caption. Refresh happens automatically on mount + after each tick /
  // untick, so no manual Refresh button is needed.

  return (
    <div data-testid="trading-checklist-root">

      {loadError && (
        <div className="mb-4 px-4 py-3 rounded-[10px] text-[13px]"
             style={{ background: "color-mix(in oklab, #e5484d 8%, var(--surface))", border: "1px solid var(--border)", color: "#e5484d" }}>
          Failed to load: {loadError}
        </div>
      )}

      {/* Add-item card */}
      <form onSubmit={onAdd}
            className="mb-5 p-4 rounded-[12px]"
            style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <div className="text-[12px] font-semibold uppercase tracking-[0.06em] mb-3"
             style={{ color: "var(--ink-3)" }}>
          Add custom item
        </div>
        <div className="flex flex-wrap items-end gap-3">
          <label className="flex flex-col gap-1 flex-1 min-w-[220px]">
            <span className="text-[11px]" style={{ color: "var(--ink-4)" }}>Name</span>
            <input type="text" value={addForm.name} required maxLength={120}
                   onChange={(e) => setAddForm(f => ({ ...f, name: e.target.value }))}
                   placeholder="e.g. Review IBD 50"
                   className="h-[36px] px-2.5 rounded-[10px] text-[13px]"
                   style={{ background: "var(--bg-2)", border: "1px solid var(--border)", color: "var(--ink-1)" }} />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-[11px]" style={{ color: "var(--ink-4)" }}>Frequency</span>
            <select value={addForm.frequency}
                    onChange={(e) => setAddForm(f => ({ ...f, frequency: e.target.value as RoutineFrequency }))}
                    className="h-[36px] px-2 rounded-[10px] text-[13px]"
                    style={{ background: "var(--bg-2)", border: "1px solid var(--border)", color: "var(--ink-1)" }}>
              {FREQUENCY_ORDER.map(f => (
                <option key={f} value={f}>{FREQUENCY_LABELS[f]}</option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-[11px]" style={{ color: "var(--ink-4)" }}>Slot</span>
            <select value={addForm.slot}
                    onChange={(e) => setAddForm(f => ({ ...f, slot: e.target.value as RoutineSlot | "" }))}
                    className="h-[36px] px-2 rounded-[10px] text-[13px]"
                    style={{ background: "var(--bg-2)", border: "1px solid var(--border)", color: "var(--ink-1)" }}>
              <option value="">— none —</option>
              {SLOT_ORDER.map(s => (
                <option key={s} value={s}>{SLOT_LABELS[s]}</option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1 flex-1 min-w-[200px]">
            <span className="text-[11px]" style={{ color: "var(--ink-4)" }}>Link (optional)</span>
            <input type="url" value={addForm.link}
                   onChange={(e) => setAddForm(f => ({ ...f, link: e.target.value }))}
                   placeholder="https://marketsmith.com/…"
                   className="h-[36px] px-2.5 rounded-[10px] text-[13px]"
                   style={{ background: "var(--bg-2)", border: "1px solid var(--border)", color: "var(--ink-1)" }} />
          </label>
          <button type="submit" disabled={addSubmitting || !addForm.name.trim()}
                  className="h-[36px] px-4 rounded-[10px] text-[13px] font-medium transition-colors disabled:opacity-50"
                  style={{ background: navColor, color: "white" }}>
            {addSubmitting ? "Adding…" : "Add item"}
          </button>
        </div>
        {addError && (
          <div className="mt-2 text-[12px]" style={{ color: "#e5484d" }}>{addError}</div>
        )}
      </form>

      {/* Groups */}
      {items === null ? (
        <div className="flex flex-col gap-3">
          {[0, 1, 2].map(i => (
            <div key={i} className="rounded-[12px] animate-pulse h-[160px]" style={{ background: "var(--bg-2)" }} />
          ))}
        </div>
      ) : groups.length === 0 ? (
        <div className="text-center py-10 text-[13px]" style={{ color: "var(--ink-3)" }}>
          No items yet. Add one above to get started.
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          {groups.map(g => (
            <div key={`${g.frequency}|${g.slot ?? ""}`}
                 className="rounded-[12px] overflow-hidden"
                 style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
              <div className="px-4 py-2.5 flex items-center justify-between text-[12px] font-semibold uppercase tracking-[0.06em]"
                   style={{ background: "var(--bg-2)", color: "var(--ink-2)", borderBottom: "1px solid var(--border)" }}>
                <span>{g.label}</span>
                <span style={{ color: "var(--ink-4)" }}>{g.items.length} item{g.items.length === 1 ? "" : "s"}</span>
              </div>
              <div>
                {g.items.map((item, idx) => {
                  // Reorder edges — computed over CUSTOM siblings only so
                  // the first/last custom item in the group correctly
                  // disables the up/down affordance the user can act on.
                  const customSiblings = g.items.filter(it => !it.is_system);
                  const customIdx = customSiblings.findIndex(it => it.id === item.id);
                  const canMoveUp = !item.is_system && customIdx > 0;
                  const canMoveDown = !item.is_system && customIdx >= 0 && customIdx < customSiblings.length - 1;
                  return (
                    <ItemRow
                      key={item.id}
                      item={item}
                      isLast={idx === g.items.length - 1}
                      isBusy={busyId === item.id}
                      isEditing={editingId === item.id}
                      canMoveUp={canMoveUp}
                      canMoveDown={canMoveDown}
                      isPendingDelete={pendingDeleteId === item.id}
                      onTick={() => void onTick(item)}
                      onUntick={() => void onUntick(item)}
                      onEditStart={() => setEditingId(item.id)}
                      onEditCancel={() => setEditingId(null)}
                      onEditSaved={async () => { setEditingId(null); await load(true); }}
                      onDelete={() => onDeleteClick(item)}
                      onMoveUp={() => void onReorder(item, "up")}
                      onMoveDown={() => void onReorder(item, "down")}
                      rowError={rowError?.id === item.id ? rowError.msg : null}
                      navColor={navColor}
                    />
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Item row ─────────────────────────────────────────────────────

function ItemRow(props: {
  item: RoutineItem;
  isLast: boolean;
  isBusy: boolean;
  isEditing: boolean;
  canMoveUp: boolean;
  canMoveDown: boolean;
  isPendingDelete: boolean;
  onTick: () => void;
  onUntick: () => void;
  onEditStart: () => void;
  onEditCancel: () => void;
  onEditSaved: () => Promise<void>;
  onDelete: () => void;
  onMoveUp: () => void;
  onMoveDown: () => void;
  rowError: string | null;
  navColor: string;
}) {
  const { item, isLast, isBusy, isEditing, canMoveUp, canMoveDown, isPendingDelete, onTick, onUntick, onEditStart, onEditCancel, onEditSaved, onDelete, onMoveUp, onMoveDown, rowError, navColor } = props;
  // Reorder affordance only appears when the group has ≥2 custom items —
  // a solo custom row would otherwise show two dimmed ghost buttons the
  // user can't act on, which read as broken.
  const showReorder = canMoveUp || canMoveDown;
  const isTask = item.item_type === "task";
  const checked = isTask && item.ticked_today;
  const chip = itemStatusChip(item);
  return (
    <div className="px-4 py-3 flex items-center gap-3"
         style={{ borderBottom: isLast ? "none" : "1px solid var(--border)" }}
         data-testid={`routine-item-${item.id}`}>
      {/* Check / counter icon */}
      {isTask ? (
        <button type="button" disabled={isBusy}
                onClick={checked ? onUntick : onTick}
                aria-label={checked ? "Untick (same-day)" : "Tick as done"}
                data-testid={`routine-tick-${item.id}`}
                className="w-[26px] h-[26px] rounded-[7px] flex items-center justify-center transition-colors disabled:opacity-50"
                style={{
                  background: checked ? navColor : "transparent",
                  border: `1.5px solid ${checked ? navColor : "var(--border-strong, var(--border))"}`,
                  color: checked ? "white" : "transparent",
                }}>
          <span style={{ fontSize: 15, lineHeight: 1 }}>✓</span>
        </button>
      ) : (
        <button type="button" disabled={isBusy}
                onClick={onTick}
                aria-label="Log an incident"
                data-testid={`routine-counter-${item.id}`}
                title="Log this — a tick means one incident happened today"
                className="w-[26px] h-[26px] rounded-full flex items-center justify-center transition-colors disabled:opacity-50"
                style={{
                  background: item.ticked_today ? "color-mix(in oklab, #e5484d 22%, var(--bg-2))" : "var(--bg-2)",
                  border: "1.5px solid var(--border)",
                  color: item.ticked_today ? "#e5484d" : "var(--ink-3)",
                }}>
          <span style={{ fontSize: 14, lineHeight: 1 }}>!</span>
        </button>
      )}

      {/* Name + link + system tag */}
      <div className="flex-1 min-w-0">
        {isEditing ? (
          <EditRow item={item} onCancel={onEditCancel} onSaved={onEditSaved} navColor={navColor} />
        ) : (
          <div className="flex items-center gap-2">
            {item.link ? (
              <a href={item.link} target="_blank" rel="noopener noreferrer"
                 className="text-[13px] hover:underline truncate"
                 style={{ color: "var(--ink-1)" }}>
                {item.name}
              </a>
            ) : (
              <span className="text-[13px] truncate" style={{ color: "var(--ink-1)" }}>{item.name}</span>
            )}
            {item.is_system && (
              <span className="text-[10px] px-1.5 py-0.5 rounded-[4px] uppercase tracking-[0.05em] shrink-0"
                    style={{ background: "var(--bg-2)", color: "var(--ink-4)" }}>system</span>
            )}
            {item.item_type === "counter" && (
              <span className="text-[10px] px-1.5 py-0.5 rounded-[4px] uppercase tracking-[0.05em] shrink-0"
                    style={{ background: "color-mix(in oklab, #e5484d 12%, var(--bg-2))", color: "#e5484d" }}>counter</span>
            )}
          </div>
        )}
        {rowError && (
          <div className="mt-1 text-[11px]" style={{ color: "#e5484d" }}>{rowError}</div>
        )}
      </div>

      {/* Status chip — omitted entirely for "never" so a bare dash doesn't
          read as another action button next to ▲▼✎×. */}
      {!isEditing && chip.kind !== "never" && (
        <div className="shrink-0 text-[12px]">
          {chip.kind === "overdue" ? (
            <span className="px-2 py-1 rounded-[6px] font-semibold"
                  style={{ background: "color-mix(in oklab, #e5484d 12%, var(--bg-2))", color: "#e5484d" }}>
              {chip.text}
            </span>
          ) : chip.kind === "counter" ? (
            <span style={{ color: "var(--ink-4)" }}>{chip.text}</span>
          ) : chip.kind === "today" ? (
            <span style={{ color: navColor }}>{chip.text}</span>
          ) : (
            <span style={{ color: "var(--ink-3)" }}>{chip.text}</span>
          )}
        </div>
      )}

      {/* Reorder / edit / delete for custom items only */}
      {!isEditing && !item.is_system && (
        <div className="shrink-0 flex items-center gap-1">
          {showReorder && (
            <>
              <button type="button" onClick={onMoveUp} disabled={isBusy || !canMoveUp}
                      aria-label="Move up"
                      data-testid={`routine-move-up-${item.id}`}
                      className="w-[28px] h-[28px] rounded-[6px] flex items-center justify-center text-[13px] transition-colors disabled:opacity-30"
                      style={{ color: "var(--ink-4)" }}
                      title="Move up">
                ▲
              </button>
              <button type="button" onClick={onMoveDown} disabled={isBusy || !canMoveDown}
                      aria-label="Move down"
                      data-testid={`routine-move-down-${item.id}`}
                      className="w-[28px] h-[28px] rounded-[6px] flex items-center justify-center text-[13px] transition-colors disabled:opacity-30"
                      style={{ color: "var(--ink-4)" }}
                      title="Move down">
                ▼
              </button>
            </>
          )}
          <button type="button" onClick={onEditStart} disabled={isBusy}
                  aria-label="Edit"
                  className="w-[28px] h-[28px] rounded-[6px] flex items-center justify-center text-[13px] transition-colors"
                  style={{ color: "var(--ink-4)" }}
                  title="Edit">
            ✎
          </button>
          {isPendingDelete ? (
            <button type="button" onClick={onDelete} disabled={isBusy}
                    aria-label="Confirm delete"
                    data-testid={`routine-delete-confirm-${item.id}`}
                    className="h-[28px] px-2 rounded-[6px] flex items-center justify-center text-[11px] font-semibold transition-colors"
                    style={{
                      background: "color-mix(in oklab, #e5484d 14%, var(--bg-2))",
                      color: "#e5484d",
                    }}
                    title="Click again to remove">
              Delete?
            </button>
          ) : (
            <button type="button" onClick={onDelete} disabled={isBusy}
                    aria-label="Delete"
                    data-testid={`routine-delete-${item.id}`}
                    className="w-[28px] h-[28px] rounded-[6px] flex items-center justify-center text-[13px] transition-colors"
                    style={{ color: "var(--ink-4)" }}
                    title="Delete">
              ×
            </button>
          )}
        </div>
      )}
    </div>
  );
}

// ── Inline edit row ──────────────────────────────────────────────

function EditRow(props: {
  item: RoutineItem;
  onCancel: () => void;
  onSaved: () => Promise<void>;
  navColor: string;
}) {
  const { item, onCancel, onSaved, navColor } = props;
  const [name, setName] = useState(item.name);
  const [frequency, setFrequency] = useState<RoutineFrequency>(item.frequency);
  const [slot, setSlot] = useState<RoutineSlot | "">(item.slot ?? "");
  const [link, setLink] = useState(item.link ?? "");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const submit = async () => {
    setSaving(true);
    setError("");
    try {
      const res = await api.routineItemsUpdate(item.id, {
        name: name.trim(),
        frequency,
        slot: slot ? slot : null,
        link: link.trim() ? link.trim() : null,
      });
      if ("error" in res) {
        setError(res.error);
      } else {
        await onSaved();
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-2">
      <input value={name} onChange={(e) => setName(e.target.value)} maxLength={120}
             className="h-[30px] px-2 rounded-[8px] text-[13px] flex-1 min-w-[160px]"
             style={{ background: "var(--bg-2)", border: "1px solid var(--border)", color: "var(--ink-1)" }} />
      <select value={frequency} onChange={(e) => setFrequency(e.target.value as RoutineFrequency)}
              className="h-[30px] px-1 rounded-[8px] text-[12px]"
              style={{ background: "var(--bg-2)", border: "1px solid var(--border)", color: "var(--ink-1)" }}>
        {FREQUENCY_ORDER.map(f => <option key={f} value={f}>{FREQUENCY_LABELS[f]}</option>)}
      </select>
      <select value={slot} onChange={(e) => setSlot(e.target.value as RoutineSlot | "")}
              className="h-[30px] px-1 rounded-[8px] text-[12px]"
              style={{ background: "var(--bg-2)", border: "1px solid var(--border)", color: "var(--ink-1)" }}>
        <option value="">— none —</option>
        {SLOT_ORDER.map(s => <option key={s} value={s}>{SLOT_LABELS[s]}</option>)}
      </select>
      <input type="url" value={link} onChange={(e) => setLink(e.target.value)}
             placeholder="https://…"
             className="h-[30px] px-2 rounded-[8px] text-[12px] flex-1 min-w-[140px]"
             style={{ background: "var(--bg-2)", border: "1px solid var(--border)", color: "var(--ink-1)" }} />
      <button type="button" onClick={() => void submit()} disabled={saving || !name.trim()}
              className="h-[30px] px-3 rounded-[8px] text-[12px] font-medium disabled:opacity-50"
              style={{ background: navColor, color: "white" }}>
        {saving ? "Saving…" : "Save"}
      </button>
      <button type="button" onClick={onCancel} disabled={saving}
              className="h-[30px] px-2 rounded-[8px] text-[12px]"
              style={{ background: "transparent", color: "var(--ink-4)" }}>
        Cancel
      </button>
      {error && <span className="text-[11px] w-full" style={{ color: "#e5484d" }}>{error}</span>}
    </div>
  );
}
