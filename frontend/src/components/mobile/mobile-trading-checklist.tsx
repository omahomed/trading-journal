"use client";

// Mobile Trading Checklist (Migration 050). Grouped tick / untick view;
// editor is desktop-only per spec. 44px tap targets, same-day undo
// enforcement mirrors desktop trading-checklist.tsx.
//
// Layout diverges from desktop in one place: monthly + quarterly items
// live below a collapsed "Longer horizon" divider so the daily/weekly
// bulk stays visible without scroll on a phone. Everything is still
// tickable — nothing is hidden, just visually deprioritized.

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { api, type RoutineItem } from "@/lib/api";
import { groupRoutineItems, itemStatusChip } from "@/lib/trading-checklist";
import { log } from "@/lib/log";

// Kept in sync with the desktop trading-checklist.tsx map. If a new
// system item gets a canonical page, add the prefix here too.
const SYSTEM_ITEM_INTERNAL_LINKS: ReadonlyArray<{ prefix: string; href: string }> = [
  { prefix: "Equity routine", href: "/nlv-entry" },
];

function internalLinkForItem(item: RoutineItem): string | null {
  if (!item.is_system) return null;
  const match = SYSTEM_ITEM_INTERNAL_LINKS.find(l => item.name.startsWith(l.prefix));
  return match?.href ?? null;
}

export function MobileTradingChecklist({ navColor }: { navColor: string }) {
  const [items, setItems] = useState<RoutineItem[] | null>(null);
  const [loadError, setLoadError] = useState<string>("");
  const [busyId, setBusyId] = useState<number | null>(null);
  const [rowError, setRowError] = useState<{ id: number; msg: string } | null>(null);
  const [longerOpen, setLongerOpen] = useState(false);

  const load = useCallback(async () => {
    setLoadError("");
    try {
      const res = await api.routineItemsList();
      if ("error" in res) setLoadError(res.error);
      else setItems(res.items);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setLoadError(msg);
      log.error("mobile-trading-checklist", "load failed", e);
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  const groups = useMemo(() => (items ? groupRoutineItems(items) : []), [items]);
  const dailyWeekly = useMemo(
    () => groups.filter(g => g.frequency === "daily" || g.frequency === "weekly"),
    [groups],
  );
  const monthlyQuarterly = useMemo(
    () => groups.filter(g => g.frequency === "monthly" || g.frequency === "quarterly"),
    [groups],
  );

  const onTick = useCallback(async (item: RoutineItem) => {
    setBusyId(item.id);
    setRowError(null);
    try {
      const res = await api.routineLogTick(item.id);
      if ("error" in res) setRowError({ id: item.id, msg: res.error });
      else await load();
    } catch (e) {
      setRowError({ id: item.id, msg: e instanceof Error ? e.message : String(e) });
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
        setRowError({ id: item.id, msg: "Undo blocked — log is from a prior day." });
      } else if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setRowError({ id: item.id, msg: body.error || `Failed: ${res.status}` });
      } else {
        await load();
      }
    } catch (e) {
      setRowError({ id: item.id, msg: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusyId(null);
    }
  }, [load]);

  return (
    <div className="pb-4" data-testid="mobile-trading-checklist-root">
      {/* Section header — small caps, matches "PORTFOLIO STATS" / "PERFORMANCE"
          / "POSITIONS OPENED" section labels elsewhere in MobileDailyJournal.
          Deliberately NOT a MobilePageHeader — the page title is "Daily
          Journal" (owned by the shell/AdaptiveShell); the checklist is one
          section within it. Rendering another page-scale wordmark here
          made the checklist read like its own page. */}
      <div className="px-1 mb-2 flex items-center justify-between">
        <div className="text-[11px] font-semibold uppercase tracking-[0.08em]"
             style={{ color: "var(--m-text-muted)" }}>
          Trading Checklist
        </div>
      </div>

      {loadError && (
        <div className="mx-5 mb-3 px-3 py-2 rounded-[10px] text-[13px]"
             style={{ background: "color-mix(in oklab, #e5484d 10%, transparent)", color: "#e5484d" }}>
          Failed to load: {loadError}
        </div>
      )}

      {items === null ? (
        <div className="px-5 flex flex-col gap-3">
          {[0, 1, 2].map(i => (
            <div key={i} className="rounded-[14px] animate-pulse h-[120px]" style={{ background: "var(--m-surface-2)" }} />
          ))}
        </div>
      ) : groups.length === 0 ? (
        <div className="text-center py-10 px-5 text-[13px]" style={{ color: "var(--m-text-muted)" }}>
          No items yet. Add one on the desktop app.
        </div>
      ) : (
        <div className="px-4 flex flex-col gap-3">
          {dailyWeekly.map(g => (
            <GroupCard key={`${g.frequency}|${g.slot ?? ""}`}
                       label={g.label}
                       items={g.items}
                       busyId={busyId}
                       rowError={rowError}
                       onTick={onTick}
                       onUntick={onUntick}
                       navColor={navColor} />
          ))}

          {monthlyQuarterly.length > 0 && (
            <>
              <button type="button" onClick={() => setLongerOpen(o => !o)}
                      className="mt-2 mx-1 px-3 py-2 rounded-[10px] text-[12px] font-semibold uppercase tracking-[0.06em] flex items-center justify-between"
                      style={{ background: "var(--m-surface-2)", color: "var(--m-text-muted)" }}>
                <span>Longer horizon · {monthlyQuarterly.reduce((n, g) => n + g.items.length, 0)}</span>
                <span>{longerOpen ? "▾" : "▸"}</span>
              </button>
              {longerOpen && monthlyQuarterly.map(g => (
                <GroupCard key={`${g.frequency}|${g.slot ?? ""}`}
                           label={g.label}
                           items={g.items}
                           busyId={busyId}
                           rowError={rowError}
                           onTick={onTick}
                           onUntick={onUntick}
                           navColor={navColor} />
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ── Group card ──────────────────────────────────────────────────

function GroupCard(props: {
  label: string;
  items: RoutineItem[];
  busyId: number | null;
  rowError: { id: number; msg: string } | null;
  onTick: (item: RoutineItem) => void | Promise<void>;
  onUntick: (item: RoutineItem) => void | Promise<void>;
  navColor: string;
}) {
  const { label, items, busyId, rowError, onTick, onUntick, navColor } = props;
  return (
    <div className="rounded-[14px] overflow-hidden"
         style={{ background: "var(--m-surface)", border: "1px solid var(--m-border)" }}>
      <div className="px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.06em]"
           style={{ color: "var(--m-text-muted)", borderBottom: "1px solid var(--m-border)" }}>
        {label} <span style={{ color: "var(--m-text-faint)" }}>· {items.length}</span>
      </div>
      <div>
        {items.map((item, idx) => (
          <MobileItemRow key={item.id}
                         item={item}
                         isLast={idx === items.length - 1}
                         isBusy={busyId === item.id}
                         onTick={() => void onTick(item)}
                         onUntick={() => void onUntick(item)}
                         rowError={rowError?.id === item.id ? rowError.msg : null}
                         navColor={navColor} />
        ))}
      </div>
    </div>
  );
}

// ── Mobile row (44px tap target) ────────────────────────────────

function MobileItemRow(props: {
  item: RoutineItem;
  isLast: boolean;
  isBusy: boolean;
  onTick: () => void;
  onUntick: () => void;
  rowError: string | null;
  navColor: string;
}) {
  const { item, isLast, isBusy, onTick, onUntick, rowError, navColor } = props;
  const isTask = item.item_type === "task";
  const checked = isTask && item.ticked_today;
  const chip = itemStatusChip(item);
  const handleToggle = () => {
    if (isBusy) return;
    if (isTask && checked) onUntick();
    else onTick();
  };
  return (
    <div className="px-3"
         style={{ borderBottom: isLast ? "none" : "1px solid var(--m-border)" }}
         data-testid={`mobile-routine-item-${item.id}`}>
      <button type="button" onClick={handleToggle} disabled={isBusy}
              className="w-full flex items-center gap-3 py-2.5 text-left disabled:opacity-60"
              style={{ minHeight: 44 }}
              data-testid={`mobile-routine-tick-${item.id}`}
              aria-pressed={isTask ? checked : undefined}
              aria-label={isTask ? (checked ? `Untick ${item.name}` : `Tick ${item.name}`) : `Log ${item.name}`}>
        {isTask ? (
          <span className="w-[28px] h-[28px] rounded-[8px] flex items-center justify-center shrink-0"
                style={{
                  background: checked ? navColor : "transparent",
                  border: `1.5px solid ${checked ? navColor : "var(--m-border)"}`,
                  color: checked ? "white" : "transparent",
                  fontSize: 16,
                  lineHeight: 1,
                }}>
            ✓
          </span>
        ) : (
          <span className="w-[28px] h-[28px] rounded-full flex items-center justify-center shrink-0"
                style={{
                  background: item.ticked_today ? "color-mix(in oklab, #e5484d 22%, var(--m-surface-2))" : "var(--m-surface-2)",
                  border: "1.5px solid var(--m-border)",
                  color: item.ticked_today ? "#e5484d" : "var(--m-text-muted)",
                  fontSize: 14,
                  lineHeight: 1,
                }}>
            !
          </span>
        )}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-[14px] truncate" style={{ color: "var(--m-text)" }}>{item.name}</span>
            {item.is_system && (
              <span className="text-[9px] px-1.5 py-0.5 rounded-[4px] uppercase tracking-[0.05em] shrink-0"
                    style={{ background: "var(--m-surface-2)", color: "var(--m-text-faint)" }}>sys</span>
            )}
          </div>
          <div className="mt-0.5 text-[11px]">
            {chip.kind === "overdue" ? (
              <span className="font-semibold" style={{ color: "#e5484d" }}>{chip.text} overdue</span>
            ) : chip.kind === "today" ? (
              <span style={{ color: navColor }}>{chip.text}</span>
            ) : (
              <span style={{ color: "var(--m-text-faint)" }}>{chip.text}</span>
            )}
          </div>
        </div>
        {item.link ? (
          <a href={item.link} target="_blank" rel="noopener noreferrer"
             onClick={(e) => e.stopPropagation()}
             className="shrink-0 text-[13px] px-2 py-1 rounded-[6px]"
             style={{ color: "var(--m-text-muted)" }}
             aria-label={`Open link for ${item.name}`}>
            ↗
          </a>
        ) : internalLinkForItem(item) ? (
          <Link href={internalLinkForItem(item)!}
                onClick={(e) => e.stopPropagation()}
                className="shrink-0 text-[13px] px-2 py-1 rounded-[6px]"
                style={{ color: "var(--m-text-muted)" }}
                aria-label={`Open capture page for ${item.name}`}>
            ↗
          </Link>
        ) : null}
      </button>
      {rowError && (
        <div className="pb-2 text-[11px]" style={{ color: "#e5484d" }}>{rowError}</div>
      )}
    </div>
  );
}
