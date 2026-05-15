"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { formatCurrency } from "@/lib/format";
import type {
  NotesRailEntityType, NotesRailItem, NotesRailYtdStats,
} from "@/lib/api";

// Phase 6 — Notion-style left rail for the Weekly Retro page (and later
// Daily Report in Phase 7). Visual contract from
// Design/design_handoff_weekly_retro/design/weekly-notes-rail-*.jsx.
//
// Entity-agnostic: items come in pre-shaped from the server; the parent
// owns the navigation handler (onItemClick) and the pin-toggle network
// call (onPinToggle). Rail-internal concerns: folder grouping, collapse
// state, search filter, keyboard nav, optimistic pin overlay.

const COLLAPSED_KEY_PREFIX = "mo-notes-rail-";

export interface NotesRailProps {
  entityType: NotesRailEntityType;
  items: NotesRailItem[];
  ytdStats: NotesRailYtdStats;
  currentEntityKey: string | null;
  onItemClick: (item: NotesRailItem) => void;
  onPinToggle: (entityId: number, currentlyPinned: boolean) => void | Promise<void>;
  collapsible?: boolean;
}

function fmtPct(v: number): string {
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toFixed(2)}%`;
}

function loadLs(key: string, fallback: any): any {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback;
    return JSON.parse(raw);
  } catch { return fallback; }
}

function saveLs(key: string, value: any): void {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* incognito */ }
}

// One per-month inline SVG sparkline — green/red bars around a midline,
// height scaled per-month so the busiest week peaks at the top. Ported
// verbatim from weekly-notes-rail.jsx:123-144 (the design's MonthSparkline)
// with the only change being that `sparkline_value` is a percent return
// rather than the design's dollar P&L (Phase 6 spec — normalized across
// account-size growth).
function MonthSparkline({ values }: { values: (number | null)[] }) {
  const real = values.filter((v): v is number => v != null);
  if (real.length === 0) return null;
  const width = 56;
  const height = 18;
  // ordered oldest → newest for left-to-right reading
  const ordered = [...real].reverse();
  const max = Math.max(...ordered.map(v => Math.abs(v)), 1);
  const bw = Math.max(2, Math.floor((width - (ordered.length - 1) * 2) / ordered.length));
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display: "block" }}>
      <line x1={0} y1={height / 2} x2={width} y2={height / 2}
            stroke="var(--border)" strokeWidth={1} />
      {ordered.map((v, i) => {
        const h = Math.max(2, (Math.abs(v) / max) * (height - 4));
        const x = i * (bw + 2);
        const positive = v >= 0;
        const y = positive ? (height / 2) - h : height / 2;
        return (
          <rect key={i} x={x} y={y} width={bw} height={h} rx={1}
                fill={positive ? "#08a86b" : "#e5484d"} opacity={0.85} />
        );
      })}
    </svg>
  );
}

function StarIcon({ filled, size = 12 }: { filled: boolean; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24"
         fill={filled ? "currentColor" : "none"}
         stroke="currentColor" strokeWidth={2}
         strokeLinecap="round" strokeLinejoin="round">
      <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
    </svg>
  );
}

function ChevronRight() {
  return (
    <svg width={12} height={12} viewBox="0 0 24 24" fill="none"
         stroke="currentColor" strokeWidth={2}
         strokeLinecap="round" strokeLinejoin="round">
      <polyline points="9 18 15 12 9 6" />
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg width={12} height={12} viewBox="0 0 24 24" fill="none"
         stroke="currentColor" strokeWidth={2}
         strokeLinecap="round" strokeLinejoin="round">
      <circle cx={11} cy={11} r={7} />
      <line x1={20} y1={20} x2={16.65} y2={16.65} />
    </svg>
  );
}

function CollapseIcon() {
  return (
    <svg width={14} height={14} viewBox="0 0 24 24" fill="none"
         stroke="currentColor" strokeWidth={2.2}
         strokeLinecap="round" strokeLinejoin="round">
      <polyline points="15 18 9 12 15 6" />
      <line x1={20} y1={6} x2={20} y2={18} />
    </svg>
  );
}

function HamburgerIcon() {
  return (
    <svg width={16} height={16} viewBox="0 0 24 24" fill="none"
         stroke="currentColor" strokeWidth={2.2}
         strokeLinecap="round" strokeLinejoin="round">
      <line x1={3} y1={6} x2={21} y2={6} />
      <line x1={3} y1={12} x2={21} y2={12} />
      <line x1={3} y1={18} x2={21} y2={18} />
    </svg>
  );
}

const MONTH_NAMES = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

// ─── RailItem ───────────────────────────────────────────────────────────
// Single row: graded-dot + title + pin star (when pinned/hovered) + return%.
// Active state adds the surface bg + 1px border + card-shadow + 3px accent
// stripe on the left edge. Negative return → red accent; positive → green.

function RailItem({
  item, active, focused, onClick, onPinToggle,
}: {
  item: NotesRailItem;
  active: boolean;
  focused: boolean;
  onClick: () => void;
  onPinToggle: () => void;
}) {
  const [hover, setHover] = useState(false);
  const positive = (item.sparkline_value ?? 0) >= 0;
  const accent = positive ? "#08a86b" : "#e5484d";
  const bg = (active || hover || focused) ? "var(--surface)" : "transparent";
  return (
    <div data-testid="rail-item"
         data-key={item.key}
         data-active={active ? "true" : "false"}
         data-pinned={item.pinned ? "true" : "false"}
         onMouseEnter={() => setHover(true)}
         onMouseLeave={() => setHover(false)}
         style={{ position: "relative" }}>
      <button onClick={onClick}
              aria-current={active ? "true" : undefined}
              style={{
                width: "100%", textAlign: "left",
                padding: "8px 32px 8px 14px",
                marginLeft: 10,
                borderRadius: 8,
                background: bg,
                border: `1px solid ${active ? "var(--border)" : "transparent"}`,
                boxShadow: active ? "var(--card-shadow)" : "none",
                display: "flex", flexDirection: "column", gap: 4,
                position: "relative", transition: "background 120ms",
                cursor: "pointer", color: "var(--ink)",
              }}>
        {active && (
          <span aria-hidden style={{
            position: "absolute", left: -1, top: 8, bottom: 8, width: 3,
            borderRadius: 99, background: accent,
          }} />
        )}
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          {/* Graded indicator — filled green when has_content, hollow when draft */}
          <span title={item.has_content ? "Graded" : "Draft — needs grading"}
                aria-label={item.has_content ? "graded" : "draft"}
                style={{
                  width: 8, height: 8, borderRadius: 999, flexShrink: 0,
                  background: item.has_content ? "#08a86b" : "transparent",
                  border: `1.5px solid ${item.has_content ? "#08a86b" : "var(--border)"}`,
                }} />
          <span style={{
            fontSize: 12.5, fontWeight: active ? 600 : 500,
            color: "var(--ink)", flex: 1, minWidth: 0,
            whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
          }}>{item.title}</span>
          {item.pinned && (
            <span style={{ color: "#f59f00", display: "inline-flex", flexShrink: 0 }}
                  title="Pinned"><StarIcon filled size={11} /></span>
          )}
        </div>
        {item.sparkline_value != null && (
          <div style={{
            fontSize: 11, display: "flex", alignItems: "center", gap: 5,
            paddingLeft: 14,
            color: accent, fontWeight: 600,
            fontFamily: "var(--font-jetbrains), monospace",
          }}>
            <span>{fmtPct(item.sparkline_value)}</span>
            {item.week_grade && (
              <>
                <span style={{ color: "var(--ink-4)", fontWeight: 400 }}>·</span>
                <span style={{ color: "var(--ink-3)", fontWeight: 500 }}>{item.week_grade}</span>
              </>
            )}
          </div>
        )}
      </button>

      {/* Pin button — surfaces on hover or when pinned. Hard-deletes when
          clicked (parent fires the API call); the visible state flips
          optimistically via the parent's state update + props change. */}
      {(hover || item.pinned) && item.id != null && (
        <button onClick={(e) => { e.stopPropagation(); onPinToggle(); }}
                aria-label={item.pinned ? "Unpin" : "Pin"}
                title={item.pinned ? "Unpin" : "Pin"}
                data-testid="rail-pin-btn"
                style={{
                  position: "absolute", right: 6, top: 8,
                  width: 22, height: 22, borderRadius: 6,
                  display: "grid", placeItems: "center",
                  background: hover ? "var(--bg)" : "transparent",
                  color: item.pinned ? "#f59f00" : "var(--ink-4)",
                  border: "none", cursor: "pointer",
                }}>
          <StarIcon filled={!!item.pinned} size={13} />
        </button>
      )}
    </div>
  );
}

// ─── MonthFolder ───────────────────────────────────────────────────────
// Renders one month worth of items under a collapsible header. The header
// shows the month name, an aggregate return summary, and the inline SVG
// sparkline. Children render in a left-bordered indent.

function MonthFolder({
  year, month, items, defaultOpen, isCurrent, forceOpen,
  storageKey, activeKey, focusedKey, onItemClick, onPinToggle,
}: {
  year: number; month: number; items: NotesRailItem[];
  defaultOpen: boolean; isCurrent: boolean;
  /** When true (e.g. an active search query), override the user's
   *  stored collapse state so matches are visible. */
  forceOpen?: boolean;
  storageKey: string;
  activeKey: string | null; focusedKey: string | null;
  onItemClick: (item: NotesRailItem) => void;
  onPinToggle: (item: NotesRailItem) => void;
}) {
  const folderKey = `${year}-${String(month).padStart(2, "0")}`;
  const [storedOpen, setStoredOpen] = useState(() => {
    const stored = loadLs(storageKey, {})[folderKey];
    return stored == null ? defaultOpen : !!stored;
  });
  useEffect(() => {
    const map = loadLs(storageKey, {});
    map[folderKey] = storedOpen;
    saveLs(storageKey, map);
  }, [storedOpen, folderKey, storageKey]);
  // Visual openness is the stored state OR-forced. The toggle still
  // writes to the stored state so collapse persists once search clears.
  const open = forceOpen || storedOpen;
  const setOpen = setStoredOpen;

  const sparklineValues = items.map(i => i.sparkline_value);
  const realValues = sparklineValues.filter((v): v is number => v != null);
  const totalPct = realValues.length > 0
    ? realValues.reduce((acc, v) => acc * (1 + v / 100), 1) - 1
    : null;
  const totalColor = totalPct == null
    ? "var(--ink-3)"
    : (totalPct >= 0 ? "#08a86b" : "#e5484d");
  const winCount = realValues.filter(v => v > 0).length;

  return (
    <div data-testid="month-folder"
         data-month-key={folderKey}
         data-open={open ? "true" : "false"}
         style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <button onClick={() => setOpen(o => !o)}
              style={{
                width: "100%", display: "flex", alignItems: "center", gap: 8,
                padding: "8px 8px 8px 6px", borderRadius: 8,
                background: "transparent", textAlign: "left",
                cursor: "pointer", border: "none", color: "var(--ink)",
                transition: "background 120ms",
              }}
              onMouseEnter={e => e.currentTarget.style.background = "var(--surface)"}
              onMouseLeave={e => e.currentTarget.style.background = "transparent"}>
        <span style={{
          display: "inline-flex", color: "var(--ink-4)",
          transform: open ? "rotate(90deg)" : "none",
          transition: "transform 150ms", width: 12,
        }}><ChevronRight /></span>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ fontSize: 13, fontWeight: 600 }}>
              {MONTH_NAMES[month - 1]}
            </span>
            {isCurrent && (
              <span style={{
                fontSize: 9, fontWeight: 700, letterSpacing: "0.08em",
                textTransform: "uppercase", color: "#b45309",
                padding: "1px 6px", borderRadius: 999,
                background: "color-mix(in oklab, #f59f00 14%, var(--surface))",
                border: "1px solid color-mix(in oklab, #f59f00 30%, var(--border))",
              }}>Current</span>
            )}
          </div>
          <div style={{
            display: "flex", alignItems: "center", gap: 5,
            marginTop: 2, fontSize: 10.5,
          }}>
            <span style={{
              color: totalColor, fontWeight: 600,
              fontFamily: "var(--font-jetbrains), monospace",
            }}>
              {totalPct == null ? "—" : fmtPct(totalPct * 100)}
            </span>
            <span style={{ color: "var(--ink-4)" }}>·</span>
            <span style={{ color: "var(--ink-3)" }}>{items.length}w</span>
            <span style={{ color: "var(--ink-4)" }}>·</span>
            <span style={{ color: "var(--ink-3)" }}>{winCount}W</span>
          </div>
        </div>
        <span style={{ flexShrink: 0, opacity: 0.92 }}>
          <MonthSparkline values={sparklineValues} />
        </span>
      </button>
      {open && (
        <div data-testid="month-folder-body"
             style={{
               display: "flex", flexDirection: "column", gap: 2,
               position: "relative", paddingLeft: 8,
             }}>
          <span aria-hidden style={{
            position: "absolute", left: 14, top: 4, bottom: 4,
            width: 1, background: "var(--border)",
          }} />
          {items.map(it => (
            <RailItem key={it.key} item={it}
                      active={activeKey === it.key}
                      focused={focusedKey === it.key}
                      onClick={() => onItemClick(it)}
                      onPinToggle={() => onPinToggle(it)} />
          ))}
        </div>
      )}
    </div>
  );
}

// ─── YearFolder ────────────────────────────────────────────────────────
// Wraps past-year months in a top-bordered collapsible. Defaults closed.

function YearFolder({
  year, months, storageKey, forceOpen,
  activeKey, focusedKey, onItemClick, onPinToggle,
}: {
  year: number; months: { month: number; items: NotesRailItem[] }[];
  storageKey: string;
  /** When true, override stored collapse so search matches surface. */
  forceOpen?: boolean;
  activeKey: string | null; focusedKey: string | null;
  onItemClick: (item: NotesRailItem) => void;
  onPinToggle: (item: NotesRailItem) => void;
}) {
  const yearKey = String(year);
  const [storedOpen, setStoredOpen] = useState(() => !!loadLs(storageKey, {})[yearKey]);
  useEffect(() => {
    const map = loadLs(storageKey, {});
    map[yearKey] = storedOpen;
    saveLs(storageKey, map);
  }, [storedOpen, yearKey, storageKey]);
  const open = forceOpen || storedOpen;
  const setOpen = setStoredOpen;

  const allItems = months.flatMap(m => m.items);
  const realValues = allItems
    .map(i => i.sparkline_value)
    .filter((v): v is number => v != null);
  const totalPct = realValues.length > 0
    ? realValues.reduce((acc, v) => acc * (1 + v / 100), 1) - 1
    : null;
  const totalColor = totalPct == null
    ? "var(--ink-3)"
    : (totalPct >= 0 ? "#08a86b" : "#e5484d");

  return (
    <div data-testid="year-folder" data-year={yearKey}
         data-open={open ? "true" : "false"}
         style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <button onClick={() => setOpen(o => !o)}
              style={{
                width: "100%", display: "flex", alignItems: "center", gap: 8,
                padding: "10px 8px 10px 6px", borderRadius: 8,
                background: "transparent", textAlign: "left", cursor: "pointer",
                borderTop: "1px solid var(--border)", border: "none",
                marginTop: 6, paddingTop: 14, color: "var(--ink)",
              }}>
        <span style={{
          display: "inline-flex", color: "var(--ink-4)",
          transform: open ? "rotate(90deg)" : "none",
          transition: "transform 150ms", width: 12,
        }}><ChevronRight /></span>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{
            fontSize: 13, fontWeight: 700,
            fontFamily: "var(--font-fraunces), Georgia, serif",
            fontStyle: "italic", color: "var(--ink-2)",
          }}>{year}</div>
          <div style={{
            display: "flex", alignItems: "center", gap: 5,
            marginTop: 2, fontSize: 10.5,
          }}>
            <span style={{
              color: totalColor, fontWeight: 600,
              fontFamily: "var(--font-jetbrains), monospace",
            }}>
              {totalPct == null ? "—" : fmtPct(totalPct * 100)}
            </span>
            <span style={{ color: "var(--ink-4)" }}>·</span>
            <span style={{ color: "var(--ink-3)" }}>{allItems.length}w</span>
          </div>
        </div>
        <span style={{
          fontSize: 9, color: "var(--ink-4)", fontWeight: 600,
          letterSpacing: "0.08em", textTransform: "uppercase",
        }}>{months.length} mo</span>
      </button>
      {open && (
        <div style={{ display: "flex", flexDirection: "column", gap: 2, paddingLeft: 8 }}>
          {months.map(m => (
            <MonthFolder key={m.month} year={year} month={m.month} items={m.items}
                         defaultOpen={false} isCurrent={false}
                         forceOpen={forceOpen}
                         storageKey={storageKey}
                         activeKey={activeKey} focusedKey={focusedKey}
                         onItemClick={onItemClick} onPinToggle={onPinToggle} />
          ))}
        </div>
      )}
    </div>
  );
}

// ─── PinnedSection ─────────────────────────────────────────────────────

function PinnedSection({
  items, activeKey, focusedKey, onItemClick, onPinToggle,
}: {
  items: NotesRailItem[];
  activeKey: string | null; focusedKey: string | null;
  onItemClick: (item: NotesRailItem) => void;
  onPinToggle: (item: NotesRailItem) => void;
}) {
  if (items.length === 0) return null;
  return (
    <div data-testid="pinned-section"
         style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div style={{
        display: "flex", alignItems: "center", gap: 6, padding: "0 8px",
      }}>
        <span style={{ color: "#f59f00", display: "inline-flex" }}>
          <StarIcon filled size={12} />
        </span>
        <span style={{
          fontSize: 9, fontWeight: 600, letterSpacing: "0.10em",
          textTransform: "uppercase", color: "var(--ink-4)",
        }}>Pinned</span>
        <span style={{ fontSize: 10, color: "var(--ink-4)" }}>· {items.length}</span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        {items.map(it => (
          <RailItem key={"pin-" + it.key} item={it}
                    active={activeKey === it.key}
                    focused={focusedKey === it.key}
                    onClick={() => onItemClick(it)}
                    onPinToggle={() => onPinToggle(it)} />
        ))}
      </div>
    </div>
  );
}

// ─── CollapsedRail ─────────────────────────────────────────────────────
// 56px sliver. Expanding via the hamburger button restores the full rail.

function CollapsedRail({ onExpand }: { onExpand: () => void }) {
  return (
    <aside data-testid="notes-rail-collapsed"
           style={{
             width: 56, flexShrink: 0,
             background: "var(--bg)",
             borderRight: "1px solid var(--border)",
             display: "flex", flexDirection: "column", alignItems: "center",
             padding: "12px 0", gap: 12, alignSelf: "stretch",
           }}>
      <button onClick={onExpand} aria-label="Expand rail" title="Expand rail"
              style={{
                width: 32, height: 32, borderRadius: 8,
                background: "var(--surface)", border: "1px solid var(--border)",
                display: "grid", placeItems: "center", color: "var(--ink-3)",
                cursor: "pointer",
              }}>
        <HamburgerIcon />
      </button>
    </aside>
  );
}

// ─── Main export ───────────────────────────────────────────────────────

export function NotesRail({
  entityType, items, ytdStats, currentEntityKey,
  onItemClick, onPinToggle, collapsible = true,
}: NotesRailProps) {
  const collapseKey = COLLAPSED_KEY_PREFIX + entityType + "-collapsed";
  const folderKey   = COLLAPSED_KEY_PREFIX + entityType + "-collapsed-folders";

  const [collapsed, setCollapsed] = useState<boolean>(() => !!loadLs(collapseKey, false));
  useEffect(() => { saveLs(collapseKey, collapsed); }, [collapsed, collapseKey]);

  const [query, setQuery] = useState("");
  const [focusedKey, setFocusedKey] = useState<string | null>(null);
  // Optimistic pin overrides: maps item.id → desired pinned bool. Cleared
  // when the parent re-renders with updated items.
  const [pinOverrides, setPinOverrides] = useState<Map<number, boolean>>(() => new Map());
  useEffect(() => { setPinOverrides(new Map()); }, [items]);

  // Apply optimistic overrides on read so the UI reflects the in-flight
  // toggle while the API call is pending.
  const itemsWithOverrides = useMemo(() => {
    if (pinOverrides.size === 0) return items;
    return items.map(it => {
      if (it.id != null && pinOverrides.has(it.id)) {
        return { ...it, pinned: pinOverrides.get(it.id)! };
      }
      return it;
    });
  }, [items, pinOverrides]);

  const handlePinToggle = useCallback(async (item: NotesRailItem) => {
    if (item.id == null) return;
    const next = !item.pinned;
    setPinOverrides(prev => {
      const m = new Map(prev);
      m.set(item.id!, next);
      return m;
    });
    try {
      await onPinToggle(item.id, item.pinned);
    } catch {
      // Roll back the optimistic flip; parent didn't accept the change.
      setPinOverrides(prev => {
        const m = new Map(prev);
        m.delete(item.id!);
        return m;
      });
    }
  }, [onPinToggle]);

  // Filter by search query — substring on title, case-insensitive. Empty
  // query passes everything through.
  const filtered = useMemo(() => {
    if (!query.trim()) return itemsWithOverrides;
    const needle = query.toLowerCase().trim();
    return itemsWithOverrides.filter(it => it.title.toLowerCase().includes(needle));
  }, [itemsWithOverrides, query]);

  // Group by year → month for rendering. Current year inlines its months
  // at the top level (matching design); past years wrap in YearFolder.
  const grouped = useMemo(() => {
    const byYear = new Map<number, Map<number, NotesRailItem[]>>();
    for (const it of filtered) {
      if (!byYear.has(it.year)) byYear.set(it.year, new Map());
      const m = byYear.get(it.year)!;
      if (!m.has(it.month)) m.set(it.month, []);
      m.get(it.month)!.push(it);
    }
    return byYear;
  }, [filtered]);

  const pinnedItems = filtered.filter(it => it.pinned);

  // Current month is the most recent (year, month) pair across all items.
  const currentYearMonth = useMemo(() => {
    if (items.length === 0) return null;
    return { year: items[0].year, month: items[0].month };
  }, [items]);

  const currentYear = currentYearMonth?.year ?? new Date().getFullYear();

  // Keyboard nav: ArrowUp / k → previous, ArrowDown / j → next. Skip when
  // focus is in a text input so the search bar isn't hijacked. Operates
  // on the visible (filtered) list, in newest-first source order.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName?.toLowerCase() ?? "";
      if (tag === "input" || tag === "textarea") return;
      if (filtered.length === 0) return;
      const idx = focusedKey != null
        ? filtered.findIndex(it => it.key === focusedKey)
        : filtered.findIndex(it => it.key === currentEntityKey);
      const at = idx === -1 ? 0 : idx;
      if (e.key === "ArrowDown" || e.key === "j") {
        e.preventDefault();
        const next = filtered[Math.min(filtered.length - 1, at + 1)];
        if (next) setFocusedKey(next.key);
      } else if (e.key === "ArrowUp" || e.key === "k") {
        e.preventDefault();
        const prev = filtered[Math.max(0, at - 1)];
        if (prev) setFocusedKey(prev.key);
      } else if (e.key === "Enter" && focusedKey) {
        const found = filtered.find(it => it.key === focusedKey);
        if (found) {
          e.preventDefault();
          onItemClick(found);
        }
      } else if (e.key === "Escape") {
        setFocusedKey(null);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [filtered, focusedKey, currentEntityKey, onItemClick]);

  // Hide entirely below lg: (1024px) per locked decision. Mobile gets
  // no rail; the date picker in the page still handles week navigation.
  const wrapperClass = "hidden lg:flex";

  if (collapsed && collapsible) {
    return (
      <div className={wrapperClass}>
        <CollapsedRail onExpand={() => setCollapsed(false)} />
      </div>
    );
  }

  return (
    <aside data-testid="notes-rail"
           className={wrapperClass}
           style={{
             width: 296, flexShrink: 0,
             background: "var(--bg)",
             borderRight: "1px solid var(--border)",
             padding: "16px 12px 0",
             flexDirection: "column", gap: 12,
             alignSelf: "stretch",
             // The wrapper class drives display via Tailwind's lg:flex,
             // but the rail's own layout direction is column.
           }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{
            fontSize: 9, fontWeight: 600, letterSpacing: "0.10em",
            textTransform: "uppercase", color: "var(--ink-4)",
          }}>
            {entityType === "weekly_retro" ? "Weekly Notes" : "Daily Notes"}
          </div>
          <div style={{ fontSize: 13, fontWeight: 600, marginTop: 2 }}>
            {items.length} {entityType === "weekly_retro" ? "weeks" : "days"}
          </div>
        </div>
        {collapsible && (
          <button onClick={() => setCollapsed(true)}
                  aria-label="Collapse rail" title="Collapse rail"
                  data-testid="rail-collapse-btn"
                  style={{
                    width: 28, height: 28, borderRadius: 8,
                    background: "transparent", color: "var(--ink-4)",
                    display: "grid", placeItems: "center", border: "none",
                    cursor: "pointer",
                  }}>
            <CollapseIcon />
          </button>
        )}
      </div>

      {/* Search */}
      <div style={{ position: "relative" }}>
        <span style={{
          position: "absolute", left: 10, top: "50%", transform: "translateY(-50%)",
          color: "var(--ink-4)", display: "inline-flex",
        }}><SearchIcon /></span>
        <input value={query} onChange={e => setQuery(e.target.value)}
               placeholder={entityType === "weekly_retro" ? "Search weeks…" : "Search days…"}
               aria-label="Search"
               data-testid="rail-search"
               style={{
                 width: "100%", height: 32, paddingLeft: 30, paddingRight: 10,
                 borderRadius: 8,
                 background: "var(--surface)", border: "1px solid var(--border)",
                 color: "var(--ink)", fontSize: 12, outline: "none",
               }} />
      </div>

      {/* List */}
      <div style={{
        display: "flex", flexDirection: "column", gap: 8,
        overflowY: "auto", flex: 1, marginTop: 2, paddingBottom: 12,
      }}>
        <PinnedSection items={pinnedItems}
                       activeKey={currentEntityKey} focusedKey={focusedKey}
                       onItemClick={onItemClick}
                       onPinToggle={handlePinToggle} />

        {/* Current year months — inline. Force folders open while a
            search query is active so matches surface even if the user
            had previously collapsed the month. */}
        {grouped.has(currentYear) && (
          <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            {Array.from(grouped.get(currentYear)!.entries())
              .sort((a, b) => b[0] - a[0])
              .map(([month, monthItems]) => (
                <MonthFolder key={month} year={currentYear} month={month} items={monthItems}
                             defaultOpen={currentYearMonth?.month === month}
                             isCurrent={currentYearMonth?.month === month}
                             forceOpen={!!query.trim()}
                             storageKey={folderKey}
                             activeKey={currentEntityKey} focusedKey={focusedKey}
                             onItemClick={onItemClick} onPinToggle={handlePinToggle} />
              ))}
          </div>
        )}

        {/* Past years wrapped */}
        {Array.from(grouped.keys())
          .filter(y => y !== currentYear)
          .sort((a, b) => b - a)
          .map(year => {
            const months = Array.from(grouped.get(year)!.entries())
              .sort((a, b) => b[0] - a[0])
              .map(([month, monthItems]) => ({ month, items: monthItems }));
            return (
              <YearFolder key={year} year={year} months={months}
                          forceOpen={!!query.trim()}
                          storageKey={folderKey}
                          activeKey={currentEntityKey} focusedKey={focusedKey}
                          onItemClick={onItemClick} onPinToggle={handlePinToggle} />
            );
          })}

        {/* Empty state */}
        {items.length === 0 && (
          <div data-testid="rail-empty-state"
               style={{
                 padding: "20px 12px", textAlign: "center",
                 fontSize: 12, color: "var(--ink-4)",
               }}>
            No {entityType === "weekly_retro" ? "weekly retros" : "daily notes"} yet
          </div>
        )}

        {/* Search miss state — separate from empty because items > 0 */}
        {items.length > 0 && filtered.length === 0 && (
          <div data-testid="rail-no-search-results"
               style={{
                 padding: "20px 12px", textAlign: "center",
                 fontSize: 12, color: "var(--ink-4)",
               }}>
            No matches.
          </div>
        )}
      </div>

      {/* Footer YTD stats — sticky bottom */}
      <div style={{
        marginTop: "auto",
        padding: "10px 8px",
        borderTop: "1px solid var(--border)",
        background: "var(--bg)",
        marginLeft: -12, marginRight: -12,
        paddingLeft: 16, paddingRight: 16,
      }}>
        <div style={{
          fontSize: 9, fontWeight: 600, letterSpacing: "0.10em",
          textTransform: "uppercase", color: "var(--ink-4)", marginBottom: 4,
        }}>{currentYear} YTD</div>
        <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
          <span style={{
            fontSize: 16, fontWeight: 600,
            fontFamily: "var(--font-jetbrains), monospace",
            color: "var(--ink)",
          }}>{ytdStats.avg_grade ?? "—"}</span>
          <span style={{ fontSize: 11, color: "var(--ink-3)" }}>
            · {ytdStats.weeks_graded}/{ytdStats.total_weeks} graded
          </span>
          {ytdStats.weeks_pinned > 0 && (
            <span style={{ fontSize: 11, color: "var(--ink-4)" }}>
              · {ytdStats.weeks_pinned} pinned
            </span>
          )}
        </div>
      </div>
    </aside>
  );
}
