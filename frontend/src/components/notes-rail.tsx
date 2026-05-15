"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { formatCurrency } from "@/lib/format";
import type {
  NotesRailEntityType, NotesRailItem, NotesRailItemTag, NotesRailYtdStats,
} from "@/lib/api";
import { TagPill } from "./tag-pill";
import { TAG_PALETTE, type TagTone } from "@/lib/tag-palette";

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

function CalendarIcon() {
  return (
    <svg width={14} height={14} viewBox="0 0 24 24" fill="none"
         stroke="currentColor" strokeWidth={2}
         strokeLinecap="round" strokeLinejoin="round">
      <rect x={3} y={4} width={18} height={18} rx={2} ry={2} />
      <line x1={16} y1={2} x2={16} y2={6} />
      <line x1={8} y1={2} x2={8} y2={6} />
      <line x1={3} y1={10} x2={21} y2={10} />
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

function FilterIcon() {
  return (
    <svg width={11} height={11} viewBox="0 0 24 24" fill="none"
         stroke="currentColor" strokeWidth={2.4}
         strokeLinecap="round" strokeLinejoin="round">
      <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
    </svg>
  );
}

function CloseXIcon({ size = 10 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
         stroke="currentColor" strokeWidth={2.5}
         strokeLinecap="round" strokeLinejoin="round">
      <line x1={18} y1={6} x2={6} y2={18} />
      <line x1={6} y1={6} x2={18} y2={18} />
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
        {/* Unified per-row stats line — same shape at every level of the
            rail: {return%} · {$P&L} · {N trades}. Each segment is
            individually optional (current week may have % but no closed
            trades; back-fill weeks may have % but null pnl). The line
            renders whenever any of the three has a real value. */}
        {(item.sparkline_value != null || item.weekly_pnl != null || item.trades_count > 0) && (
          <div style={{
            fontSize: 11, display: "flex", alignItems: "center", gap: 5,
            paddingLeft: 14,
            fontFamily: "var(--font-jetbrains), monospace",
          }}>
            {item.sparkline_value != null && (
              <span style={{ color: accent, fontWeight: 600 }}>
                {fmtPct(item.sparkline_value)}
              </span>
            )}
            {item.weekly_pnl != null && (
              <>
                {item.sparkline_value != null && (
                  <span style={{ color: "var(--ink-4)", fontWeight: 400 }}>·</span>
                )}
                <span style={{ color: accent, fontWeight: 600 }}>
                  {formatCurrency(item.weekly_pnl, { showSign: true, compact: true })}
                </span>
              </>
            )}
            {item.trades_count > 0 && (
              <>
                <span style={{ color: "var(--ink-4)", fontWeight: 400 }}>·</span>
                <span style={{ color: "var(--ink-3)", fontWeight: 500 }}>
                  {item.trades_count}T
                </span>
              </>
            )}
            {item.week_grade && (
              <>
                <span style={{ color: "var(--ink-4)", fontWeight: 400 }}>·</span>
                <span style={{ color: "var(--ink-3)", fontWeight: 500 }}>{item.week_grade}</span>
              </>
            )}
          </div>
        )}
        {/* Tag chips — reuse the Phase 1 TagPill (same visual language as
            the page's tag bar). Caps at first 3 for the rail's narrow
            column; overflow renders "+N more". */}
        {item.tags && item.tags.length > 0 && (
          <div style={{
            display: "flex", flexWrap: "wrap", gap: 4, paddingLeft: 14,
            marginTop: 2,
          }}>
            {item.tags.slice(0, 3).map((t, i) => (
              <span key={i} style={{ transform: "scale(0.85)", transformOrigin: "left center" }}>
                <TagPill label={t.name} tone={t.color as TagTone} />
              </span>
            ))}
            {item.tags.length > 3 && (
              <span style={{ fontSize: 10, color: "var(--ink-4)" }}>
                +{item.tags.length - 3}
              </span>
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
  // Aggregate $P&L + trades count across the month's child weeks for the
  // unified stats line. Both sums skip nulls / zeros naturally.
  const monthPnl = items.reduce(
    (acc, it) => acc + (it.weekly_pnl ?? 0), 0,
  );
  const hasPnl = items.some(it => it.weekly_pnl != null);
  const monthTrades = items.reduce((acc, it) => acc + (it.trades_count ?? 0), 0);
  const pnlColor = monthPnl >= 0 ? "#08a86b" : "#e5484d";

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
            fontFamily: "var(--font-jetbrains), monospace",
          }}>
            <span style={{ color: totalColor, fontWeight: 600 }}>
              {totalPct == null ? "—" : fmtPct(totalPct * 100)}
            </span>
            {hasPnl && (
              <>
                <span style={{ color: "var(--ink-4)", fontWeight: 400 }}>·</span>
                <span style={{ color: pnlColor, fontWeight: 600 }}>
                  {formatCurrency(monthPnl, { showSign: true, compact: true })}
                </span>
              </>
            )}
            {monthTrades > 0 && (
              <>
                <span style={{ color: "var(--ink-4)", fontWeight: 400 }}>·</span>
                <span style={{ color: "var(--ink-3)", fontWeight: 500 }}>
                  {monthTrades}T
                </span>
              </>
            )}
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
  // Same unified aggregation as MonthFolder, just one level up.
  const yearPnl = allItems.reduce(
    (acc, it) => acc + (it.weekly_pnl ?? 0), 0,
  );
  const hasYearPnl = allItems.some(it => it.weekly_pnl != null);
  const yearTrades = allItems.reduce((acc, it) => acc + (it.trades_count ?? 0), 0);
  const yearPnlColor = yearPnl >= 0 ? "#08a86b" : "#e5484d";

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
            fontFamily: "var(--font-jetbrains), monospace",
          }}>
            <span style={{ color: totalColor, fontWeight: 600 }}>
              {totalPct == null ? "—" : fmtPct(totalPct * 100)}
            </span>
            {hasYearPnl && (
              <>
                <span style={{ color: "var(--ink-4)", fontWeight: 400 }}>·</span>
                <span style={{ color: yearPnlColor, fontWeight: 600 }}>
                  {formatCurrency(yearPnl, { showSign: true, compact: true })}
                </span>
              </>
            )}
            {yearTrades > 0 && (
              <>
                <span style={{ color: "var(--ink-4)", fontWeight: 400 }}>·</span>
                <span style={{ color: "var(--ink-3)", fontWeight: 500 }}>
                  {yearTrades}T
                </span>
              </>
            )}
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

// ─── Filter bar (Phase 6.5) ────────────────────────────────────────────
// Derives its tag set from items[].tags (the Phase 1 polymorphic tags
// returned by /api/weekly-retros/list). Multi-select with OR semantics
// across tags; combined with the search input via AND. Hidden entirely
// when no item carries any tag — an empty filter affordance is worse
// than no affordance.

interface AvailableTag {
  name: string;
  color: string;   // TagTone key from the closed palette
  count: number;   // number of items carrying this tag
}

function ActiveTagChip({
  name, color, onRemove,
}: {
  name: string;
  color: string;
  onRemove: () => void;
}) {
  // Tone-darkened bg preserves the row-chip identity while contrasting
  // with the light-bodied row chips so the selected state reads at a
  // glance.
  const tone = TAG_PALETTE[color as TagTone] ?? TAG_PALETTE.sky;
  return (
    <span data-testid="rail-filter-chip"
          data-tag-name={name}
          style={{
            display: "inline-flex", alignItems: "center", gap: 4,
            height: 22, padding: "0 4px 0 8px",
            borderRadius: 999,
            background: tone.dot, color: "#fff",
            fontSize: 11, fontWeight: 600, whiteSpace: "nowrap",
          }}>
      <span>{name}</span>
      <button type="button"
              onClick={onRemove}
              aria-label={`Remove ${name} filter`}
              style={{
                width: 16, height: 16, borderRadius: 999,
                display: "grid", placeItems: "center",
                background: "transparent",
                color: "rgba(255,255,255,0.85)",
                border: "none", cursor: "pointer", padding: 0,
              }}>
        <CloseXIcon size={10} />
      </button>
    </span>
  );
}

function FilterPopover({
  availableTags, filters, toggle, onClear, onClose, anchorRef,
}: {
  availableTags: AvailableTag[];
  filters: string[];
  toggle: (name: string) => void;
  onClear: () => void;
  onClose: () => void;
  anchorRef: React.RefObject<HTMLDivElement | null>;
}) {
  const ref = useRef<HTMLDivElement>(null);

  // Click-outside + Esc — same idiom as ColorPicker / ToolbarDropdown.
  // Anchor element is excluded so clicking the trigger button itself
  // toggles cleanly instead of opening-then-closing on the same click.
  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (ref.current && ref.current.contains(target)) return;
      if (anchorRef.current && anchorRef.current.contains(target)) return;
      onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.preventDefault(); onClose(); }
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [onClose, anchorRef]);

  return (
    <div ref={ref}
         role="dialog"
         aria-label="Filter weeks by tag"
         data-testid="rail-filter-popover"
         style={{
           position: "absolute", top: "calc(100% + 6px)", left: 0, right: 0,
           zIndex: 40,
           background: "var(--surface)", border: "1px solid var(--border)",
           borderRadius: 12, boxShadow: "0 8px 28px rgba(14,20,38,0.14)",
           padding: 10, display: "flex", flexDirection: "column", gap: 8,
           maxHeight: 360,
         }}>
      <div style={{
        overflowY: "auto",
        display: "flex", flexDirection: "column",
        maxHeight: 280,
      }}>
        {availableTags.map(t => {
          const on = filters.includes(t.name);
          const tone = TAG_PALETTE[t.color as TagTone] ?? TAG_PALETTE.sky;
          return (
            <button key={t.name}
                    type="button"
                    data-testid="rail-filter-option"
                    data-tag-name={t.name}
                    data-selected={on ? "true" : "false"}
                    onClick={() => toggle(t.name)}
                    style={{
                      display: "flex", alignItems: "center", gap: 8,
                      padding: "6px 8px", borderRadius: 6,
                      fontSize: 12, color: "var(--ink-2)",
                      textAlign: "left", background: "transparent",
                      border: "none", cursor: "pointer",
                    }}>
              <span aria-hidden style={{
                width: 14, height: 14, borderRadius: 4,
                border: `1.5px solid ${on ? "var(--ink)" : "var(--border-2)"}`,
                background: on ? "var(--ink)" : "transparent",
                color: "#fff", display: "grid", placeItems: "center",
                flexShrink: 0,
              }}>
                {on && (
                  <svg width={10} height={10} viewBox="0 0 24 24" fill="none"
                       stroke="currentColor" strokeWidth={3}
                       strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                )}
              </span>
              <span aria-hidden style={{
                width: 6, height: 6, borderRadius: 999, background: tone.dot,
                flexShrink: 0,
              }} />
              <span style={{ flex: 1 }}>{t.name}</span>
              <span style={{
                fontSize: 11, color: "var(--ink-4)",
                fontFamily: "var(--font-jetbrains), monospace",
                fontVariantNumeric: "tabular-nums",
              }}>{t.count}</span>
            </button>
          );
        })}
      </div>
      <div style={{
        display: "flex", alignItems: "center", gap: 8,
        paddingTop: 6, borderTop: "1px solid var(--border)",
      }}>
        <button type="button"
                onClick={onClear}
                disabled={filters.length === 0}
                data-testid="rail-filter-clear"
                style={{
                  fontSize: 11, fontWeight: 500,
                  color: filters.length === 0 ? "var(--ink-4)" : "var(--ink-3)",
                  padding: "4px 8px", borderRadius: 6,
                  background: "transparent", border: "none",
                  cursor: filters.length === 0 ? "default" : "pointer",
                }}>Clear all</button>
        <span style={{ flex: 1 }} />
        <span style={{ fontSize: 11, color: "var(--ink-4)" }}>
          {filters.length} selected
        </span>
      </div>
    </div>
  );
}

function FilterBar({
  availableTags, filters, setFilters,
}: {
  availableTags: AvailableTag[];
  filters: string[];
  setFilters: React.Dispatch<React.SetStateAction<string[]>>;
}) {
  const [open, setOpen] = useState(false);
  const anchorRef = useRef<HTMLDivElement>(null);

  const toggle = useCallback((name: string) => {
    setFilters(prev => prev.includes(name)
      ? prev.filter(x => x !== name)
      : [...prev, name]);
  }, [setFilters]);

  // If a previously selected tag disappears from availableTags (e.g., its
  // last attached retro was deleted server-side) prune it from filters so
  // the bar doesn't show a chip that filters nothing.
  useEffect(() => {
    if (filters.length === 0) return;
    const names = new Set(availableTags.map(t => t.name));
    const next = filters.filter(n => names.has(n));
    if (next.length !== filters.length) setFilters(next);
  }, [availableTags, filters, setFilters]);

  // Empty-bar gate: hide entirely when no item carries a tag. An empty
  // affordance with no popover content reads as broken.
  if (availableTags.length === 0) return null;

  const colorFor = (name: string) =>
    availableTags.find(t => t.name === name)?.color ?? "sky";

  return (
    <div data-testid="rail-filter-bar" style={{ position: "relative" }}>
      <div ref={anchorRef}
           style={{
             display: "flex", alignItems: "center", flexWrap: "wrap", gap: 5,
             padding: "5px 6px 5px 8px",
             borderRadius: 10,
             background: "var(--surface)", border: "1px solid var(--border)",
             minHeight: 34,
           }}>
        <button type="button"
                onClick={() => setOpen(o => !o)}
                data-testid="rail-filter-trigger"
                aria-haspopup="dialog"
                aria-expanded={open}
                style={{
                  display: "inline-flex", alignItems: "center", gap: 5,
                  height: 22, padding: "0 8px",
                  borderRadius: 999, fontSize: 11, fontWeight: 600,
                  color: "var(--ink-2)", background: "var(--bg)",
                  border: "1px solid var(--border)", cursor: "pointer",
                }}>
          <FilterIcon />
          Filter
          {filters.length > 0 && (
            <span data-testid="rail-filter-count"
                  style={{
                    fontSize: 10, padding: "0 5px", borderRadius: 999,
                    background: "var(--ink)", color: "#fff",
                    minWidth: 14, textAlign: "center",
                  }}>{filters.length}</span>
          )}
        </button>
        {filters.length === 0 && (
          <span style={{ fontSize: 11, color: "var(--ink-4)" }}>All weeks</span>
        )}
        {filters.map(name => (
          <ActiveTagChip key={name} name={name}
                         color={colorFor(name)}
                         onRemove={() => toggle(name)} />
        ))}
      </div>
      {open && (
        <FilterPopover availableTags={availableTags}
                       filters={filters} toggle={toggle}
                       onClear={() => setFilters([])}
                       onClose={() => setOpen(false)}
                       anchorRef={anchorRef} />
      )}
    </div>
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
  // Phase 6.5: tag filter selections. Multi-select with OR semantics —
  // a week matches when it carries ANY selected tag. Combined with the
  // search query via AND in the `filtered` useMemo below.
  const [filters, setFilters] = useState<string[]>([]);
  const [focusedKey, setFocusedKey] = useState<string | null>(null);
  // Calendar jump-to-date input ref. Trigger picker via showPicker() on
  // supported browsers; falls back to a synthetic click for older Safari.
  const dateInputRef = useRef<HTMLInputElement | null>(null);
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

  // Derive the tag set for the filter bar from items[].tags. Dedupes by
  // name, counts uses, then orders by count desc / name asc — stable so
  // pills don't reorder on every render. Phase 1 enforces unique tag
  // names per portfolio, so name is a safe identity. Falls back to the
  // last-seen color when the same name appears with different colors
  // (server constraint should prevent this, but the UI shouldn't crash).
  const availableTags: AvailableTag[] = useMemo(() => {
    const byName = new Map<string, AvailableTag>();
    for (const it of items) {
      for (const t of it.tags ?? []) {
        const existing = byName.get(t.name);
        if (existing) {
          existing.count += 1;
        } else {
          byName.set(t.name, { name: t.name, color: t.color, count: 1 });
        }
      }
    }
    return Array.from(byName.values()).sort((a, b) =>
      b.count - a.count || a.name.localeCompare(b.name));
  }, [items]);

  // Filter by search (title substring) AND tag selection (OR within
  // selected tags). Search predicate runs first to take advantage of the
  // typically-larger early reduction; tag predicate runs second on the
  // already-shrunken set.
  const filtered = useMemo(() => {
    let out = itemsWithOverrides;
    if (query.trim()) {
      const needle = query.toLowerCase().trim();
      out = out.filter(it => it.title.toLowerCase().includes(needle));
    }
    if (filters.length > 0) {
      out = out.filter(it => (it.tags ?? []).some(t => filters.includes(t.name)));
    }
    return out;
  }, [itemsWithOverrides, query, filters]);

  // Folders auto-open when either search OR filters narrow the list —
  // otherwise a tag-match buried in a collapsed month would appear as a
  // false negative (the empty state would surface instead).
  const folderForceOpen = !!query.trim() || filters.length > 0;

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

  // Year span from earliest to latest item (inclusive). Drives the "·
  // {N} years" subtitle suffix. 0 → suppress (single-year or empty).
  const yearsSpanned = useMemo(() => {
    if (items.length === 0) return 0;
    const years = items.map(it => it.year);
    return Math.max(...years) - Math.min(...years) + 1;
  }, [items]);

  // Jump to the week containing a picked date. Snap to Monday and look
  // for the item by its key (week_start ISO). If no exact match (out of
  // range), pick the nearest by absolute distance — the user almost
  // certainly meant the closest known week.
  const handleJumpToDate = useCallback((iso: string) => {
    if (!iso || items.length === 0) return;
    const picked = new Date(iso + "T12:00:00");
    if (isNaN(picked.getTime())) return;
    const day = picked.getDay(); // 0=Sun..6=Sat
    const offset = day === 0 ? -6 : 1 - day;
    const monday = new Date(picked);
    monday.setDate(picked.getDate() + offset);
    const monStr = `${monday.getFullYear()}-${String(monday.getMonth() + 1).padStart(2, "0")}-${String(monday.getDate()).padStart(2, "0")}`;
    const exact = items.find(it => it.key === monStr);
    if (exact) { onItemClick(exact); return; }
    // Nearest fallback — distance in days, lowest wins.
    let best: NotesRailItem | null = null;
    let bestDelta = Infinity;
    const target = monday.getTime();
    for (const it of items) {
      const itDate = new Date(it.key + "T12:00:00").getTime();
      const delta = Math.abs(itDate - target);
      if (delta < bestDelta) { bestDelta = delta; best = it; }
    }
    if (best) onItemClick(best);
  }, [items, onItemClick]);

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
            {yearsSpanned > 0 && (
              <>
                <span style={{ color: "var(--ink-4)", margin: "0 4px" }}>·</span>
                <span style={{ color: "var(--ink-3)" }}>
                  {yearsSpanned} {yearsSpanned === 1 ? "year" : "years"}
                </span>
              </>
            )}
          </div>
        </div>
        {/* Calendar jump-to-date — native date input hidden behind an
            icon button so the rail header doesn't visually scream. On
            change, finds the week containing the picked date and fires
            onItemClick with that item. Phase 6.5 may replace this with a
            custom mini-calendar popover (design has one), but a native
            picker is cheap and works across desktop+mobile. */}
        <div style={{ position: "relative" }}>
          <button onClick={() => dateInputRef.current?.showPicker?.() ?? dateInputRef.current?.click()}
                  aria-label="Jump to date" title="Jump to date"
                  data-testid="rail-jump-date-btn"
                  style={{
                    width: 28, height: 28, borderRadius: 8,
                    background: "var(--surface)", border: "1px solid var(--border)",
                    display: "grid", placeItems: "center", color: "var(--ink-3)",
                    cursor: "pointer",
                  }}>
            <CalendarIcon />
          </button>
          <input ref={dateInputRef} type="date"
                 aria-label="Jump to date input"
                 data-testid="rail-jump-date-input"
                 onChange={e => handleJumpToDate(e.target.value)}
                 style={{
                   position: "absolute", inset: 0, opacity: 0, pointerEvents: "none",
                 }} />
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

      {/* Tag filter bar — hidden entirely when no item carries a tag. */}
      <FilterBar availableTags={availableTags}
                 filters={filters} setFilters={setFilters} />

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
                             forceOpen={folderForceOpen}
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
                          forceOpen={folderForceOpen}
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

        {/* Search miss state — separate from empty because items > 0.
            Copy + a Clear-filters affordance swap in when filters are
            active so the user has a one-click escape hatch out of an
            over-narrow filter set. */}
        {items.length > 0 && filtered.length === 0 && (
          <div data-testid="rail-no-search-results"
               style={{
                 padding: "20px 12px", textAlign: "center",
                 fontSize: 12, color: "var(--ink-4)",
                 display: "flex", flexDirection: "column",
                 alignItems: "center", gap: 8,
               }}>
            <span>
              {filters.length > 0
                ? "No weeks match the current filters."
                : "No matches."}
            </span>
            {filters.length > 0 && (
              <button type="button"
                      onClick={() => setFilters([])}
                      data-testid="rail-filter-clear-empty"
                      style={{
                        fontSize: 11, fontWeight: 600,
                        color: "var(--ink-3)",
                        background: "var(--surface)",
                        border: "1px solid var(--border)",
                        borderRadius: 6, padding: "4px 10px",
                        cursor: "pointer",
                      }}>Clear filters</button>
            )}
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
