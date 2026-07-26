// Pure helpers for the Trading Checklist page (Migration 050). Grouping,
// labels, and display-value formatting live here so the desktop + mobile
// components share one derivation and the logic can be unit-tested without
// rendering React.
//
// The backend returns items pre-sorted by (frequency, slot, sort_order),
// so the grouper just walks the list once and buckets by frequency + slot
// without re-sorting. Overdue is already derived server-side per
// db_layer._routine_overdue_days — this lib only formats it for display.

import type { RoutineFrequency, RoutineItem, RoutineSlot } from "@/lib/api";

export const FREQUENCY_ORDER: RoutineFrequency[] = [
  "daily",
  "weekly",
  "monthly",
  "quarterly",
];

export const SLOT_ORDER: RoutineSlot[] = [
  "premarket",
  "intraday",
  "end_of_shift",
  "after_close",
  "weekend",
];

export const FREQUENCY_LABELS: Record<RoutineFrequency, string> = {
  daily: "Daily",
  weekly: "Weekly",
  monthly: "Monthly",
  quarterly: "Quarterly",
};

export const SLOT_LABELS: Record<RoutineSlot, string> = {
  premarket: "Pre-market",
  intraday: "Intraday",
  end_of_shift: "End of shift",
  after_close: "After close",
  weekend: "Weekend",
};

export interface RoutineGroup {
  frequency: RoutineFrequency;
  /** null for monthly + quarterly items which have no time-of-day slot. */
  slot: RoutineSlot | null;
  /** "Daily · after close" — for section headers. */
  label: string;
  items: RoutineItem[];
}

/** Group items into ordered (frequency, slot) buckets. Input already
 *  sorted server-side; this walks once and buckets. */
export function groupRoutineItems(items: RoutineItem[]): RoutineGroup[] {
  const buckets = new Map<string, RoutineGroup>();
  const key = (f: RoutineFrequency, s: RoutineSlot | null) => `${f}|${s ?? ""}`;
  for (const item of items) {
    const k = key(item.frequency, item.slot);
    let g = buckets.get(k);
    if (!g) {
      g = {
        frequency: item.frequency,
        slot: item.slot,
        label: formatGroupLabel(item.frequency, item.slot),
        items: [],
      };
      buckets.set(k, g);
    }
    g.items.push(item);
  }
  return Array.from(buckets.values()).sort((a, b) => {
    const fA = FREQUENCY_ORDER.indexOf(a.frequency);
    const fB = FREQUENCY_ORDER.indexOf(b.frequency);
    if (fA !== fB) return fA - fB;
    const sA = a.slot ? SLOT_ORDER.indexOf(a.slot) : 99;
    const sB = b.slot ? SLOT_ORDER.indexOf(b.slot) : 99;
    return sA - sB;
  });
}

export function formatGroupLabel(
  frequency: RoutineFrequency,
  slot: RoutineSlot | null,
): string {
  if (slot === null) return FREQUENCY_LABELS[frequency];
  return `${FREQUENCY_LABELS[frequency]} · ${SLOT_LABELS[slot].toLowerCase()}`;
}

/** Rendered display for an item's status column.
 *  - counter → "log when it happens" (no cadence)
 *  - overdue → "N days" (danger token)
 *  - ticked_today (task) → "ticked today"
 *  - last_run present → "ran <MMM D>"
 *  - never run → "—" */
export type StatusChip =
  | { kind: "overdue"; text: string; days: number }
  | { kind: "counter"; text: string }
  | { kind: "today"; text: string }
  | { kind: "last_run"; text: string; date: string }
  | { kind: "never"; text: string };

export function itemStatusChip(item: RoutineItem): StatusChip {
  if (item.item_type === "counter") {
    return { kind: "counter", text: "log when it happens" };
  }
  if (item.overdue_days != null && item.overdue_days > 0) {
    const label = item.overdue_days === 1 ? "1 day" : `${item.overdue_days} days`;
    return { kind: "overdue", text: label, days: item.overdue_days };
  }
  if (item.ticked_today) {
    return { kind: "today", text: "ticked today" };
  }
  if (item.last_run_date) {
    return {
      kind: "last_run",
      text: `ran ${formatShortDate(item.last_run_date)}`,
      date: item.last_run_date,
    };
  }
  return { kind: "never", text: "—" };
}

/** "2026-07-25" → "25 Jul". Parses as UTC to avoid a client-TZ shift on
 *  dates that were computed server-side in America/Chicago. */
export function formatShortDate(iso: string): string {
  // iso is YYYY-MM-DD. Splitting avoids Date parsing which drags in the
  // browser's TZ and can shift the day by one at boundaries.
  const [y, m, d] = iso.split("-").map(Number);
  if (!y || !m || !d) return iso;
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  return `${d} ${months[m - 1] ?? ""}`.trim();
}

/** Compact today-only summary — powers the mobile "Later today" collapsed
 *  card. Counts task items due (not-yet-ticked-today) per slot for the
 *  daily frequency. Counter items are excluded (never "due"). */
export interface TodayCounts {
  premarket: number;
  intraday: number;
  end_of_shift: number;
  after_close: number;
  /** Sum of the above — one-glance "how many items outstanding today". */
  total: number;
}

export function countTodayDue(items: RoutineItem[]): TodayCounts {
  const counts: TodayCounts = {
    premarket: 0,
    intraday: 0,
    end_of_shift: 0,
    after_close: 0,
    total: 0,
  };
  for (const item of items) {
    if (item.frequency !== "daily") continue;
    if (item.item_type !== "task") continue;
    if (item.ticked_today) continue;
    const slot = item.slot;
    if (slot === "premarket") counts.premarket++;
    else if (slot === "intraday") counts.intraday++;
    else if (slot === "end_of_shift") counts.end_of_shift++;
    else if (slot === "after_close") counts.after_close++;
    else continue;
    counts.total++;
  }
  return counts;
}
