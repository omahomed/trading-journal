// Fire-and-forget auto-tick for routine_items when the user completes
// an action that IS the tick. NLV Entry save → "Equity routine";
// Recap / Thoughts save → "Journal". Never blocks the save — a network
// failure or a missing seed item logs at debug and moves on.
//
// Matching is by name-prefix against the item's `name` field. The seed
// items in db_layer._ROUTINE_SYSTEM_ITEMS carry stable prefixes ("Equity
// routine — …", "Journal — …"); if a rename ever ships, update the
// SYSTEM_ITEM_PREFIXES map below.
//
// Cache: items list is fetched once per module lifetime and reused.
// System items don't churn; custom items don't participate in autotick.
// A stale cache after a manual delete/add is harmless — worst case is a
// missed tick that the user can add manually.

import { api, type RoutineItem } from "@/lib/api";
import { log } from "@/lib/log";

/** Stable name-prefixes for system items that participate in autotick. */
export const SYSTEM_ITEM_PREFIXES = {
  equityRoutine: "Equity routine",
  journal: "Journal",
} as const;

let _cache: RoutineItem[] | null = null;

async function getItems(): Promise<RoutineItem[]> {
  if (_cache) return _cache;
  try {
    const res = await api.routineItemsList();
    if ("error" in res) return [];
    _cache = res.items;
    return res.items;
  } catch {
    return [];
  }
}

/** Test-only: reset the module cache. */
export function __resetAutotickCache() {
  _cache = null;
}

/** Today's date in America/Chicago as YYYY-MM-DD. Matches the server-
 *  side `_routine_today_ct` — a tick lives against this date so autotick
 *  callers can gate on "am I editing today?" before firing. */
export function todayInChicago(): string {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/Chicago",
    year: "numeric", month: "2-digit", day: "2-digit",
  }).format(new Date());
}

/** Silently tick the first item whose name starts with `prefix`. Returns
 *  true if a tick was attempted (item found), false if no match. Errors
 *  from the tick network call are swallowed — the caller's save has
 *  already succeeded; a failed autotick is a UX degradation, not a bug. */
export async function autoTickByPrefix(prefix: string): Promise<boolean> {
  const items = await getItems();
  const item = items.find(i => i.name.startsWith(prefix));
  if (!item) {
    log.debug.devOnly("routine-autotick", `no item matches prefix "${prefix}"`, null);
    return false;
  }
  try {
    await api.routineLogTick(item.id);
    return true;
  } catch (e) {
    log.debug.devOnly("routine-autotick", `tick failed for item ${item.id}`, e);
    return false;
  }
}
