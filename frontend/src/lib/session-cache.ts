// Versioned per-tab cache built on top of sessionStorage. Survives Cmd+R
// within the same tab, doesn't bleed across tabs, and is cleared automatically
// when the tab closes — so callers never have to think about cleanup.
//
// Why this exists: pages like Active Campaign Summary fan out 5+ API calls
// on mount, several of which hit yfinance. The user navigates between
// Dashboard / ACS / Trade Journal frequently and doesn't always want the
// freshest numbers — they want the page to *land* fast. A module-level
// cache covers most navigation patterns, but Cmd+R wipes it. This adds the
// last 5% of survival without bringing in a full data-fetching library.
//
// Each entry is namespaced by both:
//   - a per-cache `name` (e.g. "active-campaign")
//   - a per-cache `version` integer that callers bump when their stored
//     shape changes — guarantees a deploy with a different shape doesn't
//     return a stale, wrong-shape parse to the new code.
//
// Quota errors and parse errors fall through silently — the caller treats
// "no cache" the same as "fresh page," so any failure mode just means an
// extra fetch, never broken state.

const NAMESPACE = "mo-trading::session-cache::v1";

interface Envelope<T> {
  version: number;
  saved_at: number;       // epoch ms — caller can render "X min ago"
  payload: T;
}

function key(name: string): string {
  return `${NAMESPACE}::${name}`;
}

/**
 * Read a previously-cached payload. Returns null when:
 *   - sessionStorage is unavailable (SSR, blocked by browser)
 *   - the entry doesn't exist
 *   - the entry's version doesn't match the caller's expected version
 *   - the JSON fails to parse (corrupted)
 *
 * Callers should treat null as "no cache, fetch fresh."
 */
export function readCache<T>(name: string, expectedVersion: number): { payload: T; saved_at: number } | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.sessionStorage.getItem(key(name));
    if (!raw) return null;
    const env = JSON.parse(raw) as Envelope<T>;
    if (!env || env.version !== expectedVersion) return null;
    return { payload: env.payload, saved_at: env.saved_at };
  } catch {
    return null;
  }
}

/**
 * Persist a payload under `name`. Failures (quota exceeded, blocked storage,
 * unserializable shapes) are swallowed — the page already has the data in
 * memory, so a missed write is just "this entry won't survive Cmd+R."
 */
export function writeCache<T>(name: string, version: number, payload: T): void {
  if (typeof window === "undefined") return;
  try {
    const env: Envelope<T> = {
      version,
      saved_at: Date.now(),
      payload,
    };
    window.sessionStorage.setItem(key(name), JSON.stringify(env));
  } catch {
    // Quota or serialization error — caller has the data in memory anyway.
  }
}

/**
 * Drop the entry — used by Refresh buttons that explicitly want fresh data.
 */
export function clearCache(name: string): void {
  if (typeof window === "undefined") return;
  try {
    window.sessionStorage.removeItem(key(name));
  } catch {
    // Best-effort. If removeItem fails the next read will likely also fail
    // and return null, which is the same outcome.
  }
}
