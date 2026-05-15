/**
 * Thin logging wrapper. Routes through console today; forward-
 * compatible shim for future Sentry / DataDog integration.
 *
 * Convention: log.error(area, what, err) emits
 *   [area] what: err
 *
 * Use `area` = component or feature name (e.g., "weekly-retro",
 * "dashboard"). Use `what` = human-readable action that failed
 * (e.g., "rail fetch failed", "metrics fetch failed").
 *
 * Existing console.error sites in the codebase already use this
 * bracket format; this wrapper makes it callable so the format
 * can't drift across sites.
 *
 * Motivated by the Phase 6 "0 weeks" regression — a silent
 * `if ("error" in res) return` on a mount-time fetch rendered an
 * empty rail with no devtools signal. Errors must always reach
 * the console so devtools-savvy users (and future error
 * aggregators) can catch regressions of this class.
 */
export const log = {
  error: (area: string, what: string, err: unknown) =>
    console.error(`[${area}] ${what}:`, err),
  warn: (area: string, what: string, err: unknown) =>
    console.warn(`[${area}] ${what}:`, err),
  debug: (area: string, what: string, err: unknown) =>
    console.debug(`[${area}] ${what}:`, err),
};
