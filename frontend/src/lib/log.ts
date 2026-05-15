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
 *
 * Phase B (v2): log.warn.devOnly(...) and log.debug.devOnly(...)
 * fire only when NODE_ENV !== "production". Used at sites where
 * the failure is expected-noise in production (e.g., service-
 * worker registration on http, optional pre-fill fetches) but we
 * still want visibility during local development.
 */

type LogFn = (area: string, what: string, err: unknown) => void;
type LogLeveledFn = LogFn & { devOnly: LogFn };

const isDev = (): boolean => process.env.NODE_ENV !== "production";

const _warn: LogFn = (area, what, err) =>
  console.warn(`[${area}] ${what}:`, err);

const _debug: LogFn = (area, what, err) =>
  console.debug(`[${area}] ${what}:`, err);

const _devOnly = (fn: LogFn): LogFn =>
  (area, what, err) => { if (isDev()) fn(area, what, err); };

export const log = {
  error: ((area: string, what: string, err: unknown) =>
    console.error(`[${area}] ${what}:`, err)) as LogFn,
  warn: Object.assign(_warn, { devOnly: _devOnly(_warn) }) as LogLeveledFn,
  debug: Object.assign(_debug, { devOnly: _devOnly(_debug) }) as LogLeveledFn,
};
