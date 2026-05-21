// Auto-reload on chunk-load failure (typically a deploy that
// invalidated cached chunk hashes the browser is still
// referencing). One reload per session is enough to recover; a
// second chunk error in the same session signals a real network
// or CDN issue, so we stop trying and let the existing error UI
// surface the situation to the user.
//
// sessionStorage (not localStorage) — the guard clears on tab
// close, so a user returning hours later for a different deploy
// gets a fresh attempt budget. localStorage would permanently
// disable the handler after any one failure ever.

const RELOAD_KEY = "mo-chunk-reload-attempted";

/**
 * Detects dynamic-import failures across the common error
 * patterns thrown by Webpack/Vite/Next.js variants and major
 * browsers. False on non-chunk errors so callers can fall
 * through to their existing error UI.
 */
export function isChunkLoadError(error: unknown): boolean {
  if (!error) return false;
  const obj = error as { name?: unknown; message?: unknown };
  const name = String(obj.name ?? "");
  const msg = String(obj.message ?? "");

  return (
    name === "ChunkLoadError" ||
    msg.includes("Loading chunk") ||
    msg.includes("Failed to fetch dynamically imported module") ||
    msg.includes("Importing a module script failed") ||
    msg.includes("error loading dynamically imported module")
  );
}

/**
 * Attempt one full-page reload, guarded against loops by a
 * sessionStorage flag. Returns true if a reload was triggered
 * (caller should suppress further error handling), false if
 * we've already reloaded once this session (caller should
 * fall through to the existing error UI).
 *
 * Best-effort under storage-restricted browsers: if
 * sessionStorage throws (private mode quirks), skip the guard
 * and reload anyway — one false-positive reload beats no
 * reload at all.
 */
export function tryAutoReload(): boolean {
  if (typeof window === "undefined") return false;

  try {
    if (sessionStorage.getItem(RELOAD_KEY)) return false;
    sessionStorage.setItem(RELOAD_KEY, String(Date.now()));
  } catch {
    /* private mode / quota — proceed without guard */
  }

  window.location.reload();
  return true;
}

/**
 * Convenience wrapper: if the error is a chunk-load failure,
 * attempt auto-reload. Returns true when handled (caller
 * should skip its normal error path).
 */
export function handleChunkLoadError(error: unknown): boolean {
  if (!isChunkLoadError(error)) return false;
  return tryAutoReload();
}
