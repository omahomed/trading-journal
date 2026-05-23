// Phase 1 service worker — minimal app-shell cache + network-first
// for the API. Goals:
//
//   1. Make the warm-dark mobile shell available offline so the app
//      doesn't show a blank page when the network drops mid-trade
//      (the page will render with whatever the data hooks return —
//      typically empty / loading state — but the chrome is there).
//   2. Never cache authenticated API responses. Every API call is a
//      fresh network request; offline = empty data, not stale data.
//   3. Cache-first for static assets (JS, CSS, fonts, icons) so the
//      installed PWA launches instantly on repeat visits.
//
// Versioning: bump SHELL_CACHE_NAME any time you change PRECACHE_URLS
// or the cache strategy. Old caches are deleted on `activate`.
//
// v1 → v2 (one-time invalidation): the v1 PRECACHE seeded "/" and
// "/dashboard", both of which middleware-redirected through to /login
// for unauthenticated users. The cached body for "/" was therefore a
// snapshot of the /login HTML at SW install time, indefinitely served
// to returning users by the (also-removed) cache-first navigation
// strategy. The bump forces activate-time eviction of v1 so users
// returning with stale entries don't keep seeing them.
const SHELL_CACHE_NAME = "mo-shell-v2";
const RUNTIME_CACHE_NAME = "mo-runtime-v2";

// Static PWA assets that the install flow needs offline. Navigation
// HTML routes are deliberately NOT precached — those are now served
// network-first by the fetch handler below, so a fresh online client
// always gets the deploy's current HTML (and current chunk URLs).
const PRECACHE_URLS = [
  "/manifest.json",
  "/icon-192.png",
  "/icon-512.png",
  "/apple-touch-icon.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(SHELL_CACHE_NAME)
      .then((cache) => cache.addAll(PRECACHE_URLS))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((k) => k !== SHELL_CACHE_NAME && k !== RUNTIME_CACHE_NAME)
          .map((k) => caches.delete(k)),
      ),
    ).then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;

  // Only handle GET. POST/PUT/DELETE go straight to the network — they
  // mutate state and must never be served from cache.
  if (req.method !== "GET") return;

  const url = new URL(req.url);

  // Auth endpoints: always network. Never cache. The login flow must
  // see live state, and stale auth responses are a security smell.
  if (url.pathname.startsWith("/api/auth/")) {
    return; // default browser network handling
  }

  // App API: network-first, no cache write. Offline = the network
  // request rejects and the calling component handles the error
  // (loading / empty state / cached prior data via SWR in a later phase).
  if (url.pathname.startsWith("/api/")) {
    event.respondWith(fetch(req));
    return;
  }

  // Same-origin requests split by mode:
  //
  // - Navigations (HTML routes): network-FIRST. The page HTML always
  //   comes from the live deploy, so its embedded chunk URLs always
  //   match what's currently on Vercel — no more chunk-404 broken
  //   first loads from a stale cached shell. Successful responses are
  //   written to the runtime cache so offline navigations have
  //   something to fall back on. Offline chain (in order): cached
  //   match for the specific URL → cached /dashboard shell (Phase 1
  //   offline goal) → fail. The /dashboard fallback yields nothing
  //   until the user has navigated there online once and populated
  //   the runtime cache.
  //
  // - Static assets (_next chunks, fonts, images): cache-FIRST,
  //   unchanged from prior behavior. Chunks are immutable per build
  //   so the cache hit is always correct. On miss, fetch and
  //   populate. On chunk network failure, propagate the error
  //   (do NOT return HTML — that would parse as JS and bury the
  //   real failure; see src/lib/chunk-reload.ts for the in-React
  //   ChunkLoadError handler that the propagated error feeds).
  if (url.origin === self.location.origin) {
    if (req.mode === "navigate") {
      event.respondWith(
        fetch(req)
          .then((res) => {
            if (res && res.status === 200 && res.type === "basic") {
              const copy = res.clone();
              caches.open(RUNTIME_CACHE_NAME).then((cache) => cache.put(req, copy));
            }
            return res;
          })
          .catch(() =>
            caches.match(req).then((cached) => cached || caches.match("/dashboard")),
          ),
      );
      return;
    }
    event.respondWith(
      caches.match(req).then((cached) => {
        if (cached) return cached;
        return fetch(req).then((res) => {
          if (!res || res.status !== 200 || res.type !== "basic") return res;
          const copy = res.clone();
          caches.open(RUNTIME_CACHE_NAME).then((cache) => cache.put(req, copy));
          return res;
        });
      }),
    );
    return;
  }

  // Cross-origin: pass through (lets the browser handle CORS, fonts,
  // analytics, etc. without our cache getting in the way).
});
