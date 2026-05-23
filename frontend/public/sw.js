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

const SHELL_CACHE_NAME = "mo-shell-v1";
const RUNTIME_CACHE_NAME = "mo-runtime-v1";

// The minimal set of URLs that have to be available offline for the
// shell to render. Don't precache page routes whose content depends
// on auth state — those will fall back to whatever the runtime cache
// has from prior visits, which is the right behavior.
const PRECACHE_URLS = [
  "/",
  "/dashboard",
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

  // Same-origin static assets and HTML routes: cache-first, with a
  // network fallback that updates the runtime cache for next time.
  if (url.origin === self.location.origin) {
    event.respondWith(
      caches.match(req).then((cached) => {
        if (cached) return cached;
        const networkFetch = fetch(req).then((res) => {
          // Don't cache opaque or error responses.
          if (!res || res.status !== 200 || res.type !== "basic") return res;
          const copy = res.clone();
          caches.open(RUNTIME_CACHE_NAME).then((cache) => cache.put(req, copy));
          return res;
        });
        // Offline fallback to the cached /dashboard shell is ONLY safe
        // for navigation requests. Returning HTML for a failed JS
        // chunk request (mode === "script") would make the browser
        // parse HTML as JS and surface a misleading parse error —
        // and prevent Next.js's chunk-reload handler (in
        // src/lib/chunk-reload.ts via the error boundaries) from
        // seeing the real network failure. For non-navigation
        // requests, let the network error propagate.
        if (req.mode === "navigate") {
          return networkFetch.catch(() => caches.match("/dashboard"));
        }
        return networkFetch;
      }),
    );
    return;
  }

  // Cross-origin: pass through (lets the browser handle CORS, fonts,
  // analytics, etc. without our cache getting in the way).
});
