"use client";

import { useEffect } from "react";

// Inlined at build time; reflects the version of the JS bundle the browser
// currently has. /api/version returns whatever version the server is
// currently serving. If they diverge, a new deployment has landed and the
// tab is running stale code.
const LOADED_BUILD_ID = process.env.NEXT_PUBLIC_BUILD_ID || "";

// Poll cadence. Matches the industry-standard "check every ~minute" pattern
// used by Slack / Linear / Notion for their in-app update prompts.
const POLL_MS = 60_000;

// Headless reload mechanism. The moment a poll of /api/version sees a
// BUILD_ID that differs from the one baked into the running bundle, we
// reload immediately — no banner, no nav gate.
//
// The previous nav-gated implementation deferred reload until the user's
// next pathname change, which lost a race with Next.js's chunk fetcher:
// on first click after deploy, the router would request a chunk by its
// old hashed URL, the new server would 404 (or, worse, the service
// worker would serve cached /dashboard HTML for the JS request), the
// browser would surface ERR_FAILED, and the React effect that would
// have called reload() never ran because pathname never committed.
//
// Reloading immediately on detection collapses the vulnerable window to
// "between deploy and the next poll tick" (≤60s, or much shorter if the
// user focuses/visibilitychanges the tab in between). Form-state risk:
// a dirty form when the tab regains focus would be lost. Acceptable
// because the alternative is a broken click that ALSO loses the form
// AND leaves the user staring at a browser error.
export function UpdateBanner() {
  useEffect(() => {
    if (!LOADED_BUILD_ID) return;

    let cancelled = false;

    const check = async () => {
      if (cancelled) return;
      try {
        const res = await fetch("/api/version", { cache: "no-store" });
        if (!res.ok) return;
        const data = (await res.json()) as { buildId?: string };
        if (cancelled) return;
        const seen = data.buildId || "";
        if (seen && seen !== LOADED_BUILD_ID) {
          cancelled = true;
          window.location.reload();
        }
      } catch { /* network hiccup — try again next interval */ }
    };

    check();
    const id = setInterval(check, POLL_MS);

    // Re-check on focus / visibility — the most common trigger is a user
    // who has been away while a deploy landed, returning to the tab.
    const onFocus = () => check();
    const onVisibility = () => { if (!document.hidden) check(); };
    window.addEventListener("focus", onFocus);
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      cancelled = true;
      clearInterval(id);
      window.removeEventListener("focus", onFocus);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);

  return null;
}
