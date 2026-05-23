"use client";

import { useEffect, useRef, useState } from "react";
import { usePathname } from "next/navigation";

// Inlined at build time; reflects the version of the JS bundle the browser
// currently has. /api/version returns whatever version the server is
// currently serving. If they diverge, a new deployment has landed and the
// tab is running stale code.
const LOADED_BUILD_ID = process.env.NEXT_PUBLIC_BUILD_ID || "";

// Poll cadence. Matches the industry-standard "check every ~minute" pattern
// used by Slack / Linear / Notion for their in-app update prompts.
const POLL_MS = 60_000;

// Headless component: detects new versions via polling, then silently
// reloads the next time the user navigates between pages. No UI is
// rendered — the dismissable banner was removed because the version
// check is reliable enough now (build-info.json phase-gated in
// next.config.ts) that a passive auto-reload-on-nav is the calmer UX.
// Form-state risk is identical to the user clicking a link without
// saving — navigation already loses unsaved work without a warning,
// so no new footgun is introduced.
export function UpdateBanner() {
  const [updateAvailable, setUpdateAvailable] = useState(false);
  const pathname = usePathname();
  const prevPathnameRef = useRef(pathname);

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
          setUpdateAvailable(true);
        }
      } catch { /* network hiccup — try again next interval */ }
    };

    check();
    const id = setInterval(check, POLL_MS);

    // Also re-check when the tab regains focus (common case: user has been
    // away while a deploy landed).
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

  useEffect(() => {
    if (updateAvailable && pathname !== prevPathnameRef.current) {
      window.location.reload();
    }
    prevPathnameRef.current = pathname;
  }, [updateAvailable, pathname]);

  return null;
}
