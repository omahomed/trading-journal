"use client";

import { useEffect, useRef, useState } from "react";

// Inlined at build time; reflects the version of the JS bundle the browser
// currently has. /api/version returns whatever version the server is
// currently serving. If they diverge, a new deployment has landed and the
// tab is running stale code.
const LOADED_BUILD_ID = process.env.NEXT_PUBLIC_BUILD_ID || "";

// Poll cadence. Matches the industry-standard "check every ~minute" pattern
// used by Slack / Linear / Notion for their in-app update prompts.
const POLL_MS = 60_000;

export function UpdateBanner() {
  const [updateAvailable, setUpdateAvailable] = useState(false);
  const dismissedRef = useRef(false);

  useEffect(() => {
    if (!LOADED_BUILD_ID) return;

    let cancelled = false;

    const check = async () => {
      if (cancelled || dismissedRef.current) return;
      try {
        const res = await fetch("/api/version", { cache: "no-store" });
        if (!res.ok) return;
        const data = (await res.json()) as { buildId?: string };
        if (data.buildId && data.buildId !== LOADED_BUILD_ID) {
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

  if (!updateAvailable) return null;

  return (
    <div
      className="fixed bottom-5 left-1/2 -translate-x-1/2 z-[60] flex items-center gap-3 px-4 py-2.5 rounded-[12px] text-[12px] font-medium"
      style={{
        background: "var(--surface)",
        border: "1px solid var(--border)",
        color: "var(--ink)",
        boxShadow: "0 8px 30px rgba(0,0,0,0.18)",
        animation: "slide-up 0.22s ease-out",
      }}
      role="status"
      aria-live="polite"
    >
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: "var(--ink-3)" }}>
        <polyline points="23 4 23 10 17 10" />
        <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
      </svg>
      <span>New version available</span>
      <button
        onClick={() => window.location.reload()}
        className="h-[28px] px-3 rounded-[8px] text-white text-[11px] font-semibold transition-all hover:brightness-110"
        style={{ background: "#6366f1" }}
      >
        Refresh
      </button>
      <button
        onClick={() => { dismissedRef.current = true; setUpdateAvailable(false); }}
        className="h-[28px] px-2 rounded-[8px] text-[14px] leading-none"
        style={{ background: "transparent", color: "var(--ink-4)", border: "none" }}
        title="Dismiss until next session"
      >
        ✕
      </button>
    </div>
  );
}
