"use client";

import { useState, useEffect } from "react";
import { usePathname } from "next/navigation";
import { Sidebar } from "@/components/sidebar";
import { CommandPalette } from "@/components/command-palette";
import { TapeStatusPill } from "@/components/tape-status-pill";
import { getGroupForHref, getNavItemForHref } from "@/lib/nav";
import { setFocusModeActive } from "@/lib/format";

/**
 * Desktop chrome — extracted verbatim from the former `AppShell` in
 * `src/app/(app)/layout.tsx` (Phase 1 Step 5). Render output, state,
 * effects, event handlers, and styled-jsx keyframes are byte-identical
 * to the pre-extraction shell. Do not edit during Phase 1 outside the
 * AdaptiveShell wiring; this file is on the do-not-touch list as soon
 * as Step 5 is complete.
 */
export function DesktopShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [privacy, setPrivacy] = useState(false);
  const [focusMode, setFocusMode] = useState(false);
  const [dark, setDark] = useState(false);
  const [rail, setRail] = useState(false);

  const group = getGroupForHref(pathname);
  const navColor = group?.color || "#6366f1";
  const navItem = getNavItemForHref(pathname);
  const pageLabel = navItem?.label || "";

  // Load saved theme on mount
  useEffect(() => {
    const saved = localStorage.getItem("mo-theme");
    if (saved === "dark") { setDark(true); document.documentElement.classList.add("dark"); }
  }, []);

  // Load saved Privacy Mode (full-blur). Defaults OFF — previously
  // ephemeral; now persisted for consistency with Dark Mode + Focus Mode.
  useEffect(() => {
    const saved = localStorage.getItem("mo-privacy-full");
    if (saved === "on") setPrivacy(true);
  }, []);

  // Load saved Focus Mode (currency masking). Defaults ON for new users
  // — discoverable via the visible sidebar toggle, and the safer default
  // for shoulder-surfing scenarios. Update the module mirror *before*
  // setFocusMode so the first render reads the correct masking state.
  useEffect(() => {
    const saved = localStorage.getItem("mo-focus-mode");
    const initial = saved === null ? true : saved === "on";
    setFocusModeActive(initial);
    setFocusMode(initial);
  }, []);

  const toggleDark = () => {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    localStorage.setItem("mo-theme", next ? "dark" : "light");
  };

  const togglePrivacy = () => {
    const next = !privacy;
    setPrivacy(next);
    localStorage.setItem("mo-privacy-full", next ? "on" : "off");
  };

  const toggleFocus = () => {
    const next = !focusMode;
    // Update the module mirror synchronously *before* triggering the
    // React re-render — otherwise formatCurrency reads the stale value
    // during the render and the UI lags by one tick (or one refresh).
    setFocusModeActive(next);
    setFocusMode(next);
    localStorage.setItem("mo-focus-mode", next ? "on" : "off");
  };

  return (
    <div className={`flex h-screen ${privacy ? "privacy" : ""}`}>
      <Sidebar
        privacy={privacy} onTogglePrivacy={togglePrivacy}
        focusMode={focusMode} onToggleFocus={toggleFocus}
        dark={dark} onToggleDark={toggleDark}
        rail={rail} onToggleRail={() => setRail(!rail)}
      />
      <main className="flex-1 flex flex-col min-w-0" style={{ background: "var(--bg)" }}>
        <header className="h-[56px] flex items-center px-6 gap-5 sticky top-0 z-30"
                style={{ background: "var(--header-bg)", backdropFilter: "saturate(160%) blur(10px)", borderBottom: "1px solid var(--border)" }}>
          <div className="flex items-center gap-2 text-[13px] text-[var(--ink-3)]">
            <span className="font-semibold" style={{ color: navColor }}>{group?.label}</span>
            <span className="text-[#b6bac7]">/</span>
            <span className="text-[var(--ink)] font-semibold">{pageLabel}</span>
          </div>
          <div className="flex-1" />
          <TapeStatusPill />
          <button className="flex items-center gap-1.5 h-[30px] px-3 rounded-full text-xs font-medium bg-[var(--surface)] border border-[var(--border)] text-[var(--ink-2)] hover:bg-[#eef0f6] transition-colors"
                  onClick={() => document.dispatchEvent(new KeyboardEvent("keydown", { key: "k", metaKey: true }))}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
            Quick jump
            <kbd className="text-[10px] bg-[#eef0f6] border border-[var(--border)] rounded px-1" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>⌘K</kbd>
          </button>
        </header>
        <div className="flex-1 overflow-auto px-7 py-6">
          {children}
        </div>
      </main>
      <CommandPalette />

      <style jsx global>{`
        @keyframes slide-up { from { opacity: 0; transform: translateY(6px); } }
        @keyframes pulse-dot {
          0%, 100% { box-shadow: 0 0 0 3px currentColor; }
          50% { box-shadow: 0 0 0 6px transparent; }
        }
      `}</style>
    </div>
  );
}
