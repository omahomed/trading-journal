"use client";

import { useState, useEffect, useRef, useMemo } from "react";
import { getAllPages } from "@/lib/nav";

interface CommandPaletteProps {
  onNavigate: (pageId: string, tab?: string) => void;
}

export function CommandPalette({ onNavigate }: CommandPaletteProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [idx, setIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const allPages = getAllPages();

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") { e.preventDefault(); setOpen((v) => !v); }
      if (e.key === "Escape" && open) setOpen(false);
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open]);

  useEffect(() => { if (open) { setQuery(""); setIdx(0); setTimeout(() => inputRef.current?.focus(), 50); } }, [open]);

  const filtered = useMemo(() => {
    const q = query.toLowerCase().trim();
    if (!q) return allPages;
    return allPages.filter((p) => p.label.toLowerCase().includes(q) || p.group.toLowerCase().includes(q));
  }, [query, allPages]);

  useEffect(() => { setIdx(0); }, [query]);

  const grouped = useMemo(() => {
    const groups: Record<string, typeof filtered> = {};
    const order: string[] = [];
    filtered.forEach((item, i) => {
      if (!groups[item.group]) { groups[item.group] = []; order.push(item.group); }
      groups[item.group].push({ ...item, flatIdx: i } as any);
    });
    return { groups, order };
  }, [filtered]);

  const select = (item: typeof allPages[0]) => {
    setOpen(false);
    if (item.parentPage) {
      onNavigate(item.parentPage, item.tab);
    } else {
      onNavigate(item.id);
    }
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") { e.preventDefault(); setIdx((i) => Math.min(filtered.length - 1, i + 1)); }
    if (e.key === "ArrowUp") { e.preventDefault(); setIdx((i) => Math.max(0, i - 1)); }
    if (e.key === "Enter") { e.preventDefault(); if (filtered[idx]) select(filtered[idx]); }
  };

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[100] grid place-items-start justify-center pt-[12vh]"
         style={{ background: "rgba(0,0,0,0.4)", backdropFilter: "blur(4px)" }}
         onClick={() => setOpen(false)}>
      <div className="w-[560px] max-w-[90vw] rounded-[14px] overflow-hidden"
           style={{ background: "var(--surface)", boxShadow: "0 20px 48px rgba(0,0,0,0.2), 0 0 0 1px var(--border)", animation: "cmdk-rise 0.22s cubic-bezier(.2,.9,.3,1.1)" }}
           onClick={(e) => e.stopPropagation()} onKeyDown={onKeyDown}>

        {/* Input */}
        <div className="flex items-center gap-2.5 px-[18px] py-3.5" style={{ borderBottom: "1px solid var(--border)" }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--ink-4)" strokeWidth="2">
            <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input ref={inputRef} value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Jump to page..."
                 className="flex-1 border-none outline-none text-base bg-transparent" style={{ color: "var(--ink)" }} autoComplete="off" />
          <kbd className="text-[10px] rounded px-1.5 py-0.5" style={{ background: "var(--bg-2)", border: "1px solid var(--border)", color: "var(--ink-4)", fontFamily: "var(--font-jetbrains), monospace" }}>ESC</kbd>
        </div>

        {/* Results */}
        <div className="max-h-[50vh] overflow-y-auto p-1.5">
          {grouped.order.map((groupName) => (
            <div key={groupName}>
              <div className="text-[10px] uppercase tracking-[0.10em] font-semibold px-3 pt-2.5 pb-1" style={{ color: "var(--ink-4)" }}>{groupName}</div>
              {grouped.groups[groupName].map((item: any) => (
                <div key={item.id}
                     className="flex items-center gap-2.5 px-3 py-[9px] rounded-[10px] text-sm cursor-pointer transition-colors"
                     style={{ background: item.flatIdx === idx ? "var(--bg-2)" : "transparent", color: item.flatIdx === idx ? "var(--ink)" : "var(--ink-2)" }}
                     onMouseEnter={() => setIdx(item.flatIdx)} onClick={() => select(item)}>
                  <span className="w-2 h-2 rounded-full shrink-0" style={{ background: item.color }} />
                  <span className="flex-1 font-medium">{item.label}</span>
                  <span className="text-[11px]" style={{ color: "var(--ink-4)" }}>{item.group}</span>
                </div>
              ))}
            </div>
          ))}
          {filtered.length === 0 && <div className="py-7 text-center text-[13px]" style={{ color: "var(--ink-4)" }}>No matches</div>}
        </div>

        {/* Footer */}
        <div className="flex items-center gap-3.5 px-4 py-2 text-[11px]" style={{ borderTop: "1px solid var(--border)", color: "var(--ink-4)" }}>
          <span><kbd className="text-[10px] rounded px-1 mr-1" style={{ background: "var(--bg-2)", border: "1px solid var(--border)", fontFamily: "var(--font-jetbrains), monospace" }}>{"↑↓"}</kbd> navigate</span>
          <span><kbd className="text-[10px] rounded px-1 mr-1" style={{ background: "var(--bg-2)", border: "1px solid var(--border)", fontFamily: "var(--font-jetbrains), monospace" }}>{"↵"}</kbd> select</span>
          <span className="ml-auto">{filtered.length} results</span>
        </div>
      </div>

      <style jsx global>{`@keyframes cmdk-rise { from { transform: translateY(-10px) scale(0.97); opacity: 0; } }`}</style>
    </div>
  );
}
