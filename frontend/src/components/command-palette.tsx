"use client";

import { useState, useEffect, useRef, useMemo } from "react";
import { getAllPages } from "@/lib/nav";

interface CommandPaletteProps {
  onNavigate: (pageId: string) => void;
}

export function CommandPalette({ onNavigate }: CommandPaletteProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [idx, setIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const allPages = getAllPages();

  // ⌘K listener
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setOpen((v) => !v);
      }
      if (e.key === "Escape" && open) {
        setOpen(false);
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open]);

  // Focus input when opened
  useEffect(() => {
    if (open) {
      setQuery("");
      setIdx(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [open]);

  // Filter results
  const filtered = useMemo(() => {
    const q = query.toLowerCase().trim();
    if (!q) return allPages;
    return allPages.filter(
      (p) =>
        p.label.toLowerCase().includes(q) || p.group.toLowerCase().includes(q)
    );
  }, [query, allPages]);

  // Reset index on query change
  useEffect(() => {
    setIdx(0);
  }, [query]);

  // Group filtered results
  const grouped = useMemo(() => {
    const groups: Record<string, typeof filtered> = {};
    const order: string[] = [];
    filtered.forEach((item, i) => {
      if (!groups[item.group]) {
        groups[item.group] = [];
        order.push(item.group);
      }
      groups[item.group].push({ ...item, flatIdx: i } as any);
    });
    return { groups, order };
  }, [filtered]);

  const select = (pageId: string) => {
    setOpen(false);
    onNavigate(pageId);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setIdx((i) => Math.min(filtered.length - 1, i + 1));
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      setIdx((i) => Math.max(0, i - 1));
    }
    if (e.key === "Enter") {
      e.preventDefault();
      if (filtered[idx]) select(filtered[idx].id);
    }
  };

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-[100] grid place-items-start justify-center pt-[12vh] animate-fade-in"
      style={{ background: "rgba(14,20,38,0.35)", backdropFilter: "blur(4px)" }}
      onClick={() => setOpen(false)}
    >
      <div
        className="w-[560px] max-w-[90vw] bg-white rounded-[14px] overflow-hidden"
        style={{
          boxShadow: "0 20px 48px rgba(14,20,38,0.14), 0 0 0 1px rgba(14,20,38,0.06)",
          animation: "cmdk-rise 0.22s cubic-bezier(.2,.9,.3,1.1)",
        }}
        onClick={(e) => e.stopPropagation()}
        onKeyDown={onKeyDown}
      >
        {/* Input */}
        <div className="flex items-center gap-2.5 px-[18px] py-3.5 border-b border-[#e6e8ef]">
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="#8a90a2"
            strokeWidth="2"
          >
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Jump to page…"
            className="flex-1 border-none outline-none text-base bg-transparent text-[#0f1524]"
            autoComplete="off"
          />
          <kbd className="font-num text-[10px] bg-[#eef0f6] border border-[#e6e8ef] rounded px-1.5 text-[#8a90a2]">
            ESC
          </kbd>
        </div>

        {/* Results */}
        <div className="max-h-[50vh] overflow-y-auto p-1.5">
          {grouped.order.map((groupName) => (
            <div key={groupName}>
              <div className="text-[10px] uppercase tracking-[0.10em] text-[#8a90a2] font-semibold px-3 pt-2.5 pb-1">
                {groupName}
              </div>
              {grouped.groups[groupName].map((item: any) => (
                <div
                  key={item.id}
                  className="flex items-center gap-2.5 px-3 py-[9px] rounded-[10px] text-sm cursor-pointer transition-colors"
                  style={{
                    background: item.flatIdx === idx ? "#eef0f6" : "transparent",
                  }}
                  onMouseEnter={() => setIdx(item.flatIdx)}
                  onClick={() => select(item.id)}
                >
                  <span
                    className="w-2 h-2 rounded-full shrink-0"
                    style={{ background: item.color }}
                  />
                  <span className="flex-1">{item.label}</span>
                  <span className="text-[11px] text-[#8a90a2]">{item.group}</span>
                </div>
              ))}
            </div>
          ))}
          {filtered.length === 0 && (
            <div className="py-7 text-center text-[13px] text-[#8a90a2]">
              No matches
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center gap-3.5 px-4 py-2 border-t border-[#e6e8ef] text-[11px] text-[#8a90a2]">
          <span>
            <kbd className="font-num text-[10px] bg-[#eef0f6] border border-[#e6e8ef] rounded px-1 mr-1">
              ↑↓
            </kbd>
            navigate
          </span>
          <span>
            <kbd className="font-num text-[10px] bg-[#eef0f6] border border-[#e6e8ef] rounded px-1 mr-1">
              ↵
            </kbd>
            select
          </span>
          <span className="ml-auto">{filtered.length} results</span>
        </div>
      </div>

      <style jsx global>{`
        @keyframes cmdk-rise {
          from {
            transform: translateY(-10px) scale(0.97);
            opacity: 0;
          }
        }
      `}</style>
    </div>
  );
}
