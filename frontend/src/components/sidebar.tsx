"use client";

import { useState, useEffect } from "react";
import { NAV } from "@/lib/nav";
import { Icons, NAV_ICONS } from "@/components/icons";

interface SidebarProps {
  activePage: string;
  onNavigate: (pageId: string) => void;
  rail?: boolean;
  onToggleRail?: () => void;
  privacy?: boolean;
  onTogglePrivacy?: () => void;
  dark?: boolean;
  onToggleDark?: () => void;
}

export function Sidebar({ activePage, onNavigate, rail = false, onToggleRail, privacy = false, onTogglePrivacy, dark = false, onToggleDark }: SidebarProps) {
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({});

  useEffect(() => {
    const group = NAV.find((g) => g.items.some((i) => i.id === activePage));
    if (group) setOpenGroups((prev) => ({ ...prev, [group.id]: true }));
  }, [activePage]);

  const toggleGroup = (groupId: string, altKey: boolean) => {
    if (altKey) {
      setOpenGroups((prev) => ({ ...prev, [groupId]: !prev[groupId] }));
    } else {
      setOpenGroups((prev) => prev[groupId] ? { ...prev, [groupId]: false } : { [groupId]: true });
    }
  };

  return (
    <aside className="flex flex-col border-r sticky top-0 h-screen overflow-hidden transition-all duration-250"
           style={{ width: rail ? 64 : 260, background: "var(--sidebar-bg)", borderColor: "var(--border)" }}>

      {/* Brand + rail toggle */}
      <div className="flex items-center justify-between px-[18px] pt-[18px] pb-2.5">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg flex items-center justify-center shrink-0"
               style={{ background: "linear-gradient(135deg, #6366f1, #8b5cf6 55%, #0ea5a4)", boxShadow: "0 2px 8px rgba(99,102,241,0.35)" }}>
            <span className="text-white italic text-[17px] font-medium" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>M</span>
          </div>
          {!rail && (
            <div>
              <div className="font-bold text-[15px] tracking-tight">MO Money</div>
              <div className="text-[10px] text-[#8a90a2] uppercase tracking-[0.10em] font-medium">v.4</div>
            </div>
          )}
        </div>
        {!rail && onToggleRail && (
          <button className="w-7 h-7 grid place-items-center rounded-lg text-[#5a6175] hover:bg-[var(--bg-2)] transition-colors"
                  onClick={onToggleRail}>
            {Icons.panelLeft()}
          </button>
        )}
      </div>

      {!rail && (
        <>
          {/* Strategy picker */}
          <div className="mx-3.5 mb-3 px-3 py-2.5 rounded-[10px] flex items-center gap-2.5 cursor-pointer transition-colors" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
            <span className="w-2 h-2 rounded-full bg-[#6366f1]" style={{ boxShadow: "0 0 0 3px #eef0ff" }} />
            <div className="flex-1 min-w-0">
              <div className="text-[10px] text-[#8a90a2] uppercase tracking-[0.10em] font-medium">Active Strategy</div>
              <div className="text-[13px] font-semibold">CanSlim</div>
            </div>
            {Icons.chevronDown()}
          </div>

          {/* Search */}
          <div className="mx-3.5 mb-2 relative">
            <span className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[#8a90a2]">{Icons.search()}</span>
            <input
              className="w-full pl-8 pr-10 py-2 rounded-[10px] text-[13px] outline-none transition-all focus:border-[#6366f1] focus:shadow-[0_0_0_3px_rgba(99,102,241,0.1)]"
              style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)" }}
              placeholder="Search pages…"
              readOnly
              onClick={() => document.dispatchEvent(new KeyboardEvent("keydown", { key: "k", metaKey: true }))}
            />
            <kbd className="absolute right-2 top-1/2 -translate-y-1/2 text-[10px] text-[#8a90a2] bg-[#eef0f6] border border-[#e6e8ef] rounded px-1.5"
                 style={{ fontFamily: "var(--font-jetbrains), monospace" }}>⌘K</kbd>
          </div>
        </>
      )}

      {/* Nav */}
      <nav className="flex-1 overflow-y-auto overflow-x-hidden px-2 py-1 scrollbar-thin">
        {NAV.map((group) => {
          const isOpen = !!openGroups[group.id] || rail;
          const hasActive = group.items.some((i) => i.id === activePage);

          return (
            <div key={group.id} className="mb-0.5">
              <button
                className="w-full flex items-center gap-2.5 px-2.5 py-[9px] rounded-[10px] hover:bg-black/[0.03] transition-[background] duration-150"
                onClick={(e) => !rail && toggleGroup(group.id, e.altKey)}
              >
                <span className="w-[3px] rounded-full shrink-0 transition-all duration-200"
                      style={{
                        height: isOpen || hasActive ? 18 : 14,
                        background: group.color,
                        opacity: isOpen || hasActive ? 1 : 0.35,
                        boxShadow: isOpen || hasActive ? `0 0 8px color-mix(in oklab, ${group.color} 60%, transparent)` : "none",
                      }} />
                {!rail && (
                  <>
                    <span className="flex-1 text-left text-[11px] uppercase tracking-[0.10em] font-semibold"
                          style={{ color: hasActive ? group.color : "#5a6175" }}>
                      {group.label}
                    </span>
                    <span className="text-[10px] rounded-full px-[7px] min-w-[18px] text-center"
                          style={{
                            fontFamily: "var(--font-jetbrains), monospace",
                            background: hasActive ? `color-mix(in oklab, ${group.color} 12%, transparent)` : "#eef0f6",
                            color: hasActive ? group.color : "#8a90a2",
                          }}>
                      {group.items.length}
                    </span>
                    <span className="text-[#8a90a2] transition-transform duration-250 shrink-0"
                          style={{ transform: isOpen ? "rotate(90deg)" : "none" }}>
                      {Icons.chevronRight()}
                    </span>
                  </>
                )}
              </button>

              {!rail && (
                <div className="grid transition-[grid-template-rows] duration-300 ease-out"
                     style={{ gridTemplateRows: isOpen ? "1fr" : "0fr" }}>
                  <div className="min-h-0 overflow-hidden pl-5 pr-1.5 pb-1">
                    {group.items.map((item) => {
                      const isActive = activePage === item.id;
                      const IconFn = NAV_ICONS[item.id] || Icons.grid;
                      return (
                        <button key={item.id}
                                className="w-full flex items-center gap-2.5 px-2.5 py-[7px] rounded-lg text-[13px] font-medium transition-all duration-[120ms] relative text-left group/navitem"
                                style={{
                                  background: isActive ? `color-mix(in oklab, ${group.color} 14%, transparent)` : "transparent",
                                  color: isActive ? group.color : "var(--ink-2)",
                                  fontWeight: isActive ? 600 : 500,
                                  ["--hover-color" as string]: group.color,
                                }}
                                onMouseEnter={(e) => { if (!isActive) { e.currentTarget.style.background = `color-mix(in oklab, ${group.color} 8%, transparent)`; e.currentTarget.style.color = group.color; }}}
                                onMouseLeave={(e) => { if (!isActive) { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "var(--ink-2)"; }}}
                                onClick={() => onNavigate(item.id)}>
                          {isActive && (
                            <span className="absolute left-[-14px] top-1/2 -translate-y-1/2 w-[3px] h-4 rounded-full"
                                  style={{ background: group.color }} />
                          )}
                          <span className="w-4 h-4 grid place-items-center opacity-85 shrink-0">{IconFn()}</span>
                          <span>{item.label}</span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-3.5 py-3 flex flex-col gap-2.5" style={{ borderTop: "1px solid var(--border)", background: "var(--sidebar-bg)" }}>
        {!rail && (
          <>
            {/* Dark mode toggle */}
            <div className="flex items-center gap-2.5 text-xs" style={{ color: "var(--ink-3)" }}>
              <div className="w-7 h-4 rounded-full relative cursor-pointer transition-colors"
                   style={{ background: dark ? "#6366f1" : "var(--border-2)" }}
                   onClick={onToggleDark}>
                <span className="absolute top-0.5 w-3 h-3 rounded-full bg-white shadow-sm transition-all"
                      style={{ left: dark ? 14 : 2 }} />
              </div>
              <span>🌙</span>
              <span className="flex-1">Dark Mode</span>
            </div>
            {/* Privacy toggle */}
            <div className="flex items-center gap-2.5 text-xs" style={{ color: "var(--ink-3)" }}>
              <div className="w-7 h-4 rounded-full relative cursor-pointer transition-colors"
                   style={{ background: privacy ? "#6366f1" : "var(--border-2)" }}
                   onClick={onTogglePrivacy}>
                <span className="absolute top-0.5 w-3 h-3 rounded-full bg-white shadow-sm transition-all"
                      style={{ left: privacy ? 14 : 2 }} />
              </div>
              {Icons.lock()}
              <span className="flex-1">Privacy Mode</span>
            </div>
          </>
        )}
        <div className="flex items-center gap-2.5 p-1.5 rounded-[10px] hover:bg-[var(--bg-2)] transition-colors">
          <div className="w-[30px] h-[30px] rounded-full flex items-center justify-center text-white text-xs font-semibold shrink-0"
               style={{ background: "linear-gradient(135deg, #6366f1, #8b5cf6)" }}>
            M
          </div>
          {!rail && (
            <>
              <div className="flex-1 min-w-0">
                <div className="text-[13px] font-semibold">MO</div>
                <div className="text-[11px] text-[#8a90a2]">mo@momoney.app</div>
              </div>
              <button className="w-7 h-7 grid place-items-center rounded-lg text-[#5a6175] hover:bg-[var(--bg-2)]">
                {Icons.logout()}
              </button>
            </>
          )}
        </div>
      </div>
    </aside>
  );
}
