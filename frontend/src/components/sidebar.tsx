"use client";

import { useState, useEffect } from "react";
import { NAV, type NavGroup } from "@/lib/nav";

interface SidebarProps {
  activePage: string;
  onNavigate: (pageId: string) => void;
  rail?: boolean;
  onToggleRail?: () => void;
}

export function Sidebar({ activePage, onNavigate, rail = false, onToggleRail }: SidebarProps) {
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({});

  // Auto-open the group containing the active page
  useEffect(() => {
    const group = NAV.find((g) => g.items.some((i) => i.id === activePage));
    if (group) {
      setOpenGroups((prev) => ({ ...prev, [group.id]: true }));
    }
  }, [activePage]);

  const toggleGroup = (groupId: string, altKey: boolean) => {
    if (altKey) {
      // Alt-click: toggle this group without closing others
      setOpenGroups((prev) => ({ ...prev, [groupId]: !prev[groupId] }));
    } else {
      // Normal click: single-open mode
      setOpenGroups((prev) =>
        prev[groupId] ? { ...prev, [groupId]: false } : { [groupId]: true }
      );
    }
  };

  const activeGroup = NAV.find((g) => g.items.some((i) => i.id === activePage));

  return (
    <aside
      className="flex flex-col bg-[#fbfbfe] border-r border-[#e6e8ef] sticky top-0 h-screen overflow-hidden transition-all duration-250"
      style={{ width: rail ? 64 : 260 }}
    >
      {/* Brand */}
      <div className="flex items-center gap-2.5 px-[18px] pt-[18px] pb-2.5">
        <div
          className="w-7 h-7 rounded-lg flex items-center justify-center shrink-0"
          style={{
            background: "linear-gradient(135deg, #6366f1, #8b5cf6 55%, #0ea5a4)",
            boxShadow: "0 2px 8px rgba(99,102,241,0.35)",
          }}
        >
          <span className="text-white font-display italic text-[17px] font-medium">M</span>
        </div>
        {!rail && (
          <div>
            <div className="font-bold text-[15px] tracking-tight">MO Money</div>
            <div className="text-[10px] text-[#8a90a2] uppercase tracking-[0.10em] font-medium">
              Trading Journal
            </div>
          </div>
        )}
      </div>

      {/* Nav scroll area */}
      <nav className="flex-1 overflow-y-auto overflow-x-hidden px-2 py-1.5 scrollbar-thin">
        {NAV.map((group) => {
          const isOpen = !!openGroups[group.id] || rail;
          const hasActive = group.items.some((i) => i.id === activePage);

          return (
            <div key={group.id} className="mb-0.5">
              {/* Group header */}
              <button
                className="w-full flex items-center gap-2.5 px-2.5 py-[9px] rounded-[10px] hover:bg-black/[0.03] transition-[background] duration-150"
                onClick={(e) => !rail && toggleGroup(group.id, e.altKey)}
                style={{ "--nav-color": group.color } as React.CSSProperties}
              >
                {/* Color stripe */}
                <span
                  className="w-[3px] rounded-full shrink-0 transition-all duration-200"
                  style={{
                    height: isOpen || hasActive ? 18 : 14,
                    background: group.color,
                    opacity: isOpen || hasActive ? 1 : 0.35,
                    boxShadow:
                      isOpen || hasActive
                        ? `0 0 8px color-mix(in oklab, ${group.color} 60%, transparent)`
                        : "none",
                  }}
                />
                {!rail && (
                  <>
                    <span
                      className="flex-1 text-left text-[11px] uppercase tracking-[0.10em] font-semibold transition-colors"
                      style={{ color: hasActive ? group.color : "#5a6175" }}
                    >
                      {group.label}
                    </span>
                    <span
                      className="font-num text-[10px] rounded-full px-[7px] min-w-[18px] text-center"
                      style={{
                        background: hasActive
                          ? `color-mix(in oklab, ${group.color} 12%, transparent)`
                          : "#eef0f6",
                        color: hasActive ? group.color : "#8a90a2",
                      }}
                    >
                      {group.items.length}
                    </span>
                    {/* Chevron */}
                    <svg
                      className="shrink-0 text-[#8a90a2] transition-transform duration-250"
                      style={{ transform: isOpen ? "rotate(90deg)" : "rotate(0deg)" }}
                      width="12"
                      height="12"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <polyline points="9 18 15 12 9 6" />
                    </svg>
                  </>
                )}
              </button>

              {/* Items (accordion) */}
              {!rail && (
                <div
                  className="grid transition-[grid-template-rows] duration-300 ease-out"
                  style={{ gridTemplateRows: isOpen ? "1fr" : "0fr" }}
                >
                  <div className="min-h-0 overflow-hidden pl-5 pr-1.5 pb-1.5">
                    {group.items.map((item, itemIdx) => {
                      const isActive = activePage === item.id;
                      return (
                        <button
                          key={item.id}
                          className="w-full flex items-center gap-2.5 px-2.5 py-[7px] rounded-lg text-[13px] font-medium transition-all duration-[120ms] relative text-left"
                          style={{
                            "--nav-color": group.color,
                            background: isActive
                              ? `color-mix(in oklab, ${group.color} 14%, transparent)`
                              : "transparent",
                            color: isActive ? group.color : "#2c3243",
                            fontWeight: isActive ? 600 : 500,
                            animationDelay: `${itemIdx * 0.02}s`,
                          } as React.CSSProperties}
                          onClick={() => onNavigate(item.id)}
                        >
                          {isActive && (
                            <span
                              className="absolute left-[-14px] top-1/2 -translate-y-1/2 w-[3px] h-4 rounded-full"
                              style={{ background: group.color }}
                            />
                          )}
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
      <div className="border-t border-[#e6e8ef] px-3.5 py-3 bg-[#fbfbfe]">
        <div className="flex items-center gap-2.5 p-1.5 rounded-[10px] hover:bg-[#eef0f6] transition-colors">
          <div
            className="w-[30px] h-[30px] rounded-full flex items-center justify-center text-white text-xs font-semibold"
            style={{ background: "linear-gradient(135deg, #6366f1, #8b5cf6)" }}
          >
            M
          </div>
          {!rail && (
            <div className="flex-1 min-w-0">
              <div className="text-[13px] font-semibold">MO</div>
              <div className="text-[11px] text-[#8a90a2]">CanSlim</div>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}
