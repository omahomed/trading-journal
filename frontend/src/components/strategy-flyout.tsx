"use client";

import { useEffect, useRef, useState } from "react";
import { StrategyChip } from "./strategy-chip";
import type { Strategy } from "@/lib/api";

// Right-click "Set strategy →" submenu used on three surfaces (ACS, All
// Campaigns, Trade Journal). Two render modes:
//   - Desktop (mouse): hover/focus reveals a flyout to the right of the
//     parent menu item. If parent_rect.right + flyout_width would
//     overflow the viewport, the flyout flips to the left side.
//   - Coarse pointer (touch): hover doesn't exist, so we render a flat
//     "Set as: <Strategy>" cluster of menu items inline instead of a
//     reveal. Detected via matchMedia("(pointer: coarse)").
//
// `coarsePointer` is exported as a small helper so the parent context
// menu can render either the flyout or the flat list without each caller
// re-implementing the detection.

const FLYOUT_WIDTH = 200;

export function useCoarsePointer(): boolean {
  const [coarse, setCoarse] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const mql = window.matchMedia("(pointer: coarse)");
    setCoarse(mql.matches);
    const handler = (e: MediaQueryListEvent) => setCoarse(e.matches);
    mql.addEventListener("change", handler);
    return () => mql.removeEventListener("change", handler);
  }, []);
  return coarse;
}

interface StrategyFlyoutProps {
  strategies: Strategy[];
  currentStrategy?: string;
  onPick: (name: string) => void;
}

// Desktop variant: hover/focus the parent label reveals the flyout to
// the side. Auto-flips horizontally when the right edge would clip the
// viewport. The parent is responsible for closing the outer context
// menu after onPick fires.
export function StrategyFlyout({ strategies, currentStrategy, onPick }: StrategyFlyoutProps) {
  const [open, setOpen] = useState(false);
  const [flipLeft, setFlipLeft] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  // Measure the parent's bounding rect on hover-open. Compare its right
  // edge + the flyout width against window.innerWidth. If overflow,
  // anchor the flyout to the parent's LEFT edge (open leftward) instead.
  useEffect(() => {
    if (!open || !wrapperRef.current) return;
    const rect = wrapperRef.current.getBoundingClientRect();
    setFlipLeft(rect.right + FLYOUT_WIDTH > window.innerWidth);
  }, [open]);

  return (
    <div
      ref={wrapperRef}
      className="relative"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      <button
        type="button"
        className="w-full text-left px-3 py-2 text-[12px] font-medium flex items-center justify-between gap-2 transition-colors hover:brightness-95"
        style={{ color: "var(--ink)" }}
        onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-2)")}
        onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
      >
        <span className="flex items-center gap-2">
          <span style={{ color: "var(--ink-4)" }}>&#x1F3F7;</span> Set strategy
        </span>
        <span style={{ color: "var(--ink-4)" }}>›</span>
      </button>
      {open && strategies.length > 0 && (
        <div
          className="absolute top-0 z-50 rounded-[10px] py-1.5 overflow-hidden"
          style={{
            [flipLeft ? "right" : "left"]: "100%",
            [flipLeft ? "left" : "right"]: "auto",
            width: FLYOUT_WIDTH,
            background: "var(--surface)",
            border: "1px solid var(--border)",
            boxShadow: "0 8px 24px rgba(0,0,0,0.16), 0 2px 6px rgba(0,0,0,0.08)",
          }}
          data-testid="strategy-flyout"
        >
          {strategies.map(s => {
            const isCurrent = s.name === currentStrategy;
            return (
              <button
                key={s.name}
                type="button"
                className="w-full text-left px-3 py-2 text-[12px] font-medium flex items-center gap-2 transition-colors hover:brightness-95"
                style={{ color: "var(--ink)", background: isCurrent ? "var(--surface-2)" : "transparent" }}
                onMouseEnter={e => { if (!isCurrent) e.currentTarget.style.background = "var(--surface-2)"; }}
                onMouseLeave={e => { if (!isCurrent) e.currentTarget.style.background = "transparent"; }}
                onClick={e => { e.stopPropagation(); onPick(s.name); }}
              >
                <StrategyChip name={s.name} color={s.color} size="md" />
                {isCurrent && (
                  <span className="ml-auto text-[10px]" style={{ color: "var(--ink-4)" }}>current</span>
                )}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

// Touch variant: flat list of "Set as: <Strategy>" buttons, rendered
// inline by the parent context menu. No hover state; tap fires onPick.
// Used when matchMedia("(pointer: coarse)") matches.
export function StrategyFlatList({ strategies, currentStrategy, onPick }: StrategyFlyoutProps) {
  return (
    <>
      {strategies.map(s => {
        const isCurrent = s.name === currentStrategy;
        return (
          <button
            key={s.name}
            type="button"
            className="w-full text-left px-3 py-2 text-[12px] font-medium flex items-center gap-2 transition-colors"
            style={{ color: "var(--ink)", background: isCurrent ? "var(--surface-2)" : "transparent" }}
            onClick={e => { e.stopPropagation(); onPick(s.name); }}
            data-testid="strategy-flat-item"
          >
            <span style={{ color: "var(--ink-4)" }}>Set as</span>
            <StrategyChip name={s.name} color={s.color} size="md" />
            {isCurrent && (
              <span className="ml-auto text-[10px]" style={{ color: "var(--ink-4)" }}>current</span>
            )}
          </button>
        );
      })}
    </>
  );
}
