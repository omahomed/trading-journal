"use client";

import { useEffect, useState } from "react";
import { Check, ChevronDown, X } from "lucide-react";
import { usePortfolio } from "@/lib/portfolio-context";

/**
 * Mobile portfolio switcher. Mounted as the right-slot accessory in the
 * mobile page header on every route. Consumes the shared
 * `usePortfolio()` context from `src/lib/portfolio-context.tsx` — the
 * same one the desktop sidebar uses — so a switch on mobile is
 * reflected on desktop (and vice-versa) via the
 * `localStorage["mo.activePortfolioId"]` channel and the hard reload
 * baked into `setActive`.
 *
 * Closed: pill-shaped button showing the active portfolio name and a
 * chevron-down. ≥44px touch target.
 *
 * Open: full-width bottom sheet that slides up from the viewport
 * bottom (thumb-reachable). Lists every portfolio; the active one is
 * marked with a green check. Tap a row to switch (which triggers
 * `setActive` → window reload), tap the backdrop / X / press Escape
 * to dismiss without switching.
 */
export function MobilePortfolioPicker() {
  const { portfolios, activePortfolio, setActive, loading } = usePortfolio();
  const [open, setOpen] = useState(false);

  // Close on Escape.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open]);

  // Lock body scroll while the sheet is open.
  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [open]);

  const buttonLabel = activePortfolio?.name ?? (loading ? "Loading…" : "—");
  const disabled = portfolios.length === 0;

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        disabled={disabled}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label={`Active portfolio: ${buttonLabel}. Tap to switch.`}
        className="flex min-h-[36px] items-center gap-1.5 rounded-m-pill border-[0.5px] border-m-border bg-m-surface px-3 py-1 text-sm font-medium text-m-text disabled:opacity-50"
      >
        <span className="max-w-[120px] truncate font-m-num tabular-nums">{buttonLabel}</span>
        <ChevronDown size={14} strokeWidth={1.5} className="text-m-text-dim" aria-hidden="true" />
      </button>

      {open && (
        <>
          <button
            type="button"
            aria-label="Close portfolio picker"
            onClick={() => setOpen(false)}
            className="fixed inset-0 z-40 bg-black/50"
            style={{
              animation: "m-backdrop-enter var(--m-duration-tap) ease-out",
            }}
          />
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="mobile-portfolio-picker-title"
            className="fixed inset-x-0 bottom-0 z-50 flex max-h-[85vh] flex-col border-t-[0.5px] border-m-border bg-m-bg"
            style={{
              borderTopLeftRadius: "var(--m-radius-xl)",
              borderTopRightRadius: "var(--m-radius-xl)",
              animation: "m-sheet-enter var(--m-duration-sheet) var(--m-ease-spring)",
            }}
          >
            <div className="flex shrink-0 items-center justify-between border-b-[0.5px] border-m-border px-5 pt-4 pb-3">
              <h2
                id="mobile-portfolio-picker-title"
                className="text-base font-medium text-m-text"
              >
                Switch portfolio
              </h2>
              <button
                type="button"
                onClick={() => setOpen(false)}
                aria-label="Close"
                className="flex h-8 w-8 items-center justify-center text-m-text-dim"
              >
                <X size={20} strokeWidth={1.5} aria-hidden="true" />
              </button>
            </div>

            {/* Items scroll independently. min-h-0 lets flex-1 actually
                shrink the listbox below its content height (otherwise
                long lists would force the parent to grow past max-h).
                Bottom padding respects iOS safe-area so the last item
                clears the home indicator. Mirrors the MobileSelectSheet
                fix from 242621b. */}
            <div
              role="listbox"
              aria-label="Portfolios"
              className="min-h-0 flex-1 overflow-y-auto"
              style={{
                paddingBottom: "max(1.5rem, env(safe-area-inset-bottom))",
                WebkitOverflowScrolling: "touch",
              }}
            >
              {portfolios.map((p) => {
                const isActive = p.id === activePortfolio?.id;
                return (
                  <button
                    key={p.id}
                    type="button"
                    role="option"
                    aria-selected={isActive}
                    onClick={() => {
                      // Triggers window.location.reload() inside the
                      // context; closing local state is moot but tidy.
                      setActive(p);
                      setOpen(false);
                    }}
                    className="flex min-h-[52px] items-center justify-between border-b-[0.5px] border-m-border px-5 py-3.5 text-left last:border-b-0"
                  >
                    <span className={`text-base ${isActive ? "font-medium" : ""} text-m-text`}>
                      {p.name}
                    </span>
                    {isActive && (
                      <Check
                        size={20}
                        strokeWidth={2}
                        className="text-m-accent"
                        aria-hidden="true"
                      />
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        </>
      )}
    </>
  );
}
