"use client";

import { useEffect, useState, type ReactNode } from "react";
import { ChevronDown, X } from "lucide-react";

type Props = {
  /** Small label above the value (e.g., "Mode"). */
  triggerLabel: string;
  /** Selected value's primary label (e.g., "Offense"). */
  triggerValue: string;
  /** Selected value's secondary text below (e.g., "1.00%"). */
  triggerSubValue?: string;
  /** Render the trigger value in the accent (green) color. */
  triggerAccent?: boolean;
  /** Render the trigger tile with the selected-state border. */
  triggerSelected?: boolean;
  /** Title shown at the top of the bottom sheet. */
  sheetTitle: string;
  /** Tailwind max-height class for the sheet body. Default `"max-h-[85vh]"`
   *  fits the option-list use case; consumers with content-dense bodies
   *  (e.g. Trade Journal detail sheet) can opt up to `"max-h-[95vh]"`. */
  maxHeightClass?: string;
  /** Render-prop for the sheet body — typically a list of options.
   *  Receives a `close` callback so option handlers can dismiss the
   *  sheet after committing a selection. */
  children: (close: () => void) => ReactNode;
};

/**
 * Trigger tile + bottom-sheet primitive used by the mobile Position
 * Sizer's Mode / Profile / Size pickers. Mirrors the open/close
 * behavior of MobilePortfolioPicker (Escape + backdrop + body-scroll-
 * lock + slide-up animation) but exposes a tile-shaped trigger
 * instead of a header pill.
 *
 * The body is passed as a render-prop so each consumer decides how to
 * render its options list (label + sub-value, plain label, etc.)
 * while the primitive handles the shell.
 */
export function MobileSelectSheet({
  triggerLabel,
  triggerValue,
  triggerSubValue,
  triggerAccent = false,
  triggerSelected = false,
  sheetTitle,
  maxHeightClass = "max-h-[85vh]",
  children,
}: Props) {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [open]);

  const valueToneClass = triggerAccent ? "text-m-accent" : "text-m-text";
  const borderClass = triggerSelected ? "border-m-accent-border-soft" : "border-m-border";

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label={`${triggerLabel}: ${triggerValue}. Tap to change.`}
        className={`rounded-m-md border-[0.5px] ${borderClass} bg-m-surface px-3 py-[10px] text-left`}
      >
        <div className="mb-1 text-[10px] font-medium text-m-text-dim">{triggerLabel}</div>
        <div className="flex items-baseline justify-between">
          <span className={`text-sm font-medium ${valueToneClass}`}>{triggerValue}</span>
          <ChevronDown size={10} strokeWidth={1.5} className="text-m-text-dim" aria-hidden="true" />
        </div>
        {triggerSubValue && (
          <div className="mt-px font-m-num text-[11px] tabular-nums text-m-text-muted">
            {triggerSubValue}
          </div>
        )}
      </button>

      {open && (
        <>
          <button
            type="button"
            aria-label={`Close ${sheetTitle}`}
            onClick={() => setOpen(false)}
            className="fixed inset-0 z-40 bg-black/50"
            style={{ animation: "m-backdrop-enter var(--m-duration-tap) ease-out" }}
          />
          <div
            role="dialog"
            aria-modal="true"
            aria-label={sheetTitle}
            className={`fixed inset-x-0 bottom-0 z-50 flex ${maxHeightClass} flex-col border-t-[0.5px] border-m-border bg-m-bg`}
            style={{
              borderTopLeftRadius: "var(--m-radius-xl)",
              borderTopRightRadius: "var(--m-radius-xl)",
              animation: "m-sheet-enter var(--m-duration-sheet) var(--m-ease-spring)",
            }}
          >
            <div className="flex shrink-0 items-center justify-between border-b-[0.5px] border-m-border px-5 pt-4 pb-3">
              <h2 className="text-base font-medium text-m-text">{sheetTitle}</h2>
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
                clears the home indicator. */}
            <div
              role="listbox"
              aria-label={sheetTitle}
              className="min-h-0 flex-1 overflow-y-auto"
              style={{
                paddingBottom: "max(1.5rem, env(safe-area-inset-bottom))",
                WebkitOverflowScrolling: "touch",
              }}
            >
              {children(() => setOpen(false))}
            </div>
          </div>
        </>
      )}
    </>
  );
}
