"use client";

import { useCallback, useEffect, useState, type ReactNode } from "react";
import { X } from "lucide-react";

/**
 * Mobile edit sheet — Phase 2 T2-4b.
 *
 * Generic fullscreen-style sheet for hosting editable content. Hosts:
 * Daily Recap rich-text editor, Daily Thoughts rich-text editor,
 * Market Notes plain-text editor. Could also host any future
 * "edit one field" affordance.
 *
 * Composition contract:
 *   - Parent owns the open/dirty state and the value
 *   - Sheet renders header (X close · title · Save) + body slot
 *   - Dirty-dismiss is handled here: X / Escape / backdrop tap with
 *     `isDirty=true` shows a confirm action sheet (Discard / Keep
 *     Editing). Clean dismiss fires `onClose` immediately.
 *   - Save action is a `rightAction` — consumer wires the API call
 *     and clears its localStorage draft on success
 *
 * Differences from MobileSelectSheet:
 *   - Fullscreen feel (max-h-[85vh]) instead of natural content height
 *   - Drag handle bar at top (visual; gesture handling = v2)
 *   - Three-zone header with Save action on the right
 *   - Dirty-dismiss confirm path (confirm sheet renders inline)
 *
 * visualViewport observation lives in MobileRichTextEditor (the
 * keyboard-aware consumer), NOT here. Keeps this primitive generic.
 */

export type MobileEditSheetProps = {
  /** Whether the sheet is open. Parent owns this state. */
  open: boolean;
  /** Close callback — fired on clean dismiss OR after Discard confirm. */
  onClose: () => void;
  /** Sheet title shown in the header. */
  title: string;
  /** Consumer-computed dirty flag (current value !== initial). Drives
   *  the dirty-dismiss confirm path. */
  isDirty: boolean;
  /** Right-edge action button (typically Save). Consumer wires the
   *  onClick to API save + draft clear + close. */
  rightAction?: {
    label: string;
    onClick: () => void;
    disabled?: boolean;
  };
  /** Sheet body content. Editor / textarea / form. */
  children: ReactNode;
};

export function MobileEditSheet({
  open,
  onClose,
  title,
  isDirty,
  rightAction,
  children,
}: MobileEditSheetProps) {
  const [confirmOpen, setConfirmOpen] = useState(false);

  // Reset internal confirm state whenever the sheet (re)opens.
  useEffect(() => {
    if (!open) setConfirmOpen(false);
  }, [open]);

  // Escape: close the confirm if shown, else attempt close on the sheet.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (confirmOpen) {
        setConfirmOpen(false);
        return;
      }
      if (isDirty) setConfirmOpen(true);
      else onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, isDirty, onClose, confirmOpen]);

  // Body scroll lock while open. Mirrors MobileSelectSheet pattern.
  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [open]);

  const attemptClose = useCallback(() => {
    if (isDirty) setConfirmOpen(true);
    else onClose();
  }, [isDirty, onClose]);

  const handleDiscardConfirm = useCallback(() => {
    setConfirmOpen(false);
    onClose();
  }, [onClose]);

  if (!open) return null;

  return (
    <div data-testid="mobile-edit-sheet">
      {/* Backdrop — tap = attempt close (dirty → confirm). */}
      <button
        type="button"
        aria-label={`Close ${title}`}
        onClick={attemptClose}
        className="fixed inset-0 z-30 bg-black/55"
        style={{
          animation: "m-backdrop-enter var(--m-duration-tap) ease-out",
        }}
        data-testid="mobile-edit-sheet-backdrop"
      />

      {/* Sheet panel. Slides up from bottom. */}
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className="fixed inset-x-0 bottom-0 z-40 flex max-h-[85vh] flex-col border-t-[0.5px] border-m-border bg-m-bg"
        style={{
          borderTopLeftRadius: "var(--m-radius-xl)",
          borderTopRightRadius: "var(--m-radius-xl)",
          animation: "m-sheet-enter var(--m-duration-sheet) var(--m-ease-spring)",
        }}
      >
        {/* Drag handle (visual only). */}
        <div className="flex shrink-0 justify-center pt-2 pb-1" aria-hidden="true">
          <span className="h-1 w-8 rounded-full bg-m-border-strong" />
        </div>

        {/* Header: [X] · title · [Save] */}
        <div
          className="grid shrink-0 items-center gap-2 border-b-[0.5px] border-m-border px-3 pb-3 pt-1"
          style={{ gridTemplateColumns: "40px 1fr auto" }}
        >
          <button
            type="button"
            onClick={attemptClose}
            aria-label="Close"
            data-testid="mobile-edit-sheet-close"
            className="flex h-9 w-9 items-center justify-center rounded-m-pill text-m-text-dim active:opacity-80"
          >
            <X size={20} strokeWidth={1.6} aria-hidden="true" />
          </button>
          <h2
            data-testid="mobile-edit-sheet-title"
            className="min-w-0 truncate text-center text-[15px] font-medium text-m-text"
          >
            {title}
          </h2>
          {rightAction ? (
            <button
              type="button"
              onClick={rightAction.onClick}
              disabled={rightAction.disabled}
              data-testid="mobile-edit-sheet-save"
              className="rounded-m-pill bg-m-accent px-4 py-1.5 text-[13px] font-medium text-m-accent-text-on disabled:opacity-40"
            >
              {rightAction.label}
            </button>
          ) : (
            <span className="w-[60px]" aria-hidden="true" />
          )}
        </div>

        {/* Body — fills remaining height; scrolls internally. */}
        <div
          data-testid="mobile-edit-sheet-body"
          className="relative min-h-0 flex-1 overflow-y-auto"
          style={{
            paddingBottom: "max(1rem, env(safe-area-inset-bottom))",
            WebkitOverflowScrolling: "touch",
          }}
        >
          {children}
        </div>
      </div>

      {/* Dirty-dismiss confirm action sheet. Stacks above the parent
          sheet via z-50. Bottom-anchored, iOS Mail style. */}
      {confirmOpen && (
        <>
          <button
            type="button"
            aria-label="Keep editing"
            onClick={() => setConfirmOpen(false)}
            className="fixed inset-0 z-50 bg-black/35"
            data-testid="mobile-edit-sheet-confirm-backdrop"
          />
          <div
            role="alertdialog"
            aria-modal="true"
            aria-label="Unsaved changes"
            data-testid="mobile-edit-sheet-confirm"
            className="fixed inset-x-0 bottom-0 z-50 flex flex-col gap-2 px-3 pb-4"
            style={{ paddingBottom: "max(1rem, env(safe-area-inset-bottom))" }}
          >
            <div className="overflow-hidden rounded-m-md border-[0.5px] border-m-border bg-m-surface">
              <div className="border-b-[0.5px] border-m-border px-4 py-3 text-center text-[12px] text-m-text-dim">
                You have unsaved changes
              </div>
              <button
                type="button"
                onClick={handleDiscardConfirm}
                data-testid="mobile-edit-sheet-confirm-discard"
                className="block w-full px-4 py-3.5 text-center text-[16px] font-medium text-m-down active:bg-m-surface-2"
              >
                Discard Changes
              </button>
            </div>
            <button
              type="button"
              onClick={() => setConfirmOpen(false)}
              data-testid="mobile-edit-sheet-confirm-keep"
              className="rounded-m-md border-[0.5px] border-m-border bg-m-surface py-3.5 text-center text-[16px] font-semibold text-m-text active:bg-m-surface-2"
            >
              Keep Editing
            </button>
          </div>
        </>
      )}
    </div>
  );
}
