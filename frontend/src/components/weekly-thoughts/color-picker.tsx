"use client";

// Phase 4.2 — color picker for Weekly Thoughts highlight + text color
// buttons. Replaces the Phase 3 ToolbarColorBtn which only applied a
// single fixed color. Now the user picks from a swatch palette.
//
// Selection preservation: the picker saves the editor's Range when
// it opens (before focus moves into the picker), then restores it
// before invoking onPick. The parent's onPick handler calls
// execCommand("hiliteColor" / "foreColor", color), which operates on
// the restored selection.
//
// The trailing entry of the palette is interpreted as a "reset"
// sentinel — for highlight it's "transparent" (clears bg-color),
// for text color it's "inherit" (resets to editor default). The
// reset swatch renders with a diagonal-slash visual instead of a
// solid fill so it's distinguishable at a glance.
//
// Mirrors the ToolbarDropdown / UrlPopover patterns from Phase 3.5:
// onMouseDown.preventDefault on the trigger button (keeps the
// editor selection alive across click); click-outside + Esc
// dismiss; popover anchored absolutely below the trigger.

import { useCallback, useEffect, useRef, useState } from "react";

interface ColorPickerProps {
  ariaLabel: string;
  /** Icon shown on the trigger button. */
  triggerGlyph: React.ReactNode;
  /** Optional underline color on the trigger button — usually the
   *  default / first palette entry, so the button hints at the
   *  feature without re-renders on each pick. */
  triggerSwatch?: string;
  /** Color values to pick from. The last entry is treated as the
   *  reset sentinel (rendered with diagonal-slash). */
  palette: string[];
  /** Visible text inside the reset swatch (e.g., "None" or
   *  "Default"). Defaults to empty string. */
  resetLabel?: string;
  /** Called with the picked color string. Parent invokes
   *  execCommand here. The picker closes regardless of what the
   *  parent does. */
  onPick: (color: string) => void;
  /** Editor element — focused before onPick runs so execCommand
   *  operates on the restored selection. */
  editorRef?: React.RefObject<HTMLElement | null>;
}

export function ColorPicker({
  ariaLabel,
  triggerGlyph,
  triggerSwatch,
  palette,
  resetLabel = "",
  onPick,
  editorRef,
}: ColorPickerProps) {
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const savedRangeRef = useRef<Range | null>(null);

  const saveSelection = useCallback(() => {
    if (typeof window === "undefined") return;
    const sel = window.getSelection();
    savedRangeRef.current =
      sel && sel.rangeCount > 0 ? sel.getRangeAt(0).cloneRange() : null;
  }, []);

  const restoreSelection = useCallback(() => {
    if (!savedRangeRef.current) return;
    const sel = window.getSelection();
    if (!sel) return;
    sel.removeAllRanges();
    sel.addRange(savedRangeRef.current);
  }, []);

  const toggleOpen = useCallback(() => {
    setOpen(prev => {
      if (!prev) saveSelection();
      return !prev;
    });
  }, [saveSelection]);

  // Click-outside-to-close — mirrors the ToolbarDropdown idiom.
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (!wrapperRef.current) return;
      if (!wrapperRef.current.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("mousedown", handler);
    return () => window.removeEventListener("mousedown", handler);
  }, [open]);

  // Esc closes.
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.preventDefault(); setOpen(false); }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open]);

  const handlePick = useCallback((color: string) => {
    setOpen(false);
    if (editorRef?.current) editorRef.current.focus();
    restoreSelection();
    onPick(color);
  }, [editorRef, restoreSelection, onPick]);

  // Last palette entry is the reset sentinel — render differently.
  const lastIdx = palette.length - 1;

  return (
    <div ref={wrapperRef} style={{ position: "relative", display: "inline-flex" }}>
      <button
        type="button"
        onMouseDown={e => e.preventDefault() /* preserve editor selection */}
        onClick={toggleOpen}
        aria-label={ariaLabel}
        aria-haspopup="dialog"
        aria-expanded={open}
        title={ariaLabel}
        style={{
          display: "inline-flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 1,
          height: 26,
          padding: "0 6px",
          background: open ? "var(--surface-2)" : "transparent",
          border: "none",
          borderRadius: 6,
          color: "var(--ink-3)",
          cursor: "pointer",
        }}
      >
        <span style={{ display: "inline-flex" }}>{triggerGlyph}</span>
        {triggerSwatch && (
          <span style={{
            display: "block",
            width: 14,
            height: 3,
            borderRadius: 1,
            background: triggerSwatch,
          }} />
        )}
      </button>

      {open && (
        <div
          role="dialog"
          aria-label={ariaLabel}
          style={{
            position: "absolute",
            top: "100%",
            left: 0,
            marginTop: 4,
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderRadius: 10,
            boxShadow: "var(--card-shadow)",
            padding: 8,
            zIndex: 30,
            // 4-column grid sized for 24px swatches + 4px gaps + 8px
            // surrounding padding.
            display: "grid",
            gridTemplateColumns: "repeat(4, 24px)",
            gap: 4,
          }}
        >
          {palette.map((color, idx) => {
            const isReset = idx === lastIdx;
            return (
              <button
                key={`${color}-${idx}`}
                type="button"
                onMouseDown={e => e.preventDefault()}
                onClick={() => handlePick(color)}
                aria-label={isReset ? (resetLabel || "Reset") : color}
                title={isReset ? (resetLabel || "Reset") : color}
                style={{
                  width: 24,
                  height: 24,
                  borderRadius: 6,
                  border: "1px solid var(--border)",
                  // Reset swatch gets a white/transparent fill + the
                  // diagonal-slash via linear-gradient. Other swatches
                  // get the palette color directly.
                  background: isReset
                    ? "linear-gradient(to top right, transparent calc(50% - 1px), #dc2626 50%, transparent calc(50% + 1px))"
                    : color,
                  cursor: "pointer",
                  padding: 0,
                }}
              />
            );
          })}
          {resetLabel && (
            <div style={{
              gridColumn: "1 / -1",
              fontSize: 10,
              color: "var(--ink-4)",
              textAlign: "right",
              marginTop: 2,
            }}>
              ↗ {resetLabel}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
