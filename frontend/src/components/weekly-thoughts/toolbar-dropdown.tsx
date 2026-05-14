"use client";

// Phase 3.5: shared dropdown primitive used by the three toolbar selects
// (Style / Font family / Font size). Custom button + popover, NOT a
// native <select> — native controls are OS-themed and won't match the
// toolbar aesthetic, and we need to preserve the editor's selection
// across open/close (native <select> would steal focus).
//
// Selection-preservation pattern: the trigger button is `onMouseDown
// preventDefault`, which keeps the editor's selection alive while the
// dropdown opens. The current Range is also saved on open as a defense
// in depth, so option clicks can restore selection before the parent
// runs execCommand (the actual restore + focus happens here; the
// parent's onSelect is called *after* the editor is re-focused and
// the Range is back in place).

import { useCallback, useEffect, useRef, useState } from "react";
import { Icons } from "../icons";

export interface ToolbarDropdownOption {
  value: string;
  label: React.ReactNode;
}

interface ToolbarDropdownProps {
  ariaLabel: string;
  triggerLabel: React.ReactNode;
  options: ToolbarDropdownOption[];
  onSelect: (value: string) => void;
  /** Minimum popover width in px. Default 110. */
  width?: number;
  /** Editor element — focused before onSelect runs so execCommand
   *  operates on the live selection. */
  editorRef?: React.RefObject<HTMLElement | null>;
}

export function ToolbarDropdown({
  ariaLabel,
  triggerLabel,
  options,
  onSelect,
  width = 110,
  editorRef,
}: ToolbarDropdownProps) {
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

  // Click-outside-to-close. Matches the TagPicker pattern.
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

  const handlePick = useCallback((value: string) => {
    setOpen(false);
    // Restore editor focus + selection BEFORE invoking onSelect so the
    // parent's execCommand call runs against the same Range the user
    // had before opening the dropdown.
    if (editorRef?.current) editorRef.current.focus();
    restoreSelection();
    onSelect(value);
  }, [editorRef, restoreSelection, onSelect]);

  return (
    <div ref={wrapperRef} style={{ position: "relative", display: "inline-flex" }}>
      <button
        type="button"
        onMouseDown={e => e.preventDefault() /* preserve editor selection */}
        onClick={toggleOpen}
        aria-label={ariaLabel}
        aria-haspopup="listbox"
        aria-expanded={open}
        title={ariaLabel}
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 4,
          height: 26,
          padding: "0 6px",
          background: open ? "var(--surface-2)" : "transparent",
          border: "none",
          borderRadius: 6,
          color: "var(--ink-3)",
          cursor: "pointer",
        }}
      >
        <span style={{ fontSize: 11, fontWeight: 500 }}>{triggerLabel}</span>
        <Icons.chevronDown />
      </button>
      {open && (
        <div
          role="listbox"
          style={{
            position: "absolute",
            top: "100%",
            left: 0,
            marginTop: 4,
            minWidth: width,
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderRadius: 8,
            boxShadow: "var(--card-shadow)",
            padding: 4,
            zIndex: 25,
          }}
        >
          {options.map(opt => (
            <button
              key={opt.value}
              role="option"
              aria-selected={false}
              type="button"
              onMouseDown={e => e.preventDefault()}
              onClick={() => handlePick(opt.value)}
              onMouseEnter={e => { e.currentTarget.style.background = "var(--surface-2)"; }}
              onMouseLeave={e => { e.currentTarget.style.background = "transparent"; }}
              style={{
                display: "block",
                width: "100%",
                textAlign: "left",
                padding: "6px 8px",
                fontSize: 12,
                background: "transparent",
                border: "none",
                borderRadius: 4,
                color: "var(--ink)",
                cursor: "pointer",
              }}
            >
              {opt.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
