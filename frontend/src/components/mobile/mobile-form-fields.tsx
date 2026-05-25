"use client";

import type { ReactNode } from "react";

/**
 * Mobile form-field primitives shared across mobile screens
 * (position sizer, daily routine, etc.). Extracted from
 * mobile-position-sizer.tsx (Phase 2 Step 5) so multiple consumers
 * don't reinvent the same cell shape.
 *
 * Each cell renders as a single-row tile with a small uppercase
 * label and a single value/input. No multi-input layout — compose
 * cells in a grid container at the call site for that.
 */

// ── NumberFieldCell ───────────────────────────────────────────────
// Number input with inputMode="decimal" so iOS shows the right
// keypad. Stored as string at the call site (matches the form-state
// pattern across the mobile sizer / daily routine).

export function NumberFieldCell({
  label,
  value,
  onChange,
  onBlur,
  ariaLabel,
  suffix,
  placeholder,
  hasError,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  /** Optional blur handler. Mostly used by callers that want to
   *  flip a `touched` flag for validation-error gating. */
  onBlur?: () => void;
  ariaLabel: string;
  suffix?: string;
  placeholder?: string;
  /** Visual error treatment — red border + label tint. Doesn't
   *  render the error text itself (callers do that under the cell). */
  hasError?: boolean;
}) {
  const borderClass = hasError ? "border-m-down" : "border-m-border";
  const labelClass = hasError ? "text-m-down" : "text-m-text-dim";
  return (
    <label
      className={`block rounded-m-md border-[0.5px] ${borderClass} bg-m-surface px-[14px] py-[10px]`}
    >
      <span className={`mb-0.5 block text-[10px] font-medium ${labelClass}`}>
        {label}
      </span>
      <span className="flex items-baseline gap-1">
        <input
          type="text"
          inputMode="decimal"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onBlur={onBlur}
          aria-label={ariaLabel}
          placeholder={placeholder}
          className="min-w-0 flex-1 bg-transparent font-m-num text-lg font-medium tabular-nums text-m-text placeholder:text-m-text-faint focus:outline-none"
        />
        {suffix && (
          <span className="font-m-num text-lg font-medium tabular-nums text-m-text-dim">
            {suffix}
          </span>
        )}
      </span>
    </label>
  );
}

// ── ReadOnlyFieldCell ─────────────────────────────────────────────
// Display-only cell. Matches the shape of NumberFieldCell so a
// row mixing inputs and read-only metrics stays aligned.

export function ReadOnlyFieldCell({
  label,
  labelIcon,
  value,
}: {
  label: string;
  labelIcon?: ReactNode;
  value: string;
}) {
  return (
    <div className="rounded-m-md border-[0.5px] border-m-border bg-m-surface px-[14px] py-[10px]">
      <div className="mb-0.5 flex items-center gap-1 text-[10px] font-medium text-m-text-dim">
        {label}
        {labelIcon}
      </div>
      <div className="font-m-num text-lg font-medium tabular-nums text-m-text">
        {value}
      </div>
    </div>
  );
}

// ── TextFieldCell ─────────────────────────────────────────────────
// Free-form text input. Same shell as NumberFieldCell but without
// the decimal keypad / suffix slot, and the input uses normal type.
// Optional `multiline` mode renders a <textarea> instead of <input>.

export function TextFieldCell({
  label,
  value,
  onChange,
  ariaLabel,
  placeholder,
  multiline,
  rows = 3,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  ariaLabel: string;
  placeholder?: string;
  /** Render a <textarea> instead of a single-line <input>. */
  multiline?: boolean;
  /** Visible row height when multiline. */
  rows?: number;
}) {
  return (
    <label className="block rounded-m-md border-[0.5px] border-m-border bg-m-surface px-[14px] py-[10px]">
      <span className="mb-0.5 block text-[10px] font-medium text-m-text-dim">
        {label}
      </span>
      {multiline ? (
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          aria-label={ariaLabel}
          placeholder={placeholder}
          rows={rows}
          className="block w-full resize-none bg-transparent text-sm text-m-text placeholder:text-m-text-faint focus:outline-none"
        />
      ) : (
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          aria-label={ariaLabel}
          placeholder={placeholder}
          className="block w-full bg-transparent text-sm text-m-text placeholder:text-m-text-faint focus:outline-none"
        />
      )}
    </label>
  );
}
