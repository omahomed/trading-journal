"use client";

import type { ReactNode } from "react";

type Props = {
  checked: boolean;
  onChange: (next: boolean) => void;
  label: ReactNode;
  description?: ReactNode;
  /** id used to associate the label with the underlying checkbox for a11y. */
  id: string;
};

/**
 * iOS-style toggle row for mobile settings surfaces. Left side hosts a
 * label + optional description; right side is the toggle. The whole row
 * is a single clickable label so tapping anywhere on the row flips the
 * toggle — better target than the 44px switch alone.
 *
 * Uses native checkbox semantics under the hood (visually hidden, but
 * tab-focusable + screen-reader-friendly).
 */
export function MobileToggleSwitch({ checked, onChange, label, description, id }: Props) {
  return (
    <label
      htmlFor={id}
      className="flex cursor-pointer items-center justify-between gap-3 px-4 py-3"
    >
      <span className="min-w-0 flex-1">
        <span className="block text-[14px] text-m-text">{label}</span>
        {description && (
          <span className="mt-0.5 block text-[12px] text-m-text-dim">{description}</span>
        )}
      </span>
      <span className="relative inline-flex h-[26px] w-[44px] shrink-0 items-center">
        <input
          id={id}
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="peer sr-only"
        />
        <span
          aria-hidden="true"
          className={
            "block h-[26px] w-[44px] rounded-m-pill transition-colors " +
            (checked ? "bg-m-accent" : "bg-m-surface-2")
          }
        />
        <span
          aria-hidden="true"
          className={
            "absolute top-0.5 inline-block h-[22px] w-[22px] rounded-full bg-white shadow-sm transition-transform " +
            (checked ? "translate-x-[20px]" : "translate-x-0.5")
          }
        />
      </span>
    </label>
  );
}
