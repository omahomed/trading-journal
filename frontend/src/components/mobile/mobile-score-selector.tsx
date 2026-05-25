"use client";

/**
 * 1-5 chip-group score selector. Replaces the desktop daily-routine's
 * range slider — sliders are touch-hostile on small viewports (small
 * thumb, hard to land precisely). Chips give a clean tap target
 * (≥44px) per score value with color tinting that mirrors the
 * grade-tier semantics used elsewhere (gradeColor, scoreColor):
 *   - 1-2 → red (poor)
 *   - 3   → amber (mid)
 *   - 4-5 → green (good)
 *
 * Default tier mapping matches the desktop scoreColor helper at
 * daily-routine.tsx:44. Callers can override via the `tierFor` prop
 * if the calling context wants a different palette mapping (none
 * does today; left in as a forward-compat hatch).
 */

export type ScoreTier = "low" | "mid" | "high";

const VALUES = [1, 2, 3, 4, 5] as const;

function defaultTierFor(v: number): ScoreTier {
  if (v >= 4) return "high";
  if (v >= 3) return "mid";
  return "low";
}

type Props = {
  /** Sentence-case label rendered above the chip row. */
  label: string;
  /** Current 1-5 value. */
  value: number;
  /** Fires with the new value on chip tap. */
  onChange: (v: number) => void;
  /** Override the default 1-2/3/4-5 → low/mid/high mapping. */
  tierFor?: (v: number) => ScoreTier;
  /** Accessibility label override (default = label). */
  ariaLabel?: string;
};

export function MobileScoreSelector({
  label,
  value,
  onChange,
  tierFor = defaultTierFor,
  ariaLabel,
}: Props) {
  return (
    <div
      role="radiogroup"
      aria-label={ariaLabel ?? label}
      className="flex items-center justify-between gap-3"
    >
      <span className="text-[12px] font-medium text-m-text">{label}</span>
      <div className="flex items-center gap-1">
        {VALUES.map((v) => {
          const isActive = v === value;
          const tier = tierFor(v);
          const baseClass = isActive
            ? `${chipBase} ${activeChipClass(tier)}`
            : `${chipBase} border-[0.5px] border-m-border bg-m-surface text-m-text-muted`;
          // The `down` tier has no Tailwind tint utility (no --m-down-tint
          // token exists), so the active-low chip background falls back
          // to inline color-mix per the convention used in mobile-position-
          // sizer.tsx and mobile-trade-journal.tsx.
          const inlineStyle =
            isActive && tier === "low"
              ? {
                  background:
                    "color-mix(in oklab, var(--m-down) 14%, var(--m-surface))",
                }
              : undefined;
          return (
            <button
              key={v}
              type="button"
              role="radio"
              aria-checked={isActive}
              aria-label={`${label}: ${v} of 5`}
              data-testid={`score-chip-${label.toLowerCase().replace(/\s+/g, "-")}-${v}`}
              data-tier={tier}
              onClick={() => onChange(v)}
              className={baseClass}
              style={inlineStyle}
            >
              {v}
            </button>
          );
        })}
      </div>
    </div>
  );
}

const chipBase =
  "flex h-9 w-9 items-center justify-center rounded-m-pill font-m-num text-[13px] font-semibold tabular-nums transition-colors";

function activeChipClass(tier: ScoreTier): string {
  if (tier === "high") return "bg-m-accent text-m-accent-text-on";
  if (tier === "mid") return "bg-m-warn-tint text-m-warn";
  // `low` tier: only text class here — background is applied via the
  // inline color-mix style at the call site (no Tailwind tint utility).
  return "text-m-down";
}
