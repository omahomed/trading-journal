"use client";

// Phase 2 (Migration 019). Reusable visual for a strategy tag — colored
// dot + (optional) name. Six surfaces use it: Log Buy dropdown options,
// Log Buy trigger button, right-click "Set strategy →" submenu rows,
// bulk "Tag as" dropdown rows, Trade Journal card pill, Admin table.
//
// `color` and `name` are passed in from the strategies table — never
// hardcoded. Sizes:
//   sm — 8×8 dot, name optional (bulk dropdown rows)
//   md — 10×10 dot + name (default — Log Buy, submenu)
//   lg — pill shape, larger swatch (Trade Journal card footer)

export type StrategyChipSize = "sm" | "md" | "lg";

// Visual variants:
//   "dot"    — original neutral treatment (colored dot + optional name on
//              a --surface-2 pill when size=lg + showName). Backward-
//              compatible default, keeps Log Buy / bulk-tag dropdowns
//              looking the way they always have.
//   "filled" — colored pill: the strategy's own color mixed into the
//              background (14% over --surface), border at 28%, TEXT in
//              the strategy color. High-visibility variant for ACS ticker
//              cells and the mobile position cards. Same treatment
//              pattern as the SR tier chips.
export type StrategyChipVariant = "dot" | "filled";

interface StrategyChipProps {
  name: string;
  color: string;
  size?: StrategyChipSize;
  showName?: boolean;
  className?: string;
  title?: string;
  variant?: StrategyChipVariant;
}

const DOT_PX: Record<StrategyChipSize, number> = { sm: 8, md: 10, lg: 12 };
const TEXT_PX: Record<StrategyChipSize, number> = { sm: 11, md: 12, lg: 12 };
const GAP_PX: Record<StrategyChipSize, number> = { sm: 6, md: 8, lg: 8 };

export function StrategyChip({
  name,
  color,
  size = "md",
  showName = true,
  className = "",
  title,
  variant = "dot",
}: StrategyChipProps) {
  const dot = DOT_PX[size];
  const isFilled = variant === "filled";
  const isNeutralPill = !isFilled && size === "lg" && showName;

  const baseStyle: React.CSSProperties = isFilled
    ? {
        background: `color-mix(in oklab, ${color} 14%, var(--surface))`,
        border: `1px solid color-mix(in oklab, ${color} 28%, var(--border))`,
        borderRadius: 6,
        padding: "1px 6px",
      }
    : isNeutralPill
      ? {
          background: "var(--surface-2)",
          border: "1px solid var(--border)",
          borderRadius: 999,
          padding: "2px 8px 2px 6px",
        }
      : {};

  const textColor = isFilled ? color : "var(--ink)";

  return (
    <span
      className={`inline-flex items-center shrink-0 ${className}`}
      style={{ gap: GAP_PX[size], ...baseStyle }}
      title={title ?? name}
      data-strategy-variant={variant}
    >
      <span
        aria-hidden="true"
        className="inline-block rounded-full shrink-0"
        style={{ width: dot, height: dot, background: color }}
      />
      {showName && (
        <span
          className="truncate"
          style={{ fontSize: TEXT_PX[size], color: textColor, fontWeight: isFilled ? 600 : undefined }}
        >
          {name}
        </span>
      )}
    </span>
  );
}
