"use client";

import { formatCurrency } from "@/lib/format";

export type WeeklyInsightsTileFormat = "currency" | "percent" | "raw";

export interface WeeklyInsightsTileProps {
  label: string;
  value: number | null | undefined;
  formatType: WeeklyInsightsTileFormat;
  gradient: string;
  subtitle?: string;
  loading?: boolean;
}

function formatPercent(v: number): string {
  const sign = v > 0 ? "+" : v < 0 ? "" : "";
  return `${sign}${v.toFixed(2)}%`;
}

function formatValue(v: number | null | undefined, fmt: WeeklyInsightsTileFormat): string {
  if (v == null || Number.isNaN(v)) return "—";
  if (fmt === "currency") return formatCurrency(v, { showSign: true });
  if (fmt === "percent") return formatPercent(v);
  return String(v);
}

// Phase 5 — single gradient KPI tile. Visual contract pulled from
// Design/design_handoff_weekly_retro: 14px radius, white-on-gradient,
// 26px JetBrains-mono value, 9px uppercase label, 10px subtitle, with the
// radial overlay in the top-right corner.
//
// Negative values: gradient stays (the tile's identity); the number text
// is dimmed via opacity and prefixed with a "↓" glyph. Red text on
// emerald gradient is unreadable, so the sign carries the direction.
export function WeeklyInsightsTile({
  label, value, formatType, gradient, subtitle, loading,
}: WeeklyInsightsTileProps) {
  const isNegative = typeof value === "number" && value < 0;
  const display = formatValue(value, formatType);

  return (
    <div
      data-testid="weekly-insights-tile"
      data-label={label}
      data-negative={isNegative ? "true" : "false"}
      className="relative overflow-hidden rounded-[14px] flex flex-col justify-between"
      style={{
        background: gradient,
        color: "#fff",
        padding: "14px 16px",
        minHeight: 92,
        boxShadow: "0 6px 18px rgba(14,20,38,0.10)",
      }}
    >
      {/* Radial overlay — the soft white blob in the upper-right of the
          design's KPI tile. Purely decorative. */}
      <span
        aria-hidden
        className="pointer-events-none absolute"
        style={{
          right: -20, top: -20, width: 110, height: 110, borderRadius: 999,
          background: "radial-gradient(circle, rgba(255,255,255,0.22), transparent 65%)",
        }}
      />

      <div
        className="relative font-semibold uppercase"
        style={{ fontSize: 9, letterSpacing: "0.10em", opacity: 0.88 }}
      >
        {label}
      </div>

      <div
        className="relative font-semibold privacy-mask"
        style={{
          fontFamily: "var(--font-jetbrains), monospace",
          fontFeatureSettings: '"tnum" 1',
          fontSize: 26,
          letterSpacing: "-0.02em",
          marginTop: 2,
          // Slight dim on negatives — the glyph + sign do the heavy lifting.
          opacity: loading ? 0.6 : isNegative ? 0.92 : 1,
        }}
      >
        {loading ? "…" : (
          <>
            {isNegative && <span aria-hidden style={{ marginRight: 2 }}>↓</span>}
            {display}
          </>
        )}
      </div>

      {subtitle && (
        <div
          className="relative font-medium"
          style={{ fontSize: 10, opacity: 0.85 }}
        >
          {subtitle}
        </div>
      )}
    </div>
  );
}
