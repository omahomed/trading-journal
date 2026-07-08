"use client";

import Link from "next/link";
import { useRallyState, type RallyState } from "@/lib/use-rally-state";

/**
 * Mobile cycle indicator. Visual variant of the desktop `TapeStatusPill`
 * — same data, restyled to match the locked mobile anchor: warm-dark
 * purple-tinted pill, all-caps state label, separator dot, day count,
 * optional right-aligned cap indicator.
 *
 * Both pills share `useRallyState()` so a route mounting both desktop
 * and mobile chrome (Step 5's `AdaptiveShell`) makes a single fetch.
 */
export function MobileTapePill() {
  const data = useRallyState();

  if (!data) {
    return (
      <div
        className="flex items-center gap-2 rounded-m-pill border-[0.5px] border-m-border bg-m-surface px-3.5 py-2 text-xs text-m-text-dim"
        aria-label="Cycle indicator loading"
      >
        <span className="h-1.5 w-1.5 rounded-full bg-m-text-faint" />
        <span>—</span>
      </div>
    );
  }

  const detail = formatDetail(data);
  // UUP-only visual override — keeps the mobile pill's purple design
  // language for the four legacy states EXACTLY as-is. When state is
  // "UPTREND UNDER PRESSURE" (backend-only for now; dormant this
  // commit) the dot recolors to the deep-amber warn token and the
  // visible label becomes the shortened display alias. Machine string
  // stays full in aria-label so screen-reader + assertion callers get
  // the byte-identical value.
  const isUup = data.state === "UPTREND UNDER PRESSURE";
  const displayLabel = isUup ? "Uptrend · Pressure" : data.state;

  return (
    <Link
      href="/m-factor"
      className="flex items-center gap-2 rounded-m-pill border-[0.5px] border-m-purple-border bg-m-purple-tint px-3.5 py-2 text-xs"
      aria-label={`Cycle: ${data.state}${detail ? `, ${detail}` : ""}`}
    >
      <span
        className="h-1.5 w-1.5 rounded-full bg-m-purple"
        aria-hidden="true"
        style={isUup ? { backgroundColor: "var(--m-warn-deep)" } : undefined}
      />
      <span className="font-medium text-m-purple-text">{displayLabel}</span>
      {detail && (
        <>
          <span className="text-m-text-faint">·</span>
          <span className="text-m-text-muted">{detail}</span>
        </>
      )}
      {typeof data.trend_count === "number" && data.trend_count !== 0 && (
        <span className="ml-1 inline-flex items-center gap-1 text-[11px] font-semibold">
          <span className="text-m-text-muted">Trend Cycle</span>
          <span style={{ color: data.trend_count > 0 ? "#08a86b" : "#e5484d" }}>
            {data.trend_count > 0 ? `▲ Up +${data.trend_count}` : `▼ Down ${data.trend_count}`}
          </span>
        </span>
      )}
      {data.cap_at_100 && (
        <span className="ml-auto text-[11px] text-m-text-muted">Cap 100%</span>
      )}
    </Link>
  );
}

function formatDetail(d: RallyState): string {
  if (d.state === "CORRECTION") {
    return typeof d.drawdown_pct === "number"
      ? `${Math.abs(d.drawdown_pct).toFixed(1)}% off high`
      : "";
  }
  // Prefer state-onset date ("Since Apr 14") over day count when the
  // backend exposes it. Less mental math, and the absolute date stays
  // meaningful across timezones / late-night sessions where "Day N"
  // depends on which trading day the user thinks they're in.
  const since = formatSinceDate(d.power_trend_on_since);
  if (since) return `Since ${since}`;
  return d.day_num && d.day_num > 0 ? `Day ${d.day_num}` : "";
}

function formatSinceDate(iso: string | null | undefined): string | null {
  if (!iso) return null;
  // Parse the YYYY-MM-DD date as UTC midnight, then format in the same
  // UTC zone — keeps "Apr 14" stable regardless of the viewer's TZ.
  const dt = new Date(`${iso}T00:00:00Z`);
  if (Number.isNaN(dt.getTime())) return null;
  return dt.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });
}
