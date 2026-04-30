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

  return (
    <Link
      href="/m-factor"
      className="flex items-center gap-2 rounded-m-pill border-[0.5px] border-m-purple-border bg-m-purple-tint px-3.5 py-2 text-xs"
      aria-label={`Cycle: ${data.state}${detail ? `, ${detail}` : ""}`}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-m-purple" />
      <span className="font-medium text-m-purple-text">{data.state}</span>
      {detail && (
        <>
          <span className="text-m-text-faint">·</span>
          <span className="text-m-text-muted">{detail}</span>
        </>
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
  return d.day_num && d.day_num > 0 ? `Day ${d.day_num}` : "";
}
