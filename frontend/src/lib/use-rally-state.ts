"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";

export type RallyV11State = "POWERTREND" | "UPTREND" | "RALLY MODE" | "CORRECTION";

export type RallyState = {
  state: RallyV11State;
  day_num?: number;
  cap_at_100?: boolean;
  drawdown_pct?: number;
  power_trend_on_since?: string | null;
  ftd_date?: string | null;
};

const V11_STATES = ["POWERTREND", "UPTREND", "RALLY MODE", "CORRECTION"] as const;

/**
 * Subscribes to the V11 rally-prefix endpoint. Returns `null` until the
 * first successful response (or permanently on error).
 *
 * Both the desktop `TapeStatusPill` and the mobile pill consume this
 * hook so the cycle indicator visible at the top of any page reflects a
 * single shared fetch — no double request when an `AdaptiveShell`
 * mounts both desktop and mobile chrome on the same route.
 *
 * Behavior contract preserved from the inline `useEffect` it replaces:
 *   - Returns `null` while loading.
 *   - Returns `null` if the response is missing `state` or has a state
 *     value not in the V11 set.
 *   - Returns `null` on fetch error.
 *   - Re-fetches only when the component mounts (no polling).
 */
export function useRallyState(): RallyState | null {
  const [data, setData] = useState<RallyState | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .rallyPrefix()
      .then((r) => {
        if (cancelled) return;
        if (r?.state && (V11_STATES as readonly string[]).includes(r.state)) {
          setData({
            state: r.state as RallyV11State,
            day_num: r.day_num,
            cap_at_100: r.cap_at_100,
            drawdown_pct: r.drawdown_pct,
            power_trend_on_since: r.power_trend_on_since,
            ftd_date: r.ftd_date,
          });
        }
      })
      .catch(() => {
        if (!cancelled) setData(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return data;
}
