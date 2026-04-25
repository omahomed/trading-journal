"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";

type State = "POWERTREND" | "UPTREND" | "RALLY MODE" | "CORRECTION";

const STATE_STYLE: Record<State, { dot: string; ring: string; label: string }> = {
  POWERTREND: { dot: "#8A2BE2", ring: "#efe4fb", label: "Power-Trend" },
  UPTREND: { dot: "#08a86b", ring: "#e5f7ee", label: "Confirmed Uptrend" },
  "RALLY MODE": { dot: "#f59f00", ring: "#fdf2d8", label: "Rally Mode" },
  CORRECTION: { dot: "#e5484d", ring: "#fde7e8", label: "Correction" },
};

function fmtSince(iso?: string | null) {
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return "";
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

export function TapeStatusPill() {
  const router = useRouter();
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    let alive = true;
    api.rallyPrefix().then(d => { if (alive) setData(d); }).catch(() => {});
    return () => { alive = false; };
  }, []);

  const state = (data?.state || "UPTREND") as State;
  const style = STATE_STYLE[state] || STATE_STYLE.UPTREND;

  let detail = "";
  if (state === "POWERTREND") {
    const since = fmtSince((data as any)?.power_trend_on_since);
    detail = since ? ` · since ${since}` : "";
  } else if (state === "UPTREND") {
    const since = fmtSince((data as any)?.ftd_date);
    detail = since ? ` · since ${since}` : "";
  } else if (state === "RALLY MODE") {
    const dn = (data as any)?.day_num || 0;
    detail = dn ? ` · Day ${dn}` : "";
  } else if (state === "CORRECTION") {
    const dd = (data as any)?.drawdown_pct;
    detail = typeof dd === "number" ? ` · ${Math.abs(dd).toFixed(1)}% off high` : "";
  }

  return (
    <button onClick={() => router.push("/market-cycle")}
            title="Open Market Cycle Tracker"
            className="flex items-center gap-1.5 h-[30px] px-3 rounded-full text-xs font-medium bg-[var(--surface)] border border-[var(--border)] text-[var(--ink-2)] hover:bg-[#eef0f6] transition-colors">
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: style.dot, boxShadow: `0 0 0 3px ${style.ring}`, animation: "pulse-dot 2s ease-in-out infinite" }} />
      <span>{style.label}{detail}</span>
    </button>
  );
}
