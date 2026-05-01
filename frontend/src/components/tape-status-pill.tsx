"use client";

import Link from "next/link";
import { useRallyState, type RallyV11State, type RallyState as PillData } from "@/lib/use-rally-state";

type V11State = RallyV11State;

const STATE_STYLE: Record<V11State, { dot: string; ring: string; label: string }> = {
  POWERTREND:   { dot: "#8A2BE2", ring: "#efe4fb", label: "Power-Trend" },
  UPTREND:      { dot: "#08a86b", ring: "#e5f7ee", label: "Confirmed Uptrend" },
  "RALLY MODE": { dot: "#f59f00", ring: "#fdf2d8", label: "Rally Mode" },
  CORRECTION:   { dot: "#e5484d", ring: "#fde7e8", label: "Correction" },
};

function fmtSince(iso?: string | null): string {
  if (!iso) return "";
  // Parse YYYY-MM-DD as a LOCAL date — `new Date("2026-04-08")` is UTC
  // midnight which silently shifts a day west of UTC (the staging
  // user's CDT box would render "Apr 7" instead of "Apr 8"). Parsing
  // by component avoids any timezone interpretation.
  const m = iso.match(/^(\d{4})-(\d{2})-(\d{2})/);
  if (!m) return "";
  const d = new Date(parseInt(m[1], 10), parseInt(m[2], 10) - 1, parseInt(m[3], 10));
  if (isNaN(d.getTime())) return "";
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

export function TapeStatusPill() {
  const data = useRallyState();

  if (!data) {
    return (
      <div className="flex items-center gap-1.5 h-[30px] px-3 rounded-full text-xs font-medium bg-[var(--surface)] border border-[var(--border)] text-[var(--ink-3)]">
        <span className="w-1.5 h-1.5 rounded-full bg-[var(--ink-4)]" />
        <span>—</span>
      </div>
    );
  }

  const style = STATE_STYLE[data.state];
  const detail = formatDetail(data);

  return (
    <Link
      href="/m-factor"
      className="flex items-center gap-1.5 h-[30px] px-3 rounded-full text-xs font-medium bg-[var(--surface)] border border-[var(--border)] text-[var(--ink-2)] hover:bg-[#eef0f6] transition-colors"
      title="Open M Factor"
    >
      <span
        className="w-1.5 h-1.5 rounded-full"
        style={{
          backgroundColor: style.dot,
          boxShadow: `0 0 0 3px ${style.ring}`,
          animation: "pulse-dot 2s ease-in-out infinite",
        }}
      />
      <span>
        {style.label}
        {detail ? ` · ${detail}` : ""}
      </span>
      {data.cap_at_100 && <LockIcon />}
    </Link>
  );
}

function formatDetail(d: PillData): string {
  if (d.state === "POWERTREND") {
    const since = fmtSince(d.power_trend_on_since);
    return since ? `since ${since}` : "";
  }
  if (d.state === "UPTREND") {
    const since = fmtSince(d.ftd_date);
    return since ? `since ${since}` : "";
  }
  if (d.state === "RALLY MODE") {
    return d.day_num && d.day_num > 0 ? `Day ${d.day_num}` : "";
  }
  if (d.state === "CORRECTION") {
    return typeof d.drawdown_pct === "number"
      ? `${Math.abs(d.drawdown_pct).toFixed(1)}% off high`
      : "";
  }
  return "";
}

function LockIcon() {
  return (
    <svg
      width="11"
      height="11"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-label="Capped at 100%"
      role="img"
    >
      <rect x="4" y="11" width="16" height="10" rx="2" />
      <path d="M8 11V7a4 4 0 0 1 8 0v4" />
    </svg>
  );
}
