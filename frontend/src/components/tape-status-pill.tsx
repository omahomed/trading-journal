"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";

type V11State = "POWERTREND" | "UPTREND" | "RALLY MODE" | "CORRECTION";

type PillData = {
  state: V11State;
  day_num?: number;
  cap_at_100?: boolean;
  drawdown_pct?: number;
};

const STATE_STYLE: Record<V11State, { dot: string; glow: string; label: string }> = {
  POWERTREND:   { dot: "#8A2BE2", glow: "#f0e2ff", label: "Power Trend" },
  UPTREND:      { dot: "#08a86b", glow: "#e5f7ee", label: "Uptrend" },
  "RALLY MODE": { dot: "#f59f00", glow: "#fff4d6", label: "Rally Mode" },
  CORRECTION:   { dot: "#e5484d", glow: "#fdeaea", label: "Correction" },
};

const V11_STATES = ["POWERTREND", "UPTREND", "RALLY MODE", "CORRECTION"] as const;

export function TapeStatusPill() {
  const [data, setData] = useState<PillData | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .rallyPrefix()
      .then((r) => {
        if (cancelled) return;
        if (r?.state && (V11_STATES as readonly string[]).includes(r.state)) {
          setData({
            state: r.state as V11State,
            day_num: r.day_num,
            cap_at_100: r.cap_at_100,
            drawdown_pct: r.drawdown_pct,
          });
        }
      })
      .catch(() => {
        if (!cancelled) setData(null);
      });
    return () => { cancelled = true; };
  }, []);

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
      href="/market-cycle"
      className="flex items-center gap-1.5 h-[30px] px-3 rounded-full text-xs font-medium bg-[var(--surface)] border border-[var(--border)] text-[var(--ink-2)] hover:bg-[#eef0f6] transition-colors"
      title="Open Market Cycle Tracker"
    >
      <span
        className="w-1.5 h-1.5 rounded-full"
        style={{
          backgroundColor: style.dot,
          boxShadow: `0 0 0 3px ${style.glow}`,
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
  if (d.state === "POWERTREND" || d.state === "UPTREND" || d.state === "RALLY MODE") {
    return d.day_num && d.day_num > 0 ? `Day ${d.day_num}` : "";
  }
  if (d.state === "CORRECTION") {
    return typeof d.drawdown_pct === "number" ? `${d.drawdown_pct.toFixed(1)}%` : "";
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
