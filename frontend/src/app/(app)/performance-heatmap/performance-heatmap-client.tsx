"use client";

import { usePathname } from "next/navigation";
import { PerfHeatmap } from "@/components/perf-heatmap";
import { getGroupForHref } from "@/lib/nav";

export default function PerformanceHeatmapClient() {
  const navColor = getGroupForHref(usePathname())?.color || "#6366f1";
  return <PerfHeatmap navColor={navColor} />;
}
