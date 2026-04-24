"use client";

import { usePathname } from "next/navigation";
import { DailyReportCard } from "@/components/daily-report-card";
import { getGroupForHref } from "@/lib/nav";

export default function Route() {
  const navColor = getGroupForHref(usePathname())?.color || "#6366f1";
  return <DailyReportCard navColor={navColor} />;
}
