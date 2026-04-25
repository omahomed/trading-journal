"use client";

import { usePathname, useSearchParams } from "next/navigation";
import { DailyReportCard } from "@/components/daily-report-card";
import { getGroupForHref } from "@/lib/nav";

export default function Route() {
  const navColor = getGroupForHref(usePathname())?.color || "#6366f1";
  const searchParams = useSearchParams();
  const initialDate = searchParams?.get("date") || undefined;
  return <DailyReportCard navColor={navColor} initialDate={initialDate} />;
}
