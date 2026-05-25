"use client";

import { usePathname } from "next/navigation";
import { DailyReportCard } from "@/components/daily-report-card";
import { getGroupForHref } from "@/lib/nav";

type Props = { initialDate?: string };

export default function DailyReportClient({ initialDate }: Props) {
  const navColor = getGroupForHref(usePathname())?.color || "#6366f1";
  return <DailyReportCard navColor={navColor} initialDate={initialDate} />;
}
