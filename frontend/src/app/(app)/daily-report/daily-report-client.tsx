"use client";

import { usePathname } from "next/navigation";
import { DailyReportCard } from "@/components/daily-report-card";
import { MobileDailyReport } from "@/components/mobile/mobile-daily-report";
import { useIsMobile } from "@/lib/use-viewport";
import { getGroupForHref } from "@/lib/nav";

type Props = { initialDate?: string };

export default function DailyReportClient({ initialDate }: Props) {
  const isMobile = useIsMobile();
  const navColor = getGroupForHref(usePathname())?.color || "#6366f1";
  if (isMobile) return <MobileDailyReport initialDate={initialDate} />;
  return <DailyReportCard navColor={navColor} initialDate={initialDate} />;
}
