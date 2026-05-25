"use client";

import { usePathname } from "next/navigation";
import { DailyRoutine } from "@/components/daily-routine";
import { MobileDailyRoutine } from "@/components/mobile/mobile-daily-routine";
import { useIsMobile } from "@/lib/use-viewport";
import { getGroupForHref } from "@/lib/nav";

export default function DailyRoutineClient() {
  const isMobile = useIsMobile();
  const navColor = getGroupForHref(usePathname())?.color || "#6366f1";
  if (isMobile) return <MobileDailyRoutine />;
  return <DailyRoutine navColor={navColor} />;
}
