"use client";

import { usePathname } from "next/navigation";
import { WeeklyRetro } from "@/components/weekly-retro";
import { MobileWeeklyRetro } from "@/components/mobile/mobile-weekly-retro";
import { useIsMobile } from "@/lib/use-viewport";
import { getGroupForHref } from "@/lib/nav";

type Props = { initialWeek?: string };

export default function WeeklyRetroClient({ initialWeek }: Props) {
  const isMobile = useIsMobile();
  const navColor = getGroupForHref(usePathname())?.color || "#6366f1";
  if (isMobile) return <MobileWeeklyRetro />;
  return <WeeklyRetro navColor={navColor} initialWeek={initialWeek} />;
}
