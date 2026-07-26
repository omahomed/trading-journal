"use client";

import { usePathname } from "next/navigation";
import { DailyRoutine } from "@/components/daily-routine";
import { MobileDailyRoutine } from "@/components/mobile/mobile-daily-routine";
import { useIsMobile } from "@/lib/use-viewport";
import { getGroupForHref } from "@/lib/nav";

type Props = { initialDate?: string };

export default function DailyRoutineClient({ initialDate }: Props) {
  const isMobile = useIsMobile();
  const navColor = getGroupForHref(usePathname())?.color || "#f59f00";
  if (isMobile) return <MobileDailyRoutine initialDate={initialDate} />;
  return <DailyRoutine navColor={navColor} initialDate={initialDate} />;
}
