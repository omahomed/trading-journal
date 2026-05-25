"use client";

import { usePathname } from "next/navigation";
import { DailyRoutine } from "@/components/daily-routine";
import { getGroupForHref } from "@/lib/nav";

export default function DailyRoutineClient() {
  const navColor = getGroupForHref(usePathname())?.color || "#6366f1";
  return <DailyRoutine navColor={navColor} />;
}
