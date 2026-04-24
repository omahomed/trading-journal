"use client";

import { usePathname } from "next/navigation";
import { DailyJournal } from "@/components/daily-journal";
import { getGroupForHref } from "@/lib/nav";

export default function Route() {
  const navColor = getGroupForHref(usePathname())?.color || "#6366f1";
  return <DailyJournal navColor={navColor} />;
}
