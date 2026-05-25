"use client";

import { usePathname } from "next/navigation";
import { DailyJournal } from "@/components/daily-journal";
import { MobileDailyJournal } from "@/components/mobile/mobile-daily-journal";
import { useIsMobile } from "@/lib/use-viewport";
import { getGroupForHref } from "@/lib/nav";

export default function DailyJournalClient() {
  const isMobile = useIsMobile();
  const navColor = getGroupForHref(usePathname())?.color || "#6366f1";
  if (isMobile) return <MobileDailyJournal />;
  return <DailyJournal navColor={navColor} />;
}
