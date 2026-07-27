"use client";

import { usePathname } from "next/navigation";
import { DailyJournal } from "@/components/daily-journal";
import { MobileDailyJournal } from "@/components/mobile/mobile-daily-journal";
import { useIsMobile } from "@/lib/use-viewport";
import { getGroupForHref } from "@/lib/nav";

type Props = { initialDate?: string };

export default function DailyJournalClient({ initialDate }: Props) {
  const isMobile = useIsMobile();
  const navColor = getGroupForHref(usePathname())?.color || "#f59f00";
  if (isMobile) return <MobileDailyJournal initialDate={initialDate} navColor={navColor} />;
  return <DailyJournal navColor={navColor} initialDate={initialDate} />;
}
