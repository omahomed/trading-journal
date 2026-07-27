"use client";

import { usePathname } from "next/navigation";
import { JournalLog } from "@/components/journal-log";
import { MobileJournalLog } from "@/components/mobile/mobile-journal-log";
import { useIsMobile } from "@/lib/use-viewport";
import { getGroupForHref } from "@/lib/nav";

export default function JournalLogClient() {
  const isMobile = useIsMobile();
  const navColor = getGroupForHref(usePathname())?.color || "#6366f1";
  if (isMobile) return <MobileJournalLog />;
  return <JournalLog navColor={navColor} />;
}
