"use client";

import { usePathname } from "next/navigation";
import { TradeJournal } from "@/components/trade-journal";
import { MobileTradeJournal } from "@/components/mobile/mobile-trade-journal";
import { useIsMobile } from "@/lib/use-viewport";
import { getGroupForHref } from "@/lib/nav";

export default function Route() {
  const isMobile = useIsMobile();
  const navColor = getGroupForHref(usePathname())?.color || "#6366f1";
  if (isMobile) return <MobileTradeJournal />;
  return <TradeJournal navColor={navColor} />;
}
