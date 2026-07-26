"use client";

import { usePathname } from "next/navigation";
import { TradingChecklist } from "@/components/trading-checklist";
import { MobileTradingChecklist } from "@/components/mobile/mobile-trading-checklist";
import { useIsMobile } from "@/lib/use-viewport";
import { getGroupForHref } from "@/lib/nav";

export default function TradingChecklistClient() {
  const isMobile = useIsMobile();
  const navColor = getGroupForHref(usePathname())?.color || "#f59f00";
  if (isMobile) return <MobileTradingChecklist navColor={navColor} />;
  return <TradingChecklist navColor={navColor} />;
}
