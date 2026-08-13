"use client";

import { usePathname } from "next/navigation";
import { TradingOverview } from "@/components/trading-overview";
import { MobileTradingOverview } from "@/components/mobile/mobile-trading-overview";
import { useIsMobile } from "@/lib/use-viewport";
import { getGroupForHref } from "@/lib/nav";

export default function OverviewClient() {
  const isMobile = useIsMobile();
  const navColor = getGroupForHref(usePathname())?.color || "#6366f1";
  if (isMobile) return <MobileTradingOverview />;
  return <TradingOverview navColor={navColor} />;
}
