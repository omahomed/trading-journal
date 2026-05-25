"use client";

import { usePathname } from "next/navigation";
import { TradingOverview } from "@/components/trading-overview";
import { getGroupForHref } from "@/lib/nav";

export default function OverviewClient() {
  const navColor = getGroupForHref(usePathname())?.color || "#6366f1";
  return <TradingOverview navColor={navColor} />;
}
