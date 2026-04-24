"use client";

import { usePathname } from "next/navigation";
import { MarketCycle } from "@/components/market-cycle";
import { getGroupForHref } from "@/lib/nav";

export default function Route() {
  const navColor = getGroupForHref(usePathname())?.color || "#6366f1";
  return <MarketCycle navColor={navColor} />;
}
