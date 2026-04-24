"use client";

import { usePathname } from "next/navigation";
import { TradeJournal } from "@/components/trade-journal";
import { getGroupForHref } from "@/lib/nav";

export default function Route() {
  const navColor = getGroupForHref(usePathname())?.color || "#6366f1";
  return <TradeJournal navColor={navColor} />;
}
