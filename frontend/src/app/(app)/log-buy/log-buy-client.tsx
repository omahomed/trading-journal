"use client";

import { usePathname } from "next/navigation";
import { LogBuy } from "@/components/log-buy";
import { getGroupForHref } from "@/lib/nav";

export default function LogBuyClient() {
  const navColor = getGroupForHref(usePathname())?.color || "#6366f1";
  return <LogBuy navColor={navColor} />;
}
