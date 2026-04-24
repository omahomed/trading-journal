"use client";

import { usePathname } from "next/navigation";
import { LogSell } from "@/components/log-sell";
import { getGroupForHref } from "@/lib/nav";

export default function Route() {
  const navColor = getGroupForHref(usePathname())?.color || "#6366f1";
  return <LogSell navColor={navColor} />;
}
