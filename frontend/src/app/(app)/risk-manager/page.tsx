"use client";

import { usePathname } from "next/navigation";
import { RiskManager } from "@/components/risk-manager";
import { getGroupForHref } from "@/lib/nav";

export default function Route() {
  const navColor = getGroupForHref(usePathname())?.color || "#6366f1";
  return <RiskManager navColor={navColor} />;
}
