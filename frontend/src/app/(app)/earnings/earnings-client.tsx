"use client";

import { usePathname } from "next/navigation";
import { EarningsPlanner } from "@/components/earnings-planner";
import { getGroupForHref } from "@/lib/nav";

export default function EarningsClient() {
  const navColor = getGroupForHref(usePathname())?.color || "#6366f1";
  return <EarningsPlanner navColor={navColor} />;
}
