"use client";

import { usePathname } from "next/navigation";
import { EarningsPlanner } from "@/components/earnings-planner";
import { MobileEarningsPlanner } from "@/components/mobile/mobile-earnings-planner";
import { useIsMobile } from "@/lib/use-viewport";
import { getGroupForHref } from "@/lib/nav";

export default function EarningsClient() {
  const isMobile = useIsMobile();
  const navColor = getGroupForHref(usePathname())?.color || "#6366f1";
  if (isMobile) return <MobileEarningsPlanner />;
  return <EarningsPlanner navColor={navColor} />;
}
