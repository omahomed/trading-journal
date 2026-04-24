"use client";

import { usePathname } from "next/navigation";
import { WeeklyRetro } from "@/components/weekly-retro";
import { getGroupForHref } from "@/lib/nav";

export default function Route() {
  const navColor = getGroupForHref(usePathname())?.color || "#6366f1";
  return <WeeklyRetro navColor={navColor} />;
}
