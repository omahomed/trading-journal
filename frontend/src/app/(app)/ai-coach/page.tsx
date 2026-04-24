"use client";

import { usePathname } from "next/navigation";
import { AICoach } from "@/components/ai-coach";
import { getGroupForHref } from "@/lib/nav";

export default function Route() {
  const navColor = getGroupForHref(usePathname())?.color || "#6366f1";
  return <AICoach navColor={navColor} />;
}
