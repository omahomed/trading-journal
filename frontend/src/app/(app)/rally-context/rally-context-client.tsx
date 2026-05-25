"use client";

import { usePathname } from "next/navigation";
import { RallyContext } from "@/components/rally-context";
import { getGroupForHref } from "@/lib/nav";

export default function RallyContextClient() {
  const navColor = getGroupForHref(usePathname())?.color || "#6366f1";
  return <RallyContext navColor={navColor} />;
}
