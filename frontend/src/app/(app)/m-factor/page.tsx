"use client";

import { usePathname } from "next/navigation";
import { MFactor } from "@/components/m-factor";
import { getGroupForHref } from "@/lib/nav";

export default function Route() {
  const navColor = getGroupForHref(usePathname())?.color || "#6366f1";
  return <MFactor navColor={navColor} />;
}
