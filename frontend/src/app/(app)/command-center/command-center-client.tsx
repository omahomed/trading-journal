"use client";

import { usePathname } from "next/navigation";
import { CommandCenter } from "@/components/command-center";
import { getGroupForHref } from "@/lib/nav";

export default function CommandCenterClient() {
  const navColor = getGroupForHref(usePathname())?.color || "#334155";
  return <CommandCenter navColor={navColor} />;
}
