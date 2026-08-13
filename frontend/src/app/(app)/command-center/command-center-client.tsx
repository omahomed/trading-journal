"use client";

import { usePathname } from "next/navigation";
import { CommandCenter } from "@/components/command-center";
import { MobileCommandCenter } from "@/components/mobile/mobile-command-center";
import { useIsMobile } from "@/lib/use-viewport";
import { getGroupForHref } from "@/lib/nav";

export default function CommandCenterClient() {
  const isMobile = useIsMobile();
  const navColor = getGroupForHref(usePathname())?.color || "#334155";
  if (isMobile) return <MobileCommandCenter />;
  return <CommandCenter navColor={navColor} />;
}
