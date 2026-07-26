"use client";

import { usePathname } from "next/navigation";
import { NLVEntry } from "@/components/nlv-entry";
import { MobileNLVEntry } from "@/components/mobile/mobile-nlv-entry";
import { useIsMobile } from "@/lib/use-viewport";
import { getGroupForHref } from "@/lib/nav";

export default function NLVEntryClient() {
  const isMobile = useIsMobile();
  const navColor = getGroupForHref(usePathname())?.color || "#f59f00";
  if (isMobile) return <MobileNLVEntry />;
  return <NLVEntry navColor={navColor} />;
}
