"use client";

import { usePathname } from "next/navigation";
import { SectorMapping } from "@/components/sector-mapping";
import { getGroupForHref } from "@/lib/nav";

export default function SectorMappingClient() {
  const navColor = getGroupForHref(usePathname())?.color || "#e5484d";
  return <SectorMapping navColor={navColor} />;
}
