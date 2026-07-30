"use client";

import { usePathname } from "next/navigation";
import { ConcentrationRisk } from "@/components/concentration-risk";
import { getGroupForHref } from "@/lib/nav";

export default function ConcentrationRiskClient() {
  const navColor = getGroupForHref(usePathname())?.color || "#e5484d";
  return <ConcentrationRisk navColor={navColor} />;
}
