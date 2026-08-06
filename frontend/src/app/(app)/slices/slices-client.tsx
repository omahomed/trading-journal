"use client";

import { usePathname } from "next/navigation";
import { Slices } from "@/components/slices";
import { getGroupForHref } from "@/lib/nav";

export default function SlicesClient() {
  const navColor = getGroupForHref(usePathname())?.color || "#0891b2";
  return <Slices navColor={navColor} />;
}
