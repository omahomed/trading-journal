"use client";

import { usePathname } from "next/navigation";
import { Sr8Monitor } from "@/components/sr8-monitor";
import { getGroupForHref } from "@/lib/nav";

export default function Sr8MonitorClient() {
  const pathname = usePathname();
  const navColor = getGroupForHref(pathname)?.color || "#e5484d";
  return <Sr8Monitor navColor={navColor} />;
}
