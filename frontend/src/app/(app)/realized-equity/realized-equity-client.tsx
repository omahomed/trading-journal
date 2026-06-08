"use client";

import { usePathname } from "next/navigation";
import { RealizedEquity } from "@/components/realized-equity";
import { getGroupForHref } from "@/lib/nav";

export default function RealizedEquityClient() {
  const pathname = usePathname();
  const navColor = getGroupForHref(pathname)?.color || "#6366f1";
  return <RealizedEquity navColor={navColor} />;
}
