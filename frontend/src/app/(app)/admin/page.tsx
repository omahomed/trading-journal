"use client";

import { usePathname } from "next/navigation";
import { Admin } from "@/components/admin";
import { getGroupForHref } from "@/lib/nav";

export default function Route() {
  const navColor = getGroupForHref(usePathname())?.color || "#6366f1";
  return <Admin navColor={navColor} />;
}
