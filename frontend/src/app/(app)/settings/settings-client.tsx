"use client";

import { usePathname } from "next/navigation";
import { Settings } from "@/components/settings";
import { getGroupForHref } from "@/lib/nav";

export default function SettingsClient() {
  const navColor = getGroupForHref(usePathname())?.color || "#6366f1";
  return <Settings navColor={navColor} />;
}
