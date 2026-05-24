"use client";

import { usePathname } from "next/navigation";
import { Dashboard } from "@/components/dashboard";
import { getGroupForHref } from "@/lib/nav";

export default function DashboardClient() {
  const pathname = usePathname();
  const navColor = getGroupForHref(pathname)?.color || "#6366f1";
  return <Dashboard navColor={navColor} />;
}
