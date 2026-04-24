"use client";

import { usePathname } from "next/navigation";
import { Dashboard } from "@/components/dashboard";
import { getGroupForHref } from "@/lib/nav";

export default function DashboardRoute() {
  const pathname = usePathname();
  const navColor = getGroupForHref(pathname)?.color || "#6366f1";
  return <Dashboard navColor={navColor} />;
}
