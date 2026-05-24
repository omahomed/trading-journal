"use client";

import { usePathname } from "next/navigation";
import { Dashboard } from "@/components/dashboard";
import { MobileDashboard } from "@/components/mobile/mobile-dashboard";
import { getGroupForHref } from "@/lib/nav";
import { useIsMobile } from "@/lib/use-viewport";

export default function DashboardClient() {
  const pathname = usePathname();
  const isMobile = useIsMobile();
  const navColor = getGroupForHref(pathname)?.color || "#6366f1";
  return isMobile ? <MobileDashboard /> : <Dashboard navColor={navColor} />;
}
