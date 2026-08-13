"use client";

import { usePathname, useRouter } from "next/navigation";
import { ActiveCampaign } from "@/components/active-campaign";
import { MobileActiveCampaign } from "@/components/mobile/mobile-active-campaign";
import { useIsMobile } from "@/lib/use-viewport";
import { getGroupForHref, hrefForId } from "@/lib/nav";

export default function ActiveCampaignClient() {
  const pathname = usePathname();
  const router = useRouter();
  const isMobile = useIsMobile();
  const navColor = getGroupForHref(pathname)?.color || "#6366f1";
  const handleNavigate = (id: string) => {
    const href = hrefForId(id);
    if (href) router.push(href);
  };
  if (isMobile) return <MobileActiveCampaign navColor={navColor} />;
  return <ActiveCampaign navColor={navColor} onNavigate={handleNavigate} />;
}
