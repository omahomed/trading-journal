"use client";

import { usePathname, useRouter } from "next/navigation";
import { ActiveCampaign } from "@/components/active-campaign";
import { getGroupForHref, hrefForId } from "@/lib/nav";

export default function Route() {
  const pathname = usePathname();
  const router = useRouter();
  const navColor = getGroupForHref(pathname)?.color || "#6366f1";
  const handleNavigate = (id: string) => {
    const href = hrefForId(id);
    if (href) router.push(href);
  };
  return <ActiveCampaign navColor={navColor} onNavigate={handleNavigate} />;
}
