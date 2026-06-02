"use client";

import { usePathname } from "next/navigation";
import { CampaignDetail } from "@/components/campaign-detail";
import { getGroupForHref } from "@/lib/nav";

export default function CampaignDetailClient() {
  const pathname = usePathname();
  const navColor = getGroupForHref(pathname)?.color || "#08a86b";
  return <CampaignDetail navColor={navColor} />;
}
