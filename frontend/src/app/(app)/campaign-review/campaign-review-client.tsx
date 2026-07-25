"use client";

import { usePathname } from "next/navigation";
import { CampaignReview } from "@/components/campaign-review";
import { getGroupForHref } from "@/lib/nav";

export default function CampaignReviewClient() {
  const pathname = usePathname();
  const navColor = getGroupForHref(pathname)?.color || "#0d6efd";
  return <CampaignReview navColor={navColor} />;
}
