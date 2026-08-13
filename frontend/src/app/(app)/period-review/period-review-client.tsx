"use client";

import { useState, useEffect } from "react";
import { usePathname } from "next/navigation";
import { PeriodReview } from "@/components/period-review";
import { MobilePeriodReview } from "@/components/mobile/mobile-period-review";
import { useIsMobile } from "@/lib/use-viewport";
import { getGroupForHref } from "@/lib/nav";

type Props = { initialTabProp?: string };

export default function PeriodReviewClient({ initialTabProp }: Props) {
  const pathname = usePathname();
  const isMobile = useIsMobile();
  const navColor = getGroupForHref(pathname)?.color || "#6366f1";
  const [initialTab, setInitialTab] = useState<string | undefined>(initialTabProp);

  useEffect(() => { setInitialTab(initialTabProp); }, [initialTabProp]);

  if (isMobile) return <MobilePeriodReview />;

  return (
    <PeriodReview
      navColor={navColor}
      initialTab={initialTab}
      onTabConsumed={() => setInitialTab(undefined)}
    />
  );
}
