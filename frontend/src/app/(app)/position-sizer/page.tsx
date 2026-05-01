"use client";

import { useState, useEffect } from "react";
import { usePathname, useSearchParams, useRouter } from "next/navigation";
import { PositionSizer } from "@/components/position-sizer";
import { MobilePositionSizer } from "@/components/mobile/mobile-position-sizer";
import { useIsMobile } from "@/lib/use-viewport";
import { getGroupForHref, hrefForId } from "@/lib/nav";

export default function Route() {
  const isMobile = useIsMobile();
  const pathname = usePathname();
  const router = useRouter();
  const searchParams = useSearchParams();
  const navColor = getGroupForHref(pathname)?.color || "#6366f1";
  const [initialTab, setInitialTab] = useState<string | undefined>(searchParams.get("tab") || undefined);

  useEffect(() => { setInitialTab(searchParams.get("tab") || undefined); }, [searchParams]);

  if (isMobile) return <MobilePositionSizer />;

  const handleNavigate = (id: string) => {
    const href = hrefForId(id);
    if (href) router.push(href);
  };

  return (
    <PositionSizer
      navColor={navColor}
      onNavigate={handleNavigate}
      initialTab={initialTab}
      onTabConsumed={() => setInitialTab(undefined)}
    />
  );
}
