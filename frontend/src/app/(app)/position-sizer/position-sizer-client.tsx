"use client";

import { useState, useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import { PositionSizer } from "@/components/position-sizer";
import { MobilePositionSizer } from "@/components/mobile/mobile-position-sizer";
import { useIsMobile } from "@/lib/use-viewport";
import { getGroupForHref, hrefForId } from "@/lib/nav";

type Props = {
  initialTabProp?: string;
  initialTradeIdProp?: string;
};

export default function PositionSizerClient({ initialTabProp, initialTradeIdProp }: Props) {
  const isMobile = useIsMobile();
  const pathname = usePathname();
  const router = useRouter();
  const navColor = getGroupForHref(pathname)?.color || "#6366f1";
  const [initialTab, setInitialTab] = useState<string | undefined>(initialTabProp);
  const [initialHoldingTradeId, setInitialHoldingTradeId] = useState<string | undefined>(initialTradeIdProp);

  useEffect(() => { setInitialTab(initialTabProp); }, [initialTabProp]);
  useEffect(() => { setInitialHoldingTradeId(initialTradeIdProp); }, [initialTradeIdProp]);

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
      initialHoldingTradeId={initialHoldingTradeId}
      onHoldingConsumed={() => setInitialHoldingTradeId(undefined)}
    />
  );
}
