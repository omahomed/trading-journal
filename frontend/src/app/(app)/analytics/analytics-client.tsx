"use client";

import { useState, useEffect } from "react";
import { usePathname } from "next/navigation";
import { Analytics } from "@/components/analytics";
import { getGroupForHref } from "@/lib/nav";

type Props = {
  initialTabProp?: string;
  initialTradeIdProp?: string;
};

export default function AnalyticsClient({ initialTabProp, initialTradeIdProp }: Props) {
  const pathname = usePathname();
  const navColor = getGroupForHref(pathname)?.color || "#6366f1";
  const [initialTab, setInitialTab] = useState<string | undefined>(initialTabProp);
  const [initialTradeId, setInitialTradeId] = useState<string | undefined>(initialTradeIdProp);

  useEffect(() => { setInitialTab(initialTabProp); }, [initialTabProp]);
  useEffect(() => { setInitialTradeId(initialTradeIdProp); }, [initialTradeIdProp]);

  return (
    <Analytics
      navColor={navColor}
      initialTab={initialTab}
      initialTradeId={initialTradeId}
      onTabConsumed={() => setInitialTab(undefined)}
      onTradeIdConsumed={() => setInitialTradeId(undefined)}
    />
  );
}
