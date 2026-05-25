"use client";

import { useState, useEffect } from "react";
import { usePathname } from "next/navigation";
import { TradeManager } from "@/components/trade-manager";
import { getGroupForHref } from "@/lib/nav";

type Props = { initialTabProp?: string };

export default function TradeManagerClient({ initialTabProp }: Props) {
  const pathname = usePathname();
  const navColor = getGroupForHref(pathname)?.color || "#6366f1";
  const [initialTab, setInitialTab] = useState<string | undefined>(initialTabProp);

  useEffect(() => { setInitialTab(initialTabProp); }, [initialTabProp]);

  return (
    <TradeManager
      navColor={navColor}
      initialTab={initialTab}
      onTabConsumed={() => setInitialTab(undefined)}
    />
  );
}
