"use client";

import { useState, useEffect } from "react";
import { usePathname, useSearchParams } from "next/navigation";
import { TradeManager } from "@/components/trade-manager";
import { getGroupForHref } from "@/lib/nav";

export default function Route() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const navColor = getGroupForHref(pathname)?.color || "#6366f1";
  const [initialTab, setInitialTab] = useState<string | undefined>(searchParams.get("tab") || undefined);

  useEffect(() => { setInitialTab(searchParams.get("tab") || undefined); }, [searchParams]);

  return (
    <TradeManager
      navColor={navColor}
      initialTab={initialTab}
      onTabConsumed={() => setInitialTab(undefined)}
    />
  );
}
