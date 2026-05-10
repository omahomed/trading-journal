"use client";

import { useState, useEffect } from "react";
import { usePathname, useSearchParams } from "next/navigation";
import { Analytics } from "@/components/analytics";
import { getGroupForHref } from "@/lib/nav";

export default function Route() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const navColor = getGroupForHref(pathname)?.color || "#6366f1";
  const [initialTab, setInitialTab] = useState<string | undefined>(searchParams.get("tab") || undefined);
  const [initialTradeId, setInitialTradeId] = useState<string | undefined>(searchParams.get("trade_id") || undefined);

  useEffect(() => { setInitialTab(searchParams.get("tab") || undefined); }, [searchParams]);
  useEffect(() => { setInitialTradeId(searchParams.get("trade_id") || undefined); }, [searchParams]);

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
