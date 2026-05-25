"use client";

import { useState, useEffect } from "react";
import { usePathname } from "next/navigation";
import { TradeJournal } from "@/components/trade-journal";
import { MobileTradeJournal } from "@/components/mobile/mobile-trade-journal";
import { useIsMobile } from "@/lib/use-viewport";
import { getGroupForHref } from "@/lib/nav";

type Props = { initialTradeIdProp?: string };

export default function TradeJournalClient({ initialTradeIdProp }: Props) {
  const isMobile = useIsMobile();
  const navColor = getGroupForHref(usePathname())?.color || "#6366f1";
  const [initialTradeId, setInitialTradeId] = useState<string | undefined>(
    initialTradeIdProp,
  );

  useEffect(() => {
    setInitialTradeId(initialTradeIdProp);
  }, [initialTradeIdProp]);

  if (isMobile) {
    return (
      <MobileTradeJournal
        initialTradeId={initialTradeId}
        onTradeConsumed={() => setInitialTradeId(undefined)}
      />
    );
  }
  // Desktop reads ?trade_id= directly from window.location.search on mount
  // (see trade-journal.tsx:807-823) — unchanged.
  return <TradeJournal navColor={navColor} />;
}
