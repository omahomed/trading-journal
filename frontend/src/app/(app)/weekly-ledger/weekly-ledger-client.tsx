"use client";

import { usePathname } from "next/navigation";
import { WeeklyLedger } from "@/components/weekly-ledger";
import { getGroupForHref } from "@/lib/nav";

type Props = { initialWeek?: string };

export default function WeeklyLedgerClient({ initialWeek }: Props) {
  const navColor = getGroupForHref(usePathname())?.color || "#f59f00";
  return <WeeklyLedger navColor={navColor} initialWeek={initialWeek} />;
}
