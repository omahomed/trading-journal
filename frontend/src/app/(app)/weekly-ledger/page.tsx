import { connection } from "next/server";
import WeeklyLedgerClient from "./weekly-ledger-client";

export default async function WeeklyLedgerPage({
  searchParams,
}: {
  searchParams: Promise<{ week?: string }>;
}) {
  await connection();
  const sp = await searchParams;
  return <WeeklyLedgerClient initialWeek={sp.week} />;
}
