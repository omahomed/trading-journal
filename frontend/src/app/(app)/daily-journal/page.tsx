import { connection } from "next/server";
import DailyJournalClient from "./daily-journal-client";

export default async function DailyJournalPage({
  searchParams,
}: {
  searchParams: Promise<{ date?: string }>;
}) {
  await connection();
  const sp = await searchParams;
  return <DailyJournalClient initialDate={sp.date} />;
}
