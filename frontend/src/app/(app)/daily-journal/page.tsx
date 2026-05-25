import { connection } from "next/server";
import DailyJournalClient from "./daily-journal-client";

export default async function DailyJournalPage() {
  await connection();
  return <DailyJournalClient />;
}
