import { connection } from "next/server";
import TradeJournalClient from "./trade-journal-client";

export default async function TradeJournalPage() {
  await connection();
  return <TradeJournalClient />;
}
