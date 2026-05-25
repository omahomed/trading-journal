import { connection } from "next/server";
import TradeJournalClient from "./trade-journal-client";

export default async function TradeJournalPage({
  searchParams,
}: {
  searchParams: Promise<{ trade_id?: string }>;
}) {
  await connection();
  const sp = await searchParams;
  return <TradeJournalClient initialTradeIdProp={sp.trade_id} />;
}
