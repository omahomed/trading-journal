import { connection } from "next/server";
import AnalyticsClient from "./analytics-client";

export default async function AnalyticsPage({
  searchParams,
}: {
  searchParams: Promise<{ tab?: string; trade_id?: string }>;
}) {
  await connection();
  const sp = await searchParams;
  return <AnalyticsClient initialTabProp={sp.tab} initialTradeIdProp={sp.trade_id} />;
}
