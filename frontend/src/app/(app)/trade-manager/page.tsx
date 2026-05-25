import { connection } from "next/server";
import TradeManagerClient from "./trade-manager-client";

export default async function TradeManagerPage({
  searchParams,
}: {
  searchParams: Promise<{ tab?: string }>;
}) {
  await connection();
  const sp = await searchParams;
  return <TradeManagerClient initialTabProp={sp.tab} />;
}
