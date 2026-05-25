import { connection } from "next/server";
import PositionSizerClient from "./position-sizer-client";

export default async function PositionSizerPage({
  searchParams,
}: {
  searchParams: Promise<{ tab?: string; trade_id?: string }>;
}) {
  await connection();
  const sp = await searchParams;
  return <PositionSizerClient initialTabProp={sp.tab} initialTradeIdProp={sp.trade_id} />;
}
