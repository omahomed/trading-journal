import { connection } from "next/server";
import TradingChecklistClient from "./trading-checklist-client";

export default async function TradingChecklistPage() {
  await connection();
  return <TradingChecklistClient />;
}
