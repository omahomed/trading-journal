import { connection } from "next/server";
import LogSellClient from "./log-sell-client";

export default async function LogSellPage() {
  await connection();
  return <LogSellClient />;
}
