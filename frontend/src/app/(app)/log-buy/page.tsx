import { connection } from "next/server";
import LogBuyClient from "./log-buy-client";

export default async function LogBuyPage() {
  await connection();
  return <LogBuyClient />;
}
