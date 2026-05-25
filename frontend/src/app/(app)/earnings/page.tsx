import { connection } from "next/server";
import EarningsClient from "./earnings-client";

export default async function EarningsPage() {
  await connection();
  return <EarningsClient />;
}
