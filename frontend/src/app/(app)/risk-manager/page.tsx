import { connection } from "next/server";
import RiskManagerClient from "./risk-manager-client";

export default async function RiskManagerPage() {
  await connection();
  return <RiskManagerClient />;
}
