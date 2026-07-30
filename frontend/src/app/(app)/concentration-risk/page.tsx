import { connection } from "next/server";
import ConcentrationRiskClient from "./concentration-risk-client";

export default async function ConcentrationRiskPage() {
  await connection();
  return <ConcentrationRiskClient />;
}
