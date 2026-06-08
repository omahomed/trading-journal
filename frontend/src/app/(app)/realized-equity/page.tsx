import { connection } from "next/server";
import RealizedEquityClient from "./realized-equity-client";

export default async function RealizedEquityPage() {
  await connection();
  return <RealizedEquityClient />;
}
