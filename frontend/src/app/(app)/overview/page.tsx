import { connection } from "next/server";
import OverviewClient from "./overview-client";

export default async function OverviewPage() {
  await connection();
  return <OverviewClient />;
}
