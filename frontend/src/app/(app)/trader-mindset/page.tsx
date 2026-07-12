import { connection } from "next/server";
import TraderMindsetClient from "./trader-mindset-client";

export default async function TraderMindsetPage() {
  await connection();
  return <TraderMindsetClient />;
}
