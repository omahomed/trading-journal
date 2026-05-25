import { connection } from "next/server";
import DailyReportClient from "./daily-report-client";

export default async function DailyReportPage({
  searchParams,
}: {
  searchParams: Promise<{ date?: string }>;
}) {
  await connection();
  const sp = await searchParams;
  return <DailyReportClient initialDate={sp.date} />;
}
