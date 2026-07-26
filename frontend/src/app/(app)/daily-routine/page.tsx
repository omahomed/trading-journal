import { connection } from "next/server";
import DailyRoutineClient from "./daily-routine-client";

export default async function DailyRoutinePage({
  searchParams,
}: {
  searchParams: Promise<{ date?: string }>;
}) {
  await connection();
  const sp = await searchParams;
  return <DailyRoutineClient initialDate={sp.date} />;
}
