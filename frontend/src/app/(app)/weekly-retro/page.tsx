import { connection } from "next/server";
import WeeklyRetroClient from "./weekly-retro-client";

export default async function WeeklyRetroPage({
  searchParams,
}: {
  searchParams: Promise<{ week?: string }>;
}) {
  await connection();
  const sp = await searchParams;
  return <WeeklyRetroClient initialWeek={sp.week} />;
}
