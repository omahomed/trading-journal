import { connection } from "next/server";
import WeeklyRetroClient from "./weekly-retro-client";

export default async function WeeklyRetroPage() {
  await connection();
  return <WeeklyRetroClient />;
}
