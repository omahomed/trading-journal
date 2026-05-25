import { connection } from "next/server";
import DailyRoutineClient from "./daily-routine-client";

export default async function DailyRoutinePage() {
  await connection();
  return <DailyRoutineClient />;
}
