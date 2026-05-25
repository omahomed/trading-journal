import { connection } from "next/server";
import AiCoachClient from "./ai-coach-client";

export default async function AiCoachPage() {
  await connection();
  return <AiCoachClient />;
}
