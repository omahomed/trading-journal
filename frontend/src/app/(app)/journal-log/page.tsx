import { connection } from "next/server";
import JournalLogClient from "./journal-log-client";

export default async function JournalLogPage() {
  await connection();
  return <JournalLogClient />;
}
