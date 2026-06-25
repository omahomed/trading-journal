import { connection } from "next/server";
import EntryVsAddClient from "./entry-vs-add-client";

export default async function EntryVsAddPage() {
  await connection();
  return <EntryVsAddClient />;
}
